#!/usr/bin/env python3
"""Fast A/B screen of engine throughput for hill-climbing.

    scripts/hillclimb.py                      # working tree vs HEAD, all cases
    scripts/hillclimb.py --control HEAD~1 --candidate HEAD --note "paired raster"
    scripts/hillclimb.py --candidate HEAD     # A/A: measures the noise floor

Each side is a git ref, WORKTREE (the uncommitted tree), or a prebuilt
evolution-bench executable. Builds are cached by git tree hash. Each side runs
one resident `evolution-bench serve` process, so driver shader compilation is
paid once per case rather than once per trial. Trials alternate AB/BA within
each case on the frozen v1 fixtures.

Gates: every record is complete and correct, and every trial of one case and
seed ends in the same canonical replay hash on both sides. A change that is
meant to alter evolution passes --allow-behavior-change to skip the hash gate.

Verdict: FASTER or SLOWER when the 95% bootstrap interval of the equal-case
geometric-mean paired ratio excludes 1.0 and the ratio moves by at least
--min-effect (default 0.5%, the past campaigns' adoption bar), otherwise NOISE.
Fewer than three pairs per case give no usable interval and report SCREEN. One row per run is
appended to benchmarks/hillclimb/log.tsv. This is a screen; adoption still goes
through the reserved-seed confirmation in benchmarks/evolution.
"""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
import queue
import random
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main_checkout():
    """The primary checkout, so linked worktrees share fixtures, builds, the lock and the log."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=REPO, capture_output=True, text=True, check=False)
    return Path(common.stdout.strip()).parent if common.returncode == 0 else REPO


SHARED = main_checkout()
ROOT = SHARED / ".cache/hillclimb"
MANIFEST = SHARED / ".cache/evolution-bench/fixtures/v1/manifest.json"
LOCK = SHARED / ".cache/evolution-bench/benchmark.lock"
LOG = SHARED / "benchmarks/hillclimb/log.tsv"
CASES = ("ff", "mona", "darwin")
WORKTREE = "WORKTREE"
REVERT = "revert:"
KEEP_BUILDS = 8
# Everything the evolution-bench executable is built from; script and doc commits
# leave this key, and therefore the cached build, unchanged.
BUILD_INPUTS = ("src", "examples/evolution-bench.rs", "examples/check-gpu-lifecycle.rs",
                "Cargo.toml", "Cargo.lock", "rust-toolchain.toml")
LIFECYCLE = "check-gpu-lifecycle"
MAX_CANDIDATES = 6
# --confirm: the default screen, then a long horizon where host-side costs show
# (64-batch screens missed a 25-42% regression that 512 batches exposed).
CONFIRM_HORIZONS = ((64, 5), (512, 3))
RESAMPLES = 10000
BOOTSTRAP_SEED = 20261028
LOG_COLUMNS = ("utc", "control", "candidate", "cases", "pairs", "batches", "ratio",
               "low95", "high95", "verdict", "replay", "control_evals_per_s",
               "candidate_evals_per_s", "evidence", "note", "host_load", "quality_ratio",
               "quality_low95", "quality_high95", "quality_verdict", "lifecycle")
# Other processes averaging more CPU cores than this during a timed trial (e.g. a
# parallel cargo build) biased results; the one-minute load average missed them.
BUSY_CORES = 1.0
# A pair with a busy trial is discarded and re-timed up to this many times;
# parallel agents' builds kept contaminating screens despite the warning.
BUSY_RETRIES = 3
QUIET_SECONDS = 5.0
QUIET_SAMPLE_SECONDS = 1.0
QUIET_LIMIT_SECONDS = 600.0


class HillclimbError(Exception):
    """An invalid input, failed build, failed trial or failed gate."""


def require(condition, message):
    if not condition:
        raise HillclimbError(message)


def git(*args, env=None):
    result = subprocess.run(["git", *args], cwd=REPO, env=env, check=False,
                            capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def worktree_tree():
    """Hash the working tree, including untracked unignored files, without touching the index."""
    with tempfile.TemporaryDirectory() as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
        git("read-tree", "HEAD", env=env)
        git("add", "-A", env=env)
        return git("write-tree", env=env)


def build_key(tree):
    """Hash the build inputs of a tree."""
    listing = git("ls-tree", tree, "--", *BUILD_INPUTS)
    return hashlib.sha256(listing.encode()).hexdigest()[:40]


def cargo_build(source, target_dir, log):
    """Build the benchmark executable and, where the tree has it, the lifecycle checker."""
    command = ["cargo", "build", "--release", "--example", "evolution-bench"]
    if (source / "examples" / f"{LIFECYCLE}.rs").is_file():
        command += ["--example", LIFECYCLE]
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=source, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0, f"build failed in {source}; see {log}")
    return target_dir / "release/examples/evolution-bench"


def revert_commit(spec):
    """Commit `BASE` with `SHA` reverted, for `revert:SHA[@BASE]` (BASE defaults to HEAD).

    Ablation: re-check whether an earlier kept change still pays on today's code."""
    target, _, base = spec.partition("@")
    target = git("rev-parse", "--verify", f"{target}^{{commit}}")
    base = git("rev-parse", "--verify", f"{base or 'HEAD'}^{{commit}}")
    source = checkout_ref(base)
    reverted = subprocess.run(["git", "revert", "--no-commit", target], cwd=source,
                              capture_output=True, text=True, check=False)
    try:
        require(reverted.returncode == 0,
                f"cannot revert {target[:10]} on {base[:10]}: {reverted.stderr.strip()[-300:]}")
        tree = subprocess.run(["git", "write-tree"], cwd=source, capture_output=True,
                              text=True, check=True).stdout.strip()
    finally:
        subprocess.run(["git", "reset", "--hard", "--quiet", base], cwd=source, check=True)
    commit = git("commit-tree", tree, "-p", base, "-m", f"hillclimb: revert {target[:10]}")
    return commit, f"revert:{target[:10]}@{base[:10]}"


def checkout_ref(commit):
    """Check a commit out into one reusable worktree so its target dir stays incremental."""
    source = ROOT / "ref-src"
    if not (source / ".git").exists():
        git("worktree", "add", "--detach", "--force", str(source), commit)
    else:
        subprocess.run(["git", "checkout", "--detach", "--force", commit], cwd=source,
                       check=True, capture_output=True)
    return source


def resolve(side):
    """Return (label, identity, executable) for a side, building it if not cached.

    Concurrent invocations share the ref worktree and build cache, so builds are
    serialized. A build also holds the benchmark lock shared: compiling on every
    core skewed concurrent timing runs by up to 17%, so timing waits for builds
    and builds wait for timing."""
    ROOT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "build.lock").open("a") as build, LOCK.open("a") as bench:
        fcntl.flock(build, fcntl.LOCK_EX)
        fcntl.flock(bench, fcntl.LOCK_SH)
        return resolve_locked(side)


def resolve_locked(side):
    if Path(side).is_file():
        path = Path(side).resolve()
        return path.name, f"binary:{path}", path
    if side == WORKTREE:
        tree, label, source, target = worktree_tree(), "worktree", REPO, REPO / "target"
    else:
        if side.startswith(REVERT):
            commit, label = revert_commit(side[len(REVERT):])
        else:
            commit = git("rev-parse", "--verify", f"{side}^{{commit}}")
            label = f"{side}@{commit[:10]}"
        tree = git("rev-parse", f"{commit}^{{tree}}")
        source, target = None, ROOT / "ref-target"
    key = build_key(tree)
    cached = ROOT / "bin" / key / "evolution-bench"
    if cached.is_file():
        os.utime(cached.parent)
    else:
        prune_binaries()
        if source is None:
            source = checkout_ref(commit)
        cached.parent.mkdir(parents=True, exist_ok=True)
        print(f"building {label} (build inputs {key[:10]})", file=sys.stderr)
        built = cargo_build(source, target, cached.parent / "build.log")
        checker = built.with_name(LIFECYCLE)
        if checker.is_file():
            shutil.copy2(checker, cached.with_name(LIFECYCLE))
        shutil.copy2(built, cached.with_suffix(".partial"))
        cached.with_suffix(".partial").rename(cached)
    return label, f"build:{key}", cached


def prune_binaries():
    """Keep the most recently used builds; each release binary is ~160 MB."""
    builds = sorted((ROOT / "bin").glob("*/"), key=lambda d: d.stat().st_mtime, reverse=True)
    for stale in builds[KEEP_BUILDS - 1:]:
        shutil.rmtree(stale)


class Server:
    """One resident `evolution-bench serve` process answering trial requests."""

    def __init__(self, name, executable, workdir, timeout):
        self.name, self.timeout = name, timeout
        workdir.mkdir(parents=True, exist_ok=True)
        self.stderr = (workdir / "stderr.log").open("ab")
        self.process = subprocess.Popen(
            [str(executable), "serve", "--manifest", str(MANIFEST)], cwd=workdir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True)
        self.pid = self.process.pid
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def trial(self, case, seed, batches, warmup, output):
        request = (f"--case {case} --seed {seed} --batches {batches} --warmup {warmup} "
                   f"--diagnostic false --output {output}\n")
        try:
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise HillclimbError(f"{self.name}: server exited; see {self.stderr.name}") from None
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise HillclimbError(f"{self.name}: no result within {self.timeout}s") from None
            require(line is not None, f"{self.name}: server exited; see {self.stderr.name}")
            if line.startswith("@result "):
                return json.loads(line[len("@result "):])

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.stderr.close()


class OneShot:
    """Per-trial processes for executables built before `serve` existed."""

    def __init__(self, name, executable, workdir, timeout):
        self.name, self.executable, self.workdir, self.timeout = name, executable, workdir, timeout
        self.pid = None
        workdir.mkdir(parents=True, exist_ok=True)

    def trial(self, case, seed, batches, warmup, output):
        command = [str(self.executable), "trial", "--manifest", str(MANIFEST), "--case", case,
                   "--seed", str(seed), "--batches", str(batches), "--warmup", str(warmup),
                   "--diagnostic", "false", "--output", str(output)]
        try:
            result = subprocess.run(command, cwd=self.workdir, capture_output=True,
                                    text=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            raise HillclimbError(f"{self.name}: no result within {self.timeout}s") from None
        require((output / "result.json").is_file(),
                f"{self.name}: trial produced no record: {result.stderr.strip()[-500:]}")
        return json.loads((output / "result.json").read_text())

    def close(self):
        pass


def supports_serve(executable):
    """New executables exit cleanly on an empty request stream; older ones reject `serve`."""
    probe = subprocess.run([str(executable), "serve", "--manifest", str(MANIFEST)],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=False)
    return probe.returncode == 0


def gpu_share(record):
    """Estimated GPU-busy fraction of the timed clock, from the sampled stage times.

    Near 1.0 the engine is GPU-bound, so kernel work is the lever; well below 1.0
    the host (encoding, submission, readback waits) limits throughput."""
    samples = max(1, record["timestamp_samples"])
    per_iteration_s = sum(record["sampled_gpu_stage_totals_ns"]) / samples / 1e9
    iterations = record["batches"] * record["params"]["gpuBatchIters"]
    return per_iteration_s * iterations / record["elapsed_seconds"]


CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def cpu_snapshot(pids):
    """(wall seconds, system busy ticks, ticks used by this script, its children and pids)."""
    with open("/proc/stat") as stream:
        fields = [int(value) for value in stream.readline().split()[1:]]
    busy = sum(fields) - fields[3] - fields[4]  # minus idle and iowait
    usage = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    ours = sum(u.ru_utime + u.ru_stime for u in usage) * CLOCK_TICKS
    for pid in pids:
        with contextlib.suppress(OSError):
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            ours += int(stat[11]) + int(stat[12])
    return time.monotonic(), busy, ours


def wait_for_quiet(pids, settle=QUIET_SECONDS, limit=QUIET_LIMIT_SECONDS):
    """Block until other processes use at most BUSY_CORES for `settle` seconds, or `limit`."""
    deadline = time.monotonic() + limit
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        before = cpu_snapshot(pids)
        time.sleep(QUIET_SAMPLE_SECONDS)
        if foreign_cores(before, cpu_snapshot(pids)) > BUSY_CORES:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= settle:
            return True
    return False


def foreign_cores(before, after):
    """Average CPU cores used by other processes between two snapshots."""
    wall = max(after[0] - before[0], 1e-9)
    return max(0.0, (after[1] - before[1]) - (after[2] - before[2])) / CLOCK_TICKS / wall


def check_record(record, case, seed, batches):
    require(record["case_id"] == case and record["seed"] == seed
            and record["batches"] == batches, "record does not match its request")
    require(record["correctness"] and not record["errors"],
            f"{case}: trial failed numerical gates: {record['errors'][:3]}")
    rate = record["evals_per_second"]
    require(isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0,
            f"{case}: trial has no throughput score")
    return rate


def run_case(servers, case, args, evidence, rows):
    """Prime each side, then run rounds in rotated side order; return candidate/control ratios.

    With one candidate the rotation is AB/BA. Every side runs the same number of
    batches before each round, so in-process drift affects all sides alike."""
    for side, server in servers.items():
        for index in range(args.prime):
            # The first trials after engine construction can stall; they are not scored.
            record = server.trial(case, args.seed, args.batches, args.warmup,
                                  evidence / case / f"prime{index}-{side}")
            check_record(record, case, args.seed, args.batches)
    names = list(servers)
    ratios = {name: [] for name in names[1:]}
    for pair in range(args.pairs):
        shift = pair % len(names)
        order = names[shift:] + names[:shift]
        # A behavior change is judged on several trajectories; sides share each seed.
        seed = args.seed + pair if args.allow_behavior_change else args.seed
        for attempt in range(BUSY_RETRIES + 1):
            pair_rows = run_pair(servers, order, case, (pair, attempt, seed), args, evidence)
            if attempt == BUSY_RETRIES or all(
                    row["foreign_cores"] <= BUSY_CORES for row in pair_rows):
                break
            # Another process shared the CPU during this pair; time it again once
            # the host is quiet (builds last minutes, so an immediate retry fails).
            args.discarded.extend(pair_rows)
            print(f"  {case} pair {pair}: other processes were busy; waiting to retry",
                  file=sys.stderr)
            wait_for_quiet([server.pid for server in servers.values() if server.pid])
        rows.extend(pair_rows)
        rates = {row["side"]: row["rate"] for row in pair_rows}
        for name, values in ratios.items():
            values.append(rates[name] / rates["control"])
        print(f"  {case} pair {pair}: control {rates['control']:,.0f}  " + "  ".join(
            f"{name} {rates[name]:,.0f} ({ratios[name][-1]:.4f})" for name in ratios),
            file=sys.stderr)
    return ratios


def run_pair(servers, order, case, position, args, evidence):
    """Time one trial per side in `order`; return their rows."""
    pair, attempt, seed = position
    rows = []
    for side in order:
        retry = f"-retry{attempt}" if attempt else ""
        output = evidence / case / f"{pair:02d}-{side}{retry}"
        pids = [server.pid for server in servers.values() if server.pid]
        before = cpu_snapshot(pids)
        record = servers[side].trial(case, seed, args.batches, args.warmup, output)
        foreign = foreign_cores(before, cpu_snapshot(pids))
        rows.append({"case": case, "pair": pair, "seed": seed, "side": side,
                     "rate": check_record(record, case, seed, args.batches),
                     "gpu_share": gpu_share(record), "foreign_cores": foreign,
                     "gain_per_s": fitness_gain(record) / record["elapsed_seconds"],
                     "hash": record["final_state_sha256"],
                     "stage_ns": [total / max(1, record["timestamp_samples"])
                                  for total in record["sampled_gpu_stage_totals_ns"]]})
    return rows


def replay_gate(rows, candidate):
    """Per case: True when control and candidate trials of each seed end in one state."""
    hashes = {}
    for row in rows:
        if row["side"] in ("control", candidate):
            hashes.setdefault(row["case"], {}).setdefault(row["seed"], set()).add(row["hash"])
    return {case: all(len(values) == 1 for values in seeds.values())
            for case, seeds in hashes.items()}


def fitness_gain(record):
    """Best final fitness minus the starting fitness, in percentage points."""
    gain = max(record["final_fitness_percent"]) - record["initial_similarity_percent"]
    require(gain > 0, f"{record['case_id']}: no fitness gain; raise --batches")
    return gain


def quality_ratios(rows, candidate):
    """Per case: candidate/control fitness gain per second, paired by round."""
    by_round = {}
    for row in rows:
        if row["side"] in ("control", candidate):
            by_round.setdefault((row["case"], row["pair"]), {})[row["side"]] = row["gain_per_s"]
    ratios = {}
    for (case, _), sides in by_round.items():
        ratios.setdefault(case, []).append(sides[candidate] / sides["control"])
    return ratios


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    return values[low] + (values[math.ceil(position)] - values[low]) * (position - low)


def aggregate(ratios):
    """Equal-case geometric mean of paired ratios with a stratified bootstrap 95% interval."""
    logs = [[math.log(value) for value in values] for values in ratios.values()]
    point = math.exp(statistics.fmean(statistics.fmean(values) for values in logs))
    rng, samples = random.Random(BOOTSTRAP_SEED), []
    for _ in range(RESAMPLES):
        means = [statistics.fmean(rng.choices(values, k=len(values))) for values in logs]
        samples.append(statistics.fmean(means))
    samples.sort()
    return point, math.exp(percentile(samples, 0.025)), math.exp(percentile(samples, 0.975))


def verdict(point, low, high, pairs, min_effect):
    if pairs < 3:
        return "SCREEN"
    if low > 1.0 and point >= 1.0 + min_effect:
        return "FASTER"
    if high < 1.0 and point <= 1.0 - min_effect:
        return "SLOWER"
    return "NOISE"


def case_table(rows, ratios, candidate):
    """Median rate, GPU share and per-stage GPU µs per iteration, control → candidate."""
    lines = [(f"{'case':8}{'control':>12}{'candidate':>12}{'ratio':>9}{'gpu%':>12}  "
              "GPU µs/iteration control→candidate: mutate bin raster select")]
    for case, values in ratios.items():
        def median(field, side, index=None, case=case):
            return statistics.median(
                r[field] if index is None else r[field][index]
                for r in rows if r["case"] == case and r["side"] == side)
        stages = "  ".join(f"{median('stage_ns', 'control', i) / 1e3:.0f}→"
                           f"{median('stage_ns', candidate, i) / 1e3:.0f}" for i in range(4))
        lines.append(f"{case:8}{median('rate', 'control'):>12,.0f}"
                     f"{median('rate', candidate):>12,.0f}{statistics.median(values):>9.4f}"
                     f"{median('gpu_share', 'control'):>6.0%}→"
                     f"{median('gpu_share', candidate):<4.0%}  {stages}")
    return "\n".join(lines)


def append_log(entry):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(LOG_COLUMNS) + "\n"
    if LOG.exists():
        lines = LOG.read_text().splitlines(keepends=True)
        if lines and lines[0] != header:
            # Appended columns: rewrite the header; older rows leave them empty.
            LOG.write_text(header + "".join(lines[1:]))
    else:
        LOG.write_text(header)
    with LOG.open("a") as stream:
        clean = [str(entry.get(key, "")).replace("\t", " ").replace("\n", " ") for key in LOG_COLUMNS]
        stream.write("\t".join(clean) + "\n")


def median_rate(rows, side):
    return round(statistics.median(r["rate"] for r in rows if r["side"] == side))


def measure(args, sides, evidence):
    rows, args.discarded = [], []
    ratios = {name: {} for name in list(sides)[1:]}
    kinds = {side: Server if supports_serve(executable) else OneShot
             for side, (_, _, executable) in sides.items()}
    for side, kind in kinds.items():
        if kind is OneShot:
            print(f"{side}: no serve mode; one process per trial (~12 s setup each)", file=sys.stderr)
    for case in args.cases:
        # Fresh processes per case: driver submit cost drifts with every batch a
        # process has run, and differently per process. Servers reused across
        # cases showed persistent ±11% A/A offsets; fresh ones stay within ±0.4%.
        servers = {}
        try:
            for side, (_, identity, executable) in sides.items():
                workdir = ROOT / "serve" / identity.replace(":", "-").replace("/", "_")[-80:] / side
                servers[side] = kinds[side](side, executable, workdir, args.timeout)
            for name, values in run_case(servers, case, args, evidence, rows).items():
                ratios[name][case] = values
        finally:
            for server in servers.values():
                server.close()
    return rows, ratios


def summarize(args, sides, evidence, rows, ratios):
    """Log and print one verdict per candidate; return the process exit code."""
    entries = [summarize_candidate(args, sides, evidence, rows, name, by_case)
               for name, by_case in ratios.items()]
    (evidence / "summary.json").write_text(json.dumps(
        {"entries": entries, "ratios": ratios, "rows": rows, "discarded": args.discarded,
         "identities": {side: value[1] for side, value in sides.items()},
         "args": {k: v for k, v in vars(args).items() if k != "discarded"},
         "lifecycle": args.lifecycle_digests},
        indent=2, default=str))
    if args.discarded:
        print(f"re-timed {len(args.discarded)} trials that overlapped other CPU work")
    busy = [row for row in rows if row["foreign_cores"] > BUSY_CORES]
    if busy:
        worst = max(row["foreign_cores"] for row in busy)
        print(f"WARNING: other processes used up to {worst:.1f} CPU cores during {len(busy)} "
              f"of {len(rows)} timed trials; results may be biased. Rerun on an idle machine.")
    print(f"evidence {os.path.relpath(evidence, REPO)}")
    return entries


def summarize_candidate(args, sides, evidence, rows, name, ratios):
    replay = replay_gate(rows, name)
    point, low, high = aggregate(ratios)
    result = verdict(point, low, high, args.pairs, args.min_effect)
    replay_ok = all(replay.values())
    if not replay_ok and not args.allow_behavior_change:
        result = "REPLAY-MISMATCH"
    entry = {"utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
             "control": sides["control"][0], "candidate": sides[name][0],
             "cases": ",".join(args.cases), "pairs": args.pairs, "batches": args.batches,
             "ratio": f"{point:.5f}", "low95": f"{low:.5f}", "high95": f"{high:.5f}",
             "verdict": result, "replay": "same" if replay_ok else "differs",
             "control_evals_per_s": median_rate(rows, "control"),
             "candidate_evals_per_s": median_rate(rows, name),
             "evidence": os.path.relpath(evidence, REPO), "note": args.note,
             "host_load": f"{max(row['foreign_cores'] for row in rows):.1f}",
             "lifecycle": lifecycle_verdict(args.lifecycle_digests, name)}
    if args.allow_behavior_change:
        q_point, q_low, q_high = aggregate(quality_ratios(rows, name))
        entry.update(quality_ratio=f"{q_point:.5f}", quality_low95=f"{q_low:.5f}",
                     quality_high95=f"{q_high:.5f}",
                     quality_verdict=verdict(q_point, q_low, q_high, args.pairs, args.min_effect))
    entry["replay_by_case"] = replay
    append_log(entry)
    print(f"\n== {name}: {entry['candidate']}")
    print(case_table(rows, ratios, name))
    print(f"{result}: candidate/control {point:.4f}  95% [{low:.4f}, {high:.4f}]  "
          f"replay {entry['replay']}" + (f"  lifecycle {entry['lifecycle']}"
                                         if entry["lifecycle"] else ""))
    if args.allow_behavior_change:
        print(f"QUALITY {entry['quality_verdict']}: fitness gain per second, candidate/control "
              f"{entry['quality_ratio']}  95% [{entry['quality_low95']}, {entry['quality_high95']}]")
    return entry


def parse_args(argv):
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--control", default="HEAD", help="git ref, WORKTREE or executable (default HEAD)")
    cli.add_argument("--candidate", action="append",
                     help="git ref, WORKTREE, revert:SHA[@BASE] (BASE with SHA reverted, "
                          "for ablation) or executable; repeat to compare several "
                          "candidates against one control in one run (default WORKTREE)")
    cli.add_argument("--confirm", action="store_true",
                     help="run 64 batches x 5 pairs and 512 x 3; ADOPT only if neither "
                          "horizon regresses (speed or quality) and one improves")
    cli.add_argument("--lifecycle", action="store_true",
                     help="also run each side's check-gpu-lifecycle and require a "
                          "byte-identical comparison.json (implied by --confirm)")
    cli.add_argument("--build-only", action="store_true",
                     help="build or find each side and print its cached executable path")
    cli.add_argument("--cases", default=",".join(CASES), help="comma-separated subset of ff,mona,darwin")
    cli.add_argument("--pairs", type=int, default=5, help="AB/BA pairs per case, 1..20 (default 5)")
    cli.add_argument("--batches", type=int, default=64, help="timed batches per trial, 1..4096 (default 64)")
    cli.add_argument("--prime", type=int, default=1,
                     help="unscored trials per case and side before pairing, 0..4 (default 1)")
    cli.add_argument("--warmup", type=int, default=8, help="untimed warmup batches, 0..16 (default 8)")
    cli.add_argument("--seed", type=int, default=7)
    cli.add_argument("--min-effect", type=float, default=0.005,
                     help="smallest ratio change reported as FASTER/SLOWER (default 0.005)")
    cli.add_argument("--timeout", type=float, default=300.0, help="seconds per trial, including engine construction")
    cli.add_argument("--allow-behavior-change", action="store_true",
                     help="report replay mismatches instead of failing on them")
    cli.add_argument("--note", default="", help="free text recorded in the log")
    args = cli.parse_args(argv)
    args.candidate = args.candidate or [WORKTREE]
    args.lifecycle = args.lifecycle or (args.confirm and not args.allow_behavior_change)
    require(len(args.candidate) <= MAX_CANDIDATES, f"at most {MAX_CANDIDATES} candidates")
    args.cases = tuple(dict.fromkeys(args.cases.split(",")))
    require(args.cases and set(args.cases) <= set(CASES), "cases must be a subset of ff,mona,darwin")
    require(1 <= args.pairs <= 20, "pairs must be 1..20")
    require(1 <= args.batches <= 4096, "batches must be 1..4096")
    require(0 <= args.warmup <= 16, "warmup must be 0..16")
    require(0 <= args.prime <= 4, "prime must be 0..4")
    require(args.timeout > 0, "timeout must be positive")
    require(0 <= args.min_effect < 1, "min-effect must be in 0..1")
    return args


def resolve_sides(args):
    """Resolve control and candidates; a revert that does not apply is skipped, not fatal."""
    sides = {"control": resolve(args.control)}
    candidates = []
    for side in args.candidate:
        try:
            candidates.append(resolve(side))
        except HillclimbError as exc:
            if not side.startswith(REVERT):
                raise
            print(f"skipping {side}: {exc}", file=sys.stderr)
    require(candidates, "no candidate could be built")
    names = ["candidate"] if len(candidates) == 1 else [
        f"candidate{index}" for index in range(1, len(candidates) + 1)]
    sides.update(zip(names, candidates))
    return sides


def run_horizon(args, sides):
    """Measure every side at one batch budget; return one log entry per candidate."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=ROOT / "runs"))
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, ratios = measure(args, sides, evidence)
    return summarize(args, sides, evidence, rows, ratios)


def lifecycle_digests(sides, evidence):
    """Run each side's lifecycle checker; return {side: comparison.json sha256 or None}.

    comparison.json covers delivery, returned drawings, counters and state digests
    and is byte-identical across runs of one build, so any difference is real."""
    digests = {}
    for side, (_, _, executable) in sides.items():
        checker = Path(executable).with_name(LIFECYCLE)
        if not checker.is_file():
            digests[side] = None
            continue
        output = evidence / "lifecycle" / side
        output.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = subprocess.run([str(checker), "--manifest", str(MANIFEST), "--output",
                                     str(output), "--batches", "64"], cwd=output.parent,
                                    capture_output=True, text=True, timeout=1800, check=False)
        report = output / "comparison.json"
        require(result.returncode == 0 and report.is_file(),
                f"{side}: lifecycle checker failed: {result.stderr.strip()[-300:]}")
        digests[side] = hashlib.sha256(report.read_bytes()).hexdigest()
    return digests


def lifecycle_verdict(digests, side):
    if digests is None:
        return ""
    if digests.get(side) is None or digests.get("control") is None:
        return "unavailable"
    return "same" if digests[side] == digests["control"] else "differs"


def confirmation(entries):
    """Adopt only if no horizon regresses and at least one improves."""
    verdicts = [entry["verdict"] for entry in entries]
    if any(entry.get("lifecycle") in ("differs", "unavailable")
           and entry["replay"] == "same" for entry in entries):
        return "REJECT"
    verdicts += [entry["quality_verdict"] for entry in entries if "quality_verdict" in entry]
    if any(v in ("SLOWER", "REPLAY-MISMATCH", "SCREEN") for v in verdicts):
        return "REJECT"
    return "ADOPT" if "FASTER" in verdicts else "NOISE"


def main(argv=None):
    try:
        args = parse_args(argv)
        require(MANIFEST.is_file(), f"missing fixtures {MANIFEST}; run scripts/evolution-bench.py freeze")
        sides = resolve_sides(args)
        if args.build_only:
            for name, (label, _, executable) in sides.items():
                print(f"{name}\t{label}\t{executable}")
            return 0
        args.lifecycle_digests = None
        if args.lifecycle:
            stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            (ROOT / "runs").mkdir(parents=True, exist_ok=True)
            checks = Path(tempfile.mkdtemp(prefix=f"{stamp}-lifecycle-", dir=ROOT / "runs"))
            args.lifecycle_digests = lifecycle_digests(sides, checks)
        horizons = CONFIRM_HORIZONS if args.confirm else ((args.batches, args.pairs),)
        entries = []
        for batches, pairs in horizons:
            args.batches, args.pairs = batches, pairs
            entries.extend(run_horizon(args, sides))
        if args.confirm:
            print()
            for name in list(sides)[1:]:
                mine = [e for e in entries if e["candidate"] == sides[name][0]]
                print(f"CONFIRM {confirmation(mine)}: {sides[name][0]} "
                      + "  ".join(f"{e['batches']} batches {e['verdict']}"
                                  + (f"/quality {e['quality_verdict']}" if "quality_verdict" in e else "")
                                  for e in mine))
        return 1 if any(e["verdict"] == "REPLAY-MISMATCH" for e in entries) else 0
    except HillclimbError as exc:
        print(f"hillclimb: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
