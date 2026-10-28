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
import datetime
import fcntl
import json
import math
import os
import queue
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / ".cache/hillclimb"
MANIFEST = REPO / ".cache/evolution-bench/fixtures/v1/manifest.json"
LOCK = REPO / ".cache/evolution-bench/benchmark.lock"
LOG = REPO / "benchmarks/hillclimb/log.tsv"
CASES = ("ff", "mona", "darwin")
WORKTREE = "WORKTREE"
KEEP_BUILDS = 8
RESAMPLES = 10000
BOOTSTRAP_SEED = 20261028
LOG_COLUMNS = ("utc", "control", "candidate", "cases", "pairs", "batches", "ratio",
               "low95", "high95", "verdict", "replay", "control_evals_per_s",
               "candidate_evals_per_s", "evidence", "note")


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


def cargo_build(source, target_dir, log):
    command = ["cargo", "build", "--release", "--example", "evolution-bench"]
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=source, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0, f"build failed in {source}; see {log}")
    return target_dir / "release/examples/evolution-bench"


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
    """Return (label, identity, executable) for a side, building it if not cached."""
    if Path(side).is_file():
        path = Path(side).resolve()
        return path.name, f"binary:{path}", path
    if side == WORKTREE:
        tree, label, source, target = worktree_tree(), "worktree", REPO, REPO / "target"
    else:
        commit = git("rev-parse", "--verify", f"{side}^{{commit}}")
        tree, label = git("rev-parse", f"{commit}^{{tree}}"), f"{side}@{commit[:10]}"
        source, target = None, ROOT / "ref-target"
    cached = ROOT / "bin" / tree / "evolution-bench"
    if cached.is_file():
        os.utime(cached.parent)
    else:
        prune_binaries()
        if source is None:
            source = checkout_ref(commit)
        cached.parent.mkdir(parents=True, exist_ok=True)
        print(f"building {label} (tree {tree[:10]})", file=sys.stderr)
        built = cargo_build(source, target, cached.parent / "build.log")
        shutil.copy2(built, cached.with_suffix(".partial"))
        cached.with_suffix(".partial").rename(cached)
    return label, f"tree:{tree}", cached


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
    """Prime each side, then alternate AB/BA pairs; return paired candidate/control ratios."""
    for side, server in servers.items():
        for index in range(args.prime):
            # The first trials after engine construction can stall; they are not scored.
            record = server.trial(case, args.seed, args.batches, args.warmup,
                                  evidence / case / f"prime{index}-{side}")
            check_record(record, case, args.seed, args.batches)
    ratios = []
    for pair in range(args.pairs):
        order = ("control", "candidate") if pair % 2 == 0 else ("candidate", "control")
        rates = {}
        for side in order:
            output = evidence / case / f"{pair:02d}-{side}"
            record = servers[side].trial(case, args.seed, args.batches, args.warmup, output)
            rates[side] = check_record(record, case, args.seed, args.batches)
            rows.append({"case": case, "pair": pair, "side": side, "rate": rates[side],
                         "hash": record["final_state_sha256"],
                         "stage_ns": [total / max(1, record["timestamp_samples"])
                                      for total in record["sampled_gpu_stage_totals_ns"]]})
        ratios.append(rates["candidate"] / rates["control"])
        print(f"  {case} pair {pair}: control {rates['control']:,.0f}  candidate "
              f"{rates['candidate']:,.0f}  ratio {ratios[-1]:.4f}", file=sys.stderr)
    return ratios


def replay_gate(rows):
    """Return per-case hash identity: True when every trial of a case ends in one state."""
    hashes = {}
    for row in rows:
        hashes.setdefault(row["case"], set()).add(row["hash"])
    return {case: len(values) == 1 for case, values in hashes.items()}


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


def case_table(rows, ratios):
    lines = [f"{'case':8}{'control':>12}{'candidate':>12}{'ratio':>9}  GPU µs/batch mut/bin/rast/sel (cand)"]
    for case, values in ratios.items():
        rate = {side: statistics.median(r["rate"] for r in rows
                                        if r["case"] == case and r["side"] == side)
                for side in ("control", "candidate")}
        stages = [statistics.median(r["stage_ns"][i] for r in rows
                                    if r["case"] == case and r["side"] == "candidate") / 1e3
                  for i in range(4)]
        lines.append(f"{case:8}{rate['control']:>12,.0f}{rate['candidate']:>12,.0f}"
                     f"{statistics.median(values):>9.4f}  " + "/".join(f"{s:.0f}" for s in stages))
    return "\n".join(lines)


def append_log(entry):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a") as stream:
        if new:
            stream.write("\t".join(LOG_COLUMNS) + "\n")
        clean = [str(entry[key]).replace("\t", " ").replace("\n", " ") for key in LOG_COLUMNS]
        stream.write("\t".join(clean) + "\n")


def median_rate(rows, side):
    return round(statistics.median(r["rate"] for r in rows if r["side"] == side))


def measure(args, sides, evidence):
    rows, ratios, servers = [], {}, {}
    try:
        for side, (_, identity, executable) in sides.items():
            workdir = ROOT / "serve" / identity.replace(":", "-").replace("/", "_")[-80:] / side
            kind = Server if supports_serve(executable) else OneShot
            if kind is OneShot:
                print(f"{side}: no serve mode; one process per trial (~12 s setup each)",
                      file=sys.stderr)
            servers[side] = kind(side, executable, workdir, args.timeout)
        for case in args.cases:
            ratios[case] = run_case(servers, case, args, evidence, rows)
    finally:
        for server in servers.values():
            server.close()
    return rows, ratios


def summarize(args, sides, evidence, rows, ratios):
    replay = replay_gate(rows)
    point, low, high = aggregate(ratios)
    result = verdict(point, low, high, args.pairs, args.min_effect)
    replay_ok = all(replay.values())
    if not replay_ok and not args.allow_behavior_change:
        result = "REPLAY-MISMATCH"
    entry = {"utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
             "control": sides["control"][0], "candidate": sides["candidate"][0],
             "cases": ",".join(args.cases), "pairs": args.pairs, "batches": args.batches,
             "ratio": f"{point:.5f}", "low95": f"{low:.5f}", "high95": f"{high:.5f}",
             "verdict": result, "replay": "same" if replay_ok else "differs",
             "control_evals_per_s": median_rate(rows, "control"),
             "candidate_evals_per_s": median_rate(rows, "candidate"),
             "evidence": os.path.relpath(evidence, REPO), "note": args.note}
    (evidence / "summary.json").write_text(json.dumps(
        {"entry": entry, "ratios": ratios, "replay": replay, "rows": rows,
         "identities": {side: value[1] for side, value in sides.items()},
         "args": {k: v for k, v in vars(args).items()}}, indent=2, default=str))
    append_log(entry)
    print(case_table(rows, ratios))
    print(f"\n{result}: candidate/control {point:.4f}  95% [{low:.4f}, {high:.4f}]  "
          f"replay {entry['replay']}\nevidence {entry['evidence']}")
    return 0 if result != "REPLAY-MISMATCH" else 1


def parse_args(argv):
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--control", default="HEAD", help="git ref, WORKTREE or executable (default HEAD)")
    cli.add_argument("--candidate", default=WORKTREE, help="git ref, WORKTREE or executable (default WORKTREE)")
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
    args.cases = tuple(dict.fromkeys(args.cases.split(",")))
    require(args.cases and set(args.cases) <= set(CASES), "cases must be a subset of ff,mona,darwin")
    require(1 <= args.pairs <= 20, "pairs must be 1..20")
    require(1 <= args.batches <= 4096, "batches must be 1..4096")
    require(0 <= args.warmup <= 16, "warmup must be 0..16")
    require(0 <= args.prime <= 4, "prime must be 0..4")
    require(args.timeout > 0, "timeout must be positive")
    require(0 <= args.min_effect < 1, "min-effect must be in 0..1")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        require(MANIFEST.is_file(), f"missing fixtures {MANIFEST}; run scripts/evolution-bench.py freeze")
        sides = {"control": resolve(args.control), "candidate": resolve(args.candidate)}
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)
        evidence = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=ROOT / "runs"))
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows, ratios = measure(args, sides, evidence)
        return summarize(args, sides, evidence, rows, ratios)
    except HillclimbError as exc:
        print(f"hillclimb: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
