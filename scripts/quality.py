#!/usr/bin/env python3
"""Long-run quality benchmark: how far evolution gets over minutes, from blank or a mature start.

    scripts/quality.py --side master=master --seconds 300 --seeds 5        # characterize
    scripts/quality.py --side control=HEAD --side cand=WORKTREE             # build A/B
    scripts/quality.py --side control=HEAD --side nox=HEAD \\
        --params 'nox={"crossover_prob":0}'                                 # settings A/B
    scripts/quality.py --side a=HEAD --side b=HEAD --seconds 20 --seeds 2   # A/A
    scripts/quality.py --seconds 120 --seeds 3 --save-starts .cache/quality/starts/s120
    scripts/quality.py --start .cache/quality/starts/s120 --side control=HEAD \
        --side cand=WORKTREE --seconds 90 --images ff --seeds 3          # late-game screen

A side is NAME=BUILD, where BUILD is a git ref, WORKTREE, revert:SHA[@BASE] or an
evolution-bench executable (built and cached exactly as scripts/hillclimb.py does).
--params NAME=JSON|FILE merges MutationParams fields (snake_case or camelCase)
onto MutationParams::default() for that side. The first side is the control.

Each run evolves a drawing on one frozen v1 fixture image with one seed for
--seconds of wall clock, through ordinary `run_batch` calls with delivery, and
returns a trajectory of the GPU-delivered best. Every side runs the same seeds;
side order rotates per seed; each image gets fresh server processes. Runs wait
for a quiet host, hold the benchmark lock, and a run during which other processes
used more than one core is discarded and re-run.

Runs start blank unless --start DIR names saved start drawings, DIR/IMAGE/seedN.json
for every image and seed; then a run measures only the late game. --save-starts DIR
(one side only) copies each run's final drawing there, with meta.json naming the
build, params, seconds and start that produced them.

Metric at checkpoint T: remaining error r = 100 - fitness% (and r_raw without
the polygon point penalty). Comparison: geometric mean over images x seeds of
r_candidate / r_control (equal weight per image) with a stratified bootstrap 95%
interval. BETTER/WORSE need the interval to exclude 1 and a move of at least
--min-effect; fewer than three seeds give SCREEN. Also: time to reach fitness
targets taken from the control's worst seed. Records from builds with operator
stats (protocol 3) also give, per side and image, which mutation operators produced
the accepted improvements in each time window; protocol 4 adds, for a side with a
step spread, how often each step-size rung improved. See docs/quality-benchmark.md.
"""

import argparse
import datetime
import fcntl
import hashlib
import shutil
import itertools
import json
import math
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path

# Importing hillclimb must not leave scripts/__pycache__ in the tree.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hillclimb as hc
from hillclimb import HillclimbError, require

ROOT = hc.SHARED / ".cache/quality"
LOG = hc.SHARED / "benchmarks/quality/log.tsv"
# Must match FIXED_SECONDS in src/benchmark/quality.rs: the recorder takes a
# point at the first batch at or after each of these.
CHECKPOINTS = (1, 2, 5, 10, 20, 30, 60, 120, 180, 300, 600)
# Window ends of the printed operator table (and the budget); summary.json has
# a window per checkpoint.
OPERATOR_WINDOWS = (5, 30, 120, 600)
ABSOLUTE_TARGETS = (98.0, 99.0, 99.5)
WARMUP_BATCHES = 8
MAX_SIDES = 6
MAX_SEEDS = 32
MAX_SECONDS = 3600
# Engine construction (~11-15 s of shader compilation) plus the drain and slack.
REQUEST_SLACK_SECONDS = 600
NAME = re.compile(r"[A-Za-z0-9_.-]{1,32}")
LOG_COLUMNS = ("utc", "evidence", "side", "build", "identity", "params", "image", "seed",
               "seconds", "elapsed", "batches", "evals_per_s", "fitness", "raw_fitness",
               "polygons", "remaining_at", "host_load", "attempts", "note", "start")


def camel(key):
    head, *rest = key.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in rest)


def load_overrides(text):
    """A JSON object (inline, or a file path) of MutationParams fields, keys camelCased.

    The engine rejects unknown fields and values production would clamp."""
    source = text if text.lstrip().startswith("{") else Path(text).read_text()
    try:
        values = json.loads(source)
    except json.JSONDecodeError as exc:
        raise HillclimbError(f"params are not JSON: {exc}") from None
    require(isinstance(values, dict), "params must be a JSON object")
    require(all(isinstance(v, (bool, int, float, list)) for v in values.values()),
            "param values must be numbers, booleans or lists")
    return {camel(key): values[key] for key in sorted(values)}


def parse_sides(side_specs, param_specs):
    """{name: {"build": spec, "overrides": {...}}} in command-line order; first is control."""
    sides = {}
    for spec in side_specs or ["control=HEAD"]:
        name, sep, build = spec.partition("=")
        require(sep and build and NAME.fullmatch(name), f"--side must be NAME=BUILD: {spec}")
        require(name not in sides, f"duplicate side {name}")
        sides[name] = {"build": build, "overrides": {}}
    for spec in param_specs or []:
        name, sep, value = spec.partition("=")
        require(sep and name in sides, f"--params must be NAME=JSON|FILE for a side: {spec}")
        sides[name]["overrides"] = load_overrides(value)
    require(len(sides) <= MAX_SIDES, f"at most {MAX_SIDES} sides")
    return sides


def parse_args(argv):
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--side", action="append",
                     help="NAME=BUILD; repeat; first is the control (default control=HEAD)")
    cli.add_argument("--params", action="append",
                     help="NAME=JSON|FILE: MutationParams overrides for side NAME")
    cli.add_argument("--images", default=",".join(hc.CASES), help="subset of ff,mona,darwin")
    cli.add_argument("--seeds", type=int, default=5, help=f"seeds per image, 1..{MAX_SEEDS}")
    cli.add_argument("--first-seed", type=int, default=1)
    cli.add_argument("--seconds", type=float, default=120.0,
                     help=f"wall-clock budget per run, 1..{MAX_SECONDS} (default 120)")
    cli.add_argument("--server-scope", choices=("image", "run"), default="image",
                     help="fresh server per image and side (default) or per run "
                          "(+~12 s shader compile per run; no in-process submit drift)")
    cli.add_argument("--min-effect", type=float, default=0.02,
                     help="smallest remaining-error ratio change reported (default 0.02)")
    cli.add_argument("--start", type=Path,
                     help="directory of start drawings IMAGE/seedN.json (default: blank)")
    cli.add_argument("--save-starts", type=Path,
                     help="new directory to save each run's final drawing as a start")
    cli.add_argument("--build-only", action="store_true", help="resolve sides and exit")
    cli.add_argument("--note", default="", help="free text recorded in the log")
    args = cli.parse_args(argv)
    args.sides = parse_sides(args.side, args.params)
    args.images = tuple(dict.fromkeys(args.images.split(",")))
    require(args.images and set(args.images) <= set(hc.CASES),
            "images must be a subset of ff,mona,darwin")
    require(1 <= args.seeds <= MAX_SEEDS, f"seeds must be 1..{MAX_SEEDS}")
    require(args.first_seed >= 0, "first-seed must be non-negative")
    require(1 <= args.seconds <= MAX_SECONDS, f"seconds must be 1..{MAX_SECONDS}")
    require(0 <= args.min_effect < 1, "min-effect must be in 0..1")
    args.starts = start_files(args.start, args.images,
                              range(args.first_seed, args.first_seed + args.seeds))
    if args.save_starts:
        require(len(args.sides) == 1, "--save-starts needs exactly one side")
        require(not args.save_starts.exists(), f"{args.save_starts} already exists")
    return args


def start_files(directory, images, seeds):
    """{(image, seed): (absolute path, sha256)} for --start, or {} for blank starts."""
    if directory is None:
        return {}
    starts = {}
    for image in images:
        for seed in seeds:
            path = (directory / image / f"seed{seed}.json").resolve()
            require(path.is_file(), f"missing start drawing {path}")
            require(not re.search(r"\s", str(path)), f"start path has whitespace: {path}")
            starts[image, seed] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    return starts


def checkpoints(seconds):
    return [float(t) for t in CHECKPOINTS if t < seconds] + [float(seconds)]


def value_at(trajectory, t):
    """The best delivered at the first recorded batch at or after `t` (the final point if none)."""
    for point in trajectory:
        if point["seconds"] >= t:
            return point
    return trajectory[-1]


def time_to(trajectory, target):
    """Seconds until fitness first reached `target`, interpolated in log remaining error
    between the bracketing points (the recorder adds a point per 2% error drop); None if never."""
    previous = None
    for point in trajectory:
        if point["fitness"] >= target:
            if previous is None:
                return point["seconds"]
            r0, r1 = 100 - previous["fitness"], 100 - point["fitness"]
            share = (math.log(r0) - math.log(100 - target)) / (math.log(r0) - math.log(r1))
            return previous["seconds"] + share * (point["seconds"] - previous["seconds"])
        previous = point
    return None


class Servers:
    """Resident `evolution-bench serve` processes for one image, one per side."""

    def __init__(self, resolved, scope, workdir):
        self.resolved, self.scope, self.workdir, self.open = resolved, scope, workdir, {}

    def get(self, name):
        if name not in self.open:
            executable = self.resolved[name][2]
            self.open[name] = hc.Server(name, executable, self.workdir / name, timeout=0)
        return self.open[name]

    def pids(self):
        return [server.pid for server in self.open.values()]

    def done(self, name):
        if self.scope == "run":
            self.open.pop(name).close()

    def close(self):
        for server in self.open.values():
            server.close()
        self.open = {}


def request_line(side, image, seed, seconds, output, start=None):
    line = (f"quality --case {image} --seed {seed} --seconds {seconds} "
            f"--warmup {WARMUP_BATCHES} --output {output}")
    if side["overrides"]:
        line += " --params " + json.dumps(side["overrides"], separators=(",", ":"))
    if start:
        line += f" --start {start[0]}"
    return line


def check_record(record, image, seed, seconds, start=None):
    require(record.get("kind") == "quality" and record["case_id"] == image
            and record["seed"] == seed, "record does not match its request")
    require(record.get("start_sha256") == (start[1] if start else None),
            f"{image} seed {seed}: the run did not start from the requested drawing")
    # Errors include GPU device errors and a final fitness the scalar renderer disagrees
    # with (tile-bin overflow made darwin report 99.97% for an 81% drawing).
    require(not record["errors"], f"{image} seed {seed}: invalid run: {record['errors'][:3]}")
    trajectory = record["trajectory"]
    require(trajectory and trajectory[-1]["seconds"] >= seconds, f"{image}: run ended early")
    require(all(a["fitness"] <= b["fitness"] and a["batches"] <= b["batches"]
                for a, b in itertools.pairwise(trajectory)), f"{image}: trajectory not monotone")
    if record.get("protocol_version", 2) >= 3:
        names, snapshots = record["operator_names"], record["operators"]
        require(snapshots and all(len(s[kind]) == len(names) for s in snapshots
                                  for kind in OPERATOR_KINDS),
                f"{image} seed {seed}: malformed operator stats")
        require(sum(snapshots[-1]["offered"]) == snapshots[-1]["evaluations"],
                f"{image} seed {seed}: operator stats do not cover every evaluation")
    if record.get("protocol_version", 2) >= 4:
        rungs, scaled = len(record["step_multipliers"]), record["step_operators"]
        require(rungs and set(scaled) <= set(names)
                and all(len(s[kind]) == rungs for s in snapshots for kind in RUNG_KINDS),
                f"{image} seed {seed}: malformed step rung stats")
        last = snapshots[-1]
        for kind, rung_kind in zip(OPERATOR_KINDS, RUNG_KINDS):
            require(sum(last[rung_kind]) == sum(last[kind][names.index(n)] for n in scaled),
                    f"{image} seed {seed}: step rungs do not cover the step-scaled operators")


def run_one(servers, name, side, position, args, evidence):
    """One scored run of `name`, re-run while other processes disturbed it; returns
    (scored run, discarded runs)."""
    image, seed = position
    discarded = []
    for attempt in range(hc.BUSY_RETRIES + 1):
        server = servers.get(name)
        hc.wait_for_quiet(servers.pids())
        output = evidence / image / f"seed{seed}-{name}" / f"attempt{attempt}"
        start = args.starts.get(position)
        line = request_line(side, image, seed, args.seconds, output, start)
        before = hc.cpu_snapshot(servers.pids())
        try:
            record = server.request(line, args.seconds + REQUEST_SLACK_SECONDS)
        except HillclimbError as exc:
            raise HillclimbError(f"{exc} (a build older than the quality recorder "
                                 "has no `quality` request)") from None
        foreign = hc.foreign_cores(before, hc.cpu_snapshot(servers.pids()))
        servers.done(name)
        check_record(record, image, seed, args.seconds, start)
        run = summarize_run(record, name, image, seed, args.seconds)
        run.update(foreign_cores=foreign, attempts=attempt + 1, output=str(output))
        print(f"  {image} seed {seed} {name}: fitness {run['fitness']:.4f}  "
              f"{run['batches']} batches  host {foreign:.1f} cores", file=sys.stderr)
        if foreign <= hc.BUSY_CORES or attempt == hc.BUSY_RETRIES:
            return run, discarded
        discarded.append(run)
        print(f"  {image} seed {seed} {name}: other processes were busy; re-running",
              file=sys.stderr)
    raise AssertionError("unreachable: the last attempt returns")


def summarize_run(record, name, image, seed, seconds):
    trajectory = record["trajectory"]
    final = trajectory[-1]
    at = {}
    for t in checkpoints(seconds):
        point = final if t == seconds else value_at(trajectory, t)
        at[t] = {key: point[key] for key in ("fitness", "raw_fitness", "polygons", "batches")}
    return {"side": name, "image": image, "seed": seed, "elapsed": record["elapsed_seconds"],
            "batches": record["batches"], "evals_per_s": record["evals_per_second"],
            "fitness": final["fitness"], "raw_fitness": final["raw_fitness"],
            "polygons": final["polygons"], "at": at,
            "scalar_fitness": record.get("final_scalar_fitness"),
            "by_batch": {p["batches"]: p["fitness"] for p in trajectory},
            "trajectory": trajectory,
            "operator_names": record.get("operator_names"),
            "operators": record.get("operators"),
            "step_multipliers": record.get("step_multipliers")}


def measure(args, resolved, evidence):
    """Every image x seed x side; returns (scored runs, discarded runs)."""
    runs, discarded = [], []
    names = list(args.sides)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    for image in args.images:
        servers = Servers(resolved, args.server_scope, evidence / "serve" / image)
        try:
            for index, seed in enumerate(seeds):
                shift = index % len(names)
                for name in names[shift:] + names[:shift]:
                    run, extra = run_one(servers, name, args.sides[name], (image, seed),
                                         args, evidence)
                    runs.append(run)
                    discarded.extend(extra)
        finally:
            servers.close()
    return runs, discarded


def determinism(runs, resolved, sides):
    """Runs of one build and params on one image and seed must deliver the same best
    fitness at every batch index both recorded. Returns (groups, shared points, mismatches)."""
    groups = {}
    for run in runs:
        side = run["side"]
        key = (resolved[side][1], json.dumps(sides[side]["overrides"], sort_keys=True),
               run["image"], run["seed"])
        groups.setdefault(key, []).append(run)
    compared, shared, mismatches = 0, 0, []
    for key, members in groups.items():
        for other in members[1:]:
            common = members[0]["by_batch"].keys() & other["by_batch"].keys()
            compared += 1
            shared += len(common)
            if any(members[0]["by_batch"][b] != other["by_batch"][b] for b in common):
                mismatches.append(f"{key[2]} seed {key[3]}: {members[0]['side']} vs {other['side']}")
    return compared, shared, mismatches


def quartiles(values):
    ordered = sorted(values)
    return hc.percentile(ordered, 0.25), statistics.median(ordered), hc.percentile(ordered, 0.75)


def remaining(run, t, raw=False):
    value = 100.0 - run["at"][t]["raw_fitness" if raw else "fitness"]
    require(value > 0, "remaining error must be positive")
    return value


def compare(runs, control, name, t, raw, min_effect):
    """Geometric-mean r_name / r_control over images x seeds with its bootstrap interval."""
    by_key = {(r["side"], r["image"], r["seed"]): r for r in runs}
    ratios = {}
    for (side, image, seed), run in by_key.items():
        if side == name:
            base = by_key[(control, image, seed)]
            ratios.setdefault(image, []).append(remaining(run, t, raw) / remaining(base, t, raw))
    point, low, high = hc.aggregate(ratios)
    seeds = min(len(values) for values in ratios.values())
    return point, low, high, quality_verdict(point, low, high, seeds, min_effect)


def quality_verdict(point, low, high, seeds, min_effect):
    if seeds < 3:
        return "SCREEN"
    if high < 1.0 and point <= 1.0 - min_effect:
        return "BETTER"
    if low > 1.0 and point >= 1.0 + min_effect:
        return "WORSE"
    return "NOISE"


def control_targets(runs, control, image, seconds):
    """Fitness targets every control seed reached: the worst control seed's fitness at
    the final checkpoint and at the checkpoints nearest a quarter and a sixteenth of it."""
    marks = checkpoints(seconds)
    chosen = sorted({max((t for t in marks if t <= seconds / d), default=marks[0])
                     for d in (16, 4)} | {float(seconds)})
    mine = [r for r in runs if r["side"] == control and r["image"] == image]
    return [(t, min(r["at"][t]["fitness"] for r in mine)) for t in chosen]


def time_ratios(runs, control, name, image_targets):
    """Per target rank: {image: [t_name / t_control]}; a missed target counts as the
    run's full length (a lower bound) and is counted in `censored`."""
    by_key = {(r["side"], r["image"], r["seed"]): r for r in runs}
    ratios, censored = {}, 0
    for (side, image, seed), run in by_key.items():
        if side != name:
            continue
        base = by_key[(control, image, seed)]
        for rank, (_, target) in enumerate(image_targets[image]):
            reached = time_to(run["trajectory"], target)
            censored += reached is None
            ours = reached if reached is not None else run["elapsed"]
            theirs = max(time_to(base["trajectory"], target), 1e-3)
            ratios.setdefault(rank, {}).setdefault(image, []).append(max(ours, 1e-3) / theirs)
    return ratios, censored


def spread(values):
    """Seed-to-seed standard deviation of ln(values); 0 for one seed."""
    logs = [math.log(v) for v in values]
    return statistics.stdev(logs) if len(logs) > 1 else 0.0


def distribution_table(runs, sides, image, seconds):
    """Per checkpoint and side: remaining error distribution, fitness and polygons."""
    lines = [f"\n{image}: remaining error r = 100 - fitness% across seeds",
             (f"{'T s':>6}  {'side':12}{'r median':>9} {'[q25, q75]':>17} {'min..max':>17}"
              f"{'sd ln r':>8}{'fitness':>9}{'raw r':>8}{'polys':>6} {'[min..max]':>11}")]
    for t in checkpoints(seconds):
        for name in sides:
            mine = [r for r in runs if r["side"] == name and r["image"] == image]
            rs = [remaining(r, t) for r in mine]
            q25, median, q75 = quartiles(rs)
            polys = [r["at"][t]["polygons"] for r in mine]
            lines.append(
                f"{t:>6g}  {name:12}{median:>9.4f} [{q25:>7.4f}, {q75:>7.4f}] "
                f"{min(rs):>7.4f}..{max(rs):<7.4f}{spread(rs):>8.3f}"
                f"{100 - median:>9.4f}{statistics.median(remaining(r, t, True) for r in mine):>8.4f}"
                f"{statistics.median(polys):>6.0f} [{min(polys)}..{max(polys)}]")
    lines.append(f"{'':8}time to fitness (median s, reached/seeds): " + "  ".join(
        f"{name} " + " ".join(target_cell([time_to(r["trajectory"], target) for r in runs
                                           if r["side"] == name and r["image"] == image], target)
                              for target in ABSOLUTE_TARGETS) for name in sides))
    return "\n".join(lines)


def target_cell(times, target):
    reached = sorted(t for t in times if t is not None)
    median = f"{statistics.median(reached):.1f}" if len(reached) * 2 > len(times) else "-"
    return f"{target:g}%:{median}({len(reached)}/{len(times)})"


def comparison_table(runs, args):
    """Candidate vs control per checkpoint: remaining error and time-to-target ratios."""
    names = list(args.sides)
    control, entries, lines = names[0], [], []
    targets = {image: control_targets(runs, control, image, args.seconds) for image in args.images}
    for name in names[1:]:
        lines.append(f"\n{name} vs {control}: r ratio (geometric mean over images x seeds, "
                     "lower is better) and 95% bootstrap")
        for t in checkpoints(args.seconds):
            point, low, high, result = compare(runs, control, name, t, False, args.min_effect)
            rp, rl, rh, rresult = compare(runs, control, name, t, True, args.min_effect)
            entries.append({"side": name, "t": t, "ratio": point, "low95": low, "high95": high,
                            "verdict": result, "raw_ratio": rp, "raw_low95": rl,
                            "raw_high95": rh, "raw_verdict": rresult})
            lines.append(f"  T {t:>6g}s  {point:.4f} [{low:.4f}, {high:.4f}] {result:7}"
                         f"  unpenalized {rp:.4f} [{rl:.4f}, {rh:.4f}] {rresult}")
        ratios, censored = time_ratios(runs, control, name, targets)
        for rank, by_image in sorted(ratios.items()):
            point, low, high = hc.aggregate(by_image)
            label = ", ".join(f"{image} {targets[image][rank][1]:.3f}%" for image in args.images)
            lines.append(f"  time to control-worst fitness at T={targets[args.images[0]][rank][0]:g}s "
                         f"({label}): {point:.4f} [{low:.4f}, {high:.4f}]")
        if censored:
            lines.append(f"  {censored} candidate runs missed a target; their time counts as "
                         "the full run (a lower bound)")
        lines.append("  final polygons (median), control -> candidate: " + ", ".join(
            f"{image} " + " -> ".join(
                f"{statistics.median(r['polygons'] for r in runs if r['side'] == side and r['image'] == image):.0f}"
                for side in (control, name)) for image in args.images))
    return entries, targets, "\n".join(lines)


OPERATOR_KINDS = ("offered", "improved", "neutral")
# Per step rung, in the same order; protocol 4.
RUNG_KINDS = ("rung_offered", "rung_improved", "rung_neutral")


def operator_snapshot(run, t, seconds):
    """Cumulative counts at checkpoint `t`: the first snapshot at or after it (the
    post-drain one for the budget), as value_at reads the trajectory."""
    snapshots = run["operators"]
    if t < seconds:
        for snapshot in snapshots:
            if snapshot["seconds"] >= t:
                return snapshot
    return snapshots[-1]


def operator_windows(runs, side, image, ends, seconds, rungs=False):
    """(names, windows) for one side and image, counts summed over seeds; each window
    holds the counts between consecutive `ends` (from 0). None without operator stats.
    With `rungs`, the step rung counts (keys OPERATOR_KINDS), named by multiplier;
    None unless every run has them and a spread (multipliers that differ)."""
    names_key = "step_multipliers" if rungs else "operator_names"
    mine = [r for r in runs if r["side"] == side and r["image"] == image and r["operators"]]
    if not mine or (rungs and not all(r.get(names_key) and len(set(r[names_key])) > 1
                                      for r in mine)):
        return None
    names = mine[0][names_key]
    require(all(r[names_key] == names for r in mine), f"{names_key} differ between runs")
    source = dict(zip(OPERATOR_KINDS, RUNG_KINDS if rungs else OPERATOR_KINDS))
    windows, before, start = [], {kind: [0] * len(names) for kind in OPERATOR_KINDS}, 0.0
    for end in ends:
        now = {kind: [sum(column) for column in zip(*(operator_snapshot(r, end, seconds)[key]
                                                       for r in mine))]
               for kind, key in source.items()}
        windows.append({"from": start, "to": end, **{
            kind: [b - a for a, b in zip(before[kind], now[kind])] for kind in OPERATOR_KINDS}})
        before, start = now, end
    return names, windows


def table_windows(seconds):
    """Window ends of the printed operator and rung tables."""
    return [float(t) for t in OPERATOR_WINDOWS if t < seconds] + [seconds]


def operator_table(runs, side, image, seconds):
    """Per operator and window: its share of all improvements (%) and its improvement
    rate (improvements per 1000 offspring it produced); then whole-run columns."""
    found = operator_windows(runs, side, image, table_windows(seconds), seconds)
    if found is None:
        return ""
    names, windows = found
    whole = {kind: [sum(w[kind][i] for w in windows) for i in range(len(names))]
             for kind in OPERATOR_KINDS}
    seeds = sum(1 for r in runs if r["side"] == side and r["image"] == image)

    def share(values, i):
        total = sum(values)
        return 100 * values[i] / total if total else 0.0

    def rate(counts, i):
        return 1000 * counts["improved"][i] / counts["offered"][i] if counts["offered"][i] else 0.0

    labels = [f"{w['from']:g}-{w['to']:g}s" for w in windows]
    lines = [f"\n{image} {side}: operators over {seeds} seeds; per window improvement share % "
             "and improvements per 1000 offered",
             f"{'operator':14}" + "".join(f"{label:>15}" for label in labels)
             + f"{'offered%':>10}{'improved%':>10}{'per 1000':>9}{'neutral%':>9}"]
    for i, name in enumerate(names):
        if not whole["offered"][i]:
            continue
        cells = "".join(f"{share(w['improved'], i):>7.1f} {rate(w, i):>7.3f}" for w in windows)
        lines.append(f"{name:14}{cells}{share(whole['offered'], i):>10.1f}"
                     f"{share(whole['improved'], i):>10.1f}{rate(whole, i):>9.3f}"
                     f"{share(whole['neutral'], i):>9.1f}")
    lines.append(f"{'improvements':14}" + "".join(f"{sum(w['improved']):>15}" for w in windows)
                 + f"{sum(whole['offered']):>10}{sum(whole['improved']):>10}"
                 f"{1000 * sum(whole['improved']) / max(sum(whole['offered']), 1):>9.3f}"
                 f"{sum(whole['neutral']):>9}")
    return "\n".join(lines)


def rung_table(runs, side, image, seconds):
    """Per step rung of a side with a spread: its multiplier, its share of the
    step-scaled operators' offspring, its improvements per 1000 offered in each window,
    then whole-run improvement share and rate. Empty without a spread."""
    found = operator_windows(runs, side, image, table_windows(seconds), seconds, rungs=True)
    if found is None:
        return ""
    multipliers, windows = found
    whole = {kind: [sum(w[kind][i] for w in windows) for i in range(len(multipliers))]
             for kind in OPERATOR_KINDS}
    seeds = sum(1 for r in runs if r["side"] == side and r["image"] == image)

    def rate(counts, i):
        return 1000 * counts["improved"][i] / counts["offered"][i] if counts["offered"][i] else 0.0

    def share(values, i):
        return 100 * values[i] / sum(values) if sum(values) else 0.0

    labels = [f"{w['from']:g}-{w['to']:g}s" for w in windows]
    lines = [(f"\n{image} {side}: step rungs over {seeds} seeds (offset and vertex moves); "
              "improvements per 1000 offered per window"),
             f"{'rung':>4}{'step x':>8}{'offered%':>9}" + "".join(f"{label:>10}" for label in labels)
             + f"{'improved%':>10}{'per 1000':>9}"]
    for i, multiplier in enumerate(multipliers):
        cells = "".join(f"{rate(w, i):>10.3f}" for w in windows)
        lines.append(f"{i:>4}{multiplier:>8.3f}{share(whole['offered'], i):>9.1f}{cells}"
                     f"{share(whole['improved'], i):>10.1f}{rate(whole, i):>9.3f}")
    return "\n".join(lines)


def operator_summary(runs, sides, images, seconds):
    """Full counts per side and image, one window per checkpoint, for summary.json;
    `rungs` holds the step rung counts of a side with a spread."""
    entries = []
    for side in sides:
        for image in images:
            found = operator_windows(runs, side, image, checkpoints(seconds), seconds)
            if found:
                entry = {"side": side, "image": image, "names": found[0], "windows": found[1]}
                rungs = operator_windows(runs, side, image, checkpoints(seconds), seconds, True)
                if rungs:
                    entry["rungs"] = {"multipliers": rungs[0], "windows": rungs[1]}
                entries.append(entry)
    return entries


def append_log(runs, args, resolved, evidence):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("\t".join(LOG_COLUMNS) + "\n")
    utc = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    with LOG.open("a") as stream:
        for run in runs:
            side = run["side"]
            entry = {"utc": utc, "evidence": os.path.relpath(evidence, hc.REPO), "side": side,
                     "build": resolved[side][0], "identity": resolved[side][1],
                     "params": json.dumps(args.sides[side]["overrides"], separators=(",", ":")),
                     "image": run["image"], "seed": run["seed"], "seconds": f"{args.seconds:g}",
                     "elapsed": f"{run['elapsed']:.2f}", "batches": run["batches"],
                     "evals_per_s": round(run["evals_per_s"]), "fitness": f"{run['fitness']:.5f}",
                     "raw_fitness": f"{run['raw_fitness']:.5f}", "polygons": run["polygons"],
                     "remaining_at": json.dumps({f"{t:g}": round(100 - v["fitness"], 5)
                                                 for t, v in run["at"].items()},
                                                separators=(",", ":")),
                     "host_load": f"{run['foreign_cores']:.1f}", "attempts": run["attempts"],
                     "note": args.note,
                     "start": args.starts[run["image"], run["seed"]][1][:12] if args.starts else ""}
            clean = [str(entry[key]).replace("\t", " ").replace("\n", " ") for key in LOG_COLUMNS]
            stream.write("\t".join(clean) + "\n")


def report(args, resolved, evidence, runs, discarded):
    """Print tables, write the summary and log; return the exit code."""
    for image in args.images:
        print(distribution_table(runs, args.sides, image, args.seconds))
        for side in args.sides:
            for table in (operator_table(runs, side, image, args.seconds),
                          rung_table(runs, side, image, args.seconds)):
                if table:
                    print(table)
    entries, targets = [], {}
    if len(args.sides) > 1:
        entries, targets, table = comparison_table(runs, args)
        print(table)
    compared, shared, mismatches = determinism(runs + discarded, resolved, args.sides)
    if compared:
        print(f"\ndeterminism: {compared} same-build same-params run pairs, {shared} shared "
              f"batch points, " + ("identical" if not mismatches else
                                   "DETERMINISM-MISMATCH " + "; ".join(mismatches)))
    if discarded:
        print(f"re-ran {len(discarded)} runs that overlapped other CPU work")
    strip = ("trajectory", "by_batch", "operators", "operator_names", "step_multipliers")
    (evidence / "summary.json").write_text(json.dumps(
        {"args": {k: v for k, v in vars(args).items() if k != "starts"}, "identities": resolved,
         "starts": [{"image": image, "seed": seed, "path": str(path), "sha256": sha}
                    for (image, seed), (path, sha) in args.starts.items()],
         "runs": [{k: v for k, v in r.items() if k not in strip} for r in runs],
         "discarded": [{k: v for k, v in r.items() if k not in strip} for r in discarded],
         "comparisons": entries, "targets": targets,
         "operators": operator_summary(runs, args.sides, args.images, args.seconds),
         "determinism": {"pairs": compared, "points": shared, "mismatches": mismatches}},
        indent=2, default=str))
    append_log(runs, args, resolved, evidence)
    if args.save_starts:
        save_starts(runs, args, resolved, evidence)
    print(f"evidence {os.path.relpath(evidence, hc.REPO)}")
    return 1 if mismatches else 0


def save_starts(runs, args, resolved, evidence):
    """Copy each run's final delivered drawing to --save-starts as IMAGE/seedN.json."""
    (side,) = args.sides
    for run in runs:
        target = args.save_starts / run["image"] / f"seed{run['seed']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(run["output"]) / "best.json", target)
    meta = {"build": resolved[side][0], "identity": resolved[side][1],
            "params": args.sides[side]["overrides"], "seconds": args.seconds,
            "start": str(args.start) if args.start else None,
            "evidence": os.path.relpath(evidence, hc.REPO),
            "fitness": {f"{r['image']}/seed{r['seed']}": r["fitness"] for r in runs}}
    (args.save_starts / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(runs)} start drawings to {args.save_starts}")


def resolve_sides(sides):
    """(label, identity, executable) per side; sides sharing a build resolve it once."""
    builds = {}
    for side in sides.values():
        if side["build"] not in builds:
            builds[side["build"]] = hc.resolve(side["build"])
    return {name: builds[side["build"]] for name, side in sides.items()}


def main(argv=None):
    try:
        args = parse_args(argv)
        require(hc.MANIFEST.is_file(),
                f"missing fixtures {hc.MANIFEST}; run scripts/evolution-bench.py freeze")
        resolved = resolve_sides(args.sides)
        if args.build_only:
            for name, (label, _, executable) in resolved.items():
                print(f"{name}\t{label}\t{executable}")
            return 0
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)
        evidence = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=ROOT / "runs"))
        hc.LOCK.parent.mkdir(parents=True, exist_ok=True)
        with hc.LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            runs, discarded = measure(args, resolved, evidence)
        return report(args, resolved, evidence, runs, discarded)
    except HillclimbError as exc:
        print(f"quality: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
