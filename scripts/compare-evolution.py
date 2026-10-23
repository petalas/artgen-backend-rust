#!/usr/bin/env python3
"""Serial provisional comparison of two supplied v1 evolution executables.

Run with --baseline PATH/summary.json --reference PATH --candidate PATH.
Defaults inherit the capture's seed, batches and warmup; --pairs defaults to 5.
Policy 1 uses 10,000 paired, stratified percentile bootstrap resamples. Intervals
describe conditional timing uncertainty for one search seed, not search quality
or reliable coverage under machine drift. No invocation can produce KEEP.

Build the Lever supplies this rerunnable command. Test Behavior, Not
Implementation supplies fake-child tests of its public CLI and evidence.
Setup archives inputs and policy before any measurement; shared writes are
serialized by the existing benchmark lock. No builds or parallel GPU work.

Power of Ten deviations 3/8/9: Python/JSON/argparse/importlib and the existing
process helper allocate and dispatch dynamically. Only the named local helper
is imported. Bounds are three cases, ten pairs, 63 workers, 10,000 resamples,
two 512 MiB binaries, 2 MiB results and the helper's file/inventory/time limits.
Rule 10: validate with warning-enabled stdlib unittest; CI is outside this slice.
Synthetic children establish supervisor behavior, never GPU correctness.
"""

import argparse
import importlib.util
import math
import os
import platform
import random
import shutil
import signal
import sys
from pathlib import Path


def load_bench():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "evolution_comparison_helpers", Path(__file__).with_name("evolution-bench.py"))
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load local benchmark helpers")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


bench = load_bench()
LOCK = bench.DEFAULT_ROOT / "benchmark.lock"
BOOTSTRAP_SEED = 20261023
RESAMPLES = 10000
NOISE_BAND = 0.004631
FAILURES = (bench.BenchError, OSError, ValueError, KeyError, TypeError, OverflowError)
CANONICAL = ("parents", "offspring", "parent_total_errors", "parent_framebuffers",
             "offspring_errors", "chain_update_flags", "fitness_packed")
POLICY = {"version": 1, "scope": "provisional exploratory engine screen; never KEEP",
          "minimum_pairs": 5, "maximum_pairs": 10, "resamples": RESAMPLES,
          "bootstrap_seed": BOOTSTRAP_SEED, "interval_percentiles": [2.5, 97.5],
          "interval_method": "paired log-mean percentile bootstrap, independent strata per case",
          "quantiles": "linear interpolation at (n - 1) * p; exponentiate log endpoints",
          "weighting": "equal ff/mona/darwin; geometric mean of per-case paired geometric means",
          "aggregate_ratio_strict_minimum": 1.03, "aggregate_lower_strict_minimum": 1.0,
          "case_median_minimum": 1.0, "case_lower_minimum": 0.99,
          "captured_noise_band_rounded": NOISE_BAND,
          "uncertainty": "conditional on same-seed timing repeats; exploratory, not independent search seeds",
          "exclusions": "no timing-based exclusions; any invalid worker invalidates the entire comparison"}


def terminate_as_cancel(signum, _frame):
    raise KeyboardInterrupt(f"comparison received signal {signum}")


@bench.contextlib.contextmanager
def cancellation_signals():
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, terminate_as_cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    return values[low] + (values[math.ceil(position)] - values[low]) * (position - low)


def statistics(paired_ratios):
    """Return policy-1 log-mean intervals; paired_ratios maps each project to 1..10 ratios."""
    bench.require(set(paired_ratios) == set(bench.PROJECTS), "statistics needs ff, mona, darwin")
    counts = [len(paired_ratios[name]) for name in bench.PROJECTS]
    bench.require(len(set(counts)) == 1 and 1 <= counts[0] <= 10, "unequal or invalid pair counts")
    logs = [[math.log(bench.finite(value, 1e-27, 1e27, "paired ratio"))
             for value in paired_ratios[name]] for name in bench.PROJECTS]
    n = counts[0]
    rng = random.Random(BOOTSTRAP_SEED)
    sampled = [[] for _ in range(4)]
    for _ in range(RESAMPLES):
        means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for values in logs]
        for index, value in enumerate(means):
            sampled[index].append(value)
        sampled[3].append(sum(means) / 3)
    intervals = []
    for values in sampled:
        values.sort()
        intervals.append([math.exp(percentile(values, p)) for p in (0.025, 0.975)])
    cases = {name: {"paired_ratios": bench.distribution(paired_ratios[name]),
                    "geomean_ratio": math.exp(sum(logs[index]) / n),
                    "interval95": intervals[index], "interval_statistic": "exp(mean(log(paired ratio)))"}
             for index, name in enumerate(bench.PROJECTS)}
    return {"cases": cases, "aggregate": {"ratio": math.exp(sum(sum(x) / n for x in logs) / 3),
                                          "interval95": intervals[3]},
            "bootstrap_seed": BOOTSTRAP_SEED, "resamples": RESAMPLES,
            "uncertainty": POLICY["uncertainty"]}


def verify_initial_state(output, record):
    state = bench.object_value(bench.read_json(output / "initial-state.json"), "initial-state.json")
    for field in CANONICAL:
        bench.require(isinstance(state.get(field), list), f"missing initial canonical field: {field}")
    encoded = bench.json.dumps([state[key] for key in CANONICAL], separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode()
    bench.require(bench.hashlib.sha256(encoded).hexdigest() == record["initial_state_sha256"],
                  "initial_state_sha256 differs from saved initial canonical state")
    quality = record["initial_quality"]
    bench.require(len(state["parents"]) == len(quality), "saved initial parent count differs")
    bench.require(state["parent_total_errors"] == [q["stored_error"] for q in quality],
                  "saved initial errors differ from checked quality")
    for parent, checked in zip(state["parents"], quality):
        bench.object_value(parent, "saved initial parent")
        bench.require(parent.get("polygon_count") == checked["triangles"], "saved initial triangles differ")
        bits = bench.integer(parent.get("fitness_bits"), 0, 2**32 - 1, "initial fitness_bits")
        fitness = bench.struct.unpack("<f", bench.struct.pack("<I", bits))[0]
        bench.require(math.isclose(fitness, checked["stored_fitness"], abs_tol=0.00002),
                      "saved initial fitness differs from checked quality")


def control_results(summary, content):
    trials = summary.get("trials")
    bench.require(isinstance(trials, list) and 1 <= len(trials) <= bench.MAX_TRIALS,
                  "invalid baseline trial count")
    for trial in trials:
        bench.object_value(trial, "baseline worker")
        bench.require(trial.get("case_id") in bench.PROJECTS and type(trial.get("diagnostic")) is bool,
                      "unexpected baseline case/diagnostic")
    schedule = summary["arguments"]
    args = argparse.Namespace(seed=bench.integer(schedule.get("seed"), 0, 2**64 - 1, "baseline seed"),
                              batches=bench.integer(schedule.get("batches"), 1, 4096, "baseline batches"),
                              timeout=3600)
    expected, ratios = {}, {}
    for case in content["cases"]:
        selected = [t for t in trials if t.get("case_id") == case["case_id"]]
        normal = [t for t in selected if t.get("diagnostic") is False]
        bench.require(10 <= len(normal) <= 40 and len(normal) % 2 == 0, "baseline needs complete A/A controls")
        reference = normal[0]["result"]
        for trial in selected:
            bench.require(trial.get("valid") is True, "invalid baseline worker")
            bench.validate_result(trial["result"], case, args, trial["diagnostic"], summary["manifest_sha256"])
            bench.require(not bench.replay_errors(reference, trial["result"]), "baseline replays disagree")
        values = []
        for a, b in zip(normal[::2], normal[1::2]):
            bench.require(a.get("label") == "A" and b.get("label") == "B", "baseline A/A order missing")
            values.append(a["result"]["evals_per_second"] / b["result"]["evals_per_second"])
        ratios[case["case_id"]] = values
        expected[case["case_id"]] = reference
    observed = max(abs(value - 1) for values in ratios.values() for value in values)
    bench.require(math.isclose(observed, NOISE_BAND, rel_tol=0, abs_tol=0.0000005),
                  "baseline A/A noise disagrees with frozen policy source value")
    return expected, {"observed_max_absolute_aa_deviation": observed, "ratios": ratios,
                      "rounded_policy_source_verified": True, "source": "validated baseline A/B worker rates"}


def load_baseline(args):
    summary = bench.object_value(bench.read_json(args.baseline), "baseline summary")
    bench.require(summary.get("command") == "baseline" and summary.get("valid") is True
                  and type(summary.get("error_count")) is int and summary["error_count"] == 0
                  and summary.get("errors") == [], "baseline must be a valid error-free capture")
    capture = args.baseline.parent
    binary, manifest = Path(summary["binary"]["archived"]), Path(summary["manifest"])
    bench.require(binary.parent == capture and manifest.parent == capture / "fixtures",
                  "moved/restructured baseline bundles are unsupported")
    bench.require(bench.sha_file(binary, bench.MAX_BINARY_BYTES) == summary["binary"]["sha256"],
                  "baseline archived binary SHA mismatch")
    bench.require(bench.sha_file(manifest) == summary["manifest_sha256"], "baseline manifest SHA mismatch")
    bench.require(bench.inventory(manifest.parent) == summary["fixture_inventory"], "baseline fixtures changed")
    bench.require(bench.inventory(capture / "cache-seed") == summary["cache_policy"]["files"], "baseline cache changed")
    content = bench.load_manifest(manifest)
    expected, noise = control_results(summary, content)
    args.seed = summary["arguments"]["seed"] if args.seed is None else args.seed
    args.batches = summary["arguments"]["batches"] if args.batches is None else args.batches
    args.warmup = bench.integer(summary["arguments"].get("warmup"), 0, 16, "baseline warmup")
    return summary, manifest, content, expected, noise


def archive_sources(output):
    directory = output / "tool-source"
    directory.mkdir()
    sources = [Path(__file__).resolve(), Path(bench.__file__).resolve(),
               bench.REPO / "benchmarks/evolution/v1/comparison-policy.md"]
    archived = {}
    for source in sources:
        digest = bench.sha_file(source)
        target = directory / source.name
        shutil.copyfile(source, target)
        target.chmod(0o444)
        bench.require(bench.sha_file(target) == digest, "tool source changed while archiving")
        archived[str(source)] = {"archived": str(target), "sha256": digest}
    return {"files": archived, "python": sys.version, "platform": platform.platform(),
            "environment": {key: os.environ[key] for key in bench.ENV_KEYS if key in os.environ},
            "binary_source_identity": "supplied/unverified; tool source is not binary build provenance"}


def anchor_checks(args, baseline, summary):
    bench.require(bench.sha_file(args.baseline) == summary["baseline_summary_sha256"], "baseline summary changed")
    bench.require(bench.sha_file(Path(baseline["binary"]["archived"]), bench.MAX_BINARY_BYTES)
                  == baseline["binary"]["sha256"], "baseline executable changed")
    bench.require(bench.inventory(Path(baseline["manifest"]).parent) == baseline["fixture_inventory"],
                  "baseline fixtures changed")
    bench.require(bench.inventory(args.baseline.parent / "cache-seed") == baseline["cache_policy"]["files"],
                  "baseline cache changed")
    for identity in summary["binaries"].values():
        for key in ("source", "archived"):
            bench.require(bench.sha_file(Path(identity[key]), bench.MAX_BINARY_BYTES) == identity["sha256"],
                          f"{key} executable changed")


def checked_trial(args, summary, binary, manifest, case, hashes, side, pair, diagnostic=False):
    directory = args.output / "trials" / f"{case['case_id']}-{pair:02d}-{side}{'-diagnostic' if diagnostic else ''}"
    entry = {"case_id": case["case_id"], "side": side, "pair": pair, "diagnostic": diagnostic,
             "label": side, "directory": str(directory), "valid": False, "status": "invalid", "errors": []}
    summary["trials"].append(entry)
    print(f"{case['case_id']} pair {pair} {side} diagnostic={diagnostic}: started", file=sys.stderr, flush=True)
    cache = args.output / "cache-seed"
    try:
        bench.require(bench.inventory(cache) == summary["cache_policy"]["files"], "fixed cache seed changed")
        identity = summary["binaries"][side]
        bench.require(bench.sha_file(binary, bench.MAX_BINARY_BYTES) == identity["sha256"], "archived executable changed")
        entry.update(bench.execute_trial(args, binary, manifest, case, directory, diagnostic, cache, hashes))
        if entry["valid"]:
            verify_initial_state(directory / "output", entry["result"])
        bench.require(bench.sha_file(binary, bench.MAX_BINARY_BYTES) == identity["sha256"], "archived executable changed")
        bench.require(bench.inventory(cache) == summary["cache_policy"]["files"], "fixed cache seed changed")
    except FAILURES as exc:
        entry["errors"].append(str(exc))
        entry.update(valid=False, status="invalid")
    # Cancellation or an exception can interrupt assignment inside the shared
    # helper. Recover bounded evidence, including any completed counters.
    for key, path in (("result", directory / "output/result.json"),
                      ("process", directory / "process/process.json"),
                      ("before", directory / "before/telemetry.json"),
                      ("after", directory / "after/telemetry.json")):
        if key not in entry and path.is_file():
            try:
                entry[key] = bench.read_json(path, 2 * 1024 * 1024)
            except FAILURES as recovery:
                entry["errors"].append(str(recovery))
                entry.update(valid=False, status="invalid")
    if "after" not in entry and directory.is_dir():
        entry["after"] = bench.telemetry(directory, "after")
    if (directory / "output").is_dir():
        try:
            entry["artifacts"] = bench.inventory(directory / "output")
        except FAILURES as exc:
            entry["errors"].append(str(exc))
            entry.update(valid=False, status="invalid")
    if directory.is_dir():
        bench.write_json(directory / "comparison.json", entry)
    print(f"  {entry['status']}; {len(entry['errors'])} errors", file=sys.stderr, flush=True)
    if entry["status"] == "cancelled":
        raise KeyboardInterrupt
    return entry


def reference_errors(record, expected, repeated, same_seed, same_budget):
    fields = ["initial_quality", "initial_complexity", "initial_similarity_percent", "params", "gpu"]
    if same_seed:
        fields.append("initial_state_sha256")
    errors = [f"baseline initial mismatch: {field}" for field in fields if record.get(field) != expected.get(field)]
    if same_seed and same_budget:
        errors.extend(bench.replay_errors(expected, record))
    if repeated is not None:
        errors.extend(bench.replay_errors(repeated, record))
    return errors


def run_case(args, summary, manifest, case, hashes, expected):
    same_seed = summary["baseline_schedule"]["seed"] == args.seed
    same_budget = summary["baseline_schedule"]["batches"] == args.batches
    repeated = candidate = None
    for index in range(args.pairs):
        sides = ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")
        entries = {side: checked_trial(args, summary, Path(summary["binaries"][side]["archived"]),
                                       manifest, case, hashes, side, index + 1) for side in sides}
        pair = {"case_id": case["case_id"], "pair": index + 1, "order": list(sides),
                "directories": [entries[side]["directory"] for side in sides], "errors": []}
        for entry in entries.values():
            pair["errors"].extend(entry["errors"])
        a, b = entries["reference"], entries["candidate"]
        if a["valid"]:
            pair["errors"].extend(reference_errors(a["result"], expected, repeated, same_seed, same_budget))
            repeated = a["result"] if repeated is None else repeated
        if a["valid"] and b["valid"]:
            pair["errors"].extend(bench.replay_errors(a["result"], b["result"]))
            pair.update(reference_rate=a["result"]["evals_per_second"], candidate_rate=b["result"]["evals_per_second"])
            candidate = b["result"] if candidate is None else candidate
        pair["valid"] = not pair["errors"] and a["valid"] and b["valid"]
        if pair["valid"]:
            pair["ratio"] = pair["candidate_rate"] / pair["reference_rate"]
        summary["pairs"].append(pair)
        summary["errors"].extend(f"{case['case_id']} pair {index + 1}: {error}" for error in pair["errors"])
    diagnostic = checked_trial(args, summary, Path(summary["binaries"]["candidate"]["archived"]),
                               manifest, case, hashes, "candidate", 0, True)
    errors = list(diagnostic["errors"])
    if diagnostic["valid"]:
        errors.extend(bench.replay_errors(candidate, diagnostic["result"]) if candidate else ["no valid candidate replay anchor"])
    summary["errors"].extend(f"{case['case_id']} diagnostic: {error}" for error in errors)


def run_comparison(args, summary):
    baseline, source, content, expected, noise = load_baseline(args)
    summary.update(baseline_summary_sha256=bench.sha_file(args.baseline), baseline_noise=noise,
                   baseline_schedule={key: baseline["arguments"][key] for key in ("seed", "batches", "warmup")},
                   schedule={"seed": args.seed, "batches": args.batches, "warmup": args.warmup, "pairs": args.pairs},
                   baseline_final_replay_required=(args.seed == baseline["arguments"]["seed"]
                                                   and args.batches == baseline["arguments"]["batches"]))
    summary["baseline_replay_scope"] = ("full canonical/quality replay" if summary["baseline_final_replay_required"]
                                        else "different seed/budget: initial quality only; full paired and repeated replay required")
    summary["provenance"] = archive_sources(args.output)
    manifest, _, hashes = bench.prepare_fixtures(source, args.output)
    summary.update(manifest=str(manifest), manifest_sha256=bench.sha_file(manifest), fixture_inventory=hashes,
                   cache_policy=baseline["cache_policy"])
    bench.copy_inventory(args.baseline.parent / "cache-seed", args.output / "cache-seed", baseline["cache_policy"]["files"])
    for side in ("reference", "candidate"):
        directory = args.output / side
        directory.mkdir()
        _, identity = bench.prepare_binary(argparse.Namespace(binary=getattr(args, side)), directory)
        identity["source_identity"] = "supplied/unverified"
        summary["binaries"][side] = identity
    bench.write_json(args.output / "policy.json", dict(POLICY, baseline_noise=noise, schedule=summary["schedule"],
                                                       tool_sources=summary["provenance"]["files"]))
    (args.output / "trials").mkdir()
    try:
        for case in content["cases"]:
            run_case(args, summary, manifest, case, hashes, expected[case["case_id"]])
    finally:
        anchor_checks(args, baseline, summary)
        bench.require(bench.inventory(manifest.parent) == hashes, "comparison fixtures changed")


def require_complete_pairs(summary):
    n = bench.integer(summary.get("schedule", {}).get("pairs"), 1, 10, "scheduled pairs")
    pairs = summary["pairs"]
    bench.require(len(pairs) == len(bench.PROJECTS) * n, "incomplete comparison pair count")
    for name in bench.PROJECTS:
        selected = [p for p in pairs if p.get("case_id") == name]
        bench.require(len(selected) == n and {p.get("pair") for p in selected} == set(range(1, n + 1)),
                      f"incomplete comparison pairs for {name}")
        for pair in selected:
            bench.require(pair.get("valid") is True, "invalid pair cannot enter statistics")
            bench.finite(pair.get("ratio"), 1e-27, 1e27, "paired ratio")


def finish(summary):
    if not summary["errors"]:
        try:
            require_complete_pairs(summary)
        except FAILURES as exc:
            summary["errors"].append(str(exc))
    summary.update(valid=not summary["errors"], error_count=len(summary["errors"]),
                   completed_work=bench.completed_work(summary["trials"]), finished_utc=bench.utc_now())
    summary["status"] = "cancelled" if summary.get("cancelled") else "completed" if summary["valid"] else "invalid"
    summary["candidate_screen"] = "failed"
    summary["verdict"] = "REJECT" if not summary["valid"] else "INCONCLUSIVE"
    summary["statistics"] = None
    summary["worker_counts"] = {status: sum(t.get("process", {}).get("status") == status for t in summary["trials"])
                                for status in ("completed", "failed", "timed_out", "cancelled")}
    summary["worker_counts"].update(attempted=len(summary["trials"]), invalid=sum(not t["valid"] for t in summary["trials"]))
    submitted = {f"submitted_{unit}": 0 for unit in ("batches", "iterations", "evaluations")}
    for entry in summary["trials"]:
        entry["timing_score_eligible"] = summary["valid"] and not entry["diagnostic"]
        record = entry.get("result")
        counters = record.get("counters") if isinstance(record, dict) else None
        if isinstance(counters, dict):
            for key in submitted:
                value = counters.get(key)
                if type(value) is int and value >= 0:
                    submitted[key] += value
    summary["completed_work"]["reported_submitted_including_invalid_and_diagnostic"] = submitted
    summary["completed_work"]["workers_without_reported_counters"] = sum(
        not isinstance(t.get("result"), dict) or not isinstance(t["result"].get("counters"), dict)
        for t in summary["trials"])
    summary["gates_missing"] = ["live_preview_acceptance", "held_out_mature_states", "independent_search_seeds",
                                "clean_run_qualification", "broader_campaign_qualification"]
    if not summary["valid"]:
        summary["gates_missing"].append("valid_complete_comparison")
        return
    pairs = summary["pairs"]
    stats = statistics({name: [p["ratio"] for p in pairs if p["case_id"] == name] for name in bench.PROJECTS})
    for name, case in stats["cases"].items():
        selected = [p for p in pairs if p["case_id"] == name]
        case["rates"] = {side: bench.distribution([p[f"{side}_rate"] for p in selected])
                         for side in ("reference", "candidate")}
    aggregate, cases = stats["aggregate"], list(stats["cases"].values())
    gates = {"minimum_pairs": summary["schedule"]["pairs"] >= 5,
             "practical_gain": aggregate["ratio"] > 1.03 and aggregate["ratio"] > 1 + NOISE_BAND,
             "aggregate_interval": aggregate["interval95"][0] > 1,
             "case_medians": all(c["paired_ratios"]["median"] >= 1 for c in cases),
             "case_intervals": all(c["interval95"][0] >= 0.99 for c in cases)}
    summary.update(statistics=stats, screen_gates=gates,
                   candidate_screen="passed" if all(gates.values()) else "failed")
    summary["gates_missing"].extend(key for key, passed in gates.items() if not passed)
    if gates["minimum_pairs"] and (not gates["practical_gain"] or not gates["case_medians"]):
        summary["verdict"] = "REJECT"


def execute(args):
    output = args.output.resolve()
    for source in (args.baseline.parent, bench.REPO / "projects"):
        root = source.resolve()
        bench.require(not output.is_relative_to(root) and not root.is_relative_to(output), "output overlaps baseline/projects")
    for source in (args.reference, args.candidate, LOCK, Path(__file__), Path(bench.__file__)):
        bench.require(not source.resolve().is_relative_to(output), "output overlaps an input/lock")
    summary = {"protocol_version": 1, "command": "compare", "policy": POLICY, "baseline": str(args.baseline),
               "started_utc": bench.utc_now(), "evidence_directory": str(args.output), "errors": [],
               "trials": [], "pairs": [], "binaries": {}, "isolation_state": "unverified", "clean_baseline": None,
               "campaign_frozen": False, "optimization_claim": False, "primary_score_eligible": False,
               "isolation_policy": "retain telemetry and competing work; no timing-based filtering",
               "repetitions_are_independent_search_seeds": False}
    with cancellation_signals(), bench.benchmark_lock(LOCK):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.mkdir()
        try:
            run_comparison(args, summary)
        except FAILURES as exc:
            summary["errors"].append(str(exc))
        except KeyboardInterrupt:
            summary["errors"].append("comparison cancelled")
            summary["cancelled"] = True
        finally:
            finish(summary)
            bench.write_json(args.output / "summary.json", summary)
    print(args.output / "summary.json")
    return 0 if summary["valid"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "reference", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=bench.bounded_int(1, 10), default=5)
    parser.add_argument("--batches", type=bench.bounded_int(1, 4096), help="default: captured fixed budget, normally 64")
    parser.add_argument("--seed", type=bench.bounded_int(0, 2**64 - 1), help="default: captured seed, normally 7")
    parser.add_argument("--timeout", type=bench.timeout_arg, default=600.0)
    args = parser.parse_args(argv)
    if args.output is None:
        stamp = bench.datetime.datetime.now(bench.datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        args.output = bench.DEFAULT_ROOT / f"compare-{stamp}-{os.getpid()}"
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, Path(os.path.abspath(value.expanduser())))
    try:
        return execute(args)
    except FAILURES as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
