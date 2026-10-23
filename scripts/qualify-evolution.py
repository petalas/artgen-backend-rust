#!/usr/bin/env python3
"""Capped reserved-restart and live confirmation. No invocation adopts a shader.

Run ``engine --help`` or ``live --help``. A package ledger permits staged commands
to share one immutable 3600 second monotonic deadline, including lock acquisition.
The parent owns corpus construction, builds and final campaign decisions.

Power of Ten deviations 3/8/9: Python, JSON, argparse, signal callbacks and the
two named local helper modules allocate and dispatch dynamically. Bounds are
three cases, 45 engine or 20 live workers, 128 publications per live worker,
10,000 resamples, 512 MiB per binary, and the existing inventory/JSON limits.
Raw receipts use at most three 64 MiB shards, 128 MiB total and 8 MiB per line.
One child runs at a time; existing run_process kills/reaps only its owned child.
Rule 10: validate using warning-enabled stdlib unittest. CI is outside this slice.
Functions remain within 60 physical lines. Synthetic tests prove collector
behavior; they do not prove GPU correctness or performance.
"""

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import signal
import struct
import sys
import time
from pathlib import Path


def load_comparison():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        path = Path(__file__).with_name("compare-evolution.py")
        spec = importlib.util.spec_from_file_location("qualification_comparison", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load local comparison helpers")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


compare = load_comparison()
bench = compare.bench
LOCK = compare.LOCK
SEEDS = {"ff": 11, "mona": 13, "darwin": 17}
PACKAGE_SECONDS = 3600
WORKER_SECONDS = 120
# Existing helper cleanup can reap for 5s, then collect bounded GPU/WS telemetry.
# Reserve time within the same wall budget for that work and the final summary.
CLEANUP_SECONDS = 30
RESULT_BYTES = 2 * 1024 * 1024
RECEIPT_SHARD_BYTES = 64 * 1024 * 1024
RECEIPT_TOTAL_BYTES = 128 * 1024 * 1024
RECEIPT_LINE_BYTES = 8 * 1024 * 1024
FAILURES = compare.FAILURES + (RecursionError,)
LIVE_POLICY = {
    "version": 2, "case_id": "mona", "seed": 13, "batches": 256,
    "warmup_batches": 2, "publish_every": 2, "deadline_ms": 300,
    "completed_evaluations": 4194304, "publications": 128, "pairs": 5,
    "resamples": 10000, "bootstrap_seed": 20261023,
    "interval_percentiles": [2.5, 97.5],
    "interval_method": "paired run log-mean percentile bootstrap; no message resampling",
    "aa_equivalence_band": [0.99, 1.01],
    "throughput_median_minimum": 1.0, "throughput_lower_minimum": 0.99,
    "latency_median_maximum": 1.0, "latency_upper_maximum": 1.01,
    "uncertainty": "conditional on five same-seed run pairs; not independent evolution seeds",
}


class BudgetExpired(bench.BenchError):
    """The frozen package deadline leaves no further worker budget."""


class GateStop(Exception):
    """A terminal frozen gate prevents further planned work."""


def package_clock(path):
    now = time.monotonic()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if path is None:
        return {"protocol_version": 1, "clock_boot_id": boot,
                "started_monotonic_seconds": now, "deadline_monotonic_seconds": now + PACKAGE_SECONDS}
    value = bench.object_value(bench.read_json(path, RESULT_BYTES), "package ledger")
    bench.require(set(value) == {"protocol_version", "clock_boot_id", "started_monotonic_seconds",
                                "deadline_monotonic_seconds"}, "unexpected package ledger fields")
    bench.integer(value.get("protocol_version"), 1, 1, "ledger.protocol_version")
    bench.require(value.get("clock_boot_id") == boot, "package ledger belongs to a different boot")
    start = bench.finite(value.get("started_monotonic_seconds"), 0, now, "package start")
    end = bench.finite(value.get("deadline_monotonic_seconds"), 0, now + PACKAGE_SECONDS, "package deadline")
    bench.require(math.isclose(end - start, PACKAGE_SECONDS, rel_tol=0, abs_tol=1e-6),
                  "package ledger must freeze exactly 3600 seconds; extensions forbidden")
    bench.require(end - now > CLEANUP_SECONDS, "package deadline exhausted")
    return value


@contextlib.contextmanager
def package_signals(clock):
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    bench.require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "existing real timer is active")

    def stop(signum, _frame):
        for owned in handlers:
            signal.signal(owned, signal.SIG_IGN)
        if signum == signal.SIGALRM:
            raise BudgetExpired("package wall deadline reached; reserved cleanup budget")
        raise KeyboardInterrupt(f"qualification received signal {signum}")

    try:
        for signum in handlers:
            signal.signal(signum, stop)
        remaining = clock["deadline_monotonic_seconds"] - time.monotonic() - CLEANUP_SECONDS
        bench.require(remaining > 0, "package deadline exhausted before lock")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


def engine_plan():
    planned = []
    for name in bench.PROJECTS:
        for pair in range(1, 6):
            sides = ("reference", "candidate") if pair % 2 else ("candidate", "reference")
            for side in sides:
                planned.append(worker_spec(name, side, pair, 64, "timing"))
        planned.append(worker_spec(name, "candidate", 0, 64, "diagnostic"))
        for batches in (8, 32):
            for side in ("reference", "candidate"):
                planned.append(worker_spec(name, side, 0, batches, "prefix"))
    return planned


def worker_spec(name, side, pair, batches, kind, phase="engine"):
    identity = f"{phase}-{name}-{kind}-{batches}-{pair:02d}-{side}"
    return {"id": identity, "case_id": name, "side": side, "pair": pair,
            "seed": SEEDS[name], "batches": batches, "kind": kind, "phase": phase,
            "diagnostic": kind == "diagnostic", "status": "unrun", "valid": False,
            "errors": [], "timing_score_eligible": False}


def live_plan():
    planned = []
    for phase in ("aa", "ab"):
        for pair in range(1, 6):
            sides = ("reference", "candidate") if pair % 2 else ("candidate", "reference")
            for side in sides:
                planned.append(worker_spec("mona", side, pair, 256, "live", phase))
    return planned


def validate_reserved(content, original):
    bench.require(content["params"] == original["params"], "reserved settings differ from original capture")
    params = content["params"]
    for key, value in (("chainCount", 4), ("lambda", 64), ("gpuBatchIters", 64)):
        bench.require(type(params.get(key)) is int and params[key] == value, f"fixed setting {key} differs")
    originals = {case["case_id"]: case for case in original["cases"]}
    bench.require(len(content["cases"]) == 3, "reserved corpus needs exactly three cases")
    for case in content["cases"]:
        source = originals[case["case_id"]]
        bench.require(case["drawing_sha256"] != source["drawing_sha256"],
                      f"{case['case_id']} reserved drawing duplicates original development fixture")
        for field in ("reference_sha256", "source_image_sha256", "width", "height"):
            bench.require(case[field] == source[field], f"reserved reference metadata differs: {field}")


def archive_file(source, target, limit=bench.MAX_FILE_BYTES):
    digest = bench.sha_file(source, limit)
    shutil.copyfile(source, target)
    target.chmod(0o444)
    bench.require(bench.sha_file(target, limit) == digest, "input changed while archiving")
    return {"source": str(source), "archived": str(target), "sha256": digest, "limit_bytes": limit}


def provenance(args, summary):
    directory = args.output / "tool-source"
    directory.mkdir()
    sources = (Path(__file__).resolve(), Path(compare.__file__).resolve(), Path(bench.__file__).resolve(),
               bench.REPO / "benchmarks/evolution/v1/comparison-policy.md")
    summary["tool_sources"] = [archive_file(source, directory / source.name) for source in sources]
    summary["runtime"] = {"python": sys.version, "platform": compare.platform.platform(),
                          "environment": {k: os.environ[k] for k in bench.ENV_KEYS if k in os.environ}}
    for side in ("reference", "candidate"):
        directory = args.output / side
        directory.mkdir()
        _, identity = bench.prepare_binary(argparse.Namespace(binary=getattr(args, side)), directory)
        identity["source_identity"] = "supplied/unverified"
        supplied = getattr(args, side + "_provenance")
        if supplied is not None:
            bench.object_value(bench.read_json(supplied, RESULT_BYTES), "supplied build provenance")
            identity["supplied_provenance"] = archive_file(supplied, directory / "supplied-provenance.json")
        summary["binaries"][side] = identity


def prepare(args, summary):
    baseline_args = argparse.Namespace(baseline=args.capture, seed=None, batches=None)
    baseline, _, original, _, noise = compare.load_baseline(baseline_args)
    bench.require(baseline_args.warmup == 2, "original capture warmup differs from fixed two batches")
    content = bench.load_manifest(args.manifest)
    validate_reserved(content, original)
    summary["capture"] = archive_file(args.capture, args.output / "original-capture.json")
    summary["capture_cache_source"] = str(args.capture.parent / "cache-seed")
    summary["original_capture_validated"] = True
    summary["original_capture_scope"] = "integrity and frozen policy1 noise source; clean baseline remains unverified"
    summary["baseline_noise"] = noise
    summary["cache_policy"] = baseline["cache_policy"]
    summary["original_fixture_inventory"] = baseline["fixture_inventory"]
    summary["original_manifest"] = baseline["manifest"]
    summary["original_binary"] = baseline["binary"]
    summary["input_manifest"] = archive_file(args.manifest, args.output / "input-manifest.json")
    manifest, content, hashes = bench.prepare_fixtures(args.manifest, args.output)
    summary.update(manifest=str(manifest), manifest_sha256=bench.sha_file(manifest), fixture_inventory=hashes)
    bench.copy_inventory(args.capture.parent / "cache-seed", args.output / "cache-seed", baseline["cache_policy"]["files"])
    provenance(args, summary)
    if args.package_ledger is not None:
        summary["ledger_identity"] = archive_file(args.package_ledger, args.output / "source-ledger.json")
    for name in ("trials", "inputs"):
        (args.output / name).mkdir()
    bench.write_json(args.output / "policy.json", summary["policy"])
    bench.write_json(args.output / "planned-work.json", summary["planned_work"])
    return content


def check_identity(identity, binary=False):
    limit = bench.MAX_BINARY_BYTES if binary else bench.MAX_FILE_BYTES
    for key in ("source", "archived"):
        bench.require(bench.sha_file(Path(identity[key]), limit) == identity["sha256"], f"{key} input identity changed")


def anchors(args, summary):
    for key in ("capture", "input_manifest", "ledger_identity"):
        if key in summary:
            check_identity(summary[key])
    for identity in summary.get("tool_sources", []):
        check_identity(identity)
    for identity in summary["binaries"].values():
        check_identity(identity, True)
        if "supplied_provenance" in identity:
            check_identity(identity["supplied_provenance"])
    if "cache_policy" in summary:
        expected = summary["cache_policy"]["files"]
        for path in (args.capture.parent / "cache-seed", args.output / "cache-seed"):
            bench.require(bench.inventory(path) == expected, "immutable cache seed changed")
    if "original_manifest" in summary:
        bench.require(bench.inventory(Path(summary["original_manifest"]).parent) == summary["original_fixture_inventory"],
                      "original capture fixtures changed")
        original = summary["original_binary"]
        bench.require(bench.sha_file(Path(original["archived"]), bench.MAX_BINARY_BYTES) == original["sha256"],
                      "original capture binary changed")
    if "fixture_inventory" in summary:
        for root in (args.manifest.parent, args.output / "fixtures"):
            bench.require(bench.inventory(root) == summary["fixture_inventory"], "immutable fixture inventory changed")


def worker_timeout(args, clock):
    remaining = clock["deadline_monotonic_seconds"] - time.monotonic() - CLEANUP_SECONDS
    if remaining <= 0:
        raise BudgetExpired("package deadline exhausted before worker")
    return min(args.timeout, WORKER_SECONDS, remaining)


def recover(entry, directory):
    for key, relative in (("result", "output/result.json"), ("process", "process/process.json"),
                          ("before", "before/telemetry.json"), ("after", "after/telemetry.json")):
        path = directory / relative
        if key not in entry and path.is_file():
            try:
                entry[key] = bench.read_json(path, RESULT_BYTES)
            except FAILURES as exc:
                entry["errors"].append(str(exc))
                entry.update(valid=False, status="invalid")
    if (directory / "output").is_dir():
        try:
            entry["artifacts"] = bench.inventory(directory / "output")
        except FAILURES as exc:
            entry["errors"].append(str(exc))
            entry.update(valid=False, status="invalid")


def retain_finite_result(entry):
    """Keep raw JSON on disk; mark overflowed floats in the summary explicitly.

    JSON's finite decimal exponent syntax can overflow Python's float parser.
    Rejecting that value must not prevent the failure summary or lose counters.
    This bounded iterative pass covers every result root read under RESULT_BYTES.
    """
    if "result" not in entry:
        return
    root = [entry["result"]]
    pending = [root]
    for _ in range(RESULT_BYTES + 1):
        if not pending:
            entry["result"] = root[0]
            return
        container = pending.pop()
        keys = range(len(container)) if isinstance(container, list) else container
        for key in keys:
            value = container[key]
            if isinstance(value, (dict, list)):
                pending.append(value)
            elif isinstance(value, float) and not math.isfinite(value):
                container[key] = f"<invalid non-finite number: {value}>"
                entry["errors"].append("result contains a non-finite number; exact raw JSON retained")
                entry.update(valid=False, status="invalid")
    raise bench.BenchError("result object traversal exceeds byte-derived node bound")


def checked_worker(args, summary, content, planned):
    entry = dict(planned, status="started", errors=[], valid=False)
    summary["trials"].append(entry)
    planned["status"] = "started"
    directory = args.output / "trials" / entry["id"]
    entry["directory"] = str(directory)
    case = next(c for c in content["cases"] if c["case_id"] == entry["case_id"])
    try:
        anchors(args, summary)
        inputs = args.output / "inputs" / entry["id"]
        inputs.mkdir()
        fixtures = inputs / "fixtures"
        bench.copy_inventory(args.output / "fixtures", fixtures, summary["fixture_inventory"])
        manifest = fixtures / Path(summary["manifest"]).name
        local = argparse.Namespace(seed=entry["seed"], batches=entry["batches"], warmup=2,
                                   timeout=worker_timeout(args, summary["package_clock"]))
        source_side = "reference" if entry["phase"] == "aa" else entry["side"]
        binary = Path(summary["binaries"][source_side]["archived"])
        entry["binary_side"] = source_side
        if entry["kind"] == "live":
            entry.update(execute_live(local, binary, manifest, case, directory, args.output / "cache-seed"))
        else:
            entry.update(bench.execute_trial(local, binary, manifest, case, directory, entry["diagnostic"],
                                             args.output / "cache-seed", summary["fixture_inventory"]))
        if entry.get("status") == "cancelled":
            raise KeyboardInterrupt("worker cancelled")
        if entry["valid"]:
            compare.verify_initial_state(directory / "output", entry["result"])
            bench.verify_saved_state(directory / "output", entry["result"])
        bench.require(bench.inventory(fixtures) == summary["fixture_inventory"], "worker fixture integrity changed")
        anchors(args, summary)
    except FAILURES as exc:
        entry["errors"].append(str(exc))
        entry.update(valid=False, status="invalid")
    except KeyboardInterrupt:
        entry["errors"].append("qualification cancelled")
        entry.update(valid=False, status="cancelled")
        raise
    finally:
        recover(entry, directory)
        retain_finite_result(entry)
        planned["status"] = entry["status"]
        if directory.is_dir():
            bench.write_json(directory / "qualification.json", entry)
    bench.require(entry["valid"], f"{entry['id']}: {'; '.join(entry['errors']) or entry['status']}")
    return entry


def require_replay(anchor, entry):
    errors = bench.replay_errors(anchor["result"], entry["result"])
    if errors:
        entry["errors"].extend(errors)
        entry.update(valid=False, status="invalid")
        bench.write_json(Path(entry["directory"]) / "replay-failure.json",
                         {"anchor": anchor["id"], "worker": entry["id"], "errors": errors})
    bench.require(not errors, f"{entry['id']}: {'; '.join(errors)}")


def add_pair(summary, entries, phase="engine"):
    a, b = entries["reference"], entries["candidate"]
    require_replay(a, b)
    pair = {"phase": phase, "case_id": a["case_id"], "pair": a["pair"], "valid": True,
            "order": list(entries), "workers": [entry["id"] for entry in entries.values()],
            "reference_rate": a["result"]["evals_per_second"], "candidate_rate": b["result"]["evals_per_second"]}
    pair["ratio"] = pair["candidate_rate"] / pair["reference_rate"]
    if phase != "engine":
        pair["latency_ratio"] = b["result"]["p95_latency_ms"] / a["result"]["p95_latency_ms"]
        pair["missed_deadline_increase"] = b["result"]["missed_deadlines"] - a["result"]["missed_deadlines"]
    summary["pairs"].append(pair)


def engine_analysis(summary):
    proxy = {"errors": [], "trials": [], "schedule": {"pairs": 5}, "pairs": summary["pairs"]}
    compare.finish(proxy)
    bench.require(proxy["valid"], "incomplete engine pair matrix")
    summary.update(statistics=proxy["statistics"], screen_gates=proxy["screen_gates"],
                   verdict=proxy["verdict"], engine_timing_confirmation=proxy["candidate_screen"])
    if proxy["candidate_screen"] != "passed":
        summary["reserved_confirmation"] = "failed" if proxy["verdict"] == "REJECT" else "inconclusive"
        summary["stop_reason"] = "frozen policy1 engine gates did not pass; no additional samples"
        raise GateStop(summary["stop_reason"])


def run_engine(args, summary, content):
    replay = {}
    current_pair = {}
    for planned in summary["planned_work"]:
        entry = checked_worker(args, summary, content, planned)
        key = (entry["case_id"], entry["batches"])
        if key in replay:
            require_replay(replay[key], entry)
        else:
            replay[key] = entry
        if entry["kind"] != "timing":
            continue
        current_pair[entry["side"]] = entry
        if len(current_pair) != 2:
            continue
        add_pair(summary, current_pair)
        current_pair = {}
        if entry["pair"] == 5:
            ratios = [p["ratio"] for p in summary["pairs"] if p["case_id"] == entry["case_id"]]
            if bench.statistics.median(ratios) < compare.POLICY["case_median_minimum"]:
                summary.update(verdict="REJECT", reserved_confirmation="failed",
                               stop_reason=f"{entry['case_id']} five-pair median demonstrates slowdown")
                raise GateStop(summary["stop_reason"])
        if len(summary["pairs"]) == 15:
            engine_analysis(summary)
    summary["reserved_confirmation"] = "passed"


def paired_log_bootstrap(ratios):
    """Single-case run-pair intervals. Never invent additional case strata."""
    bench.require(isinstance(ratios, list) and len(ratios) == 5, "live statistics require exactly five run pairs")
    logs = [math.log(bench.finite(x, 1e-27, 1e27, "live paired ratio")) for x in ratios]
    rng = random.Random(LIVE_POLICY["bootstrap_seed"])
    samples = [sum(logs[rng.randrange(5)] for _ in range(5)) / 5 for _ in range(LIVE_POLICY["resamples"])]
    samples.sort()
    return {"paired_ratios": bench.distribution(ratios), "geomean_ratio": math.exp(sum(logs) / 5),
            "interval95": [math.exp(compare.percentile(samples, p)) for p in (0.025, 0.975)],
            "resamples": LIVE_POLICY["resamples"], "bootstrap_seed": LIVE_POLICY["bootstrap_seed"],
            "unit": "run pair", "uncertainty": LIVE_POLICY["uncertainty"]}


def live_gates(pairs, aa):
    throughput = paired_log_bootstrap([p["ratio"] for p in pairs])
    latency = paired_log_bootstrap([p["latency_ratio"] for p in pairs])
    if aa:
        gates = {"throughput_equivalence": throughput["interval95"][0] >= 0.99 and throughput["interval95"][1] <= 1.01,
                 "latency_equivalence": latency["interval95"][0] >= 0.99 and latency["interval95"][1] <= 1.01}
    else:
        gates = {"throughput_median": throughput["paired_ratios"]["median"] >= 1,
                 "throughput_lower": throughput["interval95"][0] >= 0.99,
                 "latency_median": latency["paired_ratios"]["median"] <= 1,
                 "latency_upper": latency["interval95"][1] <= 1.01,
                 "no_deadline_increase": all(p["missed_deadline_increase"] <= 0 for p in pairs)}
    return {"throughput": throughput, "latency": latency, "gates": gates, "passed": all(gates.values())}


def validate_publications(record):
    publications = record.get("publications")
    bench.require(isinstance(publications, list) and len(publications) == 128, "live requires exactly 128 publication records")
    run_id = record.get("run_id")
    bench.require(isinstance(run_id, str) and 1 <= len(run_id) <= 256, "invalid live run_id")
    latencies = []
    previous_times = [0, 0, 0]
    for sequence, publication in enumerate(publications, 1):
        bench.object_value(publication, "publication")
        for field, expected in (("run_id", run_id), ("sequence", sequence), ("completed_batches", sequence * 2),
                                ("completed_evaluations", sequence * 32768), ("payload_valid", True)):
            bench.require(type(publication.get(field)) is type(expected) and publication[field] == expected,
                          f"publication {field} mismatch")
        times = [bench.finite(publication.get(key), 0, record["elapsed_seconds"] * 1000, key)
                 for key in ("due_ms", "send_ms", "receive_ms")]
        bench.require(times == sorted(times), "publication times out of order")
        bench.require(all(now >= previous for now, previous in zip(times, previous_times, strict=True)),
                      "publication timeline regressed")
        previous_times = times
        for field in ("drawing_sha256", "png_sha256"):
            bench.digest_value(publication.get(field), field)
        bench.finite(publication.get("fitness_percent"), 0, 100, "publication fitness")
        bench.integer(publication.get("polygons"), 1, 1000, "publication polygons")
        latencies.append(times[2] - times[0])
    missed = sum(value > LIVE_POLICY["deadline_ms"] for value in latencies)
    bench.require(type(record.get("missed_deadlines")) is int and record["missed_deadlines"] == missed,
                  "live missed deadline count disagrees with receipts")
    p95 = bench.finite(record.get("p95_latency_ms"), 1e-9, WORKER_SECONDS * 1000, "p95_latency_ms")
    latencies.sort()
    bench.require(math.isclose(p95, compare.percentile(latencies, 0.95), rel_tol=1e-6, abs_tol=1e-6),
                  "live p95 latency disagrees with publication receipts")
    bench.require(missed == 0, "live publication deadline was missed")


def validate_live(record, case, args, manifest_hash):
    bench.object_value(record, "live result")
    bench.integer(record.get("live_protocol_version"), 2, 2, "live_protocol_version")
    bench.validate_result(record, case, args, False, manifest_hash)
    for field, expected in (("mode", "live_acceptance"), ("warmup_batches", 2), ("publish_every", 2),
                            ("deadline_ms", 300), ("previews_due", 128), ("previews_sent", 128),
                            ("previews_received", 128)):
        bench.require(type(record.get(field)) is type(expected) and record[field] == expected,
                      f"live {field} differs from frozen contract")
    bench.require(record["counters"]["completed_evaluations"] == 4194304, "live completed evaluation count differs")
    validate_publications(record)
    best = max(q["stored_fitness"] for q in record["final_quality"])
    bench.require(math.isclose(record["publications"][-1]["fitness_percent"], best, rel_tol=0, abs_tol=0.00002),
                  "terminal publication does not attain packed population maximum")


def receipt_object(raw, name):
    try:
        value = json.loads(raw, parse_constant=bench.reject_constant, object_pairs_hook=bench.unique_object)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise bench.BenchError(f"invalid {name} JSON: {exc}") from exc
    return bench.object_value(value, name)


def receipt_shards(record, output):
    artifacts = bench.object_value(record.get("receipt_artifacts"), "receipt_artifacts")
    bench.require(1 <= len(artifacts) <= 3, "receipt artifact count must be 1..3")
    names = [f"receipt-{index:03d}.jsonl" for index in range(len(artifacts))]
    bench.require(set(artifacts) == set(names), "invalid receipt shard names")
    total = 0
    for name in names:
        metadata = bench.object_value(artifacts[name], "receipt shard metadata")
        total += bench.integer(metadata.get("bytes"), 1, RECEIPT_SHARD_BYTES, "receipt shard bytes")
        bench.digest_value(metadata.get("sha256"), "receipt shard sha256")
    bench.require(total <= RECEIPT_TOTAL_BYTES, "receipt artifacts exceed total byte limit")
    paths = []
    for name in names:
        path = output / name
        bench.require(path.is_file() and not path.is_symlink(), f"missing or linked receipt shard: {name}")
        bench.require(path.stat().st_size == artifacts[name]["bytes"], f"receipt shard byte count differs: {name}")
        paths.append(path)
    return paths


def validate_receipt_payload(frame, publication, case):
    drawing_json = frame.get("drawingJson")
    bench.require(isinstance(drawing_json, str), "receipt drawingJson must be a string")
    bench.require(hashlib.sha256(drawing_json.encode("utf-8")).hexdigest() == publication["drawing_sha256"],
                  "receipt drawing hash differs from publication")
    drawing = receipt_object(drawing_json, "receipt drawing")
    polygons = drawing.get("polygons")
    bench.require(isinstance(polygons, list) and len(polygons) == publication["polygons"],
                  "receipt drawing polygon count differs from publication")
    bench.require(type(frame.get("polygons")) is int and frame["polygons"] == len(polygons),
                  "receipt wire polygon count differs from drawing")
    expected_score = struct.pack("!f", publication["fitness_percent"])
    for score in (drawing.get("fitness"), frame.get("fitness")):
        value = bench.finite(score, 0, 100, "receipt fitness")
        bench.require(struct.pack("!f", value) == expected_score, "receipt fitness differs from publication")
    encoded = frame.get("image")
    bench.require(isinstance(encoded, str), "receipt image must be base64 text")
    try:
        png = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise bench.BenchError(f"invalid receipt base64 image: {exc}") from exc
    bench.require(hashlib.sha256(png).hexdigest() == publication["png_sha256"],
                  "receipt PNG hash differs from publication")
    bench.require(len(png) >= 33 and png[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
                  "receipt PNG has no valid IHDR")
    bench.require(struct.unpack(">II", png[16:24]) == (case["width"], case["height"]),
                  "receipt PNG dimensions differ from fixture")


def validate_receipt_frame(frame, publication, case):
    bench.require(frame.get("type") == "update", "receipt is not a production update frame")
    metadata = bench.object_value(frame.get("liveAcceptance"), "receipt liveAcceptance")
    for key in ("run_id", "sequence", "completed_batches", "completed_evaluations", "due_ms", "send_ms",
                "drawing_sha256", "png_sha256"):
        value, expected = metadata.get(key), publication[key]
        numeric_time = key in ("due_ms", "send_ms") and type(value) in (int, float)
        bench.require((type(value) is type(expected) or numeric_time) and value == expected,
                      f"receipt {key} differs from publication")
    bench.require(type(frame.get("totalEvals")) is int
                  and frame["totalEvals"] == publication["completed_evaluations"],
                  "receipt wire completed-work origin differs from publication")
    validate_receipt_payload(frame, publication, case)


def validate_receipts(record, output, case):
    paths = receipt_shards(record, output)
    count = total = 0
    for path in paths:
        shard_bytes = 0
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for _ in range(129):
                line = stream.readline(RECEIPT_LINE_BYTES + 2)
                if not line:
                    break
                bench.require(line.endswith(b"\n") and len(line) <= RECEIPT_LINE_BYTES + 1,
                              "receipt line exceeds byte limit or lacks newline")
                count += 1
                total += len(line)
                shard_bytes += len(line)
                digest.update(line)
                bench.require(count <= 128, "receipt line count exceeds 128 publications")
                bench.require(total <= RECEIPT_TOTAL_BYTES, "receipt artifacts exceed total byte limit")
                frame = receipt_object(line, "raw receipt")
                validate_receipt_frame(frame, record["publications"][count - 1], case)
        bench.require(shard_bytes == record["receipt_artifacts"][path.name]["bytes"],
                      "receipt shard changed during audit")
        bench.require(digest.hexdigest() == record["receipt_artifacts"][path.name]["sha256"],
                      "receipt shard hash differs from declaration")
    bench.require(count == 128, "receipt line count must match 128 publications")
    return {"shards": len(paths), "publications": count, "bytes": total,
            "scope": "raw frame identities, payload hashes, PNG dimensions and wire fitness/geometry; backend owns rendering"}


def execute_live(args, binary, manifest, case, directory, cache):
    directory.mkdir()
    cwd = directory / "cwd"
    bench.copy_inventory(cache, cwd, bench.inventory(cache), writable=True)
    output = directory / "output"
    entry = {"valid": False, "errors": [], "before": bench.telemetry(directory, "before")}
    command = [str(binary), "--live-acceptance", "--manifest", str(manifest), "--case", "mona",
               "--seed", "13", "--batches", "256", "--warmup", "2", "--publish-every", "2",
               "--deadline-ms", "300", "--output", str(output)]
    try:
        entry["process"] = bench.run_process(command, cwd, directory / "process", args.timeout)
        entry["status"] = entry["process"]["status"]
        if (output / "result.json").is_file():
            entry["result"] = bench.read_json(output / "result.json", RESULT_BYTES)
        bench.require(entry["status"] == "completed", f"live subprocess {entry['status']}")
        bench.require("result" in entry, "missing live result.json")
        validate_live(entry["result"], case, args, bench.sha_file(manifest))
        entry["receipt_audit"] = validate_receipts(entry["result"], output, case)
        bench.require(entry["result"].get("cache_seed") == bench.inventory(cache),
                      "live backend cache seed differs from original capture seed")
        entry["valid"] = True
    finally:
        entry["after"] = bench.telemetry(directory, "after")
    return entry


def run_live(args, summary, content):
    replay = None
    current_pair = {}
    run_ids = set()
    for planned in summary["planned_work"]:
        entry = checked_worker(args, summary, content, planned)
        bench.require(entry["result"]["run_id"] not in run_ids, "live reused an earlier worker run_id")
        run_ids.add(entry["result"]["run_id"])
        if replay is not None:
            require_replay(replay, entry)
        else:
            replay = entry
        current_pair[entry["side"]] = entry
        if len(current_pair) != 2:
            continue
        add_pair(summary, current_pair, entry["phase"])
        current_pair = {}
        if entry["pair"] != 5:
            continue
        phase = entry["phase"]
        pairs = [p for p in summary["pairs"] if p["phase"] == phase]
        result = live_gates(pairs, phase == "aa")
        summary[phase + "_statistics"] = result
        summary[phase + "_qualification"] = "passed" if result["passed"] else "inconclusive"
        if not result["passed"]:
            summary["live_confirmation"] = "inconclusive"
            summary["stop_reason"] = f"live {phase.upper()} frozen gates did not pass; no retries or extensions"
            if phase == "ab" and (not result["gates"]["throughput_median"] or not result["gates"]["latency_median"]):
                summary["verdict"] = "REJECT"
                summary["live_confirmation"] = "failed"
            raise GateStop(summary["stop_reason"])
    summary["live_confirmation"] = "passed"


def work_accounting(summary):
    counts = {status: 0 for status in ("completed", "failed", "timed_out", "cancelled", "invalid", "unrun")}
    reported = {f"{phase}_{unit}": 0 for phase in ("submitted", "completed")
                for unit in ("batches", "iterations", "evaluations")}
    missing = 0
    for entry in summary["trials"]:
        process_status = entry.get("process", {}).get("status")
        if process_status in counts:
            counts[process_status] += 1
        counts["invalid"] += not entry["valid"]
        record = entry.get("result")
        counters = record.get("counters") if isinstance(record, dict) else None
        if not isinstance(counters, dict):
            missing += 1
            continue
        for key in reported:
            value = counters.get(key)
            if type(value) is int and 0 <= value <= 2**64 - 1:
                reported[key] += value
    counts.update(attempted=len(summary["trials"]), planned=len(summary["planned_work"]),
                  valid=sum(t["valid"] for t in summary["trials"]),
                  unrun=sum(p["status"] == "unrun" for p in summary["planned_work"]))
    kinds = {kind: {"workers": sum(t["kind"] == kind for t in summary["trials"]),
                    "invalid": sum(t["kind"] == kind and not t["valid"] for t in summary["trials"]),
                    "errors": sum(len(t["errors"]) for t in summary["trials"] if t["kind"] == kind)}
             for kind in ("timing", "diagnostic", "prefix", "live")}
    return {"worker_counts": counts, "reported_including_invalid_diagnostic_and_prefix": reported,
            "worker_kinds": kinds, "workers_without_reported_counters": missing,
            "retained_worker_error_count": sum(len(t["errors"]) for t in summary["trials"]),
            "policy": "retain every worker; diagnostic, prefix and invalid work never enters timing scores"}


def finish(summary):
    summary["package_elapsed_seconds"] = time.monotonic() - summary["package_clock"]["started_monotonic_seconds"]
    summary["budget_exceeded"] = summary["package_elapsed_seconds"] > PACKAGE_SECONDS
    if summary["budget_exceeded"]:
        summary["errors"].append("hard package wall budget exceeded")
    final_statuses = {t["id"]: t["status"] for t in summary["trials"]}
    for planned in summary["planned_work"]:
        if planned["id"] in final_statuses:
            planned["status"] = final_statuses[planned["id"]]
    summary.update(finished_utc=bench.utc_now(), valid=not summary["errors"], error_count=len(summary["errors"]),
                   completed_work=work_accounting(summary))
    summary["status"] = "cancelled" if summary.get("cancelled") else "invalid" if summary["errors"] else "completed"
    if summary["errors"]:
        summary["verdict"] = "INVALID"
        summary["statistics"] = None
        for key in ("aa_statistics", "ab_statistics"):
            if key in summary:
                summary[key]["score_eligible"] = False
        summary["reserved_confirmation"] = "invalid" if summary["command"] == "engine" else "unrun"
        summary["live_confirmation"] = "invalid" if summary["command"] == "live" else "unrun"
    complete = all(p["status"] != "unrun" for p in summary["planned_work"])
    summary["planned_work_complete"] = complete
    for planned in summary["planned_work"]:
        if planned["status"] == "unrun":
            planned["unrun_reason"] = summary.get("stop_reason", "setup or validity failure")
    for entry in summary["trials"]:
        entry["timing_score_eligible"] = summary["valid"] and entry["valid"] and entry["kind"] == "timing"


def new_summary(args, clock):
    return {"protocol_version": 1, "command": args.command, "started_utc": bench.utc_now(),
            "package_clock": clock, "package_wall_limit_seconds": PACKAGE_SECONDS,
            "worker_timeout_seconds": args.timeout, "cleanup_reserve_seconds": CLEANUP_SECONDS,
            "policy": {"engine": compare.POLICY, "live": LIVE_POLICY, "case_seeds": SEEDS,
                       "engine_pairs": 5, "engine_batches": 64, "prefix_batches": [8, 32], "warmup": 2,
                       "planned_engine_workers": 45, "planned_engine_completed_evaluations": 38535168,
                       "no_retries": True, "no_extensions": True, "no_sample_filtering": True},
            "isolation_state": "unverified", "clean_baseline": None, "adoption": False,
            "optimization_claim": False, "broader_campaign_qualification": False,
            "isolation_policy": "background activity allowed; preserve other processes and every slow pair",
            "restart_scope": "mature drawing restarts; prefix executions do not establish continuation",
            "verdict": "INCONCLUSIVE", "reserved_confirmation": "unrun", "live_confirmation": "unrun",
            "planned_work": engine_plan() if args.command == "engine" else live_plan(),
            "trials": [], "pairs": [], "errors": [], "binaries": {}, "statistics": None}


def execute(args):
    bench.require(not args.output.exists(), "output already exists; immutable packages cannot be resumed")
    inputs = [args.manifest.parent, args.capture.parent, args.reference, args.candidate]
    inputs += [p for p in (args.package_ledger, args.reference_provenance, args.candidate_provenance) if p is not None]
    for source in inputs:
        source_root, output_root = source.resolve(), args.output.resolve()
        bench.require(not source_root.is_relative_to(output_root) and not output_root.is_relative_to(source_root),
                      "output and immutable inputs must not overlap")
    clock = package_clock(args.package_ledger)
    args.output.mkdir(parents=True)
    summary = new_summary(args, clock)
    try:
        with package_signals(clock), bench.benchmark_lock(LOCK):
            try:
                content = prepare(args, summary)
                if args.command == "engine":
                    run_engine(args, summary, content)
                else:
                    run_live(args, summary, content)
            finally:
                anchors(args, summary)
    except GateStop as exc:
        summary["stop_reason"] = str(exc)
    except FAILURES as exc:
        summary["errors"].append(str(exc))
        summary["stop_reason"] = "validity or package budget failure; fail fast"
    except KeyboardInterrupt as exc:
        summary["errors"].append(str(exc) or "qualification cancelled")
        summary.update(cancelled=True, stop_reason="cancelled; remaining work was not run")
    finish(summary)
    bench.write_json(args.output / "summary.json", summary)
    print(f"{summary['verdict']}: {args.output / 'summary.json'}")
    return 0 if summary["valid"] else 1


def timeout_value(raw):
    try:
        return bench.finite(float(raw), 0.01, WORKER_SECONDS, "timeout")
    except FAILURES as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("engine", "live"):
        sub = commands.add_parser(name)
        for option in ("manifest", "capture", "reference", "candidate", "output"):
            sub.add_argument("--" + option, required=True, type=Path)
        for option in ("package-ledger", "reference-provenance", "candidate-provenance"):
            sub.add_argument("--" + option, type=Path)
        sub.add_argument("--timeout", type=timeout_value, default=WORKER_SECONDS,
                         help="per-worker seconds, at most 120; cannot extend package deadline")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    try:
        return execute(args)
    except FAILURES as exc:
        print(f"qualification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
