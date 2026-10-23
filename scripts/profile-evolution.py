#!/usr/bin/env python3
"""Profile a frozen v1 reference without adding timings to its primary score.

Usage: python3 scripts/profile-evolution.py --baseline PATH/summary.json
Optional: --case ff|mona|darwin --host-only|--envelope-only --binary PATH
          --output NEW_DIRECTORY --timeout 600

Linux host profiling requires perf and readable /proc. Missing perf and recognized
perf permission/event failures are not_available; worker failures are invalid.
Baselines with zero warmup are unsupported; the steady marker is the second #1.
Unsupported steady windows fail rather than guessing.
Host mode runs the baseline's archived executable. Envelope mode archives the
supplied executable, or builds once through evolution-bench.py's Cargo artifact
helper. Neither mode changes a baseline, project drawing, or primary score.

Build the Lever chooses one serial script using the existing benchmark helpers.
Test Behavior, Not Implementation requires malformed evidence and real fake-child
timeout tests. Setup verifies binary/fixture/cache identity before either mode;
only the new bundle is shared, under the existing benchmark lock. GPU execution
is serial. Real GPU proof belongs to the caller; tests use synthetic processes.

Power of Ten deviations, rules 3/8/9: Python, importlib, JSON, argparse, selectors,
subprocess and unittest allocate and use platform callbacks. Only the named local
benchmark module is dynamically imported. There is one owned process group, at
most six trials, 144001 samples per trial, 16 markers, 256 perf tasks and children
per task, 64 KiB pipe reads, and existing benchmark file/inventory limits. A perf
data file and host child files are capped at 512 MiB. Logs and result files over
64 MiB fail validation. Timeout and cancellation
kill/reap only that group. Retain these bounds when extending the script.
Rule 10: warning-enabled stdlib unittest is the available recurring check; CI
configuration is outside these two files. Tests do not establish GPU correctness.
"""

import argparse
import collections
import errno
import functools
import importlib.util
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time


def load_bench():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "evolution_bench_profile_helpers", Path(__file__).with_name("evolution-bench.py"))
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load local evolution benchmark helpers")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


bench = load_bench()
PERIOD = 0.025
GUARD = 1.0
MIN_SAMPLES = 20
MAX_PROC_TASKS = 256
MAX_MARKERS = 16
LOCK = bench.DEFAULT_ROOT / "benchmark.lock"
ROOT = bench.REPO / ".cache/evolution-profile"


def read_small(path):
    with path.open("rb") as stream:
        data = stream.read(16385)
    bench.require(len(data) <= 16384, f"proc file exceeds byte limit: {path}")
    return data.decode("utf-8", errors="strict")


def parse_proc_stat(text):
    """comm may contain spaces and closing parentheses; fields follow the last )."""
    end = text.rfind(")")
    bench.require("(" in text and end >= 0, "malformed proc stat comm")
    fields = text[end + 1:].split()
    bench.require(20 <= len(fields) <= 128, "malformed proc stat fields")
    return {"state": fields[0],
            "utime": bench.integer(int(fields[11]), 0, 2**64 - 1, "utime"),
            "stime": bench.integer(int(fields[12]), 0, 2**64 - 1, "stime"),
            "starttime": bench.integer(int(fields[19]), 0, 2**64 - 1, "starttime")}


def find_worker(perf_pid, binary):
    """Only inspect direct children of threads belonging to our perf process."""
    identity = binary.stat()
    try:
        with os.scandir(f"/proc/{perf_pid}/task") as tasks:
            for index, task in enumerate(tasks):
                bench.require(index < MAX_PROC_TASKS, "perf task count exceeds limit")
                try:
                    children = read_small(Path(task.path) / "children").split()
                    bench.require(len(children) <= MAX_PROC_TASKS, "perf children exceed limit")
                    for child in children:
                        pid = bench.integer(int(child), 1, 2**31 - 1, "worker PID")
                        candidate = Path(f"/proc/{pid}/exe").stat()
                        if (candidate.st_dev, candidate.st_ino) == (identity.st_dev, identity.st_ino):
                            return pid
                except (FileNotFoundError, ProcessLookupError):
                    continue
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def sample_worker(raw, binary):
    if raw["worker_pid"] is None:
        raw["worker_pid"] = find_worker(raw["perf_pid"], binary)
    pid = raw["worker_pid"]
    if pid is None:
        return
    try:
        group = parse_proc_stat(read_small(Path(f"/proc/{pid}/stat")))
        main = parse_proc_stat(read_small(Path(f"/proc/{pid}/task/{pid}/stat")))
        wchan = read_small(Path(f"/proc/{pid}/task/{pid}/wchan")).strip()
        bench.require(len(wchan) <= 128, "wait channel name exceeds limit")
        if raw.get("worker_starttime") is None:
            raw["worker_starttime"] = group["starttime"]
        bench.require(group["starttime"] == main["starttime"] == raw["worker_starttime"],
                      "worker PID identity changed while sampling")
        bench.require(len(raw["samples"]) < raw["sample_limit"], "host sample limit exceeded")
        raw["samples"].append([time.monotonic(), group["utime"], group["stime"],
                               main["utime"], main["stime"], main["state"], wchan])
    except (FileNotFoundError, ProcessLookupError):
        pass  # An owned worker can exit between the individual proc reads.


def drain_stderr(selector, stderr, raw, pending):
    events = selector.select(PERIOD)
    for key, _ in events:  # Exactly one registered pipe.
        chunk = os.read(key.fileobj.fileno(), 65536)
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        bench.require(stderr.tell() + len(chunk) <= bench.MAX_FILE_BYTES, "stderr exceeds byte limit")
        stderr.write(chunk)
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        bench.require(len(pending) <= 65536, "stderr line exceeds byte limit")
        for line in lines:  # At most 65536 newlines in one bounded pipe read.
            if b"[timestamps] sample #1:" in line:
                bench.require(len(raw["first_sample_messages"]) < MAX_MARKERS, "too many timing markers")
                raw["first_sample_messages"].append(time.monotonic())
    return pending


def stop_group(process):
    # start_new_session makes this PID the group ID, including perf's worker.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait(timeout=5)


def observe_process(process, stderr, raw, timeout, binary):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stderr, selectors.EVENT_READ)
        pending = b""
        next_sample = raw["started_monotonic"]
        for _ in range(raw["sample_limit"] * 4 + 128):
            now = time.monotonic()
            if now - raw["started_monotonic"] >= timeout:
                return "timed_out"
            pending = drain_stderr(selector, stderr, raw, pending)
            if binary is not None and now >= next_sample:
                sample_worker(raw, binary)
                next_sample = now + PERIOD
            if process.poll() is not None and not selector.get_map():
                return "completed" if process.returncode == 0 else "failed"
        raise bench.BenchError("process observation iteration budget exhausted")


def run_owned(command, cwd, evidence, timeout, binary=None, artifact_limit=None):
    """Bound streams/lifetime and retain evidence, including cancellation paths."""
    evidence.mkdir()
    raw = {"samples": [], "first_sample_messages": [], "worker_pid": None,
           "sample_period_target_seconds": PERIOD, "sample_limit": math.ceil(timeout / PERIOD) + 1,
           "ticks_per_second": os.sysconf("SC_CLK_TCK")}
    limit = artifact_limit or bench.MAX_FILE_BYTES
    record = {"command": [str(p) for p in command], "cwd": str(cwd), "status": "failed",
              "returncode": None, "artifact_file_byte_limit": limit}
    with (evidence / "stdout.log").open("xb") as stdout, (evidence / "stderr.log").open("xb") as stderr:
        process = None
        raw["started_monotonic"] = time.monotonic()
        try:
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=subprocess.PIPE, start_new_session=True,
                                       preexec_fn=functools.partial(bench.limit_child_artifacts, limit))
            raw["perf_pid"] = process.pid
            record["status"] = observe_process(process, stderr, raw, timeout, binary)
        except (OSError, ValueError, bench.BenchError) as exc:
            record["error"] = str(exc)
            if process is None and isinstance(exc, OSError):
                record["spawn_errno"] = exc.errno
        except BaseException:
            record["status"] = "cancelled"
            raise
        finally:
            if process is not None:
                record["returncode"] = stop_group(process)
                process.stderr.close()
            raw["finished_monotonic"] = time.monotonic()
            record["process_wall_seconds"] = raw["finished_monotonic"] - raw["started_monotonic"]
            if stdout.tell() > bench.MAX_FILE_BYTES:
                record.update(status="failed", error="stdout exceeds byte limit")
            bench.write_json(evidence / "process.json", record)
            if binary is not None:
                bench.write_json(evidence / "host-samples.json", raw)
    return record, raw


def validate_profile_record(record, expected, case, args, manifest_hash, envelope):
    """Validate profile output directly; its nullable rate cannot pass normal scoring."""
    bench.object_value(record, "result")
    for field, value in (("protocol_version", 1), ("status", "completed"), ("correctness", True),
                         ("errors", []), ("error_count", 0), ("diagnostic", False),
                         ("case_id", case["case_id"]), ("seed", args.seed), ("batches", args.batches),
                         ("fixture_manifest_sha256", manifest_hash)):
        bench.require(type(record.get(field)) is type(value) and record[field] == value,
                      f"wrong or missing profile {field}")
    mismatches = bench.replay_errors(record, expected)
    bench.require(not mismatches, "; ".join(mismatches))
    elapsed = bench.finite(record.get("elapsed_seconds"), 1e-9, args.timeout, "elapsed_seconds")
    bench.finite(record.get("setup_seconds"), 0, args.timeout, "setup_seconds")
    bench.finite(record.get("drain_seconds"), 0, elapsed, "drain_seconds")
    bench.validate_quality_record(record, case)
    bench.integer(record.get("timestamp_samples"), args.batches, args.batches, "timestamp_samples")
    if not envelope:
        bench.validate_result(record, case, args, False, manifest_hash)
        return None
    bench.require(record.get("timestamp_mode") == "batch_envelope", "missing batch_envelope timestamp_mode")
    bench.require("evals_per_second" in record and record["evals_per_second"] is None,
                  "profile evals_per_second must be null")
    span = bench.object_value(record.get("batch_compute_envelope"), "batch_compute_envelope")
    bench.integer(span.get("sample_count"), args.batches, args.batches, "envelope sample_count")
    # 1% or 1 ms permits timestamp/wall-clock granularity; it cannot excuse a missing batch.
    compute = bench.finite(span.get("compute_ns"), 1e-9,
                           elapsed * 1e9 + max(1e6, elapsed * 1e7), "compute_ns") / 1e9
    bench.require(record["sampled_gpu_stage_totals_ns"][3] == 0, "selection interval must be unavailable/zero")
    return {"compute_envelope_seconds": compute, "wall_seconds": elapsed,
            "gpu_queue_critical_span_fraction": compute / elapsed,
            "scope": "all iterations per batch; excludes uploads and following copy/readback; not utilization",
            "last_iteration_samples": {"mutate_bin_raster_ns": record["sampled_gpu_stage_totals_ns"][:3],
                                       "sample_count": record["timestamp_samples"], "selection": "unavailable"}}


def analyze_host(raw, result):
    markers = raw["first_sample_messages"]
    bench.require(2 <= len(markers) <= MAX_MARKERS, "need warmup and measured sample #1 markers")
    spawned, finished = raw["started_monotonic"], raw["finished_monotonic"]
    begin = markers[1] + GUARD
    end = spawned + result["setup_seconds"] + result["elapsed_seconds"] - GUARD
    bench.require(spawned <= markers[0] < markers[1] < finished, "implausible timing markers")
    bench.require(spawned + result["setup_seconds"] <= begin < end <= finished - GUARD,
                  "guarded steady window is not contained in worker lifetime/measurement")
    bench.require(end - begin <= result["elapsed_seconds"] - 2 * GUARD,
                  "steady window exceeds measured interval")
    bench.require(len(raw["samples"]) <= raw["sample_limit"], "host sample limit exceeded")
    interior = [s for s in raw["samples"] if begin <= s[0] <= end]
    bench.require(len(interior) >= MIN_SAMPLES, "fewer than 20 guarded steady samples")
    duration = interior[-1][0] - interior[0][0]
    bench.require(duration > 0 and all(a[0] < b[0] for a, b in zip(interior, interior[1:])),
                  "host sample clock did not advance")
    ticks = bench.integer(raw["ticks_per_second"], 1, 1000000, "ticks_per_second")
    group = ((interior[-1][1] + interior[-1][2]) - (interior[0][1] + interior[0][2])) / ticks
    main = ((interior[-1][3] + interior[-1][4]) - (interior[0][3] + interior[0][4])) / ticks
    bench.require(group >= 0 and main >= 0, "CPU counters moved backwards")
    states = dict(collections.Counter(s[5] for s in interior))
    channels = dict(collections.Counter(s[6] for s in interior))
    return {"window_monotonic": [begin, end], "guard_seconds": GUARD,
            "window_policy": "second sample #1 + 1s to perf spawn + setup + elapsed - 1s; conservative early end",
            "sample_period_target_seconds": PERIOD, "sample_limit": raw["sample_limit"],
            "minimum_samples": MIN_SAMPLES, "interior_samples": len(interior),
            "observed_window_seconds": duration, "process_cpu_seconds": group,
            "process_one_core_fraction": group / duration, "main_cpu_seconds": main,
            "main_one_core_fraction": main / duration, "main_states": states,
            "main_wait_channels": channels,
            "main_wait_channel_fractions": {k: v / len(interior) for k, v in channels.items()},
            "cpu_scope": "/proc utime+stime, all threads; perf cpu-clock:u samples user CPU time only",
            "interpretation": "CPU wait observations; not GPU utilization"}


def load_baseline(args):
    summary = bench.object_value(bench.read_json(args.baseline), "baseline summary")
    bench.require(summary.get("command") == "baseline" and summary.get("valid") is True
                  and type(summary.get("error_count")) is int and summary["error_count"] == 0,
                  "baseline must be a valid error-free reference capture")
    capture = args.baseline.parent
    binary = Path(summary["binary"]["archived"])
    manifest = Path(summary["manifest"])
    bench.require(binary.parent == capture and manifest.parent == capture / "fixtures",
                  "moved/restructured baseline bundles are unsupported")
    bench.require(bench.sha_file(binary, bench.MAX_BINARY_BYTES) == summary["binary"]["sha256"],
                  "baseline archived binary SHA mismatch")
    bench.require(bench.sha_file(manifest) == summary["manifest_sha256"], "baseline manifest SHA mismatch")
    bench.require(bench.inventory(manifest.parent) == summary["fixture_inventory"], "baseline fixture inventory mismatch")
    bench.require(bench.inventory(capture / "cache-seed") == summary["cache_policy"]["files"],
                  "baseline cache seed inventory mismatch")
    arguments = summary["arguments"]
    args.seed = bench.integer(arguments.get("seed"), 0, 2**64 - 1, "baseline seed")
    args.batches = bench.integer(arguments.get("batches"), 1, 4096, "baseline batches")
    args.warmup = bench.integer(arguments.get("warmup"), 1, 16, "baseline warmup")
    trials = summary.get("trials")
    bench.require(isinstance(trials, list) and 1 <= len(trials) <= bench.MAX_TRIALS, "invalid baseline trial count")
    content = bench.load_manifest(manifest)
    expected = {}
    for case in content["cases"]:
        if args.case and case["case_id"] != args.case:
            continue
        normals = [t for t in trials if t.get("case_id") == case["case_id"] and t.get("diagnostic") is False]
        bench.require(bool(normals) and all(t.get("valid") is True for t in normals), "missing valid normal baseline trials")
        reference = normals[0]["result"]
        for trial in normals:
            baseline_args = argparse.Namespace(seed=args.seed, batches=args.batches, timeout=3600)
            bench.validate_result(trial["result"], case, baseline_args, False, summary["manifest_sha256"])
            bench.require(not bench.replay_errors(reference, trial["result"]), "baseline normal replays disagree")
        expected[case["case_id"]] = reference
    return summary, binary, manifest, content, expected


def trial_command(binary, manifest, case, args, output, envelope):
    command = [str(binary), "trial", "--manifest", str(manifest), "--case", case["case_id"],
               "--seed", str(args.seed), "--batches", str(args.batches), "--warmup", str(args.warmup),
               "--diagnostic", "false", "--output", str(output)]
    if envelope:
        command.extend(["--profile-mode", "batch-envelope"])
    return command


def perf_unavailable(process, raw, evidence):
    """Only recognized perf startup failures may omit the worker result."""
    if process["status"] != "failed" or raw.get("worker_pid") is not None or raw.get("first_sample_messages"):
        return False
    if process.get("spawn_errno") == errno.ENOENT:
        return True
    path = evidence / "stderr.log"
    with path.open("rb") as stream:
        text = stream.read(16384).decode("utf-8", errors="replace")
    permission = ("No permission to enable cpu-clock" in text
                  or "Access to performance monitoring and observability operations is limited" in text)
    unsupported = ("event syntax error: 'cpu-clock:u'" in text
                   or ("perf_event_open" in text and any(s in text for s in
                       ("Operation not supported", "Function not implemented"))))
    return permission or unsupported


def profile_trial(args, binary, identity, manifest, hashes, case, expected, entry):
    directory = Path(entry["directory"])
    directory.mkdir()
    envelope = entry["mode"] == "batch_envelope"
    entry["binary"] = identity
    entry["evidence"] = {"process": str(directory / "process"), "worker_output": str(directory / "output"),
                         "analysis": str(directory / ("envelope-analysis.json" if envelope else "host-analysis.json"))}
    if not envelope:
        entry["evidence"].update(samples=str(directory / "process/host-samples.json"),
                                  perf_data=str(directory / "perf.data"), report=str(directory / "perf-report"))
    cwd = directory / "cwd"
    cache = args.output / "cache-seed"
    bench.require(bench.inventory(cache) == args.cache_hashes, "fixed cache seed changed")
    bench.copy_inventory(cache, cwd, args.cache_hashes, writable=True)
    output = directory / "output"
    bench.require(bench.inventory(manifest.parent) == hashes, "fixture changed before profile")
    command = trial_command(binary, manifest, case, args, output, envelope)
    if not envelope:
        perf = shutil.which("perf")
        if perf is None or not Path("/proc/self/task").is_dir():
            entry.update(status="not_available", reason="perf or Linux /proc is unavailable")
            return
        command = [perf, "record", "-e", "cpu-clock:u", "-F", "499", "--call-graph", "dwarf,8192",
                   "--clockid", "mono", "-o", str(directory / "perf.data"), "--"] + command
    process, raw = run_owned(command, cwd, directory / "process", args.timeout,
                             None if envelope else binary,
                             bench.MAX_FILE_BYTES if envelope else bench.MAX_BINARY_BYTES)
    entry["process"] = process
    bench.require(bench.inventory(manifest.parent) == hashes, "fixture changed during profile")
    bench.require(bench.sha_file(binary, bench.MAX_BINARY_BYTES) == identity["sha256"], "profile binary changed")
    if output.exists():
        entry["artifacts"] = bench.inventory(output)
    if (output / "result.json").is_file():
        entry["result"] = bench.read_json(output / "result.json", 2 * 1024 * 1024)
    if not envelope and "result" not in entry and perf_unavailable(process, raw, directory / "process"):
        entry.update(status="not_available", reason="recognized perf startup failure; inspect process logs")
        return
    bench.require(process["status"] == "completed", f"profile process {process['status']}")
    bench.require("result" in entry, "missing profile result.json")
    entry["analysis"] = validate_profile_record(entry["result"], expected, case, args,
                                                bench.sha_file(manifest), envelope)
    bench.verify_saved_state(output, entry["result"])
    if not envelope:
        entry["analysis"] = analyze_host(raw, entry["result"])
        begin, end = entry["analysis"]["window_monotonic"]
        report, _ = run_owned(["perf", "report", "-i", str(directory / "perf.data"), "--stdio",
                               "--no-children", "--percent-limit", "1", "--time", f"{begin:.9f},{end:.9f}"],
                              cwd, directory / "perf-report", min(args.timeout, 30))
        entry["perf_report"] = report
        entry["perf_report_status"] = "available" if report["status"] == "completed" else "not_available"
    bench.write_json(directory / ("envelope-analysis.json" if envelope else "host-analysis.json"), entry["analysis"])
    entry.update(status="completed", valid=True)


def run_profiles(args, summary):
    baseline, reference_binary, source_manifest, content, expected = load_baseline(args)
    summary.update(baseline_binary=baseline["binary"], baseline_manifest_sha256=baseline["manifest_sha256"],
                   schedule={"seed": args.seed, "batches": args.batches, "warmup": args.warmup},
                   reference_results=expected)
    manifest, _, hashes = bench.prepare_fixtures(source_manifest, args.output)
    cache = args.output / "cache-seed"
    args.cache_hashes = baseline["cache_policy"]["files"]
    bench.copy_inventory(args.baseline.parent / "cache-seed", cache, args.cache_hashes)
    envelope_binary = identity = None
    if not args.host_only:
        envelope_binary, identity = bench.prepare_binary(args, args.output)
        summary["envelope_binary"] = identity
    if not args.envelope_only:
        version, _ = run_owned(["perf", "--version"], args.output, args.output / "perf-version", min(args.timeout, 5))
        summary["perf_version"] = version
    for case in content["cases"]:
        if case["case_id"] not in expected:
            continue
        modes = ("batch_envelope",) if args.envelope_only else ("host",) if args.host_only else ("host", "batch_envelope")
        for mode in modes:  # At most three cases times two modes; no overlapping GPU work.
            entry = {"case_id": case["case_id"], "mode": mode, "valid": False, "errors": [],
                     "directory": str(args.output / f"{case['case_id']}-{mode}")}
            summary["profiles"].append(entry)
            try:
                profile_trial(args, envelope_binary if mode == "batch_envelope" else reference_binary,
                              identity if mode == "batch_envelope" else baseline["binary"],
                              manifest, hashes, case, expected[case["case_id"]], entry)
            except (bench.BenchError, OSError, ValueError, KeyError, TypeError) as exc:
                entry.update(status="invalid", errors=[str(exc)])
                summary["errors"].append(f"{case['case_id']} {mode}: {exc}")
    bench.require(bench.sha_file(reference_binary, bench.MAX_BINARY_BYTES) == baseline["binary"]["sha256"],
                  "baseline archived executable changed")
    bench.require(bench.inventory(source_manifest.parent) == baseline["fixture_inventory"], "baseline fixtures changed")
    bench.require(bench.inventory(args.baseline.parent / "cache-seed") == args.cache_hashes, "baseline cache seed changed")


def execute(args):
    output = args.output
    for source in (args.baseline.parent, bench.REPO / "projects"):
        root = source.resolve()
        bench.require(not output.is_relative_to(root) and not root.is_relative_to(output), "output overlaps baseline/projects")
    bench.require(not LOCK.resolve().is_relative_to(output), "output overlaps benchmark lock")
    summary = {"protocol_version": 1, "command": "profile", "scope": "diagnostic evidence only",
               "baseline": str(args.baseline), "baseline_summary_sha256": bench.sha_file(args.baseline),
               "started_utc": bench.utc_now(), "campaign_frozen": False, "optimization_claim": False,
               "primary_score_eligible": False, "verdict": "no throughput verdict", "errors": [], "profiles": [],
               "evidence_directory": str(output), "cpu_count": os.cpu_count(),
               "cpu_load_average": list(os.getloadavg()), "isolation_state": "unverified",
               "scope_limits": "single diagnostic replay per mode/case; no speedup or utilization claim"}
    with bench.benchmark_lock(LOCK):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        try:
            run_profiles(args, summary)
        except (bench.BenchError, OSError, ValueError, KeyError, TypeError) as exc:
            summary["errors"].append(str(exc))
        except KeyboardInterrupt:
            summary["errors"].append("profiling cancelled; owned process group stopped")
            summary["cancelled"] = True
        finally:
            summary["completed_work"] = bench.completed_work(summary["profiles"])
            summary["error_count"] = len(summary["errors"])
            summary["valid"] = not summary["errors"]
            summary["unavailable_profiles"] = sum(e.get("status") == "not_available" for e in summary["profiles"])
            summary["status"] = "invalid" if summary["errors"] else "partial" if summary["unavailable_profiles"] else "completed"
            summary["finished_utc"] = bench.utc_now()
            bench.write_json(output / "summary.json", summary)
    print(str(output / "summary.json"))
    return 0 if summary["valid"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=bench.PROJECTS)
    parser.add_argument("--binary", type=Path, help="envelope executable; otherwise build release example once")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--host-only", action="store_true")
    modes.add_argument("--envelope-only", action="store_true")
    parser.add_argument("--timeout", type=bench.timeout_arg, default=600.0)
    args = parser.parse_args(argv)
    if args.output is None:
        stamp = bench.datetime.datetime.now(bench.datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        args.output = ROOT / f"profile-{stamp}-{os.getpid()}"
    for key in ("baseline", "output", "binary"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    try:
        return execute(args)
    except (bench.BenchError, OSError, ValueError) as exc:
        print(f"profile failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
