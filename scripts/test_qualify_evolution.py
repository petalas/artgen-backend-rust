"""CPU-only behavioral proofs for the capped qualification CLI.

Run python3 -B -W error -m unittest discover -s scripts -p test_qualify_evolution.py.
Synthetic subprocesses exercise real process/file ownership, never GPUs or Cargo.
Python/unittest dispatch and allocation use the bounded deviations in the collector.
"""

import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


def local_module(name, filename):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
        if spec is None or spec.loader is None:
            raise RuntimeError("missing local test support")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


qualify = local_module("qualification_under_test", "qualify-evolution.py")
support = local_module("qualification_compare_support", "test_compare_evolution.py")
bench = qualify.bench
LIVE_ARTIFACT = '''
if LIVE:
    record.update(mode="live_acceptance", live_protocol_version=2, warmup_batches=2, publish_every=2,
                  deadline_ms=int(arg("--deadline-ms")), run_id=str(os.getpid()),
                  previews_due=128, previews_sent=128, previews_received=128,
                  missed_deadlines=0, p95_latency_ms=5.0, publications=[], cache_seed=CACHE_SEED)
    if MODE == "live_protocol_missing":
        del record["live_protocol_version"]
    elif MODE.startswith("live_protocol_"):
        record["live_protocol_version"] = json.loads(MODE.removeprefix("live_protocol_"))
    record["elapsed_seconds"] = 1.0
    record["evals_per_second"] = evals
    for seq in range(1,129):
        record["publications"].append(dict(
            run_id=record["run_id"], sequence=seq, completed_batches=seq*2,
            completed_evaluations=seq*32768, due_ms=seq*5.0, send_ms=seq*5.0+1,
            receive_ms=seq*5.0+5, drawing_sha256=hashed("drawing"),
            png_sha256=hashed("png"), fitness_percent=98.5, polygons=400,
            payload_valid=True))
    if MODE == "live_slow" and label.endswith("candidate"):
        record["elapsed_seconds"] = 1.1
        record["evals_per_second"] = evals/1.1
    if MODE == "live_latency" and label.endswith("candidate"):
        record["p95_latency_ms"] = 6
        for publication in record["publications"]:
            publication["receive_ms"] += 1
    if MODE == "live_aa_noise" and label.endswith("candidate"):
        record["elapsed_seconds"] = 1.05
        record["evals_per_second"] = evals/1.05
    if MODE == "live_missing":
        record["publications"].pop()
        record["previews_received"] -= 1
    if MODE == "live_bad_p95":
        record["p95_latency_ms"] = 4
    if MODE == "live_payload":
        record["publications"][0]["payload_valid"] = False
    if MODE == "live_wrong_origin":
        record["publications"][0]["completed_evaluations"] -= 1
    if MODE == "live_duplicate":
        record["publications"][1]["sequence"] = 1
    if MODE == "live_inverted_time":
        record["publications"][0]["receive_ms"] = -1
    if MODE == "live_timeline":
        record["publications"][1]["due_ms"] = 0
        record["publications"][1]["send_ms"] = 1
        record["publications"][1]["receive_ms"] = 5
    if MODE == "live_cache":
        record["cache_seed"] = {}
    if MODE == "live_deadline":
        record["publications"][-1]["receive_ms"] += 300
        record["missed_deadlines"] = 1
    if MODE == "live_terminal":
        record["publications"][-1]["fitness_percent"] -= .1
'''
LIVE_RECEIPTS = r'''
if LIVE:
    import base64, struct, zlib
    def png_chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    width, height = case["width"], case["height"]
    if MODE == "receipt_png_dimensions":
        width += 1
    png = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
           + png_chunk(b"IDAT", zlib.compress((b"\0" + b"\xff" * width * 4) * height)) + png_chunk(b"IEND", b""))
    polygon = {"points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}],
               "color": {"r": 255, "g": 255, "b": 255, "a": 255}}
    drawing = {"polygons": [polygon] * 400, "isDirty": False, "fitness": 98.5}
    if MODE == "receipt_drawing_fitness":
        drawing["fitness"] = 97
    if MODE == "receipt_drawing_polygons":
        drawing["polygons"].pop()
    drawing_json = json.dumps(drawing, separators=(",", ":"))
    frames = []
    fields = ("run_id", "sequence", "completed_batches", "completed_evaluations", "due_ms", "send_ms",
              "drawing_sha256", "png_sha256")
    for publication in record["publications"]:
        publication.update(drawing_sha256=hashed(drawing_json), png_sha256=hashlib.sha256(png).hexdigest())
        frames.append({"type": "update", "image": base64.b64encode(png).decode(),
                       "fitness": publication["fitness_percent"], "polygons": publication["polygons"],
                       "totalEvals": publication["completed_evaluations"], "elapsed": publication["send_ms"] / 1000,
                       "drawingJson": drawing_json, "liveAcceptance": {key: publication[key] for key in fields}})
    if MODE.startswith("receipt_meta_"):
        key = MODE.removeprefix("receipt_meta_")
        frames[0]["liveAcceptance"][key] = "wrong" if isinstance(frames[0]["liveAcceptance"][key], str) else -1
    if MODE == "receipt_png_hash":
        frames[0]["image"] = base64.b64encode(b"corrupt PNG").decode()
    if MODE == "receipt_bad_base64":
        frames[0]["image"] = "!invalid!"
    if MODE == "receipt_drawing_hash":
        frames[0]["drawingJson"] += " "
    if MODE == "receipt_fitness":
        frames[0]["fitness"] = 97
    if MODE == "receipt_polygons":
        frames[0]["polygons"] = 399
    if MODE == "receipt_wrong_type":
        frames[0]["type"] = "stats"
    if MODE == "receipt_wire_origin":
        frames[0]["totalEvals"] -= 1
    if MODE == "receipt_missing_line":
        frames.pop()
    if MODE == "receipt_extra_line":
        frames.append(frames[0])
    lines = [(json.dumps(frame, separators=(",", ":")) + "\n").encode() for frame in frames]
    if MODE == "receipt_bad_json":
        lines[0] = b"not JSON\n"
    if MODE == "receipt_duplicate_key":
        lines[0] = b'{"type":"update",' + lines[0][1:]
    if MODE == "receipt_long_line":
        lines[0] = b" " * (8 * 1024 * 1024) + lines[0]
    if MODE == "receipt_no_newline":
        lines[-1] = lines[-1].rstrip(b"\n")
    if MODE == "receipt_nested_json":
        lines[0] = b"[" * 1000 + b"0" + b"]" * 1000 + b"\n"
    if MODE == "receipt_nonfinite_json":
        lines[0] = lines[0].replace(b'"fitness":98.5', b'"fitness":1e309', 1)
    record["receipt_artifacts"] = {}
    shards = 4 if MODE == "receipt_too_many" else 3 if MODE in ("receipt_three_shards", "receipt_total_limit") else 1
    for index in range(shards):
        name = f"receipt-{index:03d}.jsonl"
        raw = b"".join(lines[index * len(lines) // shards:(index + 1) * len(lines) // shards])
        (output / name).write_bytes(raw)
        record["receipt_artifacts"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    if MODE == "receipt_no_manifest":
        del record["receipt_artifacts"]
    if MODE == "receipt_empty_manifest":
        record["receipt_artifacts"] = {}
    if MODE == "receipt_missing_file":
        (output / "receipt-000.jsonl").unlink()
    if MODE == "receipt_corrupt_file":
        with (output / "receipt-000.jsonl").open("ab") as stream:
            stream.write(b"corrupt\n")
    if MODE == "receipt_bad_hash":
        record["receipt_artifacts"]["receipt-000.jsonl"]["sha256"] = "0" * 64
    if MODE == "receipt_bad_size":
        record["receipt_artifacts"]["receipt-000.jsonl"]["bytes"] += 1
    if MODE == "receipt_shard_limit":
        record["receipt_artifacts"]["receipt-000.jsonl"]["bytes"] = 64 * 1024 * 1024 + 1
    if MODE == "receipt_total_limit":
        for metadata in record["receipt_artifacts"].values():
            metadata["bytes"] = 50 * 1024 * 1024
    if MODE == "receipt_path_escape":
        record["receipt_artifacts"]["../receipt-000.jsonl"] = record["receipt_artifacts"].pop("receipt-000.jsonl")
    if MODE == "receipt_symlink":
        (output / "receipt-000.jsonl").rename(output / "raw-receipt")
        (output / "receipt-000.jsonl").symlink_to("raw-receipt")
'''
CHILD = support.CHILD.replace('MODE = "ok"', 'MODE = "ok"\nLIVE = "--live-acceptance" in sys.argv')
CHILD = CHILD.replace('def arg(name):\n', 'def arg(name):\n    if name == "--diagnostic" and LIVE: return "false"\n')
CHILD = CHILD.replace('label.endswith("diagnostic")', '"-diagnostic-" in label')
CHILD = CHILD.replace('\nassert Path("pipeline_cache_test.bin").read_bytes()', '''
CACHE_SEED = {"pipeline_cache_test.bin": {"bytes": len(Path("pipeline_cache_test.bin").read_bytes()),
              "sha256": hashlib.sha256(Path("pipeline_cache_test.bin").read_bytes()).hexdigest()}}
assert Path("pipeline_cache_test.bin").read_bytes()''')
CHILD = CHILD.replace('(output / "result.json").write_text(json.dumps(record))',
                      LIVE_ARTIFACT + LIVE_RECEIPTS + '\n(output / "result.json").write_text(json.dumps(record))')
CHILD = CHILD.replace('if (MODE == "replay_mismatch"', '''if MODE == "prefix_mismatch" and batches == 8 and label.endswith("candidate"):
    state["parents"][0]["rng_state"] += 1
if (MODE == "replay_mismatch"''')
CHILD = CHILD.replace('if MODE == "failure":', '''if MODE == "invalid_setup":
    (output / "result.json").write_text(json.dumps(dict(status="invalid", error_count=1,
        errors=["adapter unavailable"], counters=counters if "counters" in globals() else {},
        evals_per_second=None)))
    sys.exit(2)
if MODE == "failure":''')
CHILD = CHILD.replace('(output / "result.json").write_text(json.dumps(record))', '''
if MODE == "decimal_overflow":
    record["elapsed_seconds"] = 1.234567
    (output / "result.json").write_text(json.dumps(record).replace('1.234567', '1e309'))
elif MODE == "root_overflow":
    (output / "result.json").write_text('1e309')
elif MODE == "negative_root_overflow":
    (output / "result.json").write_text('-1e309')
elif MODE == "deeply_nested_json":
    (output / "result.json").write_text('[' * 1000 + '0' + ']' * 1000)
else:
    (output / "result.json").write_text(json.dumps(record))''')


class CollectorBehaviorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.capture = self.root / "capture"
        self.capture.mkdir()
        self.original_manifest = support.support.fixture_corpus(self.capture / "fixtures")
        manifest = bench.read_json(self.original_manifest)
        manifest["params"] = {"chainCount": 4, "lambda": 64, "gpuBatchIters": 64}
        self.original_manifest.write_text(json.dumps(manifest))
        self.manifest = support.support.fixture_corpus(self.root / "reserved")
        reserved = copy.deepcopy(manifest)
        for case in reserved["cases"]:
            drawing = self.manifest.parent / case["drawing_file"]
            drawing.write_text('{"polygons":[],"fitness":98.5,"reserved":true}')
            case["drawing_sha256"] = bench.sha_file(drawing)
        self.manifest.write_text(json.dumps(reserved))
        cache = self.capture / "cache-seed"
        cache.mkdir()
        (cache / "pipeline_cache_test.bin").write_bytes(b"fixed cache seed")
        self.original = self.capture / "evolution-bench"
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        for path in (self.original, self.reference, self.candidate):
            self.binary(path, delay=0.3 if path == self.candidate else 1)
        self.summary_path = self.make_capture()
        self.output = self.root / "qualification"
        self.before = bench.inventory(self.capture)
        self.patch(qualify, "LOCK", self.root / "benchmark.lock")
        self.patch(bench, "telemetry", support.support.telemetry_sample)

    def patch(self, target, key, value):
        patcher = mock.patch.object(target, key, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def binary(self, path, mode="ok", delay=1):
        text = CHILD.replace('MODE = "ok"', f'MODE = {mode!r}')
        path.write_text(text.replace("DELAY_FACTOR = 1.0", f"DELAY_FACTOR = {delay}"))
        path.chmod(0o755)

    def make_capture(self):
        trials = []
        for case in bench.PROJECTS:
            directory = self.capture / f"control-{case}"
            cwd = self.root / f"control-cwd-{case}"
            bench.copy_inventory(self.capture / "cache-seed", cwd, bench.inventory(self.capture / "cache-seed"), True)
            process = subprocess.run([str(self.original), "trial", "--manifest", str(self.original_manifest),
                                      "--case", case, "--seed", "7", "--batches", "64", "--warmup", "2",
                                      "--diagnostic", "false", "--output", str(directory)],
                                     cwd=cwd, capture_output=True, timeout=3, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            record = bench.read_json(directory / "result.json")
            for pair in range(5):
                for label in ("A", "B"):
                    result = copy.deepcopy(record)
                    rate = 1048576 / .25
                    if case == "ff" and pair == 0 and label == "A":
                        rate *= 1 - qualify.compare.NOISE_BAND
                    result.update(elapsed_seconds=1048576 / rate, evals_per_second=rate)
                    trials.append({"case_id": case, "label": label, "diagnostic": False, "valid": True, "result": result})
        summary = {"command": "baseline", "valid": True, "errors": [], "error_count": 0,
                   "binary": {"archived": str(self.original), "sha256": bench.sha_file(self.original)},
                   "manifest": str(self.original_manifest), "manifest_sha256": bench.sha_file(self.original_manifest),
                   "fixture_inventory": bench.inventory(self.original_manifest.parent),
                   "cache_policy": {"files": bench.inventory(self.capture / "cache-seed")},
                   "arguments": {"seed": 7, "batches": 64, "warmup": 2}, "trials": trials}
        path = self.capture / "summary.json"
        bench.write_json(path, summary)
        return path

    def arguments(self, command="engine", *extra):
        return [command, "--manifest", str(self.manifest), "--capture", str(self.summary_path),
                "--reference", str(self.reference), "--candidate", str(self.candidate),
                "--output", str(self.output), "--timeout", "3", *extra]

    def run_cli(self, command="engine", *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = qualify.main(self.arguments(command, *extra))
        path = self.output / "summary.json"
        return code, bench.read_json(path) if path.is_file() else None

    def invalid(self, mode, expected, command="engine", *extra):
        self.binary(self.reference, mode, .3)
        code, summary = self.run_cli(command, *extra)
        self.assertEqual(code, 1, summary["errors"] if summary else "no summary.json")
        self.assertEqual(summary["verdict"], "INVALID")
        self.assertFalse(summary["valid"])
        self.assertTrue(any(expected in e for e in summary["errors"]), summary["errors"])
        self.assertFalse(any(t["timing_score_eligible"] for t in summary["trials"]))
        self.assertGreater(summary["completed_work"]["worker_counts"]["unrun"], 0)
        self.assertEqual(bench.inventory(self.capture), self.before)
        return summary

    def test_engine_exact_matrix_seeds_counters_prefixes_and_no_adoption(self):
        code, summary = self.run_cli()
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["verdict"], "INCONCLUSIVE")
        self.assertEqual(summary["reserved_confirmation"], "passed")
        self.assertFalse(summary["adoption"])
        self.assertFalse(summary["optimization_claim"])
        self.assertIsNone(summary["clean_baseline"])
        self.assertEqual(summary["isolation_state"], "unverified")
        self.assertEqual(len(summary["trials"]), 45)
        self.assertEqual(len(summary["pairs"]), 15)
        counts = summary["completed_work"]
        self.assertEqual(counts["reported_including_invalid_diagnostic_and_prefix"]["completed_evaluations"], 38535168)
        self.assertEqual(counts["worker_counts"]["unrun"], 0)
        self.assertEqual(sum(t["timing_score_eligible"] for t in summary["trials"]), 30)
        self.assertEqual(sum(t["diagnostic"] for t in summary["trials"]), 3)
        self.assertEqual(sum(t["kind"] == "prefix" for t in summary["trials"]), 12)
        for case, seed in (("ff", 11), ("mona", 13), ("darwin", 17)):
            trials = [t for t in summary["trials"] if t["case_id"] == case]
            self.assertEqual({t["result"]["seed"] for t in trials}, {seed})
            pairs = [p for p in summary["pairs"] if p["case_id"] == case]
            self.assertEqual([p["order"] for p in pairs], [["reference", "candidate"], ["candidate", "reference"],
                             ["reference", "candidate"], ["candidate", "reference"], ["reference", "candidate"]])
        self.assertEqual(bench.inventory(self.capture), self.before)
        self.assertEqual(len({t["process"]["cwd"] for t in summary["trials"]}), 45)
        for trial in summary["trials"]:
            directory = Path(trial["directory"])
            self.assertIn("raw child stderr", (directory / "process/stderr.log").read_text())
            manifest = Path(trial["process"]["command"][3])
            self.assertEqual(bench.inventory(manifest.parent), summary["fixture_inventory"])

    def test_terminal_slowdown_stops_after_five_pairs_without_filtering(self):
        self.binary(self.candidate, delay=3)
        code, summary = self.run_cli()
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["verdict"], "REJECT")
        self.assertEqual(len(summary["trials"]), 10)
        self.assertEqual(len(summary["pairs"]), 5)
        self.assertEqual(summary["completed_work"]["worker_counts"]["unrun"], 35)
        self.assertTrue(all(p["unrun_reason"] for p in summary["planned_work"] if p["status"] == "unrun"))

    def test_invalid_first_worker_fails_fast_and_retains_partial_files(self):
        summary = self.invalid("failure", "subprocess failed")
        self.assertEqual(len(summary["trials"]), 1)
        trial = summary["trials"][0]
        self.assertEqual(trial["process"]["returncode"], 3)
        self.assertTrue((Path(trial["directory"]) / "output/partial.txt").is_file())

    def test_finite_overflow_boundary_retains_raw_counters(self):
        summary = self.invalid("oversized_number", "too large")
        self.assertEqual(summary["trials"][0]["result"]["elapsed_seconds"], 10**309)
        self.assertEqual(summary["completed_work"]["reported_including_invalid_diagnostic_and_prefix"]["completed_evaluations"], 1048576)

    def test_decimal_float_overflow_keeps_summary_and_raw_json(self):
        summary = self.invalid("decimal_overflow", "non-finite number")
        trial = summary["trials"][0]
        self.assertIn("non-finite", trial["result"]["elapsed_seconds"])
        self.assertEqual(summary["completed_work"]["reported_including_invalid_diagnostic_and_prefix"]["completed_evaluations"], 1048576)
        self.assertIn("1e309", (Path(trial["directory"]) / "output/result.json").read_text())

    def test_nonfinite_scalar_root_keeps_partials_and_stops_engine_and_live(self):
        for command, planned in (("engine", 45), ("live", 20)):
            for mode, raw in (("root_overflow", "1e309"), ("negative_root_overflow", "-1e309")):
                with self.subTest(command=command, raw=raw):
                    self.output = self.root / f"{command}-{mode}"
                    self.binary(self.reference, mode, .3)
                    code, summary = self.run_cli(command)
                    self.assertEqual(code, 1)
                    self.assertIsNotNone(summary, "worker failure must retain summary.json")
                    self.assertEqual(summary["verdict"], "INVALID")
                    self.assertFalse(summary["valid"])
                    self.assertFalse(summary["adoption"])
                    self.assertTrue(any("non-finite number" in error for error in summary["errors"]))
                    self.assertEqual(len(summary["trials"]), 1)
                    trial = summary["trials"][0]
                    self.assertFalse(trial["valid"])
                    self.assertFalse(trial["timing_score_eligible"])
                    self.assertEqual(trial["status"], "invalid")
                    self.assertIsInstance(trial["result"], str)
                    self.assertIn("non-finite", trial["result"])
                    directory = Path(trial["directory"])
                    self.assertEqual((directory / "output/result.json").read_text(), raw)
                    self.assertEqual(bench.read_json(directory / "qualification.json"), trial)
                    self.assertEqual(trial["process"]["returncode"], 0)
                    work = summary["completed_work"]
                    self.assertEqual(work["worker_counts"]["attempted"], 1)
                    self.assertEqual(work["worker_counts"]["invalid"], 1)
                    self.assertEqual(work["worker_counts"]["unrun"], planned - 1)
                    self.assertEqual(work["workers_without_reported_counters"], 1)
                    self.assertEqual(work["reported_including_invalid_diagnostic_and_prefix"]["completed_evaluations"], 0)
                    self.assertTrue(all(p["unrun_reason"] for p in summary["planned_work"][1:]))
                    self.assertEqual(bench.inventory(self.capture), self.before)

    def test_deeply_nested_worker_json_keeps_partials_and_raw_artifact(self):
        for command, planned in (("engine", 45), ("live", 20)):
            with self.subTest(command=command):
                self.output = self.root / f"{command}-deeply-nested"
                self.binary(self.reference, "deeply_nested_json", .3)
                code, summary = self.run_cli(command)
                self.assertEqual(code, 1)
                self.assertIsNotNone(summary, "malformed worker JSON must retain summary.json")
                self.assertEqual(summary["verdict"], "INVALID")
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["adoption"])
                self.assertTrue(summary["errors"])
                self.assertEqual(len(summary["trials"]), 1)
                trial = summary["trials"][0]
                self.assertEqual(trial["status"], "invalid")
                self.assertFalse(trial["valid"])
                self.assertFalse(trial["timing_score_eligible"])
                directory = Path(trial["directory"])
                self.assertEqual((directory / "output/result.json").read_text(), '[' * 1000 + '0' + ']' * 1000)
                self.assertFalse(bench.read_json(directory / "qualification.json")["valid"])
                work = summary["completed_work"]
                self.assertEqual(work["worker_counts"]["attempted"], 1)
                self.assertEqual(work["worker_counts"]["invalid"], 1)
                self.assertEqual(work["worker_counts"]["unrun"], planned - 1)
                self.assertTrue(all(p["unrun_reason"] for p in summary["planned_work"][1:]))
                self.assertEqual(bench.inventory(self.capture), self.before)

    def test_artifact_and_numerical_corruption_invalidates_without_further_work(self):
        cases = (("corrupt_initial", "initial_state_sha256"), ("state_hash_mismatch", "saved canonical state"),
                 ("wrong_completed", "completed_evaluations"), ("unexplained_pixel", "pixel audit"),
                 ("immature", "97"), ("fixture_mutation", "fixture"))
        for mode, expected in cases:
            with self.subTest(mode=mode):
                self.output = self.root / mode
                summary = self.invalid(mode, expected)
                self.assertEqual(len(summary["trials"]), 1)

    def test_candidate_exact_replay_failure_stops_after_two_workers(self):
        self.binary(self.candidate, "replay_mismatch", .3)
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(len(summary["trials"]), 2)
        self.assertTrue(any("final_state_sha256" in e for e in summary["errors"]))

    def test_diagnostic_overflow_is_counted_and_stops_before_prefixes(self):
        self.binary(self.candidate, "overflow", .3)
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(len(summary["trials"]), 11)
        self.assertTrue(summary["trials"][-1]["diagnostic"])
        self.assertTrue(any("validation_max_tile_occupancy" in e for e in summary["errors"]))

    def test_correctness_prefix_is_restarted_and_requires_exact_replay(self):
        self.binary(self.candidate, "prefix_mismatch", .3)
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(len(summary["trials"]), 13)
        self.assertEqual(summary["trials"][-1]["batches"], 8)
        self.assertEqual(summary["trials"][-1]["kind"], "prefix")
        self.assertEqual(summary["completed_work"]["worker_kinds"]["prefix"]["invalid"], 1)
        self.assertTrue((Path(summary["trials"][-1]["directory"]) / "replay-failure.json").is_file())

    def test_fixed_default_settings_are_required_before_measurement(self):
        content = bench.read_json(self.manifest)
        content["params"]["chainCount"] = 3
        self.manifest.write_text(json.dumps(content))
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(summary["trials"], [])
        self.assertTrue(any("settings differ" in e for e in summary["errors"]))

    def test_supplied_build_provenance_is_archived_without_verifying_build(self):
        provenance = self.root / "build.json"
        provenance.write_text(json.dumps({"source_revision": "synthetic", "profile": "release"}))
        self.binary(self.reference, "failure")
        code, summary = self.run_cli("engine", "--reference-provenance", str(provenance))
        self.assertEqual(code, 1)
        identity = summary["binaries"]["reference"]
        self.assertEqual(identity["build"], "supplied_binary_unverified_build")
        self.assertEqual(identity["supplied_provenance"]["sha256"], bench.sha_file(provenance))
        self.assertEqual(bench.read_json(Path(identity["supplied_provenance"]["archived"])), bench.read_json(provenance))

    def test_package_alarm_terminates_worker_under_same_ledger_without_extension(self):
        ledger = qualify.package_clock(None)
        ledger["started_monotonic_seconds"] -= 3568
        ledger["deadline_monotonic_seconds"] -= 3568
        path = self.root / "ledger.json"
        path.write_text(json.dumps(ledger))
        summary = self.invalid("timeout", "package wall deadline", "engine", "--package-ledger", str(path), "--timeout", "120")
        self.assertEqual(len(summary["trials"]), 1)
        self.assertFalse(summary["budget_exceeded"])
        self.assertEqual(summary["package_clock"], ledger)
        self.assertEqual(summary["trials"][0]["process"]["status"], "cancelled")
        pid = int((Path(summary["trials"][0]["directory"]) / "output/worker.pid").read_text())
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.assertEqual(bench.read_json(path), ledger)

    def test_timeout_retains_raw_artifacts_and_leaves_unrelated_child_alive(self):
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as unrelated:
            try:
                summary = self.invalid("timeout", "timed_out", "engine", "--timeout", ".1")
                self.assertIsNone(unrelated.poll())
                self.assertEqual(summary["completed_work"]["worker_counts"]["timed_out"], 1)
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=3)

    def test_duplicate_reserved_drawings_rejected_before_worker(self):
        shutil = qualify.shutil
        for case in bench.read_json(self.original_manifest)["cases"]:
            shutil.copyfile(self.original_manifest.parent / case["drawing_file"], self.manifest.parent / case["drawing_file"])
        original = bench.read_json(self.original_manifest)
        self.manifest.write_text(json.dumps(original))
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(summary["trials"], [])
        self.assertTrue(any("duplicates original" in e for e in summary["errors"]))

    def test_fixed_settings_and_binary_size_are_checked_before_any_worker(self):
        with self.candidate.open("wb") as stream:
            stream.truncate(bench.MAX_BINARY_BYTES + 1)
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(summary["trials"], [])
        self.assertTrue(any("file exceeds" in e for e in summary["errors"]))

    def test_existing_output_immutable_and_timeout_cannot_exceed_cap(self):
        self.output.mkdir()
        sentinel = self.output / "sentinel"
        sentinel.write_text("keep")
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            qualify.main(self.arguments("engine", "--timeout", "121"))

    def test_held_benchmark_lock_is_nonblocking_and_retained_in_summary(self):
        with bench.benchmark_lock(qualify.LOCK):
            code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(summary["trials"], [])
        self.assertTrue(any("lock is held" in e for e in summary["errors"]))

    def test_live_exact_work_aa_before_ab_uses_reference_for_both_aa_sides(self):
        code, summary = self.run_cli("live")
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["protocol_version"], 1)
        self.assertEqual(summary["policy"]["live"]["version"], 2)
        self.assertEqual(summary["live_confirmation"], "passed")
        self.assertEqual(summary["verdict"], "INCONCLUSIVE")
        self.assertEqual(len(summary["trials"]), 20)
        self.assertEqual(summary["completed_work"]["reported_including_invalid_diagnostic_and_prefix"]["completed_evaluations"], 83886080)
        self.assertEqual({t["binary_side"] for t in summary["trials"][:10]}, {"reference"})
        self.assertEqual({t["phase"] for t in summary["trials"][:10]}, {"aa"})
        self.assertTrue(summary["aa_statistics"]["passed"])
        self.assertTrue(summary["ab_statistics"]["passed"])
        self.assertEqual(summary["aa_statistics"]["latency"]["unit"], "run pair")
        self.assertEqual(summary["policy"]["live"]["deadline_ms"], 300)
        for trial in summary["trials"]:
            self.assertEqual(trial["result"]["protocol_version"], 1)
            self.assertEqual(trial["result"]["live_protocol_version"], 2)
            self.assertEqual(trial["receipt_audit"]["publications"], 128)
            self.assertEqual(trial["receipt_audit"]["shards"], 1)
            output = Path(trial["directory"]) / "output"
            metadata = trial["result"]["receipt_artifacts"]["receipt-000.jsonl"]
            self.assertEqual(bench.sha_file(output / "receipt-000.jsonl"), metadata["sha256"])
            with (output / "receipt-000.jsonl").open("rb") as stream:
                frame = json.loads(stream.readline())
            publication = trial["result"]["publications"][0]
            self.assertEqual(hashlib.sha256(base64.b64decode(frame["image"])).hexdigest(), publication["png_sha256"])
            self.assertEqual(hashlib.sha256(frame["drawingJson"].encode()).hexdigest(), publication["drawing_sha256"])

    def test_live_legacy_missing_or_noninteger_protocol_cannot_qualify(self):
        modes = ("live_protocol_missing", "live_protocol_1", "live_protocol_0", "live_protocol_3",
                 "live_protocol_2.0", "live_protocol_true", "live_protocol_null", 'live_protocol_"2"')
        for mode in modes:
            with self.subTest(mode=mode):
                self.output = self.root / mode
                summary = self.invalid(mode, "live_protocol_version", "live")
                self.assertEqual(len(summary["trials"]), 1)
                self.assertEqual(summary["completed_work"]["worker_counts"]["invalid"], 1)
                self.assertEqual(summary["completed_work"]["worker_counts"]["unrun"], 19)
                self.assertFalse(summary["planned_work_complete"])

    def test_aa_noise_stops_before_ab_without_adding_samples(self):
        self.binary(self.reference, "live_aa_noise")
        code, summary = self.run_cli("live")
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["aa_qualification"], "inconclusive")
        self.assertEqual(summary["verdict"], "INCONCLUSIVE")
        self.assertEqual(len(summary["trials"]), 10)
        self.assertEqual(summary["completed_work"]["worker_counts"]["unrun"], 10)

    def test_live_receipt_sequence_spans_three_shards(self):
        self.binary(self.reference, "receipt_three_shards")
        code, summary = self.run_cli("live")
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["live_confirmation"], "passed")
        self.assertEqual({trial["receipt_audit"]["shards"] for trial in summary["trials"]}, {1, 3})
        for trial in summary["trials"]:
            self.assertEqual(trial["receipt_audit"]["publications"], 128)
            self.assertEqual(trial["receipt_audit"]["bytes"],
                             sum(metadata["bytes"] for metadata in trial["result"]["receipt_artifacts"].values()))

    def test_live_malformed_publications_cannot_qualify(self):
        cases = (("live_missing", "previews_received"), ("live_bad_p95", "p95 latency"),
                 ("live_payload", "payload_valid"), ("live_wrong_origin", "completed_evaluations"),
                 ("live_duplicate", "sequence"), ("live_inverted_time", "receive_ms"),
                 ("live_timeline", "timeline regressed"), ("live_cache", "cache seed differs"),
                 ("live_deadline", "deadline was missed"), ("live_terminal", "packed population maximum"))
        for mode, expected in cases:
            with self.subTest(mode=mode):
                self.output = self.root / mode
                summary = self.invalid(mode, expected, "live")
                self.assertEqual(len(summary["trials"]), 1)

    def test_live_missing_receipt_manifest_cannot_qualify(self):
        summary = self.invalid("receipt_no_manifest", "receipt", "live")
        self.assertEqual(len(summary["trials"]), 1)

    def test_live_corrupt_or_unbounded_receipt_shards_cannot_qualify(self):
        cases = (("receipt_empty_manifest", "receipt"), ("receipt_missing_file", "receipt"),
                 ("receipt_corrupt_file", "receipt"), ("receipt_bad_hash", "receipt"),
                 ("receipt_bad_size", "receipt"), ("receipt_path_escape", "receipt"),
                 ("receipt_symlink", "receipt"), ("receipt_too_many", "receipt"),
                 ("receipt_shard_limit", "receipt"), ("receipt_total_limit", "receipt"))
        for mode, expected in cases:
            with self.subTest(mode=mode):
                self.output = self.root / mode
                summary = self.invalid(mode, expected, "live")
                self.assertEqual(len(summary["trials"]), 1)
                self.assertTrue((Path(summary["trials"][0]["directory"]) / "qualification.json").is_file())

    def test_live_raw_frames_must_match_publications_and_payloads(self):
        modes = ("receipt_meta_run_id", "receipt_meta_sequence", "receipt_meta_completed_batches",
                 "receipt_meta_completed_evaluations", "receipt_meta_due_ms", "receipt_meta_send_ms",
                 "receipt_meta_drawing_sha256", "receipt_meta_png_sha256", "receipt_png_hash",
                 "receipt_bad_base64", "receipt_png_dimensions", "receipt_drawing_hash",
                 "receipt_fitness", "receipt_polygons", "receipt_drawing_fitness", "receipt_drawing_polygons",
                 "receipt_wrong_type", "receipt_wire_origin", "receipt_missing_line", "receipt_extra_line",
                 "receipt_bad_json", "receipt_duplicate_key", "receipt_long_line", "receipt_no_newline",
                 "receipt_nested_json", "receipt_nonfinite_json")
        for mode in modes:
            with self.subTest(mode=mode):
                self.output = self.root / mode
                summary = self.invalid(mode, "receipt", "live")
                self.assertEqual(len(summary["trials"]), 1)
                trial = summary["trials"][0]
                self.assertEqual(summary["completed_work"]["worker_counts"]["invalid"], 1)
                self.assertEqual(summary["completed_work"]["reported_including_invalid_diagnostic_and_prefix"]["completed_evaluations"], 4194304)
                self.assertTrue((Path(trial["directory"]) / "output/receipt-000.jsonl").is_file())

    def test_invalid_live_setup_keeps_partial_result_with_unavailable_fields(self):
        summary = self.invalid("invalid_setup", "subprocess failed", "live")
        result = summary["trials"][0]["result"]
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["evals_per_second"])
        self.assertNotIn("initial_state_sha256", result)

    def test_live_candidate_slowdown_rejects_without_adoption(self):
        self.binary(self.candidate, "live_slow", .3)
        code, summary = self.run_cli("live")
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["verdict"], "REJECT")
        self.assertFalse(summary["ab_statistics"]["passed"])
        self.assertFalse(summary["adoption"])

    def test_sigterm_and_sigint_cleanup_owned_child_and_write_summary(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=sig):
                self.output = self.root / str(sig)
                self.check_cancellation(sig)

    def check_cancellation(self, sig):
        self.binary(self.reference, "timeout")
        wrapper = self.root / "supervisor.py"
        wrapper.write_text("import importlib.util,signal,sys\nfrom pathlib import Path\n"
                           f"s=importlib.util.spec_from_file_location('support',{str(Path(__file__).resolve())!r})\n"
                           "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                           f"m.qualify.LOCK=Path({str(self.root / 'benchmark.lock')!r})\n"
                           "m.bench.telemetry=m.support.support.telemetry_sample\n"
                           "before=signal.getsignal(signal.SIGTERM)\n"
                           "code=m.qualify.main(sys.argv[1:])\n"
                           "assert signal.getsignal(signal.SIGTERM)==before\n"
                           "sys.exit(code)\n")
        command = [sys.executable, "-B", "-W", "error", str(wrapper), *self.arguments()]
        pid_path = self.output / "trials/engine-ff-timing-64-01-reference/output/worker.pid"
        worker = None
        with ((self.root / f"supervisor-{sig}.log").open("wb") as log,
              subprocess.Popen(command, stdout=log, stderr=log) as supervisor):
            try:
                for _ in range(300):
                    if pid_path.is_file() and pid_path.stat().st_size:
                        worker = int(pid_path.read_text())
                        break
                    time.sleep(.01)
                self.assertIsNotNone(worker)
                supervisor.send_signal(sig)
                self.assertEqual(supervisor.wait(timeout=6), 1)
            finally:
                if supervisor.poll() is None:
                    supervisor.kill()
                    supervisor.wait(timeout=3)
        summary = bench.read_json(self.output / "summary.json")
        self.assertTrue(summary["cancelled"])
        self.assertEqual(summary["status"], "cancelled")
        self.assertEqual(summary["trials"][0]["process"]["status"], "cancelled")
        self.assertFalse(Path(f"/proc/{worker}").exists())


class StatisticsAndBudgetTests(unittest.TestCase):
    def test_live_constant_ratios_and_fixed_seed_resample_run_pairs(self):
        stats = qualify.paired_log_bootstrap([1.01] * 5)
        self.assertAlmostEqual(stats["interval95"][0], 1.01)
        self.assertAlmostEqual(stats["interval95"][1], 1.01)
        self.assertEqual(stats["resamples"], 10000)
        self.assertEqual(stats["bootstrap_seed"], 20261023)
        first = qualify.paired_log_bootstrap([1, 1, 1, 1, 2])
        self.assertEqual(first, qualify.paired_log_bootstrap([1, 1, 1, 1, 2]))
        self.assertEqual(first["paired_ratios"]["median"], 1)
        self.assertAlmostEqual(first["interval95"][1], 2 ** .6)

    def test_live_equivalence_and_ab_threshold_boundaries(self):
        pairs = [{"ratio": .99, "latency_ratio": 1.01, "missed_deadline_increase": 0} for _ in range(5)]
        self.assertTrue(qualify.live_gates(pairs, True)["passed"])
        self.assertFalse(qualify.live_gates(pairs, False)["passed"])
        for pair in pairs:
            pair.update(ratio=1, latency_ratio=1)
        self.assertTrue(qualify.live_gates(pairs, False)["passed"])
        pairs[0]["missed_deadline_increase"] = 1
        self.assertFalse(qualify.live_gates(pairs, False)["passed"])
        for pair in pairs:
            pair.update(ratio=.989, latency_ratio=1)
        self.assertFalse(qualify.live_gates(pairs, True)["passed"])

    def test_invalid_live_statistics_are_not_repeated_into_three_cases(self):
        for ratios in ([1], [1] * 6, [0] * 5, [10**309] * 5, [True] * 5):
            with self.subTest(ratios=ratios), self.assertRaises(qualify.FAILURES):
                qualify.paired_log_bootstrap(ratios)

    def test_engine_three_percent_threshold_remains_strict_policy1(self):
        summary = {"pairs": [{"case_id": name, "pair": i + 1, "valid": True,
                               "ratio": 1.03, "reference_rate": 100, "candidate_rate": 103}
                              for name in bench.PROJECTS for i in range(5)]}
        with self.assertRaises(qualify.GateStop):
            qualify.engine_analysis(summary)
        self.assertEqual(summary["verdict"], "REJECT")
        self.assertFalse(summary["screen_gates"]["practical_gain"])

    def test_ledger_rejects_extension_crossboot_expiry_and_future_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            current = qualify.package_clock(None)
            path.write_text(json.dumps(current))
            self.assertEqual(qualify.package_clock(path), current)
            for updates in ({"deadline_monotonic_seconds": current["deadline_monotonic_seconds"] + 1},
                            {"clock_boot_id": "different"},
                            {"started_monotonic_seconds": time.monotonic() + 100},
                            {"started_monotonic_seconds": 0, "deadline_monotonic_seconds": 3600}):
                path.write_text(json.dumps(dict(current, **updates)))
                with self.subTest(updates=updates), self.assertRaises(bench.BenchError):
                    qualify.package_clock(path)


if __name__ == "__main__":
    unittest.main()
