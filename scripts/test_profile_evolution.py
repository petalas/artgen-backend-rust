"""Profile contract tests; children are Python fakes, never perf/Cargo/GPU jobs.

Run: python3 -B -W error -m unittest discover -s scripts -p test_profile_evolution.py
"""

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def local_module(name, filename):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


profile = local_module("evolution_profile", "profile-evolution.py")
support = local_module("evolution_profile_test_support", "test_evolution_bench.py")
bench = profile.bench
NORMAL_FAKE = support.FAKE_EXECUTABLE.replace('"elapsed_seconds": 0.25,',
                                            '"elapsed_seconds": 0.25, "setup_seconds": 0.1,')
ENVELOPE_FAKE = NORMAL_FAKE.replace(
    '(output / "state.json").write_text(json.dumps(state))',
    '''if "--profile-mode" in sys.argv:
    record.update(timestamp_mode="batch_envelope", evals_per_second=None,
                  batch_compute_envelope={"compute_ns": 100000000, "sample_count": batches},
                  sampled_gpu_stage_totals_ns=[10, 20, 30, 0])
if MODE == "saved_state_corruption":
    state["parents"][0]["rng_state"] += 1
(output / "state.json").write_text(json.dumps(state))''')


class ProfileBehaviorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.capture = self.root / "baseline"
        self.capture.mkdir()
        self.manifest = support.fixture_corpus(self.capture / "fixtures")
        self.reference_binary = self.capture / "evolution-bench"
        self.reference_binary.write_text(NORMAL_FAKE)
        self.reference_binary.chmod(0o755)
        self.output = self.root / "profile"
        self.envelope_binary = self.root / "envelope-bench"
        self.set_mode("ok")
        self.schedule = argparse.Namespace(seed=7, batches=64, warmup=2, timeout=3)
        normal = self.capture / "normal"
        case = next(c for c in bench.load_manifest(self.manifest)["cases"] if c["case_id"] == "ff")
        self.case = case
        command = profile.trial_command(self.reference_binary, self.manifest, case, self.schedule, normal, False)
        process = subprocess.run(command, cwd=self.root, capture_output=True, timeout=3, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.reference = bench.read_json(normal / "result.json")
        self.manifest_hash = bench.sha_file(self.manifest)
        cache = self.capture / "cache-seed"
        cache.mkdir()
        (cache / "seed.bin").write_bytes(b"immutable seed")
        summary = {"command": "baseline", "valid": True, "error_count": 0,
                   "binary": {"archived": str(self.reference_binary), "sha256": bench.sha_file(self.reference_binary)},
                   "manifest": str(self.manifest), "manifest_sha256": self.manifest_hash,
                   "fixture_inventory": bench.inventory(self.manifest.parent),
                   "cache_policy": {"files": bench.inventory(cache)},
                   "arguments": {"seed": 7, "batches": 64, "warmup": 2},
                   "trials": [{"case_id": "ff", "diagnostic": False, "valid": True, "result": self.reference}]}
        bench.write_json(self.capture / "summary.json", summary)
        self.before = bench.inventory(self.capture)
        patch = mock.patch.object(profile, "LOCK", self.root / "benchmark.lock")
        patch.start()
        self.addCleanup(patch.stop)

    def set_mode(self, mode):
        self.envelope_binary.write_text(ENVELOPE_FAKE.replace('MODE = "ok"', f'MODE = "{mode}"'))
        self.envelope_binary.chmod(0o755)

    def envelope_record(self):
        record = copy.deepcopy(self.reference)
        record.update(timestamp_mode="batch_envelope", evals_per_second=None,
                      batch_compute_envelope={"compute_ns": 100000000, "sample_count": 64},
                      sampled_gpu_stage_totals_ns=[10, 20, 30, 0])
        return record

    def validate(self, record):
        return profile.validate_profile_record(record, self.reference, self.case,
                                               self.schedule, self.manifest_hash, True)

    def run_cli(self, *extra):
        argv = ["--baseline", str(self.capture / "summary.json"), "--output", str(self.output),
                "--case", "ff", "--binary", str(self.envelope_binary), "--timeout", "3", *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = profile.main(argv)
        path = self.output / "summary.json"
        return code, bench.read_json(path) if path.is_file() else None

    def fake_perf(self, body):
        directory = self.root / "bin"
        directory.mkdir()
        path = directory / "perf"
        path.write_text(f"#!{sys.executable}\nimport subprocess,sys\n"
                        "if '--version' in sys.argv:\n print('synthetic perf'); sys.exit(0)\n" + body)
        path.chmod(0o755)
        return mock.patch.dict(os.environ, {"PATH": str(directory) + os.pathsep + os.environ["PATH"]})

    def test_envelope_only_cli_preserves_reference_and_records_completed_work(self):
        with mock.patch.object(profile.shutil, "which", side_effect=AssertionError("must not invoke perf")):
            code, summary = self.run_cli("--envelope-only")
        self.assertEqual(code, 0, summary)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["error_count"], 0)
        self.assertFalse(summary["campaign_frozen"])
        self.assertFalse(summary["optimization_claim"])
        self.assertFalse(summary["primary_score_eligible"])
        self.assertEqual(summary["schedule"], {"seed": 7, "batches": 64, "warmup": 2})
        entry, = summary["profiles"]
        self.assertEqual(entry["mode"], "batch_envelope")
        self.assertTrue(entry["valid"])
        self.assertIsNone(entry["result"]["evals_per_second"])
        self.assertEqual(entry["result"]["final_state_sha256"], self.reference["final_state_sha256"])
        self.assertAlmostEqual(entry["analysis"]["gpu_queue_critical_span_fraction"], 0.4)
        self.assertEqual(entry["analysis"]["last_iteration_samples"]["selection"], "unavailable")
        self.assertEqual(summary["completed_work"]["reported_including_invalid_and_diagnostic"],
                         {"completed_batches": 64, "completed_iterations": 192, "completed_evaluations": 1536})
        self.assertEqual(bench.inventory(self.capture), self.before)
        self.assertTrue((self.output / "ff-batch_envelope/process/stderr.log").is_file())

    def test_saved_state_corruption_fails_even_with_matching_reported_hashes(self):
        self.set_mode("saved_state_corruption")
        code, summary = self.run_cli("--envelope-only")
        self.assertEqual(code, 1)
        self.assertFalse(summary["valid"])
        self.assertIn("saved canonical state", summary["errors"][0])
        self.assertEqual(summary["profiles"][0]["result"]["final_state_sha256"], self.reference["final_state_sha256"])

    def test_clone_mutation_is_rejected_without_changing_baseline(self):
        self.set_mode("fixture_mutation")
        code, summary = self.run_cli("--envelope-only")
        self.assertEqual(code, 1)
        self.assertIn("fixture changed", summary["errors"][0])
        self.assertEqual(bench.inventory(self.capture), self.before)

    def test_failed_worker_keeps_artifacts_and_error_count(self):
        self.set_mode("failure")
        code, summary = self.run_cli("--envelope-only")
        self.assertEqual(code, 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertTrue((self.output / "ff-batch_envelope/output/partial.txt").is_file())

    def test_missing_perf_is_partial_evidence_without_building(self):
        with mock.patch.object(profile.shutil, "which", return_value=None), \
                mock.patch.object(profile, "run_owned", return_value=({"status": "failed"}, {})), \
                mock.patch.object(bench, "prepare_binary", side_effect=AssertionError("host must not build")):
            code, summary = self.run_cli("--host-only")
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["unavailable_profiles"], 1)
        self.assertFalse(summary["profiles"][0]["valid"])

    def test_perf_failure_records_not_available(self):
        with self.fake_perf("print('No permission to enable cpu-clock:u event.', file=sys.stderr); sys.exit(255)\n"):
            code, summary = self.run_cli("--host-only")
        self.assertEqual(code, 0)
        self.assertEqual(summary["profiles"][0]["status"], "not_available")

    def test_worker_crash_after_start_is_an_error_not_missing_perf(self):
        source = NORMAL_FAKE.replace('MODE = "ok"', 'MODE = "failure"').replace(
            'print("raw child stderr", file=sys.stderr, flush=True)',
            'print("GPU initialized; [timestamps] sample #1: fake", file=sys.stderr, flush=True)')
        self.reference_binary.write_text(source)
        path = self.capture / "summary.json"
        summary = bench.read_json(path)
        summary["binary"]["sha256"] = bench.sha_file(self.reference_binary)
        path.write_text(json.dumps(summary))
        # A permission-like line after a worker started must not hide its crash.
        body = ("code=subprocess.call(sys.argv[sys.argv.index('--')+1:])\n"
                "print('No permission to enable cpu-clock:u event.', file=sys.stderr)\nsys.exit(code)\n")
        with self.fake_perf(body):
            code, summary = self.run_cli("--host-only")
        self.assertEqual(code, 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["profiles"][0]["status"], "invalid")
        self.assertEqual(summary["unavailable_profiles"], 0)
        self.assertTrue((self.output / "ff-host/output/partial.txt").is_file())

    def test_tampered_archived_binary_fails_before_execution(self):
        self.reference_binary.write_text("not the archived executable")
        with mock.patch.object(profile, "run_owned", side_effect=AssertionError("must not execute")):
            code, summary = self.run_cli("--envelope-only")
        self.assertEqual(code, 1)
        self.assertIn("binary SHA", summary["errors"][0])

    def test_existing_output_is_refused(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        code, summary = self.run_cli("--envelope-only")
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_profile_mode_requires_null_score_and_complete_samples(self):
        original = self.envelope_record()
        variants = [dict(original, timestamp_mode=None), dict(original, timestamp_samples=66),
                    dict(original, evals_per_second=123), dict(original, error_count=1),
                    dict(original, final_state_sha256="0" * 64), dict(original, gpu={"name": "changed"}),
                    dict(original, sampled_gpu_stage_totals_ns=[10, 20, 30, 1]),
                    dict(original, batch_compute_envelope={"compute_ns": 1, "sample_count": 66}),
                    dict(original, batch_compute_envelope={"compute_ns": 0, "sample_count": 64}),
                    dict(original, batch_compute_envelope={"compute_ns": 300000000, "sample_count": 64})]
        missing = copy.deepcopy(original)
        del missing["evals_per_second"]
        variants.append(missing)
        for index, record in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(bench.BenchError):
                self.validate(record)

    def test_pixel_audit_must_remain_exact_even_when_new_audit_is_valid(self):
        record = self.envelope_record()
        record["final_quality"][0]["pixel_audit"]["outside_nominal_quantum"] = 1
        with self.assertRaises(bench.BenchError):
            self.validate(record)

    def test_valid_profile_is_ineligible_for_primary_controller(self):
        record = self.envelope_record()
        self.assertAlmostEqual(self.validate(record)["gpu_queue_critical_span_fraction"], 0.4)
        with self.assertRaises(bench.BenchError):
            bench.validate_result(record, self.case, self.schedule, False, self.manifest_hash)

    def test_zero_warmup_baselines_are_explicitly_unsupported(self):
        path = self.capture / "summary.json"
        value = bench.read_json(path)
        value["arguments"]["warmup"] = 0
        path.write_text(json.dumps(value))
        code, summary = self.run_cli("--envelope-only")
        self.assertEqual(code, 1)
        self.assertIn("warmup", summary["errors"][0])


class HostEvidenceTests(unittest.TestCase):
    def raw_samples(self):
        return {"started_monotonic": 100, "finished_monotonic": 113,
                "first_sample_messages": [100.5, 102], "sample_limit": 100, "ticks_per_second": 100,
                "samples": [[103 + i * .025, i, 0, i, 0, "S", "do_sys_poll"] for i in range(81)]}

    def test_proc_stat_preserves_field_offsets_with_spaces_and_parentheses(self):
        fields = ["S"] + ["0"] * 49
        fields[11], fields[12], fields[19] = "123", "45", "6789"
        actual = profile.parse_proc_stat("1234 (thread ) with spaces) " + " ".join(fields))
        self.assertEqual(actual, {"state": "S", "utime": 123, "stime": 45, "starttime": 6789})

    def test_guarded_window_wait_summary_and_cpu_tick_scope(self):
        report = profile.analyze_host(self.raw_samples(), {"setup_seconds": 2, "elapsed_seconds": 10})
        self.assertEqual(report["window_monotonic"], [103, 111])
        self.assertEqual(report["interior_samples"], 81)
        self.assertAlmostEqual(report["process_cpu_seconds"], .8)
        self.assertAlmostEqual(report["process_one_core_fraction"], .4)
        self.assertEqual(report["main_wait_channels"], {"do_sys_poll": 81})
        self.assertEqual(report["main_wait_channel_fractions"], {"do_sys_poll": 1})
        self.assertIn("not GPU utilization", report["interpretation"])

    def test_missing_marker_or_too_few_samples_cannot_produce_cpu_claim(self):
        for raw in (dict(self.raw_samples(), first_sample_messages=[102]),
                    dict(self.raw_samples(), samples=self.raw_samples()["samples"][:19]),
                    dict(self.raw_samples(), first_sample_messages=[100.5, 111])):
            with self.subTest(raw=raw), self.assertRaises(bench.BenchError):
                profile.analyze_host(raw, {"setup_seconds": 2, "elapsed_seconds": 10})

    def test_timeout_kills_owned_descendants_preserves_unrelated_process(self):
        child = "import time; time.sleep(30)"
        program = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); "
                   "print(p.pid,flush=True); time.sleep(30)")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with subprocess.Popen([sys.executable, "-c", child], start_new_session=True) as unrelated:
                try:
                    record, raw = profile.run_owned([sys.executable, "-c", program], root, root / "process",
                                                    .3, Path(sys.executable))
                    self.assertEqual(record["status"], "timed_out")
                    self.assertIsNone(unrelated.poll())
                    worker = int((root / "process/stdout.log").read_text())
                    self.assertEqual(raw["worker_pid"], worker)
                    self.assertTrue(raw["samples"])
                    path = Path(f"/proc/{worker}/stat")
                    if path.exists():
                        self.assertEqual(profile.parse_proc_stat(profile.read_small(path))["state"], "Z")
                    self.assertTrue((root / "process/host-samples.json").is_file())
                finally:
                    unrelated.kill()
                    unrelated.wait(timeout=3)

    def test_cancellation_reaps_child_and_retains_process_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(profile, "observe_process", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    profile.run_owned([sys.executable, "-c", "import time; time.sleep(30)"],
                                      root, root / "process", 1)
            record = bench.read_json(root / "process/process.json")
            self.assertEqual(record["status"], "cancelled")
            self.assertIsNotNone(record["returncode"])


if __name__ == "__main__":
    unittest.main()
