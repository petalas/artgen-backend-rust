"""Behavioral supervisor tests. Synthetic child processes never initialize a GPU.

Run: python3 -W error -m unittest discover -s scripts -p test_evolution_bench.py
The child executable exercises the subprocess/filesystem protocol. It makes no
claim about the numerical or shader correctness owned by the Rust runner.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("evolution_bench", Path(__file__).with_name("evolution-bench.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

FAKE_EXECUTABLE = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

MODE = "ok"
def arg(name):
    return sys.argv[sys.argv.index(name) + 1]
def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()
print("raw child stdout", flush=True)
print("raw child stderr", file=sys.stderr, flush=True)
if sys.argv[1] == "freeze":
    shutil.copytree(arg("--projects-dir"), arg("--fixtures"))
    sys.exit(0)
output = Path(arg("--output"))
output.mkdir()
if MODE == "timeout":
    (output / "partial.txt").write_text("started")
    time.sleep(30)
if MODE == "signal":
    os.kill(os.getpid(), signal.SIGTERM)
if MODE == "failure":
    (output / "partial.txt").write_text("failed")
    sys.exit(3)
if MODE == "missing":
    sys.exit(0)
if MODE == "unparseable":
    (output / "result.json").write_text("not json")
    sys.exit(0)
if MODE == "bad_top_level":
    (output / "result.json").write_text("[]")
    sys.exit(0)
manifest_path = Path(arg("--manifest"))
manifest = json.loads(manifest_path.read_text())
params = manifest["params"]
batches = int(arg("--batches"))
iterations = batches * params["gpuBatchIters"]
evals = iterations * params["chainCount"] * params["lambda"]
counters = {"submitted_batches": batches, "completed_batches": batches,
            "submitted_iterations": iterations, "completed_iterations": iterations,
            "submitted_evaluations": evals, "completed_evaluations": evals}
case = next(case for case in manifest["cases"] if case["id"] == arg("--case"))
initial_quality = [{"chain": index, "triangles": 400, "stored_error": 100,
                    "framebuffer_error": 100, "scalar_error": 100,
                    "stored_fitness": 98.25, "expected_stored_fitness": 98.25,
                    "scalar_fitness": 98.25, "differing_pixels": 0,
                    "max_channel_delta": 0, "total_channel_delta": 0,
                    "pixel_audit": {"outside_nominal_quantum": 0,
                                    "unexplained_pixels": 0, "widest_channel_envelope": 0}}
                   for index in range(params["chainCount"])]
final_quality = [dict(value, stored_fitness=98.5, expected_stored_fitness=98.5,
                      scalar_fitness=98.5) for value in initial_quality]
record = {"protocol_version": 1, "status": "completed", "error_count": 0,
          "errors": [], "case_id": arg("--case"), "seed": int(arg("--seed")),
          "batches": batches, "diagnostic": arg("--diagnostic") == "true",
          "elapsed_seconds": 0.25, "drain_seconds": 0.02,
          "evals_per_second": None if arg("--diagnostic") == "true" else evals / 0.25, "counters": counters,
          "initial_state_sha256": hashed("initial"), "final_state_sha256": hashed("final"),
          "fixture_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          "initial_similarity_percent": 98.25,
          "final_fitness_percent": [98.5] * params["chainCount"],
          "initial_complexity": case["complexity"],
          "final_complexity": [case["complexity"]] * params["chainCount"],
          "initial_quality": initial_quality, "final_quality": final_quality,
          "start_mode": "mature_drawing_restart", "timestamp_samples": batches,
          "sampled_gpu_stage_totals_ns": [10, 20, 30, 40],
          "validation_max_tile_occupancy": 200 if arg("--diagnostic") == "true" else None,
          "correctness": True, "params": params,
          "gpu": {"name": "synthetic subprocess, no GPU", "driver": "not applicable"}}
label = output.parent.name
if MODE == "wrong_completed":
    record["counters"]["completed_evaluations"] -= 1
if MODE == "malformed_counters":
    record["counters"] = ["wrong shape"]
if MODE == "wrong_submitted":
    record["counters"]["submitted_batches"] += 1
if MODE == "wrong_iterations":
    record["counters"]["submitted_iterations"] += 1
    record["counters"]["completed_iterations"] += 1
if MODE == "wrong_case":
    record["case_id"] = "unrequested"
if MODE == "wrong_manifest":
    record["fixture_manifest_sha256"] = hashed("wrong")
if MODE == "wrong_rate":
    if record["evals_per_second"] is not None:
        record["evals_per_second"] *= 2
if MODE == "quality_mismatch" and label.endswith("B"):
    record["final_fitness_percent"][0] -= 0.125
if MODE == "overflow" and label.endswith("diagnostic"):
    record["validation_max_tile_occupancy"] = 257
if MODE == "immature":
    record["initial_similarity_percent"] = 97.0
if MODE == "missing_metrics":
    del record["initial_complexity"]
if MODE == "missing_errors":
    del record["error_count"]
if MODE == "missing_quality":
    del record["final_quality"]
if MODE == "quality_replay" and label.endswith("B"):
    record["final_quality"][0].update(scalar_error=101, max_channel_delta=1,
                                     total_channel_delta=1, differing_pixels=1)
if MODE == "regression":
    record["final_fitness_percent"][0] = 98.0
    record["final_quality"][0].update(stored_fitness=98.0, expected_stored_fitness=98.0,
                                     scalar_fitness=98.0)
if MODE == "unexplained_pixel":
    record["final_quality"][0]["pixel_audit"]["unexplained_pixels"] = 1
if MODE == "missing_audit":
    del record["final_quality"][0]["pixel_audit"]
if MODE == "audited_edge":
    for quality in record["final_quality"]:
        quality.update(scalar_error=148, max_channel_delta=9,
                       total_channel_delta=48, differing_pixels=6,
                       pixel_audit={"outside_nominal_quantum": 6,
                                    "unexplained_pixels": 0, "widest_channel_envelope": 9})
if MODE == "reported_error":
    record["error_count"] = 1
    record["errors"] = ["independent numerical mismatch"]
    record["status"] = "invalid"
    record["correctness"] = False
if MODE == "fixture_mutation":
    target = manifest_path.parent / manifest["cases"][0]["drawing_file"]
    target.chmod(0o644)
    target.write_text("mutated")
if MODE == "duplicate_json":
    (output / "result.json").write_text('{"status":"completed","status":"invalid"}')
    sys.exit(0)
if MODE == "nan":
    record["elapsed_seconds"] = float("nan")
if MODE == "cache_check":
    assert Path("pipeline_cache_test.bin").read_bytes() == b"fixed cache seed"
    Path("pipeline_cache_test.bin").write_bytes(b"mutated child cache")
Path("child-cwd.txt").write_text(str(Path.cwd()))
parents = [{"polygon_count": 400, "fitness_bits": 1120206848,
            "mutation_scale_bits": 1065353216, "stagnation_counter": 0,
            "rng_state": index, "polygons": [[0, 0, 0, 0]] * 400}
           for index in range(params["chainCount"])]
state = {"parents": parents, "offspring": [], "parent_total_errors": [100] * len(parents),
         "parent_framebuffers": [[0] * case["width"] * case["height"]] * len(parents),
         "offspring_errors": [], "chain_update_flags": [], "fitness_packed": [],
         "counters": counters, "validation_max_tile_occupancy": record["validation_max_tile_occupancy"]}
if (MODE == "replay_mismatch" and label.endswith("B") or
        MODE == "diagnostic_mismatch" and label.endswith("diagnostic")):
    state["parents"][0]["rng_state"] = 9999
canonical = [state[field] for field in ("parents", "offspring", "parent_total_errors",
             "parent_framebuffers", "offspring_errors", "chain_update_flags", "fitness_packed")]
record["final_state_sha256"] = hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()
if MODE == "no_evolution":
    record["initial_state_sha256"] = record["final_state_sha256"]
if MODE == "profiling_score":
    record["timestamp_mode"] = "batch_envelope"
    record["batch_compute_envelope"] = {"compute_ns": 1000, "sample_count": batches}
if MODE == "state_hash_mismatch":
    record["final_state_sha256"] = hashed("wrong canonical state hash")
(output / "state.json").write_text(json.dumps(state))
(output / "drawings.json").write_text(json.dumps([{"polygons": []}] * len(parents)))
(output / "result.json").write_text(json.dumps(record))
if MODE == "completed_nonzero":
    sys.exit(5)
'''


def write_json(path, content):
    path.write_text(json.dumps(content), encoding="utf-8")


def fixture_corpus(root):
    root.mkdir()
    cases = []
    for project in ("ff", "mona", "darwin"):
        directory = root / project
        directory.mkdir()
        drawing = directory / "drawing.json"
        drawing.write_text('{"polygons":[],"fitness":98.25}')
        reference = directory / "reference.rgba"
        reference.write_bytes(bytes([12, 34, 56, 255]) * 384 * 2)
        complexity = {"triangles": 400, "vertices": 1200, "overflowed_tiles": 0,
                      "tile_count": 24, "triangle_area": {"min": 0, "median": 0.1, "p95": 0.2, "max": 1},
                      "bbox_area": {"min": 0, "median": 0.1, "p95": 0.2, "max": 1},
                      "tile_overlaps": {"min": 0, "p50": 100, "p95": 120, "max": 200}}
        cases.append({"id": project, "width": 384, "height": 2,
                      "drawing_file": f"{project}/drawing.json",
                      "reference_file": f"{project}/reference.rgba",
                      "drawing_sha256": hashlib.sha256(drawing.read_bytes()).hexdigest(),
                      "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                      "source_image_sha256": "a" * 64, "stored_fitness_percent": 98.25,
                      "scalar_fitness_percent": 98.25, "complexity": complexity})
    write_json(root / "manifest.json", {"protocol_version": 1, "resolution": 384,
               "start_mode": "mature_drawing_restart", "minimum_similarity_percent": 97.0,
               "params": {"gpuBatchIters": 3, "chainCount": 2, "lambda": 4}, "cases": cases})
    return root / "manifest.json"


def telemetry_sample(output, label):
    path = output / label
    path.mkdir()
    value = {"utc": "synthetic", "cpu_count": 4, "cpu_load_average": [1, 1, 1],
             "gpu": {"status": "unavailable", "reason": "test does not access GPUs"},
             "backend": {"status": "observed", "evolving": True}}
    write_json(path / "telemetry.json", value)
    return value


class SupervisorBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = fixture_corpus(self.root / "corpus")
        self.executable = self.root / "fake-bench"
        self.set_mode("ok")
        self.output = self.root / "evidence"
        self.lock = self.root / "benchmark.lock"
        # Machine probes are separate boundaries. These tests run only the
        # synthetic child, never cargo, toolchain probes, telemetry, or a backend.
        self.provenance = mock.patch.object(bench, "machine_provenance", return_value={"test": True})
        self.provenance.start()
        self.addCleanup(self.provenance.stop)
        self.telemetry = mock.patch.object(bench, "telemetry", side_effect=telemetry_sample)
        self.telemetry.start()
        self.addCleanup(self.telemetry.stop)

    def set_mode(self, mode):
        self.executable.write_text(FAKE_EXECUTABLE.replace('MODE = "ok"', f'MODE = "{mode}"'))
        self.executable.chmod(0o755)

    def run_supervisor(self, mode="ok", command="validate", extra=()):
        self.set_mode(mode)
        argv = [command, "--binary", str(self.executable), "--output", str(self.output),
                "--lock", str(self.lock), "--timeout", "2"]
        if command != "freeze":
            argv.extend(["--manifest", str(self.manifest)])
        argv.extend(extra)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = bench.main(argv)
        summary_path = self.output / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        return code, summary

    def assert_failure(self, mode, message):
        code, summary = self.run_supervisor(mode)
        self.assertEqual(code, 1)
        self.assertFalse(summary["valid"])
        self.assertGreater(summary["error_count"], 0)
        self.assertTrue(any(message in error for error in summary["errors"]), summary["errors"])
        self.assertIsNone(summary.get("aggregation", {}).get("aggregate_evals_per_second"))
        return summary

    def test_validate_success_retains_raw_logs_and_completed_work(self):
        code, summary = self.run_supervisor()
        self.assertEqual(code, 0, summary["errors"])
        self.assertTrue(summary["valid"])
        self.assertFalse(summary["clean_baseline"])
        self.assertEqual(summary["isolation_state"], "unverified")
        self.assertEqual(summary["measurement_context"], "development_contended")
        self.assertFalse(summary["campaign_frozen"])
        self.assertFalse(summary["optimization_claim"])
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(len(summary["trials"]), 9)
        self.assertEqual(summary["completed_work"]["reported_including_invalid_and_diagnostic"]
                         ["completed_evaluations"], 432)
        for trial in summary["trials"]:
            path = Path(trial["directory"])
            self.assertTrue(trial["valid"])
            self.assertIn("raw child stdout", (path / "process/stdout.log").read_text())
            self.assertIn("raw child stderr", (path / "process/stderr.log").read_text())
            self.assertEqual(Path(trial["process"]["cwd"]), path / "cwd")
            self.assertTrue(Path(trial["process"]["command"][3]).is_absolute())
        self.assertEqual(len({trial["process"]["cwd"] for trial in summary["trials"]}), 9)

    def test_default_build_uses_cargos_redirected_executable_not_a_stale_default(self):
        checkout = self.root / "checkout"
        stale = checkout / "target/release/examples/evolution-bench"
        stale.parent.mkdir(parents=True)
        stale.write_text("#!/bin/sh\nexit 91\n")
        stale.chmod(0o755)
        redirected = self.root / "redirected-target/release/examples/evolution-bench"
        redirected.parent.mkdir(parents=True)
        redirected.write_bytes(self.executable.read_bytes())
        redirected.chmod(0o755)
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        cargo = fake_bin / "cargo"
        cargo.write_text("#!/usr/bin/env python3\nimport json, os\n"
            "print(json.dumps({'reason':'compiler-artifact','target':"
            "{'name':'evolution-bench','kind':['example']},'profile':{'opt_level':'3'},"
            "'executable':os.environ['CARGO_TARGET_DIR']+'/release/examples/evolution-bench'}))\n")
        cargo.chmod(0o755)
        argv = ["validate", "--output", str(self.output), "--manifest", str(self.manifest),
                "--lock", str(self.lock), "--timeout", "2"]
        with mock.patch.object(bench, "REPO", checkout), mock.patch.dict(os.environ,
                {"PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                 "CARGO_TARGET_DIR": str(self.root / "redirected-target")}):
            with contextlib.redirect_stdout(io.StringIO()):
                code = bench.main(argv)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["binary"]["source"], str(redirected))
        self.assertEqual(summary["binary"]["build"], "release_locked")
        self.assertEqual(summary["binary"]["sha256"], hashlib.sha256(redirected.read_bytes()).hexdigest())

    def test_a_successful_build_without_the_requested_artifact_is_rejected(self):
        path = self.root / "cargo.jsonl"
        path.write_text(json.dumps({"reason": "build-finished", "success": True}) + "\n")
        with self.assertRaisesRegex(bench.BenchError, "exactly one"):
            bench.cargo_artifact(path)

    def test_baseline_pairs_are_repetitions_of_one_search_seed(self):
        code, summary = self.run_supervisor(command="baseline", extra=("--pairs", "2"))
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(len(summary["trials"]), 18)
        self.assertEqual(summary["measurement_context"], "baseline_contended")
        self.assertEqual(summary["contamination"]["backend_evolving_observations"], 36)
        self.assertIsNone(summary["clean_baseline"])
        self.assertEqual(summary["isolation_state"], "unverified")
        self.assertFalse(summary["seed_schedule"]["repetitions_are_independent_search_seeds"])
        for case in summary["cases"]:
            self.assertEqual(case["aa_ratios"]["raw"], [1, 1])
            self.assertEqual(case["evals_per_second"]["n"], 4)
            self.assertEqual(case["evals_per_second"]["cv"], 0)
            self.assertEqual(case["comparisons"][0]["order"], ["A", "B", "diagnostic"])
        projects = summary["aggregation"]["projects"]
        self.assertEqual([entry["weight"] for entry in projects], [1 / 3] * 3)

    def test_subprocess_timeout_retains_partial_output_and_only_kills_owned_child(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        try:
            code, summary = self.run_supervisor("timeout", extra=("--timeout", "0.1"))
            self.assertEqual(code, 1)
            self.assertIsNone(unrelated.poll())
            self.assertEqual(len(summary["trials"]), 9)
            for trial in summary["trials"]:
                self.assertEqual(trial["process"]["status"], "timed_out")
                self.assertEqual(trial["process"]["returncode"], -signal.SIGKILL)
                self.assertTrue((Path(trial["directory"]) / "output/partial.txt").exists())
                self.assertIn("raw child stdout", (Path(trial["directory"]) / "process/stdout.log").read_text())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_subprocess_failure_retains_exit_stderr_and_partial_files(self):
        summary = self.assert_failure("failure", "exit 3")
        trial = summary["trials"][0]
        self.assertEqual(trial["process"]["returncode"], 3)
        self.assertIn("raw child stderr", (Path(trial["directory"]) / "process/stderr.log").read_text())
        self.assertTrue((Path(trial["directory"]) / "output/partial.txt").exists())

    def test_native_signal_is_failure_even_without_result(self):
        summary = self.assert_failure("signal", "exit -15")
        self.assertEqual(summary["trials"][0]["process"]["returncode"], -signal.SIGTERM)

    def test_missing_result_is_rejected(self):
        self.assert_failure("missing", "missing result.json")

    def test_unparseable_result_is_rejected_and_raw_bytes_survive(self):
        summary = self.assert_failure("unparseable", "invalid JSON")
        path = Path(summary["trials"][0]["directory"]) / "output/result.json"
        self.assertEqual(path.read_text(), "not json")

    def test_wrong_top_level_is_rejected(self):
        self.assert_failure("bad_top_level", "result must be a nonempty object")

    def test_duplicate_keys_are_rejected(self):
        self.assert_failure("duplicate_json", "duplicate JSON key")

    def test_nonfinite_metrics_are_rejected(self):
        self.assert_failure("nan", "non-finite")

    def test_wrong_completed_work_is_rejected(self):
        self.assert_failure("wrong_completed", "wrong completed work: completed_evaluations")

    def test_malformed_counters_preserve_summary_and_raw_result(self):
        summary = self.assert_failure("malformed_counters", "result.counters must be a nonempty object")
        self.assertEqual(summary["trials"][0]["result"]["counters"], ["wrong shape"])

    def test_wrong_submitted_work_is_rejected(self):
        self.assert_failure("wrong_submitted", "wrong completed work: submitted_batches")

    def test_wrong_work_even_when_submitted_equals_completed_is_rejected(self):
        self.assert_failure("wrong_iterations", "wrong completed work: submitted_iterations")

    def test_unrequested_case_is_rejected(self):
        self.assert_failure("wrong_case", "result case_id differs")

    def test_manifest_identity_mismatch_is_rejected(self):
        self.assert_failure("wrong_manifest", "fixture manifest hash mismatch")

    def test_throughput_must_use_completed_work_inside_elapsed(self):
        self.assert_failure("wrong_rate", "throughput does not match")

    def test_same_seed_replay_mismatch_is_rejected(self):
        self.assert_failure("replay_mismatch", "replay/quality mismatch: final_state_sha256")

    def test_quality_mismatch_is_rejected_even_with_equal_state_hashes(self):
        self.assert_failure("quality_mismatch", "quality/hash mismatch")

    def test_independent_quality_replay_mismatch_is_rejected_with_equal_state_hashes(self):
        self.assert_failure("quality_replay", "replay/quality mismatch: final_quality")

    def test_saved_canonical_state_hash_must_match_result(self):
        self.assert_failure("state_hash_mismatch", "final_state_sha256 differs from saved canonical state")

    def test_unchanged_state_cannot_claim_completed_evolution_work(self):
        self.assert_failure("no_evolution", "no evolution state changed")

    def test_profiled_records_cannot_be_used_as_reference_timing_samples(self):
        self.assert_failure("profiling_score", "profiling records cannot enter")

    def test_missing_independent_quality_metrics_are_rejected(self):
        self.assert_failure("missing_quality", "final_quality needs one numerical check")

    def test_parent_fitness_cannot_regress_against_initial_quality(self):
        self.assert_failure("regression", "final parent fitness regressed")

    def test_unexplained_pixel_is_rejected(self):
        self.assert_failure("unexplained_pixel", "pixel audit has unexplained pixels")

    def test_missing_pixel_audit_is_rejected(self):
        self.assert_failure("missing_audit", "pixel_audit must be a nonempty object")

    def test_audited_edge_differences_retain_nominal_error_metrics(self):
        code, summary = self.run_supervisor("audited_edge")
        self.assertEqual(code, 0, summary["errors"])
        quality = summary["trials"][0]["result"]["final_quality"][0]
        self.assertEqual(quality["max_channel_delta"], 9)
        self.assertEqual(quality["total_channel_delta"], 48)

    def test_nonzero_exit_rejects_even_a_complete_valid_result(self):
        summary = self.assert_failure("completed_nonzero", "exit 5")
        self.assertEqual(summary["trials"][0]["result"]["status"], "completed")

    def test_diagnostic_replay_must_match_normal_trajectory(self):
        self.assert_failure("diagnostic_mismatch", "replay/quality mismatch: final_state_sha256")

    def test_intermediate_tile_overflow_is_rejected(self):
        self.assert_failure("overflow", "validation_max_tile_occupancy")

    def test_initial_similarity_must_be_strictly_above_97(self):
        self.assert_failure("immature", "initial_similarity_percent")

    def test_missing_complexity_metrics_are_rejected(self):
        self.assert_failure("missing_metrics", "initial_complexity")

    def test_missing_error_count_is_rejected(self):
        self.assert_failure("missing_errors", "error_count")

    def test_runner_reported_numerical_failure_is_rejected_and_retained(self):
        summary = self.assert_failure("reported_error", "trial did not complete validly")
        self.assertEqual(summary["trials"][0]["result"]["errors"], ["independent numerical mismatch"])

    def test_fixture_integrity_is_checked_before_subprocess_even_if_child_would_ignore_it(self):
        (self.manifest.parent / "ff/drawing.json").write_text("tampered input")
        code, summary = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertIn("fixture integrity mismatch", summary["errors"][0])
        self.assertFalse((self.output / "trials").exists())

    def test_missing_frozen_file_is_rejected_before_any_trial(self):
        (self.manifest.parent / "mona/reference.rgba").unlink()
        code, summary = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertIn("missing frozen file", summary["errors"][0])

    def test_mutating_snapshot_is_rejected_and_original_corpus_is_unchanged(self):
        original = (self.manifest.parent / "ff/drawing.json").read_bytes()
        summary = self.assert_failure("fixture_mutation", "fixture integrity changed during trial")
        self.assertEqual((self.manifest.parent / "ff/drawing.json").read_bytes(), original)
        self.assertFalse(summary["trials"][0]["valid"])

    def test_cache_seed_is_copied_for_every_process_and_caller_cache_is_unchanged(self):
        cache = self.root / "seed"
        cache.mkdir()
        (cache / "pipeline_cache_test.bin").write_bytes(b"fixed cache seed")
        code, summary = self.run_supervisor("cache_check", extra=("--cache-seed", str(cache)))
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual((cache / "pipeline_cache_test.bin").read_bytes(), b"fixed cache seed")
        for trial in summary["trials"]:
            self.assertEqual((Path(trial["process"]["cwd"]) / "pipeline_cache_test.bin").read_bytes(),
                             b"mutated child cache")

    def test_clean_output_directory_is_required(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("existing")
        code, summary = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertEqual((self.output / "keep.txt").read_text(), "existing")

    def test_exclusive_lock_refuses_overlapping_invocations(self):
        with bench.benchmark_lock(self.lock):
            code, summary = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertFalse(self.output.exists())

    def test_freeze_has_one_command_and_refuses_existing_destination(self):
        destination = self.root / "frozen"
        code, summary = self.run_supervisor(command="freeze", extra=(
            "--projects-dir", str(self.manifest.parent), "--fixtures", str(destination)))
        self.assertEqual(code, 0, summary["errors"])
        self.assertTrue((destination / "manifest.json").exists())
        self.assertNotIn("trials", summary)
        first_manifest = (destination / "manifest.json").read_bytes()
        self.output = self.root / "second-evidence"
        code, summary = self.run_supervisor(command="freeze", extra=(
            "--projects-dir", str(self.manifest.parent), "--fixtures", str(destination)))
        self.assertEqual(code, 1)
        self.assertIn("already exists", summary["errors"][0])
        self.assertEqual((destination / "manifest.json").read_bytes(), first_manifest)

    def test_bounds_and_help(self):
        for value in ("0", "4097", "-1"):
            with self.subTest(batches=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    bench.parse_args(["validate", "--batches", value])
                self.assertEqual(exc.exception.code, 2)
        for flag, value in (("--pairs", "21"), ("--timeout", "nan"), ("--timeout", "inf"),
                            ("--warmup", "17"), ("--seed", str(2**64))):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    bench.parse_args(["baseline", flag, value])
                self.assertEqual(exc.exception.code, 2)
        for command in ("freeze", "validate", "baseline"):
            with contextlib.redirect_stdout(io.StringIO()) as help_text:
                with self.assertRaises(SystemExit) as exc:
                    bench.parse_args([command, "--help"])
            self.assertEqual(exc.exception.code, 0)
            self.assertIn("--binary", help_text.getvalue())

    def test_relative_paths_are_absolute_before_child_cwd_changes(self):
        args = bench.parse_args(["validate", "--manifest", "relative/manifest.json",
                                 "--binary", "relative/bench", "--output", "relative/output"])
        self.assertTrue(args.manifest.is_absolute())
        self.assertTrue(args.binary.is_absolute())
        self.assertTrue(args.output.is_absolute())

    def test_output_cannot_be_inside_immutable_fixture_directory(self):
        self.output = self.manifest.parent / "accidental-output"
        code, summary = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertFalse(self.output.exists())

    def test_output_cannot_contain_the_input_corpus(self):
        self.output = self.root
        code, summary = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertTrue(self.manifest.exists())

    def test_existing_binary_is_archived_without_a_build(self):
        with mock.patch.object(bench, "capture_command", side_effect=AssertionError("unexpected command")):
            code, summary = self.run_supervisor()
        self.assertEqual(code, 0, summary["errors"])
        archived = self.output / "evolution-bench"
        self.assertEqual(archived.read_bytes(), self.executable.read_bytes())
        self.assertEqual(summary["binary"]["sha256"], hashlib.sha256(archived.read_bytes()).hexdigest())
        self.assertEqual(summary["binary"]["build"], "supplied_binary_unverified_build")

    def test_default_build_happens_once_before_all_trials(self):
        fake_repo = self.root / "repo"
        target = fake_repo / "target/release/examples/evolution-bench"
        target.parent.mkdir(parents=True)
        real_process = bench.run_process
        builds = []

        def run(command, cwd, evidence, timeout, env=None, artifact_limit=bench.MAX_FILE_BYTES):
            if command[0] == "cargo":
                builds.append(command)
                target.write_bytes(self.executable.read_bytes())
                target.chmod(0o755)
                evidence.mkdir()
                (evidence / "stdout.log").write_text(json.dumps({
                    "reason": "compiler-artifact", "target": {
                        "name": "evolution-bench", "kind": ["example"]},
                    "executable": str(target)}) + "\n")
                return {"status": "completed", "returncode": 0}
            return real_process(command, cwd, evidence, timeout, env, artifact_limit)

        argv = ["validate", "--manifest", str(self.manifest), "--output", str(self.output),
                "--lock", str(self.lock), "--timeout", "2"]
        with mock.patch.object(bench, "REPO", fake_repo), mock.patch.object(bench, "run_process", side_effect=run):
            with contextlib.redirect_stdout(io.StringIO()):
                code = bench.main(argv)
        self.assertEqual(code, 0)
        self.assertEqual(builds, [["cargo", "build", "--release", "--locked", "--example",
                                  "evolution-bench", "--message-format=json-render-diagnostics"]])

    def test_debug_symbol_release_binary_can_exceed_json_and_log_limit(self):
        large = self.root / "large-release-example"
        with large.open("wb") as stream:
            stream.write(b"executable with debug symbols")
            stream.truncate(bench.MAX_FILE_BYTES + 1)
        large.chmod(0o755)
        self.output.mkdir()
        args = bench.parse_args(["validate", "--binary", str(large)])
        archived, identity = bench.prepare_binary(args, self.output)
        self.assertEqual(archived.stat().st_size, bench.MAX_FILE_BYTES + 1)
        self.assertEqual(identity["sha256"], bench.sha_file(large, bench.MAX_BINARY_BYTES))

    def test_fixture_symlinks_cannot_escape_integrity_snapshot(self):
        target = self.manifest.parent / "ff/drawing.json"
        target.unlink()
        external = self.root / "external.json"
        external.write_text('{"polygons":[],"fitness":98.25}')
        target.symlink_to(external)
        code, summary = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertIn("missing frozen file", summary["errors"][0])

    def test_project_aggregation_does_not_weight_by_workload_count(self):
        def case(name, rate):
            return {"case_id": name, "project": name.split("-")[0], "valid": True,
                    "error_count": 0, "evals_per_second": {"median": rate, "raw": [rate]},
                    "aa_ratios": {"raw": [1]}}
        aggregation = bench.project_statistics([case("ff-one", 100), case("ff-two", 100),
                                                case("mona-one", 1000), case("darwin-one", 10)])
        self.assertAlmostEqual(aggregation["aggregate_evals_per_second"], 100)
        self.assertEqual(aggregation["projects"][0]["case_weight_within_project"], 0.5)


if __name__ == "__main__":
    unittest.main()
