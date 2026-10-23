"""CPU-only CLI proofs. Children are synthetic Python workers, never GPU/Cargo.

Run: python3 -B -W error -m unittest discover -s scripts -p test_compare_evolution.py
Python/unittest allocation and dispatch use the bounded deviations documented in
compare-evolution.py. Temporary corpora and children have scoped owners/cleanup.
"""

import contextlib
import copy
import importlib.util
import io
import os
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
            raise RuntimeError("missing local test module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


compare = local_module("evolution_comparison", "compare-evolution.py")
support = local_module("evolution_comparison_test_support", "test_evolution_bench.py")
bench = compare.bench
INITIAL_ARTIFACT = '''
import copy, struct
state["parents"][0]["rng_state"] = int(arg("--seed")) * 1000 + batches
initial = copy.deepcopy(state)
for parent in initial["parents"]:
    parent["fitness_bits"] = struct.unpack("<I", struct.pack("<f", 98.25))[0]
    parent["rng_state"] = int(arg("--seed"))
initial["counters"] = {key: 0 for key in counters}
if MODE == "initial_error_mismatch":
    initial["parent_total_errors"][0] += 1
if MODE == "initial_fitness_mismatch":
    initial["parents"][0]["fitness_bits"] = struct.unpack("<I", struct.pack("<f", 98.0))[0]
fields = ("parents", "offspring", "parent_total_errors", "parent_framebuffers",
          "offspring_errors", "chain_update_flags", "fitness_packed")
record["initial_state_sha256"] = hashlib.sha256(json.dumps(
    [initial[key] for key in fields], separators=(",", ":")).encode()).hexdigest()
if MODE == "corrupt_initial":
    initial["parents"][0]["rng_state"] += 1
if MODE != "missing_initial":
    (output / "initial-state.json").write_text(json.dumps(initial))
if MODE == "repeat_mismatch" and "-02-" in label:
    state["parents"][0]["rng_state"] += 99
if MODE == "diagnostic_score" and record["diagnostic"]:
    record["evals_per_second"] = 123
if MODE == "diagnostic_budget" and record["diagnostic"]:
    record["batches"] += 1
if MODE == "oversized_number":
    record["elapsed_seconds"] = 10 ** 309
'''
CHILD = support.FAKE_EXECUTABLE.replace('label.endswith("B")', 'label.endswith("candidate")')
CHILD = CHILD.replace('record = {"protocol_version":', '''
assert Path("pipeline_cache_test.bin").read_bytes() == b"fixed cache seed"
Path("pipeline_cache_test.bin").write_bytes(b"private modified cache")
started = time.monotonic()
time.sleep(0.015 * DELAY_FACTOR)
measured = time.monotonic() - started
record = {"protocol_version":''')
CHILD = CHILD.replace('"elapsed_seconds": 0.25, "drain_seconds": 0.02,',
                      '"elapsed_seconds": measured, "drain_seconds": 0.0001,')
CHILD = CHILD.replace('else evals / 0.25,', 'else evals / measured,')
CHILD = CHILD.replace('if (MODE == "replay_mismatch"', INITIAL_ARTIFACT + '\nif (MODE == "replay_mismatch"')
CHILD = CHILD.replace('MODE = "ok"', 'MODE = "ok"\nDELAY_FACTOR = 1.0')
CHILD = CHILD.replace('output.mkdir()', 'output.mkdir()\n(output / "worker.pid").write_text(str(os.getpid()))')


class ComparisonBehaviorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.capture = self.root / "baseline"
        self.capture.mkdir()
        self.manifest = support.fixture_corpus(self.capture / "fixtures")
        cache = self.capture / "cache-seed"
        cache.mkdir()
        (cache / "pipeline_cache_test.bin").write_bytes(b"fixed cache seed")
        self.original = self.capture / "evolution-bench"
        self.binary(self.original)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.binary(self.reference)
        self.binary(self.candidate, delay=0.3)
        self.output = self.root / "comparison"
        self.baseline = self.make_baseline()
        self.before = bench.inventory(self.capture)
        for target, name, replacement in ((compare, "LOCK", self.root / "benchmark.lock"),
                                           (bench, "telemetry", support.telemetry_sample)):
            patch = mock.patch.object(target, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def binary(self, path, mode="ok", delay=1.0):
        path.write_text(CHILD.replace('MODE = "ok"', f'MODE = "{mode}"').replace(
            "DELAY_FACTOR = 1.0", f"DELAY_FACTOR = {delay}"))
        path.chmod(0o755)

    def make_baseline(self):
        trials = []
        for case in bench.PROJECTS:
            directory = self.capture / f"control-{case}"
            cwd = self.root / f"baseline-cwd-{case}"
            bench.copy_inventory(self.capture / "cache-seed", cwd, bench.inventory(self.capture / "cache-seed"), writable=True)
            command = [str(self.original), "trial", "--manifest", str(self.manifest), "--case", case,
                       "--seed", "7", "--batches", "64", "--warmup", "2", "--diagnostic", "false",
                       "--output", str(directory)]
            result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=3, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            normal = bench.read_json(directory / "result.json")
            for index in range(5):
                for label in ("A", "B", "diagnostic"):
                    record = copy.deepcopy(normal)
                    diagnostic = label == "diagnostic"
                    record.update(diagnostic=diagnostic, validation_max_tile_occupancy=200 if diagnostic else None)
                    record["elapsed_seconds"] = 0.25
                    rate = record["counters"]["completed_evaluations"] / 0.25
                    if case == "ff" and index == 0 and label == "A":
                        rate *= 1 - compare.NOISE_BAND
                        record["elapsed_seconds"] = record["counters"]["completed_evaluations"] / rate
                    record["evals_per_second"] = None if diagnostic else rate
                    trials.append({"case_id": case, "label": label, "diagnostic": diagnostic,
                                   "valid": True, "result": record})
        summary = {"command": "baseline", "valid": True, "errors": [], "error_count": 0,
                   "binary": {"archived": str(self.original), "sha256": bench.sha_file(self.original)},
                   "manifest": str(self.manifest), "manifest_sha256": bench.sha_file(self.manifest),
                   "fixture_inventory": bench.inventory(self.manifest.parent),
                   "cache_policy": {"files": bench.inventory(self.capture / "cache-seed")},
                   "arguments": {"seed": 7, "batches": 64, "warmup": 2}, "trials": trials}
        path = self.capture / "summary.json"
        bench.write_json(path, summary)
        return path

    def run_cli(self, *extra):
        argv = ["--baseline", str(self.baseline), "--reference", str(self.reference),
                "--candidate", str(self.candidate), "--output", str(self.output), "--timeout", "3", *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = compare.main(argv)
        path = self.output / "summary.json"
        return code, bench.read_json(path) if path.is_file() else None

    def assert_invalid(self, mode, message, *extra):
        self.binary(self.candidate, mode, delay=0.3)
        code, summary = self.run_cli("--pairs", "1", *extra)
        self.assertEqual(code, 1, summary)
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["verdict"], "REJECT")
        self.assertIsNone(summary["statistics"])
        self.assertEqual(summary["candidate_screen"], "failed")
        self.assertTrue(any(message in error for error in summary["errors"]), summary["errors"])
        self.assertEqual(bench.inventory(self.capture), self.before)
        return summary

    def test_fast_candidate_passes_screen_with_no_adoption_or_clean_claim(self):
        code, summary = self.run_cli()
        self.assertEqual(code, 0, summary["errors"])
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["candidate_screen"], "passed")
        self.assertEqual(summary["verdict"], "INCONCLUSIVE")
        self.assertIsNone(summary["clean_baseline"])
        self.assertEqual(summary["isolation_state"], "unverified")
        self.assertFalse(summary["optimization_claim"])
        self.assertFalse(summary["primary_score_eligible"])
        self.assertIn("live_preview_acceptance", summary["gates_missing"])
        self.assertIn("held_out_mature_states", summary["gates_missing"])
        self.assertEqual(summary["schedule"], {"seed": 7, "batches": 64, "warmup": 2, "pairs": 5})
        self.assertEqual(len(summary["trials"]), 33)
        self.assertEqual(summary["completed_work"]["reported_including_invalid_and_diagnostic"]["completed_evaluations"], 33 * 1536)
        self.assertEqual(bench.inventory(self.capture), self.before)
        self.assertGreater(summary["statistics"]["aggregate"]["ratio"], 1.5)
        self.assertEqual(summary["policy"]["bootstrap_seed"], 20261023)
        self.assertAlmostEqual(summary["baseline_noise"]["observed_max_absolute_aa_deviation"], compare.NOISE_BAND)
        for identity in summary["binaries"].values():
            self.assertEqual(identity["build"], "supplied_binary_unverified_build")
            self.assertEqual(identity["source_identity"], "supplied/unverified")
            self.assertEqual(bench.sha_file(Path(identity["archived"])), identity["sha256"])
        self.assertNotEqual(summary["binaries"]["reference"]["sha256"], summary["binaries"]["candidate"]["sha256"])
        for trial in summary["trials"]:
            path = Path(trial["directory"])
            self.assertTrue((path / "output/initial-state.json").is_file())
            self.assertIn("raw child stdout", (path / "process/stdout.log").read_text())
            self.assertIn("raw child stderr", (path / "process/stderr.log").read_text())
            self.assertTrue(trial["before"]["backend"]["evolving"])
        self.assertEqual(len({t["process"]["cwd"] for t in summary["trials"]}), 33)

    def test_slow_candidate_rejects_without_discarding_unfavorable_pairs(self):
        self.binary(self.candidate, delay=3)
        code, summary = self.run_cli()
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["candidate_screen"], "failed")
        self.assertEqual(summary["verdict"], "REJECT")
        self.assertEqual(len(summary["pairs"]), 15)
        self.assertTrue(all(p["valid"] for p in summary["pairs"]))
        self.assertLess(summary["statistics"]["aggregate"]["ratio"], 0.75)

    def test_order_alternates_ab_ba_and_diagnostic_follows_each_case(self):
        code, summary = self.run_cli()
        self.assertEqual(code, 0, summary["errors"])
        for case in bench.PROJECTS:
            pairs = [p for p in summary["pairs"] if p["case_id"] == case]
            self.assertEqual([p["order"] for p in pairs], [["reference", "candidate"], ["candidate", "reference"],
                             ["reference", "candidate"], ["candidate", "reference"], ["reference", "candidate"]])
            trials = [t for t in summary["trials"] if t["case_id"] == case]
            self.assertFalse(any(t["diagnostic"] for t in trials[:-1]))
            self.assertTrue(trials[-1]["diagnostic"])
            self.assertEqual(trials[-1]["side"], "candidate")
            self.assertIsNone(trials[-1]["result"]["evals_per_second"])
            self.assertFalse(trials[-1]["timing_score_eligible"])

    def test_short_invocation_is_inconclusive_even_when_ratios_are_large(self):
        code, summary = self.run_cli("--pairs", "1", "--batches", "1")
        self.assertEqual(code, 0, summary["errors"])
        self.assertEqual(summary["verdict"], "INCONCLUSIVE")
        self.assertIn("minimum_pairs", summary["gates_missing"])
        self.assertFalse(summary["baseline_final_replay_required"])
        self.assertEqual(len(summary["trials"]), 9)

    def test_different_seed_and_budget_validate_initial_quality_and_full_pairs(self):
        code, summary = self.run_cli("--pairs", "1", "--seed", "11", "--batches", "3")
        self.assertEqual(code, 0, summary["errors"])
        self.assertFalse(summary["baseline_final_replay_required"])
        self.assertIn("different seed/budget", summary["baseline_replay_scope"])
        self.assertEqual({t["result"]["seed"] for t in summary["trials"]}, {11})
        old = bench.read_json(self.baseline)["trials"][0]["result"]
        self.assertNotEqual(old["initial_state_sha256"], summary["trials"][0]["result"]["initial_state_sha256"])
        self.assertTrue(all(p["valid"] for p in summary["pairs"]))

    def test_replay_mismatch_invalidates_whole_score_and_retains_all_work(self):
        summary = self.assert_invalid("replay_mismatch", "final_state_sha256")
        self.assertEqual(len(summary["trials"]), 9)
        self.assertEqual(summary["completed_work"]["reported_including_invalid_and_diagnostic"]["completed_evaluations"], 9 * 1536)
        self.assertFalse(any(t["timing_score_eligible"] for t in summary["trials"]))

    def test_wrong_work_cannot_be_scored(self):
        self.assert_invalid("wrong_completed", "completed_evaluations")

    def test_oversized_json_integer_retains_summary_raw_record_and_counters(self):
        summary = self.assert_invalid("oversized_number", "too large")
        trial = next(t for t in summary["trials"] if t["side"] == "candidate")
        self.assertEqual(trial["result"]["elapsed_seconds"], 10 ** 309)
        self.assertEqual(trial["process"]["status"], "completed")
        self.assertIn("after", trial)
        self.assertTrue((Path(trial["directory"]) / "output/result.json").is_file())
        self.assertEqual(summary["completed_work"]["reported_including_invalid_and_diagnostic"]["completed_evaluations"], 9 * 1536)

    def test_different_seed_still_requires_exact_paired_replay(self):
        self.assert_invalid("replay_mismatch", "final_state_sha256", "--seed", "11", "--batches", "3")

    def test_saved_final_canonical_hash_is_checked(self):
        self.assert_invalid("state_hash_mismatch", "saved canonical state")

    def test_failed_worker_retains_exit_status_stderr_and_partial_output(self):
        summary = self.assert_invalid("failure", "subprocess failed")
        trial = next(t for t in summary["trials"] if t["side"] == "candidate")
        self.assertEqual(trial["process"]["returncode"], 3)
        self.assertTrue((Path(trial["directory"]) / "output/partial.txt").is_file())
        self.assertGreater(summary["completed_work"]["workers_without_reported_counters"], 0)

    def test_scalar_quality_mismatch_cannot_hide_behind_canonical_hash(self):
        self.assert_invalid("quality_replay", "final_quality")

    def test_diagnostic_must_replay_candidate_and_stay_within_capacity(self):
        for mode, message in (("diagnostic_mismatch", "final_state_sha256"),
                              ("overflow", "validation_max_tile_occupancy"),
                              ("diagnostic_score", "diagnostic evals_per_second must be null"),
                              ("diagnostic_budget", "batches differs from requested trial")):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                self.assert_invalid(mode, message)

    def test_profiling_candidate_is_rejected_from_primary_timing(self):
        self.assert_invalid("profiling_score", "profiling records cannot enter")

    def test_fixture_mutation_invalidates_clone_and_preserves_baseline(self):
        self.assert_invalid("fixture_mutation", "fixture integrity changed")

    def test_missing_or_corrupt_initial_artifacts_are_rejected(self):
        for mode, message in (("missing_initial", "initial-state.json"),
                              ("corrupt_initial", "initial_state_sha256"),
                              ("initial_error_mismatch", "saved initial errors"),
                              ("initial_fitness_mismatch", "saved initial fitness")):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                self.assert_invalid(mode, message)

    def test_reference_must_match_baseline_and_repeats(self):
        self.binary(self.reference, "repeat_mismatch")
        code, summary = self.run_cli("--pairs", "2")
        self.assertEqual(code, 1)
        self.assertTrue(any("final_state_sha256" in e for e in summary["errors"]))
        self.assertIsNone(summary["statistics"])

    def test_timeout_keeps_partial_artifacts_and_other_process_alive(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            summary = self.assert_invalid("timeout", "timed_out", "--timeout", "0.15")
            self.assertIsNone(unrelated.poll())
            self.assertGreater(summary["worker_counts"]["timed_out"], 0)
            trial = next(t for t in summary["trials"] if t["side"] == "candidate")
            self.assertTrue((Path(trial["directory"]) / "output/partial.txt").is_file())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_sigterm_cancels_owned_worker_keeps_other_process_and_restores_handler(self):
        self.binary(self.reference, "timeout")
        wrapper = self.root / "supervisor.py"
        wrapper.write_text("import importlib.util,signal,sys\nfrom pathlib import Path\n"
                           f"s=importlib.util.spec_from_file_location('cancel_support',{str(Path(__file__).resolve())!r})\n"
                           "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                           f"m.compare.LOCK=Path({str(self.root / 'benchmark.lock')!r})\n"
                           "m.bench.telemetry=m.support.telemetry_sample\n"
                           "previous=signal.getsignal(signal.SIGTERM)\n"
                           "code=m.compare.main(sys.argv[1:])\n"
                           "assert signal.getsignal(signal.SIGTERM)==previous\n"
                           "sys.exit(code)\n")
        command = [sys.executable, "-B", str(wrapper), "--baseline", str(self.baseline), "--reference", str(self.reference),
                   "--candidate", str(self.candidate), "--output", str(self.output), "--pairs", "1", "--timeout", "10"]
        pid_file = self.output / "trials/ff-01-reference/output/worker.pid"
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        worker_pid = None
        try:
            with ((self.root / "supervisor.log").open("wb") as log,
                  subprocess.Popen(command, stdout=log, stderr=log) as supervisor):
                for _ in range(200):
                    if pid_file.is_file() and pid_file.stat().st_size:
                        worker_pid = int(pid_file.read_text())
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(worker_pid, (self.root / "supervisor.log").read_text())
                supervisor.send_signal(signal.SIGTERM)
                self.assertEqual(supervisor.wait(timeout=5), 1)
            summary = bench.read_json(self.output / "summary.json")
            self.assertTrue(summary["cancelled"])
            self.assertEqual(summary["status"], "cancelled")
            self.assertFalse(summary["valid"])
            self.assertIsNone(summary["statistics"])
            self.assertFalse(Path(f"/proc/{worker_pid}").exists())
            self.assertIsNone(unrelated.poll())
            self.assertEqual(summary["trials"][0]["process"]["status"], "cancelled")
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)
            if worker_pid is not None and Path(f"/proc/{worker_pid}/cmdline").exists():
                # A failed regression must clean up only this unique fake worker.
                command_line = Path(f"/proc/{worker_pid}/cmdline").read_bytes()
                if str(self.output).encode() in command_line:
                    os.kill(worker_pid, signal.SIGKILL)

    def test_missing_supplied_binary_retains_failure_summary_without_build(self):
        self.candidate.unlink()
        code, summary = self.run_cli("--pairs", "1")
        self.assertEqual(code, 1)
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["trials"], [])
        self.assertTrue(any("not executable" in e for e in summary["errors"]))

    def test_existing_output_is_immutable(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        code, summary = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_binary_archive_enforces_512_mib_limit_before_measurements(self):
        with self.candidate.open("wb") as stream:
            stream.truncate(bench.MAX_BINARY_BYTES + 1)
        code, summary = self.run_cli("--pairs", "1")
        self.assertEqual(code, 1)
        self.assertEqual(summary["trials"], [])
        self.assertTrue(any("file exceeds" in error for error in summary["errors"]))

    def test_pair_cap_is_enforced_before_any_child(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_cli("--pairs", "11")
        self.assertFalse(self.output.exists())


class StatisticsBehaviorTests(unittest.TestCase):
    def summary(self, ratio):
        return {"errors": [], "trials": [], "schedule": {"pairs": 5},
                "pairs": [{"case_id": name, "pair": index + 1, "valid": True,
                           "ratio": ratio, "reference_rate": 100, "candidate_rate": 100 * ratio}
                          for name in bench.PROJECTS for index in range(5)]}

    def test_exact_three_percent_gain_fails_the_strict_threshold(self):
        summary = self.summary(1.03)
        compare.finish(summary)
        self.assertEqual(summary["statistics"]["aggregate"]["ratio"], 1.03)
        self.assertFalse(summary["screen_gates"]["practical_gain"])
        self.assertEqual(summary["candidate_screen"], "failed")
        self.assertEqual(summary["verdict"], "REJECT")

    def test_incomplete_pair_set_is_invalid_without_partial_statistics(self):
        summary = self.summary(1.1)
        summary["pairs"].pop()
        compare.finish(summary)
        self.assertFalse(summary["valid"])
        self.assertIsNone(summary["statistics"])
        self.assertTrue(any("incomplete comparison" in error for error in summary["errors"]))

    def test_constant_ratios_have_known_geometric_aggregate_and_intervals(self):
        ratios = {"ff": [1.21] * 5, "mona": [1.0] * 5, "darwin": [1.0] * 5}
        result = compare.statistics(ratios)
        expected = 1.21 ** (1 / 3)
        self.assertAlmostEqual(result["aggregate"]["ratio"], expected)
        for bound in result["aggregate"]["interval95"]:
            self.assertAlmostEqual(bound, expected)
        self.assertEqual(result["bootstrap_seed"], 20261023)
        self.assertEqual(result["resamples"], 10000)

    def test_intervals_resample_log_means_while_median_is_separate(self):
        ratios = {name: [1, 1, 1, 1, 2] for name in bench.PROJECTS}
        first = compare.statistics(ratios)
        self.assertEqual(first, compare.statistics(ratios))
        for case in first["cases"].values():
            self.assertEqual(case["paired_ratios"]["median"], 1)
            self.assertAlmostEqual(case["geomean_ratio"], 2 ** 0.2)
            self.assertEqual(case["interval95"][0], 1)
            self.assertAlmostEqual(case["interval95"][1], 2 ** 0.6)
        self.assertEqual(first["aggregate"]["interval95"][0], 1)
        self.assertAlmostEqual(first["aggregate"]["interval95"][1], 2 ** (6 / 15))

    def test_invalid_or_unbalanced_ratios_are_rejected(self):
        for ratios in ({"ff": [1]}, {name: [0] for name in bench.PROJECTS},
                       {"ff": [1, 1], "mona": [1], "darwin": [1]}):
            with self.subTest(ratios=ratios), self.assertRaises(bench.BenchError):
                compare.statistics(ratios)


if __name__ == "__main__":
    unittest.main()
