"""CPU-only qualification checks using a retained native producer record."""

import argparse
import copy
import importlib.util
import json
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("qualification_bench", Path(__file__).with_name("evolution-bench.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
FIXTURES = Path(__file__).parent / "fixtures" / "trial-qualification"


class QualificationContractTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((FIXTURES / "qualification.json").read_text())
        self.case = json.loads((FIXTURES / "case.json").read_text())
        self.args = argparse.Namespace(seed=7, batches=8, timeout=600)
        self.manifest_hash = "82dfe207ecd20ae964edf4528f1a90fe4176b21f4ac0911b305da1e1edf1c5aa"

    def qualify(self, record):
        return bench.validate_qualification(record, self.case, self.args, self.manifest_hash)

    def test_native_qualification_is_valid_but_cannot_enter_timing_score(self):
        self.assertIs(self.qualify(self.record), self.record)
        self.assertFalse(self.record["diagnostic"])
        self.assertIsNone(self.record["evals_per_second"])
        with self.assertRaisesRegex(bench.BenchError, "evals_per_second"):
            bench.validate_result(self.record, self.case, self.args, False, self.manifest_hash)

    def test_qualification_rejects_invalid_work_clocks_and_quality(self):
        mutations = [
            ("counters", "submitted_batches", 7),
            ("counters", "completed_batches", 7),
            ("counters", "submitted_iterations", 511),
            ("counters", "completed_iterations", 511),
            ("counters", "submitted_evaluations", 131071),
            ("counters", "completed_evaluations", 131071),
            (None, "elapsed_seconds", 0),
            (None, "elapsed_seconds", float("nan")),
            (None, "drain_seconds", 600),
            (None, "evals_per_second", 131072 / self.record["elapsed_seconds"]),
            (None, "correctness", False),
            (None, "diagnostic", True),
            (None, "timestamp_mode", "batch_envelope"),
            (None, "initial_state_sha256", self.record["final_state_sha256"]),
            (None, "final_quality", []),
            ("params", "chainCount", 5),
        ]
        for section, field, value in mutations:
            with self.subTest(section=section, field=field, value=value):
                record = copy.deepcopy(self.record)
                owner = record if section is None else record[section]
                owner[field] = value
                with self.assertRaises(bench.BenchError):
                    self.qualify(record)
        del self.record["evals_per_second"]
        with self.assertRaisesRegex(bench.BenchError, "evals_per_second"):
            self.qualify(self.record)

    def test_scored_record_requires_the_correct_finite_completed_work_rate(self):
        # A derived timing shape tests score policy without relabeling historical
        # evidence as an accepted performance observation.
        expected = 131072 / self.record["elapsed_seconds"]
        for rate in [expected, None, expected * 2, float("inf")]:
            with self.subTest(rate=rate):
                record = copy.deepcopy(self.record)
                record["evals_per_second"] = rate
                if rate == expected:
                    self.assertIs(bench.validate_result(
                        record, self.case, self.args, False, self.manifest_hash), record)
                else:
                    with self.assertRaises(bench.BenchError):
                        bench.validate_result(record, self.case, self.args, False, self.manifest_hash)


if __name__ == "__main__":
    unittest.main()
