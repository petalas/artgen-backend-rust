"""CPU-only proofs of the hill-climb driver. Children are fake Python executables.

Run: python3 -B -W error -m unittest discover -s scripts -p test_hillclimb.py
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("hillclimb", Path(__file__).with_name("hillclimb.py"))
hillclimb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hillclimb)

# Speaks the `serve` and `trial` protocols. Rate, final hash, correctness and serve
# support come from the executable's own name so each side can differ.
FAKE = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name           # e.g. rate-100-hash-a-serve-1-ok-1-gain-1
fields = dict(zip(name.split("-")[::2], name.split("-")[1::2]))

def record(flags):
    ok = fields["ok"] == "1"
    rec = {"case_id": flags["case"], "seed": int(flags["seed"]),
           "batches": int(flags["batches"]), "correctness": ok,
           "errors": [] if ok else ["pixel mismatch"],
           "evals_per_second": float(fields["rate"]) if ok else None,
           "final_state_sha256": fields["hash"] + flags["case"],
           "timestamp_samples": 4, "sampled_gpu_stage_totals_ns": [4e3, 4e3, 4e3, 4e3],
           "elapsed_seconds": 0.01, "params": {"gpuBatchIters": 4},
           "initial_similarity_percent": 97.0,
           "final_fitness_percent": [97.0, 97.0 + float(fields["gain"]) / 100]}
    out = Path(flags["output"]); out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(rec))
    return rec

def flags(words):
    return dict(zip([w[2:] for w in words[::2]], words[1::2]))

if sys.argv[1] == "serve":
    if fields["serve"] != "1":
        sys.exit("unknown command serve")
    for line in sys.stdin:
        print("engine log line", flush=True)
        print("@result " + json.dumps(record(flags(line.split()))), flush=True)
elif sys.argv[1] == "trial":
    rec = record(flags(sys.argv[2:]))
    print(json.dumps(rec))
    sys.exit(0 if rec["correctness"] else 1)
'''


LIFECYCLE_FAKE = """#!/usr/bin/env python3
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output") + 1]); out.mkdir(parents=True)
(out / "comparison.json").write_text("CONTENT")
"""


class HillclimbCli(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        manifest = self.root / "manifest.json"
        manifest.write_text("{}")
        quiet = mock.patch.object(hillclimb, "wait_for_quiet", return_value=True)
        quiet.start()
        self.addCleanup(quiet.stop)
        for name, value in (("ROOT", self.root / "hc"), ("MANIFEST", manifest),
                            ("LOCK", self.root / "lock"), ("LOG", self.root / "log.tsv")):
            patcher = mock.patch.object(hillclimb, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake(self, rate=100, digest="a", serve=1, ok=1, gain=1, lifecycle="same"):
        folder = self.root / f"bin-{lifecycle}"
        path = folder / f"rate-{rate}-hash-{digest}-serve-{serve}-ok-{ok}-gain-{gain}"
        folder.mkdir(exist_ok=True)
        path.write_text(FAKE)
        path.chmod(0o755)
        checker = folder / "check-gpu-lifecycle"
        checker.write_text(LIFECYCLE_FAKE.replace("CONTENT", lifecycle))
        checker.chmod(0o755)
        return str(path)

    def run_cli(self, control, candidate, *extra):
        out, err = io.StringIO(), io.StringIO()
        argv = ["--control", control, "--candidate", candidate, "--cases", "ff,mona",
                "--pairs", "3", "--batches", "2", "--timeout", "20", *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = hillclimb.main(argv)
        return code, out.getvalue(), err.getvalue()

    def log_rows(self):
        lines = hillclimb.LOG.read_text().splitlines()
        header = lines[0].split("\t")
        return [dict(zip(header, line.split("\t"))) for line in lines[1:]]

    def test_identical_sides_are_noise_and_logged(self):
        code, out, _ = self.run_cli(self.fake(), self.fake())
        self.assertEqual(code, 0, out)
        self.assertIn("NOISE", out)
        [row] = self.log_rows()
        self.assertEqual((row["verdict"], row["replay"], row["ratio"]), ("NOISE", "same", "1.00000"))

    def test_clear_speedup_is_faster_and_counts_every_trial(self):
        code, out, _ = self.run_cli(self.fake(rate=100), self.fake(rate=110))
        self.assertEqual(code, 0, out)
        [row] = self.log_rows()
        self.assertEqual(row["verdict"], "FASTER")
        self.assertAlmostEqual(float(row["ratio"]), 1.1, places=6)
        evidence = Path(out.rsplit("evidence ", 1)[1].strip())
        rows = json.loads((hillclimb.REPO / evidence / "summary.json").read_text())["rows"]
        self.assertEqual(len(rows), 2 * 3 * 2)  # primes are not scored rows

    def test_several_candidates_share_one_control_and_log_a_row_each(self):
        code, out, _ = self.run_cli(self.fake(rate=100), self.fake(rate=110),
                                    "--candidate", self.fake(rate=90),
                                    "--candidate", self.fake(digest="b"))
        self.assertEqual(code, 1, out)
        self.assertEqual([r["verdict"] for r in self.log_rows()],
                         ["FASTER", "SLOWER", "REPLAY-MISMATCH"])
        evidence = Path(out.rsplit("evidence ", 1)[1].strip())
        rows = json.loads((hillclimb.REPO / evidence / "summary.json").read_text())["rows"]
        self.assertEqual(len(rows), 4 * 3 * 2)

    def test_build_only_prints_executables_without_trials(self):
        code, out, _ = self.run_cli(self.fake(), self.fake(rate=110), "--build-only")
        self.assertEqual(code, 0)
        self.assertEqual([line.split("\t")[0] for line in out.splitlines()], ["control", "candidate"])
        self.assertFalse(hillclimb.LOG.exists())

    def test_busy_host_is_recorded_and_warned(self):
        with mock.patch.object(hillclimb, "foreign_cores", return_value=9.0):
            code, out, _ = self.run_cli(self.fake(), self.fake())
        self.assertEqual(code, 0)
        self.assertIn("WARNING: other processes used up to 9.0 CPU cores during 12 of 12", out)
        self.assertIn("re-timed 36 trials", out)
        self.assertEqual(self.log_rows()[0]["host_load"], "9.0")

    def test_pair_disturbed_once_is_retimed_and_scored_clean(self):
        readings = iter([5.0] + [0.0] * 100)
        with mock.patch.object(hillclimb, "foreign_cores", lambda *_: next(readings)):
            code, out, _ = self.run_cli(self.fake(), self.fake(rate=110))
        self.assertEqual(code, 0, out)
        self.assertIn("re-timed 2 trials", out)
        self.assertNotIn("WARNING", out)
        self.assertEqual(self.log_rows()[0]["host_load"], "0.0")

    def test_confirm_runs_both_horizons_and_adopts_only_without_regressions(self):
        code, out, _ = self.run_cli(self.fake(rate=100), self.fake(rate=110),
                                    "--candidate", self.fake(rate=110, digest="b"), "--confirm")
        self.assertEqual(code, 1, out)
        rows = self.log_rows()
        self.assertEqual([(r["batches"], r["pairs"]) for r in rows],
                         [("64", "5"), ("64", "5"), ("512", "3"), ("512", "3")])
        self.assertIn("CONFIRM ADOPT: rate-110-hash-a", out)
        self.assertIn("CONFIRM REJECT: rate-110-hash-b", out)

    def test_confirm_rejects_a_lifecycle_difference(self):
        code, out, _ = self.run_cli(self.fake(), self.fake(rate=110, lifecycle="other"), "--confirm")
        self.assertEqual(code, 0, out)
        self.assertIn("lifecycle differs", out)
        self.assertIn("CONFIRM REJECT", out)
        self.assertEqual({r["lifecycle"] for r in self.log_rows()}, {"differs"})

    def test_unappliable_revert_is_skipped(self):
        code, out, err = self.run_cli(self.fake(), self.fake(rate=110),
                                      "--candidate", "revert:0000000000")
        self.assertEqual(code, 0, out + err)
        self.assertIn("skipping revert:0000000000", err)
        self.assertEqual(len(self.log_rows()), 1)

    def test_effect_below_min_effect_is_noise(self):
        code, out, _ = self.run_cli(self.fake(rate=1000), self.fake(rate=1003))
        self.assertEqual(code, 0, out)
        self.assertEqual(self.log_rows()[0]["verdict"], "NOISE")

    def test_replay_mismatch_fails_unless_behavior_change_is_allowed(self):
        code, out, _ = self.run_cli(self.fake(digest="a"), self.fake(digest="b"))
        self.assertEqual(code, 1)
        self.assertIn("REPLAY-MISMATCH", out)
        code, out, _ = self.run_cli(self.fake(digest="a"), self.fake(digest="b", rate=120),
                                    "--allow-behavior-change")
        self.assertEqual(code, 0, out)
        self.assertEqual([r["verdict"] for r in self.log_rows()], ["REPLAY-MISMATCH", "FASTER"])
        self.assertEqual(self.log_rows()[1]["replay"], "differs")

    def test_behavior_change_reports_fitness_gain_per_second(self):
        code, out, _ = self.run_cli(self.fake(), self.fake(digest="b", gain=2),
                                    "--allow-behavior-change")
        self.assertEqual(code, 0, out)
        [row] = self.log_rows()
        self.assertEqual((row["verdict"], row["quality_verdict"]), ("NOISE", "FASTER"))
        self.assertAlmostEqual(float(row["quality_ratio"]), 2.0, places=3)
        evidence = Path(out.rsplit("evidence ", 1)[1].strip())
        rows = json.loads((hillclimb.REPO / evidence / "summary.json").read_text())["rows"]
        self.assertEqual(sorted({r["seed"] for r in rows}), [7, 8, 9])

    def test_executables_without_serve_run_one_process_per_trial(self):
        code, out, err = self.run_cli(self.fake(serve=0), self.fake())
        self.assertEqual(code, 0, out + err)
        self.assertIn("control: no serve mode", err)
        self.assertEqual(self.log_rows()[0]["replay"], "same")

    def test_incorrect_trial_fails_without_a_log_row(self):
        for serve in (1, 0):
            code, _, err = self.run_cli(self.fake(), self.fake(ok=0, serve=serve))
            self.assertEqual(code, 1)
            self.assertIn("failed numerical gates", err)
        self.assertFalse(hillclimb.LOG.exists())

    def test_two_pairs_only_screen(self):
        code, out, _ = self.run_cli(self.fake(rate=100), self.fake(rate=150), "--pairs", "2")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.log_rows()[0]["verdict"], "SCREEN")

    def test_invalid_arguments_are_rejected_before_any_trial(self):
        for extra in (("--cases", "ff,lena"), ("--pairs", "0"), ("--warmup", "17")):
            code, _, err = self.run_cli(self.fake(), self.fake(), *extra)
            self.assertEqual(code, 1)
            self.assertIn("hillclimb:", err)
        self.assertFalse((hillclimb.ROOT / "runs").exists())


class RevertSide(unittest.TestCase):
    def test_revert_builds_base_without_the_named_commit(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        repo = Path(scratch.name) / "repo"
        repo.mkdir()
        def run(*args):
            return hillclimb.subprocess.run(args, cwd=repo, check=True, capture_output=True,
                                            text=True).stdout.strip()
        run("git", "init", "-q")
        run("git", "config", "user.email", "t@example.com")
        run("git", "config", "user.name", "t")
        for name in ("a", "b", "c"):
            (repo / name).write_text(name)
            run("git", "add", name)
            run("git", "commit", "-qm", name)
        middle = run("git", "rev-parse", "HEAD~1")
        with mock.patch.object(hillclimb, "REPO", repo), \
                mock.patch.object(hillclimb, "ROOT", Path(scratch.name) / "hc"):
            commit, label = hillclimb.revert_commit(middle)
            content = hillclimb.git("ls-tree", "--name-only", commit)
            clean = hillclimb.git("-C", str(hillclimb.ROOT / "ref-src"), "status", "--porcelain")
        self.assertTrue(label.startswith(f"revert:{middle[:10]}@"))
        self.assertEqual(content.split(), ["a", "c"])
        self.assertEqual(clean, "")


class ForeignCpu(unittest.TestCase):
    def test_wait_for_quiet_returns_once_settled_and_gives_up_at_the_limit(self):
        with mock.patch.object(hillclimb, "QUIET_SAMPLE_SECONDS", 0.01), \
                mock.patch.object(hillclimb, "foreign_cores", return_value=0.0):
            self.assertTrue(hillclimb.wait_for_quiet([], settle=0.03, limit=5))
        with mock.patch.object(hillclimb, "QUIET_SAMPLE_SECONDS", 0.01), \
                mock.patch.object(hillclimb, "foreign_cores", return_value=8.0):
            self.assertFalse(hillclimb.wait_for_quiet([], settle=0.03, limit=0.1))

    def test_own_cpu_time_is_attributed_to_us(self):
        # Independent of other load on the machine: our busy loop lands in `ours`.
        before = hillclimb.cpu_snapshot([])
        deadline = hillclimb.time.monotonic() + 0.3
        while hillclimb.time.monotonic() < deadline:
            pass
        after = hillclimb.cpu_snapshot([])
        ours = (after[2] - before[2]) / hillclimb.CLOCK_TICKS
        self.assertGreater(ours, 0.2)
        self.assertLessEqual(ours, (after[1] - before[1]) / hillclimb.CLOCK_TICKS + 0.05)


class Statistics(unittest.TestCase):
    def test_bootstrap_interval_brackets_the_point(self):
        point, low, high = hillclimb.aggregate({"ff": [1.01, 1.02, 1.03], "mona": [0.99, 1.0, 1.01]})
        self.assertLess(low, point)
        self.assertLess(point, high)
        self.assertAlmostEqual(point, (1.02 * 1.0) ** 0.5, places=3)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    unittest.main()
