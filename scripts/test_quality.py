"""CPU-only proofs of the long-run quality driver. Children are fake Python executables.

Run: python3 -B -W error -m unittest discover -s scripts -p test_quality.py
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
_spec = importlib.util.spec_from_file_location("quality", Path(__file__).with_name("quality.py"))
quality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quality)
hc = quality.hc

# Speaks `serve` with `quality` requests. Remaining error falls as 1/sqrt(batch),
# scaled by the executable's name (scale-1.0-nd-0), the seed and crossoverProb=0.
# nd-1 adds per-process noise, so the same seed no longer replays; err-1 reports
# the recorder's scalar-renderer disagreement; ign-1 ignores --start. Each run
# writes best.json naming its case and seed.
FAKE = r'''#!/usr/bin/env python3
import hashlib, json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
fields = dict(zip(name.split("-")[::2], name.split("-")[1::2]))
assert sys.argv[1] == "serve"
for line in sys.stdin:
    words = line.split()
    assert words[0] == "quality"
    flags = dict(zip([w[2:] for w in words[1::2]], words[2::2]))
    params = json.loads(flags.get("params", "{}"))
    if set(params) - {"crossoverProb", "lambda"}:
        sys.exit("unknown MutationParams field")
    seed, seconds = int(flags["seed"]), float(flags["seconds"])
    scale = float(fields["scale"]) * (1 + 0.02 * seed)
    scale *= 1.2 if params.get("crossoverProb") == 0 else 1.0
    noise = (os.getpid() % 97) * 1e-6 if fields["nd"] == "1" else 0.0
    points = []
    for batch in range(int(seconds * 10) + 2):
        r = 30 * scale / (1 + batch) ** 0.5 + noise
        points.append({"seconds": batch * 0.1, "batches": batch, "evaluations": batch * 16384,
                       "fitness": 100 - r, "raw_fitness": 100 - r * 0.99,
                       "polygons": min(batch, 1000), "points": 3 * min(batch, 1000)})
    record = {"kind": "quality", "case_id": flags["case"], "seed": seed,
              "elapsed_seconds": points[-1]["seconds"], "batches": points[-1]["batches"],
              "evals_per_second": 163840.0, "trajectory": points,
              "errors": ["delivered fitness differs from the scalar renderer's"]
              if fields["err"] == "1" else []}
    if "start" in flags and fields["ign"] == "0":
        record["start_sha256"] = hashlib.sha256(Path(flags["start"]).read_bytes()).hexdigest()
    out = Path(flags["output"]); out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(record))
    (out / "best.json").write_text(json.dumps({"case": flags["case"], "seed": seed}))
    print("engine log line", flush=True)
    print("@result " + json.dumps(record), flush=True)
'''


class QualityCli(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        manifest = self.root / "manifest.json"
        manifest.write_text("{}")
        patches = [mock.patch.object(hc, "wait_for_quiet", return_value=True),
                   mock.patch.object(hc, "foreign_cores", return_value=0.0),
                   mock.patch.object(hc, "MANIFEST", manifest),
                   mock.patch.object(hc, "LOCK", self.root / "lock"),
                   mock.patch.object(hc, "ROOT", self.root / "hc"),
                   mock.patch.object(quality, "ROOT", self.root / "quality"),
                   mock.patch.object(quality, "LOG", self.root / "log.tsv")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def fake(self, scale=1.0, nd=0, err=0, ign=0):
        path = self.root / f"scale-{scale}-nd-{nd}-err-{err}-ign-{ign}"
        path.write_text(FAKE)
        path.chmod(0o755)
        return str(path)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = quality.main(["--images", "ff,mona", "--seeds", "3", "--seconds", "20", *argv])
        return code, out.getvalue(), err.getvalue()

    def log_rows(self):
        lines = quality.LOG.read_text().splitlines()
        header = lines[0].split("\t")
        return [dict(zip(header, line.split("\t"))) for line in lines[1:]]

    def summary(self, out):
        evidence = hc.REPO / out.rsplit("evidence ", 1)[1].strip()
        return json.loads((evidence / "summary.json").read_text())

    def test_single_side_characterization_prints_distributions_and_logs_each_run(self):
        code, out, err = self.run_cli("--side", f"master={self.fake()}")
        self.assertEqual(code, 0, out + err)
        self.assertIn("ff: remaining error", out)
        self.assertIn("sd ln r", out)
        self.assertIn("99%:", out)
        self.assertNotIn(" vs ", out)
        rows = self.log_rows()
        self.assertEqual(len(rows), 2 * 3)
        self.assertEqual({row["seed"] for row in rows}, {"1", "2", "3"})
        remaining = json.loads(rows[0]["remaining_at"])
        self.assertEqual(list(remaining), ["1", "2", "5", "10", "20"])

    def test_saved_starts_seed_later_runs_and_are_logged(self):
        fake = self.fake()
        starts = self.root / "starts"
        code, out, err = self.run_cli("--side", f"base={fake}", "--save-starts", str(starts))
        self.assertEqual(code, 0, out + err)
        self.assertEqual(json.loads((starts / "mona" / "seed2.json").read_text()),
                         {"case": "mona", "seed": 2})
        meta = json.loads((starts / "meta.json").read_text())
        self.assertEqual((meta["seconds"], len(meta["fitness"])), (20.0, 6))
        self.assertEqual({row["start"] for row in self.log_rows()}, {""})

        code, out, err = self.run_cli("--side", f"a={fake}", "--side", f"b={fake}",
                                      "--start", str(starts))
        self.assertEqual(code, 0, out + err)
        started = self.log_rows()[6:]
        self.assertEqual(len(started), 12)
        self.assertTrue(all(len(row["start"]) == 12 for row in started))
        self.assertEqual(len({(row["image"], row["seed"], row["start"]) for row in started}), 6)

        code, _, err = self.run_cli("--side", f"old={self.fake(ign=1)}", "--start", str(starts))
        self.assertEqual(code, 1)
        self.assertIn("did not start from the requested drawing", err)

        code, _, err = self.run_cli("--side", f"a={fake}", "--start", str(starts), "--seeds", "4")
        self.assertEqual(code, 1)
        self.assertIn("missing start drawing", err)
        code, _, err = self.run_cli("--side", f"a={fake}", "--save-starts", str(starts))
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)
        code, _, err = self.run_cli("--side", f"a={fake}", "--side", f"b={fake}",
                                    "--save-starts", str(self.root / "new"))
        self.assertEqual(code, 1)
        self.assertIn("exactly one side", err)

    def test_identical_sides_are_noise_and_replay_identically(self):
        fake = self.fake()
        code, out, _ = self.run_cli("--side", f"a={fake}", "--side", f"b={fake}")
        self.assertEqual(code, 0, out)
        self.assertIn("1.0000 [1.0000, 1.0000] NOISE", out)
        self.assertIn("determinism: 6 same-build same-params run pairs", out)
        self.assertIn("identical", out)

    def test_params_override_is_converted_and_judged_worse(self):
        fake = self.fake()
        code, out, _ = self.run_cli("--side", f"control={fake}", "--side", f"nox={fake}",
                                    "--params", 'nox={"crossover_prob": 0}')
        self.assertEqual(code, 0, out)
        entries = self.summary(out)["comparisons"]
        self.assertEqual({e["verdict"] for e in entries}, {"WORSE"})
        self.assertAlmostEqual(entries[-1]["ratio"], 1.2, places=6)
        self.assertNotIn("determinism", out)  # different params are not a replay pair
        self.assertEqual(self.log_rows()[-1]["params"], '{"crossoverProb":0}')

    def test_better_build_is_better_and_reaches_targets_sooner(self):
        code, out, _ = self.run_cli("--side", f"control={self.fake()}",
                                    "--side", f"cand={self.fake(scale=0.8)}")
        self.assertEqual(code, 0, out)
        self.assertEqual({e["verdict"] for e in self.summary(out)["comparisons"]}, {"BETTER"})
        self.assertIn("time to control-worst fitness", out)
        # r ~ 1/sqrt(t): 0.8x the error reaches a target in 0.64x the time.
        line = next(x for x in out.splitlines() if "control-worst fitness at T=20s" in x)
        self.assertAlmostEqual(float(line.split("): ")[1].split()[0]), 0.64, delta=0.03)

    def test_two_seeds_only_screen(self):
        code, out, _ = self.run_cli("--side", f"control={self.fake()}",
                                    "--side", f"cand={self.fake(scale=0.5)}", "--seeds", "2")
        self.assertEqual(code, 0, out)
        self.assertEqual({e["verdict"] for e in self.summary(out)["comparisons"]}, {"SCREEN"})

    def test_nondeterministic_replay_fails(self):
        fake = self.fake(nd=1)
        code, out, _ = self.run_cli("--side", f"a={fake}", "--side", f"b={fake}",
                                    "--server-scope", "run")
        self.assertEqual(code, 1)
        self.assertIn("DETERMINISM-MISMATCH", out)

    def test_disturbed_run_is_rerun_and_still_checked_for_replay(self):
        readings = iter([5.0] + [0.0] * 100)
        with mock.patch.object(hc, "foreign_cores", lambda *_: next(readings)):
            code, out, _ = self.run_cli("--side", f"master={self.fake()}")
        self.assertEqual(code, 0, out)
        self.assertIn("re-ran 1 runs", out)
        self.assertIn("determinism: 1 same-build", out)
        self.assertEqual(len(self.log_rows()), 6)
        self.assertEqual(max(r["attempts"] for r in self.log_rows()), "2")

    def test_run_whose_fitness_fails_the_scalar_check_fails_without_a_log_row(self):
        code, _, err = self.run_cli("--side", f"a={self.fake(err=1)}")
        self.assertEqual(code, 1)
        self.assertIn("ff seed 1: invalid run: [\"delivered fitness differs", err)
        self.assertFalse(quality.LOG.exists())

    def test_rejected_params_end_the_run_with_an_error(self):
        code, _, err = self.run_cli("--side", f"a={self.fake()}", "--params", 'a={"bogus": 1}')
        self.assertEqual(code, 1)
        self.assertIn("server exited", err)
        self.assertFalse(quality.LOG.exists())

    def test_invalid_arguments_are_rejected_before_any_run(self):
        fake = self.fake()
        for extra in (("--side", "a"), ("--side", f"a={fake}", "--side", f"a={fake}"),
                      ("--side", f"a={fake}", "--params", 'b={"x":1}'),
                      ("--side", f"a={fake}", "--params", "a={not json"),
                      ("--side", f"a={fake}", "--params", 'a={"x":"word"}'),
                      ("--side", f"a={fake}", "--images", "lena"),
                      ("--side", f"a={fake}", "--seconds", "0")):
            code, _, err = self.run_cli(*extra)
            self.assertEqual(code, 1, extra)
            self.assertIn("quality:", err)
        self.assertFalse((quality.ROOT / "runs").exists())


POINTS = ({"seconds": 0.0, "fitness": 60.0}, {"seconds": 1.0, "fitness": 90.0},
          {"seconds": 3.0, "fitness": 99.0})


class Trajectory(unittest.TestCase):

    def test_value_at_takes_the_first_point_at_or_after(self):
        self.assertEqual(quality.value_at(POINTS, 1.0)["fitness"], 90.0)
        self.assertEqual(quality.value_at(POINTS, 1.5)["fitness"], 99.0)
        self.assertEqual(quality.value_at(POINTS, 9.0)["fitness"], 99.0)

    def test_time_to_interpolates_in_log_remaining_error(self):
        self.assertEqual(quality.time_to(POINTS, 50.0), 0.0)
        self.assertAlmostEqual(quality.time_to(POINTS, 100 - 10 ** 0.5), 2.0)  # r 10 -> 1
        self.assertIsNone(quality.time_to(POINTS, 99.5))

    def test_checkpoints_end_at_the_budget(self):
        self.assertEqual(quality.checkpoints(45), [1, 2, 5, 10, 20, 30, 45])
        self.assertEqual(quality.checkpoints(300)[-2:], [180, 300])


if __name__ == "__main__":
    unittest.main()
