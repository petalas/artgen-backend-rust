# Long-run quality benchmark

`scripts/quality.py` measures how far evolution gets from a blank drawing over
minutes of wall clock. Throughput screens (`scripts/hillclimb.py`) time 64–512
batches from mature fixtures and cannot show late-game behavior. Use this
benchmark for evolution settings (mutation probabilities and sizes, adaptive
mutation, polygon penalty, chains vs lambda, crossover and tournament) and for
search changes. Sides can differ by build, by settings, or both.

```sh
scripts/quality.py --side master=master --images ff,mona --seeds 5 --seconds 300   # characterize
scripts/quality.py --side control=HEAD --side cand=WORKTREE                          # build A/B
scripts/quality.py --side control=HEAD --side nox=HEAD --params 'nox={"crossover_prob":0}'
scripts/quality.py --side a=HEAD --side b=HEAD --seconds 20 --seeds 2                # A/A
```

- `--side NAME=BUILD` (repeatable; the first side is the control). BUILD is a git
  ref, `WORKTREE`, `revert:SHA[@BASE]` or an `evolution-bench` executable. Builds
  share hillclimb's cache (`.cache/hillclimb/bin`), so sides with the same build
  compile once.
- `--params NAME=JSON|FILE` gives one side's `MutationParams` overrides, merged
  onto `MutationParams::default()`. Keys may be snake_case or camelCase. The engine
  rejects unknown fields, wrong types and values that `sanitize` would change.
- `--images` (default ff,mona,darwin; see darwin below), `--seeds` (default 5),
  `--first-seed`, `--seconds` (default 120, at most 3600), `--min-effect` (default
  0.02), `--server-scope image|run`, `--build-only`, `--note`.

Only builds that include the quality recorder can be sides. Older executables
have no `quality` request, and their server exits on the first run.

## What a run is

`evolution-bench` accepts a `quality` request, in `serve` mode or as a command:

```text
quality --case ff --seed 3 --seconds 300 --warmup 8 [--params {"crossoverProb":0}] --output DIR
```

`src/benchmark/quality.rs` restarts the resident engine on a blank drawing with
`prepare_seeded_benchmark`, then calls `run_batch(params, true)` until the budget
passes. This is the app's call, with timestamps, one-batch-behind delivery and
the strict-improvement global best. The final pending batch is flushed and the
GPU is drained inside the clock. Every value comes from a drawing the engine had
already delivered, so recording adds no GPU round-trip. The fitness is the GPU's
fitness. The recorder only rescores the final drawing, after the clock stops.

Trajectory points (`seconds, batches, evaluations, fitness, raw_fitness, polygons,
points`) are recorded:

- at every batch during the first second (at most 512) and at power-of-two batch
  counts;
- each time elapsed time grows 5%, and at the first batch at or after each of
  1, 2, 5, 10, 20, 30, 60, 120, 180, 300 and 600 s;
- each time the best remaining error falls by 2%;
- at the end, after the drain.

`batches` and `evaluations` count submitted work, which runs one batch ahead of
the delivered drawing. `raw_fitness = fitness / (1 - PER_POINT_MULTIPLIER *
points)` removes the point penalty from `compute_fitness`.

After the clock stops, the final delivered drawing is rendered by the scalar
renderer (`reference::evaluate`) and described by `reference::complexity`. A gap
above 0.05 fitness points, or any wgpu device error, makes the run invalid, and
the script stops. `DIR` gets `result.json` (the full record) and `best.json`.

## Metrics

At checkpoint T, which is each fixed second below the budget plus the budget:

- **Remaining error** r = 100 − fitness% of the best delivered at the first
  recorded batch at or after T, overshooting T by at most one batch. The final
  checkpoint uses the post-drain point. `raw r` uses the unpenalized fitness.
- **Per side and image:** median r with its interquartile range, minimum and
  maximum, `sd ln r` (the seed-to-seed standard deviation of ln r), median fitness,
  median unpenalized r, and polygon median and range. Also reported: the median
  time to 98, 99 and 99.5% fitness, shown only when more than half of the seeds
  reached the target.
- **Comparison:** the geometric mean over images × seeds of
  r_candidate / r_control, paired by image and seed. Each image gets equal
  weight. The 95% interval is hillclimb's stratified bootstrap. A result is
  `BETTER` when the interval is below 1 and the ratio is ≤ 1 − min-effect, and
  `WORSE` in the mirror case. Otherwise it is `NOISE`. With fewer than three
  seeds, every result is `SCREEN`. The same comparison is made for raw r.
- **Time to target:** this is the time ratio, candidate over control. The targets are
  the worst control seed's fitness at the final checkpoint and at the checkpoints
  nearest a quarter and a sixteenth of the budget. Every control seed reaches
  them by construction. Times are interpolated in ln r between the points that
  bracket a target. A candidate that misses a target is charged its full run
  length, a lower bound, and the script counts these.
- **Final polygons:** the median per image, control → candidate.

`sd ln r` sets how many seeds a comparison needs. The paired ratio of one seed
has a log SD of about √2 · sd ln r when sides are uncorrelated.

## Determinism and noise

Given a seed, the delivered best is a function of the batch index. Wall clock
only maps batches to seconds. Whenever two runs share a build, params, image and
seed (A/A sides and discarded re-runs), the script compares delivered fitness at
every batch index both recorded. It prints `identical` or `DETERMINISM-MISMATCH`;
a mismatch gives exit code 1. Measured on 2026-10-29 (ff and mona, seeds 1–2,
fresh servers per image):

- A/A, 20 s: 491 shared batch points, all identical. r ratio 1.0002 at 20 s, and
  at most ±0.5% at any checkpoint, from timing only. The batch counts of the two
  sides differed by at most 7 of ~1500.
- A/A, 10 s: 474 points identical. Ratios were 0.9996–1.0000.
- `crossover_prob 0` vs default, 20 s: r ratio 1.14 [1.11, 1.18] at 20 s and 1.13
  at 5 s, with fewer polygons (ff 834 → 678). That is a SCREEN with two seeds.

Batch rate depends on the image and the seed: ff runs ~70–80 batches/s and mona
~25–38, so equal seconds mean unequal work across seeds. That is intended,
because settings that slow the engine pay for it here.

The quiet-host check, lock and retry come from hillclimb. Before each run the
script waits until other processes use at most one core for 5 s. A run during
which they averaged more than one core is discarded and re-run, up to three
times. Each image gets fresh server processes, one per side. The engine's host
submit time grows over the first ~4000 batches of a process (LEARNINGS, "Host
submit time grows"). Every side runs its seeds in the same order, so that growth
is matched between sides, but a process's first seed runs on a fresh process.
`--server-scope run` starts a process per run for characterizations that must
exclude it.

## Cost

A run costs its budget plus ~6 s for the quiet wait, the seeded reset and the
scalar check. Engine construction (shader compilation, ~12 s) is paid once per
image and side, or once per run with `--server-scope run`. Measured: 12 runs of
10 s (2 images × 2 seeds × 3 sides) took 4 min 14 s. Estimate:
`images × seeds × sides × (seconds + 7) + images × sides × 13` seconds. For 2
images, 5 seeds, 300 s and one side, that is ~52 minutes.

## Limits

- **darwin is not usable yet.** From a blank start, its tile bins overflow
  (`TILE_MAX_POLYS` = 256) within ~3 s. The GPU then reports 99.97% for a drawing
  the scalar renderer scores at 81%, and the run fails its scalar check (see
  LEARNINGS). Pass `--images ff,mona` until the engine handles overflow.
- Only the frozen v1 fixture references at 384 px are available (ff, mona,
  darwin). milkshakes has no fixture, and other resolutions are not supported.
- Logs: one row per scored run goes to `benchmarks/quality/log.tsv`, with the
  side, build, params, image, seed, final fitness and polygons, r at each
  checkpoint and host load. Evidence goes to `.cache/quality/runs/<utc>-*/`: each
  run's `result.json` and `best.json`, `serve/*/stderr.log` and a `summary.json`
  with the comparisons and targets.

## Power of Ten notes

- Rule 2: the batch loop stops at the budget and at a hard 2^24 batches. Grid
  walks are capped at 4096 steps. The trajectory is preallocated to 2048 points,
  and exhausting it is an error. The budget is limited to 3600 s.
- Rule 3: `run_batch` allocates each delivered `Drawing`, as it does in the app.
  The recorder keeps only the best one and the preallocated points.
- Rule 10: `evals_per_second` casts `u64` to `f64`, as trial records do (Clippy
  `cast_precision_loss`). Evaluation counts stay far below 2^53.
