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
scripts/quality.py --side cap1000=HEAD --side cap2000=HEAD --params 'cap2000={"max_polygons":2000}'
scripts/quality.py --side p1=HEAD --side p16=HEAD --params 'p16={"per_point_penalty":3.2e-6}'
scripts/quality.py --side a=HEAD --side b=HEAD --seconds 20 --seeds 2                # A/A
```

- `--side NAME=BUILD` (repeatable; the first side is the control). BUILD is a git
  ref, `WORKTREE`, `revert:SHA[@BASE]` or an `evolution-bench` executable. Builds
  share hillclimb's cache (`.cache/hillclimb/bin`), so sides with the same build
  compile once.
- `--params NAME=JSON|FILE` gives one side's `MutationParams` overrides, merged
  onto `MutationParams::default()`. Keys may be snake_case or camelCase. The engine
  rejects unknown fields, wrong types and values that `sanitize` would change.
  Two settings shape the late game: `max_polygons`, the search cap (default
  1000, at most the polygon capacity 2048; see
  [engine architecture](engine-architecture.md#polygon-capacity-and-the-search-cap)),
  and `per_point_penalty` (default 2e-7 = 1/5e6, at most 1e-4), the fitness
  factor `1 - penalty * points`. One build serves every cap and penalty, so such
  sides differ by settings only. Each side's engine is allocated for its own
  chain count and λ.
- `--images` (default ff,mona,darwin; see darwin under Limits), `--seeds` (default 5),
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
the delivered drawing. `raw_fitness = fitness / (1 - per_point_penalty *
points)` removes the run's point penalty from `compute_fitness`, so `raw r`
compares sides with different penalties on error alone.

After the clock stops, the final delivered drawing is rendered by the scalar
renderer (`reference::evaluate`, with the run's `per_point_penalty`) and
described by `reference::complexity`. A gap
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

- **darwin builds full tile lists.** From a blank start it reaches the
  256-polygon tile lists (`TILE_MAX_POLYS`) within ~3 s. Builds before the
  overflow rejection reported 99.97% for a drawing the scalar renderer scores at
  81% and fail its scalar check (see LEARNINGS); use `--images ff,mona` with such
  a side. Current builds reject offspring that overflow a list, so darwin is
  scored honestly. On 2026-10-29 (seeds 1–2, 30 s) darwin reached 98.54 and
  98.37%, equal to the scalar rescore, with 920 and 782 triangles and at most
  233 polygons in a tile. It ran 667–1503 batches against ff's ~2400.
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

## Master baseline, 2026-10-29

Defaults (4 chains, λ 64), 5 seeds × 300 s from blank, fresh server per run
(`7183161`; evidence `.cache/quality/runs/20261029T133826Z-5nb53zzs`).
Remaining error r = 100 − fitness%, median [min..max], and median triangles:

| T s | ff r | ff triangles | mona r | mona triangles |
| --- | --- | --- | --- | --- |
| 5 | 2.37 [2.24..2.70] | 549 | 3.79 [3.66..5.97] | 336 |
| 30 | 1.50 [1.47..1.55] | 853 | 2.48 [2.40..2.70] | 713 |
| 60 | 1.37 [1.36..1.44] | 942 | 2.33 [2.28..2.46] | 842 |
| 120 | 1.32 [1.28..1.36] | 1000 | 2.20 [2.18..2.32] | 972 |
| 300 | 1.28 [1.26..1.31] | 1000 | 2.17 [2.16..2.21] | 1000 |

- Both images reach the 1000-triangle cap by 120–180 s; after it, r falls
  only 3% (ff) and 1.5% (mona) in the next 180 s. The cap, not the search,
  limits the late game at the current penalty (1000 triangles cost 0.06%).
  The cap (`max_polygons`, up to 2048) and the penalty (`per_point_penalty`)
  are parameters since the polygon capacity was raised; this baseline used
  1000 and 2e-7.
- Seed-to-seed spread (sd of ln r) is 0.08–0.20 up to 5 s and 0.01–0.03 from
  30 s, so 5 seeds resolve about 2–3% differences late; the default
  `--min-effect 0.02` matches.
- Batches per run vary between seeds (mona 10k–19k in 300 s), because the
  raster cost follows the drawing; compare at fixed seconds, not batches.

## Cap and penalty sweep, 2026-10-29

`2f79e3d`, 3 seeds × 300 s per image, settings-only sides. Ratio of
unpenalized remaining error to the default (cap 1000, penalty 2e-7), lower is
better, with the 95% interval, and final median triangles. Penalized and
unpenalized ratios agree within 2%.

| Side | T | ff | mona | darwin |
| --- | --- | --- | --- | --- |
| cap 2000 | 120 s | 0.974 | 0.999 | 0.966 |
| cap 2000 | 300 s | 0.910 [0.899, 0.929], 1128 | 0.947 [0.945, 0.948], 1176 | 0.894 [0.858, 0.918], 1251 |
| penalty 8e-7 (cap 1000 or 2000) | 300 s | 1.590, 517 | 1.422, 399 | 1.412, 539 |

- Cap 2000 gains nothing until the default runs reach 1000 triangles
  (120–180 s), then 5–11% by 300 s and still widening; it is not binding
  at 300 s.
- A 4x penalty halves the triangle count and costs 41–59% more remaining
  error, and never reaches either cap. Penalized drawings converge to a
  different, much smaller drawing; any useful penalty is well below 4x.

Evidence: `.cache/quality/runs/20261029T164405Z-p_5yl0m9` (ff),
`20261029T174525Z-jf8bdj4a` (mona), `20261029T184656Z-pm8z1wan` (darwin).
