# Long-run quality benchmark

`scripts/quality.py` measures how far evolution gets from a blank drawing, or
from a saved mature drawing, over minutes of wall clock. Throughput screens (`scripts/hillclimb.py`) time 64–512
batches from mature fixtures and cannot show late-game behavior. Use this
benchmark for evolution settings (mutation probabilities and sizes, adaptive
mutation, polygon penalty, chains vs lambda, crossover and tournament) and for
search changes. Sides can differ by build, by settings, or both.

```sh
scripts/quality.py --side master=master --images ff,mona --seeds 5 --seconds 300   # characterize
scripts/quality.py --side control=HEAD --side cand=WORKTREE                          # build A/B
scripts/quality.py --side control=HEAD --side nox=HEAD --params 'nox={"crossover_prob":0}'
scripts/quality.py --side cap2000=HEAD --side cap1000=HEAD --params 'cap1000={"max_polygons":1000}'
scripts/quality.py --side p1=HEAD --side p16=HEAD --params 'p16={"per_point_penalty":3.2e-6}'
scripts/quality.py --side c=HEAD --side sp=HEAD --params 'sp={"step_spread_min":0.25,"step_spread_max":4}'
scripts/quality.py --side a=HEAD --side b=HEAD --seconds 20 --seeds 2                # A/A
scripts/quality.py --seconds 120 --seeds 5 --save-starts .cache/quality/starts/s120   # make starts
scripts/quality.py --start .cache/quality/starts/s120 --side control=HEAD --side cand=WORKTREE \
    --images ff --seeds 3 --seconds 90                                               # late-game screen
```

- `--side NAME=BUILD` (repeatable; the first side is the control). BUILD is a git
  ref, `WORKTREE`, `revert:SHA[@BASE]` or an `evolution-bench` executable. Builds
  share hillclimb's cache (`.cache/hillclimb/bin`), so sides with the same build
  compile once.
- `--params NAME=JSON|FILE` gives one side's `MutationParams` overrides, merged
  onto `MutationParams::default()`. Keys may be snake_case or camelCase. The engine
  rejects unknown fields, wrong types and values that `sanitize` would change.
  Two settings shape the late game: `max_polygons`, the search cap (default
  2000 since the cap sweep below; 1000 before, at most the polygon capacity 2048; see
  [engine architecture](engine-architecture.md#polygon-capacity-and-the-search-cap)),
  and `per_point_penalty` (default 2e-7 = 1/5e6, at most 1e-4), the fitness
  factor `1 - penalty * points`. One build serves every cap and penalty, so such
  sides differ by settings only. Each side's engine is allocated for its own
  chain count and λ.
- `step_spread_min` and `step_spread_max` (default 0.25 and 4; 1 each is no
  spread, and a record or setting without them reads as no spread; each within
  [1/16, 16], min at most max) spread offset and vertex move sizes over
  each chain's offspring: offspring i uses rung i mod 8, multiplier
  `min * (max/min)^(rung/7)`, on top of the adaptive scale (see
  [engine architecture](engine-architecture.md#step-spread)). A side with a
  spread gets a step rung table (below).
- `--images` (default ff,mona,darwin; see darwin under Limits), `--seeds` (default 5),
  `--first-seed`, `--seconds` (default 120, at most 3600), `--min-effect` (default
  0.02), `--server-scope image|run`, `--build-only`, `--note`.
- `--start DIR` starts every run from `DIR/IMAGE/seedN.json` instead of blank;
  each image and seed needs its file. `--save-starts DIR` (one side, `DIR` must
  not exist) copies each run's final delivered drawing there, plus `meta.json`
  with the build, params, budget and start that made them. See
  [Mature starts](#mature-starts).

Only builds that include the quality recorder can be sides. Older executables
have no `quality` request, and their server exits on the first run.

## What a run is

`evolution-bench` accepts a `quality` request, in `serve` mode or as a command:

```text
quality --case ff --seed 3 --seconds 300 --warmup 8 [--params {"crossoverProb":0}] [--start FILE] --output DIR
```

`src/benchmark/quality.rs` restarts the resident engine on a blank drawing, or
on the `--start` drawing in every chain, with `prepare_seeded_benchmark`, then calls `run_batch(params, true)` until the budget
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
The record (protocol 5) carries `start_sha256`, the start file's SHA-256 or
null for a blank start; the script checks it against the file it named.
Protocol 3 added operator stats (below), protocol 4 their step rung counts and
protocol 5 their gain; the script still accepts protocol 2 to 4 records, which
lack them.

## Operator stats

The record's `operator_names` and `operators` say which search operators
produced the accepted winners (see
[engine architecture](engine-architecture.md#operator-stats)). Each snapshot
holds cumulative `offered`, `improved` and `neutral` counts per operator,
taken at the first batch at or after each fixed second and after the drain.
`batches` and `evaluations` in a snapshot count the collected batches, one
fewer than submitted except in the final snapshot; offered always sums to
`evaluations`, and the script rejects a run where it does not. Counting starts
at the seeded start; warmup is excluded.

The script prints one table per side and image, counts summed over seeds, with
windows ending at 5, 30, 120 and 600 s and the budget. Per window: the
operator's share of all improvements (%) and its rate, improvements per 1000
offspring it produced. The last four columns cover the whole run: share of
offspring, share of improvements, rate and share of neutral acceptances.
`summary.json` (`operators`) has the counts for a window per checkpoint.

Share says where improvements came from; rate says how productive an
operator's offspring are, so a high-share, low-rate operator is common, not
good. Neutral acceptances are mostly copies: crossover of near-identical
parents yields offspring equal to the parent, which select takes half the time.

### Step rungs

Protocol 4 records also hold, per snapshot, `rung_offered`, `rung_improved` and
`rung_neutral`: the same counts per step rung, for the operators that scale
with the step only (`step_operators`: offset, move point, medium move, micro
move), with the rungs' multipliers in `step_multipliers`. Each rung block sums
to those operators' counts; the script rejects a run where it does not.

For a side whose records have a spread (multipliers that differ), the script
prints a rung table per image after the operator table, summed over seeds:
the rung, its multiplier (`step x`), its share of the step-scaled offspring
(12.5% each at λ 64), improvements per 1000 offered in each window (the
operator table's windows), then the whole run's share of the step-scaled
improvements and rate. Compare rates down a column: the rung with the highest
rate is the step size that paid in that window, and a peak that moves to
smaller rungs in later windows says the search wants finer steps as the
drawing matures. Rates count improvements, not their size, and offspring of
every rung compete in the same select, so a rung's rate also depends on the
others. `summary.json` has the rung counts under `rungs` in the side's
`operators` entry.

### Gain

Protocol 5 snapshots also hold `gain` per operator and `rung_gain` per rung:
the summed fitness gain of their improvements, in error units (fitness points
times max total error / 100, so one unit is about 9e-7 points at 384 px). It is
the error drop less the change in point penalty, signed: an addition that
lowers error by less than its points cost would count negative, though select
never accepts one. The rung gains sum to the step-scaled operators'; the script
rejects a run where they do not.

Both tables then end in four gain columns: `gain%`, the row's share of all gain
in the table over the run; `last gain%`, the same for the last window; `per win`,
gain per improvement relative to the table's average (gain share over
improvement share); and `per offer`, gain per offspring relative to the average
(gain share over offspring share). `per offer` above 1 means the row's offspring
are worth more than average; it is the column to weigh operators and rungs by.
Gain sums every chain's improvements, so a crossover that pulls in another
chain's polygons gets credit for gains that chain's own operators found first.

Smoke run, 2026-10-30 (mona, seed 1, 60 s from `s120v2`, spread 0.25–4,
evidence `.cache/quality/runs/20261030T183122Z-_5hdokr8`): rates fell
steadily with step size in every window, 0.195 per 1000 at ×0.25 to 0.080 at
×4 over the run (0.390 → 0.141 in 0–5 s, 0.122 → 0.054 in 30–60 s). One
seed, one image: the mechanism works, not a tuning result.

Smoke run, 2026-10-30 (mona, seed 1, 60 s from the s120 starts): crossover
made 20% of offspring and 75% of the 15,919 improvements (1.1 per 1000),
micro_move 11% (0.20 per 1000), offset 0.3% from 10.7% of offspring
(0.009 per 1000). Offered summed to the record's 54,427,648 evaluations.

## Mature starts

From blank, most of a run repeats the early game: ff reaches r 1.50 by 30 s and
1.32 by 120 s (baseline below), and settings that matter only late spend their
budget getting there. A start set is the final drawings of one characterization
run (`--save-starts`), one per image and seed. A side then starts from those
drawings, with its own params and seed, so the whole budget measures the late
game. The frozen v1 fixture drawings are not mature enough: they score 97.2–97.4%
with 423–539 triangles, worse than a blank run's 5 s.

What a start changes: every chain begins from the same single drawing, not the
population that made it, and the starts were evolved under the start set's
params. So a start-set result says how a side continues a default-evolved
drawing, not how it does from blank; early-game effects are not measured.
Starts for seed N are used by seed N, so seed-to-seed variation includes the
start. A typical workflow is a one-image, three-seed screen from starts (about 9
minutes for two sides at 90 s), then the full three images from starts for
candidates that pass.

Smoke check, 2026-10-30: a 5 s mona run saved a 95.92% drawing; a run started
from it recorded initial fitness 95.92382, equal to the saved fitness, and the
start's hash in the log.

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
  checkpoint, host load and the first 12 hex digits of the start drawing's
  SHA-256 (empty for a blank start). Evidence goes to `.cache/quality/runs/<utc>-*/`: each
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

Default cap raised to 2000 after this sweep. Results above and the master
baseline were measured with the old default of 1000.

## Population shape sweep, 2026-10-29

`62a961c` (cap 2000), 256 offspring per iteration split as chains × λ,
3 seeds × 300 s. Unpenalized remaining error relative to 4×64 (the
default), and batches completed in 300 s:

| Shape | T | ff | mona | darwin |
| --- | --- | --- | --- | --- |
| 16×16 | 30 s | 1.052 WORSE | 1.017 NOISE | 1.084 NOISE |
| 16×16 | 300 s | 1.031 NOISE | 1.013 NOISE | 1.053 NOISE |
| 64×4 | 30 s | 1.320 WORSE | 1.162 WORSE | 1.134 WORSE |
| 64×4 | 300 s | 1.090 WORSE | 1.082 WORSE | 1.043 WORSE |
| batches, 4×64 / 16×16 / 64×4 | 300 s | 25k / 23k / 12k | 17k / 14k / 6k | 6–18k / 6–7k / 5–6k |

- The default 4×64 stays best. Larger populations close most of the gap by
  300 s, but 64×4 completes half as many batches (same evaluations per
  batch), so part of its deficit is engine cost per chain, not search.
- Darwin's batch rate varies 3x between seeds of one side; its intervals are
  wide.

Evidence: `.cache/quality/runs/` for the "population shape sweep" rows in
`benchmarks/quality/log.tsv`.

## Crossover probability sweep, 2026-10-30

`b6cee9c`, 4×64, 3 seeds × 300 s. Unpenalized remaining error relative to
`crossover_prob` 0.2 (the default):

| crossover_prob | ff | mona | darwin |
| --- | --- | --- | --- |
| 0.1 | 1.023 [1.005, 1.056] WORSE | 0.997 NOISE | 0.987 NOISE |
| 0.4 | 1.048 [1.016, 1.106] WORSE | 1.020 [1.005, 1.048] NOISE | 0.974 NOISE |

0.2 stays: no setting is better on any image, and each alternative is worse
on ff. Darwin's batch count again varied 3x between seeds of one side
(6.8k–21.7k), which keeps its intervals wide. An earlier 20 s screen showed
crossover off (0.0) about 13% worse.

## Late-game characterization from s120, 2026-10-30

`efe837f`, defaults, start set `s120` (defaults, 120 s from blank, seeds 1–3),
120 s per run (evidence `.cache/quality/runs/20261030T115540Z-4yfm7t_t`).

| image | r at start | r at 120 s | change | sd ln r at 120 s |
| --- | --- | --- | --- | --- |
| ff | 1.279 | 1.205 | −5.8% | 0.006 |
| mona | 2.174 | 2.081 | −4.3% | 0.016 |
| darwin | 1.324 | 1.229 | −7.2% | 0.051 |

- From a mature start, a run removes only 4–7% of r in 120 s, so a setting
  that makes progress 20% faster moves r by about 1%. Judge start-based
  comparisons by time to target (the control-worst fitness at the budget),
  not by the r ratio against the default 2% `--min-effect`.
- Operators, 30–120 s, all three images: crossover is 20% of offspring and
  73–76% of improvements. With 4 chains an improvement that one chain's
  mutation found is counted again each time crossover copies it into another
  chain, so this share overstates what crossover creates. Of the mutations,
  micro vertex moves give 10–13% of improvements from 15.8% of offspring and
  full colour changes 2–4% from 3.6%. Offset (10.7% of offspring) gives 0.2–0.3%,
  medium moves (22.7%) 3–4%, micro colour (21.1%) 5–6%, large point moves (1.7%)
  0.1%. Reorder has the highest rate (4–10 per 1000 offered) but is offered
  almost never, because its weight does not scale with polygon count. Counts
  say how often an operator wins, not by how much.
- Accepted winners: 0.7–1.6% of chain-iterations improve, but 48% accept a
  neutral winner (equal fitness, the 50% coin flip), mostly crossover copies
  identical to the parent. Each acceptance pays the polygon copy and the
  framebuffer update.

## Adaptive mutation screen from s120, 2026-10-30

`efe837f`, ff, seeds 1–3, 90 s from `s120`, `adaptive_mutation` true vs false
(evidence `.cache/quality/runs/20261030T121440Z-7tsb32oc`).

- r ratio at 90 s 0.9953 [0.9909, 0.9986], NOISE against the 2% bar.
- Time to the control's worst final fitness: 0.84 [0.76, 0.91], so the
  adaptive side reached the control's 90 s result about 16% sooner. It made
  52,485 improvements against 46,241, mostly from micro vertex moves early
  and crossover. Adaptive scales offset, point-move and micro vertex move
  sizes by 1.46–2.0 once improvements are rare (the stagnation reset holds
  it there), so here larger geometric steps helped. One image and three
  seeds: a screen, not an adoption result.

## Adaptive confirm and operator weight screen from s120, 2026-10-30

`b454465` (exact parent copies never accepted), seeds 1–3, 90 s from `s120`.
Time ratio = time to the control's worst final fitness, candidate / control.

Adaptive mutation, ff + mona + darwin (evidence
`.cache/quality/runs/20261030T133740Z-zahk6o2z`): r ratio 0.9947 [0.9902,
0.9991], NOISE; time ratio 0.83 [0.69, 0.98]. Median r at 90 s: ff 1.220 →
1.197, darwin 1.249 → 1.241, but mona 2.096 → 2.109 (worse).

Operator weights, ff only (evidence `.cache/quality/runs/20261030T140631Z-8sb6nxan`):

| side | change | r ratio at 90 s | time ratio | offered (M) | improvements |
| --- | --- | --- | --- | --- | --- |
| control | defaults | 1 | 1 | 395 | 46,345 |
| cut | offset 0.000565 (÷10), medium move 0.001 (÷4) | 0.9955 [0.9899, 0.9985] | 0.88 [0.65, 1.06] | 395 | 51,305 |
| struct | reorder 0.5, adjacent swap 0.5 (~1% of offspring each) | 1.0297 [1.0250, 1.0342] WORSE | 1.48 [1.00, 3.09] | 135 | 13,537 |
| colour | change colour 0.0019 (×4) | 1.0056 [1.0035, 1.0069] | 1.12 [1.00, 1.34] | 397 | 41,800 |
| combo | all three | 1.0289 [1.0239, 1.0331] WORSE | 1.48 [1.00, 3.09] | 137 | 15,584 |

- More reordering kills crossover. Uniform crossover pairs polygons by index,
  which works while chains share one polygon order; reordering scrambles it
  and crossover's improvements fall from 75% of all to 0% after 30 s. Those
  sides also evaluated a third as many offspring (larger dirty boxes, and
  crossover offspring with many changed slots blend every pixel). Reorder's
  high per-offer rate does not survive being offered more (4.5 → 0.9 per 1000).
- Cutting offset and medium moves moves the freed offspring mostly to micro
  vertex moves and gives 11% more improvements; its time ratio interval still
  includes 1.
- More full colour changes did not help: their rate fell from 0.075 to 0.030
  per 1000 offered.

## Cut, adaptive and micro step confirm from s120, 2026-10-30

`54ffd03`, ff + mona + darwin, seeds 1–3, 90 s from `s120` (evidence
`.cache/quality/runs/20261030T143916Z-95dggffl`). cut = offset 0.000565,
medium move 0.001; micro2 = `micro_adjust_delta` 0.002 (default 0.001).

| side | time ratio at 90 s | r ratio at 90 s | median r at 90 s, ff / mona / darwin | improvements, ff / mona / darwin |
| --- | --- | --- | --- | --- |
| control | 1 | 1 | 1.2203 / 2.0963 / 1.2486 | 46,477 / 65,585 / 47,330 |
| cut | 0.78 [0.65, 0.90] | 0.9912 [0.9836, 0.9971] | 1.2083 / 2.0810 / 1.2423 | 51,422 / 81,813 / 56,072 |
| adaptive | 0.83 [0.69, 0.98] | 0.9948 [0.9902, 0.9991] | 1.1965 / 2.1092 / 1.2409 | 50,784 / 66,954 / 55,134 |
| micro2 | 0.87 [0.75, 1.01] | 0.9996 [0.9940, 1.0051] | 1.2061 / 2.1177 / 1.2548 | 46,145 / 63,785 / 48,671 |
| cut + micro2 | 0.70 [0.56, 0.80] | 0.9874 [0.9780, 0.9951] | 1.2046 / 2.0786 / 1.2333 | 55,524 / 81,633 / 63,131 |

- cut and cut + micro2 beat the control on every image; adaptive and micro2
  alone lose on mona. All r ratios are NOISE against the 2% bar, which is too
  coarse from a mature start (see the characterization above).
- micro2 raises micro vertex moves' improvement rate right after the start
  (ff 0.109 → 0.223 per 1000 offered in 0–5 s) but not after 30 s (0.081 →
  0.075; mona and darwin equal or lower): larger steps pay early in the window
  and stop paying later.
- The cut sides keep 13–55 more triangles at 90 s.

## Cut + micro2 hold-out and blank check; adopted, 2026-10-30

`c5a66ca`, control vs cut + micro2 (offset 0.000565, medium move 0.001,
`micro_adjust_delta` 0.002). These became the defaults in the next commit;
the frozen v1 fixtures keep their explicit old values.

- Hold-out seeds 4–5, 90 s from `s120` (evidence
  `.cache/quality/runs/20261030T163323Z-8xt8izpr`): time ratio 0.75 [0.60,
  0.95], r ratio 0.9861 [0.9790, 0.9933] (SCREEN, two seeds). Median r at 90 s
  lower on every image: ff 1.251 → 1.218, mona 2.167 → 2.148, darwin 1.257 →
  1.248.
- From blank, 120 s, seeds 1–3 (evidence
  `.cache/quality/runs/20261030T165246Z-slntxqs6`): r ratio 0.9751 [0.9593,
  0.9929] BETTER at 120 s (0.9606 BETTER at 30 s); time ratio 0.69 [0.57,
  0.85]. mona (2.174 → 2.115) and darwin (1.325 → 1.276) gain; ff is mixed,
  worse at 5 s (2.32 → 2.89) and by median at 120 s (1.280 → 1.293, range
  1.234–1.322 against 1.254–1.283). Triangles at 120 s rise: ff 1049 → 1083,
  mona 999 → 1141, darwin 1079 → 1250.

## Step spread A/B from s120v2, 2026-10-30

`ce59872` (defaults include cut + micro2), ff + mona + darwin, seeds 1–3, 90 s
from `s120v2` (evidence `.cache/quality/runs/20261030T183441Z-bfdzw_71`).

| side | step rungs | time ratio at 90 s | r ratio at 90 s | median r at 90 s, ff / mona / darwin | improvements, ff / mona / darwin |
| --- | --- | --- | --- | --- | --- |
| control | 1 | 1 | 1 | 1.2267 / 2.0565 / 1.2110 | 44,433 / 59,206 / 44,322 |
| wide | 0.25–4 | 0.68 [0.60, 0.76] | 0.9916 [0.9869, 0.9962] | 1.2112 / 2.0396 / 1.2057 | 61,509 / 70,825 / 52,967 |
| small | 0.25–1 | 0.86 [0.71, 1.00] | 1.0015 [0.9988, 1.0043] | 1.2284 / 2.0508 / 1.2216 | 51,295 / 60,474 / 38,691 |
| mid | 0.5–2 | 0.71 [0.61, 0.85] | 0.9960 [0.9923, 0.9998] | 1.2261 / 2.0345 / 1.1955 | 53,146 / 70,188 / 48,645 |

- wide is ahead on every image; mid next; small barely differs from the control.
- Within every spread the improvement rate falls with step size in every
  window (wide on ff, 30–90 s: 0.137 per 1000 offered at 0.25×, 0.033 at 4×),
  yet the side without large steps (small) gains least, and wide's small rungs
  succeed more often than small's (0.171 vs 0.104 per 1000 at 0.25× on ff).
  Large steps win rarely but appear to open room that small steps then use;
  counts do not show how much each improvement is worth.

## Step spread hold-out and blank check; 0.25–4 adopted, 2026-10-30

`de6e700`, control (no spread) vs wide (0.25–4) and wider (0.125–8).

- Hold-out seeds 4–5, 90 s from `s120v2` (evidence
  `.cache/quality/runs/20261030T195616Z-r6xewowa`): wide time ratio 0.75
  [0.65, 0.87], r ratio 0.9896 [0.9854, 0.9938]; wider time ratio 0.49 [0.38,
  0.63], r ratio 0.9820 [0.9759, 0.9881] (SCREEN, two seeds). Both lower r on
  every image; wider lowest on all three (ff 1.184 → 1.167 → 1.158, mona 2.092
  → 2.082 → 2.079, darwin 1.241 → 1.225 → 1.207).
- Wide from blank, 120 s, seeds 1–3 (evidence
  `.cache/quality/runs/20261030T202513Z-ht8giulz`): r ratio 0.9757 [0.9577,
  0.9987] BETTER at 120 s and 0.8256 BETTER at 5 s; time ratio 0.76 [0.61,
  1.00]. ff 1.294 → 1.237 and darwin 1.276 → 1.233 gain, mona is level (2.115 →
  2.120). With fewer triangles: mona 1141 → 1064, darwin 1250 → 1169.

0.25–4 became the default in the next commit; wider was seen on two seeds only
and is the next candidate. A missing `stepSpreadMin`/`stepSpreadMax` still reads
as no spread, so the frozen v1 fixtures, earlier records and saved settings keep
their meaning (hillclimb against the previous commit: replay and lifecycle same).

## Step spread width from s120v2; 0.25–4 kept, 2026-10-30

`6ba827f` (default 0.25–4), ff + mona + darwin, seeds 1–3, 90 s from `s120v2`
(evidence `.cache/quality/runs/20261030T223504Z-wvm44md_`), wide (the default)
vs wider (0.125–8) and widest (1/16–16).

| side | step rungs | time ratio at 90 s | r ratio at 90 s | median r at 90 s, ff / mona / darwin | offered, ff / mona / darwin (M) |
| --- | --- | --- | --- | --- | --- |
| wide | 0.25–4 | 1 | 1 | 1.2112 / 2.0397 / 1.2057 | 312 / 287 / 171 |
| wider | 0.125–8 | 0.93 [0.80, 1.03] | 0.9974 [0.9926, 1.0017] NOISE | 1.2178 / 2.0473 / 1.1973 | 299 / 264 / 157 |
| widest | 1/16–16 | 0.95 [0.86, 1.06] | 0.9977 [0.9909, 1.0031] NOISE | 1.2218 / 2.0452 / 1.1996 | 274 / 235 / 147 |

- The hold-out's 0.49 for wider did not replicate: on seeds 1–3 against the new
  default, neither wider spread is distinguishable from 0.25–4, and medians go
  both ways per image.
- Wider spreads offer 4–18% fewer offspring in the same time (larger moves dirty
  larger boxes) and win that back with a higher improvement rate per offspring;
  the two roughly cancel.
- The largest rungs win least (widest on darwin, 30–90 s: 0.037 per 1000 at 16×
  vs 0.123 at 1/16×), and below about 0.3× the rate stops rising.

0.25–4 stays the default; no blank check was run since no candidate survived.

## Improvement gain per operator and rung from s120v2, 2026-10-30

`3500b9c` (defaults, spread 0.25–4), ff + mona + darwin, seeds 1–3, 90 s from
`s120v2` (evidence `.cache/quality/runs/20261030T233322Z-5fbt7995`). Counts
match the wide side of the width A/B exactly, so the gain tally does not
change the search. `per offer` is gain per offspring relative to the table's
average; ff / mona / darwin.

| operator | offered% | gain%, ff / mona / darwin | per offer |
| --- | --- | --- | --- |
| crossover | 20.0 | 74.9 / 75.0 / 75.0 | 3.74 / 3.75 / 3.75 |
| micro_move | 23.8 | 14.3 / 15.2 / 14.0 | 0.60 / 0.64 / 0.59 |
| change_color | 5.4 | 2.5 / 3.6 / 3.0 | 0.46 / 0.67 / 0.56 |
| medium_move | 8.5 | 3.5 / 2.8 / 3.8 | 0.41 / 0.33 / 0.45 |
| move_point | 2.5 | 0.7 / 0.4 / 1.1 | 0.29 / 0.17 / 0.43 |
| micro_color | 31.7 | 2.9 / 1.8 / 2.0 | 0.09 / 0.06 / 0.06 |
| saturation + brightness | 5.1 | 0.3 / 0.1 / 0.3 | ≤ 0.07 |
| offset | 1.6 | 0.2 / 0.1 / 0.1 | 0.10 / 0.06 / 0.06 |
| reorder | 0.05 | 0.2 / 0.2 / 0.2 | 29 / 44 / 38 |

- Step rungs: worth per offspring is nearly flat, 0.81–1.24 across 0.25×–4×.
  Large steps win 2–4× less often, but each win is worth 1.4–1.9× the average.
  This is why the spread's width barely mattered.
- micro_color makes 32% of offspring for 2–3% of gain. It also makes 26–38% of
  neutral acceptances, whose value gain cannot see.
- Crossover's 3.75 includes gains another chain's operators found first.
- Reorder is the most valuable per offspring, at a very small offered share
  (reorder probability 0.002). An earlier sweep found that more reordering
  breaks crossover's slot alignment, so its value is not linear in its weight.
