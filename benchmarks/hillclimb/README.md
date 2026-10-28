# Hill-climb campaign, 2026-10-28

Metric: completed evaluations/s on the frozen v1 fixtures (ff, mona, darwin;
384 px, 4 chains, λ 64, 64 iterations/batch), equal-case geometric mean of
paired ratios. Behavior changes are judged on fitness gained per timed second.
Every row of [`log.tsv`](log.tsv) is one `scripts/hillclimb.py` run; see
[docs/hillclimb.md](../../docs/hillclimb.md) for the method.

## Engine changes

| Commit | Change | Measured when kept | Ablation after round 4 (256 batches) |
| --- | --- | --- | --- |
| `30b3ca3` | Crossover spread over the 64 mutate lanes | 1.1385 [1.1328, 1.1438] | conflicts with `bde7fbd`; not ablated |
| `344eb99` | Framebuffer update reads the winner's tile lists | 1.0905 [1.0848, 1.0961] | revert 0.7629 |
| `f7ec23f` | Double-buffered best-drawing readback | 1.0238 [1.0201, 1.0277] | revert 0.7880 |
| `bd2f732` | Raster skips clean tiles without a barrier | 1.0150 [1.0078, 1.0233] | revert 0.9916 |
| `bde7fbd` | Crossover partners drawn from active chains (bug fix, behavior change) | quality 1.3446 [1.2947, 1.3990]; 512 batches 1.2777 | — |
| `352951f` | One command buffer per batch (undoes `70c9031`) | 512 batches 1.2541 [1.2452, 1.2630]; 64 batches 1.0236 | — |

`70c9031` (4-iteration chunked submits, +13.7% when kept) was later shown to
lock the driver into a slow submit state once readback was double-buffered and
was removed by `352951f`. Round-2 ratios were measured before servers were
restarted per case; the ablation and cumulative rows use the corrected harness.

Cumulative against the session start (`f76e8cd`, one fresh process per trial),
quality mode because evolution changed:

| Horizon | evals/s ratio | Fitness gain per second |
| --- | --- | --- |
| 64 batches × 5 pairs | 1.3969 [1.3942, 1.3996] | 1.9497 [1.8779, 2.0283] |
| 512 batches × 3 pairs | 1.4362 [1.4283, 1.4448] | 1.7340 [1.7015, 1.7666] |

Master runs at about 516k / 418k / 415k evals/s (ff / mona / darwin) at 64
batches, against 358k / 301k / 304k at the start.

## Rejected

- Exact dirty region for crossover offspring: replay identical, raster time
  unchanged (0.9932, NOISE). Its premise (25% self-crossover) assumed the
  tournament sampled active chains; it sampled the 1024-slot buffer instead,
  which led to `bde7fbd`.
- Chunk sizes 1, 2, 8, 16 against 4 (before `352951f`); deduplicating the params
  `set_bind_group` calls (no change in `finish` time).

## Harness lessons, in the order they were found

1. Engine construction (NVIDIA shader compile, ~11 s) dominated per-trial
   processes: resident `serve` mode.
2. Host load on the Windows side (a game, video) produced a false 3.4% SLOWER A/A.
3. A candidate's first trial after construction could stall: unscored priming.
4. Sweeps rebuilt and re-ran the control: multi-candidate runs, `--build-only`.
5. Script and doc commits forced engine rebuilds: cache keyed on build inputs.
6. Parallel agents could not share fixtures and the lock: state resolved via the
   git common dir; builds and timing serialized.
7. Parallel cargo builds biased two screens and the load average missed them:
   per-trial foreign-CPU measurement.
8. Long-lived servers carried per-process driver drift (±11% A/A offsets):
   fresh servers per case.
9. Behavior changes need a quality metric: fitness gain per second.
10. 64-batch screens hid a 25–42% long-run regression: confirm host-side
    changes at 512 batches.
11. A kept change turned harmful later: `revert:SHA` ablation after each round.

## Remaining ideas

- Raster is ~80% of GPU time (400–520 µs/iteration); prior campaigns already
  mined tiling, binning and compaction. The next raster gain likely needs a
  different algorithm (for example, caching per-tile parent prefixes).
- `src/auto_tune.rs` tunes mutation parameters at runtime. Its tuning predates
  the crossover fix, so crossover probability and tournament size may now have
  better values.
- `finish` still grows slowly in long processes (12 to 14.5 ms per batch over
  4096 batches); it is hidden while host work per batch stays below GPU time.
