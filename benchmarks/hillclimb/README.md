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
| `554d828` | Skip blending pixels no changed polygon covers (single mutations) | confirm 1.1174 / 1.1161 | — |
| `3f8e9e4` | Delivered chain chosen from fitness readback (determinism fix) | 512 batches 1.0002 | — |
| `dd7e1ad` | Bitmask binning of dirty tiles, 4 row bands per offspring | confirm 1.0530 / 1.0630 | — |
| `15519ce` | Unchanged-pixel skip for crossover offspring (≤4 changed slots) | confirm 1.7380 / 1.8355, lifecycle same | — |
| `54286cc` + `50854a0` | Unchanged-pixel skip in framebuffer updates; raster tiles no changed polygon meets exit early | stacked confirm 1.2257 / 1.2027, lifecycle same | `54286cc` revert 0.8994 (512 batches, after round 7); `50854a0` code already removed by `c93ba70` |
| `8e383a9` | Each batch finished as 4 command buffers on parallel threads, one submit | confirm 1.0354 / 1.1339; gpu% 82–89% to 97–99% | — |
| `c93ba70` | Raster loops over a worklist of tiles with work, built by the binner; fixed 8192-workgroup grid | confirm on top of `8e383a9` 1.2575 / 1.3018 | — |

Confirm ratios are 64 batches × 5 pairs / 512 batches × 3 pairs. In crossovers
between real chains, 38–53% of offspring are exact copies of parent A and
almost all differ in at most 4 slots, which is why `15519ce` is so large.

`54286cc` keeps the parent versions of a winner's changed polygons in polygon
slots 996–999 of the chain (past every drawing's count, outside the replay
hash); the skip turns itself off for drawings above 996 polygons. Code that
reads past `polygon_count` or raises the polygon limit must account for it.

`70c9031` (4-iteration chunked submits, +13.7% when kept) was later shown to
lock the driver into a slow submit state once readback was double-buffered and
was removed by `352951f`. Round-2 ratios were measured before servers were
restarted per case; the ablation and cumulative rows use the corrected harness.

Cumulative against the session start (`f76e8cd`, one fresh process per trial),
quality mode because evolution changed:

| After | Horizon | evals/s ratio | Fitness gain per second |
| --- | --- | --- | --- |
| round 4 | 64 batches × 5 pairs | 1.3969 [1.3942, 1.3996] | 1.9497 [1.8779, 2.0283] |
| round 4 | 512 batches × 3 pairs | 1.4362 [1.4283, 1.4448] | 1.7340 [1.7015, 1.7666] |
| round 5 | 64 batches × 5 pairs | 2.9726 [2.9622, 2.9826] | 4.1490 [4.0023, 4.3110] |
| round 5 | 512 batches × 3 pairs | 3.3511 [3.3249, 3.3761] | 4.0459 [3.9685, 4.1222] |
| round 7 | 64 batches × 5 pairs | 4.5272 [4.4536, 4.5938] | 6.3187 [6.0869, 6.5563] |
| round 7 | 512 batches × 3 pairs | 5.3296 [5.3044, 5.3512] | 6.4346 [6.3089, 6.5534] |

Master runs at about 1.63M / 1.48M / 1.30M evals/s (ff / mona / darwin) at 64
batches and 1.81M / 1.60M / 1.50M at 512, against 340–359k / 293–301k /
291–303k at the start. A16 alone read NOISE at 512 batches because its branch
predated `8e383a9`: the faster GPU made the single-threaded `finish` the limiter.

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
12. Every adoption needed two horizons by hand: `--confirm`.
13. The foreign-CPU warning did not stop parallel agents' builds from skewing
    runs: disturbed pairs are re-timed automatically.
14. Returned-drawing delivery had become racy (fixed in `3f8e9e4`); agents ran the
    lifecycle checker by hand: `--lifecycle`, implied by `--confirm`.
15. Immediate retries landed inside the same multi-minute build: retries wait
    for a quiet host.
16. Agents' tables showed only candidate stage times: control→candidate columns.
17. An agent judged a branch that lacked a newer master commit and the verdict
    flipped: candidates missing control commits are warned about and logged.
18. A revert whose code a later change had already removed silently ran an A/A:
    no-op reverts are skipped.

## Remaining ideas

- Per iteration at 512 batches: mutate ~10, bin ~28–30, raster ~80–107,
  select+update ~22–25 µs. Raster is still the largest stage, now almost only
  pixels a changed polygon covers. Bin rose by ~4 µs to build the worklist.
- At ~1.8M evals/s, `finish` on 4 threads (~4.5 ms per batch) leaves the host
  ~6 ms ahead of a ~9 ms GPU batch; further GPU gains will bring the host back
  as the limiter.
- `src/auto_tune.rs` tunes mutation parameters at runtime. Its tuning predates
  the crossover fix, so crossover probability and tournament size may now have
  better values.
- `finish` still grows slowly in long processes (12 to 14.5 ms per batch over
  4096 batches); it is hidden while host work per batch stays below GPU time.
