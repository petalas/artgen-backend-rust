# Engine architecture verification

The four refactors preserve the evolutionary rules and all six shader files.
Capacity and dispatch planning, installed shader programs, pending readback and
trial qualification each have one owner. The first three commits are `8ae2145`,
`dcca024` and `f71bf66`; this record accompanies the qualification commit.

## Correctness and artifact identity

- 152 Rust tests and 137 Python tests pass.
- The normal application and benchmark/lifecycle examples build in release mode.
  All-target compilation and all/pedantic Clippy complete. Existing repository
  diagnostics remain; the new engine modules and GPU checkers have no diagnostics.
- The reference executable exactly matches the previously qualified
  [release record](../benchmarks/evolution/continuation-20261024/profiles/pixel-pair-release.json),
  SHA-256 `1cf30fea6f7cb91f645b192c9b6a148fa500196eef1d463ebd17701e884a2fa2`.
- All six complete shader strings match the frozen reference in the final normal
  application, evolution worker, resident lab and lifecycle executables.
- The main-thread program checker rejects a late framebuffer-program failure,
  retains all six installed handles and tile metadata, then installs a valid tile.
  The raster checker passes all five shapes, odd edges and its numerical fixtures.
- Six ordinary lifecycle runs match byte for byte before and after readback
  ownership and again after qualification integration. They cover both timestamp
  modes, pending reset/repreparation, drawing delivery, final maximum-parent
  association and submitted/completed work.
- Fresh native unscored qualification output passes its real Python consumer and
  saved-state checks, and the timing validator rejects it. JSON protocols and
  independent numerical oracles remain unchanged.

## Original 64-batch comparison

The 33-worker matrix contains five alternating reference/candidate pairs per
image and three separate candidate overflow replays. All workers complete with
zero errors and exact initial/final canonical replay, settings, device identity,
quality and work accounting. Timed work totals 31,457,280 completed evaluations;
separate diagnostics add 3,145,728.

Each timing run completes 64 batches, 4096 iterations and 1,048,576 evaluations
with seed 7 and two warmups. Rates include ordinary batches, required readback
and checked final drain. Preparation, compilation, snapshots and numerical
qualification stay outside the clock. Cache policy and evolution parameters
match between arms. Runs are serial under the benchmark lock.

Rates are evals/sec, median [minimum, maximum] of five runs per arm. Ratios use
paired geometric means and conditional 95% intervals, rather than dividing the
rate medians.

| Drawing | Reference | Refactor | Paired ratio [95% interval] |
| --- | ---: | ---: | ---: |
| ff | 319,505 [317,185, 324,150] | 320,717 [316,067, 321,690] | 0.998645 [0.986119, 1.007849] |
| mona | 275,540 [274,654, 277,168] | 275,895 [275,606, 276,345] | 1.000122 [0.997218, 1.002889] |
| darwin | 269,617 [267,657, 270,483] | 276,706 [268,759, 279,003] | 1.019196 [1.004820, 1.032880] |

The equal-image aggregate is 1.005944 [0.999846, 1.011552]. It supports aggregate
non-regression at a 1% margin and does not establish an aggregate speedup.
The matrix does not establish strict +/-1% equivalence: ff's lower bound crosses
0.99 and the aggregate upper bound exceeds 1.01. Every sample remains retained.
The existing optimization policy's 3% gain screen remains separate from this
structural refactor's performance question; its original verdict is unchanged.

## Separate longer ff comparison

The separately frozen followup uses five alternating pairs at 256 batches,
seed 7 and two warmups, plus one separate diagnostic replay. It addresses the
original ff interval's uncertainty over a longer engine epoch. It is not pooled
with, substituted for, or used to relabel the 64-batch matrix.

All 11 workers pass with zero errors and exact replay. Timed work totals
41,943,040 evaluations; the separate diagnostic adds 4,194,304. Reference rate is
336,607 [333,957, 339,205] evals/sec; candidate rate is
337,701 [335,232, 338,826], median [minimum, maximum] of five runs per arm.
The paired geometric ratio is 1.003282 [0.995714, 1.009226]. It clears the
predeclared 0.99 lower-bound gate and fits within +/-1% at this longer budget.
It does not establish a speedup. The original 64-batch ff interval remains
inconclusive for that margin.

## Reserved replay and attribution

Nine separate unscored workers compare reference and candidate on the existing
reserved mature drawings with seeds 11, 13 and 17, including candidate overflow
replays. All complete with exact replay and zero errors. A tenth unscored worker
profiles the full Mona compute envelope. Its 64 spans cover 3.030818 seconds of a
4.452231-second measured engine epoch, 68.1%. This is a queue compute interval,
including GPU stalls, not hardware utilization or recoverable CPU time.
Profiles and reserved replay rates do not contribute to either timing cohort.

These results describe RTX 5090 through WSL Dozen/Vulkan in the recorded session.
Background GPU isolation remains unverified. Repeated timing seeds do not prove
independent search quality or behavior on other backends. Startup, live viewer
latency and broader optimization adoption are outside these engine measurements.

## Independent audit

A separate read-only audit passes with no blocking findings. It verifies 160
saved state artifacts, 640 parent numerical associations and all 54 serial worker
spans, and independently reproduces both bootstrap calculations. All frozen
source/binary identities and actual release settings match. The audit recomputes
RGB-byte-L1 and IEEE f32 fitness from saved parent pixels; it checks the native
scalar-policy records without rerunning the scalar renderer.

The original ff64 uncertainty, separate ff256 conclusion, unverified GPU isolation
and absence of an established aggregate speedup remain explicit limitations.

## Local evidence and method

Raw workers, source/binary freezes, specifications, reviews and the append-only
trail are under `.cache/architecture-refactor-20261024/`. `comparison/summary.json`
retains the original matrix; `ff-followup/summary.json` is a separate cohort.
`reserved-summary.json`, `qualification-proof.json`, `readback-proof.json` and
`final-check-summary.json` retain the correctness/attribution evidence.

Foundational Thinking put related state in owned types. Laziness Protocol removed
parallel bookkeeping and duplicated rules without generic cache abstractions.
Test Behavior, Not Implementation selected ordinary engine returns and public
installation results as the test surface. Prove It Works required real executable
and GPU evidence. Explain the Number kept completed work, setup, diagnostics and
conditional timing inference separate.
