# Reject mutation bbox reuse without a measured gain

Reject this candidate. The completed-engine candidate/reference ratio is
0.999530, with conditional 95% interval [0.996403, 1.002741]. This shows no
measurable throughput benefit. Both aggregate gain requirements fail: the
ratio is not above 1.005 and the lower bound is not above 1. All case regression
guards pass. No extra samples, 368-fixture checker, reserved run or Rust build
follows. The ordinary engine and every evolution parameter remain unchanged.

| Case | Paired ratio median | Minimum to maximum | 95% interval for paired geometric mean |
| --- | ---: | ---: | ---: |
| ff | 0.999028 | 0.994307 to 1.008228 | 0.995496 to 1.005259 |
| mona | 0.995924 | 0.990015 to 1.007553 | 0.993386 to 1.004421 |
| darwin | 0.998241 | 0.991373 to 1.011974 | 0.994065 to 1.006032 |

Each case has five alternating pairs, 64 measured batches per arm, seed 7 and
eight warmups. All fifteen pairs remain included. Statistics equally weight the
cases and use 10,000 stratified paired-log bootstrap draws with seed 20261024.
The interval is conditional on same-seed timing repeats. It does not cover
independent search seeds, reserved drawings or other hardware.

The [source patch](attempts/18-mutation-bbox-reuse.patch) captures the three original
packed vertex words in the final per-polygon path of `single_mutate_offspring`.
After the original mutation, repack and polygon store, it compares the resulting
words. Exact equality returns the existing bbox. A changed word follows the
original new-bbox calculation and union. Color, vertices, polygon writes and RNG
calls keep their original order. Empty drawings, earlier returns, crossover and
multi-mutation remain unchanged.

This does not assume that vertex unpack/repack round trips exactly. Equal packed
words give the bbox helper identical inputs; its integer min/max union is
idempotent. The source inverse restores every G1 byte, and the archived patch
reconstructs the candidate without fuzz. Candidate SHA256 is
`f3dd47b2dc6c82e12c925c4d4cb41b2f5aeda28a016dc246796a61d9d17988dc`.
No shared allocation, barrier, dispatch or parameter changes. Capturing three
private words, testing equality and branching may cost more than the removed
calculation. Guard frequency and compiled instruction savings remain unmeasured;
no helper-level speedup is established.

The [frozen protocol](profiles/mutation-bbox-reuse-spec.json), committed before
execution in d0c0e34, uses the existing immutable prepared release and hot mutation
installer. Actual source and trial records confirm G1, WG64, automatic mutation
initialization TRUE and finalization overlap FALSE for both arms. Binning and
raster sources retain their accepted identities. Earlier A/A64 supplies calibration
and GPU identities only; it is not fresh noise evidence and contributes no rates.

The [training runner](attempts/20-bbox-training-runner.patch) preserves the complete
matrix gate and deadline. The [stage contract](attempts/19-explicit-shader-stage-contract.patch)
checks bin/raster/mutation provenance and requires explicit G1/TRUE metadata for
mutation sources. Thirty CPU tests pass, including the original 26 and four
[stage regressions](attempts/21-shader-stage-contract-tests.patch). Synthetic
converted wire fixtures are not native mutation qualification. The initial test
discovery skipped hyphenated filenames; the explicit loader ran all 30. Scoped
Ruff passes without suppressions. The cohesive phase validator is 62 physical
lines after explicit stage propagation, preserving its bounded association logic.

## Work and verification

One native process exits zero after 275.048 seconds. The controller takes 277.393
seconds, within its 1,800-second budget. Thirty timing trials complete 31,457,280
evaluations. Six unscored diagnostics add 6,291,456 and three reference
qualifications add 3,145,728. Thirty-nine successful preparations account for
5,111,808 warmup evaluations, giving a source-accounted logical lower bound of
46,006,272. Full-fitness/reset and other setup GPU work stay separate.

The parent offline audit passes all 522 predicates. It reuses the unchanged,
hash-pinned byte-L1, f32 fitness and independent bootstrap functions from the
previous independent audit. Eighteen initial/final snapshots reproduce their
seven-field canonical hashes, both diagnostic arms replay their reference
qualification, and all 72 parents match exact cached-versus-framebuffer RGB-byte
L1. Maximum independent f32 fitness difference is zero under the existing 2e-5
policy. No scalar rerender or new independent reviewer is claimed.

All 30 native rows retain correct work counters, parameters, source identities,
execution options and 64 Stages samples. Rates equal completed work divided by
elapsed time. Protected hashes preserve 78 runtime source files, three ordinary
binaries and two comparison helpers. The raw output inventory remains unchanged.
The audit retains its actual executed source; subsequent lint-only cleanup changes
an unused variable name and the equivalent datetime.UTC alias, with no remeasurement.

Completed-engine timing includes readback and final drain. Preparation, source
installation and oracle export remain outside. Median sampled mutation averages
are 90.651 versus 90.481 microseconds for ff, 108.915 versus 108.858 for mona and
99.512 versus 99.484 for darwin. These medians summarize five trial averages, each
sampling only 64 final iterations. They do not establish a stage speedup, guard
frequency, hardware limiter or whole-run shares. The compiler may already remove
the duplicate work, or the guard's cost may offset it; neither cause was measured.

The host reported load average 0.01 on 16 logical CPUs at launch, with no other
known benchmark/build. WSL host/GPU background isolation is unverified. The
candidate never advanced to the direct 368-fixture or reserved-data gates and
has no broader runtime qualification. Its rejection is final for this cohort.

The [machine record](profiles/mutation-bbox-reuse-result.json) and
`.cache/evolution-mechanism-20261024/` retain rows, snapshots, clocks, the actual
command, source proof, setup distributions, CPU logs and rerunnable offline checks.
Prove It Works drove raw state/error recomputation. Attack the Premise selected a
different mechanism from the failed cache layouts. Explain the Number prevents a
claim of savings from source-level redundancy alone.
