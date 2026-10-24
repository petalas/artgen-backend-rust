# Rejected dirty raster dispatch compaction, tool removed

Historical disposition: both frozen short filters failed and the compaction family
is stopped. Neither variant was adopted. The experimental runtime, lab flag and
mode fields, source pin, resources, APIs, generated shader fragments and
`check-gpu-raster-grid` example have been removed. Commit `83999f8` introduced the
default-off prototype and remains in Git. The description below records its behavior
and evidence before removal; it does not describe an available checkout mode.

The final scoped preimages, SHA-256 inventory and full scoped patch are preserved in
`.cache/evolution-overnight-20261024/compact-rejected-final/`. That patch is against
`83999f8^` and contains the current uncommitted mutation-group context from tasks 25
and 31. It is not a pure compaction delta. The cleanup preserved that group feature,
the ordinary a292/e133/598 shader bytes, all parameters and RNG, and the ordinary
two-pass bulk/four-pass sampled schedule with its original RW-to-RW boundaries and
eight timestamp meanings. The current lab rejects `--compact-raster` as unknown.

This default-off benchmark mode changed scheduling only. It built a prefix of
consumed tile counts from the current offspring headers, then indirectly dispatched
those workgroups. Binning, raster arithmetic, selection, mutation, RNG streams and
complete MutationParams remained unchanged. The ordinary full-image fitness/reset
path retained its original raster pipeline.

`benchmark_set_compact_raster(bool)` drained before toggling and lazily created
bounded persistent prefix/argument buffers. The WG1 prefix builder rewrote every
active boundary and all three scalar indirect arguments each iteration. Consumption
used the rasterizer's strict overlap rule, including clipped edges, out-of-image
controls, interior zero-width and some inverted boxes. It did not reinterpret
those boxes as empty merely because their minima reach their maxima.

A balanced 2D indirect grid respected the 65,535 limit per axis. For 73,728 groups
it used 36,864 by 2. A uniform return handled any padded final group; zero total
used arguments 0/1/1. The group leader found the owning offspring with a bounded
strict upper-bound search, skipped duplicate prefix boundaries, then published
its tile mapping at the existing metadata barrier. Pixel coverage, original bin
strides, decoded polygon loading, blend order, byte errors and subgroup reduction
retained their accepted-C implementation.

Bulk encoding used mutate/bin/prefix, indirect raster, then select/update passes.
The sampled iteration kept the four existing timestamp pairs and inserted an
untimestamped prefix pass. Wall time and the batch envelope included that overhead;
stage samples alone could not establish its whole-run cost. Indirect validation stayed
enabled. Logical evaluations and completion accounting remained unchanged.

The historical resident lab accepted `--compact-raster true` only with the raster stage and
exact accepted C bytes for **both** the reference and every candidate. The reference
used ordinary dispatch; candidates used compact dispatch. The mode was recorded from
the actual evolver in each normal and diagnostic row. Preflight rejected a different
but numerically equivalent reference before GPU construction. This guard was added
after source review identified a possible baseline-attribution error.

The direct main-thread `check-gpu-raster-grid` example checked actual generated
prefix and compact sources against ordinary C. Its runtime proof used 75 fixtures across all five
raster workgroups, representative lambdas, retained histories, zero/shrinking/growing
prefixes, partial and unusual bounds, and totals 65,534/65,535/65,536/65,537/73,728.
All original tile visits had to match the independent strict predicate exactly once.
Both error accumulators also had to match before selection cleared them. Its 225
explicit compute passes were checker work, not evolution evaluations.

The release checker passed all 75 fixtures. Two separate injected missing/duplicate
faults each exited with the expected failure and a retained `negative_proof` record.
Their old/new error arrays stayed equal, so coverage itself rejected them. The
actual engine then passed a one-batch ff reference/compact smoke, two normal and two
overflow executions, with exact canonical state, byte-L1, fitness and completed work.
Those timings are integration evidence only. The subsequent two-pair, eight-batch
filter rejected WG1 compaction: aggregate ratio **0.659089**, conditional interval
**[0.655904, 0.662290]**. ff/mona/darwin median paired ratios were
0.619353/0.684103/0.675772. All twelve normal workers and six diagnostic replays
passed, with 1,572,864 normal evaluations. No pair was excluded and no full-budget
confirmation was run. Its exact aggregate ratio was **0.6590888793067157**, with
conditional interval **[0.6559036150329431, 0.6622896121771861]**, recorded in
`.cache/evolution-overnight-20261024/compact-raster-screen-01/output/analysis.json`.

The second frozen filter used the WG64 prefix builder and also failed. Its aggregate
ratio was **0.6537716356593324**, with conditional interval
**[0.6460889295408405, 0.6615456975813435]**, taken directly from
`.cache/evolution-overnight-20261024/compact-prefix64-screen-01/output/analysis.json`.
Its summary recorded all twelve normal workers and six diagnostic replays valid,
zero errors and 1,572,864 normal evaluations. Neither filter advanced or qualified
for confirmation. These were two-pair, eight-batch exploratory filters with
same-seed timing repeats, not independent search seeds or full-budget performance
confirmations. No new timing or profiler run was performed for this cleanup.

The WG64 prefix passed 75 runtime coverage/error fixtures in
`compact-prefix64-proof-v1/` using archived build v7. The retained missing/duplicate
fault evidence is in `raster-grid-proof-v2/`; the WG64 positive proof did not rerun
those injections. The final prefix preimage SHA-256 is
`e80a9e518cfddd810a412fc86f71379863fe35547af074cdf05d983981c81010`.
The final grid checker preimage SHA-256 is
`b3ab63164ea063d1065f77507a1386ca48bd58e86bfa19ae9985b705a2967dca`.
That checker had 79 cases and only CPU tests for that version. The four cases added
after the 75-case v7 runtime proof were **unrun on GPU**. Removal does not upgrade
that proof to 79 cases.

Historical CPU verification passed the library tests, generated WGSL validation across all
five workgroups, lab preflight tests and seven checker tests. Check/Clippy reported
no diagnostics in the new module, lab or checker; existing repository warnings
remain. Source review found no other material issue, and a fresh review confirmed
the reference-source guard correction.

Evidence is retained under `.cache/evolution-overnight-20261024/` in
`raster-grid-proof-v2/`, `compact-raster-smoke-01/`, `build-v5/` (checker) and
`build-v6/` (corrected lab), `build-v7/` (WG64 prefix lab/checker),
`compact-raster-screen-01/`, `compact-prefix64-proof-v1/` and
`compact-prefix64-screen-01/`. `raster-grid-proof-v1/` retains a launcher failure before
GPU execution: the immutable-artifact writer was incorrectly used for mutable
checkpoint state. The replacement preserves the same checker and fixtures.

Power of Ten rule 4 deviation during the prototype: the 65-line paired bind-group
descriptor table stayed together so its matching layouts were directly auditable.
The existing large constructor and batch encoder retained explicit resource/pass
order; the added path was limited to the documented three/five-pass schedule.
Fixed buffers, checked capacities, bounded searches, source preservation,
75 physical coverage/error cases and actual-loop replay were the compensating
controls. The failed filters settled the disposition by removing the experimental
path. No default enablement followed.

After source freeze, one empty EOF line was removed from the prologue fragment to
satisfy whitespace checks. Token order was identical, the final newline remained,
and the focused generated-source tests passed again. The exact before/after hashes
and archived preimage are in `compact-prologue-format-proof.json`. Existing GPU
proofs retain their original binaries and source inventories.

The v6/v7 binaries, build commands, launch commands, raw caches and historical
records remain unchanged. The exact filter arguments remain in each screen's
`launch.json`; build commands remain in `build-v6/command.json` and
`build-v7/command.json`. Replays must use the archived binaries, not the current
lab, and a fresh output directory. For example, from the repository root:

```sh
timeout 300 .cache/evolution-overnight-20261024/build-v6/engine-lab \
  --manifest .cache/evolution-bench/baseline-20261023T172054.919294Z-95742/fixtures/manifest.json \
  --output .cache/evolution-overnight-20261024/compact-wg1-replay-NEW \
  --shader-stage raster \
  --reference-shader .cache/evolution-overnight-20261024/build-v6/source/src/shaders/rasterize_error.wgsl \
  --variant compact=.cache/evolution-overnight-20261024/build-v6/source/src/shaders/rasterize_error.wgsl \
  --compact-raster true --pairs 2 --batches 8 --diagnostics true

timeout 300 .cache/evolution-overnight-20261024/build-v7/engine-lab \
  --manifest .cache/evolution-bench/baseline-20261023T172054.919294Z-95742/fixtures/manifest.json \
  --output .cache/evolution-overnight-20261024/compact-wg64-replay-NEW \
  --shader-stage raster \
  --reference-shader .cache/evolution-overnight-20261024/build-v7/source/src/shaders/rasterize_error.wgsl \
  --variant compact=.cache/evolution-overnight-20261024/build-v7/source/src/shaders/rasterize_error.wgsl \
  --compact-raster true --pairs 2 --batches 8 --diagnostics true
```

These are historical reproduction commands, not an instruction to run during the
parent's timing window. After cleanup, 31 focused GPU-evolver library tests, 12 resident-lab tests and
seven mutation-checker CPU tests passed. All/pedantic Clippy completed successfully
with no errors; legacy diagnostics remain. Independent source review confirmed
the ordinary pass/timestamp schedule and source-coupled mutation dispatch were
preserved. Fresh release and runtime evidence use the separate v8 archive.
