# Compare exact dirty raster dispatch compaction

This default-off benchmark mode changes scheduling only. It builds a prefix of
consumed tile counts from the current offspring headers, then indirectly dispatches
those workgroups. Binning, raster arithmetic, selection, mutation, RNG streams and
complete MutationParams remain unchanged. The ordinary full-image fitness/reset
path retains its original raster pipeline.

`benchmark_set_compact_raster(bool)` drains before toggling and lazily creates
bounded persistent prefix/argument buffers. The WG1 prefix builder rewrites every
active boundary and all three scalar indirect arguments each iteration. Consumption
uses the rasterizer's strict overlap rule, including clipped edges, out-of-image
controls, interior zero-width and some inverted boxes. It does not reinterpret
those boxes as empty merely because their minima reach their maxima.

A balanced 2D indirect grid respects the 65,535 limit per axis. For 73,728 groups
it uses 36,864 by 2. A uniform return handles any padded final group; zero total
uses arguments 0/1/1. The group leader finds the owning offspring with a bounded
strict upper-bound search, skipping duplicate prefix boundaries, then publishes
its tile mapping at the existing metadata barrier. Pixel coverage, original bin
strides, decoded polygon loading, blend order, byte errors and subgroup reduction
retain their accepted-C implementation.

Bulk encoding uses mutate/bin/prefix, indirect raster, then select/update passes.
The sampled iteration keeps the four existing timestamp pairs and inserts an
untimestamped prefix pass. Wall time and the batch envelope include that overhead;
stage samples alone cannot establish its whole-run cost. Indirect validation stays
enabled. Logical evaluations and completion accounting remain unchanged.

The resident lab accepts `--compact-raster true` only with the raster stage and
exact accepted C bytes for **both** the reference and every candidate. The reference
runs ordinary dispatch; candidates run compact dispatch. The mode is recorded from
the actual evolver in each normal and diagnostic row. Preflight rejects a different
but numerically equivalent reference before GPU construction. This guard was added
after source review identified a possible baseline-attribution error.

The direct main-thread `check-gpu-raster-grid` example checks actual generated
prefix and compact sources against ordinary C. It uses 75 fixtures across all five
raster workgroups, representative lambdas, retained histories, zero/shrinking/growing
prefixes, partial and unusual bounds, and totals 65,534/65,535/65,536/65,537/73,728.
All original tile visits must match the independent strict predicate exactly once.
Both error accumulators must also match before selection clears them. Its 225
explicit compute passes are checker work, not evolution evaluations.

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
confirmation was run. The runtime default remains ordinary. The validated default-off
benchmark seam is retained to test a narrower prefix-construction hypothesis.

CPU verification passed the library tests, generated WGSL validation across all
five workgroups, lab preflight tests and seven checker tests. Check/Clippy reported
no diagnostics in the new module, lab or checker; existing repository warnings
remain. Source review found no other material issue, and a fresh review confirmed
the reference-source guard correction.

Evidence is retained under `.cache/evolution-overnight-20261024/` in
`raster-grid-proof-v2/`, `compact-raster-smoke-01/`, `build-v5/` (checker) and
`build-v6/` (corrected lab) and `compact-raster-screen-01/`. `raster-grid-proof-v1/` retains a launcher failure before
GPU execution: the immutable-artifact writer was incorrectly used for mutable
checkpoint state. The replacement preserves the same checker and fixtures.

Power of Ten rule 4 deviation: the new 65-line paired bind-group descriptor table
stays together so its matching layouts remain directly auditable. The existing
large constructor and batch encoder retain explicit resource/pass order during
this finite prototype; the added path is limited to the documented three/five-pass
schedule. Fixed buffers, checked capacities, bounded searches, source preservation,
75 physical coverage/error cases and actual-loop replay compensate. The attempt's
keep/reject decision will determine whether to deepen the encoder seam or remove
this experimental path. No default enablement or performance claim follows here.

After source freeze, one empty EOF line was removed from the prologue fragment to
satisfy whitespace checks. Token order is identical, the final newline remains,
and the focused generated-source tests pass again. The exact before/after hashes
and archived preimage are in `compact-prologue-format-proof.json`. Existing GPU
proofs retain their original binaries and source inventories.
