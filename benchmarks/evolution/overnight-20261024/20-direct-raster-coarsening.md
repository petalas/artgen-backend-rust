# Direct raster tile grouping prototype

This benchmark prototype processes two or four adjacent logical tiles in each
directly dispatched workgroup. The production shader stays exact AE1 until the
separate performance and reserved gates pass. No prefix, indirect dispatch,
additional production buffer, compute pass or evolution parameter is introduced.
The stopped indirect family supplies no evidence for this implementation.

The source's bounded RASTER_TILES_PER_GROUP literal selects 1, 2 or 4; an absent
marker means 1. It travels with source, pipeline and logical workgroup through
construction, cache hits, preparation and diagnostics. All three raster dispatch
sites use the actual helper. Both framebuffer updates retain the full grid.
Logical 32x16 tiles, every valid consumed pixel, decoded geometry, original float
helpers, layer order, byte quantization, integer reductions and logical evaluation
counts remain unchanged. Actual execution shape is recorded outside MutationParams
and GPU identity.

Each group computes an immutable strict-overlap mask for adjacent original tiles,
including padding guards. A workgroup-uniform load permits uniform skip paths.
Consumed tiles reuse the decoded cache. The old reduction barrier completes every
prior polygon read before the next cache writes; the following load barrier waits
for lane zero's prior error reads before any next error writes. Source review found
no material synchronization, installation or coordinate issue. No extra iteration
barrier was inserted to mask an unproved ordering assumption.

The source snapshot and GPU proof use the factory's exact specialized source,
pipeline, workgroup, T and compilation options. Vulkan keeps its existing FALSE
initialization policy. This prevents the earlier compact-factory policy mismatch.
Production G1 and binner 598 remain byte-identical.

Portable CPU tests pin exact AE1/T2/T4 fixtures. Nineteen pipeline and thirteen lab
tests pass; the unignored matrix retains all thirty Naga specializations. The new
checker adds four CPU tests, fifty ordinary/coverage/fault Naga variants and whole
float-helper byte identity. Owned all/pedantic diagnostics are clean; repository
warnings remain. The actual release archive is build-v10, whose lab SHA is
67c9080cb409667e2d699aaa98c705946f7cda91695ffb72be335d9cfcd8ed27 and checker SHA is
542a53b2a1c96fc483c912ae1fa6770c2e643294be3e195305de23da9d221e15.
Cargo's reported executable paths, optimization 3, source inventory and hashes
are retained. The three production shader bytes are embedded exactly.

Fresh main-thread GPU proof passed 180 fixtures and 540 real raster dispatches.
For all five logical workgroups, T1/2/4, lambdas 1/64 and X tile counts
1/2/3/4/5/9, it compares baseline, candidate and instrumented BOTH error arrays
before selection. Independent strict-predicate visits require each consumed
original tile exactly once and every skipped/inactive/padding/sentinel word
untouched. CPU cached-parent byte L1 independently checks old errors.
Adjacent 0/1/255/256 counts and mixed bytes stress cache reuse; source options
come directly from the actual factory snapshot.

Four separate actual GpuEvolver cases passed right-edge acceptance. T2/T4 at
one- and two-iteration batches change only a final-column pixel in a 17x19
fixture under logical 8x8 tiles. Completed snapshots prove a strict accepted
improvement, byte-L1/cache equality and unchanged other pixels; full recomputation
agrees. This covers the full-fitness, bulk and sampled raster paths and catches
accidentally shrinking the framebuffer-update grid. These correctness-only color
mutation settings are separate from the fixed timing workload.

Both real missing- and duplicate-tile faults were rejected. Their white empty
tiles contribute zero to both error totals, so numerical agreement alone passes.
The actual changed work mapping nevertheless fails visit coverage. Each negative
runs one fixture and three dispatches, exits 1 with negative_detected=true, and
records successful error equality plus explicit negative rejection. Compilation,
timeout and readback failures do not count as a successful fault demonstration.

Evidence under .cache/evolution-overnight-20261024/raster-direct-coarsen/ includes
design.md, implementation-notes.md, frozen-spec.json, CPU logs, fixture-pin/,
checker/, and gpu-proof-v1/. Default source is pinned as raster-ae1-reference.wgsl;
candidate hashes are c3b2d775f76fcd9f8ebadf77d957cad40663fb766641e662174e245f6de80da3
and 64d4584f32865fed0d15f1d422d4afcbd482b175755bbc606eb9b1879b36feb4.
Patches 41 and 42 reconstruct the independent candidates.

The frozen short screen uses two alternating pairs per project, eight batches,
seed 7, two matching warmups and separate overflow replays. Only the highest
passing point may advance to five-pair 64-batch training and reserved confirmation.
No candidate is adopted from this prototype proof. Timing, broader backend
qualification and hardware attribution remain unproved here.
