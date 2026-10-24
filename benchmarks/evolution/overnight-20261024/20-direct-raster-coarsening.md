# Reject direct raster tile grouping

REJECT both T2 and T4 at the frozen short screen. The equal-project geometric
throughput ratios are 0.968418649371 for T2, with conditional 95% interval
[0.967582123663, 0.969255898299], and 0.968871553587 for T4, with interval
[0.966512340108, 0.971236525800]. Every paired candidate rate is below its
reference rate. Neither clears the 1.005 aggregate gate, and every case median
is below the 0.98 floor. Neither advances to full-budget training or reserved
confirmation. Prototype commit 05100b546766f54ad09ecf93b27aeef249bbdd81 remains
unadopted. There is no combined-gain claim.

The predeclared schedule uses two alternating pairs per case, eight batches,
seed 7 and two matching warmup batches. Pair zero runs reference/candidate;
pair one runs candidate/reference. All twelve pairs remain, including every
slow sample. These are two timing repeats per side, not the five repeats
required for acceptance. The fixed screen permits early rejection at this
budget; no extra sampling was added.

Rates below are evaluations/second, with median and observed minimum/maximum.
Each reference belongs to its own candidate comparison.

| Candidate | Case | Reference median [min, max] | Candidate median [min, max] | Paired ratio median [min, max] |
| --- | --- | --- | --- | --- |
| T2 | ff | 335886.482 [334931.146, 336841.817] | 324931.560 [323771.670, 326091.450] | 0.967383053 [0.966681284, 0.968084822] |
| T2 | mona | 275661.555 [275533.989, 275789.121] | 267036.301 [266712.502, 267360.100] | 0.968711463 [0.967088554, 0.970334372] |
| T2 | darwin | 272519.977 [271810.793, 273229.161] | 264117.274 [263187.708, 265046.839] | 0.969164339 [0.968275414, 0.970053264] |
| T4 | ff | 331708.992 [331658.328, 331759.656] | 322988.620 [322571.416, 323405.823] | 0.973710606 [0.972601588, 0.974819625] |
| T4 | mona | 273244.455 [270871.748, 275617.162] | 265806.561 [264846.057, 266767.064] | 0.972822171 [0.967889889, 0.977754453] |
| T4 | darwin | 269731.418 [268838.020, 270624.816] | 258982.560 [258705.873, 259259.247] | 0.960156910 [0.958002490, 0.962311330] |

The independent CPU audit recomputes each rate as 131072/elapsed_seconds and
each paired ratio as reference_elapsed/candidate_elapsed. It repeats the
frozen 10000-resample bootstrap, seed 20261024, with log-ratio resampling within
ff/mona/darwin and equal case weighting. Both aggregate ratios, aggregate
intervals and case intervals reproduce the saved analysis exactly. Intervals
are conditional on this eight-batch, same-seed screen. They do not describe
independent search seeds or establish a general hardware regression.

## Work and exactness

All 24 expected normal trials and nine expected overflow diagnostics complete
correctly with zero reported errors. Every trajectory submits and completes
eight batches, 512 iterations and 131072 logical evaluations. Four chains and
lambda 64 give 256 evaluations per iteration.

| Recorded work | Trajectories | Submitted/completed batches | Submitted/completed iterations | Submitted/completed evaluations |
| --- | --- | --- | --- | --- |
| Normal timing | 24 | 192 | 12288 | 3145728 |
| Separate diagnostics | 9 | 72 | 4608 | 1179648 |
| Reference qualification | 3 | 24 | 1536 | 393216 |

These counts exclude setup/warmup work and the separate checker proof.
Diagnostics have null rates and zero elapsed/drain timing fields; they never
enter throughput statistics. Initial/final maximum occupancies are ff 63/66,
mona 78/79 and darwin 78/80 for each source, below capacity 256. Normal
occupancy fields are null. All 33 rows report actual raster T1/T2/T4 and
mutation group size 1, with identical parameters and GPU identity per case.

The reference receives the existing independent scalar numerical qualification
once per case and eight-batch budget. All four initial parents exceed 97%
fitness. Every initial/final quality entry satisfies exact framebuffer-byte
RGB L1 versus cached error, stored-fitness association and zero unexplained
numerical pixels. Scalar and GPU pixels may differ within the existing
explained arithmetic/coverage envelope. Qualification under that policy does
not imply scalar byte identity.

The archived lab compares all seven canonical fields directly before and
after each normal and diagnostic trajectory. They are packed parents, packed
offspring, parent errors, parent framebuffers, offspring errors, chain update
flags and packed fitness. Reporting-only control IDs remain excluded. The CPU
audit independently rehashes the six unique qualified initial/final states and
all 18 saved diagnostic snapshots, compares the seven fields exactly and
checks all 33 rows' state associations. It recomputes 24 parent byte-L1 totals
across those six states, verifies packed fitness and triangle counts, and
decodes every saved final drawing to its matching packed geometry and color.
Normal per-trial snapshots are not separately persisted; their direct exact
comparisons are enforced by the archived lab and represented by matching
canonical hashes in the raw rows.

## Sampled stages and limits

These are medians of the two per-trial average stage samples, in nanoseconds.
Each trial samples only the final iteration of each of its eight batches.
Columns list mutate, bin, raster/error and select in that order.

| Comparison | Case | Reference stage ns | Candidate stage ns |
| --- | --- | --- | --- |
| T2 | ff | 88356, 47382, 427252, 77412 | 88640, 48870, 443106, 77278 |
| T2 | mona | 107972, 60932, 558388, 82366 | 107954, 62620, 590494, 86684 |
| T2 | darwin | 98786, 52942, 553376, 94412 | 98374, 54334, 581004, 96282 |
| T4 | ff | 88750, 48016, 414540, 77906 | 88576, 48632, 444762, 76810 |
| T4 | mona | 108580, 62308, 558804, 85700 | 108072, 62766, 598448, 87380 |
| T4 | darwin | 98710, 52914, 553018, 93180 | 98756, 53252, 615482, 92674 |

Raster/error is the largest sampled stage and rises in all six comparisons,
consistent with lower engine rates. This does not identify a hardware limiter.
No matched occupancy, bandwidth or stall counters establish whether barriers,
serial tile work or driver compilation caused the change. Stage samples cannot
be summed into whole-run wall time, treated as GPU utilization or multiplied
by 64 as a measured batch total. Engine EPS includes ordinary encoding,
submission, polling, drawing readbacks and the checked final drain. Setup,
compilation, added qualification, snapshots, hashing and report writes are
outside the normal timed region. No result establishes an application gain.

The adapter reports Vulkan through Dozen, Mesa 25.3.6, on an RTX 5090.
Recorded CPU load averages are below one on 16 logical CPUs. Before/after GPU
point samples cannot establish isolation during the run; competing background
activity remains uncontrolled. The verdict is rejection at the frozen budget.

## Retained prototype and correctness proof

This benchmark prototype processes two or four adjacent logical tiles in each
directly dispatched workgroup. Production remains exact AE1. Both candidates failed before full-budget
confirmation. No prefix, indirect dispatch,
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
For logical workgroups 32x16, 32x8, 16x16, 16x8 and 8x8, T1/2/4, lambdas 1/64 and X tile counts
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

The CPU audit independently verifies those event counts, all saved error-array
deltas and old-L1 associations, and all specialized factory snapshot hashes.
Every factory snapshot records empty constants and Vulkan initialization FALSE.
The four host integrations are additional correctness cases; their host batch
dispatches are not part of the 540 checker dispatches. Each changes only pixel
[16, 9] and reduces byte L1 from 765 to 433, with cached and full recomputation
agreement. The missing fault reports one missing visit and zero duplicate
visits. The duplicate fault reports one missing and one duplicate visit. Both
report zero unexpected visits and valid GPU setup. This finite fixture matrix
and four integrations do not prove every image, backend or schedule.

Evidence under .cache/evolution-overnight-20261024/raster-direct-coarsen/ includes
design.md, implementation-notes.md, frozen-spec.json, CPU logs, fixture-pin/,
checker/, and gpu-proof-v1/. Default source is pinned as raster-ae1-reference.wgsl;
candidate hashes are c3b2d775f76fcd9f8ebadf77d957cad40663fb766641e662174e245f6de80da3
and 64d4584f32865fed0d15f1d422d4afcbd482b175755bbc606eb9b1879b36feb4.
Patches 41 and 42 reconstruct the independent candidates.

The offline audit rehashes every build-v10 archived source and both binaries,
and finds exact G1/AE1/598 production shader bytes inside the lab executable.
Build provenance's commit label predates the prototype; the matching source
inventory identifies the scored source. Patches 41 and 42 independently
reconstruct the candidates from AE1. The full prototype commit patch is also
archived. Artifact hashes and raw statistical/state checks are recorded in
.cache/evolution-overnight-20261024/direct-result-audit-55/audit.json.
Its audit.py is CPU-only; source-freeze.json pins the lab, checker, candidate
sources and relevant implementation sources. The raw screen remains in
raster-direct-coarsen-screen-01/output/ without edits.

Full five-pair 64-batch training and reserved confirmation were not run.
Neither candidate is adopted. Removal of rejected runtime support remains
pending source verification by the parent and cleanupowner53. This report does
not certify cleanup completion or change production adoption status.

## Runtime cleanup

Commit `fe5e829` removes the rejected grouping APIs, runtime branches, execution
metadata and dedicated checker. The four surviving engine/lab files match the
preceding `2165b2c` versions byte for byte. Historical fixtures and patches stay
available for reproducing the experiment. Static checks preserve the mutation
group parser/cache, both mutation dispatches, all three ordinary raster dispatches,
both framebuffer updates, pass boundaries, timestamp meanings and hot installers.
All three accepted production shader hashes remain unchanged.

The scoped check, 64 library tests and twelve lab tests pass. Scoped all/pedantic
Clippy exits zero with existing warnings and none on restored lines. A fresh
release default worker also reproduces the original mona endpoint under a
separate unscored host profile. Cleanup preimages, patch, protected-file hashes,
command logs and parent verification are in `direct-rejected-cleanup/`. The new
release's source inventory and exact embedded shader proof are in `build-v11/`.
