# Backend finalization is a distinct host cost

One qualified private mona observation separates front-end command recording
from backend finalization. Across 64 measured batches, recording compute passes
takes 0.003068 seconds; `encoder.finish()` takes 1.177429 seconds of elapsed
host-thread time. The former is too small to explain the full-batch residual.
The latter warrants a dependency-preserving experiment, not a savings claim.

| Direct bracket | Total elapsed seconds |
| --- | ---: |
| Early conflict collection bracket | 1.065405 |
| Parameter/control uploads | 0.009504 |
| Front-end compute/pass recording | 0.003068 |
| Trailing query/copy recording | 0.000063 |
| Encoder finish | 1.177429 |
| Queue submit | 0.038640 |
| Post-submit prior collection bracket | 1.652106 |
| Async-map registration | 0.000232 |

These brackets are sequential within submitted batch calls. Their durations
include off-CPU waits and driver activity. They are not CPU-cycle counters.
Whole batch rows total 3.946519215 seconds; checked final drain is 0.052399626.
Gaps outside those rows total 565,362 ns, and gaps between direct brackets inside
the rows total 71,809 ns. They close to the sidecar's 3.999484203-second epoch;
the ordinary result's 3.999484173 seconds excludes its 30 ns end-marker offset.
Returned Drawing destruction remains inside measured wall time after its batch
row closes and is part of the unattributed gap.

Readback rows total 2.769839017 seconds inclusively. Their nested phases total
2.769782398, including 2.767204845 seconds in polling and 0.001743452 seconds
in drawing decode/drop/unmap. These rows are already contained in the collection
and drain brackets above. The GPU compute envelope is 3.243450080 seconds and
overlaps host work. Adding any of these inclusive or cross-device totals would
produce a false whole-run share.

The first two measured finishes take 154.116 and 225.342 ms. Rows 3–64 have
median 10.011 ms, p95 19.280 ms and maximum 23.881 ms. Their cause is unproven;
no cold outlier is excluded and warmup remains exactly two batches.

## A measured question for one experiment

Twenty-one early conflicts occur on caller batches 4, 7 through 64, collecting
targets 3, 6 through 63. Backend finalization on those same caller rows totals
390.249432 ms, median 18.328485 ms, range 16.973189–23.880967 ms. Under the
restrictive assumption that this cost stays unchanged and can all overlap
remaining previous-GPU execution, it is a ceiling of 9.76% of the observed wall
before added costs. Driver blocking, caching and the extra command-buffer
boundary may invalidate that assumption. The 1.177-second total is not available
headroom.

The source audit supports a guarded test: on existing early-conflict calls only,
finalize an independent compute/timestamp prefix before collecting the previous
readback. Then collect that same result, construct/upload the same parameters
and updated control reset, encode the control→fitness→drawing-copy tail, and
submit the two command buffers together once. Keep the single mapped drawing
staging buffer out of the prefix. Preserve every dispatch, pass, query, copy,
logical evaluation, RNG draw, return and flush behavior. Other calls retain the
original single command buffer. This is a proposed experiment, not adoption.

The earlier rejected later-readback prototype moved only front-end recording;
its backend finish still followed collection. The new profile identifies the
unmeasured distinction rather than changing that rejection's disposition.
Actual ordinary-API equality must pass before any alternating timing screen.

## Observer proof and limits

The observer exists only in a copied private workspace. It buffers 64 batch rows
and at most 65 readbacks, excludes warmup explicitly, and exports a separate
118,436-byte sidecar after elapsed capture and checked drain. Production source,
parameters, callbacks and GPU operations were unchanged during this observation.
Removing instrumentation restores all accepted source bytes. The finish/submit
expression is split into local statements with the same evaluation order.

Source review found missing interval validation and an incorrect synchronous
fallback target could pass. Both are corrected before any physical observer run.
Actual-export regressions demonstrate RED then GREEN, ending with four passing
CPU tests and preserved partial rows. The first build remains archived and
unexecuted. The final build uses locked dependencies, optimization 3 and debug
assertions false, with full production shader bytes verified.

The physical worker completes 1,048,576 evaluations, 64 timestamp samples and
exact initial/final replay against all five accepted mona envelope workers.
Independent audit finds no material issue: interval order, containment, staging,
pending-target identity, all 64 asynchronous completions and the final synchronous
fallback target 64 pass. Byte-L1 and f32 fitness recompute exactly within the
unchanged 2e-5 policy. Scalar rasterization is not rerun. This one instrumented
worker is diagnostic and supplies no speedup, repeatability or hardware-utilization
claim; timer/recorder overhead remains uncalibrated.

Observer binary SHA256 is
50604668356d7b76a14c2b42216b2105a41eea3301d3900ca764278cc4872793.
Evidence is retained under `.cache/evolution-profiling-20261024/` in
`cpu-phases-77/`, `cpu-phases-order-fix-80/`, `cpu-phases-target-fix-82/`,
`cpu-phase-runtime-v2/`, `cpu-phase-audit-83/` and `finalize-overlap-84/`.
The pre-existing root app build remains untouched; its 11:17 identity was already
recorded before private validation. Original accepted archives remain unchanged.

Explain the Number separates the affected-call ceiling from total finalization,
and elapsed waits from recoverable work. Test Behavior requires malformed exports
to fail while retaining evidence. Prove It Works requires actual state and work
replay before interpreting the sidecar.
