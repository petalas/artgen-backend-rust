# Keep one polygon traversal for paired pixels

ADOPTED under the fixed-work engine gate. Reserved completed-engine throughput
is **5.49% higher** than AE1, ratio **1.054929**, with conditional 95% interval
**[1.050581, 1.059429]**. Training independently passes at 1.050842
[1.048352, 1.053630]. Both phases retain all five alternating pairs per image.
All evolution parameters, RNG streams, precision, fitness and selection stay fixed.

The shader fetches each cached polygon once in the source and calls the unchanged
blend function separately for the two pixels owned by a lane. Each pixel keeps
its ascending polygon order. Logical tiles remain 32 by 16 with 256 physical
lanes; the other four workgroup shapes retain their original one-pixel path.
Shared memory, bindings, dispatches, barriers and integer reduction stay unchanged.
The new private RGB scalars avoid the reproduced Naga struct-member pointer bug.

The [protocol](profiles/pixel-pair-safe-spec.json) and exact
[source patch](attempts/24-raster-pixel-pair-safe.patch) were frozen in adc8d23.
The earlier compiler-failed attempt remains closed INCONCLUSIVE with no timings.
This cohort reuses none of its samples or correctness records.

## Completed throughput

Each timed trial completes 64 batches, 4096 iterations and 1,048,576 evaluations.
Five pairs per image use AB/BA/AB/BA/AB. Training uses seed 7; distinct reserved
drawings use seeds 11/13/17. Both arms have eight warmups and ordinary Stages
timestamps. Timing includes ordinary readback and checked final drain; preparation,
compilation, warmups, numerical checks and exports are outside engine elapsed time.

Rates are evaluations/second, median [minimum, maximum] of five runs. The paired
column reports median [minimum, maximum] and need not equal the ratio of rate medians.

| Phase | Image | AE1 rate | Paired-pixel rate | Paired ratio |
| --- | --- | ---: | ---: | ---: |
| Training | darwin | 254,449.9 [251,746.4, 254,601.1] | 267,364.5 [264,319.1, 269,153.2] | 1.050755 [1.047872, 1.057599] |
| Training | ff | 315,385.6 [305,265.8, 325,536.1] | 328,604.4 [318,892.0, 338,728.6] | 1.044637 [1.040525, 1.061694] |
| Training | mona | 254,600.6 [253,310.9, 256,836.2] | 268,481.6 [266,534.8, 270,199.6] | 1.054226 [1.046874, 1.055446] |
| Reserved | ff | 310,456.2 [305,879.4, 323,760.4] | 328,185.2 [317,394.0, 337,655.3] | 1.047141 [1.037460, 1.072079] |
| Reserved | mona | 270,400.7 [267,519.8, 276,958.6] | 285,379.7 [280,699.7, 292,173.4] | 1.055396 [1.049267, 1.060425] |
| Reserved | darwin | 264,495.8 [261,827.4, 271,903.0] | 278,909.3 [275,154.3, 285,513.2] | 1.054494 [1.050055, 1.064577] |

The aggregate is the equal-image geometric mean of paired ratios. The frozen
stratified log bootstrap uses 10,000 draws and seed 20261024. Both aggregate
point/lower-bound gates and all per-image non-regression gates pass. Intervals
condition on these repeated timing seeds and recorded sessions. The earlier
A/A64 control supplies calibration and device identity only; no samples or
correction enter this comparison. Background GPU isolation remains unverified.

## Correctness and audit

All 60 timing records, 12 separate diagnostics and six qualifications complete
with zero reported errors. They account for 62,914,560 measured, 12,582,912
diagnostic and 6,291,456 qualification evaluations. Eight-warmup preparations add
10,223,616 source-accounted evaluations. Total logical work is 92,012,544,
excluding setup/full-fitness resets and the unscored direct fixture job.

The [confirmed machine record](profiles/pixel-pair-safe-confirmation.json) retains
rates, distributions, sampled stages, counters and evidence hashes. Parent audits
pass 1047 predicates across 36 snapshots and 144 saved parent associations. A fresh
independent audit passes 1500 checks, reproduces every ratio/interval, and uses
new byte-L1/f32 arithmetic for all 48 unique parent associations. All seven
canonical fields replay exactly; cached errors equal framebuffer RGB-byte L1.
The existing 2e-5 CPU/GPU fitness bound is unchanged.

The direct GPU checker passes 91 fixtures: 65 raw cases plus 26 existing engine
checks. It compares uninstrumented candidate/reference new and old accumulators
before selection, checks exact pixel visits and sentinels, independently checks
old-byte L1, and exercises full-fitness/incremental paths. All five shapes, counts
0/1/194/255/256/300, dirty rectangles, odd image edges, mixed bytes and degenerates
are covered. The 300-polygon case matches its original first-256 prefix. The
coverage specialization reverses exactly to the measured candidate. CPU compiler
checks remain separate from this real GPU proof.

The immutable lab is 97d176f3…; the fixture checker is e4b2b5b8…. The selected
source is 83c505bc…, installed byte-for-byte in src/shaders/rasterize_error.wgsl.
All five native workers finish serially before the original 1800-second deadline.
There is no retry, sample extension, exclusion, threshold change or competing
candidate in this cohort. Source review passes all 28 declared invariants.

## Source and remaining work

Focused CPU checks pass: 18 pipeline tests and 14 lab tests. The normal release
build passes for the application, evolution-bench and engine-lab with optimization
3 and debug assertions disabled. All six complete shader strings are verified in
each fresh executable; the [release record](profiles/pixel-pair-release.json) pins
the actual Cargo artifacts and 83-file source freeze. The source keep is
committed as bb255c0. A separate, precommitted default-path worker passes Mona,
seed 7, 64 batches and the ordinary two warmups using the rebuilt bundled shader,
without a source override. Its 1,048,576 evaluations, two snapshots and eight
parents match both the qualified candidate and the reference; 51 recorded
predicates pass. Its [delivery record](profiles/pixel-pair-default-verification-result.json)
is explicitly unscored and adds no samples to the completed five-worker cohort.
The first offline metadata preflight caught the compact LabRow/TrialRecord
representation difference before GPU execution; it was corrected without a rerun.
All raw qualification files remain unchanged. The normal application was rebuilt
but its UI/WebSocket path was not launched; no preview or startup claim follows.

The raster/error interval remains the largest sampled GPU stage. Training
trial-average raster medians decrease from about 436/560/556 microseconds to
394/507/506 for ff/Mona/Darwin. These final-iteration samples support attribution
to this pass; they are not whole-run shares, utilization or proof of a specific
load/register/hardware mechanism. A new emitted-code observation is the next
way to check whether the shared loads were actually reused.

Power of Ten keeps the new helpers below one page with fixed private scalar state,
a maximum 256-polygon traversal, and maximum per-lane error 1530. The existing
main orchestration remains a documented long-function exception: its loader,
three barriers and reduction suffix are byte-identical; the only extension selects
the paired helper. Refactoring that synchronization during adoption would change
the measured source. Prove It Works required actual GPU fixtures and exact replay;
Explain the Number kept the gain tied to completed engine work.

The historical direct original-to-baseline result remains 4.584441 times. No new
original-to-current ratio is claimed or inferred by multiplying these gains. The
broader policy-1/live-preview qualification remains outside this engine-only keep.
