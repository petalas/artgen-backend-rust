# Binning amplification diagnostic

Four optimization screens did not clear promotion. The next question is whether
extra bin-construction work changes whole-engine completion time. The old sampled
bin interval covers only the final iteration and has different pass boundaries
from bulk iterations. It is not a measured whole-run share.

This is a diagnostic, not a fifth optimization attempt or an adoption score.
Keep the N=1 reference shader exact. For N=2, repeat the complete original tile
count reset, polygon bounds and ordered scatter body twice inside the same WG1
dispatch. Each repeat rebuilds identical bins. Mutation, rasterization, selection,
RNG, dispatch count and completed logical evaluations remain unchanged. The
single textual append marker still instruments every append in diagnostic mode.
The instrumentation tail is outside both reset loops.

## Schedule fixed before measurement

Use the original mature ff/mona/darwin reference bundle, all initial parents
above 97%, every frozen parameter and seed 7. Run five N=1/N=2 pairs per case,
reversing order each pair. Each worker uses two warmup batches, 64 measured
batches and 64 iterations per batch. This is 1,048,576 completed logical
evaluations per worker, with the final checked drain inside wall time.

Use the existing profiler's envelope-only path. It retains the normal mutation,
bin and raster samples and measures the full batch compute envelope separately.
The selection interval is unavailable in this mode. Require exactly 64 measured
samples after warmup, null throughput and primary-score eligibility false.
After all pairs, run three separate full-budget N=2 overflow diagnostics outside
response statistics. Require occupancy at most 256 and exact canonical replay.

There are 30 timed diagnostic workers and three overflow workers. Do not filter
slow pairs or add repetitions after seeing data. Any process, device, numerical,
artifact, count or replay error invalidates interpretation. Preserve raw output,
errors and completed work. Background activity remains uncontrolled; leave other
processes untouched. All binaries, source patches, schedule, scripts and caches
are retained under `.cache/evolution-profile/bin-sensitivity-05/`.

The only allowed escalation is a separately archived N=4 shader and one fixed
five-pair N=1/N=4 series if the valid N=2 series leaves any case's sampled-bin or
wall-time interval lower bound at or below 1.0. No further escalation or sample
extension follows. A correctness failure does not trigger escalation. A compiler
that removes repeated work leaves retention unresolved, not evidence of cheap
binning. N=4 is not built or run unless that rule applies.

## Correctness and response

Before timing, compare complete final count/index buffers against the frozen
original GPU binner. `--scope full-buffers` checks even unconsumed and unwritten
poison/stale words, while retaining consumed-prefix and 500/999/1000 truth checks.
It uses five workgroups and seven images, 805 paired checks and 1,610 dispatches.
The earlier strict and consumed-only modes keep their original contracts.

Require exact initial/final canonical state and numerical quality agreement with
the archived 64-batch trajectory for every worker, and recompute both endpoint
hashes from retained states. Submitted and completed logical work must agree.
Nominal bin-body executions are N times completed logical evaluations. Label
this as a count derived from the literal loop and drained work, not a hardware
execution counter. It excludes warmup and preparation.

Let T be checked evolution wall time, C the summed full-batch compute envelope,
and b the summed last-iteration bin samples. Report paired T_N/T_1, C_N/C_1 and
b_N/b_1, with per-side medians/ranges. Reuse policy 1's equal project weighting,
paired log-mean statistic, 10,000 stratified bootstrap resamples and seed
20261023. These are response ratios, not throughput promotion scores.

A reproducible sampled-bin increase is evidence that the perturbation happened.
Report envelope and wall response alongside it. Under the restrictive model
T_N = U + N*B, the repeated body's inferred wall share is
(T_N/T_1 - 1)/(N-1). This requires retained work, comparable per-repeat cost and
unchanged other costs. It excludes dispatch/synchronization costs that were not
repeated. Compiler transformations, repeat-local caches and scheduling can break
those assumptions. Do not substitute the nominal repeat factor for measurement
or infer GPU utilization or a hardware bottleneck.

Attack the Premise changes the question before another optimization. Explain
the Number requires retention evidence and separates sampled attribution from
whole-run response. Prove It Works requires direct GPU buffers and saved values.

## Result: retained work affects whole-run completion

All 30 normal profiling workers and three overflow replays completed with zero
errors. Five alternating pairs per project produced these equal-weight aggregate
response ratios, N=2 divided by N=1:

| Measurement | Response ratio | Conditional 95% interval |
| --- | --- | --- |
| Checked evolution wall time | 1.691465 | 1.689657 to 1.693245 |
| Full-batch compute envelope | 1.715879 | 1.714882 to 1.716881 |
| Sampled final-iteration bin time | 1.988340 | 1.987311 to 1.989415 |

The sampled bin interval nearly doubled, providing reproducible retention
evidence. Whole-run elapsed time increased by 69.15%, and the GPU compute
envelope increased by 71.59%, with unchanged evolved states. All per-case bin
and wall interval lower bounds exceed 1.0, so the declared N=4 escalation was
not triggered. No N=4 shader was built or timed.

Per-case wall seconds, median and range over five workers per side:

| Project | N=1 wall seconds | N=2 wall seconds | Paired geometric ratio |
| --- | --- | --- | --- |
| ff | 15.100 (15.043 to 15.129) | 25.327 (25.306 to 25.417) | 1.679013 |
| mona | 18.843 (18.759 to 18.855) | 31.945 (31.914 to 32.046) | 1.697577 |
| darwin | 18.498 (18.458 to 18.568) | 31.440 (31.370 to 31.472) | 1.697875 |

The normal profiles account for 31,457,280 completed logical evaluations; the
three overflow replays add 3,145,728, for 34,603,008. The nominal normal bin-body
count is 47,185,920, derived from the literal repeats and completed logical work.
It is not a hardware execution counter. All profile throughput fields are null,
and no record contributes an optimization/adoption score. Every timing pair is
included. Overflow maxima match the baseline and remain below 256.

Independent verification recomputed 132 output artifact hashes, all 66 canonical
endpoint hashes and all 264 cached framebuffer byte-L1 totals. All match the
original trajectory exactly. The full-buffer GPU checker passed 805 paired
checks and 1,610 dispatches. Fifty Rust tests passed. New checker formatting and
Clippy checks passed; existing library warnings remain. Invalid scope arguments
reject before GPU initialization.

This intervention establishes that bin-construction work materially affects
completion time on these mature workloads. The restrictive linear model would
assign about 69% of wall time to the repeated body, but that is a model-based
estimate. Repeat-local caches and compiler transformations are not separately
measured. Dispatch and synchronization costs were not repeated. The observed
response does not identify a specific hardware resource or GPU utilization.

The next isolated optimization uses unique tile ownership to fill bins in
parallel, preserving each tile's ascending polygon order and all original bin
contents. That is a hypothesis, not a speedup predicted from these ratios.

The runtime shader and ordinary benchmark executable remain the reference. The
probe is retained as an experimental patch, with source and binary identities in
`.cache/evolution-profile/bin-sensitivity-05/identity.json`. All raw paired data
and intervals remain in `series-2/summary.json`, with direct verification in
`series-2/independent-verification.json`. Background activity remains unverified.
