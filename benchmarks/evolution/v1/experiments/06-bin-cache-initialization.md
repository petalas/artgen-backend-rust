# Experiment 6, remove redundant bin-cache initialization

Candidate 5 passed its full core-throughput screen at 2.3145 times the original
reference, but benchmark setup rose by roughly 6.6 seconds. This experiment
compares directly against that tested fast implementation. Its primary goal is
to recover setup time while preserving completed throughput and exact evolution.
The original throughput policy and its 3% promotion threshold remain unchanged.

## Mechanism and safety contract

Keep candidate 5's shader byte-identical. Disable automatic workgroup-memory
zero initialization only for binning pipeline compilation, including tile-size
recreation and the separate benchmark diagnostic variant. Leave mutation,
selection, rasterization and framebuffer pipeline options unchanged.

The bin cache has 1,000 vec4<u32> slots. For a validated polygon count P in
0..1000, lane i writes slots i+32*k below P, and all lanes reach the explicit
workgroup barrier before reading. Tile owners read only slots 0..P. No suffix
slot is consumed; P=0 reads no cache slots and still resets tile counts. This
proof is independent of the contents supplied by automatic initialization.

The candidate must pass complete GPU bin-buffer comparison against the frozen
original, including poison/stale words, counts 500/999/1000 and five workgroups.
It must also pass rasterization, compiler and exact seeded evolution checks.
Normal and overflow-diagnostic variants must use the same bin compilation option.

## Fixed setup and throughput comparison

Use the same archived candidate-5 executable as the reference, not the original
slower engine. Preserve the original mature ff/mona/darwin fixtures above 97%,
resolution, packed complexity, seed 7, four chains, 64 offspring per chain,
64 iterations per batch and two warmup batches.

Run one fixed series of five alternating pairs per project at 64 measured
batches, plus one separate candidate overflow replay per project. Each worker
completes 1,048,576 measured evaluations; all 33 workers report 34,603,008,
including 3,145,728 diagnostic evaluations excluded from statistics. Do not pool
previous data, add samples or filter slow pairs. Any process, numerical,
capacity, overflow, replay or artifact error invalidates the experiment.

The existing comparison command collects both setup and completed-throughput
measurements. Its unchanged 3% throughput-promotion verdict is retained as such;
it does not answer this separate setup/non-regression question. Compute setup
speed ratios as reference setup seconds divided by candidate setup seconds.
Use the same paired log-mean statistic, equal project weights, 10,000 stratified
bootstrap draws and fixed random seed as policy 1.

The setup goal passes only if aggregate setup speed exceeds 1/0.9, its interval
lower bound exceeds 1, and every project's median setup improves by at least one
second with a median paired setup speed above 1. This tests a material reduction
rather than interpreting tiny timer variation.

Throughput non-regression requires aggregate paired geometric throughput at
least 1.0 and aggregate interval lower bound at least 0.99. Every case must have
median paired throughput at least 1.0 and interval lower bound at least 0.99.
An interval entirely below 1.0 demonstrates a slowdown and fails this gate.
If a point estimate misses without a demonstrated slowdown, classify the result
as inconclusive rather than claiming equivalence or extending the schedule.
All quality, exact endpoint replay, completed-work and overflow gates also pass
before any positive setup disposition. These are conditional same-seed timing
bounds, not a guarantee about all future workloads.

Setup seconds cover the benchmark's preparation, not isolated browser readiness
or an individual compiler phase. A result here will not be called a direct UI
loading measurement. Keep background isolation unverified and leave other
processes untouched. Live-preview, reserved-state and independent-seed gates
remain open before default adoption, even if this setup test succeeds.

Archive immutable binaries, the incremental patch relative to candidate 5, the
combined patch relative to dd52d50, build/profile/lock identity, all raw outputs
and the pre-data contract under `.cache/evolution-experiments/bin-nozero-06/`.
Prove It Works requires actual uninitialized-cache execution and full buffers;
Explain the Number separates setup recovery from evolution throughput.

## Verification completed before timing

The full-buffer GPU checker passed 805 paired checks and 1,610 dispatches,
including 500/999/1000 polygons. Rasterization passed with the candidate's
actual pipeline options. All 50 Rust tests passed. Independent source review
found no material issue in normal/recreated/diagnostic pipeline coverage or
prefix initialization. Mutation and other pipelines remain unchanged.

Candidate shader SHA-256 remains
`a219175f3fd395600f0ecbef3ac099b7450aeaf972e1e5c91dfc5dcde1fb901b`.
Candidate pipeline source SHA-256 is
`dd6ee2b355fc100f9fbcaeb1fc2b4246ba7b073d44faa373bef5b5789fcd49c3`.
Candidate executable SHA-256 is
`60e1e076876e5a46c0329d701ac1f4d1950d7ae9a1c2826c08a1bf6d56ef1b1f`.
Reference candidate-5 executable SHA-256 is
`60b0ccd82956ed88cdeaf20f6fb818deb90be19ee86194f4ba480b0a29a9bad6`.
Cargo reports locked release, optimization 3 and disabled debug assertions for
both. Exact source archives, combined and incremental patches and toolchain
identity are retained in the local experiment bundle.

The new helper passes formatting and whitespace checks. Whole-file formatting
already differs at HEAD, and Clippy reports pre-existing repository warnings;
neither prompted unrelated rewrites. A local proof-run aggregation script
attempted to exclusively create its summary a second time after both GPU
children had completed successfully. The original partial summary, both raw
process records and a corrected final aggregation are retained. No GPU job
failed or was rerun for that reporting error.

The original shader, pipeline source and normal benchmark executable were
restored before the timing series. The two tested implementations run only
from immutable archived executables. The setup analysis code and its hash were
frozen before timing; synthetic checks reject a known setup regression and a
2% throughput slowdown.

The setup timer also includes the initial snapshot, its JSON write and numerical
qualification in `src/benchmark/trial.rs`. These operations are identical on
both sides. The result therefore measures complete benchmark preparation, not
a separated shader-compiler interval. Final snapshot/qualification remains
outside the completed-throughput interval, whose ordinary readback and final
drain are unchanged.

## Result: reject the setup change

All 33 workers completed with zero process, device, numerical, capacity,
overflow, replay or artifact errors. Every worker completed 1,048,576 logical
evaluations, giving 34,603,008 including 3,145,728 diagnostic evaluations excluded
from timing. All five pairs in each project remain included.

Complete benchmark setup improved materially. Its aggregate paired speed ratio
was **1.66191682**, with conditional 95% interval
**1.65455928 to 1.66999118**. This is
**39.83% less setup time**. Every setup gate passed,
with roughly 6.7 to 6.8 seconds saved in each project's median.

The throughput non-regression gate failed. The aggregate ratio against fast
candidate 5 was **0.99802930**, with conditional 95% interval
**0.99263963 to 1.00280423**.
The aggregate interval includes equality, but mona's interval lies below 1 and
its lower bound misses 0.99. Mona's paired median is 0.99790972. The aggregate
point estimate also misses the frozen minimum of 1. These failed checks yield
**REJECT** for the combined setup/non-regression objective. Policy 1 separately
returns REJECT under its unchanged throughput-promotion rules.

| Project | Reference setup median, range in seconds | Candidate setup median, range in seconds | Paired throughput geometric ratio | Conditional 95% interval |
| --- | --- | --- | --- | --- |
| ff | 17.0324, 16.8765 to 17.1621 | 10.2210, 10.1416 to 10.2992 | 1.00196527 | 0.99803440 to 1.00591162 |
| mona | 17.1961, 17.1219 to 17.3716 | 10.3631, 10.2297 to 10.3862 | 0.99087546 | 0.97819046 to 0.99936482 |
| darwin | 17.0845, 16.8675 to 17.1749 | 10.3404, 10.1364 to 10.4885 | 1.00128596 | 0.99151981 to 1.01165303 |

The slowest mona ratio, 0.96631733 in pair 2, remains in the result. No pair was
excluded and no sample extension or retry was launched. Background activity
was uncontrolled under the user's instruction. This series therefore does not
identify a universal kernel slowdown or its hardware cause. It does fail the
predeclared guard, so a loading gain is not accepted in exchange for uncertain
or lower sustained performance.

The shader and evolutionary settings were identical to candidate 5. The
single changed compilation option therefore isolates the source change behind
the preparation-time response in this cache/setup regime. Individual compiler
phase timings and direct browser readiness were not measured. The frozen cache
seed was copied afresh for every worker; warmed application reload behavior is
outside this claim.

Keep candidate 5, with its original bin-memory initialization, as the best
archived throughput candidate. Do not install candidate 6 or claim that the
original 2.3145-times-throughput result was remeasured here. The ordinary runtime
source and benchmark executable remain on the original reference. The next
useful step is live-preview and distinct-state/seed qualification of candidate
5, before another compiler-option experiment or default adoption.

The complete run is retained at
`.cache/evolution-experiments/bin-nozero-06/comparison/summary.json`.
`analysis.json` preserves the separate setup gates, throughput gates and all raw
per-case ratios. `pre-data-hashes.json` still matches the analysis, schedule and
pre-data report. The combined and candidate-5-relative patches remain available
for review without changing the installed engine.
