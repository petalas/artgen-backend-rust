# Experiment 5, tile-parallel ordered bin filling

The bin-body amplification diagnostic measured a near doubling of sampled bin
time and a 69.15% increase in whole-run wall time with identical evolved states.
This supports investigating bin construction. It is not a hardware-utilization
measurement or a predicted speedup for this candidate. See the
[diagnostic evidence](../binning-sensitivity.md).

## Mechanism and exactness

Use 32 lanes per offspring workgroup. First compute the original final integer
polygon tile bounds in parallel, into a 1,000-entry vec4<u32> workgroup array.
After one workgroup barrier, each lane owns a distinct sequence of tile indices.
For each owned tile it resets the count and scans polygon indices in ascending
order, appending exactly when the original half-open tile range contains that
tile. Different tiles can fill concurrently while each tile retains layer order.

Build every original tile. Do not include dirty-region clipping, packed rejection
or the N=2 diagnostic loop. Preserve original float decoding, AABB and tile-bound
expressions, bindings, one workgroup per offspring, dispatch count, overflow count and ordered
first-256 indices. Keep the existing atomic append and default workgroup zeroing
for this candidate. Removing either is a separate future change.

Every consumed shared slot has one writer before the barrier. The only return
before that barrier is workgroup-uniform. Empty drawings still reset all tile
counts; unwritten index words retain their original history. Each tile has one
owner, and each owner's polygon scan is serial. Thus final complete count/index
buffers must match the original, including unconsumed bins and stale words.

The shared array is 16,000 bytes, within the existing 16,352-byte requested limit.
The workgroup size is 32, within existing limits. No host limits, dispatches,
settings, host buffer allocations, bindings or pipeline options change. Scan work can grow
with tile count; parallelism is a hypothesis that must earn a measured gain.

## Proof and fixed measurement stages

Archive the candidate patch, executable, profile, toolchain and lock identity.
Run the unchanged full-buffer checker against the frozen original GPU binner:
805 paired checks and 1,610 dispatches, five tile workgroups, seven image sizes,
packed boundaries, zero/inverted boxes and counts 500/999/1000. Run the existing
rasterization regression and compiler shader checks. Independently review shared
initialization, tile ownership, ordering and arithmetic before scoring.

Reuse [comparison policy 1](../comparison-policy.md) without changes. Preserve
mature ff/mona/darwin parents above 97%, actual resolution and complexity, seed 7,
four chains, 64 offspring per chain, 64 iterations per batch, every mutation
setting and two warmup batches. Initial and final canonical state, numerical
quality, parameters, counters and adapter identity must match exactly. Ordinary
readback and final completion remain inside the timed interval.

First run five alternating pairs per project at eight measured batches and one
separate candidate overflow replay per project. All 33 workers account for
4,325,376 completed evaluations including diagnostics. Keep every pair and error.
Advance only if all unchanged preliminary gates pass, to five pairs per project
at 64 batches with the same archived candidate. Otherwise stop without a full
budget performance claim. Do not tune the candidate from its preliminary result.

Production remains on the reference pending live, held-out and broader campaign
qualification. No KEEP is possible from this screen alone. Background activity
is uncontrolled and other processes remain untouched. All proof and timing
records are retained under `.cache/evolution-experiments/bin-tile-05/`.

Prove It Works requires full-bin equality and exact evolution replay. Explain
the Number uses the causal diagnostic to choose the next code path, while keeping
sampled attribution and hardware-resource claims separate.

## Verification completed before scoring

The full-buffer checker passed 805 paired checks and 1,610 GPU dispatches,
including all unconsumed and stale words. The GPU rasterization regression
passed, and all 50 Rust tests passed, including shader variants. Independent
source review found no material issue in shared-prefix initialization, unique
tile ownership, ordering, arithmetic or host compatibility.

Shader SHA-256:
`a219175f3fd395600f0ecbef3ac099b7450aeaf972e1e5c91dfc5dcde1fb901b`.
Executable SHA-256:
`60b0ccd82956ed88cdeaf20f6fb818deb90be19ee86194f4ba480b0a29a9bad6`.
The ordinary executable and runtime shader were restored before timing.

## Preliminary result: screen passed

All 33 eight-batch workers completed with zero errors and exact saved-state
replay. The aggregate paired throughput ratio was **1.95904675**, with
conditional 95% interval **1.92325670 to 1.99731779**. Every frozen preliminary
gate passed. The generic comparator reports INCONCLUSIVE for adoption, as
required while broader qualification is incomplete.

The separately frozen 64-batch confirmation will now run with the same archived
shader and executable. Its observations will not be pooled with this preliminary
series. No implementation or acceptance threshold is changed from this result.

## Full-budget result: timing screen passed, adoption pending

All 33 confirmation workers completed with zero errors. Every frozen timing
and correctness gate passed. The aggregate paired completed-throughput ratio
was **2.31453649**, a measured **131.45% increase**, with conditional 95% interval
**2.30891556 to 2.32134700**. The generic verdict remains INCONCLUSIVE for
adoption because live, reserved-state, independent-seed and broader campaign
qualification remain open. No KEEP is claimed.

Five alternating pairs per project used 64 measured batches. Medians and ranges
for completed eval/s are below. The start uses the actual frozen image and
packed drawing; initial similarity and complexity remain identical on both sides.

| Project | Initial fitness | Triangles | Reference eval/s | Candidate eval/s | Paired geometric ratio | Conditional 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| ff | 97.34536% | 423 | 68,984 (68,626 to 69,149) | 153,378 (152,613 to 153,684) | 2.222740 | 2.219765 to 2.225572 |
| mona | 97.18222% | 539 | 55,158 (55,048 to 55,218) | 129,425 (128,616 to 130,291) | 2.345814 | 2.337702 to 2.354368 |
| darwin | 97.40385% | 474 | 56,701 (56,563 to 56,930) | 134,509 (134,008 to 136,659) | 2.377989 | 2.364087 to 2.397627 |

Each worker completed 1,048,576 evaluations across 4,096 iterations and 64
batches. Total reported work was 34,603,008 evaluations. The 3,145,728
diagnostic evaluations are excluded from timing; the 30 normal workers account
for 31,457,280. Submitted and completed counts agree. Diagnostics match their
normal trajectories, and no consumed bin exceeds capacity. No slow pair or
failed worker was excluded. Confirmation is analyzed separately from the short
screen and uses the exact same archived executable.

The source change parallelizes bin filling by tile while retaining a serial
polygon scan within each tile. All original float bounds and indices are
preserved. The measured gain does not depend on reducing image resolution,
precision, candidates, chains or drawing complexity. The direct GPU suites and
full trajectory checks support this claim on the recorded backend and corpus.

## Setup tradeoff and remaining qualification

Benchmark setup medians rose from roughly 10.3 to 10.6 seconds to 17.0 to 17.1
seconds. The added shared array still uses default workgroup zero initialization.
Compilation of that initialization is a plausible cause of the increase, but
setup contains more than compilation and is not an isolated preview-loading
measurement. The new array's consumed prefix is explicitly initialized before
its barrier; removing redundant automatic initialization is a separate next
experiment, with throughput and correctness non-regression required.

The sampled bin intervals and full setup records are retained in
`stage-analysis.json`. Sampled stage totals cover only the final iteration of
each batch and are not whole-run percentages or GPU utilization. Background
activity was uncontrolled. Timing repeats share seed 7; they do not establish
independent search-seed or held-out quality. A longer live-preview acceptance
case remains necessary before changing the runtime default.

The ordinary runtime shader and executable remain at the reference. The tested
candidate is retained as [05-tile-parallel-binning.patch](05-tile-parallel-binning.patch)
and an immutable executable. Its identity and all full-budget records are in
`.cache/evolution-experiments/bin-tile-05/confirmation/summary.json`.

## Independent final audit

A separate read-only audit reproduced the aggregate interval and all per-case
statistics. It verified 132 output hashes, all 66 canonical hashes and all 264
framebuffer byte-L1 totals against the baseline, as well as binary identities,
embedded shader bytes, pairing and frozen-budget chronology. No material finding
remains. The passing timing screen and INCONCLUSIVE adoption verdict are retained
with their stated setup, same-seed and backend limitations.
