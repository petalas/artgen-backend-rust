# Experiment 3, parallel bounds with ordered scatter

The first two candidates removed work outside consumed tiles but failed the
practical-gain gate. Their common premise was that this avoidable work dominated
enough whole-engine time to clear 3%. The paired measurements did not support
that premise at their declared budgets. This experiment changes family, without
combining either rejected patch.

## Mechanism and invariant

The original binner uses one invocation per offspring for vertex decoding,
floating-point bounds, count resets and scatter. Candidate 3 uses 32 lanes to
compute the original final integer tile bounds into a workgroup array. Each lane
owns indices `local_idx + 32*k`. After a workgroup barrier, lane zero performs the
original full count resets and serial polygon scatter in ascending polygon order.

All 15 original decode, AABB, pixel and tile expressions remain source-identical.
The reset and scatter statements, overflow counts, first-256 index order,
bindings and dispatch counts remain unchanged. There are still one logical
candidate evaluation and one workgroup per offspring. Physical lanes do not
increase the evaluation counter. No dirty filtering or new culling is added.

The array has 1,000 entries of four u32 bounds, totaling 16,000 bytes. The existing
requested downlevel workgroup-storage limit is 16,352 bytes. No host limit or
pipeline compilation option changes. Automatic workgroup zero initialization
remains enabled, and its cost is part of this candidate. Each consumed slot has
exactly one writer, including counts below 32 or not divisible by 32. Empty
drawings reach the barrier and reset every count. The sole pre-barrier return
uses a workgroup-uniform offspring-range condition.

Parallel storage access and preprocessing may reduce the serial cost, but that
is a hypothesis. The sampled bin pass was the largest measured stage in prior
profiles. Neither that sample nor this design identifies a hardware bottleneck.
Shared-memory initialization, occupancy and serial scatter can limit this change.

## Proof and measurement protocol fixed before timing

Build the candidate with the existing locked release profile. Archive its
executable, shader, patch, toolchain, lock hash and Cargo-reported profile before
restoring the production source. Run the unchanged consumed-only GPU checker
against the frozen original GPU shader. It covers five tile workgroups, seven
image sizes, packed boundaries, empty and changing dirty regions, and 500-layer
count/order checks. Run the existing GPU rasterization regression as well.
Source equivalence alone does not establish identical compiler output.

Reuse [comparison policy 1](../comparison-policy.md) without threshold changes.
All ff, mona and darwin inputs, above-97% mature parents, resolution, mutation
settings, seed 7, four chains, 64 offspring per chain and 64 iterations per batch
remain frozen. Each timed worker includes ordinary readback and its final drain.

First run five alternating pairs per project at eight measured batches after two
warmup batches, plus one separate candidate diagnostic per project. This is
33 workers at 131,072 completed evaluations each. Advance only if the unchanged
practical-gain, uncertainty, per-case non-regression and correctness gates pass.
If they do, run the same five-pair series at the full 64-batch budget. Otherwise
stop without inferring full-budget throughput. Do not tune this implementation
from the preliminary results.

Every worker must match initial and final canonical state, cached byte-L1 error
and quality. Diagnostic occupancy must remain at most 256, with its throughput
excluded. Keep all pairs, failures and work counts. Isolation remains unverified.
Even a passing full screen cannot return KEEP until broader campaign and live
acceptance gates pass. Production remains on the reference in the meantime.

The frozen schedule and all build, checker and trial evidence are retained under
`.cache/evolution-experiments/bin-parallel-03/`. The candidate patch is
[03-parallel-bounds.patch](03-parallel-bounds.patch).

Attack the Premise changed the hypothesis family after two failed screens.
Prove It Works requires GPU bin equality and exact evolution replay. Explain the
Number keeps stage attribution separate from completed whole-engine throughput.

## Verification completed before scoring

The frozen-original GPU comparison passed all 770 paired checks, comprising
1,540 dispatches. The rasterization regression passed. All 50 Rust tests and 99
Python supervisor/comparator tests passed. Clippy reported no warnings in the
checker or benchmark example; 581 existing library warnings remain. Independent
source review found no material issue. The initial GPU fixtures stopped at 500 polygons. A later checker extension,
without any candidate change, passed 805 paired checks and 1,610 dispatches,
including exact reference counts and ordered prefixes at 999 and 1,000 polygons.
The original 500-layer full, partial and zero-width cases remain. Source index
coverage is exhaustive for counts 0 through 1,000. Broader live and quality
qualification remains open.

The archived shader SHA-256 is
`9a1cf2e57e9470389266ffb66b63cf915c64e742bd3aeeba70dcdf3855b6c34e`.
The benchmark executable SHA-256 is
`21b48ab3590d2e904a54ffd0e5f0963dfc53511313463e1d161adcee2f652025`.
The ordinary production shader was restored byte-for-byte before scoring; its
benchmark executable matches the reference SHA-256
`7bd7fd3410073ac6937fd1e3046e75183ca92e0532d210c333a484a2b9c4f94f`.

## Preliminary result: REJECT

All 33 workers completed with zero errors. The equal-weight paired geometric
mean ratio was **0.99907049**, with conditional 95% bootstrap interval
**0.96725338 to 1.01943438**. This screen found no measurable whole-engine gain.
The practical-gain, aggregate-interval and per-case-interval gates failed.
Per-case median ratios and the minimum pair count passed. No full 64-batch
comparison was run or inferred.

Five alternating timing pairs per project gave these eval/s medians and ranges:

| Project | Reference eval/s | Candidate eval/s | Paired geometric ratio | Paired ratio range |
| --- | --- | --- | --- | --- |
| ff | 65,865 (65,311 to 66,627) | 67,308 (53,820 to 67,482) | 0.976925 | 0.8078 to 1.0309 |
| mona | 53,124 (52,924 to 53,142) | 53,735 (52,931 to 53,822) | 1.008920 | 0.9962 to 1.0156 |
| darwin | 52,774 (52,629 to 54,247) | 53,428 (53,089 to 54,916) | 1.011743 | 0.9865 to 1.0435 |

One ff candidate trial was substantially slower, with paired ratio 0.807773.
It remains included under the frozen no-exclusion policy. The interval captures
that variation; uncontrolled background activity prevents attributing the slow
pair to a specific cause. These data fail the screen. They do not prove that
parallel bounds have no benefit under other conditions.

Every worker completed 131,072 evaluations over 512 iterations and eight batches.
The total was 4,325,376 completed evaluations, of which 393,216 diagnostic
evaluations were excluded from timing. All canonical endpoints and recorded
quality matched their references. Timing repetition uses one seed, rather than
independent search seeds. The same mature inputs and complexity remain intact.

Raw records are retained in
`.cache/evolution-experiments/bin-parallel-03/screen/summary.json`. The original
production shader and ordinary benchmark executable remain restored to the
reference. The candidate is stopped under its preliminary rule. Broader live,
held-out and quality qualification is still open, and no engine change is kept.

## Retained verification

Independent artifact verification recomputed all 66 canonical endpoint hashes
and all 264 cached parent framebuffer byte-L1 totals. All match exactly. The
follow-up capacity checker, with source hash
`68fa8f2d2ce597911a0e47147b996c7f51167abf33d510616c485422aaa8bf21`,
passed on the same unchanged candidate shader. Its binary and process logs are
retained separately under `checkers-capacity/`, so the earlier 770-check proof
remains immutable. All shader, input and timing identities remain unchanged.

## Attribution and limits

The median sampled bin intervals changed from 2.466 to 2.422 ms for ff, 3.240 to
3.182 ms for mona and 3.109 to 3.048 ms for darwin. These are medians over five
workers per side, using only the last-iteration sample from each of eight
batches. Selection still includes framebuffer updates. The intervals are
diagnostic attribution, not whole-run shares or a hardware-utilization estimate.
They show only a small sampled bin reduction, consistent with the screen failing
to establish a material whole-engine gain. They do not identify the hardware
resource that limits scatter. Full per-stage records remain in `stage-analysis.json`.

Benchmark setup medians were about 10.1 to 10.4 seconds for reference and 17.2 to
17.4 seconds for candidate. Setup is outside the throughput score and includes
more than pipeline compilation. This is not an isolated preview-loading result.
The larger shared array's compilation and zeroing costs remain unseparated.

Scoped Rust formatting passes. Workspace-wide `cargo fmt --all -- --check`
reports existing differences in untouched files, beginning with `shared/src/lib.rs`
and `shared/src/mutation_params.rs`. Those files remain unchanged. Logs retain
that failure as well as the passing scoped formatting and Clippy checks.
