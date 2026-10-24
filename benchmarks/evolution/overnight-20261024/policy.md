# Overnight engine throughput campaign

The user redirected work on 2026-10-24 to GPU evolution engine throughput only,
requested autonomous overnight work, fixed evolution parameters and frequent
commits. WebSocket, UI, loading and mutation-parameter tuning are outside this
campaign. Earlier policies and their INCONCLUSIVE or REJECT results remain
unchanged. This is a new engine-only acceptance contract, not a retroactive
pass of the earlier live-preview gates.

The planning target is at least another 10% over the already confirmed fast
binner, with at least twelve isolated implementation attempts while the night
budget permits. Continue useful work until the morning cutoff rather than
stopping as soon as that target is reached. A documented exhausted hypothesis
family triggers re-profiling, not a parameter search or invented gain.

Work ends at 07:00 UTC, 08:00 Europe/London, after finishing the current bounded
unit. An hourly wake checkpoint checks the same campaign state and never starts
duplicate GPU workers. The mutable checkpoint is
`.cache/evolution-overnight-20261024/state.json`. Every attempt retains its source,
hashes, raw measurements, errors and decision. Commit accepted changes and
rejected experiment records. Do not push or change unrelated processes.

## Fixed work and correctness

Use the immutable ff/mona/darwin mature-drawing corpus from the original user
capture. Every active initial parent must exceed 97% fitness. Keep the complete
MutationParams object fixed, including four chains, 64 offspring, 64 iterations
per batch, raster workgroups, mutation probabilities, adaptive behavior and
crossover. Keep reference images, resolutions, precision, fitness, selection,
polygon limits, RNG rules and candidate work unchanged.

Internal shader execution choices, such as the binner's lane count, are code
changes rather than changes to those parameters. They must preserve exact
results. Reinitialize identical seeded state for each side and warm the same
profiling path. Count completed work after the final checked drain. Include
ordinary encoding, submission, polling and drawing readback in engine time.
Keep resets, pipeline compilation, added qualification, snapshots, hashing and
report writes outside it. Report every submitted and completed count and error.

Compare all seven canonical state fields: packed parents and offspring,
parent errors and framebuffers, offspring errors, chain update flags and packed
fitness. Preserve the existing exclusion of reporting-only control IDs. Require
exact initial and final replay and exact framebuffer-byte RGB L1 against cached
errors. Verify every parent's stored fitness and geometry association. Any
process, device, capacity, replay, overflow or numerical error rejects the
candidate; it cannot become a fast timing sample.

Qualify a reference trajectory once per case and budget using the existing
independent numerical policy. Subsequent exact snapshot equality is a
transitive correctness check; repeating the same scalar rendering for every
timing sample adds no evidence. New shader mechanisms still receive focused
buffer/raster checks before acceptance. Use the existing full-bin checker when
bin contents should remain identical and consumed-bin checks only when an
explicit, proven unread-region optimization permits stale unconsumed words.
Overflow diagnostics must instrument the actual candidate and match its normal
trajectory. Never disable a validity check to accommodate a new mechanism.

## Fast feedback and confirmation

Reuse GPU devices, buffers, compiled shader variants and qualified inputs
within one main-thread process. Serialize GPU jobs under the shared benchmark
lock. Fixed short work budgets are permitted; simpler images or fewer triangles
are not scoring substitutes. Stage timestamps are attribution for sampled
iterations, not utilization or a whole-run percentage.

First measure resident reference/reference repeatability. Freeze that evidence
before interpreting candidate screens. Use two alternating pairs per project
at eight batches for cheap filters, seed 7 and two warmup batches. A filter is
not an acceptance result. Retain all samples, including slow ones.

Advance a correct filter only when its equal-project geometric mean ratio is
at least 1.005 and no case median ratio is below 0.98. Otherwise reject at that
declared budget, without adding samples. Promising candidates get five paired
64-batch trials per project. Resample paired log ratios independently per case
with 10,000 draws, seed 20261024. Require aggregate ratio at least 1.005 and its
conditional 95% lower bound above 1.0; require each case median at least 0.995
and lower bound at least 0.99. These engine-only microoptimization thresholds
are fixed before new candidate data and do not rewrite screening policy 1.

Confirm a proposed accepted change once on the distinct reserved mature corpus
with seeds ff 11, mona 13 and darwin 17, preserving all parameters. Freeze the
confirmation's five paired 64-batch runs per project before starting them and
apply the same gates. Do not tune from confirmation data. Reject an uncertain
result rather than extending sampling until it passes. Profile separately from
scored wall-time comparisons, and explain any micro result with a matching
whole-engine result before adopting it.

## Starting reference and attempts

Candidate 5 from the earlier campaign is the starting fast implementation. Its
exact shader already passed full-buffer, numerical, original-corpus and
distinct-state/seed confirmation at approximately 2.3 times original throughput.
Recheck its identity and seeded engine behavior before committing it as the
engine-only baseline. Keep its automatic bin-cache zero initialization. The
earlier zeroing-removal candidate had no established engine throughput benefit.

Try isolated mechanisms before combining them. First investigate internal bin
workgroup width and redundant global occupancy atomics. Re-profile after wins;
do not assume the original bottleneck remains dominant. Use hyperfine for
isolated host costs when a profile identifies them; process startup is not the
GPU evolution throughput score. No parameter search, approximate rendering,
skipped evaluations, lighter fixtures or selection changes are allowed.
