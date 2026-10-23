# Experiment 4, combine dirty tiles and packed rejection

The first two mechanisms were measured independently. Dirty-region binning gave
an observed 1.625% gain at its full budget. Packed polygon rejection gave 2.117%
at its preliminary budget. Both failed the strict 3% practical-gain gate. These
are different budgets and conditions, and their gains cannot be added to predict
this candidate. Experiment 3's parallel bounds mechanism is not included.

## Hypothesis and preserved behavior

Combine work avoidance at two points in the original serial binner. Reset and
fill only bins that rasterization can consume. Before unpacking polygon vertices,
reject a polygon only when conservative packed integer bounds prove it cannot
contribute to any consumed bin. Surviving polygons retain the original float
bounds and their original ascending layer order, with scatter clipped to the
consumed tile rectangle.

Both changes remove operations that cannot affect a consumed bin. For each
consumed tile, omitted polygons contribute no entries, and clipping retains all
entries that the tile would receive. Thus raw counts and ordered first-256
indices must match the original binner, including overflow counts. Every consumed
count resets, including when no polygon overlaps. Unconsumed words remain stale.
A newly consumed tile resets before its next use.

Use the rasterizer's literal strict per-axis predicate, with clipped final tiles.
Interior zero-width and inverted dirty boxes can still consume a tile. Decoded
u16 bounds outside the image cannot create an out-of-image consumer. The packed
pretest retains experiment 2's one-pixel guard, only for positive dimensions up
to 512 on both axes. Outside that range, retain dirty-tile filtering but disable
packed rejection. At a partial final image tile, right rejection remains disabled
when its consumed end reaches the image edge. Full-image consumption bypasses the
packed test. No guard subtracts or divides outside an actual checked branch.

The reference-byte rendering, original float AABB calculations, opacity, geometry,
RNG, mutation, fitness, selection, capacities, dispatch and settings remain
unchanged. Keep one invocation per offspring and the existing diagnostic marker.
No new bindings, allocations, shared arrays or compilation options are added.
The performance effect of composing the changes is unknown until measured.

## Gates fixed before timing

Archive the exact shader patch, executable, build profile, lock/toolchain and
reference identity before timing. The shader differs from `dd52d50` only in bin
construction. Production is restored to the original before interpreting results.

Run both existing bin checks from the unchanged checker. Strict mode tests all
consumed counts/indices and every stale word, with its 255-dispatch suite.
Consumed-only mode compares against the frozen original GPU binner across five
workgroups and seven images, including packed-coordinate boundaries, partial
edges, zero/inverted boxes, changing dirty regions and counts 500/999/1000. It
performs 805 paired checks and 1,610 dispatches. Run the existing GPU rasterization
regression too. Check compiler validation and source invariants independently.

Reuse [comparison policy 1](../comparison-policy.md) with its exact script and
thresholds. The frozen mature ff, mona and darwin parents all start above 97%, at
the actual 384-pixel longest side. Keep seed 7, four chains, 64 offspring per
chain, 64 iterations per batch, every mutation setting and two warmup batches.
Setup and extra verification stay outside the score; ordinary readback and the
final checked drain stay inside it.

First run five alternating pairs per project at eight measured batches, plus
one separate candidate overflow replay per project. All 33 workers complete
131,072 evaluations each, totaling 4,325,376 evaluations including diagnostics.
Use the unchanged strict aggregate ratio above 1.03, uncertainty, per-case
non-regression and correctness gates. Preserve every slow pair and error.

Advance only on a passing preliminary screen, to a separately frozen series of
five pairs per project at 64 batches. Otherwise stop this candidate without a
full-budget throughput claim. Do not tune the candidate after its preliminary
result. No KEEP is possible before live, held-out and broader campaign gates.
Background activity is uncontrolled, and isolation remains unverified. Leave
other processes untouched.

The schedule and all evidence are retained under
`.cache/evolution-experiments/bin-combined-04/`. Build the Lever reuses the same
runner; Prove It Works requires real GPU outputs and exact saved-state replay;
Explain the Number keeps completed engine throughput separate from sampled stages.

## Verification completed before scoring

Both GPU bin suites passed: 255 strict dispatches and 805 paired comparisons,
or 1,610 actual dispatches, against the frozen original GPU binner. The GPU
rasterization regression passed. All 50 Rust unit tests passed, including shader
variant validation. At scoring time, the checker and Python scripts were unchanged from their
previously passing formatting, Clippy and 99-test supervisor/comparator suite.
No Python test rerun was needed for this shader-only candidate.

Independent source review found no material issue. Its CPU checks covered
9,043,968 tile cuts and 262,656 guarded thresholds, with maximum numerator
33,554,431. These are source/CPU checks, separate from the actual GPU suites.

Candidate shader SHA-256:
`3acd05792f2334314e7900399111c3484d3b0e0c40d7903de49c6637cf72c034`.
Candidate executable SHA-256:
`a8e29b580824c1d13e3ccba7a2f72c6dd530e96a4181813efc295aa49b610b8f`.
The normal executable was restored to reference SHA-256
`7bd7fd3410073ac6937fd1e3046e75183ca92e0532d210c333a484a2b9c4f94f`,
and the production shader is byte-identical to `dd52d50`. Both runtime checks
and timing use immutable archived executables.

## Preliminary result: REJECT

All 33 workers completed with zero errors. The equal-weight paired geometric
mean ratio was **1.00226610**, with conditional 95% bootstrap interval
**0.99538079 to 1.00934560**. This screen found no measurable whole-engine gain.
The unchanged practical-gain, aggregate-interval and per-case-interval gates
failed. Per-case median ratios and the minimum pair count passed. No full
64-batch series was run or inferred.

Five alternating pairs per project produced these eval/s medians and ranges:

| Project | Reference eval/s | Candidate eval/s | Paired geometric ratio | Paired ratio range |
| --- | --- | --- | --- | --- |
| ff | 66,540 (65,644 to 66,627) | 66,755 (65,882 to 67,101) | 1.003702 | 0.9911 to 1.0102 |
| mona | 53,046 (52,925 to 53,146) | 53,274 (52,559 to 53,457) | 1.002255 | 0.9908 to 1.0063 |
| darwin | 54,334 (52,614 to 54,418) | 53,268 (52,996 to 54,708) | 1.000843 | 0.9754 to 1.0374 |

Every worker completed 131,072 measured evaluations. The total was 4,325,376,
including 393,216 diagnostic evaluations excluded from timing. All canonical
endpoints, quality, complexity, parameters and adapter identity match. Every
pair remains included. Repeated seed 7 estimates timing variation rather than
independent search quality; background activity remains uncontrolled.

The combination did not reproduce an additive benefit. These data do not
identify whether overlapping work savings, compiler effects or machine activity
explain that result. The next step returns to attribution: amplify the complete
original bin body without changing logical evolution, and compare sampled bin,
full-batch compute and checked wall time. This is a diagnostic, not an adoption
score or a fifth optimization attempt.

Production stays at the original shader and reference executable. The candidate
patch is [04-combined-binning.patch](04-combined-binning.patch). Raw evidence is
retained in `.cache/evolution-experiments/bin-combined-04/screen/summary.json`.

Independent saved-state verification recomputed all 66 canonical endpoint hashes
and all 264 cached framebuffer byte-L1 totals, with exact agreement. A later
checker extension supports the separate bin-amplification diagnostic; candidate
4's earlier checker source, executable and logs remain immutable in its bundle.
