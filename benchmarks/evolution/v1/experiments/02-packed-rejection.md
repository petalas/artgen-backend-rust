# Experiment 2, reject irrelevant packed polygons before decoding

This candidate is measured alone against `dd52d50`, without experiment 1's tile
clipping. It skips whole polygon scans when integer bounds prove that the polygon
cannot enter any tile consumed by rasterization. All tile counts still reset.
Surviving polygons use the original floating-point bounds, full scatter ranges,
layer ordering and overflow handling. No mutation, fitness, precision, population
or image setting changes.

## Bound and scope

The fast path applies only when both positive image axes are at most 512 pixels.
Other dimensions use the original full binner. Full-image consumption bypasses
the packed test. Empty consumer ranges return only after all tile counts reset.
Consumer bounds follow the rasterizer's strict tile overlap predicate, including
zero-width and inverted boxes that still consume an interior tile.

For one image axis of size d, let S be the first consumed tile's pixel start and
E the last consumed tile's exclusive pixel end. Packed extrema are qmin/qmax in
0..65535. Reject on that axis only under one of these conditions:

- S is positive and qmax is at most floor((S-1) * 65535 / d).
- E is less than d and qmin is at least ceil((E+1) * 65535 / d).

The one-pixel guard covers old floating-point error. A conservative bound for
the original decode and multiplication at d <= 512 is
512 * 2.5 * 2^-23 + 2^-14 = 0.000213624 pixels, below one pixel. The bound uses
[WGSL floating-point accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy)
and accounts for permitted
[reassociation](https://www.w3.org/TR/WGSL/#reassociation-and-fusion).
This is a conservative enclosure, not a claim of CPU/GPU bit identity.

Right rejection is disabled at the image edge. The original scatter can append
a degenerate polygon at normalized coordinate 1 to a partial final tile. That
index must remain when the tile is consumed. Threshold products and the integer
ceiling addition fit u32 in the guarded domain; true branches prevent unsigned
subtraction or division outside its range.

## Verification design

CPU probes checked the guarded enclosure over 67,108,864 coordinate/dimension
pairs and 201,326,592 endpoint cases. They also found that division, reciprocal
lowering and exact rational arithmetic can disagree at integer pixel or tile
boundaries. Those probes are source/CPU evidence only.

The explicit `--scope consumed-only` GPU checker therefore compares against a
frozen copy of the original GPU binner, compiled with matching workgroups and
bound to separate readable buffers. It checks consumed counts and ordered
first-256 indices. The unchanged strict default retains experiment 1's
poison/stale-word and rational-oracle contract.

Cases include 512 and 513 on both axes, partial image edges, threshold-adjacent
packed coordinates, zero/inverted dirty boxes, full-image bypass, changing dirty
regions and 500 overlapping polygons. The frozen original binner is
[reference-bin-dd52d50.wgsl](reference-bin-dd52d50.wgsl), SHA-256
`27bbf8473e9ae1f68cb7ccb3a747e6f6fb8b9106bd7a42e16e88ea017a88f61f`.
Reference output is not inferred from CPU floating-point behavior.

The differential GPU run passed 770 paired checks, or 1,540 actual dispatches,
over five workgroups and seven image sizes. The existing GPU rasterization
regression also passed. All 50 Rust tests pass, and Clippy reports no warning in
the updated checker. Source and checker reviews found no material issue. These
checks qualify the candidate for measurement, not production adoption.

## Measurement stages fixed before timing

Reuse [comparison policy 1](../comparison-policy.md), its statistical rules and
the immutable ff/mona/darwin inputs. Each project retains its mature drawing,
384-pixel longest side, above-97% initial fitness, seed 7, four chains, 64 offspring
per chain, 64 iterations per batch and two warmup batches.

First run five alternating pairs per project at eight measured batches, plus
one separate candidate diagnostic per project. Each worker completes 131,072
evaluations; all 33 workers account for 4,325,376 evaluations including diagnostics.
This is explicitly a preliminary screen, not a replacement for the full budget.
Both sides receive exactly the same logical work and complete the final drain.

Advance to the full 64-batch, five-pair series only if this preliminary run passes
the unchanged practical gain, confidence, per-case non-regression and correctness
gates. Otherwise stop this candidate without claiming its full-budget throughput.
Do not tune its implementation from these timing results. Production adoption
still requires the remaining live, held-out and campaign qualification gates;
the generic comparator cannot return KEEP.

The reference/candidate executables, source patch, CPU probes, GPU logs and all
run records are retained under `.cache/evolution-experiments/bin-scan-02/`.
The preliminary schedule is frozen in `screen-budget.json` there. No clean-machine
or hardware-utilization claim is made.

Explain the Number separates a source opportunity from a measured gain. Build
the Lever reuses the paired runner. Prove It Works requires differential GPU bins
and exact evolution replay before interpreting throughput.

## Preliminary result: REJECT

All 33 workers completed successfully with zero errors. The equal-weight paired
geometric mean ratio was **1.02117024**, a measured **2.12%** gain over the
reference at the eight-batch budget. Its conditional 95% bootstrap interval was
**1.00991579–1.03491225**. The point estimate fails the unchanged strict 3%
practical-gain gate. No full 64-batch series was run or inferred.

Five alternating pairs per project produced these eval/s medians and ranges:

| Project | Reference eval/s | Candidate eval/s | Paired geometric gain | Paired ratio range |
| --- | --- | --- | --- | --- |
| ff | 65,999 (64,199–66,035) | 66,700 (66,465–67,565) | 2.11% | 1.0066–1.0390 |
| mona | 52,394 (48,226–52,558) | 53,044 (53,000–53,070) | 2.95% | 1.0097–1.0990 |
| darwin | 52,654 (52,398–54,035) | 53,147 (52,733–54,700) | 1.30% | 0.9788–1.0389 |

The total was 4,325,376 completed evaluations, including 393,216 diagnostic
evaluations excluded from timing. All initial and final canonical states matched
their reference trajectories. Independent saved-state verification recomputed all
264 parent framebuffer byte-L1 totals and found exact cached-error agreement.
No pair was removed for being slow. Background activity was not controlled, and
these five timing pairs repeat one seed rather than independent search seeds.

The production shader and ordinary benchmark executable were restored to the
reference. The patch is retained in [02-packed-rejection.patch](02-packed-rejection.patch);
raw records are in `.cache/evolution-experiments/bin-scan-02/screen/summary.json`
and `independent-verification.json`. This candidate is stopped under its declared
preliminary rule. It establishes no full-budget or production speedup.

Both avoid-work attempts remained below the practical threshold. Their shared
premise was that removing work outside dirty tiles would provide a material
whole-engine gain. The two paired measurements did not support that threshold.
The next isolated experiment changes family: parallel bounds preprocessing while
retaining serial ordered scatter. It does not combine the rejected patches.
