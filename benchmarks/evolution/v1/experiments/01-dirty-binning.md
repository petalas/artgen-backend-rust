# Experiment 1, dirty-region binning

This experiment removes bin construction for tiles the rasterizer cannot read.
It changes no mutation, proposal, scoring, precision, population or image setting.
The candidate is retained as [a patch](01-dirty-binning.patch) and an archived
executable. The production shader remains the reference pending the campaign's
remaining acceptance gates.

## Mechanism

The reference resets every tile count and scatters every polygon into its full
tile AABB for each offspring. Rasterization then skips tiles outside the dirty
box before reading their bin contents. The candidate resets only consumed tile
counts and intersects each polygon's tile range with the consumed rectangle.
It preserves ascending polygon order, full counts and the first 256 indices.

Consumption must match `tile_end > dirty_min && tile_start < dirty_max` on both
axes. Interior zero-width and some inverted dirty boxes still consume a tile.
The candidate preserves that behavior and clips partial edge tiles exactly.
Newly consumed bins always reset before append, so old counts and indices cannot
affect a later dirty region. Boxes with no consumed tiles return before reading
polygons.

The original floating-point polygon AABB expressions remain unchanged. The
diagnostic specialization still instruments the single append marker and tracks
maximum occupancy over all actual bin construction. Full-fitness evaluation
sets the dirty box to the full image before dispatch.

## Identity and checks

Both executables use the same locked optimized build, toolchain, GPU, fixture
manifest and cache seed. The reference is `dd52d50`. The only engine source
difference in the candidate is `src/shaders/bin_polygons.wgsl`.

- Reference executable SHA-256:
  `7bd7fd3410073ac6937fd1e3046e75183ca92e0532d210c333a484a2b9c4f94f`.
- Candidate executable SHA-256:
  `621b4e112cf0a3b56590ee084896fa603a56cbff30ede4c30c1df388fc50da28`.
- Patch SHA-256:
  `64bb6201961794fa782611cc01dbac293b5cbf27da9225ddbb619fb7e767136f`.

Nine full-budget qualification workers passed with zero errors and 9,437,184
completed evaluations including diagnostics. All initial/final canonical states,
quality records and work counters match the user-captured reference. Independent
checks matched all 72 initial/final framebuffer byte-L1 totals.

The candidate-only GPU checker passed 255 dispatches across five supported
workgroups and images 64x48, 65x51 and 17x19. It uses the real pipeline with
checker-owned readable buffers, verifies consumed counts and ordered indices,
and checks that skipped words retain poison or prior contents. It covers dirty
A/B/A transitions, 30 boundary boxes, empty drawings and 255/256/257 layers.
Workgroup cases use fresh bin buffers; retained-buffer workgroup transitions
are not covered by this checker. The existing GPU rasterization checks also
passed. Both checkers run on the main thread with bounded process/readback waits.

The new checker has no Clippy warnings. Existing repository diagnostics remain.
The preserved checker executable/source record precedes two host-only Clippy
cleanups to checked integer conversion and fixed-array chunk iteration. Neither
changes the GPU shader or fixture contents.

## Paired measurements

The frozen [screening policy](../comparison-policy.md) specifies five alternating
reference/candidate pairs per project, the same 64-batch seed-7 trajectory, equal
project weights, separate candidate diagnostics, exact replay and conditional
paired bootstrap intervals. The broader live-preview, held-out state, independent
seed and clean-run qualification gates remain open. A screen cannot return KEEP.

The paired series completed 33 workers with zero errors and **34,603,008
completed evaluations**, including diagnostics. The 30 normal workers account
for 31,457,280 scored evaluations; three diagnostics contribute no timing score.
Each project has five paired timing repetitions of the same seed-7 trajectory.

| Project | Reference median eval/s, range | Candidate median eval/s, range | Paired geometric-mean gain | 95% conditional interval |
|---|---:|---:|---:|---:|
| ff | 68,972; 68,689 to 69,164 | 70,602; 69,964 to 70,775 | +2.20% | +2.00% to +2.35% |
| mona | 54,697; 54,201 to 55,325 | 55,787; 54,849 to 55,951 | +1.30% | +0.70% to +1.99% |
| darwin | 56,415; 56,256 to 56,711 | 57,303; 57,028 to 57,534 | +1.38% | +1.21% to +1.55% |

The equal-weight paired aggregate improves **1.63%**, with a 95% conditional
interval of **1.42% to 1.86%**. This exceeds zero and the historical observed A/A
spread, but falls below the frozen practical threshold of strictly greater than
3%. Every other screen gate passes. The recorded verdict is **REJECT**. The
production shader and normal release benchmark executable were restored to the
reference; no engine throughput change is adopted.

These are exploratory timing observations on one fixed trajectory per project.
Machine isolation remains unverified. Same-seed timing intervals do not measure
independent search quality or future-session noise. The user-captured reference
score remains unchanged; this series compares both binaries again under matched
conditions rather than dividing by historical rates.

Last-iteration sampled bin time falls from 2.541 to 2.467 ms for ff, 3.256 to
3.207 ms for mona, and 3.173 to 3.102 ms for darwin. These are medians across five
workers per side, each averaging 64 final-iteration samples. Binning remains the
largest sampled stage. This is not a whole-run attribution or utilization claim.
The serial per-polygon scan and floating-point AABB calculations remain intact;
they are the next profiling hypothesis rather than an established hardware
bottleneck.

Retained evidence:

- [Complete paired series](../../../../.cache/evolution-experiments/bin-dirty-01/comparison/summary.json).
- [Independent point-ratio and sampled-stage calculation](../../../../.cache/evolution-experiments/bin-dirty-01/paired-analysis.json).
- [Full-budget qualification checks](../../../../.cache/evolution-experiments/bin-dirty-01/qualification/independent-verification.json).
- The paired series' `independent-verification.json` recomputes saved endpoint
  hashes and all 264 framebuffer errors.
- [Independent read-only artifact audit](../../../../.cache/evolution-experiments/bin-dirty-01/result-audit.json)
  reproduced all endpoint hashes, the paired aggregate and bootstrap interval,
  and 24 unique framebuffer error totals. It found no material issue and
  confirmed REJECT under the frozen rule.

## Reproduce

The existing local archived executables can be compared without changing source:

```sh
python3 scripts/compare-evolution.py \
  --baseline .cache/evolution-bench/baseline-20261023T172054.919294Z-95742/summary.json \
  --reference .cache/evolution-experiments/bin-dirty-01/reference/evolution-bench \
  --candidate .cache/evolution-experiments/bin-dirty-01/candidate/evolution-bench \
  --pairs 5 --batches 64 --seed 7
```

To rebuild the candidate, apply `01-dirty-binning.patch` to the reference source,
build the release benchmark and candidate checker with locked dependencies, and
archive them before restoring the reference shader. The checker deliberately
asserts skipped-bin preservation and requires the experimental patch at build
time. Run its archived executable for the retained proof:

```sh
timeout 600 .cache/evolution-experiments/bin-dirty-01/checkers/check-gpu-binning
```

Local build identity, source patch, raw logs, worker states and verification
records are retained under `.cache/evolution-experiments/bin-dirty-01/`.
The append-only decision log is `.cache/hillclimb/decisions.tsv`.

Explain the Number selected binning from separate CPU/GPU profiles and requires
paired completed-work measurements. Build the Lever produced the comparison
command. Prove It Works requires exact replay and direct bin/framebuffer checks.
