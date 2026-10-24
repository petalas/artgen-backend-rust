# Reject subtile masks and cached alpha

Both isolated candidates failed the frozen short throughput gate. Production
remains the kept 256-lane decoded raster AE1. All 24 normal trials, nine overflow
replays and three reference qualifications passed exact state, cached-byte-L1,
fitness, work and device checks. No failure was accepted as a fast sample.

| Attempt | Mechanism | Aggregate ratio | Conditional 95% interval | Decision |
| --- | --- | ---: | --- | --- |
| 32 | Conservative 8×8 subtile mask in padding | 0.945503801 | [0.942705527, 0.948310381] | REJECT |
| 33 | Original alpha unpack cached in padding | 0.997701456 | [0.993335873, 1.002086226] | REJECT |

Each candidate used two alternating pairs per ff/mona/darwin case, eight
measured batches, two matching warmups, seed 7 and the entire fixed manifest.
Normal work was 3,145,728 evaluations; diagnostic work was 1,179,648 and had
null rates. Qualification contributed 393,216 evaluations. Slow pairs remain
included. The two-pair intervals are exploratory; neither candidate advanced,
so no full-budget or reserved run was added.

The mask conservatively encloses packed-coordinate bounds with a one-pixel
margin, then skips polygons that cannot reach the current 8×8 cluster. It uses
byte 44 of the 48-byte decoded record and preserves original blend arithmetic,
pixel ownership, barriers and integer reduction. Source review found no false
negative under its documented dimensions/fallback proof. Exact trajectory and
framebuffer totals passed, but the extra mask construction and per-pixel guard
did not improve the measured engine. These timings do not identify which added
instruction or resource caused the loss.

Alpha caching evaluates the original unpack expression once in the cooperative
loader and stores its f32 alpha at byte 44, again preserving the 48-byte stride.
Packed RGB unpacking and blend operations remain unchanged. Source and GPU
replay support exactness on the recorded backend. Whether the compiler removed
any repeated instruction is unverified; the observed result supplies no useful
throughput gain. No alpha endpoint shortcut or precision reduction was used.

Both experiments used immutable release build-v9, which bundles the kept G1
mutation by default. All three numerically qualified reference endpoints match
the historical G64 mutation endpoints at eight batches, and actual row metadata
records G1. This supplies a bundled-default integration check independent of
the earlier mutation source installer. Binner, raster reference, parameters,
resolution and objective were fixed.

Evidence under `.cache/evolution-overnight-20261024/`: source generators, exact
diffs and hashes in `raster-mask/` and `raster-alpha/`; the frozen schedule in
`raster-mask-alpha-screen-spec.json`; raw rows, complete medians/ranges and stage
samples in `raster-mask-alpha-screen-01/output/`. Patches 32 and 33 apply
independently to exact AE1. Candidate-specific broader raster fixtures were not
run for these rejected filters. Background isolation remains unverified.
