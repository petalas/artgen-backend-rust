# Raster edge caches and fixed-tile physical lanes

These independent candidates use the kept decoded-AABB raster as reference,
with the same 32×16 logical tiles, complete MutationParams, binner, mutation
shader and trajectories. Only shader execution and cached intermediate values
change. Attempt 26 is now adopted after full confirmation. The final keep report is
[Fixed-tile physical raster lanes](11-fixed-tile-physical-lanes.md).

| Attempt | Mechanism | Preliminary ratio | Filter disposition |
| --- | --- | ---: | --- |
| 17 | Cache vertices, AABBs and edge deltas; 72-byte record | Unrun | Reject capacity proof |
| 25 | Cache vertices and edge deltas instead of AABBs; 56-byte record | 1.016220 | Pass; not selected |
| 26 | At most 256 physical lanes, integer-sum two pixels per lane at 32×16 | 1.072576 | Adopted after confirmation; see report 11 |
| 27 | At most 128 physical lanes, integer-sum four pixels per lane at 32×16 | 1.071301 | Pass; not selected |

Attempt 17 needs 18,432 polygon-cache bytes alone, exceeding the existing
16,352-byte requested workgroup-storage limit. It was rejected from source layout
before GPU execution. The compact edge alternative has a 56-byte stride and a
conservative 15,440-byte total, within that same limit. It preserves the original
pixel AABB expressions and caches the original edge differences, without expanding
or reassociating the edge equations.

The lane candidates retain logical tile dimensions and all original pixel work.
Physical lane count is clamped to the tile pixel count for smaller supported tiles.
Every pixel ordinal is assigned once; each lane evaluates its assigned pixels with
the original floating-point operations, then adds their already-integer new/old
errors. The subgroup reduction uses actual subgroup IDs and sums the same tile
values. The maximum tile error is 512×765=391,680, safely within u32. Cooperative
polygon loading and barriers retain partial-image participation. No cache capacity,
precision, bin clamp, ordering or genetic setting is reduced.

The frozen short screen used two alternating pairs per project, eight batches,
seed 7, two matching warmups, stage timestamps and independent overflow replays.
All 36 normal trials and twelve diagnostics passed exact canonical, cached-byte-L1,
fitness, work and device gates. Normal work totaled 4,718,592 evaluations; diagnostic
rates are null. All slow pairs remain included.

The highest aggregate point estimate, lanes256, advanced to a predeclared five-pair,
64-batch training/reserved confirmation. The other two received no extra samples.
Its candidate-specific release numerical checker passed both fixture loops across
all five supported raster workgroups, including odd image bounds, layered colors,
all reference bytes and the visible-layer regression. Full throughput confirmation
and the independent endpoint audit subsequently passed. Report 11 records the
final keep decision, rates, ranges and exact source identity.

Source proofs, hashes and generators are under `.cache/evolution-overnight-20261024/`
in `raster-edge/`, `raster-edge-compact/` and `raster-lanes/`. Raw short records and
complete rates/ranges are in `raster-lanes-edge-screen-01/`, using immutable
`build-v6/`. The frozen full package is `raster-lanes-confirmation/spec.json`.
A candidate-specific checker build succeeded but its temporary-source restoration
wrapper hit a variable-shadowing error afterward; `restoration.json` records the
verified restoration from immutable C. No compilation or GPU work was repeated
for that reporting error. That restoration was temporary. Production now uses
AE1 intentionally after the separate keep gate passed, as recorded in report 11.

Intervals from two-pair filters are exploratory. Background isolation remains
unverified; no broader speed, hardware utilization or evolution-quality claim
follows from these preliminary ratios.
