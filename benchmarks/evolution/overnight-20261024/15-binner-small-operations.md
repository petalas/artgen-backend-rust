# Reject three small binner operations

All three candidates failed the unchanged preliminary practical gate. They
preserved exact engine results; production remains the 512-lane local-count
binner 598. No full-budget or reserved samples were added.

| Attempt | Mechanism | Aggregate ratio | Conditional 95% interval | Decision |
| --- | --- | ---: | --- | --- |
| 34 | Unsigned extent membership | 0.997214475 | [0.995036533, 0.999397184] | REJECT |
| 35 | Hoist tile-data base | 1.002021900 | [0.997883557, 1.006177405] | REJECT |
| 36 | Plain u32 count store | 0.998671395 | [0.991477445, 1.005917543] | REJECT |

The frozen filter used two alternating pairs per project, eight batches, two
matching warmups and seed 7. All 36 timing trials and twelve independent overflow
replays completed with zero device, quality, state, fitness, capacity or work
errors. Timing work was 4,718,592 evaluations; diagnostics contributed 1,572,864
and were excluded from rates. Three reference qualification runs added 393,216.
Every slow pair remains included. The intervals from two pairs are exploratory.

Unsigned extents replace two half-open comparisons per axis with wrapping
subtraction and an extent comparison; the source proof includes empty/inverted
bounds. The offset candidate hoists the identical original integer expression.
The plain-store candidate relies on the unique tile owner, retains the full
uncapped local count and every first-256 ordered index write, and changes the
count-array view from atomic to plain u32. These are independent changes.
Original float bounds, polygon order, WG512, buffer layouts, mutation G1,
raster AE1 and the complete parameters stay fixed.

The existing diagnostic factory handles the plain-store form only in untimed
replays, rewriting its binding to atomic and retaining the reserved-tail maximum.
All three physical plain-store normal/diagnostic trajectories passed. This does
not upgrade the prior CPU/Naga 257/1000/0 proof into an 835-fixture full-buffer
result; broader buffer checks were not run for these rejected filters.

No useful engine gain was observed. Source operation counts do not prove machine
instructions disappeared, and this screen establishes no hardware bottleneck or
universal statement about atomics or bounds packing.

Evidence under .cache/evolution-overnight-20261024/: generators, proofs, hashes
and independent diffs in bin-micros/; pre-data schedule in
bin-micros-screen-spec.json; immutable release/default sources in build-v9/;
raw rows, medians/ranges and stage samples in bin-micros-screen-01/output/.
Patches 34-36 apply independently to production 598. Background isolation remains
unverified on the recorded RTX 5090 Vulkan/Dozen backend.
