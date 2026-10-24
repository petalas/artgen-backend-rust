# Raster cache filters on the faster binner

The first two raster candidates use the committed 512-lane local-count binner
and preserve the complete evolution parameters. Each has two alternating pairs
per project, eight batches, seed 7 and two warmups. All 24 normal trials and nine
untimed overflow replays passed exact canonical, byte-L1, fitness, work and
device checks. Normal work totals 3,145,728 evaluations; diagnostic work is
reported separately and excluded from the score.

| Attempt | Change | Preliminary aggregate ratio | Disposition |
| --- | --- | ---: | --- |
| 11 | Size the shared polygon array to `min(TILE_CAP, 256)` | 1.05931 | Advance filter; no adoption yet |
| 12 | One 256-slot cooperative load, ordered blend, no cache-reuse barrier | 1.12154 | Leading cache filter; test decoded alternatives next |

The first change keeps every original executable loop and arithmetic expression.
At the fixed 32 by 16 raster workgroup it reduces polygon storage from 24,576
bytes to 4,096 bytes. It retains the old 192-slot capacity at 8 by 8, where the
unchanged chunk loop can use two chunks. Each chunk accesses at most
`min(TILE_CAP, tile_poly_count)`, and `tile_poly_count` remains clamped to 256.

The second change replaces chunking with one cooperative load of the entire
clamped tile prefix. At 8 by 8 each lane can load four entries; at the fixed
32 by 16 each lane loads at most one. Every lane reaches the loading barrier,
including those outside partial image tiles. Blend order and arithmetic remain
unchanged. The cache is never overwritten afterward, allowing removal of its
reuse barrier while retaining the separate reduction barrier. Empty tiles gain
one loading barrier. The old main remains cohesive and becomes shorter; its
existing over-page length is retained to avoid unrelated FP helper changes.

The sampled raster intervals moved with engine time. ff's median last-iteration
interval was approximately 0.712 ms in the reference, 0.633 ms with smaller
storage, and 0.580 ms with cooperative loading. These are sampled attribution,
not utilization or full-batch percentages. Full-budget and reserved confirmation
remain required before keeping either source.

The actual raster source installer first rejected a deliberately wrong shader
that reported zero candidate error. Its initial fitness and state failed, and
its throughput was null. Evidence is in
`.cache/evolution-overnight-20261024/raster-negative-source-gate/`.
Filter rows, complete ranges, stage samples and overflow snapshots are retained
in `raster-cache-screen-01/`. The source proofs and capacity predeclaration are
in `raster-variants/`. Patches 11 and 12 apply independently to the unchanged
original raster source. Both templates pass production Naga and GPU creation
through the real resident loop. Candidate-specific numerical GPU fixtures remain
part of any eventual keep decision.
