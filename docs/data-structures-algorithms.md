# Data Structures & Algorithm Recommendations

**Expert focus**: AABB culling, delta evaluation, tile-based binning, error metrics, polygon representation

---

## Executive Summary

The biggest wins come from reducing redundant work: AABB culling eliminates 90-97% of polygon-pixel tests, and dirty-rectangle evaluation can reduce per-iteration cost by 100-1000x for common mutations. The current AoS layout is already correct for the broadcast read pattern — no change needed.

---

## 1. Bounding Box Early-Out in Rasterization

**Problem**: Every pixel tests every polygon with 3 edge evaluations. Most polygons are small (~3% of image area at `NEW_POINT_MAX_DISTANCE = 0.03`). A pixel tests all 1000 polygons but is inside maybe 5-10.

**Fix**: Compute AABB from the 3 vertices before edge testing:

```wgsl
let min_x = min(poly.v0.x, min(poly.v1.x, poly.v2.x));
let max_x = max(poly.v0.x, max(poly.v1.x, poly.v2.x));
let min_y = min(poly.v0.y, min(poly.v1.y, poly.v2.y));
let max_y = max(poly.v0.y, max(poly.v1.y, poly.v2.y));

if fx < min_x || fx > max_x || fy < min_y || fy > max_y {
    continue;
}
```

4 comparisons vs 6 mul + 6 add for edges. The min/max is only 4 instructions on data already in registers.

**Expected speedup**: 3-10x reduction in rasterize work depending on average polygon size.

---

## 2. Drop sqrt in Error Computation

**Current**: `sqrt(dr*dr + dg*dg + db*db)` — sqrt costs 4-8 GPU cycles vs 1 for multiply.

**Options**:
- **Sum of absolute differences**: `|dr| + |dg| + |db|`, max = 765. Simple, no overflow.
- **Squared differences** (better ranking properties but risks u32 overflow at 384x384):
  - `(dr*dr + dg*dg + db*db) / 64`, max per pixel ≈ 3048. Total fits u32.
  - Or accumulate as f32 in shared memory.

For practical simplicity, L1 (sum of abs diff) is recommended. Update `MAX_ERROR_PER_PIXEL` from 441.67 to 765.0.

---

## 3. Tile-Based Polygon Binning

**Problem**: Even with AABB culling, each pixel iterates 1000 times checking bounding boxes (48 bytes per polygon load = 48 KB of sequential reads per pixel).

**Approach**: Divide image into 16x16 tiles. Pre-pass bins each polygon into tiles its AABB overlaps. Rasterize kernel only iterates polygons in its tile.

**Sizing** (384x384 with 16x16 tiles = 24x24 = 576 tiles):
- Average polygon overlaps ~4 tiles, 500 active polygons → ~3-4 polygons per tile
- Max 256 per tile: 576 * 256 * 4 bytes = 590 KB per chain, ~37.7 MB for 64 chains

**Ordering**: Alpha blending requires polygon order. Since binning iterates polygons in order and appends to lists, order is automatically preserved.

**Expected speedup**: Each pixel processes 3-10 polygons instead of 1000 → 10-50x net improvement.

---

## 4. Dirty-Rectangle Delta Evaluation (Highest Potential)

Most mutations affect only one polygon. Re-rasterizing the entire image is massive overkill.

### 4a. Dirty-rectangle-only rasterization

Track which polygon was mutated and the union of old/new AABBs. Only rasterize and evaluate error within the dirty rectangle.

For a typical small polygon mutation (AABB ~6x6 pixels):
- Current: 147K pixels × 1000 polygons = 147M operations
- Delta: 36 pixels × 1000 polygons = 36K operations → **~4000x reduction**

**Requirements**:
- `dirty_bbox: vec4<u32>` in `DrawingState`
- `per_pixel_error` buffer (384*384*4 = 590 KB per chain)
- Error delta shader: subtract old dirty-region error, add new
- Full rasterization fallback for structural mutations (add/remove/reorder)

### 4b. Layer checkpoints (maximum win, maximum effort)

Store composited images at every 50th polygon (20 checkpoints). When polygon i changes, restart from checkpoint[i/50] and re-composite 50 polygons over the dirty rectangle.

Memory: 20 * 590 KB = 11.8 MB per chain — too much at scale. **Not recommended**.

**Approach 4a is the sweet spot**: 100-500x improvement for common mutations.

---

## 5. Error-Guided Polygon Placement

**Problem**: Random polygon placement wastes mutations on already-well-matched regions.

**Fix**: Compute a small error heatmap (24x24 tiles) as a byproduct of error_reduce. In mutate, use rejection sampling to bias new polygons toward high-error tiles:

```wgsl
var tile_x = rand_u32(&rng, 24u);
var tile_y = rand_u32(&rng, 24u);
let tile_error = error_heatmap[tile_y * 24 + tile_x];
if rand_f32(&rng) > tile_error / max_tile_error {
    // retry or fall back to random
}
origin_x = (f32(tile_x) + rand_f32(&rng)) / 24.0;
origin_y = (f32(tile_y) + rand_f32(&rng)) / 24.0;
```

Doesn't speed up execution but improves **convergence rate by 3-10x**.

---

## 6. SoA vs AoS Layout — No Change Needed

The current AoS layout is correct for the GPU access pattern. All threads in a workgroup read the same polygon for the same chain — this is a broadcast read that GPU hardware handles via L1 cache. SoA would actually hurt by spreading data across multiple buffers.

---

## 7. Avoid Full DrawingState Copy in Mutate

The mutate shader already copies only up to `polygon_count` (not all 1000). This is already correctly bounded. No change needed.

A more advanced approach (copy only mutated polygon, read unmutated from chain_states directly in rasterize) interacts with dirty-rectangle evaluation — implement them together.

---

## 8. Multi-Scale / Coarse-to-Fine Evolution

Early in evolution, run at lower resolution (96x96 or 192x192) where rasterization is 4-16x cheaper. Once improvement rate plateaus, switch to full 384x384.

**Expected**: Early evolution 4-16x faster. Total time to given fitness drops 30-50%.

---

## 9. Perceptual Error Metric

Humans are more sensitive to luminance differences than chrominance. Simple weighted channels:

```wgsl
let dist = 2.0 * dr * dr + 4.0 * dg * dg + 3.0 * db * db;
```

Weights green (dominates human luminance perception) most heavily. 80% of perceptual benefit for zero extra cost. Full CIE Lab requires expensive cube root — not worth it.

---

## Priority Summary

| # | Recommendation | Speedup | Effort |
|---|---------------|---------|--------|
| 1 | AABB early-out in rasterize | 3-10x on rasterize | ~10 lines WGSL |
| 2 | Drop sqrt (L1 or scaled L2) | 5-10% on error pass | ~5 lines |
| 5 | Error-guided polygon placement | 3-10x convergence | ~30 lines WGSL |
| 9 | Weighted RGB error | Better visual quality | ~3 lines WGSL |
| 3 | Tile-based polygon binning | 10-50x on rasterize | New shader + buffer |
| 4a | Dirty-rectangle evaluation | 100-1000x common mutations | Architectural change |
| 8 | Multi-scale resolution | 30-50% total time | Moderate pipeline changes |
