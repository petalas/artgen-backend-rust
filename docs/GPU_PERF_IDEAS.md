# GPU Evolution Performance Ideas

Brainstormed 2026-02-21. Organized by expected impact.

## High Impact

### 1. Delta / Incremental Evaluation
Most mutations change only 1-2 polygons, yet we re-rasterize the entire image from scratch. Instead:
- Cache the fully rasterized image per chain (one extra W×H×4 buffer per chain)
- Track which polygon(s) the mutation changed (small metadata output from mutate)
- Only re-composite the affected polygon's bounding box: erase old contribution, apply new
- Compute error delta only in the affected region

Turns rasterize cost from O(polygons × pixels) to O(affected_pixels) for single-polygon mutations. Could be 10-50× faster for the rasterize pass. Tricky part is z-order — erasing a polygon mid-stack means re-compositing everything above it in that region.

**Status:** Not started. See detailed analysis below.

### 2. Coarse-to-Fine Screening
Most offspring are rejected. Evaluate at 1/4 or 1/8 resolution first, then only full-res evaluate promising candidates:
- Dispatch cheap low-res rasterize+error for all λ offspring
- Compare to parent's low-res error (precomputed once on acceptance)
- Only dispatch full-res for top-k or within-threshold offspring
- Could skip 80-90% of full-res work

Low-res pass is essentially free on GPU. Needs a filter/compact step between passes.

### 3. Fused Mega-Kernel (Persistent Threads)
Instead of 3-4 separate dispatches per iteration, fuse into one persistent kernel that loops `gpu_batch_iters` times internally using `storageBarrier()`. Eliminates:
- Per-iteration dispatch overhead
- Global memory round-trips between passes
- Separate bin_polygons pass (bin in shared memory instead)

Challenge: rasterize needs many workgroups per chain, so can't truly fuse across chains without device-scope sync.

## Medium Impact

### 4. Stratified / Orthogonal Mutations
Currently each offspring picks independently from the same distribution. Instead ensure λ offspring cover different strategies:
- Offspring 0: move a point (fine)
- Offspring 1: change a color
- Offspring 2: add/remove polygon (structural)
- Offspring 3: crossover
- etc.

Avoids redundant exploration, improves per-iteration information gain. Cheap — just index into mutation schedule in mutate shader.

### 5. Half-Precision (f16) for Color Math
Rasterize shader does f32 color blending. On NVIDIA Ampere+, f16 runs at 2× throughput. Color channels are 8-bit so precision loss is negligible. Use `vec4<f16>` for blend accumulator.

### 6. Polygon Sorting by Screen Coverage
Sort polygons by bounding box area (largest first). Larger polygons establish base colors early. Improves tile culling efficiency.

### 7. Error-Weighted Mutation Targeting
Track per-region error (e.g., 4×4 grid, 16 u32s per chain). Bias point movements and new polygon placement toward high-error regions. Cheap metadata to maintain in select, could improve convergence rate significantly.

## Speculative / Longer-Term

### 8. Population-Level Gradient Estimation (NES/OpenES)
Use fitness of all λ offspring to estimate gradient in parameter space, step in estimated direction. Much better convergence than (1+λ)-ES but complex to implement on GPU.

### 9. Learned Mutation Distributions
Per-chain statistics of which mutation types have been productive recently (accept rate per mutation type). Shift probability toward productive mutations. Lightweight reinforcement signal.

### 10. Multi-Objective / Novelty Selection
Occasionally accept offspring that are *different* from neighbors even if fitness is slightly worse. Prevents premature convergence more gracefully than stagnation burst.

---

## Detailed Analysis: Delta / Incremental Evaluation (#1)

### Core Insight
In a typical iteration with 150+ polygons at 512×512, rasterize_error processes ~39M polygon-pixel tests. But a single-polygon mutation only affects the pixels within that polygon's bounding box (old and new positions). If the polygon covers 5% of the image, we could skip 95% of the work.

### What Changes Per Mutation Type

| Mutation | Polygons affected | Notes |
|----------|------------------|-------|
| move_point | 1 | Bounding box of old + new triangle |
| micro_adjust_point | 1 | Very small region delta |
| change_color | 1 | Same bounding box, just reblend |
| micro_adjust_color | 1 | Same bounding box |
| lighten/darken | 1 | Same bounding box |
| offset_polygon | 1 | Old + new bounding box |
| rotate_polygon | 1 | Old + new bounding box |
| add_polygon | 1 | New polygon's bounding box only |
| remove_polygon | 1 | Removed polygon's bounding box |
| reorder/swap | 2 | Union of both bounding boxes |
| crossover | ALL | Full re-rasterize needed |

~90%+ of mutations affect exactly 1 polygon. Crossover is the exception and falls back to full rasterize.

### Architecture Options

#### Option A: Cached Framebuffer + Dirty Region
- Store a rasterized RGBA image per chain (512×512×4 = 1MB per chain)
- On mutation, mutate shader outputs: `changed_poly_index`, `old_bbox`, `new_bbox`
- New "patch_rasterize" shader:
  1. For each pixel in the dirty region (union of old + new bbox):
     - Re-composite from polygon 0 (or from the changed polygon's layer up)
     - Write result to cached framebuffer
  2. Compute error delta vs reference for dirty pixels
  3. Compute error delta vs old cached pixels for same region
  4. New total error = old_total_error - old_region_error + new_region_error
- On acceptance: cached framebuffer is already correct
- On rejection: restore dirty region from... what? Need a backup.

**Problem:** Need to restore the cached image on rejection. Options:
- Copy dirty region to temp buffer before patching (extra copy)
- Keep two framebuffers and swap (2× memory)
- Re-rasterize dirty region from parent state on rejection (but rejection is 90%+ of the time, so this is expensive)

#### Option B: Layered Prefix-Sum Framebuffer
- Pre-compute "prefix sum" images: image after polygon 0, after 0+1, after 0+1+2, etc.
- To re-evaluate polygon k, blend prefix[k-1] with new polygon k, then continue from k+1
- O(N-k) work per mutation at polygon index k instead of O(N)
- Memory: N × W × H × 4 bytes = 150 × 1MB = 150MB per chain — too expensive

#### Option C: Error-Delta Only (No Cached Framebuffer)
Instead of caching the framebuffer, compute the error *delta* directly:
- Rasterize full image ONLY in the dirty region (old bbox ∪ new bbox)
- Twice: once with old polygon state, once with new polygon state
- new_error = old_total_error + (new_region_error - old_region_error)
- Still O(polygons × dirty_pixels) but dirty_pixels << total_pixels

This avoids the framebuffer cache entirely but still iterates all polygons within the dirty region. For small mutations, dirty_pixels might be 1-5% of total, giving 20-100× speedup on the pixel dimension.

#### Option D: Cached Framebuffer + Snapshot-on-Mutate (Recommended)
Hybrid approach:
1. Each chain has a cached framebuffer (1MB per chain)
2. Mutate shader outputs dirty region metadata
3. Before patching, copy dirty region to a small temp buffer (bounded by max polygon bbox)
4. Patch: re-rasterize dirty region into cached framebuffer
5. Compute error for dirty region
6. If rejected: restore dirty region from temp buffer
7. If accepted: framebuffer already correct, update total error

Memory overhead: 1MB per chain (framebuffer) + small temp buffer (shared across chains since they execute sequentially per offspring)

### Complications

1. **Z-order dependency**: Changing polygon at index k requires re-compositing polygons k through N-1 in the dirty region. Average case: re-composite N/2 polygons. Worst case (polygon 0): all N polygons. But only within the dirty bbox.

2. **Add/remove polygon**: Shifts all subsequent polygon indices. Dirty region = bbox of added/removed polygon, but all polygons above must be re-composited.

3. **Reorder/swap**: Two polygons change position. Dirty region = union of both bboxes. Everything between the two indices must be re-composited in both regions.

4. **Crossover**: Changes everything. Must fall back to full rasterize.

5. **Concurrency**: Each offspring needs independent temp storage. With λ offspring per chain, need λ temp buffers (or serialize offspring evaluation, losing parallelism).

6. **Dispatch shape changes**: Dirty-region rasterize has variable workload per offspring. Either pad to worst case (losing benefit) or use indirect dispatch (additional complexity).

### Estimated Speedup

Assumptions: 150 polygons, 512×512 image, average polygon covers 5% of pixels.

| Scenario | Current cost | Delta cost | Speedup |
|----------|-------------|-----------|---------|
| move_point (early polygon) | 150×262K = 39M ops | 75×13K = 975K ops | ~40× |
| move_point (late polygon) | 39M ops | 10×13K = 130K ops | ~300× |
| change_color | 39M ops | 75×13K = 975K ops | ~40× |
| micro_adjust | 39M ops | 75×1.3K = 97K ops | ~400× |
| add/remove | 39M ops | 75×13K = 975K ops | ~40× |
| reorder | 39M ops | 75×26K = 1.95M ops | ~20× |
| crossover | 39M ops | 39M ops | 1× (fallback) |

Weighted average (crossover ~5% of time): **~30-50× speedup on rasterize pass.**

### Implementation Complexity
High. Requires:
- New buffer: cached framebuffer per chain
- Modified mutate shader: output dirty region metadata
- New shader: patch_rasterize (dirty region only)
- Modified select shader: handle error delta math + framebuffer restore/commit
- Fallback path for crossover / multi-polygon mutations
- Careful synchronization of framebuffer state
