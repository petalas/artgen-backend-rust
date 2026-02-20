# Data Structures & Algorithms Analysis

Detailed analysis of data layout, memory efficiency, and algorithmic complexity in the artgen-backend-rust GPU evolution pipeline. Focuses on new optimization opportunities beyond the already-implemented GpuPolygon packing, tiled polygon prefetch, workgroup error reduction, and per-offspring persistent RNG.

---

## 1. GPU Memory Layout: AoS vs SoA for Polygons

### Current Layout (AoS)

`GpuDrawingState` stores polygons as an Array-of-Structures:

```
struct DrawingState {
    polygon_count: u32,      // 4B
    fitness_bits: u32,       // 4B
    mutation_scale: f32,     // 4B
    stagnation_counter: u32, // 4B
    rng_state: vec4<u32>,    // 16B
    polygons: array<Polygon, 1000>,  // 16000B
}
// Total: 16032 bytes per chain
```

Each `Polygon` is `vec4<u32>` = 16 bytes: `[color_packed, v0, v1, v2]`.

### Opportunity: Split Color from Geometry

In the rasterize_error shader, every pixel iterates over all polygons but only accesses color data when the pixel is **inside** the triangle (typically a minority of iterations for small triangles). The AABB cull skips many polygons entirely, but for polygons that pass the AABB check, color and vertices are in the same 16-byte record, so loading the polygon into registers/shared memory always loads the color even if the half-space test rejects the pixel.

**Proposed SoA layout** (per-chain):
```
vertices: array<vec3<u32>, 1000>   // 12B per polygon (v0, v1, v2)
colors:   array<u32, 1000>         //  4B per polygon (packed RGBA)
```

**Estimated benefit**: During rasterization, the shared memory tile (currently 768 polygons * 16B = 12,288B) could hold 768 * 12B = 9,216B for vertices only. This is a 25% reduction in shared memory pressure per tile, or equivalently the tile capacity increases to 1024 polygons per tile (1024 * 12B = 12,288B), reducing tile count and barrier synchronization by ~25% for large drawings.

**Estimated cost**: The mutation shader and select shader would need to access two separate arrays instead of one `vec4<u32>`. More complex addressing. The color array would need a second cooperative load pass in the rasterize shader for polygons that pass the half-space test.

**Verdict**: Moderate benefit. The main advantage is fitting more polygons per shared memory tile in the rasterizer. Worth prototyping if rasterize_error is the bottleneck (which timestamp profiling shows it is, typically 60-80% of GPU time). However, the complication of lazy color loading (only when inside) requires per-pixel divergent memory access, which may reduce occupancy benefits. **Recommend profiling the ratio of inside-pixels to AABB-pass-pixels to quantify potential gains before implementing.**

---

## 2. Buffer Packing and Padding Analysis

### GpuDrawingState Padding Audit

```
polygon_count:      offset 0,  4 bytes
fitness_bits:       offset 4,  4 bytes
mutation_scale:     offset 8,  4 bytes
stagnation_counter: offset 12, 4 bytes
rng_state:          offset 16, 16 bytes (vec4<u32>)
polygons:           offset 32, 16000 bytes
Total: 16032 bytes
```

This is tightly packed with no wasted padding. The 32-byte header aligns perfectly to `vec4<u32>` boundaries (WGSL struct alignment rules).

### Opportunity: Reduce Header to Enable Power-of-2 Alignment

The total state size (16032B) is not a power of two. GPU memory controllers work most efficiently with power-of-2 aligned strides for coalesced access patterns. Padding each state to 16384B (16 KiB) would waste 352 bytes per chain (2.2% overhead) but could improve memory access coalescing when multiple chains are accessed in strided patterns (e.g., the migration shader reading `chain_states[neighbor_id]`).

**Verdict**: Low priority. The SSBO access pattern is primarily sequential within a single chain (one thread reads its own chain). Strided cross-chain access only happens during migration (rare) and tournament selection (one chain at a time). The 2.2% memory overhead per chain is not justified for the marginal coalescing improvement.

### Opportunity: Compact Header by Combining Fields

The `stagnation_counter` (u32, max value ~10000) and `polygon_count` (u32, max 1000) could be packed into a single u32 (10 bits for polygon_count, 22 bits for stagnation_counter = up to 4M). This would save 4 bytes per chain but complicate shader code with bitfield extraction.

**Verdict**: Not worthwhile. The header is already only 32 bytes vs 16000 bytes of polygon data. Saving 4 bytes (0.025% of total) adds complexity for no measurable gain.

---

## 3. Polygon Sorting for Better Rasterization Cache Behavior

### Current Behavior

Polygons are stored in painter's algorithm order (back-to-front for correct alpha blending). This order is essential for visual correctness and cannot be changed for the rasterization pass.

### Opportunity: Spatial Sorting Within Same-Depth Groups

Since alpha blending is order-dependent, polygons cannot be globally reordered. However, within the AABB culling step, spatial locality matters: if polygons that are spatially close are also close in the array, the AABB cull can reject entire contiguous ranges early, improving branch prediction and reducing wasted iterations.

**Analysis**: In practice, the evolutionary process naturally tends to produce polygons that are somewhat spatially distributed (the algorithm adds polygons near reference features). The polygons are not randomly scattered. Furthermore, the AABB cull is already per-pixel, so spatial ordering would only help if it improved the **early exit** rate, which requires a two-pass approach (bounding box scan + inside test) that would add more overhead than it saves.

**Verdict**: Not recommended. The painter's order constraint is fundamental to correctness. Any reordering within "same-depth groups" would require defining what constitutes a group (polygons at the same z-layer), which doesn't exist in this flat 2D model. The AABB cull already provides per-polygon spatial filtering.

---

## 4. Spatial Data Structures for Rasterization

### Current Approach

The rasterize_error shader does a brute-force loop over all polygons per pixel, with two optimizations:
1. Tiled shared memory prefetch (768 polygons per tile, cooperative load)
2. AABB culling per polygon per pixel

### Opportunity A: Precomputed Bounding Box Buffer

Separate the bounding box computation from the per-pixel loop. Currently each pixel recomputes AABB min/max for every polygon. With 1000 polygons and 512x512=262,144 pixels, that's 262M redundant AABB computations per offspring.

**Proposed approach**: Add a lightweight pass (or compute at mutation time) that stores precomputed `vec4<f32>` bounding boxes (min_x, min_y, max_x, max_y) per polygon in a separate buffer. The rasterizer loads these from shared memory and performs the AABB test before loading the full polygon data.

**Analysis**: The AABB computation is 6 min/max operations on unpacked vertices. Unpacking the vertex (2 shifts, 2 ANDs, 2 multiplies) is the expensive part. With precomputed AABBs, the rasterizer would:
1. Load AABB (4 floats = 16B) from shared memory
2. Test pixel against AABB (4 comparisons)
3. Only on hit: load full polygon (16B) and do half-space test

However, this doubles the shared memory requirements (16B AABB + 16B polygon per entry), halving the tile size from 768 to ~384 polygons and doubling the number of tiles and barriers. The savings from avoiding vertex unpacking on rejected polygons must outweigh the cost of more tiles.

**Estimated break-even**: If >50% of polygons are rejected by AABB per workgroup (which is typical for small polygons), the precomputed AABB approach wins. For large polygons that cover most of the image, it loses.

**Verdict**: Worth investigating for drawings with many small polygons (the common case at high polygon counts). Could be implemented as a separate `precompute_aabb` pass that runs once per iteration, storing results in a transient buffer that the rasterizer reads.

### Opportunity B: Hierarchical Tile Culling (Two-Level Rasterization)

Instead of testing every polygon against every pixel, use a coarse pass that determines which polygons overlap each 16x16 tile, then a fine pass that only tests the relevant polygons.

**Proposed approach**:
1. Coarse pass: For each 16x16 tile, test all polygon AABBs against the tile bounds. Produce a per-tile polygon list (indices).
2. Fine pass: Each workgroup only iterates over its tile's polygon list.

**Analysis**: This is essentially a GPU-side bounding volume hierarchy / binning approach. The coarse pass would require:
- A buffer to store per-tile polygon lists (variable length)
- Either a fixed-size allocation (e.g., max 1000 entries per tile) or a prefix-sum-based dynamic allocation

For a 512x512 image with 16x16 tiles: 32x32 = 1024 tiles. With 1000 polygons and average ~10% coverage per polygon, each tile would have ~100 relevant polygons (vs 1000 brute-force). This is a 10x reduction in inner-loop iterations.

**Cost**: The coarse pass itself is O(tiles * polygons) = O(1024 * 1000) = ~1M operations, which is cheap. The main challenge is memory allocation for variable-length per-tile lists.

**Implementation strategy using indirect dispatch**:
1. `coarse_cull` shader: For each (tile, polygon) pair, atomicAdd a counter per tile, write polygon index to a flat buffer at the computed offset.
2. `rasterize_error` shader: Each workgroup reads its tile's polygon count and iterates only over the relevant polygons.

**Verdict**: High potential for large polygon counts (500+). The brute-force approach scales as O(pixels * polygons), while the tiled approach scales as O(pixels * avg_polygons_per_tile). For 1000 polygons with ~10% average tile coverage, this could reduce rasterization work by ~10x. **This is the single highest-impact optimization in this analysis.** The main implementation challenge is the per-tile polygon list memory management, which requires either a worst-case allocation or a prefix-sum compaction pass.

---

## 5. Error Reduction Algorithm

### Current Approach

The rasterize_error shader uses a standard binary reduction in shared memory:
```wgsl
var stride = 128u;
while stride > 0u {
    if local_idx < stride {
        shared_errors[local_idx] += shared_errors[local_idx + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
}
```
This is 8 steps for 256 threads, with a `workgroupBarrier()` at each step.

### Opportunity: Warp-Level Reduction to Eliminate Barriers

On modern GPUs (Vulkan subgroup operations), threads within a single warp/wavefront can communicate without shared memory or barriers. WGSL supports `subgroupAdd` (via the `subgroups` extension in newer wgpu versions).

**Proposed approach**:
```wgsl
// Phase 1: Subgroup reduction (no barriers needed)
let subgroup_sum = subgroupAdd(pixel_error);
// Phase 2: One thread per subgroup writes to shared memory
if subgroupElect() {
    shared_errors[subgroup_id] = subgroup_sum;
}
workgroupBarrier();
// Phase 3: Final reduction across subgroups (only 4-8 active threads for 256-thread workgroup)
if local_idx < subgroup_count {
    // tree reduction on just 4-8 values
}
```

For a 256-thread workgroup with 32-thread warps: 8 subgroups. Current approach: 8 barriers. Proposed approach: 1 barrier. This eliminates 7 barrier synchronizations per workgroup per iteration.

**Availability**: The `subgroups` feature requires wgpu 0.20+ and Vulkan 1.1 with `VK_KHR_shader_subgroup`. Most modern discrete GPUs support this. The WSL2/Dozen driver may or may not expose it.

**Verdict**: Medium priority. The reduction is a small fraction of the rasterizer's total work (the polygon loop dominates), but eliminating 7 barriers is free performance. Requires checking adapter feature support at runtime and falling back to the current approach if unavailable.

---

## 6. Memory Bandwidth Analysis

### Per-Iteration Bandwidth Budget

For a configuration with K=16 chains, lambda=8, 512x512 image, 150 polygons per drawing:

**Mutate pass**:
- Read: K * DrawingState = 16 * 16032B = 250 KB (chain_states)
- Write: K * lambda * DrawingState = 128 * 16032B = 2.0 MB (working_states, mostly polygon copy)
- Total: ~2.3 MB

**Rasterize+Error pass** (dominant):
- Read: K * lambda * (polygon_count * 16B tiled reads + reference texture reads)
  - Polygon data: 128 * 150 * 16B = 300 KB (but each read through shared memory, so global read is 1x)
  - Reference texture: 512 * 512 * 4B = 1 MB (per offspring, but texture cache reuse across offspring is near-perfect for same-tile workgroups)
  - Effective: ~1.3 MB global reads (polygon data) + ~1 MB texture (heavily cached)
- Write: 128 * 4B = 512B (error accumulators, atomic adds)
- Total: ~2.3 MB

**Select pass**:
- Read: K * lambda * 4B (error accumulators) + K * DrawingState (conditional polygon copy on accept)
- Write: K * DrawingState (conditional) + K * 4B (fitness_packed)
- With ~5% acceptance rate: ~0.05 * 16 * 16032B = ~12.5 KB average
- Total: ~270 KB average

**Per-iteration total**: ~5 MB. At 50 iterations per batch: ~250 MB per batch.

### Opportunity: Reduce Mutation Copy Bandwidth

The mutate shader copies the **entire** parent drawing (all 1000 polygon slots) to the offspring slot, even though only `polygon_count` polygons are active (typically 150-300). The copy loop is:
```wgsl
for (var i = 0u; i < poly_count; i++) {
    working_states[offspring_id].polygons[i] = chain_states[chain_id].polygons[i];
}
```

This already only copies `poly_count` polygons (good), but each polygon access to `working_states[offspring_id]` involves a scatter write to a potentially cold memory location. With lambda=8 offspring per chain and 16 chains, that's 128 offspring copies per iteration.

**Opportunity**: Since the mutate shader typically modifies only 1-3 polygons per offspring, a "copy-on-write" approach could be considered: store only the delta (which polygon index changed and the new value) and apply it during rasterization. However, this would complicate the rasterizer significantly and break the tile prefetch pattern.

**Verdict**: The current approach of copying only `poly_count` polygons is already reasonably efficient. The full 1000-polygon buffer is allocated but not accessed beyond `poly_count`. No change recommended.

---

## 7. Quantization Tradeoffs

### Current Vertex Quantization

Vertices are quantized to u16 (65536 levels) packed as two u16 per u32 word. For a 512x512 image:
- Resolution: 65536 / 512 = 128 sub-pixel positions per pixel
- Precision: 1/65536 = 0.0000153 in normalized coords = 0.0078 pixels at 512px
- This is well beyond perceptual resolution.

### Opportunity: Reduce to 12-bit Quantization

With 12-bit quantization (4096 levels):
- Resolution: 4096 / 512 = 8 sub-pixel positions per pixel
- Precision: 0.000244 normalized = 0.125 pixels at 512px
- Still sub-pixel, but approaching visible quantization artifacts for fine detail

Packing: Two 12-bit values in a u32 leaves 8 bits unused. Could pack all 3 vertices into 2 u32s (3 * 24 bits = 72 bits) instead of 3 u32s (96 bits), saving 1 word per polygon. Total polygon size: 12B instead of 16B, a 25% reduction.

**Analysis**: The 12-bit quantization adds visible stepping artifacts at high zoom levels and for polygons near the image edges. The u16 quantization is already compact and has no visible artifacts. The 25% size savings per polygon translates to 25% more polygons per shared memory tile (from 768 to ~1024), which reduces tile count.

**Verdict**: Not recommended. The u16 quantization is a good balance of precision and compactness. Going to 12-bit would introduce visible artifacts and make pack/unpack more complex (non-aligned bit extraction). The tile capacity improvement is better achieved through the SoA layout or hierarchical culling approaches.

### Opportunity: Use u8 for Color Components Instead of pack4x8unorm

Color is already stored as 4x u8 packed via `pack4x8unorm`. This is optimal -- no further compression is useful without sacrificing color fidelity.

---

## 8. Indirect Dispatch for Variable Workload

### Current Dispatch Strategy

The rasterize_error shader is dispatched as:
```rust
pass.dispatch_workgroups(wg_x, wg_y, active * lambda);
```
where `wg_x = (W+15)/16`, `wg_y = (H+15)/16`. Every offspring gets the same dispatch dimensions.

### Opportunity: Per-Offspring Variable Dispatch via Indirect

Offspring with fewer polygons need less rasterization work. An indirect dispatch could skip or reduce workgroups for offspring with very few polygons (e.g., an offspring with 10 polygons doesn't need the shared-memory tiling overhead).

**Analysis**: The dispatch dimensions are per-pixel (image space), not per-polygon. Even an offspring with 1 polygon still needs all pixels evaluated to compute the full-image error. The polygon loop is the inner variable, not the dispatch dimensions. Therefore, indirect dispatch based on polygon count doesn't reduce the pixel-level work.

**Potential variant**: For offspring where `polygon_count < tile_cap` (i.e., all polygons fit in a single tile), the tiling loop overhead (extra barrier, tile management) could be avoided with a specialized shader variant. But WGSL/wgpu doesn't support shader specialization constants (override constants are limited), so this would require maintaining two separate shader modules.

**Verdict**: Not applicable for the current architecture. Indirect dispatch would only help if the dispatch dimensions themselves varied per offspring, which they don't (all offspring render the full image).

---

## 9. Prefix Sum / Scan for Dynamic Workloads

### Opportunity: Prefix Sum for Hierarchical Tile Culling

If the hierarchical tile culling from Section 4B is implemented, a prefix sum is needed to allocate variable-length per-tile polygon lists in a flat buffer:

1. **Count pass**: Each tile counts how many polygons overlap it
2. **Prefix sum**: Exclusive scan over tile counts to compute per-tile offsets into a flat buffer
3. **Scatter pass**: Write polygon indices to the flat buffer at the computed offsets
4. **Rasterize pass**: Each tile reads its polygon list from `offset[tile_id]` to `offset[tile_id] + count[tile_id]`

**Implementation**: For 1024 tiles, the prefix sum is trivially small (fits in one workgroup). A simple Blelloch scan in shared memory would suffice:

```wgsl
@compute @workgroup_size(1024)
fn prefix_sum(@builtin(local_invocation_index) lid: u32) {
    shared_data[lid] = tile_counts[lid];
    // Up-sweep
    for (var d = 1u; d < 1024u; d *= 2u) {
        workgroupBarrier();
        if lid % (2u * d) == 2u * d - 1u { shared_data[lid] += shared_data[lid - d]; }
    }
    // Down-sweep
    if lid == 1023u { shared_data[lid] = 0u; }
    for (var d = 512u; d > 0u; d /= 2u) {
        workgroupBarrier();
        if lid % (2u * d) == 2u * d - 1u {
            let t = shared_data[lid - d];
            shared_data[lid - d] = shared_data[lid];
            shared_data[lid] += t;
        }
    }
    workgroupBarrier();
    tile_offsets[lid] = shared_data[lid];
}
```

**Verdict**: Only applicable if hierarchical tile culling is implemented. The prefix sum itself is trivial for 1024 tiles.

---

## 10. Buffer Reuse and Aliasing

### Current Buffer Allocation

The pipeline allocates these buffers:

| Buffer | Size (K=16, lambda=8, 512x512) | Usage Pattern |
|--------|------|---------------|
| `chain_states_buf` | 16 * 16032B = 250 KB | Persistent, R/W |
| `working_states_buf` | 128 * 16032B = 2.0 MB | Per-iteration, R/W |
| `error_accumulators_buf` | 128 * 4B = 512B | Per-iteration, atomic R/W |
| `reference_texture` | 512 * 512 * 4B = 1.0 MB | Read-only |
| `control_flags_buf` | 16B | Per-batch, atomic R/W |
| `params_buf` | 128B | Per-batch, read-only |
| `readback_staging_buf` | 16032B | On-demand |
| `control_staging_bufs` (x2) | 16B each | Per-batch |
| `fitness_packed_buf` | 64B | Per-iteration |
| `fitness_staging_bufs` (x2) | 64B each | Per-batch |
| **Total** | **~3.3 MB** | |

### Opportunity: Alias `error_accumulators_buf` With Padding in `working_states_buf`

Since each offspring already has a `fitness_bits` field in its `GpuDrawingState`, the separate `error_accumulators_buf` could theoretically use those fields directly. However, the error accumulator requires `atomic<u32>` access, and atomics on SSBO fields within a struct array are syntactically different in WGSL than atomics on a flat array.

**Analysis**: The `error_accumulators_buf` is only 512B for typical configurations. This is negligible compared to the 2 MB `working_states_buf`. No meaningful savings from aliasing.

**Verdict**: Not worthwhile. The buffer is too small to matter, and aliasing would add shader complexity.

### Opportunity: Temporal Buffer Aliasing Between Passes

The `working_states_buf` is only needed during mutate and rasterize passes. After the select pass extracts results, it could theoretically be reused for other purposes. However, within a single batch of 50 iterations, the buffer is needed for every iteration, so there's no temporal window for reuse.

**Verdict**: No opportunity for temporal aliasing within the current batch-of-50 architecture.

---

## 11. CPU Path Algorithmic Improvements

### Scanline Fill vs Half-Space Rasterization

The CPU path (`utils.rs`) maintains two rasterization algorithms:
1. `fill_shape`: Scanline fill using edge tables, HashMap-based. O(n * h) where n = edges, h = height.
2. `fill_triangle`: Half-space (Fgiesen-style) with 8x8 blocking. O(bbox_area) per triangle.

The scanline fill (`fill_shape`) allocates a `HashMap<usize, Vec<Line>>` per polygon, a `Vec<Point>` per scanline, sorts intersection points per row, and produces individual `Point` objects for every filled pixel. This is heavily allocation-bound.

### Opportunity: Eliminate Scanline Path

The GPU path only uses triangles (fan-triangulated). The CPU `fill_shape` is only used for polygons with >3 points. Since the GPU path already fan-triangulates all polygons, the CPU path could do the same: fan-triangulate any multi-point polygon into triangles and use `fill_triangle` for all of them. This would eliminate the allocation-heavy scanline code path entirely.

**Verdict**: Low priority since the CPU path is secondary to the GPU path. But if CPU performance matters, fan-triangulating and using `fill_triangle` exclusively would be faster due to zero allocations per polygon.

### Opportunity: SIMD Error Computation

The CPU evaluator (`evaluator.rs`) computes per-pixel L2 error using scalar `f32::sqrt()`:
```rust
let sqrt = f32::sqrt(((re * re) + (ge * ge) + (be * be)) as f32);
```

This could be vectorized using SIMD to process 4 pixels at once (each pixel has 3 channels). However, the `sqrt` operation is the bottleneck, and portable SIMD for `sqrt` is not yet stable. An alternative is to skip `sqrt` and use squared error (L2 squared), which is monotonically equivalent for comparison purposes and avoids the sqrt entirely.

**Caveat**: Changing from L2 to L2-squared would change the fitness scale and make CPU/GPU results diverge unless the GPU shader is also updated. The `MAX_ERROR_PER_PIXEL` constant would need to change from `sqrt(255^2 * 3) = 441.67` to `255^2 * 3 = 195075`.

**Verdict**: If CPU/GPU parity matters, not recommended without changing both. If the CPU path is only for fallback, switching to L2-squared is a free performance win.

---

## 12. Select Shader: Parallel Degenerate Triangle Culling

### Current Approach

After accepting an offspring, thread 0 sequentially scans all polygons for degenerate triangles (cross product < 0.00001):

```wgsl
if shared_accept == 1u && local_id == 0u {
    for (var r = 0u; r < count; r++) {
        // ... cross product check, compact if degenerate
    }
}
```

This is O(polygon_count) work on a single thread while 63 other threads in the workgroup are idle.

### Opportunity: Parallel Culling With Prefix Sum Compaction

1. All 64 threads cooperatively test polygons (thread i tests polygons i, i+64, i+128, ...)
2. Each thread marks keep/remove in shared memory
3. A workgroup-level prefix sum computes new indices
4. All threads cooperatively copy polygons to their compacted positions

**Analysis**: For 150 polygons, single-threaded culling takes ~150 iterations. Parallelized across 64 threads: ~3 iterations per thread. The prefix sum adds ~10 steps. Total: ~13 steps vs 150. A ~10x speedup for the culling operation.

**However**: The culling only runs on acceptance (~5% of iterations), and the culling itself is fast (simple arithmetic, no memory-bound operations). The total time spent in culling is negligible compared to rasterization.

**Verdict**: Theoretically clean but practically negligible. Only worth implementing if the acceptance rate increases substantially or polygon counts grow much larger.

---

## Summary of Recommendations

### High Impact (Recommended)

| # | Optimization | Expected Benefit | Effort |
|---|-------------|-----------------|--------|
| 4B | Hierarchical tile culling | ~5-10x rasterization speedup for 500+ polygon drawings | High |
| 5 | Subgroup reduction for error accumulation | Eliminate 7 barriers per workgroup per iteration | Medium |
| 4A | Precomputed AABB buffer | Avoid redundant vertex unpacking for rejected polygons | Medium |

### Medium Impact (Consider)

| # | Optimization | Expected Benefit | Effort |
|---|-------------|-----------------|--------|
| 1 | SoA polygon layout (split color from geometry) | 25% more polygons per tile, reduced shared memory pressure | Medium |
| 11 | Eliminate CPU scanline path (fan-triangulate all) | Faster CPU rasterization, zero allocations | Low |

### Low Impact (Not Recommended)

| # | Optimization | Reason to Skip |
|---|-------------|---------------|
| 2 | Power-of-2 state alignment | Negligible coalescing benefit for single-chain access |
| 3 | Spatial polygon sorting | Painter's order constraint prevents reordering |
| 7 | 12-bit vertex quantization | Visible artifacts, marginal size savings |
| 8 | Indirect dispatch for variable polygon count | Dispatch is per-pixel, not per-polygon |
| 10 | Buffer aliasing | Buffers too small to matter |
| 12 | Parallel degenerate culling | Only runs on rare acceptance events |
