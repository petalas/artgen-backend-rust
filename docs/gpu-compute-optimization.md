# GPU Compute Optimization Analysis

Analysis of the artgen-backend-rust GPU evolution pipeline for further performance improvements.
Target hardware: NVIDIA RTX 5090 (Blackwell, 170 SMs, 32-wide warps, 128KB L1/shared per SM, 96MB L2).
Current workload: 4-pass evolutionary loop (mutate, rasterize+error, select, migrate) dispatched 50 times per batch.

## Already-Completed Optimizations (not discussed further)

- Fused rasterize+error pass with workgroup reduction
- Tiled polygon prefetch (768-polygon shared-memory tiles)
- Adaptive mutation scale (1.2x accept, pow(0.8, 1/lambda) reject, burst to 5.0 on stagnation)
- Double-buffered async staging readback
- Per-offspring persistent PCG RNG
- Pipeline cache for shader binaries
- Degenerate triangle culling on acceptance
- Variable lambda (1-64 offspring)
- Island population with migration (intra + inter-island ring topology)
- Timestamp query profiling
- Workgroup size 16x16 for rasterize, 64 for mutate/select

---

## 1. Mutate Shader: Massive Warp Divergence and Serial Memory Access

### Current State

The mutate shader dispatches one workgroup of 64 threads per chain (`@workgroup_size(64, 1, 1)`), where thread `i` generates offspring `i` for that chain. Each thread runs a long, deeply-branched mutation function that:

1. Copies the full parent drawing state (up to 1000 polygons x 16 bytes = 16KB) from `chain_states` to `working_states` in a serial loop.
2. Runs mutation logic with many conditional branches (add/remove/reorder/scale/rotate/swap/offset/move/color/lighten/darken).
3. In multi-mutation mode, iterates over every polygon applying per-polygon mutations with independent RNG draws.

### Problem: Serial Parent Copy (16KB per thread)

Each thread independently copies up to 1000 polygons from parent to offspring:

```wgsl
for (var i = 0u; i < poly_count; i++) {
    var poly = chain_states[chain_id].polygons[i];
    // clamp alpha ...
    working_states[offspring_id].polygons[i] = poly;
}
```

With lambda=8 and 16 chains, that is 128 threads each doing 1000 x 16-byte reads from the same parent address range. All 8 threads within a chain read the exact same source data. This is a classic **broadcast read** opportunity being wasted with individual loads.

### Optimization: Cooperative Parent Copy via Shared Memory

Load the parent's polygon data once into shared memory using all 64 threads cooperatively (similar to how `rasterize_error.wgsl` loads polygon tiles). Then each thread copies from shared memory to its offspring slot:

```wgsl
var<workgroup> shared_parent: array<Polygon, 1000>;

// Cooperative load: 64 threads x ceil(1000/64) = ~16 loads each
for (var i = local_id; i < parent_poly_count; i += 64u) {
    shared_parent[i] = chain_states[chain_id].polygons[i];
}
workgroupBarrier();

// Each offspring copies from shared (fast) instead of global (slow)
for (var i = 0u; i < parent_poly_count; i++) {
    working_states[offspring_id].polygons[i] = shared_parent[i];
}
```

This reduces global memory reads by lambda-fold (8x at default lambda). The 16KB shared memory cost fits within the SM's 128KB budget.

**Estimated impact**: 2-4x faster parent copy phase. At lambda=8 with 1000 polygons, this eliminates 7/8 of global memory traffic for the copy.

### Problem: Warp Divergence in Mutation Branches

In single-mutation mode, each thread picks one mutation via weighted random selection. Within a warp of 32 threads, different threads will pick different mutation types, causing severe warp divergence. In the worst case, if all 13 mutation types are selected by at least one thread, the warp executes all 13 branches serially.

### Optimization: Sort-by-Mutation-Type Before Executing

This is difficult in WGSL due to limited cross-thread communication. A more practical approach:

- **Accept the divergence** but minimize its cost by reducing per-branch instruction count.
- **Factor out the polygon index selection** from the mutation itself -- currently each branch re-reads the polygon and re-unpacks vertices.
- **Pre-unpack the target polygon** into registers before the branch, so all branches start from the same unpacked state. Currently, vertex unpack (`unpack_vertex`) is repeated in every branch that touches vertices.

### Problem: Per-Polygon Loop in Multi-Mutation Mode

In multi-mutation mode, each thread iterates over all `poly_count` polygons (up to 1000), performing multiple independent RNG draws per polygon (move_point x3, micro_adjust x3, change_color x4, micro_color x4, lighten, darken, offset = 18 RNG draws per polygon). With 1000 polygons that is 18,000 RNG evaluations per thread. The PCG step function itself does a 64-bit multiply emulated with four 32-bit multiplies -- this is expensive.

### Optimization: Reduce RNG Calls in Multi-Mutation Mode

Bundle correlated mutations. Instead of 18 independent RNG draws per polygon, draw a single random number and use bit partitioning to select which mutations fire. For mutations with the same probability, a single draw can drive multiple decisions:

```wgsl
let r = pcg_step(&rng);
let should_move_v0 = (r & 0xFFu) < threshold_move;
let should_move_v1 = ((r >> 8u) & 0xFFu) < threshold_move;
let should_micro_v0 = ((r >> 16u) & 0xFFu) < threshold_micro;
// etc.
```

This could reduce RNG calls from ~18 to ~3-4 per polygon.

**Estimated impact**: 2-3x faster per-polygon mutation loop in multi-mutation mode.

---

## 2. Rasterize+Error Shader: Shared Memory Pressure and Reduction Inefficiency

### Current State

Dispatches `(ceil(W/16), ceil(H/16), active_chains * lambda)` workgroups of 16x16=256 threads. Each workgroup rasterizes one offspring's drawing at a 16x16 pixel tile. Uses two shared memory arrays:

- `shared_polys: array<Polygon, 768>` = 12,288 bytes
- `shared_errors: array<u32, 256>` = 1,024 bytes
- **Total: 13,312 bytes per workgroup**

### Problem: Shared Memory Limits Occupancy

On RTX 5090 (128KB shared memory per SM), 13,312 bytes per workgroup allows at most 9 concurrent workgroups per SM. However, the register pressure from the rasterization loop (maintaining running r/g/b accumulators, polygon vertices, edge function intermediates) likely limits this further to 4-6 workgroups. With 170 SMs and 256 threads per workgroup, that gives 170 x 5 x 256 = ~217K concurrent threads. For a typical run (256x256 image, 16 chains, lambda=8, tile size 16x16), we dispatch 16 x 16 x 128 = 32,768 workgroups with 256 threads each = 8.3M thread-items. The GPU can process ~217K at a time, so the compute grid is heavily oversubscribed (good for latency hiding), but shared memory is the bottleneck preventing higher occupancy.

### Optimization: Reduce Polygon Tile Size to Improve Occupancy

The 768-polygon tile was chosen to minimize barrier count, but it uses 12KB of shared memory. Reducing to 512 polygons (8KB) would cut shared memory to 9KB, allowing up to 14 concurrent workgroups per SM. The tradeoff is ~50% more tiles (and barriers) for drawings with >512 polygons, but the occupancy gain may compensate:

| Tile size | Shared mem | Max WGs/SM | Tiles for 1000 polys |
|-----------|-----------|------------|---------------------|
| 768       | 13,312 B  | 9          | 2                   |
| 512       | 9,216 B   | 13         | 2                   |
| 384       | 7,168 B   | 17         | 3                   |
| 256       | 5,120 B   | 25         | 4                   |

At typical polygon counts (100-500), even 384 would require only 1-2 tiles. Profile with timestamp queries at different tile sizes to find the sweet spot.

**Estimated impact**: 10-30% faster rasterize+error pass depending on polygon count.

### Problem: Reduction Has Unnecessary Barriers

The binary tree reduction uses `workgroupBarrier()` at every halving step (8 barriers for 256->1):

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

For NVIDIA GPUs with 32-wide warps, the last 5 levels of reduction (stride <= 16) operate within a single warp and do not need a barrier.

### Optimization: Warp-Synchronous Final Reduction

Replace the final 5 iterations (stride 16, 8, 4, 2, 1) with explicit subgroup operations or at minimum drop the barriers for stride <= 16. WGSL now supports `subgroupAdd()` through the `subgroups` feature, though wgpu support may be limited. A simpler approach:

```wgsl
// Last barrier needed at stride = 32 (cross-warp boundary)
if stride <= 16u {
    // These are warp-synchronous on NVIDIA — no barrier needed
    // But WGSL doesn't guarantee this. Use subgroupAdd if available.
}
```

If `subgroups` feature is available in wgpu 22.1, replace the entire reduction with:
```wgsl
let warp_sum = subgroupAdd(pixel_error);
if subgroupElect() {
    atomicAdd(&error_accumulators[chain_id], warp_sum);
}
```

This eliminates shared memory for reduction entirely, freeing 1KB.

**Estimated impact**: 5-15% faster reduction phase (small portion of total rasterize time).

### Problem: AABB Culling Has Divergent Early-Exit

The per-polygon AABB check causes warp divergence when some threads' pixels are inside the AABB and others aren't:

```wgsl
if fx < bb_min_x || fx > bb_max_x || fy < bb_min_y || fy > bb_max_y {
    continue;
}
```

For a 16x16 tile, polygons that partially overlap the tile cause some threads to skip (the `continue`) while others proceed to the half-space test. This is inherent to the algorithm but can be mitigated.

### Optimization: Tile-Level AABB Pre-Filter

Before the per-thread polygon loop, have all threads cooperatively compute the workgroup's tile bounding box (trivial: min/max of pixel coordinates), then skip polygons whose AABB doesn't intersect the tile at all. This can be done during the shared memory load phase:

```wgsl
// During cooperative load, also compute tile AABB intersection
let tile_min_x = f32(wid.x * 16u) / f32(w);
let tile_max_x = f32((wid.x + 1u) * 16u) / f32(w);
// ...
// Set a flag in shared memory: shared_visible[slot] = intersects_tile;
```

Then in the per-thread loop, skip polygons with `shared_visible[i] == false`. This eliminates all per-thread AABB computation for polygons that miss the tile entirely, reducing divergence.

**Estimated impact**: 10-25% reduction in per-pixel work for typical scenes where most polygons are small relative to the image.

---

## 3. Select Shader: Thread 0 Bottleneck and Serialized Decision Logic

### Current State

Dispatches one workgroup of 64 threads per chain. Thread 0 does all decision logic (find best offspring, compare fitness, update adaptive mutation state), then broadcasts via shared memory. All 64 threads cooperate on the polygon copy.

### Problem: Thread 0 Serial Lambda Loop

Thread 0 loops over lambda offspring to find the best:
```wgsl
for (var i = 0u; i < lambda; i++) {
    let oid = chain_id * lambda + i;
    let err = atomicExchange(&error_accumulators[oid], 0u);
    if err < best_error { ... }
}
```

With lambda=64, this is 64 serial `atomicExchange` operations plus comparisons. The other 63 threads in the workgroup are idle during this.

### Optimization: Parallel Min-Reduction for Best Offspring

Use all 64 threads to participate in finding the minimum error:

```wgsl
var<workgroup> shared_errors_sel: array<u32, 64>;
var<workgroup> shared_indices: array<u32, 64>;

// Each thread reads one offspring's error (thread i reads offspring i)
if local_id < lambda {
    shared_errors_sel[local_id] = atomicExchange(&error_accumulators[chain_id * lambda + local_id], 0u);
    shared_indices[local_id] = local_id;
} else {
    shared_errors_sel[local_id] = 0xFFFFFFFFu;
    shared_indices[local_id] = 0u;
}
workgroupBarrier();

// Binary reduction to find min
for (var s = 32u; s > 0u; s >>= 1u) {
    if local_id < s && shared_errors_sel[local_id + s] < shared_errors_sel[local_id] {
        shared_errors_sel[local_id] = shared_errors_sel[local_id + s];
        shared_indices[local_id] = shared_indices[local_id + s];
    }
    workgroupBarrier();
}
```

This parallelizes the lambda-fold search, reducing it from O(lambda) serial atomics to O(log2(lambda)) parallel steps.

**Estimated impact**: 2-4x faster select decision phase at lambda=64, minimal impact at lambda=1.

### Problem: Degenerate Culling is Single-Threaded

After acceptance, thread 0 serially scans all polygons for degenerate triangles:
```wgsl
for (var r = 0u; r < count; r++) {
    // cross product check ...
    if cross > 0.00001 { write_idx++; }
}
```

With 1000 polygons this is 1000 serial iterations with global memory reads while 63 threads wait.

### Optimization: Parallel Compaction for Degenerate Culling

Use all 64 threads to evaluate the cross products in parallel, then prefix-sum to compute compacted indices:

1. Each thread checks ceil(1000/64) = 16 polygons and writes a 1/0 bitmask.
2. Prefix-sum over the bitmask to get write indices.
3. Threads cooperatively write surviving polygons to compacted positions.

This turns an O(N) serial operation into O(N/64) parallel work. However, since culling only runs on acceptance (rare, ~5% of iterations), the absolute time savings may be small.

**Estimated impact**: Negligible overall (only fires on acceptance), but improves worst-case latency.

---

## 4. Dispatch Strategy: Per-Iteration Compute Pass Overhead

### Current State

Each of the 50 iterations in a batch encodes 3-4 separate compute passes:

```rust
for i in 0..iterations {
    // Pass 1: mutate (begin_compute_pass + dispatch + end)
    // Pass 2: rasterize_error (begin_compute_pass + dispatch + end)
    // Pass 3: select (begin_compute_pass + dispatch + end)
    // Pass 4: migrate (conditional, begin_compute_pass + dispatch + end)
}
```

Each `begin_compute_pass`/`end_compute_pass` pair in wgpu inserts implicit barriers and may flush caches. Over 50 iterations, that is 150-200 compute pass transitions.

### Problem: Pass Transition Overhead

wgpu translates each compute pass to a Vulkan command buffer with `vkCmdPipelineBarrier` calls between passes. On NVIDIA drivers, each barrier has ~1-5us overhead. With 150 barriers per batch, that is 150-750us of pure barrier overhead, which can be significant for small images (256x256) where the actual compute is only a few milliseconds.

### Optimization: Merge Passes Within a Single Compute Pass

Instead of separate compute passes per pipeline, use a single compute pass with pipeline switches:

```rust
let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { ... });
for i in 0..iterations {
    pass.set_pipeline(&p.mutate_pipeline);
    pass.set_bind_group(0, &p.mutate_bind_group, &[]);
    pass.dispatch_workgroups(active, 1, 1);

    pass.set_pipeline(&p.rasterize_error_pipeline);
    pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
    pass.dispatch_workgroups(wg_x, wg_y, active * lambda);

    pass.set_pipeline(&p.select_pipeline);
    pass.set_bind_group(0, &p.select_bind_group, &[]);
    pass.dispatch_workgroups(active, 1, 1);
    // migrate...
}
drop(pass); // single end
```

**Important caveat**: wgpu may not insert the necessary memory barriers between pipeline dispatches within a single compute pass. The mutate->rasterize dependency (working_states written then read) requires a `STORAGE_BUFFER` barrier. Verify that wgpu's internal tracking handles this correctly, or keep separate passes but batch them more aggressively.

**Alternative**: Reduce iterations per batch from 50 to a smaller number and overlap batches more aggressively via the existing double-buffer scheme.

**Estimated impact**: 5-15% reduction in per-batch overhead for small images.

### Optimization: Indirect Dispatch for Variable Lambda

Currently, the rasterize dispatch count `active * lambda` is set at encode time. If lambda changes between batches, the entire params buffer must be re-uploaded. Consider using `dispatch_workgroups_indirect` with a GPU-side dispatch args buffer that the mutate shader could update. This would allow dynamic lambda per chain (some chains could explore with high lambda while converged chains use lambda=1).

**Estimated impact**: Primarily an architectural improvement for future work, not immediate throughput.

---

## 5. Memory Access Patterns and Coalescing

### Problem: DrawingState Stride is 16,032 Bytes

`GpuDrawingState` is 16,032 bytes (32-byte header + 1000 x 16-byte polygons). When the rasterize shader reads `working_states[chain_id].polygons[...]`, adjacent chain_ids are 16KB apart. For `active * lambda = 128` offspring, the working_states buffer is 128 x 16,032 = ~2MB. L2 cache on RTX 5090 is 96MB, so the entire working_states buffer fits easily. However, the access pattern within the rasterize shader is:

1. All threads in a workgroup (z-dim = same offspring) read the same offspring's polygon data sequentially.
2. Different workgroups for different offspring read different 16KB regions.

This is actually a favorable pattern -- within a workgroup, the cooperative tile load produces coalesced 128-byte cache line fetches from the shared memory prefetch. Cross-offspring accesses hit different cache lines but the L2 is large enough to hold everything.

### Observation: Reference Image Texture is Well-Optimized

The reference image is stored as a 2D texture (`Rgba8Unorm`) which leverages the GPU's texture cache with spatial locality (2D tiling in hardware). This is optimal for the access pattern where nearby pixel threads read nearby reference pixels.

### Optimization: Pack GpuPolygon More Tightly

Currently each `GpuPolygon` is 16 bytes (4 x u32). The color uses 4 bytes (RGBA8), and each vertex uses 4 bytes (2 x u16). This is already quite compact. However, if vertex precision could be reduced to 12 bits per component (4096 levels instead of 65536), two vertices could fit in 3 bytes instead of 4, saving 3 bytes per polygon. At 1000 polygons that saves 3KB per offspring. This is probably not worth the precision loss.

### Optimization: Structure-of-Arrays for Polygon Data

Instead of the current array-of-structures layout:

```
DrawingState {
    header: 32 bytes
    polygons: [Polygon; 1000]  // each 16 bytes
}
```

Consider splitting into separate buffers:
```
polygon_colors: array<u32>    // 4 bytes each
polygon_v0: array<u32>        // 4 bytes each
polygon_v1: array<u32>        // 4 bytes each
polygon_v2: array<u32>        // 4 bytes each
```

This would improve coalescing when shaders access only one field (e.g., centroid computation only needs vertices, not colors). However, the current fused rasterize shader reads all 4 fields for every polygon test, so the benefit is limited for the hot path. **Not recommended** given the complexity cost.

---

## 6. Batch Size Tuning

### Current State

`GPU_ITERATIONS_PER_BATCH = 50`. Each batch of 50 iterations is encoded as one command buffer, submitted, then results are read back via double-buffered staging.

### Analysis

The batch size affects two opposing factors:
1. **Larger batches**: Amortize CPU-side submission and readback overhead. Fewer `queue.submit()` calls, fewer `map_async` operations.
2. **Smaller batches**: Faster response to new global best (lower latency between improvement found and CPU learning about it). More frequent parameter updates.

At 50 iterations with lambda=8 and 16 chains, each batch produces 50 x 8 x 16 = 6,400 offspring evaluations. If rasterize+error takes ~1ms per iteration, the batch takes ~50ms. The double-buffer scheme adds one batch of latency (so improvements are reported ~50ms late). This seems reasonable.

### Optimization: Adaptive Batch Size

Increase batch size when no improvements are being found (steady state) and decrease it when improvements are frequent (early evolution or after migration). This maximizes throughput during plateaus while maintaining responsiveness during rapid improvement phases.

```rust
let iterations = if improvements_in_last_10_batches > 5 {
    25  // responsive mode
} else {
    100 // throughput mode
};
```

**Estimated impact**: 5-10% throughput improvement in steady state by reducing submission overhead.

---

## 7. RTX 5090-Specific Considerations

### Blackwell Architecture Features

1. **170 SMs** with 128 CUDA cores each = 21,760 cores. The rasterize pass with a 256x256 image dispatches 16 x 16 x 128 = 32,768 workgroups of 256 threads = 8.3M threads. This easily saturates the GPU.

2. **128KB L1/shared per SM**: The current 13KB shared memory usage per workgroup is modest. Could increase tile size or add more shared memory caching.

3. **96MB L2 cache**: Entire working_states buffer (2MB at 128 offspring) fits in L2 with room to spare. Reference texture (256x256x4 = 256KB) also fits. Cache thrashing is unlikely.

4. **Warp size 32**: The 16x16 workgroup maps to 8 warps. Polygon AABB culling creates divergence across warps within a workgroup (tiles at polygon edges). This is inherent and cannot be eliminated.

5. **Clock speed ~2.4 GHz boost**: Individual thread performance is high, so single-threaded bottlenecks (select decision, degenerate culling) complete fast. The parallel optimization of select's lambda loop is most impactful at high lambda.

### Optimization: Increase Chain Count for Full GPU Utilization

With 170 SMs and the rasterize pass being the dominant cost, full utilization requires enough z-dimension dispatches. At 256x256, each offspring needs 16x16=256 workgroups. With 128 offspring (16 chains x 8 lambda), that is 32,768 workgroups. At 5 concurrent workgroups per SM, the GPU can run 170 x 5 = 850 workgroups simultaneously. It takes 32768/850 = ~39 waves to complete. This is good occupancy.

At 512x512, each offspring needs 32x32=1024 workgroups, and 128 offspring = 131,072 workgroups. This is 154 waves -- even better utilization.

However, the mutate and select passes only dispatch `active` workgroups (16 by default). With 170 SMs, only 16/170 = 9.4% of SMs are utilized during mutate/select. **Increasing chain count to 128-256** would better utilize the GPU during these phases.

**Estimated impact**: 3-8x faster mutate+select passes (currently ~10% of total time, so ~3-8% overall).

---

## 8. Alpha Clamping During Copy

### Current State

During the parent-to-offspring copy in the mutate shader, every polygon's alpha is clamped:

```wgsl
for (var i = 0u; i < poly_count; i++) {
    var poly = chain_states[chain_id].polygons[i];
    var color = unpack_color(poly);
    color.w = clamp(color.w, params.min_alpha_norm, params.max_alpha_norm);
    poly.data.x = pack_color(color);
    working_states[offspring_id].polygons[i] = poly;
}
```

This unpack-clamp-repack cycle runs on every polygon, every copy. Since alpha values are already clamped from the previous iteration's mutation, the clamp is almost always a no-op.

### Optimization: Defer Alpha Clamping

Only clamp alpha on mutation (when the color actually changes) and on crossover. Skip the per-polygon clamp during copy. This eliminates `unpack4x8unorm` + `clamp` + `pack4x8unorm` for every polygon in the copy loop, allowing a direct 16-byte copy:

```wgsl
for (var i = 0u; i < poly_count; i++) {
    working_states[offspring_id].polygons[i] = chain_states[chain_id].polygons[i];
}
```

**Estimated impact**: 10-20% faster parent copy in mutate shader.

---

## 9. Error Metric: Integer Arithmetic Instead of sqrt()

### Current State

The per-pixel error uses `sqrt()`:

```wgsl
pixel_error = u32(sqrt(dr * dr + dg * dg + db * db));
```

`sqrt()` on NVIDIA hardware is a special function unit (SFU) operation that takes 4-8 cycles. With 256x256 = 65,536 pixels per offspring, that is 65K sqrt operations per offspring.

### Optimization: Use Squared Error

Replace L2 distance with squared L2 distance:

```wgsl
pixel_error = u32(dr * dr + dg * dg + db * db);
```

The ranking of offspring by error is preserved (sqrt is monotonic), so selection will choose the same best offspring. The absolute fitness values reported to CPU will differ, but the GPU-side comparison remains correct. Adjust `max_error_per_pixel` on the CPU side: `255^2 * 3 = 195,075` instead of `sqrt(255^2 * 3) = 441.67`.

This eliminates the sqrt entirely. The `u32(...)` conversion from `f32` is a single instruction.

**Caveat**: The fitness function in the select shader also uses total error, and the atomicMax on control flags uses fitness bits for cross-chain comparison. As long as all chains use the same metric, the relative ordering is preserved. The CPU-side fitness display would need to apply `sqrt` to convert back to the L2 scale.

**Estimated impact**: 5-10% faster per-pixel computation in rasterize+error pass.

---

## 10. Workgroup Size Exploration

### Current Sizes

| Shader          | Workgroup size | Threads | Rationale                          |
|-----------------|---------------|---------|-------------------------------------|
| mutate          | (64, 1, 1)    | 64      | 1 thread per offspring, max lambda=64 |
| rasterize_error | (16, 16, 1)   | 256     | 2D tile matching pixel grid         |
| select          | (64, 1, 1)    | 64      | Parallel polygon copy               |

### Rasterize: Consider 8x8 Workgroups

Smaller workgroups (8x8=64 threads) would:
- Use 4x less shared memory for `shared_errors` (64 vs 256 entries, saving 768 bytes)
- Use 4x less shared memory for `shared_polys` if tile size scales proportionally
- Allow 4x more concurrent workgroups per SM (more occupancy)
- Require 4x more workgroups (more dispatch overhead)
- Reduce reduction tree depth from 8 to 6 steps

The tradeoff: more workgroups means more atomic additions to error accumulators, and each workgroup's reduction produces a smaller partial sum. For a 256x256 image, 8x8 tiles = 32x32 = 1024 workgroups per offspring vs 16x16 = 256 per offspring. Each workgroup contributes one atomicAdd. With 128 offspring that is 131K vs 32K atomicAdds total. On Blackwell with 96MB L2 and dedicated atomic units, this should not be a bottleneck.

### Recommendation

Profile both 8x8 and 16x16 workgroup sizes with the timestamp query infrastructure. The occupancy improvement from 8x8 may outweigh the increased atomicAdd traffic.

---

## Summary: Priority-Ranked Optimization Opportunities

| # | Optimization                                        | Pass          | Est. Impact | Effort |
|---|-----------------------------------------------------|---------------|-------------|--------|
| 1 | Drop sqrt in error metric (use squared error)       | rasterize     | 5-10%       | Low    |
| 2 | Skip alpha clamping during parent copy               | mutate        | 10-20% of copy | Low    |
| 3 | Cooperative parent copy via shared memory             | mutate        | 2-4x copy speed | Medium |
| 4 | Tile-level AABB pre-filter                            | rasterize     | 10-25%      | Medium |
| 5 | Reduce polygon tile size (768->512 or 384)           | rasterize     | 10-30%      | Low    |
| 6 | Parallel min-reduction in select (lambda>1)          | select        | 2-4x select | Medium |
| 7 | Batch RNG draws in multi-mutation mode                | mutate        | 2-3x mutation| Medium |
| 8 | Merge compute passes (reduce barrier overhead)        | dispatch      | 5-15%       | Low    |
| 9 | Increase chain count for mutate/select utilization    | all           | 3-8%        | Low    |
| 10| Workgroup size tuning (8x8 vs 16x16)                | rasterize     | Variable    | Low    |
| 11| Warp-synchronous / subgroup reduction                | rasterize     | 5-15% of red| Medium |
| 12| Adaptive batch size                                  | dispatch      | 5-10%       | Low    |
