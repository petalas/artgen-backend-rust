# GPU Compute Optimization Analysis

**Expert focus**: GPU occupancy, memory bandwidth, shader dispatch efficiency, buffer layout

---

## Executive Summary

The RTX 5090 has ~21,760 CUDA cores (170 SMs with 128 CUDA cores each), 32 GB VRAM, and the current pipeline leaves the vast majority of the GPU idle. The bottleneck breakdown, in order of severity:

1. **Mutate shader**: 64 threads total, `workgroup_size(1)` -- uses 0.02% of GPU
2. **Select/Migrate shaders**: Same problem -- 64 threads, `workgroup_size(1)`
3. **Rasterize and error_reduce**: The only passes with real parallelism, but only 64 chains
4. **Memory bandwidth**: Each pixel reads up to 1000 polygons (48 bytes each) = 48 KB per pixel from global memory with no locality exploitation

---

## 1. Scale Up Chain Count

**Current**: 64 chains. **Recommended**: 512-1024 chains.

The rasterize and error_reduce passes are the only well-parallelized shaders. Mutate and select launch only 64 threads total. Going to 512-1024 chains would make mutate/select still small but 8-16x better, and improve evolutionary search by exploring more of the fitness landscape.

**Memory at 512 chains, 384x384**:
- `chain_states`: 512 * 48,032 = ~23.4 MB
- `working_states`: ~23.4 MB
- `render_targets`: 512 * 384 * 384 * 4 = ~301 MB
- **Total: ~348 MB** -- well within 32 GB

At 4096 chains: ~2.8 GB total. Still plenty of headroom.

---

## 2. AABB Culling in Rasterize (Highest Impact Shader Change)

Every pixel tests every polygon with 3 edge evaluations. Typical triangles span ~3% of the image, so **~97% of polygon-pixel tests are rejections**.

Add a bounding box early-out before the edge test:

```wgsl
let min_x = min(poly.v0.x, min(poly.v1.x, poly.v2.x));
let max_x = max(poly.v0.x, max(poly.v1.x, poly.v2.x));
let min_y = min(poly.v0.y, min(poly.v1.y, poly.v2.y));
let max_y = max(poly.v0.y, max(poly.v1.y, poly.v2.y));

if fx < min_x || fx > max_x || fy < min_y || fy > max_y {
    continue;
}
```

4 comparisons vs 6 mul + 6 add for edge functions. Expected **3-5x rasterization speedup**.

---

## 3. Shared Memory Polygon Loading

Every thread in a workgroup independently loads the same polygon data from global memory. An 8x8 workgroup = 64 threads all reading the same polygon. L1 cache pressure is enormous (1000 * 48 bytes = 48 KB exceeds L1).

Use cooperative loading into shared memory:

```wgsl
var<workgroup> shared_polys: array<Polygon, 64>;

for (var tile_start = 0u; tile_start < poly_count; tile_start += 64u) {
    if local_idx < tile_count {
        shared_polys[local_idx] = working_states[chain_id].polygons[tile_start + local_idx];
    }
    workgroupBarrier();

    for (var i = 0u; i < tile_count; i++) {
        let poly = shared_polys[i];
        // ... AABB + edge test + blend ...
    }
    workgroupBarrier();
}
```

Shared memory usage: 64 * 48 = 3,072 bytes per workgroup (well within SM budget). Expected **2-4x reduction in global memory traffic**.

---

## 4. Drop sqrt in Error Computation

`sqrt()` costs 4-8 cycles vs 1 cycle for FMA. For relative comparison, ordering is preserved by sum of absolute differences:

```wgsl
let dr = abs(rr - refr);
let dg = abs(rg - refg);
let db = abs(rb - refb);
pixel_error = u32(dr + dg + db);
```

Max per pixel = 765 (255*3), no u32 overflow risk. Update `MAX_ERROR_PER_PIXEL` to 765.0.

---

## 5. Merge Rasterize + Error_Reduce (Eliminate render_targets Buffer)

Currently rasterize writes W*H*K pixels to `render_targets`, then error_reduce reads them back. This is a full round-trip through global memory.

Merging eliminates:
- The `render_targets` buffer (at 512 chains + 384x384 = ~301 MB saved)
- An entire compute pass dispatch
- ~72 MB of global memory traffic per iteration

Pixel color stays in registers, never touches global memory. **Probably the single highest-ROI change.**

---

## 6. Mutate Shader Improvements

### 6a. workgroup_size(1) wastes entire warps
Each SM runs 1 useful lane and 31 idle lanes. Options:
- Parallelize polygon copy across threads (workgroup_size(64))
- Split copy into a separate parallel pass
- Increase chain count so more workgroups are available

### 6b. Consider splitting copy and mutate
The polygon copy (48KB sequential) can be parallelized as a separate dispatch with workgroup_size(256).

---

## 7. Increase Batch Size

Current: 10 iterations per batch. Recommended: 50-100.

Amortizes CPU-GPU synchronization overhead. The GPU's select pass already handles error accumulator reset via `atomicExchange`, so the CPU-side zero-fill may even be redundant.

---

## 8. Double-Buffered Readback

Use two staging buffers, alternating each batch. While GPU runs batch N+1, CPU reads batch N's results. Eliminates GPU idle bubble between batches.

Expected: 20-30% throughput improvement by hiding readback latency.

---

## Priority Summary

| Priority | Change | Effort | Expected Speedup |
|----------|--------|--------|-----------------|
| 1 | Merge rasterize + error_reduce | Medium | 20-40% (eliminate buffer + pass) |
| 2 | AABB culling in rasterize | Low | 3-5x for rasterize pass |
| 3 | Drop sqrt, use L1 distance | Low | 10-15% for error computation |
| 4 | Increase chain count to 512-1024 | Trivial | Better search + GPU utilization |
| 5 | Increase batch size to 50-100 | Trivial | Reduce CPU-GPU sync overhead |
| 6 | Shared memory polygon loading | Medium | 2-4x memory bandwidth reduction |
| 7 | Parallelize mutate copy (workgroup_size 32+) | High | 10-20x for mutate pass |
| 8 | Double-buffered readback | Medium | Hides readback latency |
