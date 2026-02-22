# GPU Compute Optimization Analysis (2026-02-22)

Comprehensive analysis of the artgen-backend-rust GPU evolution pipeline.
Target hardware: NVIDIA RTX 5090 (Blackwell, 170 SMs, 32-wide warps, 128KB L1/shared per SM, 96MB L2).
Current workload: 3-pass evolutionary loop (mutate -> rasterize+error -> select) dispatched N times per batch (default 64 iterations).

## Already-Completed Optimizations

These are implemented in the current codebase and not discussed further:

- Fused rasterize+error pass with workgroup-level subgroup reduction (`subgroupAdd`)
- Tiled polygon prefetch into shared memory (TILE_CAP = THREAD_COUNT * 3 polygons per tile)
- Cooperative parent loading via shared memory in mutate shader
- Parallel min-reduction in select shader (binary tree across 64 threads)
- Adaptive mutation scale (1.2x accept, pow(0.99, 1/lambda) reject, burst to 1.5 on stagnation)
- Double-buffered async staging readback
- Per-offspring persistent PCG RNG (in working_states, not copied from parent)
- Pipeline cache for shader binaries
- Degenerate triangle culling on acceptance
- Variable lambda (1-64 offspring, power-of-2)
- Timestamp query profiling (opt-in, last iteration only)
- Configurable workgroup size for rasterize (default 32x16, supports 8x8 to 32x16)
- Compute pass consolidation (bulk iterations in single pass, timestamps only on last iter)
- Tile culling via bin_polygons pass (optional, per-workgroup polygon lists)
- Incremental evaluation with dirty bounding boxes (optional)
- Crossover with tournament selection across chains
- Alpha clamping removed from parent copy (only on crossover path)

---

## 1. bin_polygons Shader: Single-Threaded Serial Bottleneck

### Current State

`bin_polygons.wgsl` dispatches with `@workgroup_size(1, 1, 1)` -- a single thread per offspring. That single thread serially iterates over ALL polygons (up to 1000) and for each polygon, iterates over all overlapping tiles, doing `atomicAdd` on the tile count plus a global memory write to `tile_data`.

```wgsl
// bin_polygons.wgsl line 103
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    // ... serial over all polygons, serial over all tiles
    for (var pi = 0u; pi < poly_count; pi++) {
        for (var ty = tile_min_y; ty < tile_max_y; ty++) {
            for (var tx = tile_min_x; tx < tile_max_x; tx++) {
                let slot = atomicAdd(&tile_counts[tile_global], 1u);
                // ...
```

With 256 offspring (4 chains * 64 lambda), this dispatches 256 single-thread workgroups on a 170-SM GPU. Each workgroup uses just 1 thread of a 32-thread warp, wasting 96.9% of available hardware.

### Problem Analysis

For 500 polygons, each covering ~5 tiles on average, the single thread performs ~2500 `atomicAdd` + ~2500 global writes sequentially. With 256 offspring, that is 256 single-thread workgroups -- the GPU has 170 SMs that can each run many concurrent workgroups, but each workgroup completes slowly due to serialization.

The deeper issue: the serial loop is needed to preserve alpha blending order (polygon indices must appear in order in tile_data). However, the serial work is per-offspring -- no cross-offspring dependency exists.

### Optimization: Multi-Threaded Binning with Cooperative Parallelism

Use a larger workgroup (e.g., 64 threads per offspring) and distribute polygons across threads in stripes. Since each offspring's tiles are independent, threads can safely do atomicAdd on different tiles. The ordering constraint is satisfied because thread `t` processes polygons `t, t+64, t+128...` which are already in ascending order, and atomicAdd serializes writes to the same tile.

Wait -- that breaks ordering. If thread 0 processes polygon 0 (tiles A,B) and thread 1 processes polygon 1 (tile A), thread 1's atomicAdd on tile A might complete before thread 0's, inserting polygon 1 before polygon 0 in tile_data.

**Better approach**: Two-pass binning.
1. **Count pass** (parallel): Each thread processes a stripe of polygons, computes which tiles each polygon overlaps, and atomically increments tile_counts. Ordering doesn't matter for counting.
2. **Scatter pass** (serial): Single thread (or cooperative with prefix sum) writes polygon indices in order.

Alternatively, since tile_data ordering only affects rasterization correctness when tile culling is on, and the rasterize shader already reads polygons from tile_data in the order they appear, the simplest optimization is to **parallelize the tile count reset** (currently serial over all tiles) and keep the polygon scatter serial. The tile count reset loop is:

```wgsl
for (var t = 0u; t < num_tiles; t++) {
    atomicStore(&tile_counts[offspring_id * num_tiles + t], 0u);
}
```

At 512x512 with 16x16 tiles = 1024 tiles per offspring, that's 1024 serial atomicStores. With 64 threads, this drops to 16 iterations per thread.

**Estimated impact**: The binning pass is a small fraction of total time (included in rasterize_error timing window). At lambda=64 with 4 chains, parallelizing the reset loop alone saves ~1000 iterations per offspring. The polygon scatter remains serial but is I/O-bound (global memory writes), not compute-bound. **5-15% improvement on binning pass**.

---

## 2. Rasterize Shader: Redundant Parent Rasterization in Incremental Mode

### Current State

When `incremental_eval == 1`, the rasterize_error shader rasterizes the offspring drawing AND re-rasterizes the parent drawing at every pixel in the dirty bbox:

```wgsl
// rasterize_error.wgsl line 253-265
if incremental {
    var old_r = 255.0; var old_g = 255.0; var old_b = 255.0;
    let parent_poly_count = chain_states[parent_chain].polygon_count;
    for (var pi = 0u; pi < parent_poly_count; pi++) {
        rasterize_blend(chain_states[parent_chain].polygons[pi], fx, fy, &old_r, &old_g, &old_b);
    }
    // ...
}
```

This means **every pixel in the dirty bbox rasterizes the FULL parent drawing** (all polygons, not just affected ones) to compute `pixel_error_old`. For a parent with 500 polygons and a dirty bbox covering 10% of pixels, this is 500 * 26K = 13M polygon-pixel tests just for the "old" error. The offspring rasterization is another 13M.

### Problem

The parent rasterization is the SAME for every offspring within a chain (all offspring share the same parent). With lambda=64, this redundantly rasterizes the parent 64 times. Furthermore, the parent's framebuffer was already computed during `init_framebuffers` and is stored in `chain_framebuffers`.

### Optimization: Read Parent Error from Cached Framebuffer

Instead of re-rasterizing the parent in the dirty bbox, read the cached framebuffer pixel and compute error directly:

```wgsl
if incremental {
    let fb_idx = parent_chain * w * h + py * w + px;
    let packed = chain_framebuffers[fb_idx];
    let old_r = f32(packed & 0xFFu);
    let old_g = f32((packed >> 8u) & 0xFFu);
    let old_b = f32((packed >> 16u) & 0xFFu);
    pixel_error_old = u32(abs(old_r - refr) + abs(old_g - refg) + abs(old_b - refb));
}
```

This replaces O(polygon_count) rasterization per pixel with a single buffer read. The framebuffer is already maintained by `init_framebuffers` and updated by the select shader on acceptance.

**Caveat**: The current code comments say "re-rasterize parent to avoid quantization mismatch that causes error drift." The framebuffer stores u8 RGB values (packed into u32), while the rasterization produces f32 values that are clamped to [0, 255]. The quantization difference is at most 0.5 per channel per pixel. Over time this could accumulate, but the select shader already uses `chain_total_errors` (computed with quantized values from init_framebuffers) for parent fitness, so the mismatch already exists at the selection level.

**Implementation note**: The rasterize_error shader already has `chain_states` bound at `@group(0) @binding(5)`, and `chain_framebuffers` is bound in the select and init_framebuffers shaders. To read it from rasterize_error, it would need an additional binding (binding 6 or use the existing chain_framebuffers binding from select). Alternatively, reuse the existing init_framebuffers buffer binding.

**Estimated impact**: For typical incremental eval with 500 polygons, this eliminates the parent rasterization entirely (50% of the per-pixel work). With lambda=64 and dirty bbox covering ~5% of image, this saves ~64 * 500 * 1.3K = 41.6M polygon-pixel tests per iteration. **2x faster rasterize pass in incremental mode**.

---

## 3. Mutate Shader: Excessive Global Memory Writes in Per-Polygon Loop

### Current State

In multi-mutation mode, the per-polygon mutation loop reads, potentially mutates, and writes EVERY polygon back to global memory regardless of whether it changed:

```wgsl
// mutate.wgsl line 823-926
for (var pi = 0u; pi < count; pi++) {
    var poly = working_states[offspring_id].polygons[pi];
    // ... many independent RNG draws, most of which don't fire ...
    poly.data = vec4<u32>(pack_color(color), pack_vertex(v0), pack_vertex(v1), pack_vertex(v2));
    working_states[offspring_id].polygons[pi] = poly;  // ALWAYS writes back
}
```

With default mutation probabilities, most per-polygon mutations have probability 0.001-0.004. For a drawing with 500 polygons, each polygon has ~18 independent RNG draws; on average only ~0.04 mutations fire per polygon. Yet all 500 polygons are unconditionally read, unpacked, repacked, and written.

### Problem

The pack/unpack overhead is non-trivial: `unpack_color` (1 call to `unpack4x8unorm`), 3x `unpack_vertex` (3 ALU ops each), then `pack_color` + 3x `pack_vertex` at the end. This is ~30 ALU ops per polygon even when nothing changed. With 500 polygons per offspring and 64 offspring = 32K unnecessary pack/unpack cycles per iteration.

The global memory write is even more expensive: 16 bytes per polygon written regardless of modification. That's 500 * 16 = 8KB per offspring, 64 * 8KB = 512KB per iteration of unnecessary writes.

### Optimization: Track Dirty Flag Per Polygon, Skip Writeback

```wgsl
for (var pi = 0u; pi < count; pi++) {
    var poly = working_states[offspring_id].polygons[pi];
    var modified = false;
    // ... mutations set modified = true when they fire ...
    if modified {
        poly.data = vec4<u32>(pack_color(color), pack_vertex(v0), pack_vertex(v1), pack_vertex(v2));
        working_states[offspring_id].polygons[pi] = poly;
    }
}
```

Since the polygon was already copied from shared memory to the offspring slot before the mutation loop (line 622), unmodified polygons are already correct in global memory. The write is purely redundant.

**Estimated impact**: With ~4% mutation rate per polygon, this eliminates ~96% of per-polygon writebacks and all associated pack operations. **10-20% faster multi-mutation mode**. Less impactful in single-mutation mode (which exits early after one mutation).

---

## 4. Select Shader: Degenerate Culling is Single-Threaded on Acceptance

### Current State

After accepting an offspring, thread 0 serially scans all polygons for degenerate triangles while 63 threads wait:

```wgsl
// select.wgsl line 278-296
if shared_accept == 1u && local_id == 0u {
    let count = shared_copy_count;
    var write_idx = 0u;
    for (var r = 0u; r < count; r++) {
        // ... cross product check, compaction ...
    }
}
```

With 500+ polygons, this is 500 serial global memory reads + cross product computations + conditional writes. The 63 idle threads could help.

### Optimization: Parallel Degenerate Check + Cooperative Compaction

Phase 1: All 64 threads check ceil(count/64) polygons each, writing 1/0 to shared memory.
Phase 2: Prefix sum over the bitmask to compute write indices.
Phase 3: Threads cooperatively write surviving polygons.

However, since acceptance rate is ~5% and culling is only meaningful when degenerate triangles exist (rare in practice), the absolute time savings are small. **Estimated impact: negligible overall, but improves worst-case tail latency.**

A simpler partial optimization: parallelize just the read+check phase and have thread 0 do the compaction. This avoids prefix sum complexity while parallelizing the expensive part (global reads + FP math).

---

## 5. GpuParams Uniform Buffer: Could Use Push Constants

### Current State

`GpuParams` (256 bytes) is uploaded via `queue.write_buffer` every batch and read through a uniform buffer binding at `@group(1) @binding(0)`. Every compute pass binds it.

```rust
// mod.rs line 231
p.queue.write_buffer(&p.params_buf, 0, params_bytes);
```

### Problem

The 256-byte uniform buffer requires: (1) CPU staging allocation, (2) DMA transfer to GPU, (3) descriptor set binding per pass. On Vulkan, push constants up to 128 bytes are guaranteed; NVIDIA typically supports 256 bytes. Push constants are embedded directly in the command stream and accessed from SM-local registers with zero latency.

However, GpuParams is currently **256 bytes** -- exceeding the 128-byte Vulkan minimum guarantee. The NVIDIA RTX 5090 likely supports 256 bytes, but this is adapter-specific.

### Optimization: Split GpuParams into Hot (128B push constants) + Cold (uniform)

The most frequently-accessed params (image dimensions, lambda, chain_count, mutation mode flags, max_polygons) fit easily in 128 bytes. Rarely-changing params (individual mutation probabilities) can remain in the uniform buffer.

Alternatively, if the adapter reports `max_push_constant_size >= 256`, use push constants for the full struct. Add a runtime check:

```rust
let push_constant_size = if adapter_limits.max_push_constant_size >= 256 { 256 } else { 0 };
```

**Estimated impact**: Eliminates one `queue.write_buffer` per batch plus one descriptor binding per pass. The data is already in L1 cache after the first access, so the steady-state benefit is small. **1-3% overall improvement**, primarily in CPU-side encoding cost reduction.

---

## 6. Rasterize Shader: Color Unpack Only on Hit

### Current State

In `rasterize_blend`, the color is unpacked AFTER the inside-triangle test:

```wgsl
// rasterize_error.wgsl line 169-179
if all_pos || all_neg {
    let pcolor = unpack_color(poly);
    // ... alpha blend ...
}
```

This is already well-optimized for the brute-force path. However, in the shared memory prefetch path, the polygon is loaded as a complete 16-byte `Polygon` struct. The color word (`poly.data.x`) is loaded even for polygons that fail AABB culling.

### Optimization: Split Geometry and Color in Shared Memory (SoA)

Store only the 12 bytes of geometry (3 packed vertices) in shared_polys during cooperative load. Defer the 4-byte color load to only threads that pass the half-space test. This increases shared memory polygon capacity by 33% at the same memory budget:

| Layout | Bytes/poly | Polys at 12KB budget |
|--------|-----------|---------------------|
| Current AoS | 16 | 768 |
| SoA (geometry only in shared) | 12 | 1024 |

With 1024 polygons per tile, drawings with up to 1024 polygons need only 1 tile pass (vs 2 currently for 769-1000 polygons), eliminating a `workgroupBarrier`.

**Implementation**: Change `shared_polys` to `array<vec3<u32>, 1024>` (vertex-only). Color is read from global memory only on hit: `let color_packed = working_states[chain_id].polygons[tile_base + i].data.x`. Since hits are a small fraction of polygon tests (~5-10% for small polygons), the extra global reads are minimal and likely L2-cached.

**Estimated impact**: For drawings with 769-1000 polygons, eliminates one tile pass + barrier = **10-20% faster rasterize**. For drawings under 768 polygons, the benefit is the slightly reduced shared memory footprint improving occupancy. **Medium effort** -- touches shared memory layout, cooperative load, and inner loop.

---

## 7. Dispatch Efficiency: Mutate/Select Underutilize GPU at Low Chain Counts

### Current State

Default is 4 chains with lambda=64. The mutate shader dispatches 4 workgroups of 64 threads each = 256 threads total. The select shader dispatches 4 workgroups of 64 threads = 256 threads. On an RTX 5090 with 170 SMs capable of running thousands of concurrent threads, this is severe underutilization:

- Mutate: 4 workgroups / 170 SMs = 2.4% SM utilization
- Select: 4 workgroups / 170 SMs = 2.4% SM utilization
- Rasterize: 4*64 = 256 offspring, each with (512/32)*(512/16) = 16*32 = 512 workgroups = 131,072 total workgroups = well-saturated

The rasterize pass is the only one that actually saturates the GPU. Mutate and select are severely under-subscribed.

### Problem

With 4 chains and lambda=64, the mutate shader runs 4 workgroups. Each workgroup's 64 threads do independent work (one offspring each), so there is no warp divergence issue -- but there are only 4*2 = 8 warps total, occupying at most 4 SMs out of 170.

### Optimization: Increase Default Chain Count or Restructure Dispatch

**Option A**: Increase GPU_DEFAULT_CHAIN_COUNT from 4 to 32-64. This would dispatch 32-64 workgroups for mutate/select, improving SM utilization to 19-38%. The memory cost is modest: 64 chains * 16KB = 1MB for chain_states (trivial on RTX 5090 with 32GB VRAM).

**Option B**: For the mutate shader specifically, restructure the dispatch so each offspring is a separate workgroup instead of using workgroup-internal parallelism. Currently the shared memory cooperative parent load requires a workgroup per chain. If the parent data were pre-staged (e.g., by a lightweight "copy parent to offspring" pass), the mutate workgroup size could be reduced to 1, dispatching `active * lambda` single-thread workgroups.

**Concern**: More chains means more independent evolutionary runs, which doesn't help convergence per chain. The benefit is purely GPU utilization.

**Estimated impact**: Mutate+select are typically <15% of total batch time (rasterize dominates). Even fully saturating the GPU during these phases would save at most 12-13% overall. **At higher polygon counts where mutate is more expensive, this matters more.**

---

## 8. Rasterize Shader: Thread 0 Cross-Subgroup Summation Loop

### Current State

After `subgroupAdd`, thread 0 serially sums across subgroups:

```wgsl
// rasterize_error.wgsl line 371-377
if local_idx == 0u {
    let num_subgroups = (THREAD_COUNT + sg_size - 1u) / sg_size;
    var total = 0u;
    for (var i = 0u; i < num_subgroups; i++) {
        total += shared_errors[i];
    }
    atomicAdd(&error_accumulators[chain_id * 2u], total);
}
```

With the default 32x16 = 512-thread workgroup and sg_size=32, there are 16 subgroups. Thread 0 does 16 serial shared memory reads. With 16x16 = 256 threads, there are 8 subgroups. This is a minor serial tail but it runs on EVERY workgroup dispatch.

### Optimization: Second-Level Subgroup Reduction

If the number of subgroups (e.g., 16) is <= sg_size (32 on NVIDIA), the first 16 threads of the first subgroup can do a second `subgroupAdd` instead of a serial loop:

```wgsl
workgroupBarrier();
if local_idx < num_subgroups {
    let val = shared_errors[local_idx];
    let final_sum = subgroupAdd(val);
    if local_idx == 0u {
        atomicAdd(&error_accumulators[chain_id * 2u], final_sum);
    }
}
```

This replaces the serial loop with a single subgroup intrinsic. For 16 subgroups, this is 1 cycle vs 16 shared memory reads.

**Caveat**: The `subgroupAdd` must only operate on the first `num_subgroups` lanes. Threads beyond `num_subgroups` contribute 0 (since `shared_errors` beyond that is undefined). The solution is to read 0 for lanes >= num_subgroups:

```wgsl
if local_idx < num_subgroups {
    let val = select(0u, shared_errors[local_idx], local_idx < num_subgroups);
    let final_sum = subgroupAdd(val);
    if local_idx == 0u {
        atomicAdd(&error_accumulators[chain_id * 2u], final_sum);
    }
}
```

Wait -- only threads 0..num_subgroups-1 enter the `if` block, so `subgroupAdd` only operates on those threads. On NVIDIA, inactive threads in a warp don't participate in subgroup ops. This is correct.

**Estimated impact**: Saves ~15 shared memory reads per workgroup. With 131K workgroups per iteration (512x512, 256 offspring), that's ~2M fewer shared memory reads. **1-3% faster rasterize reduction phase**. Minor but free improvement.

---

## 9. Memory Layout: Error Accumulators Stride-2 Causes Bank Conflicts

### Current State

Error accumulators use stride-2 layout: `[new_error_0, old_error_0, new_error_1, old_error_1, ...]`. In the select shader, threads 0..lambda-1 each read their error at index `oid * 2`:

```wgsl
// select.wgsl line 149-150
let new_err = atomicExchange(&error_accumulators[oid * 2u], 0u);
let old_err = atomicExchange(&error_accumulators[oid * 2u + 1u], 0u);
```

With lambda=64, thread `k` accesses indices `2k` and `2k+1`. These are contiguous and should not cause bank conflicts in shared memory. However, in the rasterize shader, the atomicAdd writes to `chain_id * 2u` where chain_id = wid.z. Multiple workgroups for the same offspring all atomicAdd to the same address -- this is expected and unavoidable for atomic accumulation.

No significant optimization here -- stride-2 is fine.

---

## 10. wgpu Overhead: Per-Dispatch Bind Group Setting

### Current State

Inside the bulk iteration loop, every iteration sets bind groups for 3 (or 4 with tile culling) pipelines:

```rust
// mod.rs line 285-307
for _ in 0..bulk_count {
    pass.set_pipeline(&p.mutate_pipeline);
    pass.set_bind_group(0, &p.mutate_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(active, 1, 1);
    // ... 2-3 more pipeline+bind_group sets ...
}
```

With 64 iterations and 3 pipelines per iteration, that's 192 `set_pipeline` + 384 `set_bind_group` calls. Each `set_bind_group` call has CPU-side validation overhead in wgpu.

### Optimization: Hoist Params Bind Group Outside Loop

The `params_bind_group` at group(1) is identical across all pipelines and never changes within a batch. It only needs to be set once per pass, not re-set after every `set_pipeline`. However, wgpu resets all bind group state when `set_pipeline` is called (the pipeline layout might differ). Since all 4 pipelines use the same params_bgl at group(1), the re-set is technically required by the wgpu API but results in the same descriptor set being redundantly validated.

**Practical mitigation**: The `min_binding_size` optimization (already partially done -- pipeline.rs sets it for some entries). Verify ALL bind group layout entries across all shaders have `min_binding_size` set to skip runtime validation. Currently, several entries use the correct `NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64)` but others use `NonZeroU64::new(std::mem::size_of::<u32>() as u64)` -- the latter should be set to the actual minimum expected buffer size for the shader.

**Estimated impact**: **1-5% reduction in CPU-side command encoding time**. Mostly matters when GPU work per iteration is very short (small images, low polygon count).

---

## 11. Rasterize Shader: AABB Culling Could Use Integer Pixel Coords

### Current State

AABB culling operates in normalized [0,1] float coordinates:

```wgsl
// rasterize_error.wgsl line 150-157
let bb_min_x = min(pv0.x, min(pv1.x, pv2.x));
let bb_max_x = max(pv0.x, max(pv1.x, pv2.x));
// ...
if fx < bb_min_x || fx > bb_max_x || fy < bb_min_y || fy > bb_max_y {
    return;
}
```

This requires unpacking all 3 vertices (3x `unpack_vertex`, each doing 2 u32-to-f32 conversions + division) before the AABB test.

### Optimization: Integer-Space AABB Check Before Vertex Unpack

Pack the AABB as integer pixel coordinates during binning or shared memory load, and compare against integer pixel coords (px, py) directly:

```wgsl
// In cooperative load, precompute integer AABB per polygon:
let v0_raw = poly.data.y;
let v1_raw = poly.data.z;
let v2_raw = poly.data.w;
// x components in lower 16 bits, y in upper 16 bits
let min_x_16 = min(v0_raw & 0xFFFFu, min(v1_raw & 0xFFFFu, v2_raw & 0xFFFFu));
let max_x_16 = max(v0_raw & 0xFFFFu, max(v1_raw & 0xFFFFu, v2_raw & 0xFFFFu));
// Convert from u16 normalized to pixel: (val * image_width) / 65535
let bb_min_px = (min_x_16 * w) >> 16u;  // approximate: val * w / 65536
// ...
if px < bb_min_px || px > bb_max_px || py < bb_min_py || py > bb_max_py {
    continue; // skip without unpacking to float
}
```

This performs the AABB cull using integer arithmetic (cheaper than float) and avoids the full vertex unpack for polygons that miss. For the ~90% of polygons that fail AABB for any given pixel, this saves 3 float divisions.

**Estimated impact**: For 500 polygons where ~450 fail AABB per pixel, saves 450 * 6 float ops = 2700 FP ops per pixel. With 262K pixels per offspring, that's ~700M saved FP ops per offspring per iteration. **5-10% faster per-pixel inner loop**.

**Risk**: The integer approximation (`>> 16` instead of `/ 65535`) introduces up to 1 pixel of error in the AABB bounds. This means the integer AABB is slightly conservative (may include 1 extra pixel row/column), but this is safe -- it just means a few extra pixels proceed to the exact float half-space test, which correctly rejects them. No false negatives.

---

## 12. Incremental Eval: Framebuffer Update on Acceptance is Missing

### Current State

The init_framebuffers shader rasterizes the initial chain drawing into `chain_framebuffers` and computes `chain_total_errors`. But when the select shader accepts an offspring, it only updates `chain_total_errors`:

```wgsl
// select.wgsl line 301-304
if incremental && shared_accept == 1u {
    if local_id == 0u {
        atomicStore(&chain_total_errors[chain_id], shared_accepted_total_error);
    }
}
```

The `chain_framebuffers` buffer is NOT updated after acceptance. This means the cached framebuffer becomes stale after the first acceptance, and subsequent incremental evaluations re-rasterize the parent from scratch (because `chain_states` was updated but `chain_framebuffers` was not).

### Problem

After acceptance, `chain_states[chain_id]` has the new drawing but `chain_framebuffers[chain_id]` still has the old framebuffer. If the rasterize shader were optimized to read from `chain_framebuffers` (Optimization #2), it would get wrong old-error values. The current code works around this by re-rasterizing the parent every time, which is the source of the performance problem identified in #2.

### Optimization: Update Framebuffer on Acceptance

After the select shader accepts an offspring, dispatch the `init_framebuffers` shader for just that chain (or add framebuffer update logic to the select shader). This keeps the framebuffer in sync with `chain_states`, enabling Optimization #2.

However, triggering a per-chain init_framebuffers dispatch after select would require either:
- A second select-like pass that checks acceptance and dispatches conditionally (indirect dispatch)
- A GPU-driven conditional dispatch based on `control.new_best_found`

**Simpler approach**: The select shader already has all 64 threads cooperating on polygon copy after acceptance. After the polygon copy, the same 64 threads could update the framebuffer for the dirty bbox region. But the framebuffer update requires rasterizing ALL polygons in the dirty region (not just the changed one), which is the same O(polygons * dirty_pixels) work -- more suited to a dedicated dispatch with 2D spatial workgroups.

**Most practical implementation**: After the batch's select passes complete, check `control.new_best_found` on the CPU side. If set, dispatch `init_framebuffers` for the accepted chain(s) before the next batch. This adds one synchronous dispatch per improvement, which is rare (~5% of batches).

**Estimated impact**: Enables Optimization #2, which is the larger win. On its own, **necessary infrastructure** for 2x incremental eval speedup.

---

## Summary: Priority-Ranked New Optimization Opportunities

| # | Optimization | Pass | Est. Impact | Effort | Risk |
|---|-------------|------|-------------|--------|------|
| 2 | Read parent error from cached framebuffer (+ #12) | rasterize (incremental) | **2x faster incremental rasterize** | Medium | Medium (quantization drift) |
| 3 | Skip writeback for unmodified polygons in multi-mutation | mutate | 10-20% faster multi-mutation | Low | None |
| 6 | SoA shared memory layout (geometry-only prefetch) | rasterize | 10-20% for high polygon counts | Medium | Low |
| 11 | Integer-space AABB culling before float unpack | rasterize | 5-10% faster inner loop | Medium | Low |
| 1 | Parallelize bin_polygons tile count reset | binning | 5-15% of binning pass | Low | None |
| 8 | Second-level subgroup reduction | rasterize | 1-3% of reduction | Low | None |
| 5 | Push constants for hot params | all | 1-3% overall | Medium | Low |
| 7 | Increase default chain count for mutate/select saturation | mutate/select | 3-8% overall | Low | None (memory cost) |
| 10 | Verify min_binding_size on all BGL entries | CPU encoding | 1-5% CPU-side | Low | None |
| 4 | Parallel degenerate culling | select | Negligible (rare path) | Medium | None |
