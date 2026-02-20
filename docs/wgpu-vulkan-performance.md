# wgpu/Vulkan Performance Analysis

**Expert focus**: wgpu API usage, occupancy, buffer access patterns, async readback, subgroup ops, profiling

---

## Executive Summary

Three dominant bottlenecks:
1. **Mutate shader is catastrophically underutilized** — 64 workgroups of size 1 on 21,760 CUDA cores = 0.02% occupancy
2. **Synchronous readback blocks the CPU** — mandatory GPU bubble between every batch
3. **Rasterize has poor memory access** — each thread iterates all 1000 polygons from global memory

---

## 1. Workgroup Sizing

### Mutate: workgroup_size(1) — single biggest problem

64 threads total on a GPU with 170 SMs × 1536 threads per SM = 0.02% occupancy. Each SM runs 1 useful lane and 31 idle ones. Memory latency cannot be hidden.

**Best fix**: Split copy and mutate into separate passes:

```wgsl
// Pass 1a: Bulk copy (workgroup_size(256), fully parallel)
@compute @workgroup_size(256)
fn copy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_id = gid.x / 1000u;
    let poly_idx = gid.x % 1000u;
    if poly_idx < chain_states[chain_id].polygon_count {
        working_states[chain_id].polygons[poly_idx] = chain_states[chain_id].polygons[poly_idx];
    }
}

// Pass 1b: Mutate (workgroup_size(1), sequential per chain)
fn mutate_main(...) { /* just mutation logic */ }
```

For K=64: copy dispatch = 250 workgroups × 256 threads = 64,000 threads.

### Rasterize/Error: Consider 16x16 workgroups

Current 8x8 = 64 threads (2 warps). Try 16x16 = 256 threads (8 warps) for better per-SM occupancy. Shared memory for reduction: 256 × 4 = 1,024 bytes — trivial.

---

## 2. Occupancy Analysis (RTX 5090)

| Pass | Workgroups | Threads/WG | Total | Occupancy |
|------|-----------|-----------|-------|-----------|
| Mutate | 64 | 1 | 64 | ~0.02% (catastrophic) |
| Rasterize | 147,456 | 64 | 9.4M | ~100% (excellent) |
| Error reduce | 147,456 | 64 | 9.4M | ~100% (excellent) |
| Select | 64 | 1 | 64 | ~0.02% (catastrophic) |

**Target**: At 512 chains, mutate still only 512 threads, but rasterize goes to 1.18M workgroups.

---

## 3. Buffer Access Patterns

### Rasterize: broadcast reads from global memory

All 64 threads in an 8x8 workgroup read the same polygon. L1 cache handles this but 48 KB per chain exceeds L1 (128 KB per SM shared across all warps).

**Fix**: Cooperative loading into shared memory:

```wgsl
var<workgroup> shared_polys: array<Polygon, 64>;

for (var tile_start = 0u; tile_start < poly_count; tile_start += 64u) {
    if local_idx < tile_count {
        shared_polys[local_idx] = working_states[chain_id].polygons[tile_start + local_idx];
    }
    workgroupBarrier();
    for (var i = 0u; i < tile_count; i++) {
        let poly = shared_polys[i];
        // ... test + blend ...
    }
    workgroupBarrier();
}
```

Converts scattered global reads into coalesced loads → shared memory broadcasts. Shared memory is ~100x faster than global memory.

### Render target writes — already optimal

`render_targets[chain_id * w * h + py * w + px]` — adjacent threads write adjacent addresses. No change needed.

### Reference image as texture — not worth it

No interpolation needed (exact pixel access), so texture cache benefit is marginal. Not worth the complexity.

---

## 4. Double-Buffered Readback

Two control staging buffers + two readback staging buffers, alternating each batch:

```rust
pub fn run_batch(&mut self, ...) -> Option<Drawing> {
    // 1. Read PREVIOUS batch results (other buffer)
    let prev_result = if let Some(pending) = self.pending_readback.take() {
        self.read_pending(pending)
    } else { None };

    // 2. Encode and submit NEXT batch (current buffer)
    // ... encode passes ...
    encoder.copy_buffer_to_buffer(control -> staging[current_buf]);
    let sub_idx = queue.submit(encoder.finish());

    // 3. Start async map for this buffer
    staging[current_buf].map_async(Read, |_| {});
    self.pending_readback = Some(PendingReadback { buf_idx, sub_idx });
    self.current_buf_idx = 1 - self.current_buf_idx;

    // 4. Return PREVIOUS batch's result
    prev_result
}
```

GPU never stalls between batches. Expected 20-30% throughput improvement.

---

## 5. Queue and Command Buffer Notes

- **Multiple queues**: wgpu exposes single queue. NVIDIA serializes compute anyway. Not worth pursuing.
- **Multiple command buffers**: No benefit — driver batches them. Current single-buffer-per-batch is correct.
- **Error accumulator reset**: The `atomicExchange` in select already resets accumulators. The CPU-side `queue.write_buffer` zero-fill may be redundant. Can remove after verification.

---

## 6. Shader Compilation / Pipeline Caching

wgpu 22.1.0 supports `PipelineCache`:

```rust
let cache = device.create_pipeline_cache(&PipelineCacheDescriptor {
    label: Some("shader_cache"),
    data: load_cache_from_disk(),
    fallback: true,
});
// Pass cache to create_compute_pipeline
// After creation: cache.get_data() → save to disk
```

Eliminates 100-500ms shader recompilation on subsequent runs.

---

## 7. Push Constants vs Uniform Buffer

wgpu supports push constants via `PipelineLayoutDescriptor::push_constant_ranges`. Avoids one pointer dereference per access. But params (96 bytes) fits entirely in L1 after first access. **Not worth the refactoring.**

---

## 8. Subgroup Operations

RTX 5090 fully supports subgroup ops. Request `Features::SUBGROUP`.

### Error reduction: eliminate most shared memory

```wgsl
enable subgroups;

var sum = subgroupAdd(pixel_error);
// Only need shared mem for cross-subgroup reduction (2 subgroups at WG size 64)
```

`subgroupAdd` → single `redux.sync.add.u32` PTX instruction (~3 cycles vs ~30 for tree reduction).

### Rasterize: warp-level polygon skip

```wgsl
if !subgroupAny(hits) {
    continue; // entire warp skips this polygon
}
```

Helps for small polygons affecting only a few workgroups.

---

## 9. Timestamp Queries — Do This First

This is how you find the actual bottleneck instead of guessing:

```rust
required_features: Features::TIMESTAMP_QUERY,

let timestamp_query_set = device.create_query_set(&QuerySetDescriptor {
    label: Some("timestamps"),
    ty: QueryType::Timestamp,
    count: 12,
});
```

Instrument each compute pass with `timestamp_writes` in `ComputePassDescriptor`. Resolve, read back, convert to nanoseconds via `queue.get_timestamp_period()`.

**This should be the first action item.** Measure each pass, then focus optimization on whichever is actually dominant.

Predicted order: mutate > rasterize > error_reduce > select > migrate.

---

## Priority Summary

| Priority | Change | Effort | Expected Speedup |
|----------|--------|--------|-----------------|
| 1 | Timestamp queries (profiling) | 30 min | Informs all other decisions |
| 2 | Split polygon copy from mutate | 2 hours | Removes 48 KB sequential bottleneck |
| 3 | Double-buffered readback | 2-3 hours | Eliminates GPU idle bubble |
| 4 | Increase chain count to 256-512 | 30 min | Better GPU utilization + search |
| 5 | Shared memory polygon loading | 2 hours | 16x less global memory bandwidth |
| 6 | Increase batch size to 50 | 5 min | Reduce sync overhead |
| 7 | Try workgroup_size(16,16,1) | 15 min | May improve SM utilization |
| 8 | Subgroup ops in error_reduce | 30 min | Small but free improvement |
| 9 | Pipeline caching | 30 min | Eliminates startup latency |
