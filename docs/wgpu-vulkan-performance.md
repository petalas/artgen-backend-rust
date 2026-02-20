# wgpu/Vulkan Performance Analysis

**System:** RTX 5090 (32GB VRAM), WSL2, wgpu 22.1.0
**Current:** 512 chains, 50 iterations/batch, ~2.8GB VRAM used, 5-pass compute pipeline

---

## 1. Command Buffer Submission: Synchronous Readback Stalls (HIGH IMPACT)

### Problem

The main loop in `run_batch()` is entirely synchronous:

```
encode 50 iterations -> submit -> device.poll(Maintain::Wait) -> map_async -> block on recv()
```

`device.poll(Maintain::Wait)` in `read_control_flags()` (line 207 of `mod.rs`) blocks the CPU thread until ALL GPU work finishes. This means:

1. The CPU is idle while the GPU executes 50 iterations of 5 passes each.
2. The GPU is idle while the CPU processes the result and re-encodes the next command buffer.
3. There is zero overlap between CPU work and GPU work.

### Fix: Double-Buffered Command Submission

Use two sets of staging buffers and alternate between them. While the GPU executes batch N+1, the CPU maps and reads the results from batch N.

```rust
// Pseudocode for double-buffered submission
struct DoubleBuffer {
    control_staging: [Buffer; 2],
    current: usize,
    pending_submission: Option<SubmissionIndex>,
}

fn run_batch(&mut self) -> Option<Drawing> {
    let read_idx = self.current;
    let write_idx = 1 - self.current;

    // 1. If there's a pending submission, poll it and read results
    //    (this should already be done since we submitted it last frame)
    let result = if self.pending_submission.is_some() {
        self.read_results(read_idx)
    } else {
        None
    };

    // 2. Encode and submit the NEXT batch (using write_idx staging)
    let encoder = self.encode_batch(write_idx);
    let idx = self.queue.submit(once(encoder.finish()));
    self.pending_submission = Some(idx);

    // 3. Swap
    self.current = write_idx;

    result // from the PREVIOUS batch
}
```

The key wgpu API is `queue.submit()` returning a `SubmissionIndex`, and then using `device.poll(Maintain::WaitForSubmissionIndex(idx))` to wait only for a specific submission rather than all work.

**Expected improvement:** 10-30% throughput increase from CPU/GPU overlap. The GPU should rarely idle between batches.

### Additional: `queue.write_buffer()` Implicit Synchronization

Three `queue.write_buffer()` calls happen before each batch (lines 87, 97, 101 of `mod.rs`):

```rust
p.queue.write_buffer(&p.params_buf, 0, ...);
p.queue.write_buffer(&p.control_flags_buf, 0, ...);
p.queue.write_buffer(&p.error_accumulators_buf, 0, &zeros);
```

In wgpu, `write_buffer` is internally staged -- it does not block -- but it does create an implicit dependency: the writes must complete before the next `submit()` uses them. This is fine with the current single-submit pattern, but with double-buffering you would want to ensure these writes go into the same command encoder as the compute passes, or at least are submitted before the compute work. As-is, this is not a bottleneck.

---

## 2. Error Accumulator Reset: Unnecessary CPU-Side Buffer Write (MEDIUM IMPACT)

### Problem

Every batch, the CPU writes a zero-filled buffer to reset error accumulators:

```rust
let zeros = vec![0u8; p.chain_count as usize * 4]; // 2KB allocation per batch
p.queue.write_buffer(&p.error_accumulators_buf, 0, &zeros);
```

This creates a `Vec` allocation + memset + a staged write every batch. With 50 iterations/batch, and the select shader already doing `atomicExchange(..., 0)` to reset per-iteration, this is only needed at the start of each batch.

### Fix: GPU-Side Clear via `encoder.clear_buffer()`

```rust
encoder.clear_buffer(&p.error_accumulators_buf, 0, None);
```

`clear_buffer()` issues a `vkCmdFillBuffer` on the GPU side. Zero CPU allocation, zero staging transfer. This is a pure GPU-side operation with much lower overhead.

**Expected improvement:** Eliminates a small per-batch allocation and transfer. Minor but clean.

---

## 3. Mutate Shader: Workgroup Size 1 (HIGH IMPACT)

### Problem

The mutate shader uses `@workgroup_size(1)`:

```wgsl
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
```

Dispatched as `dispatch_workgroups(chain_count, 1, 1)` = 512 workgroups of 1 thread each.

On NVIDIA GPUs, a warp is 32 threads. A workgroup of size 1 means 31 of every 32 ALU lanes are permanently masked off. The SM scheduler can interleave warps from different workgroups to hide latency, but:

- Each workgroup still occupies a full warp slot.
- Shared memory/register file per-workgroup overhead is paid 512 times instead of 16 times (512/32).
- Occupancy is constrained by workgroup count limits per SM.

The real issue is that this shader is fundamentally serial per-chain (loops over all polygons, does sequential mutations). There is no easy way to parallelize mutation within a chain because each mutation depends on the previous one.

### Partial Fix: Pack Multiple Chains per Workgroup

Instead of 1 chain per workgroup, pack 32 chains into one workgroup so each thread handles one chain but they share a warp:

```wgsl
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_id = gid.x; // 0..511
```

Dispatch as `dispatch_workgroups(chain_count / 32, 1, 1)` = 16 workgroups.

This does NOT change the algorithmic work per thread, but it ensures each warp is fully occupied. The 32 threads in a warp execute in lockstep but because each thread operates on independent chain data, there are no warp divergence issues (each thread follows its own random mutation path, but the hardware just masks as needed -- this is inherently divergent work, so the improvement is modest).

**Expected improvement:** 5-15% for the mutate pass. The mutate pass is not the bottleneck (rasterize is), so total improvement is small.

---

## 4. Select/Migrate Shader: Also Workgroup Size 1 (LOW-MEDIUM IMPACT)

Same issue as mutate. `select_main` and `migrate_main` both use `@workgroup_size(1)`. Same fix: bump to `@workgroup_size(32)` or `@workgroup_size(64)` and dispatch fewer workgroups.

The select shader does a per-polygon copy loop when the candidate wins:

```wgsl
for (var i = 0u; i < pc; i++) {
    chain_states[chain_id].polygons[i] = working_states[chain_id].polygons[i];
}
```

This is a lot of sequential memory traffic (up to 1000 * 48 = 48KB per chain). With workgroup_size(1), each SM processes these one chain at a time. With workgroup_size(32), 32 chains can be in flight simultaneously on one SM, allowing the memory controller to coalesce and pipeline the loads/stores much better.

**Expected improvement:** 5-10% on the select pass.

---

## 5. Rasterize: Per-Pixel Polygon Array Reads from Global Memory (HIGH IMPACT -- MOST EXPENSIVE PASS)

### Problem

The rasterize shader is dispatched as `(W/8, H/8, chain_count)` with workgroup_size `(8, 8, 1)`. For a 384x384 image with 512 chains, that is:

- 48 * 48 * 512 = 1,179,648 workgroups
- 64 threads per workgroup = ~75 million threads

Each thread reads the polygon array from the `working_states` storage buffer:

```wgsl
for (var i = 0u; i < poly_count; i++) {
    let poly = working_states[chain_id].polygons[i];
    // AABB cull + edge test + blend
}
```

Each polygon is 48 bytes. With 1000 polygons, each thread potentially reads 48KB of global memory. The 64 threads in a workgroup all read the SAME polygon data (same chain_id for all pixels in the 8x8 tile), but because this is storage buffer memory (not uniform/texture), the GPU cannot broadcast the read. Each thread issues its own load, and the L1/L2 caches must handle 64 simultaneous reads of the same address.

### Fix: Load Polygons into Workgroup Shared Memory

The 8x8 workgroup has 64 threads. They can cooperatively load polygon data into `var<workgroup>` shared memory before doing per-pixel processing:

```wgsl
var<workgroup> shared_polys: array<Polygon, 64>; // 48 * 64 = 3072 bytes

// Process polygons in tiles of 64
for (var tile_start = 0u; tile_start < poly_count; tile_start += 64u) {
    // Cooperative load: each thread loads one polygon
    let load_idx = tile_start + local_idx;
    if load_idx < poly_count {
        shared_polys[local_idx] = working_states[chain_id].polygons[load_idx];
    }
    workgroupBarrier();

    // Each thread processes all loaded polygons against its pixel
    let tile_end = min(tile_start + 64u, poly_count);
    for (var i = tile_start; i < tile_end; i++) {
        let poly = shared_polys[i - tile_start];
        // AABB + edge test + blend
    }
    workgroupBarrier();
}
```

This reduces global memory reads by 64x (one load per polygon per workgroup instead of per thread). Shared memory on NVIDIA has ~100x lower latency than global memory.

**Caveat:** wgpu/WGSL has a limit on shared memory per workgroup (typically 16KB for Vulkan, 32KB on NVIDIA). At 48 bytes/polygon, you can fit 341 polygons in 16KB. Tiling with 64-polygon chunks as shown above stays well within limits (3072 bytes).

**Expected improvement:** 30-60% reduction in rasterize pass time. This is the bottleneck pass, so overall throughput could improve 20-40%.

---

## 6. Reference Image: Storage Buffer vs Texture (MEDIUM IMPACT)

### Problem

The reference image is stored as a `storage<read>` buffer of packed u32:

```wgsl
@group(0) @binding(1) var<storage, read> reference_image: array<u32>;
```

In the error_reduce shader, each thread reads one pixel:

```wgsl
let reference = reference_image[ref_idx];
```

Storage buffers go through the general-purpose L1/L2 cache hierarchy. Texture memory, on the other hand, has a dedicated texture cache optimized for 2D spatial locality -- when one thread reads pixel (x, y), neighboring threads reading (x+1, y) or (x, y+1) get cache hits from the texture cache's space-filling curve layout.

### Fix: Use a wgpu Texture + `textureLoad()`

Replace the storage buffer with a `texture_2d<f32>`:

```rust
// Rust: create texture instead of buffer
let reference_texture = device.create_texture(&TextureDescriptor {
    size: Extent3d { width: image_width, height: image_height, depth_or_array_layers: 1 },
    format: TextureFormat::Rgba8Unorm,
    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    // ...
});
queue.write_texture(/* upload reference RGBA */);
```

```wgsl
// WGSL: read from texture
@group(0) @binding(1) var reference_image: texture_2d<f32>;

let ref_color = textureLoad(reference_image, vec2<u32>(px, py), 0);
let refr = ref_color.x * 255.0;
let refg = ref_color.y * 255.0;
let refb = ref_color.z * 255.0;
```

The texture cache is purpose-built for 2D access patterns and the error_reduce shader's access pattern (8x8 tile of spatially adjacent pixels) is the ideal case.

**Expected improvement:** 10-20% on error_reduce pass. The workgroup reduction is also significant work in this shader, so the texture cache wins apply only to the load portion.

---

## 7. Fused Rasterize + Error Reduce Pass (MEDIUM-HIGH IMPACT)

### Problem

Currently, rasterize writes pixels to `render_targets` (a huge buffer: 512 * 384 * 384 * 4 = 302MB), and then error_reduce reads them back. This is a round-trip through global memory:

```
rasterize: compute pixel -> write to render_targets[pixel_idx]
   (implicit barrier between compute passes)
error_reduce: read render_targets[pixel_idx] -> compute error -> reduce
```

The `render_targets` buffer exists solely to transfer per-pixel colors between these two passes. At 302MB, it is by far the largest buffer and dominates VRAM usage.

### Fix: Fuse Into a Single Shader

Compute the pixel color and immediately compare against the reference, accumulating error in-place:

```wgsl
@compute @workgroup_size(8, 8, 1)
fn rasterize_and_error(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    // ... rasterize pixel (same as current rasterize.wgsl) ...

    // Immediately compare against reference (no intermediate buffer)
    let ref_pixel = reference_image[ref_idx]; // or textureLoad()
    let refr = f32(ref_pixel & 0xFFu);
    // ... compute L1 error ...

    // Workgroup reduce (same as current error_reduce.wgsl)
    shared_errors[local_idx] = pixel_error;
    workgroupBarrier();
    // ... reduction tree ...
    if local_idx == 0u {
        atomicAdd(&error_accumulators[chain_id], shared_errors[0]);
    }
}
```

Benefits:
1. **Eliminate the 302MB render_targets buffer entirely.** VRAM drops from ~2.8GB to ~2.5GB.
2. **Eliminate one full compute pass dispatch per iteration.** That is 50 * (barrier + dispatch) overhead removed.
3. **Each pixel value stays in registers** -- no round-trip through global memory. On the RTX 5090, global memory bandwidth is shared across all SMs; eliminating 302MB * 2 (write + read) * 512 chains per iteration of memory traffic is massive.
4. **Bind group simplification:** One fewer pipeline, one fewer bind group layout, fewer descriptor sets.

**Caveat:** The fused shader will need bindings for both `working_states` (for rasterize) and `reference_image` (for error). This means more bindings in one bind group, but wgpu supports up to 8 bindings per group easily.

**Expected improvement:** 15-30% total throughput improvement. Memory traffic reduction is the dominant win.

---

## 8. `queue.write_buffer` for Params: Consider Inline Push Constants (LOW IMPACT)

### Problem

`GpuParams` is 96 bytes, written via `queue.write_buffer` every batch:

```rust
p.queue.write_buffer(&p.params_buf, 0, bytemuck::bytes_of(&params));
```

This goes through wgpu's staging belt (allocate staging buffer, memcpy, schedule DMA transfer). For 96 bytes, the staging overhead dominates.

### Alternative: Push Constants

wgpu supports push constants via `PushConstantRange` in the pipeline layout. Push constants are written directly into the command buffer -- zero allocation, zero DMA transfer, immediate availability at shader execution time.

```rust
// Pipeline layout
let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
    push_constant_ranges: &[PushConstantRange {
        stages: ShaderStages::COMPUTE,
        range: 0..96,
    }],
    // ...
});

// In command recording
pass.set_push_constants(0, bytemuck::bytes_of(&params));
```

**Caveat:** Push constants have a size limit of 128 bytes (guaranteed by Vulkan spec). At 96 bytes, `GpuParams` fits but leaves minimal room for growth. Also, WGSL does not natively support push constants -- wgpu maps them through a polyfill that uses a small uniform buffer internally. The real-world benefit over `write_buffer` for a 96-byte uniform is negligible.

**Expected improvement:** Negligible. Not worth the complexity.

---

## 9. Timestamp Queries for Profiling (DIAGNOSTIC -- NO THROUGHPUT CHANGE)

### Problem

There is no way to know which pass is the bottleneck without measurement. All analysis above is theoretical.

### Fix: Enable Timestamp Queries

wgpu 22.1.0 supports `Features::TIMESTAMP_QUERY`. Use `ComputePassDescriptor::timestamp_writes` (currently set to `None` on every pass).

```rust
// Request feature
required_features: Features::TIMESTAMP_QUERY,

// Create query set
let query_set = device.create_query_set(&QuerySetDescriptor {
    ty: QueryType::Timestamp,
    count: 12, // 2 per pass * 5 passes + 2 for batch start/end
    label: Some("perf_queries"),
});

// On each pass
let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
    label: Some("rasterize"),
    timestamp_writes: Some(ComputePassTimestampWrites {
        query_set: &query_set,
        beginning_of_pass_write_index: Some(2),
        end_of_pass_write_index: Some(3),
    }),
});

// After submit, resolve and read back
encoder.resolve_query_set(&query_set, 0..12, &timestamp_buf, 0);
// Map timestamp_buf, convert to nanoseconds using queue.get_timestamp_period()
```

This would immediately reveal whether rasterize, error_reduce, mutate, or select is the bottleneck, and by how much. All other optimizations should be prioritized based on this data.

**Recommendation:** Implement this first. It costs nothing at runtime (timestamp queries are essentially free on NVIDIA) and eliminates guesswork.

---

## 10. WSL2-Specific Considerations

### Known Issues

- **Dozen driver (Vulkan-on-D3D12):** Already handled by `ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` and chain count capping. The max_storage_buffer_binding_size is typically lower on Dozen than native Vulkan.

- **PCI-e passthrough overhead:** WSL2 uses a virtual GPU (dxgkrnl VMBus). Every GPU submission crosses the VM boundary. This adds ~10-50us per `queue.submit()` call. With the current pattern of 1 submit per batch (50 iterations), this is already well-amortized. Increasing batch size further (e.g., 100 or 200 iterations) would reduce this overhead proportionally but would also increase latency for detecting new best solutions.

- **Memory mapping latency:** `buffer.map_async()` + `device.poll(Maintain::Wait)` in WSL2 has higher latency than native because the readback must cross the VM boundary. Double-buffering (item 1) mitigates this by overlapping the map with GPU execution.

### Potential: Native Vulkan

Running natively on Linux (not WSL2) would eliminate the VMBus overhead entirely. If the workload is throughput-bound (which it should be at 512 chains), the difference is likely <5%. If the workload is submission-bound, the difference could be 10-20%.

---

## 11. Increasing Chain Count with Available VRAM (LOW-MEDIUM IMPACT)

### Problem

Only ~2.8GB of 32GB VRAM is used. The RTX 5090 has substantial spare capacity.

### Analysis

The dominant buffer is `render_targets`: `chain_count * W * H * 4` bytes. At 512 chains and 384x384:
- render_targets = 512 * 384 * 384 * 4 = 302MB
- chain_states = 512 * 48032 = 23.5MB
- working_states = 512 * 48032 = 23.5MB
- reference = 384 * 384 * 4 = 0.6MB
- Total ~350MB (matches the ~2.8GB estimate if accounting for wgpu internal overhead)

If the render_targets buffer is eliminated by fusing rasterize + error_reduce (item 7), the per-chain cost drops to ~96KB (two DrawingState buffers). With 30GB available, you could theoretically run ~300,000 chains. The bottleneck shifts to compute throughput and dispatch limits.

Practically, increasing from 512 to 2048 or 4096 chains (with the fused shader) would be feasible and would provide more population diversity, potentially improving convergence quality.

**Caveat:** The `max_storage_buffer_binding_size` limit (typically 128MB-2GB depending on adapter) may cap the chain_states buffer before VRAM runs out. The existing capping logic handles this correctly.

---

## 12. Batch Size vs Latency Tradeoff

### Current State

`GPU_ITERATIONS_PER_BATCH = 50` encodes 50 * 5 = 250 compute passes into one command buffer. This is good for amortizing submission overhead. However:

- All 250 passes execute before the CPU can check for a new global best.
- If a global best is found on iteration 5 of 50, the remaining 45 iterations still execute (their results are valid -- each iteration operates on its chain-local best, not the global best, so no work is wasted from a correctness standpoint).

Increasing to 100 or 200 iterations per batch would reduce CPU-side overhead proportionally but is unlikely to improve throughput significantly since the current 50 already amortizes well.

**Recommendation:** Leave at 50 unless profiling shows significant CPU overhead per batch.

---

## Priority-Ordered Recommendations

| Priority | Item | Expected Impact | Effort |
|----------|------|----------------|--------|
| 1 | Timestamp queries (#9) | Diagnostic | Low |
| 2 | Fuse rasterize + error_reduce (#7) | 15-30% throughput | Medium |
| 3 | Shared memory polygon tiling in rasterize (#5) | 20-40% on rasterize | Medium |
| 4 | Double-buffered submission (#1) | 10-30% throughput | Medium |
| 5 | Reference image as texture (#6) | 10-20% on error pass | Low |
| 6 | GPU-side `clear_buffer` for accumulators (#2) | Minor cleanup | Trivial |
| 7 | Mutate/select workgroup size > 1 (#3, #4) | 5-15% on those passes | Low |
| 8 | Increase chain count post-fusion (#11) | Quality improvement | Low |
