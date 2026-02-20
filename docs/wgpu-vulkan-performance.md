# wgpu / Vulkan Performance Analysis

Deep analysis of the artgen-backend-rust GPU compute pipeline for performance improvement opportunities, focusing on wgpu API patterns, Vulkan backend behavior, and shader optimization. Covers `src/gpu_evolver/` (mod.rs, pipeline.rs, buffers.rs), `src/shaders/*.wgsl`, and host-side orchestration in `src/main.rs`.

## Current Architecture Summary

The GPU pipeline implements a (1+lambda)-ES evolutionary algorithm across K independent chains:

- **4 compute passes per iteration**: mutate -> rasterize_error (fused) -> select -> migrate (periodic)
- **50 iterations** batched into a single command buffer
- **Double-buffered staging** for async readback of control flags, timestamps, and fitness values
- **Pipeline cache** persisted to disk (Vulkan `VK_EXT_pipeline_creation_cache_control`)
- **Timestamp queries** for per-pass profiling

---

## 1. Push Constants for GpuParams (High Impact)

### Current State
`GpuParams` (128 bytes) is written via `queue.write_buffer()` every batch. This creates a staging copy, then a DMA transfer before the first dispatch. Every compute pass reads this buffer through a uniform binding.

### Opportunity
Vulkan push constants allow up to 128 bytes of data to be inlined directly into the command buffer -- no buffer allocation, no descriptor binding, no cache miss on first access. The current `GpuParams` struct is exactly 128 bytes, which is the minimum guaranteed push constant size in Vulkan.

### Implementation
wgpu exposes push constants via `Features::PUSH_CONSTANTS`. The pipeline layout declares `push_constant_ranges`, and you call `pass.set_push_constants()` instead of writing a buffer.

```rust
// In pipeline creation:
let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
    push_constant_ranges: &[PushConstantRange {
        stages: ShaderStages::COMPUTE,
        range: 0..128,
    }],
    ..
});

// In command encoding:
pass.set_push_constants(ShaderStages::COMPUTE, 0, bytemuck::bytes_of(&params));
```

In WGSL, replace `var<uniform> params: Params` with `var<push_constant> params: Params`.

### Expected Benefit
- Eliminates one buffer write per batch (currently `queue.write_buffer` to `params_buf`)
- Removes a descriptor set binding slot (frees up one uniform binding per pass)
- Push constant data is embedded directly in the command stream, accessed from GPU-local registers or L0 cache on NVIDIA hardware
- On RTX 5090, push constants live in the command processor's dedicated register file, giving effectively zero-latency reads

### Caveats
- `max_push_constant_size` must be >= 128 bytes. All desktop Vulkan drivers guarantee at least 128 bytes (NVIDIA typically supports 256). The code should query the adapter limit and fall back to the current uniform buffer path if the limit is insufficient.
- wgpu's `Features::PUSH_CONSTANTS` is not part of the WebGPU standard, so this is a native-only optimization.

---

## 2. Subgroup Operations for Error Reduction (High Impact)

### Current State
The rasterize_error shader performs a binary tree reduction in shared memory to sum per-pixel errors within each 16x16 workgroup (256 threads). This takes 8 `workgroupBarrier()` steps:

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

### Opportunity
Vulkan subgroup operations (`VK_KHR_shader_subgroup`) allow threads within a warp/wave (32 on NVIDIA, 64 on AMD) to communicate directly through register shuffles, without shared memory or barriers. wgpu exposes this via `Features::SUBGROUP`.

A 256-thread reduction can be done in two phases:
1. **Intra-subgroup**: `subgroupAdd()` across 32 threads (zero shared memory, zero barriers)
2. **Inter-subgroup**: Only 8 partial sums (256/32) need the shared memory path

This replaces 8 barrier-synchronized steps with 1 subgroup intrinsic + ~3 barrier steps.

### Implementation
```wgsl
// Phase 1: subgroup reduction (no barriers needed)
let subgroup_sum = subgroupAdd(pixel_error);

// Phase 2: one thread per subgroup writes to shared
if subgroupElect() {
    shared_errors[local_idx / subgroup_size] = subgroup_sum;
}
workgroupBarrier();

// Phase 3: first subgroup reduces the 8 partial sums
if local_idx < 8u {
    let val = shared_errors[local_idx];
    let final_sum = subgroupAdd(val);
    if local_idx == 0u {
        atomicAdd(&error_accumulators[chain_id], final_sum);
    }
}
```

### Expected Benefit
- On NVIDIA RTX 5090 (warp size 32): eliminates 5 of 8 barrier steps, reduces shared memory traffic by ~87%
- Subgroup shuffles are single-cycle on NVIDIA SM hardware
- The rasterize_error pass is typically the bottleneck (50-70% of GPU time), so even a modest per-workgroup speedup compounds across millions of dispatches

### Caveats
- Requires `Features::SUBGROUP` in wgpu (landed in wgpu 0.19+, available in wgpu 22.1.0)
- Need a fallback path if the feature is not available (keep current shared memory reduction)
- Subgroup size varies by vendor (32 NVIDIA, 32/64 AMD) -- use `@builtin(subgroup_size)` in the shader

---

## 3. Rasterize_Error Workgroup Size Tuning (Medium Impact)

### Current State
The rasterize_error shader uses `@workgroup_size(16, 16, 1)` = 256 threads. Each workgroup processes a 16x16 pixel tile for a single offspring, loading polygons cooperatively into shared memory in tiles of 768.

### Opportunity: Occupancy vs. Register Pressure
On NVIDIA SM hardware (RTX 5090, SM 100), each SM can execute multiple workgroups concurrently. The maximum occupancy depends on:

- **Register count per thread**: The rasterize_error shader has ~20 live registers per thread. With 256 threads/workgroup, that is ~5120 registers per workgroup. NVIDIA SM 100 has 65536 registers per SM, allowing ~12 concurrent workgroups, but shared memory is the limiter.
- **Shared memory**: `shared_polys` (12288 bytes) + `shared_errors` (1024 bytes) = 13312 bytes per workgroup. RTX 5090 has 128KB shared memory per SM, allowing ~9 concurrent workgroups.

Reducing workgroup size to `(8, 8, 1)` = 64 threads would:
- Reduce shared memory to ~3.3KB per workgroup (768/4 = 192 polygons in tile + 256 bytes for errors)
- Allow more concurrent workgroups per SM (up to ~38)
- But increase the number of polygon tile passes (more shared memory loads per pixel)

The optimal tradeoff is empirical. A good middle ground might be `(16, 8, 1)` = 128 threads with a tile size of 384 polygons (~6.4KB shared), which would allow ~18-20 concurrent workgroups.

### Implementation
This requires changing both the shader `@workgroup_size` and the host dispatch calculation. The polygon tile size in `shared_polys` should scale proportionally.

### Expected Benefit
- 10-30% improvement in rasterize_error pass depending on polygon count
- Higher occupancy means better latency hiding for memory-bound texture reads
- Most impactful at lower polygon counts where the shader is more memory-bound

---

## 4. `min_binding_size` for Bind Group Layout Validation (Low-Medium Impact)

### Current State
All `BindGroupLayoutEntry` declarations use `min_binding_size: None`:

```rust
BindGroupLayoutEntry {
    binding: 0,
    ty: BindingType::Buffer {
        min_binding_size: None,
        ..
    },
    ..
}
```

### Opportunity
Setting `min_binding_size` to the actual minimum struct size (e.g., `NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64)`) enables wgpu to skip per-draw-call validation of buffer sizes. With `None`, wgpu validates every `set_bind_group()` call at runtime.

In a tight loop with 50 iterations x 5 passes = 250 `set_bind_group()` calls per batch, this validation overhead adds up. It is a CPU-side optimization that reduces command encoding time.

### Implementation
```rust
use std::num::NonZeroU64;

BindGroupLayoutEntry {
    binding: 0,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
    },
    count: None,
}
```

### Expected Benefit
- Eliminates per-dispatch buffer size validation on the CPU side
- Marginal per-call, but with 250+ bind group sets per batch it accumulates
- Also catches binding mismatches at pipeline creation time rather than at dispatch time (better error messages)

---

## 5. Pre-encoded Command Buffers for Repeated Iterations (Medium Impact)

### Current State
Every `run_batch()` call encodes a fresh command buffer with 50 iterations of the 4-pass loop. Since the dispatch dimensions, pipeline bindings, and workgroup counts are identical across iterations (only `params.iteration_number` changes), much of this encoding is redundant.

### Opportunity: Reusable Render Bundles / Command Buffer Templates
While wgpu does not directly support Vulkan secondary command buffers for compute, there are two approaches:

**Approach A: Encode once, submit multiple times**
If the params_buf write and control_flags_buf write were moved to a separate (small) command buffer that runs first, the main iteration command buffer could be pre-encoded once and resubmitted. The main iteration loop only depends on:
- `params_buf` contents (written by `queue.write_buffer` before submission)
- `control_flags_buf` contents (written by `queue.write_buffer` before submission)

Since `queue.write_buffer` happens before `queue.submit`, a pre-encoded command buffer would see the updated data.

However, wgpu `CommandBuffer` is consumed on submission (single-use). The alternative is to pre-encode the compute passes into a helper function that produces command buffers from a cached pattern, minimizing the encoding overhead.

**Approach B: Vulkan indirect dispatch**
The dispatch dimensions are constant, so indirect dispatch is not helpful here. However, if chain counts were dynamic, `dispatch_workgroups_indirect` would avoid re-encoding.

### Implementation
Factor out the pass encoding into a builder that caches pipeline/bind group references and generates command buffers with minimal overhead. The key insight is that `set_pipeline` and `set_bind_group` are the expensive encoding calls -- if these are the same every iteration, the command encoder's internal state tracking is doing redundant work.

```rust
// Encode the inner loop body once into a closure
let encode_iteration = |encoder: &mut CommandEncoder, ts: Option<&QuerySet>| {
    // ... 4 compute passes (same code as current, but factored out)
};
```

### Expected Benefit
- Reduces CPU encoding time per batch by eliminating redundant pipeline/bind group state transitions
- On NVIDIA drivers, the Vulkan command buffer compiler can better optimize repeated identical dispatches
- Most impactful when the GPU is fast enough that CPU encoding becomes the bottleneck

---

## 6. Eliminate Redundant `set_bind_group` in Inner Loop (Medium Impact)

### Current State
Inside the 50-iteration loop, each compute pass calls `set_pipeline()` and `set_bind_group()`:

```rust
for i in 0..iterations {
    // Pass 1: Mutate
    pass.set_pipeline(&p.mutate_pipeline);
    pass.set_bind_group(0, &p.mutate_bind_group, &[]);
    pass.dispatch_workgroups(active, 1, 1);
    // ... 3 more passes
}
```

Each `begin_compute_pass` / `end` pair also adds overhead (synchronization barriers). Since all 50 iterations use the same pipelines and bind groups, this creates 200+ redundant API calls per batch.

### Opportunity
Restructure to minimize pass boundaries. Currently there are 50 x 4 = 200 compute passes (each with begin/end). By using a single compute pass per pipeline and dispatching all iterations worth of work, you could reduce to 4 passes total. However, the iteration loop requires synchronization between passes (mutate must finish before rasterize_error reads the output).

A more practical optimization: use fewer, larger compute passes. Each iteration requires 3-4 pipeline switches, but within a single compute pass you can switch pipelines without ending the pass. The key constraint is timestamp writes, which are per-pass.

For the 49 non-profiled iterations, encode all 4 pipeline dispatches within a single compute pass (no timestamps). Only the last iteration uses separate passes for timestamp measurement.

### Implementation
```rust
// Non-profiled iterations: single compute pass, multiple pipeline dispatches
{
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("batch_inner"),
        timestamp_writes: None,
    });
    for i in 0..iterations - 1 {
        pass.set_pipeline(&p.mutate_pipeline);
        pass.set_bind_group(0, &p.mutate_bind_group, &[]);
        pass.dispatch_workgroups(active, 1, 1);

        pass.set_pipeline(&p.rasterize_error_pipeline);
        pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, active * lambda);

        pass.set_pipeline(&p.select_pipeline);
        pass.set_bind_group(0, &p.select_bind_group, &[]);
        pass.dispatch_workgroups(active, 1, 1);

        // Migration check...
    }
}
// Last iteration: separate passes for timestamps
```

### Expected Benefit
- Reduces compute pass begin/end overhead from ~200 to ~6 (1 big pass + 4-5 profiled passes)
- On Vulkan, each compute pass boundary inserts a full pipeline barrier. Combining dispatches within a single pass allows the driver to optimize barrier placement (only inserting where UAV hazards actually exist)
- NVIDIA drivers on Vulkan can coalesce multiple dispatches within a single pass into fewer GPU-side submissions

### Caveat
wgpu inserts implicit storage buffer barriers between dispatches within the same compute pass when it detects UAV (read-after-write / write-after-read) hazards. Since mutate writes `working_states` and rasterize_error reads it, wgpu will correctly insert a barrier. But it may be overly conservative -- worth benchmarking.

---

## 7. Texture Storage Format for Reference Image (Low Impact)

### Current State
The reference image is stored as `TextureFormat::Rgba8Unorm` and loaded in the shader via `textureLoad()`. This goes through the texture unit, which provides spatial locality caching.

### Opportunity
The texture unit is well-suited here because adjacent threads in a 16x16 workgroup access adjacent pixels, giving good 2D locality. No change is recommended for the reference image format.

However, consider using `Rgba8Uint` instead of `Rgba8Unorm` to avoid the automatic `[0,1]` normalization in the texture unit. The shader currently loads normalized floats and multiplies by 255.0:

```wgsl
let ref_color = textureLoad(reference_image, vec2<i32>(i32(px), i32(py)), 0);
let refr = ref_color.x * 255.0;
```

With `Rgba8Uint`, the shader would get integer values directly:
```wgsl
let ref_color = textureLoad(reference_image, vec2<i32>(i32(px), i32(py)), 0);
let refr = f32(ref_color.x);
```

### Expected Benefit
- Saves one multiply per channel per pixel (3 multiplies x W x H x K*lambda pixels per iteration)
- Negligible in practice -- the FMA units are not the bottleneck

---

## 8. Buffer Memory Allocation Strategy (Medium Impact)

### Current State
All buffers are created individually via `device.create_buffer()`. Each buffer gets its own Vulkan memory allocation through wgpu's internal allocator (gpu-allocator crate).

### Opportunity: Explicit Memory Placement
wgpu 22.1.0's `MemoryHints::Performance` (already used) tells the allocator to prefer GPU-local memory. No further control is available through the wgpu API.

However, specific buffer usage patterns can be optimized:

**Staging buffers should use `mapped_at_creation: true`**
The `readback_staging_buf` and multi-readback staging buffer in `readback_chains()` are created unmapped, then mapped later. Creating them pre-mapped avoids a map/unmap cycle:

```rust
// For the multi-readback temp buffer in readback_chains():
let staging = p.device.create_buffer(&BufferDescriptor {
    mapped_at_creation: true,  // Pre-map to avoid a round-trip
    ..
});
```

Wait -- this only helps if you write to the buffer from CPU before submitting. For readback (GPU writes, CPU reads), `mapped_at_creation` does not help because the buffer must be unmapped before GPU can write to it, then re-mapped after.

A better optimization for `readback_chains()`: **reuse a persistent staging buffer** instead of creating a new one each call. Buffer creation has non-trivial overhead (Vulkan memory allocation + page table updates).

### Implementation
Add a persistent `multi_readback_staging_buf` to `GpuPipeline`, sized for `max_chain_count * GPU_DRAWING_STATE_SIZE`. Reuse it across calls to `readback_chains()`.

### Expected Benefit
- Eliminates buffer creation/destruction overhead in `readback_chains()` (called every ~2 seconds for island thumbnails)
- Avoids Vulkan `vkAllocateMemory`/`vkFreeMemory` calls and page table management
- Minor but prevents potential memory fragmentation over long runs

---

## 9. Workgroup Size for Mutate Shader (Low-Medium Impact)

### Current State
The mutate shader uses `@workgroup_size(64, 1, 1)` with one workgroup per chain. Threads 0..lambda-1 are active; threads lambda..63 exit early. With the default lambda=8, only 8 of 64 threads are active (12.5% occupancy within the workgroup).

### Opportunity
When lambda is small (1-8), most threads in the workgroup are wasted. Since the mutate shader is purely divergent (each thread follows a different mutation path based on RNG), there is no benefit to having inactive threads in the warp.

Options:
1. **Dynamic workgroup size at pipeline creation**: Create multiple mutate pipelines for different lambda values (1, 2, 4, 8, 16, 32, 64) with matching workgroup sizes. Select the right pipeline at dispatch time.
2. **Use `@workgroup_size(1)` and dispatch `active * lambda` workgroups**: Each thread is a separate workgroup. This wastes no threads but loses any potential for shared memory cooperation in the mutate pass (which is not used anyway -- the mutate shader has no shared memory).

Approach 2 is simpler and eliminates all thread waste:

```rust
// Instead of:
pass.dispatch_workgroups(active, 1, 1);  // 64 threads/workgroup, lambda active

// Use:
pass.dispatch_workgroups(active * lambda, 1, 1);  // 1 thread/workgroup
```

### Expected Benefit
- At lambda=8, eliminates 87.5% wasted thread launches in the mutate pass
- Mutate is typically <10% of total GPU time, so absolute impact is small
- Simpler dispatch logic

### Caveat
Very small workgroups (size 1) can hurt GPU throughput because the hardware schedules work in warp-sized (32-thread) chunks. Having 32 single-thread workgroups is no worse than one 32-thread workgroup with only 1 active thread. But the dispatch overhead per workgroup is higher. Benchmark to confirm.

---

## 10. Async Pipeline Compilation (Low Impact at Steady State)

### Current State
All 5 compute pipelines are created synchronously during `GpuPipeline::new()`. With the pipeline cache, this is fast on subsequent launches, but the first launch compiles all shaders from WGSL -> SPIR-V -> GPU binary.

### Opportunity
wgpu supports `device.create_compute_pipeline_async()` which returns a future. All 5 pipelines could be compiled in parallel:

```rust
let mutate_future = device.create_compute_pipeline_async(&mutate_desc);
let rasterize_future = device.create_compute_pipeline_async(&rasterize_desc);
let select_future = device.create_compute_pipeline_async(&select_desc);
// ... await all
```

### Expected Benefit
- First-launch time reduced by overlapping shader compilation
- Negligible after first launch (pipeline cache handles it)
- Only matters for developer iteration speed, not runtime performance

---

## 11. RTX 5090-Specific Device Limits Tuning

### Current State
```rust
let required_limits = Limits {
    max_storage_buffer_binding_size: max_buffer_size as u32,
    max_buffer_size,
    max_compute_workgroups_per_dimension: 65535,
    max_compute_invocations_per_workgroup: 256,
    max_storage_buffers_per_shader_stage: 6,
    ..Limits::downlevel_defaults()
};
```

### Opportunity
`Limits::downlevel_defaults()` sets conservative values for WebGL2 compatibility. Several limits should be raised for desktop Vulkan:

- **`max_storage_buffer_binding_size`**: RTX 5090 supports up to 2GB (`0x80000000`). The current code already calculates this dynamically, which is correct.
- **`max_compute_work_group_storage_size`**: Not explicitly set. The rasterize_error shader uses 13312 bytes of shared memory. NVIDIA SM 100 supports 99KB, but wgpu defaults to 16384 bytes (16KB). This is sufficient for the current shader but would need to be raised if the polygon tile size increases.
- **`max_compute_invocations_per_workgroup`**: Set to 256, matching the rasterize_error workgroup. NVIDIA supports 1024. If workgroup size experiments (item 3) try larger sizes, this needs to increase.

### Implementation
```rust
let required_limits = Limits {
    max_compute_work_group_storage_size: 32768,  // 32KB, room for larger tiles
    ..current_limits
};
```

---

## 12. Eliminate Per-Polygon Alpha Clamping in Mutate Shader (Low Impact)

### Current State
The mutate shader clamps alpha on **every polygon** during the parent-to-offspring copy:

```wgsl
for (var i = 0u; i < poly_count; i++) {
    var poly = chain_states[chain_id].polygons[i];
    var color = unpack_color(poly);
    color.w = clamp(color.w, params.min_alpha_norm, params.max_alpha_norm);
    poly.data.x = pack_color(color);
    working_states[offspring_id].polygons[i] = poly;
}
```

This unpacks, clamps, and repacks every polygon's color even when alpha is already in range (which it should be, since every mutation path also clamps).

### Opportunity
Replace the unpack-clamp-repack loop with a raw copy, and only clamp alpha in the mutation paths that actually modify it. Since the data is validated on first upload and every mutation clamps, the invariant is maintained.

```wgsl
for (var i = 0u; i < poly_count; i++) {
    working_states[offspring_id].polygons[i] = chain_states[chain_id].polygons[i];
}
```

### Expected Benefit
- Saves 2 unpack + 1 clamp + 1 pack per polygon per offspring per iteration
- With 1000 polygons x 8 lambda x 50 iterations = 400,000 unnecessary unpack-repack operations per batch
- Minor but the mutate shader is per-offspring, so it compounds

---

## 13. Shared Memory Polygon Prefetch Tile Size (Medium Impact)

### Current State
The rasterize_error shader uses a tile size of 768 polygons in shared memory (12288 bytes). With 256 threads, each thread loads 3 polygons per tile pass.

### Opportunity: Adaptive Tile Size
For drawings with fewer polygons (e.g., <200), the tile loop only runs once. The shared memory is allocated but mostly unused, reducing per-SM occupancy for no benefit.

For drawings with many polygons (500-1000), the tile loop runs 1-2 times. The cooperative load pattern is effective here.

Consider dynamically setting the tile size based on polygon count. This could be done at pipeline creation time (specialization constants, not available in WGSL) or by having two shader variants:

- **Small variant**: No tiling, direct reads from storage buffer, smaller workgroup (8x8). Better for <200 polygons.
- **Large variant**: Current tiling approach. Better for 200+ polygons.

### Implementation
Create two rasterize_error pipelines and select at dispatch time:

```rust
if max_polygon_count < 200 {
    pass.set_pipeline(&p.rasterize_error_small_pipeline);
} else {
    pass.set_pipeline(&p.rasterize_error_large_pipeline);
}
```

### Expected Benefit
- 10-20% improvement for low polygon count drawings (early evolution stages)
- No change for high polygon count drawings
- The transition point should be profiled empirically

---

## 14. `queue.write_buffer` vs. Mapped Staging for Params (Low Impact)

### Current State
Two `queue.write_buffer` calls per batch:
```rust
p.queue.write_buffer(&p.params_buf, 0, bytemuck::bytes_of(&params));
p.queue.write_buffer(&p.control_flags_buf, 0, bytemuck::bytes_of(&control_reset));
```

### Opportunity
`queue.write_buffer` internally allocates a staging buffer, copies data, and issues a DMA transfer. For small writes (128 bytes + 16 bytes), this is fine. If push constants are adopted (item 1), the params write is eliminated entirely.

The control flags write (16 bytes) is unavoidable since it resets an SSBO that the GPU reads atomically. This is already optimal.

### Expected Benefit
Minimal. The writes are small and the staging allocator is pooled.

---

## Priority Ranking

| # | Optimization | Impact | Effort | Risk |
|---|-------------|--------|--------|------|
| 1 | Push constants for GpuParams | High | Low | Low |
| 2 | Subgroup ops for error reduction | High | Medium | Low |
| 6 | Fewer compute pass boundaries | Medium | Medium | Medium |
| 3 | Workgroup size tuning (rasterize) | Medium | Medium | Medium |
| 4 | `min_binding_size` validation skip | Low-Med | Low | None |
| 12 | Remove redundant alpha clamping | Low | Low | None |
| 8 | Persistent multi-readback staging | Low-Med | Low | None |
| 13 | Adaptive polygon tile size | Medium | Medium | Low |
| 9 | Mutate workgroup size / dispatch | Low-Med | Low | Low |
| 5 | Pre-encoded command patterns | Medium | High | Medium |
| 11 | RTX 5090 limits tuning | Low | Low | None |
| 10 | Async pipeline compilation | Low | Low | None |
| 7 | Texture format for reference | Low | Low | None |
| 14 | Params write elimination | Low | Low | None |

## Recommended Implementation Order

1. **Push constants** (item 1) -- highest reward/effort ratio, 128-byte struct is a perfect fit
2. **Subgroup reduction** (item 2) -- targets the dominant bottleneck (rasterize_error pass)
3. **Compute pass consolidation** (item 6) -- reduces CPU-side encoding overhead and Vulkan barrier spam
4. **`min_binding_size`** (item 4) + **remove alpha clamping** (item 12) -- quick wins with zero risk
5. **Workgroup size experiments** (items 3, 9, 13) -- requires benchmarking, do after establishing baseline measurements
