# wgpu/Vulkan Performance Analysis

Analysis of the GPU evolution pipeline's wgpu API usage and Vulkan-level performance patterns. Targets RTX 5090 on WSL2 with wgpu v22.1.0 (Vulkan backend).

**Already-completed optimizations (out of scope):**
1. Fused rasterize + error_reduce into single shader
2. Batched 50 iterations per batch to amortize submission overhead
3. Per-polygon AABB early-out in rasterize_error
4. Configurable chain count up to adapter limit

---

## 1. Triple Buffer Map Stalls (Critical)

**File:** `src/gpu_evolver/mod.rs`, lines 322-427

Every `run_batch()` call performs three synchronous buffer maps in sequence:

```rust
// read_control_flags() — line 329
p.device.poll(Maintain::Wait);

// read_timestamps() — line 383
p.device.poll(Maintain::Wait);

// read_chain_fitness() — line 417
p.device.poll(Maintain::Wait);
```

Each `poll(Maintain::Wait)` is a full CPU-GPU synchronization barrier. The first call already ensures all GPU work has completed, making the subsequent two calls redundant. Worse, each `map_async` + `poll(Wait)` + `unmap` cycle has non-trivial Vulkan overhead (VkMapMemory, cache invalidation, VkUnmapMemory).

**Recommendation:** Consolidate all three staging readbacks into a single buffer. Copy control flags (16 bytes) + fitness_packed (K * 4 bytes) + timestamps (64 bytes) into one contiguous staging buffer, then map it once:

```rust
// In pipeline.rs — replace three staging buffers with one:
let combined_staging_size = 16 + (chain_count as u64 * 4) + 64;
let combined_staging_buf = device.create_buffer(&BufferDescriptor {
    label: Some("combined_staging"),
    size: combined_staging_size,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// In mod.rs run_batch() — single copy sequence, single map:
encoder.copy_buffer_to_buffer(&p.control_flags_buf, 0, &p.combined_staging_buf, 0, 16);
encoder.copy_buffer_to_buffer(&p.fitness_packed_buf, 0, &p.combined_staging_buf, 16, fitness_size);
encoder.copy_buffer_to_buffer(&p.timestamp_resolve_buf, 0, &p.combined_staging_buf, 16 + fitness_size, 64);
// ...submit, then single map_async + single poll(Wait) + parse all three regions
```

This eliminates two of three synchronization points per batch (saves ~0.1-0.5ms on WSL2 where Vulkan-on-D3D12 has higher sync overhead).

---

## 2. Double-Buffered Staging for Overlap (High Impact)

**File:** `src/gpu_evolver/mod.rs`, lines 143-320

Currently, the CPU blocks waiting for GPU completion before starting the next batch. With double-buffered staging, batch N+1 can be submitted while batch N's results are still being read back:

```
Current:  [GPU batch N]---[map+read]---[GPU batch N+1]---[map+read]
Proposed: [GPU batch N]---[GPU batch N+1]---[GPU batch N+2]
                    \---[map+read N]---/\---[map+read N+1]---/
```

**Implementation:** Create two staging buffers (A and B). After submitting batch N, map staging buffer from batch N-1 while the GPU is already working on N. Alternate between A and B each batch. This requires tracking which staging buffer was last used and deferring the readback by one batch.

On an RTX 5090 with 50-iteration batches, the GPU compute time is likely 2-5ms and the map+read cycle is 0.1-0.5ms. Double buffering would overlap these entirely, approaching zero CPU-side idle time.

---

## 3. `queue.write_buffer()` Stalls Before Batch (Medium Impact)

**File:** `src/gpu_evolver/mod.rs`, lines 152-166

Three `write_buffer` calls happen before each batch:

```rust
p.queue.write_buffer(&p.params_buf, 0, bytemuck::bytes_of(&params));        // 128 bytes
p.queue.write_buffer(&p.control_flags_buf, 0, bytemuck::bytes_of(&control)); // 16 bytes
p.queue.write_buffer(&p.error_accumulators_buf, 0, &zeros);                  // K * 4 bytes
```

`queue.write_buffer()` in wgpu internally creates a staging buffer, copies data into it, and inserts a copy command. For the error accumulators, this means allocating up to 2048 bytes of staging memory every batch (512 chains * 4). Since the select shader already does `atomicExchange(&error_accumulators[chain_id], 0u)` to reset errors, the explicit CPU-side zero-fill of `error_accumulators` is partially redundant.

**Recommendation:**
- The error accumulator reset via `queue.write_buffer` is needed because `atomicExchange` only resets accumulators that were actually read (chains that ran select). However, since all active chains always run select, this should indeed be handled by `atomicExchange` alone. Verify this is the case and remove the CPU-side zero-fill.
- For `params_buf` (128 bytes) and `control_flags_buf` (16 bytes), push constants would be ideal (see section 7), but these are small enough that `write_buffer` overhead is minimal.

---

## 4. Buffer Usage Flags Over-Provisioning (Low-Medium Impact)

**File:** `src/gpu_evolver/pipeline.rs`, lines 134-217

Several buffers have more `BufferUsages` flags than needed:

| Buffer | Current flags | Actually needed |
|--------|--------------|-----------------|
| `chain_states_buf` | STORAGE \| COPY_SRC \| COPY_DST | STORAGE \| COPY_SRC (COPY_DST only needed for `reinit_chains` — could use a separate init path) |
| `working_states_buf` | STORAGE \| COPY_DST | STORAGE only (the COPY_DST is unused — no `copy_buffer_to_buffer` targets this) |
| `error_accumulators_buf` | STORAGE \| COPY_DST | STORAGE only if CPU zero-fill is removed per section 3 |
| `control_flags_buf` | STORAGE \| COPY_SRC \| COPY_DST | Correct (read back via COPY_SRC, reset via COPY_DST) |

On Vulkan, extra usage flags can prevent the driver from placing buffers in optimal memory pools. A buffer with `COPY_DST` must be visible to the transfer engine, which may force it out of device-local-only memory on some architectures. On an RTX 5090 with large VRAM this is unlikely to matter for placement, but cleaner flags communicate intent better to the driver's internal heuristics.

**Recommendation:** Remove `COPY_DST` from `working_states_buf`. Evaluate whether `reinit_chains` can use a compute shader to copy data from a small upload buffer rather than `write_buffer` directly to `chain_states_buf`.

---

## 5. Readback Chain Creates a Second Command Buffer (Medium Impact)

**File:** `src/gpu_evolver/mod.rs`, lines 339-374

`readback_chain()` creates a new command encoder, submits a single copy command, then blocks on `poll(Wait)`:

```rust
let mut encoder = p.device.create_command_encoder(...);
encoder.copy_buffer_to_buffer(&p.chain_states_buf, src_offset, &p.readback_staging_buf, 0, ...);
p.queue.submit(std::iter::once(encoder.finish()));
// ... map + poll(Wait)
```

This is called from `run_batch()` only when `new_best_found != 0`, so it adds a second full submit+sync cycle on improvement batches. It's also called from `build_gpu_stats()` once per island for the island best drawings (up to 8 extra readbacks every 2 seconds).

**Recommendation:** For the `run_batch()` case, copy the best chain's state to staging as part of the main batch command buffer. Since `best_chain_id` is only known after the select shader runs (GPU-side), this requires either:

1. **Deferred readback:** After reading control flags, if `new_best_found`, include the copy in the *next* batch's command buffer and defer the Drawing conversion by one batch. This is the simplest change and costs zero extra syncs.

2. **GPU-side conditional copy:** Use an indirect dispatch or a simple "copy_best" compute shader that reads `control.best_chain_id` and copies that chain to a dedicated output slot. This eliminates the second submission entirely.

For `build_gpu_stats()`, batch all island-best readbacks into a single command buffer with multiple `copy_buffer_to_buffer` commands, then do one map/poll cycle instead of N.

---

## 6. Compute Pass Timestamp Profiling Overhead (Low Impact)

**File:** `src/gpu_evolver/mod.rs`, lines 176-267

Timestamps are only recorded on the last iteration of the batch (`is_last_iter`), which is good. However, the timestamp query set has 8 slots and is resolved every batch regardless of whether migration ran. If migration didn't run on the last iteration, slots 6-7 contain stale data from a previous batch.

The `read_timestamps` method handles this correctly by checking `migrate_ran`, so there's no correctness issue. However, the `resolve_query_set` for all 8 queries and the copy of 64 bytes could be reduced to only the used queries.

**Recommendation:** Conditional resolve:
```rust
let query_count = if migrate_ran { 8 } else { 6 };
encoder.resolve_query_set(&p.timestamp_query_set, 0..query_count, &p.timestamp_resolve_buf, 0);
```

This is a micro-optimization that saves one or two Vulkan vkCmdCopyQueryPoolResults calls.

---

## 7. Push Constants vs Uniform Buffer (Medium Impact)

**File:** `src/gpu_evolver/pipeline.rs`, lines 411-470

The `GpuParams` struct is 128 bytes, which fits exactly within the Vulkan minimum guaranteed push constant size (128 bytes). Currently it's uploaded via a uniform buffer with `queue.write_buffer()` every batch.

Push constants are embedded directly in the command buffer, eliminating:
- The staging buffer allocation in `write_buffer`
- The internal buffer-to-buffer copy
- A buffer binding slot in every bind group

**Recommendation:**
```rust
// Pipeline layout with push constants:
let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
    label: Some("mutate_layout"),
    bind_group_layouts: &[&mutate_bgl],
    push_constant_ranges: &[PushConstantRange {
        stages: ShaderStages::COMPUTE,
        range: 0..128,
    }],
});

// In command encoding (replaces write_buffer):
pass.set_push_constants(ShaderStages::COMPUTE, 0, bytemuck::bytes_of(&params));
```

In WGSL, replace `@group(0) @binding(N) var<uniform> params: Params` with `var<push_constant> params: Params`. This removes one binding from every bind group layout and eliminates the `params_buf` entirely.

**Caveat:** Vulkan push constants have a minimum guarantee of 128 bytes, but RTX 5090 supports 256 bytes. At exactly 128 bytes, `GpuParams` fits, but verify `adapter.limits().max_push_constant_size >= 128` at init time.

---

## 8. Shader Compilation and Pipeline Caching (Medium Impact)

**File:** `src/gpu_evolver/pipeline.rs`, lines 243-470

All five compute pipelines are created with `cache: None`:

```rust
let mutate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    // ...
    cache: None,
});
```

wgpu v22 supports `PipelineCache` objects that persist compiled shader binaries across runs. Without caching, every application launch recompiles all shaders from WGSL -> SPIR-V -> driver-specific ISA. On an RTX 5090, this compilation happens during `create_compute_pipeline()` and can take 100-500ms per shader, adding 0.5-2.5s to startup.

**Recommendation:**
```rust
// At init, create or load pipeline cache:
let cache = device.create_pipeline_cache(&PipelineCacheDescriptor {
    label: Some("artgen_cache"),
    data: std::fs::read("pipeline_cache.bin").ok().as_deref(),
    fallback: true,
});

// Pass to all pipeline creation calls:
let mutate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    // ...
    cache: Some(&cache),
});

// At shutdown, save:
if let Some(data) = cache.get_data() {
    std::fs::write("pipeline_cache.bin", data).ok();
}
```

This turns 0.5-2.5s startup into ~50ms on subsequent launches.

---

## 9. Reference Image as Texture vs Storage Buffer (Medium-High Impact)

**File:** `src/gpu_evolver/pipeline.rs`, lines 149-157; `src/shaders/rasterize_error.wgsl`

The reference image is stored as a flat `array<u32>` in a storage buffer:

```wgsl
@group(0) @binding(1) var<storage, read> reference_image: array<u32>;
// ...
let ref_idx = py * w + px;
let reference = reference_image[ref_idx];
```

Storage buffer reads go through the L1/L2 cache hierarchy but miss the dedicated texture sampling hardware entirely. The GPU's texture units have:
- **Dedicated texture caches** (separate from L1, often larger for 2D spatial locality)
- **Hardware-accelerated bilinear interpolation** (free, though not needed here)
- **2D spatial tiling** that matches the 16x16 workgroup access pattern perfectly

Each 16x16 workgroup reads a contiguous 16x16 tile of the reference image. With a storage buffer using row-major layout, adjacent threads in the Y direction access data 512 pixels apart (for a 512-wide image), causing poor cache utilization. A texture's internal tiled/swizzled layout keeps 16x16 blocks physically contiguous.

**Recommendation:**
```rust
// Create as texture instead of buffer:
let reference_texture = device.create_texture(&TextureDescriptor {
    label: Some("reference_image"),
    size: Extent3d { width: image_width, height: image_height, depth_or_array_layers: 1 },
    mip_level_count: 1,
    sample_count: 1,
    dimension: TextureDimension::D2,
    format: TextureFormat::Rgba8Unorm,
    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    view_formats: &[],
});

// Upload via queue.write_texture()
queue.write_texture(
    reference_texture.as_image_copy(),
    reference_rgba,
    ImageDataLayout { offset: 0, bytes_per_row: Some(image_width * 4), rows_per_image: None },
    Extent3d { width: image_width, height: image_height, depth_or_array_layers: 1 },
);
```

In the shader:
```wgsl
@group(0) @binding(1) var reference_texture: texture_2d<f32>;

// In main():
let ref_color = textureLoad(reference_texture, vec2<u32>(px, py), 0);
let refr = ref_color.r * 255.0;
let refg = ref_color.g * 255.0;
let refb = ref_color.b * 255.0;
```

This eliminates the manual bit-unpacking (`reference & 0xFF`, shifts) and leverages the texture unit's hardware decompression. The Rgba8Unorm format returns normalized [0,1] floats directly.

On an RTX 5090 with 16384 CUDA cores, the texture cache bandwidth advantage over storage buffer is substantial for 2D access patterns. Expect 10-30% improvement in the rasterize_error pass, which is the dominant cost.

---

## 10. Workgroup Size Tuning (Medium Impact)

### 10a. Rasterize + Error: 16x16 vs 8x8

**File:** `src/shaders/rasterize_error.wgsl`, line 81

The current workgroup size is `@workgroup_size(16, 16, 1)` = 256 threads. This is reasonable but may not be optimal:

- **Shared memory usage:** `shared_polys` = 256 * 48 = 12,288 bytes + `shared_errors` = 256 * 4 = 1,024 bytes = **13,312 bytes total**. On an RTX 5090 SM (48KB shared memory), this allows 3 workgroups per SM, which is good occupancy.
- **Register pressure:** The rasterization loop has moderate register usage (pixel color accumulators, loop variables, polygon fields). With 256 threads per workgroup and 3 workgroups per SM, that's 768 threads per SM using ~65,536 registers (32-bit) = ~85 registers per thread. This is within the 5090's 256 registers per thread limit.

An alternative of `@workgroup_size(8, 8, 1)` = 64 threads would:
- Reduce shared memory to 64 * 48 + 64 * 4 = 3,328 bytes, allowing 14 workgroups per SM
- But require 4x more workgroups total, increasing dispatch overhead
- And reduce the efficiency of the cooperative polygon loading (loading 64 polygons per tile instead of 256, requiring 4x more tiles)

**Recommendation:** 16x16 is likely already optimal for this workload. Profile with both 16x16 and 8x8 using the existing timestamp infrastructure to confirm.

### 10b. Mutate/Select/Migrate: Workgroup Size 1

**File:** `src/shaders/mutate.wgsl`, line 267; `src/shaders/select.wgsl`, lines 79, 163, 180

All three non-rasterization shaders use `@workgroup_size(1)`. With 512 chains, this means 512 workgroups of 1 thread each. On an RTX 5090 SM that can run 2048 threads, this means each SM runs 2048 single-thread workgroups concurrently, but each thread is serialized within its workgroup.

This is actually the correct design for this algorithm since each chain's mutation/selection is independent and sequential. The GPU's warp scheduler fills SMs by running multiple single-thread workgroups simultaneously. However, there's a subtle issue: **single-thread workgroups waste warp lanes**. On NVIDIA, a warp is 32 threads. A workgroup of size 1 occupies one full warp with 31 idle lanes.

With 512 chains, that's 512 warps with 31/32 wasted occupancy = **97% wasted compute**. However, these shaders are memory-bound (copying 48KB drawing states), not compute-bound, so the wasted lanes don't necessarily reduce throughput. The memory pipeline can saturate regardless.

**Recommendation:** For mutate/select/migrate, the algorithmic structure genuinely requires one thread per chain. The wasted warp lanes are unavoidable without a fundamental redesign (e.g., using 32 threads cooperatively per chain to parallelize the polygon copy loops). This is a potential future optimization but requires significant shader refactoring.

---

## 11. Memory Allocation Patterns (Good as-is)

**File:** `src/gpu_evolver/pipeline.rs`, lines 127-238

All buffers are created once in `GpuPipeline::new()` and persist for the lifetime of the application. There are no per-frame or per-batch buffer allocations. This is correct.

The one exception is `reinit_chains()` in `mod.rs` (line 431), which calls `queue.write_buffer()` to upload new chain states. This internally allocates a staging buffer, but wgpu's belt allocator handles this efficiently.

**Assessment:** No issues here.

---

## 12. WSL2-Specific Considerations

### 12a. Vulkan-on-D3D12 Translation Layer

On WSL2, Vulkan calls go through Microsoft's D3D12 translation layer ("Dozen"). This adds overhead to:
- **Buffer mapping:** Each map/unmap involves D3D12 resource state transitions
- **Pipeline barriers:** Implicit barriers between compute passes are translated to D3D12 resource barriers, which may be more conservative than native Vulkan
- **Queue submission:** Each `queue.submit()` translates to D3D12 command list submission

The triple-poll-Wait pattern (section 1) is particularly expensive on WSL2 because each `poll(Wait)` must wait for the D3D12 fence to signal through the WSL2 VM boundary.

### 12b. ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER

**File:** `src/gpu_evolver/pipeline.rs`, line 67

```rust
flags: wgpu::InstanceFlags::default()
    | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
```

This is correctly set for WSL2's Dozen driver. Without it, wgpu may reject the adapter.

### 12c. MemoryHints::Performance

**File:** `src/gpu_evolver/pipeline.rs`, line 118

```rust
memory_hints: MemoryHints::Performance,
```

Good. This hints to wgpu/Vulkan to prefer speed over memory usage.

---

## 13. Device Polling Strategy Optimization (Medium Impact)

**File:** `src/gpu_evolver/mod.rs`

The current pattern is always `device.poll(Maintain::Wait)`, which blocks until all submitted work completes. An alternative is to use `device.poll(Maintain::Poll)` in a spin loop, which returns immediately if work is not yet done.

For this use case, `Maintain::Wait` is correct because the CPU has nothing useful to do while waiting for GPU results. However, combining this with double buffering (section 2) would change the calculus: the CPU could be preparing the next batch's parameters while polling.

**Recommendation:** Implement double-buffered staging first, then switch to `Maintain::Poll` in a loop that alternates between checking GPU completion and doing CPU-side work (parameter updates, WS state updates, etc.).

---

## 14. Bind Group Caching (Good as-is)

**File:** `src/gpu_evolver/pipeline.rs`, lines 472-519

All bind groups are created once during init and reused for every dispatch. This is correct. No bind group is ever recreated.

**Assessment:** No issues here.

---

## 15. Subgroup Operations for Error Reduction (Medium-High Impact)

**File:** `src/shaders/rasterize_error.wgsl`, lines 183-191

The current error reduction uses shared memory with explicit barriers:

```wgsl
shared_errors[local_idx] = pixel_error;
workgroupBarrier();

var stride = 128u;
while stride > 0u {
    if local_idx < stride {
        shared_errors[local_idx] += shared_errors[local_idx + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
}
```

This is 8 iterations with 8 `workgroupBarrier()` calls. On NVIDIA, `workgroupBarrier()` translates to `__syncthreads()`, which is relatively cheap but still forces all warps to reach the barrier before any can proceed.

NVIDIA RTX 5090 supports **subgroup (warp-level) operations** that can replace the lower iterations of the reduction tree. With subgroups of size 32:

```wgsl
// Enable subgroup features
enable subgroups;

// Intra-warp reduction (no barriers needed):
var error = pixel_error;
error = subgroupAdd(error);

// Only one thread per subgroup writes to shared memory
shared_errors[local_idx / 32u] = error; // 8 values for 256 threads
workgroupBarrier();

// Final reduction of 8 values (only thread 0)
if local_idx == 0u {
    var total = 0u;
    for (var i = 0u; i < 8u; i++) {
        total += shared_errors[i];
    }
    atomicAdd(&error_accumulators[chain_id], total);
}
```

This replaces 8 barrier-synchronized iterations with 1 subgroup operation + 1 barrier + 1 serial loop. On an RTX 5090, `subgroupAdd` compiles to a single `redux.sync.add` PTX instruction.

**Caveat:** wgpu subgroup support requires `Features::SUBGROUP` and is not yet stabilized. Check wgpu v22.1.0's subgroup support status. If not available, this is a future optimization.

---

## 16. Specialize Shaders for Common Cases (Low Impact)

**File:** `src/shaders/mutate.wgsl`

The mutate shader has a hot loop (`while !is_dirty && attempts < 1000u`) that iterates until at least one mutation fires. With typical mutation probabilities (1/50 to 1/750 per polygon, ~150 polygons), the expected number of iterations is very small (1-2). The `1000u` cap is a safety net.

No optimization needed here — the shader compiler will handle this well.

---

## Priority-Ranked Summary

| # | Optimization | Impact | Effort | Section |
|---|-------------|--------|--------|---------|
| 1 | Consolidate staging buffer maps | High | Low | 1 |
| 2 | Reference image as texture | Medium-High | Medium | 9 |
| 3 | Push constants for params | Medium | Medium | 7 |
| 4 | Pipeline cache for fast startup | Medium | Low | 8 |
| 5 | Double-buffered staging | High | High | 2 |
| 6 | Batch readback_chain copies | Medium | Low | 5 |
| 7 | Subgroup reduction | Medium-High | Medium | 15 |
| 8 | Remove redundant error accumulator reset | Low-Medium | Low | 3 |
| 9 | Clean up buffer usage flags | Low | Low | 4 |
| 10 | Conditional timestamp resolve | Low | Low | 6 |
