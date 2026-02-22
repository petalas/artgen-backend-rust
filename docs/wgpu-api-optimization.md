# wgpu API & Driver-Level Optimization Analysis

Target: wgpu 28.0.0, Vulkan backend, NVIDIA RTX 5090 (Blackwell SM100, 170 SMs, 32-wide warps, 128KB shared/SM, 96MB L2).

This analysis focuses exclusively on wgpu API usage patterns, driver-level optimizations, and Vulkan-specific opportunities that have NOT already been covered in existing docs (`wgpu-vulkan-performance.md`, `gpu-compute-optimization.md`, `GPU_PERF_IDEAS.md`, `gpu-optimization-todo.md`). Items already documented or implemented are noted as such and skipped.

---

## Already Implemented / Documented (skipped)

These items appear in existing docs and/or are already present in the codebase:

- **Subgroup operations for error reduction** -- Already implemented in `rasterize_error.wgsl` (lines 353-385) using `subgroupAdd()`, `@builtin(subgroup_invocation_id)`, `@builtin(subgroup_size)`. The 1D workgroup layout required by naga is used.
- **Pipeline cache** -- Implemented in `pipeline.rs` (lines 184-199, 936-955). Uses `PipelineCache` with disk persistence keyed by adapter.
- **`min_binding_size` on bind group layout entries** -- Already implemented on all BGL entries (e.g., `pipeline.rs` line 449: `min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64)`).
- **Compute pass consolidation (single pass for bulk iterations)** -- Already implemented in `mod.rs` lines 278-308. All non-timestamped iterations go into ONE compute pass with inline pipeline switches.
- **Double-buffered async staging readback** -- Implemented (`staging_idx`, `PendingBatch`, `start_async_map`, etc.).
- **Cooperative parent copy via shared memory** -- Already implemented in `mutate.wgsl` lines 148-567. Uses `shared_parent_polygons` workgroup memory with cooperative striped loading.
- **Parallel min-reduction in select shader** -- Already implemented in `select.wgsl` lines 147-181. Uses `reduction_err`/`reduction_idx` shared arrays with binary tree reduction.
- **MemoryHints::Performance** -- Already set in device request (`pipeline.rs` line 177).
- **Timestamp queries** -- Fully implemented with double-buffered staging and per-pass profiling.
- **Alpha clamping removal from parent copy** -- Already done. The mutate shader copies from shared memory directly without unpack-clamp-repack (lines 622-624).
- **Tile culling / polygon binning** -- Implemented as a separate `bin_polygons.wgsl` pass.
- **Configurable workgroup sizes** -- Implemented via string replacement at pipeline creation time (lines 967-969).

---

## NEW Optimization Opportunities

### 1. Push Constants for GpuParams (HIGH IMPACT)

**Current state:** GpuParams (256 bytes) is written via `queue.write_buffer()` every batch (`mod.rs` lines 228-231). This creates a staging copy and DMA transfer. Every compute pass reads this through a uniform buffer binding at `@group(1) @binding(0)`.

**Opportunity:** Vulkan push constants allow up to 256 bytes (NVIDIA typically supports 256, Vulkan minimum is 128) to be inlined directly into the command buffer. The RTX 5090 stores push constants in the command processor's dedicated register file, giving effectively zero-latency reads -- no buffer allocation, no descriptor binding, no cache miss.

**However:** The GpuParams struct is 256 bytes, which is exactly the typical NVIDIA push constant limit. The Vulkan minimum guarantee is only 128 bytes. While the RTX 5090 supports 256, this should be checked at runtime via `adapter.limits().max_push_constant_size`.

**Why it matters for this workload:** Within the consolidated bulk compute pass (`mod.rs` lines 279-308), `set_bind_group(1, &p.params_bind_group, &[])` is called 3-4 times per iteration (mutate, optional bin_polygons, rasterize_error, select) x `bulk_count` iterations. At 64 iterations with 4 passes each, that is 256 `set_bind_group` calls for the params bind group alone. Push constants would replace ALL of these with `set_push_constants()` -- which inlines the data into the command stream and avoids any descriptor set management overhead.

**wgpu API:** Requires `Features::PUSH_CONSTANTS`. In WGSL, replace `var<uniform>` with `var<push_constant>`. In Rust, use `PipelineLayoutDescriptor::immediate_size` (which the code already has set to 0 -- see `pipeline.rs` line 684) and `ComputePass::set_push_constants()`.

**Note on `immediate_size`:** The code already has `immediate_size: 0` on all pipeline layouts. This is the wgpu 28.0.0 API for push constants. Changing it to `256` and setting push constants via the compute pass would be a drop-in replacement.

**Code reference:** `pipeline.rs` lines 680-684 (mutate layout), 695-698 (rasterize layout), 707-710 (bin_polygons layout), 718-721 (select layout).

**Estimated impact:** Eliminates one `queue.write_buffer` per batch plus 256+ `set_bind_group` calls per batch. Primary benefit is reduced CPU-side encoding overhead and eliminated UBO read latency on the GPU side. At 64 iterations per batch, this is meaningful. **Medium-high impact on CPU encoding time, low impact on GPU compute time.**

---

### 2. Submission Index Tracking for Targeted Polling (MEDIUM IMPACT)

**Current state:** Every `device.poll()` call uses `submission_index: None`, which waits for ALL pending GPU work to complete:

```rust
// mod.rs line 466
p.device.poll(PollType::Wait { submission_index: None, timeout: None }).unwrap();
```

This is called in `finish_pending_readback`, `readback_chain`, `readback_chains`, `evaluate_chain_fitness`, and `dispatch_init_framebuffers`.

**Opportunity:** `queue.submit()` returns a `SubmissionIndex` that can be passed to `device.poll()` to wait only for that specific submission. This avoids blocking on unrelated GPU work that may have been submitted by other codepaths.

Currently, the double-buffer scheme submits a new batch (step 6 in `run_batch`, line 402) and then issues `map_async` (step 7), but the poll in `finish_pending_readback` waits for ALL submissions -- including the just-submitted new batch. If the new batch takes significantly longer than the map operations from the previous batch, the CPU blocks unnecessarily.

**Implementation:**
```rust
// In run_batch, capture the submission index
let submission_idx = p.queue.submit(std::iter::once(encoder.finish()));

// Store it in PendingBatch
self.pending_batch = Some(PendingBatch {
    staging_idx: write_idx,
    collect_timestamps,
    active_chain_count: active,
    submission_index: submission_idx,
});

// In finish_pending_readback, wait for THAT specific submission
p.device.poll(PollType::Wait {
    submission_index: Some(pending.submission_index),
    timeout: None,
}).unwrap();
```

**Why it matters:** In the current double-buffer scheme, `finish_pending_readback` is called at the start of the NEXT `run_batch`. At that point, two submissions are outstanding: (1) the previous batch whose staging buffers need reading, and (2) a potentially in-progress batch if anything else submitted work. With `submission_index: None`, the CPU waits for everything. With a targeted submission index, it only waits for the specific batch it needs.

**Estimated impact:** In practice, the current architecture only has one or two submissions in flight at a time, so the difference may be small. But it is architecturally correct and prevents future surprises if additional GPU work is submitted between batches (e.g., readback_chain for display, or init_framebuffers). **Low-medium impact, near-zero effort.**

---

### 3. Transient Staging Buffer in `readback_chains` and `evaluate_chain_fitness` (MEDIUM IMPACT)

**Current state:** Both `readback_chains()` (`mod.rs` lines 579-584) and `evaluate_chain_fitness()` (`mod.rs` lines 723-728) create a new staging buffer on every call via `device.create_buffer()`. After reading, the buffer is unmapped and dropped.

```rust
// readback_chains: called for viewer display / benchmark readback
let staging = p.device.create_buffer(&BufferDescriptor {
    label: Some("multi_readback_staging"),
    size: count as u64 * state_size,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

**Problem:** Each `create_buffer` call triggers a Vulkan `vkAllocateMemory` (or sub-allocation from wgpu's gpu-allocator). Each drop triggers `vkFreeMemory` (or sub-allocator return). These are relatively expensive operations that involve kernel-level page table management.

**Opportunity:** Pre-allocate a persistent staging buffer at pipeline creation time, sized for the worst case (`chain_count * GPU_DRAWING_STATE_SIZE`), and reuse it across calls. This eliminates per-call allocation overhead and prevents memory fragmentation over long runs.

For `evaluate_chain_fitness`, the staging buffer size depends on `active` chains. A persistent buffer sized for `chain_count` (max) would cover all cases.

**Implementation:** Add `readback_multi_staging_buf: Buffer` and `eval_staging_buf: Buffer` to `GpuPipeline`, created once in `GpuPipeline::new()`.

**Code reference:** `mod.rs` lines 579-584 (`readback_chains`), lines 723-728 (`evaluate_chain_fitness`).

**Estimated impact:** `readback_chains` is called periodically for viewer display (every few seconds). `evaluate_chain_fitness` is called in `prepare_for_benchmark`. Neither is hot-loop critical, but eliminating allocation churn is good practice. **Low impact on throughput, medium impact on allocation hygiene.**

---

### 4. `queue.write_buffer` Batching via Write-Combined Staging (LOW-MEDIUM IMPACT)

**Current state:** Each `run_batch` call makes 2 separate `queue.write_buffer` calls:

```rust
// mod.rs lines 231-232
p.queue.write_buffer(&p.params_buf, 0, params_bytes);     // 256 bytes

// mod.rs line 241
p.queue.write_buffer(&p.control_flags_buf, 0, bytemuck::bytes_of(&control_reset)); // 16 bytes
```

Each `queue.write_buffer` internally allocates from wgpu's staging belt, copies the data, and records a DMA transfer command. Two separate small writes create two separate DMA operations.

**Opportunity:** If push constants are adopted (item 1), the params_buf write disappears entirely. The control_flags_buf write (16 bytes) is too small to optimize further -- the overhead is in the staging belt allocation, not the transfer. This item becomes irrelevant if push constants are implemented.

**Alternative for non-push-constant path:** Combine both writes into a single staging buffer operation using `queue.write_buffer_with()` (if available in wgpu 28) or by mapping a single staging buffer at creation and copying both payloads.

**Estimated impact:** Negligible. The writes are tiny and wgpu's staging belt is efficient for small writes. **Low impact.**

---

### 5. Vulkan Descriptor Indexing / Bindless for Tile Culling Buffers (LOW IMPACT, EXPLORATORY)

**Current state:** Tile culling uses two large storage buffers (`tile_data_buf`, `tile_counts_buf`) bound as entire buffers:

```rust
// pipeline.rs lines 819-820
BindGroupEntry { binding: 1, resource: tile_data_buf.as_entire_binding() },
BindGroupEntry { binding: 2, resource: tile_counts_buf.as_entire_binding() },
```

These buffers can be very large (up to SSBO limit) because they're sized for `offspring_capacity * max_num_tiles * TILE_MAX_POLYS`.

**Observation:** The current approach is correct for wgpu's API model. Vulkan descriptor indexing (`VK_EXT_descriptor_indexing`) would allow accessing these as arrays of buffer descriptors, but wgpu does not expose this for storage buffers in a useful way for this pattern. The data is already efficiently accessed via manual indexing in the shader.

**Conclusion:** No change recommended. The current approach of large SSBOs with manual indexing is optimal for this use case.

---

### 6. SPIR-V Pre-Compilation to Avoid naga Overhead (LOW-MEDIUM IMPACT)

**Current state:** All shaders are compiled from WGSL source via `include_str!` at pipeline creation time. Naga's WGSL frontend parses the source, validates it, and generates SPIR-V for the Vulkan backend. For the rasterize_error, bin_polygons, and init_framebuffers shaders, this happens every time the workgroup size changes (`set_rasterize_wg`, `pipeline.rs` lines 900-933), because string replacement generates new WGSL source.

```rust
// pipeline.rs lines 967-974
let source = include_str!("../shaders/rasterize_error.wgsl")
    .replace("const WG_X: u32 = 16;", &format!("const WG_X: u32 = {};", wg[0]))
    .replace("const WG_Y: u32 = 16;", &format!("const WG_Y: u32 = {};", wg[1]));

let shader = device.create_shader_module(ShaderModuleDescriptor {
    source: ShaderSource::Wgsl(source.into()),
    ..
});
```

**Opportunity:** wgpu supports `ShaderSource::SpirV` for pre-compiled SPIR-V. You could:
1. Pre-compile the shader variants at build time using `naga-cli` or `naga` as a build dependency
2. Ship SPIR-V binaries for common workgroup sizes (8x8, 16x8, 16x16, 32x16)
3. Fall back to WGSL+naga for uncommon sizes

**However:** The pipeline cache (`PipelineCache`) already caches the compiled GPU binary across runs. The naga compilation overhead is only paid on first launch or when the workgroup size changes. Since workgroup size changes are rare (user-triggered via UI), the compile latency is not a hot-path concern.

**Additionally:** wgpu 28 flags `ShaderSource::SpirV` as unsafe because it bypasses naga validation. This is acceptable for this project but adds maintenance burden (must regenerate SPIR-V when shaders change).

**Estimated impact:** First-launch time reduced by ~100-500ms (naga compilation for 5 shaders). Negligible for runtime performance. **Low impact.**

---

### 7. Requesting Optimal RTX 5090 Device Features and Limits (LOW IMPACT, CORRECTNESS)

**Current state:**

```rust
// pipeline.rs lines 152-160
let required_limits = Limits {
    max_storage_buffer_binding_size: max_buffer_size as u32,
    max_buffer_size,
    max_compute_workgroups_per_dimension: 65535,
    max_compute_invocations_per_workgroup: 512,
    max_compute_workgroup_size_x: 512,
    max_storage_buffers_per_shader_stage: 7,
    ..Limits::downlevel_defaults()
};
```

The base is `Limits::downlevel_defaults()`, which is designed for WebGL2 compatibility and sets many limits conservatively. Notable suboptimal defaults:

| Limit | `downlevel_defaults()` | RTX 5090 actual | Used by this project |
|-------|----------------------|-----------------|---------------------|
| `max_uniform_buffer_binding_size` | 16384 | 65536 | 256 (GpuParams) |
| `max_compute_work_group_storage_size` | 16384 | 99152+ | ~25KB (rasterize at 32x16) |
| `max_uniform_buffers_per_shader_stage` | 11 | 15+ | 1 |
| `min_storage_buffer_offset_alignment` | 256 | 16 (NVIDIA) | N/A (no dynamic offsets) |

**Concern:** The `max_compute_work_group_storage_size` of 16384 from `downlevel_defaults()` could be a problem. The rasterize_error shader at 32x16 workgroup size uses:
- `shared_polys: array<Polygon, 1536>` = 1536 * 16 = 24,576 bytes
- `shared_errors: array<u32, 128>` = 512 bytes
- `shared_errors_old: array<u32, 128>` = 512 bytes
- `shared_skip_tile: u32` + dirty bbox vars = ~20 bytes
- **Total: ~25,620 bytes**

This exceeds the 16,384 byte limit from `downlevel_defaults()`. If wgpu validates this limit during pipeline creation, it would fail. The fact that it works suggests either (a) wgpu doesn't enforce this limit for compute shaders, (b) the adapter's reported limit is automatically used, or (c) naga calculates the actual used amount rather than the declared array size.

**Recommendation:** Explicitly request a higher `max_compute_work_group_storage_size`:

```rust
let required_limits = Limits {
    max_compute_work_group_storage_size: 49152,  // 48KB, half of SM's 128KB
    ..current
};
```

This ensures correctness and also unlocks future shader variants that use more shared memory.

**Estimated impact:** Correctness improvement. May also prevent future breakage if wgpu tightens limit validation. **Low impact on performance, medium importance for robustness.**

---

### 8. Error Handling on `device.poll()` Timeout (LOW IMPACT, ROBUSTNESS)

**Current state:** All `device.poll()` calls use `timeout: None` (infinite wait) and unwrap the result:

```rust
p.device.poll(PollType::Wait { submission_index: None, timeout: None }).unwrap();
```

**Concern:** If the GPU hangs (driver TDR, kernel-level timeout), this will panic. On Windows/WSL2, the OS may kill the GPU context after ~2 seconds of unresponsiveness (TDR timeout), and the `unwrap()` will panic with an unhelpful error.

**Recommendation:** Use a finite timeout (e.g., 10 seconds) and handle timeout gracefully:

```rust
match p.device.poll(PollType::Wait {
    submission_index: None,
    timeout: Some(std::time::Duration::from_secs(10)),
}) {
    Ok(wgpu::PollStatus::Complete) => {},
    Ok(wgpu::PollStatus::SubmissionQueueEmpty) => {},
    Err(e) => {
        eprintln!("GPU poll error: {:?}", e);
        // Handle gracefully: skip readback, reset state, etc.
    }
}
```

**Estimated impact:** No performance impact. Improves resilience against GPU hangs. **Low priority but good practice.**

---

### 9. Redundant `set_bind_group` Calls Within Consolidated Compute Pass (MEDIUM IMPACT)

**Current state:** Within the bulk compute pass (`mod.rs` lines 279-308), every iteration re-sets all bind groups:

```rust
for _ in 0..bulk_count {
    pass.set_pipeline(&p.mutate_pipeline);
    pass.set_bind_group(0, &p.mutate_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(active, 1, 1);

    // ... bin_polygons ...

    pass.set_pipeline(&p.rasterize_error_pipeline);
    pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(wg_x, wg_y, active * lambda);

    pass.set_pipeline(&p.select_pipeline);
    pass.set_bind_group(0, &p.select_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(active, 1, 1);
}
```

The `params_bind_group` at group(1) is the SAME bind group for all pipelines and never changes within the batch. It is re-set 3-4 times per iteration x 64 iterations = 192-256 redundant calls.

**Opportunity:** wgpu tracks bind group state per slot. If a `set_bind_group` call sets the same bind group that is already bound at that slot, wgpu may short-circuit internally. However, checking the wgpu source, the bind group is always recorded in the command encoder regardless of whether it changed. Each call generates a command in the internal command list.

The params bind group at slot 1 is shared across all pipeline layouts (all layouts include `&params_bgl` at index 1). When switching from mutate to rasterize_error, the bind group at slot 1 should be preserved by Vulkan (bind groups are orthogonal to pipeline binds as long as the layout is compatible).

**However:** wgpu's current implementation may not take advantage of Vulkan's bind group compatibility rules. When `set_pipeline` is called with a different pipeline, wgpu may invalidate all bind groups (conservative behavior). This means the `set_bind_group(1, ...)` calls are necessary from wgpu's perspective, even though Vulkan wouldn't require them.

**If push constants are adopted (item 1):** The params bind group disappears entirely, eliminating all 256 redundant `set_bind_group(1, ...)` calls. This is the cleanest solution.

**Estimated impact:** Without push constants: wgpu internal overhead of ~256 extra set_bind_group commands per batch. With push constants: eliminated entirely. **Medium impact on CPU encoding time.**

---

### 10. Async Pipeline Compilation for Workgroup Size Changes (LOW IMPACT)

**Current state:** `set_rasterize_wg()` (`pipeline.rs` lines 900-933) synchronously creates 3 new pipelines (rasterize_error, bin_polygons, init_framebuffers) when the user changes workgroup size. This blocks the evolution loop during compilation.

```rust
pub fn set_rasterize_wg(&mut self, wg: [u32; 2]) {
    if wg == self.rasterize_wg { return; }
    // Synchronous: blocks until shader compiles and pipeline is ready
    let (pipeline, shader) = create_rasterize_pipeline(...);
    self.rasterize_error_pipeline = pipeline;
    // ... etc
}
```

**Opportunity:** Use `device.create_compute_pipeline_async()` for the pipeline recreation. Continue using the old pipeline for one more batch while the new one compiles. Swap when ready.

**Implementation complexity:** Moderate. Requires storing an `Option<Future<ComputePipeline>>` and checking completion at the start of each batch.

**Estimated impact:** Eliminates a ~10-50ms stall when the user changes workgroup size via UI. Only matters for interactive use, not benchmarks or headless mode. **Low impact.**

---

### 11. `mapped_at_creation: true` for One-Time Upload Buffers (NEGLIGIBLE IMPACT)

**Current state:** `chain_states_buf` is created via `create_buffer_init` which internally creates the buffer and issues a write. The error_accumulators buffer is similar.

**Observation:** `create_buffer_init` is already optimal -- it uses `mapped_at_creation: true` internally and unmaps after the copy. No improvement possible here.

For staging buffers (control, fitness, timestamp), `mapped_at_creation: false` is correct because these are GPU-write, CPU-read buffers that must be unmapped for GPU writes.

**Conclusion:** No change recommended.

---

## Summary: Priority-Ranked Recommendations

| # | Optimization | Impact | Effort | Risk |
|---|-------------|--------|--------|------|
| 1 | Push constants for GpuParams (256B) | High | Medium | Low |
| 2 | Submission index tracking for targeted polling | Medium | Low | None |
| 3 | Persistent staging buffers (eliminate per-call allocation) | Low-Med | Low | None |
| 7 | Request `max_compute_work_group_storage_size` explicitly | Low | Low | None |
| 9 | Eliminate redundant `set_bind_group(1, ...)` (via push constants) | Medium | Included in #1 | None |
| 8 | Finite timeout on `device.poll()` | Low | Low | None |
| 6 | SPIR-V pre-compilation | Low | Medium | Low |
| 4 | `queue.write_buffer` batching | Negligible | Low | None |
| 5 | Descriptor indexing for tile buffers | None | N/A | N/A |
| 10 | Async pipeline recompilation | Low | Medium | Low |

## Recommended Implementation Order

1. **Push constants (item 1 + 9)** -- Biggest single win. Eliminates `params_buf` write, removes `params_bind_group` and ~256 `set_bind_group(1, ...)` calls per batch, and provides zero-latency GPU access to params. The `immediate_size` field already exists in the pipeline layout descriptors (currently 0).

2. **Submission index tracking (item 2)** -- Trivial change: capture return value of `queue.submit()`, store in `PendingBatch`, use in `device.poll()`. Prevents unnecessary blocking.

3. **Persistent staging buffers (item 3)** -- Simple refactor: move buffer creation from `readback_chains`/`evaluate_chain_fitness` into `GpuPipeline::new()`.

4. **Explicit shared memory limit (item 7)** -- One-line change to `required_limits`. Prevents potential future breakage.

5. **Robust poll timeout (item 8)** -- Replace `unwrap()` with graceful error handling.
