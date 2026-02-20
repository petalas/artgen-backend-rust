# GPU Compute Optimization Analysis

**Target hardware:** NVIDIA RTX 5090 (Blackwell, SM 120, 170 SMs, 21760 CUDA cores, 32 threads/warp)
**Current config:** 512 chains, 50 iterations/batch, 384x384 max image, 1000 max polygons per chain

---

## 1. Mutate Shader: Workgroup Size 1 (Critical)

**File:** `src/shaders/mutate.wgsl`, line 140
**Current:** `@workgroup_size(1)` dispatched as `(chain_count, 1, 1)` = 512 workgroups of 1 thread each

**Problem:** Each SM on the RTX 5090 can host multiple warps (up to 48 warps = 1536 threads at full occupancy). A workgroup of size 1 means every warp is 1/32 utilized -- 31 out of 32 lanes are permanently masked off. With 512 single-thread workgroups across 170 SMs, you get ~3 workgroups per SM, which is 3 active threads out of a potential 1536. That is **0.2% occupancy**.

The mutate shader is inherently serial per chain (it mutates polygons sequentially with data dependencies on the RNG state and polygon array). However, the copy loop at the top (lines 159-163) and the polygon array shifts during add/remove (lines 204-206, 216-218) could be parallelized across threads in a workgroup.

**Recommendation:** This shader's serial RNG-dependent logic makes it hard to parallelize within a chain. The real fix is to increase chain count well beyond 512 (discussed in section 7) so the GPU has enough independent workgroups to saturate. For the mutate pass specifically, grouping multiple chains into one workgroup (e.g., `@workgroup_size(32)` with 32 chains per workgroup) would improve warp utilization but requires restructuring the indexing.

**Expected impact:** Low-to-moderate for mutate alone (it is not the bottleneck), but fixing occupancy here prevents it from becoming one when rasterize/error are optimized.

---

## 2. Rasterize Shader: Polygon Loop Serialization (High Impact)

**File:** `src/shaders/rasterize.wgsl`, lines 92-125
**Current:** `@workgroup_size(8, 8, 1)` = 64 threads, dispatched as `(W/8, H/8, chain_count)`

For a 384x384 image with 512 chains: `48 * 48 * 512 = 1,179,648` workgroups of 64 threads = 75.5M threads total. This gives excellent occupancy.

**Problem:** Each thread loops over ALL polygons (up to 1000) sequentially. With AABB culling already in place, the remaining bottleneck is the **sequential polygon load from global memory**. Each polygon is 48 bytes. Loading 1000 polygons = 48 KB of global memory reads per thread. With 64 threads in a workgroup all reading the same polygon data, this should hit L1 cache well (broadcast pattern), but there is no explicit use of workgroup shared memory to prefetch polygon batches.

**Recommendation: Tiled polygon loading via shared memory.** Have the 64 threads in a workgroup cooperatively load batches of polygons (e.g., 64 at a time) into `var<workgroup>` shared memory, then each thread processes those 64 polygons from shared memory before loading the next batch. This converts 64 independent global memory streams into 1 cooperative load, reducing L1 cache pressure and improving memory bandwidth utilization.

```wgsl
var<workgroup> shared_polys: array<Polygon, 64>;

for (var batch = 0u; batch < poly_count; batch += 64u) {
    // Cooperative load: each thread loads one polygon
    let load_idx = batch + local_idx;
    if load_idx < poly_count {
        shared_polys[local_idx] = working_states[chain_id].polygons[load_idx];
    }
    workgroupBarrier();

    let batch_end = min(64u, poly_count - batch);
    for (var i = 0u; i < batch_end; i++) {
        let poly = shared_polys[i];
        // ... AABB + half-space test + blend ...
    }
    workgroupBarrier();
}
```

**Expected impact:** High. Polygon data is the dominant memory access in this shader. Shared memory prefetching reduces global memory transactions by ~64x for the polygon data stream. For 500+ polygon drawings, this could be a 2-4x speedup on the rasterize pass.

---

## 3. Error Reduce: Subgroup Intrinsics for Faster Reduction (Medium Impact)

**File:** `src/shaders/error_reduce.wgsl`, lines 87-95
**Current:** Binary tree reduction in shared memory with 6 barriers (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1).

**Problem:** Each `workgroupBarrier()` is a full memory fence + execution barrier. On NVIDIA hardware, the first reduction step (64 -> 32) and often the second (32 -> 16) can be done within a single warp using **subgroup operations** (shuffle/reduce), which are barrier-free and execute in a single cycle.

**Recommendation:** Use WGSL subgroup operations (available in WebGPU with the `subgroups` feature). Since RTX 5090 has warp size 32:

```wgsl
// Phase 1: Subgroup reduce (no barrier needed, warp-level)
var val = pixel_error;
val = subgroupAdd(val);  // reduces 32 values within each warp

// Phase 2: Two warps -> shared memory for cross-warp reduction
if subgroupInvocationId == 0u {
    shared_errors[local_idx / 32u] = val;  // 2 partial sums
}
workgroupBarrier();

if local_idx == 0u {
    let total = shared_errors[0] + shared_errors[1];
    atomicAdd(&error_accumulators[chain_id], total);
}
```

This replaces 6 barriers with 1 barrier, and eliminates 5 shared memory round-trips.

**Caveat:** Subgroup operations require `enable subgroups;` in WGSL and the `SUBGROUP` feature on the device. The RTX 5090 supports this, but wgpu feature availability should be checked at runtime.

**Expected impact:** Moderate. Reduces barrier overhead from 6 to 1 per workgroup. With 1.18M workgroups dispatched per error_reduce pass, this eliminates ~5.9M barrier synchronizations per iteration.

---

## 4. Fused Rasterize + Error Pass (High Impact)

**Files:** `src/shaders/rasterize.wgsl` + `src/shaders/error_reduce.wgsl`
**Current:** Two separate dispatches -- rasterize writes packed RGBA u32 to `render_targets`, then error_reduce reads it back alongside the reference image.

**Problem:** The render_targets buffer is `chain_count * W * H * 4` bytes = `512 * 384 * 384 * 4` = 301 MB. Rasterize writes this entire buffer to global memory, then error_reduce reads it all back. This is a 602 MB round-trip through VRAM that exists solely as an intermediate result -- no other pass reads `render_targets`.

**Recommendation:** Fuse rasterize and error_reduce into a single shader. After compositing all polygons at a pixel, immediately compute the error diff against the reference image and accumulate into shared memory, then reduce. This eliminates the render_targets buffer entirely and the 602 MB of memory traffic.

```wgsl
@compute @workgroup_size(8, 8, 1)
fn rasterize_and_error(...) {
    // ... composite polygons (same as current rasterize) ...

    // Compute error inline instead of writing to render_targets
    let ref_pixel = reference_image[py * w + px];
    let refr = f32(ref_pixel & 0xFFu);
    // ... unpack + L1 diff ...
    let pixel_error = u32(dr + dg + db);

    // Workgroup reduction (same as current error_reduce)
    shared_errors[local_idx] = pixel_error;
    workgroupBarrier();
    // ... reduce ...
}
```

**Benefits:**
- Eliminates `render_targets` buffer (301 MB VRAM savings)
- Eliminates 602 MB/iteration of global memory bandwidth
- Removes one full dispatch + implicit barrier between passes
- Computed pixel color stays in registers -- never touches memory

**Expected impact:** High. This is likely the single biggest optimization available. The render_targets buffer bandwidth is the dominant memory cost in the pipeline.

---

## 5. Select/Migrate: Large Struct Copy via Single Thread (Medium Impact)

**File:** `src/shaders/select.wgsl`, lines 100-108 and 141-149
**Current:** `@workgroup_size(1)`, single thread copies up to 1000 polygons (48 KB) in a loop.

**Problem:** Copying a 48 KB `DrawingState` struct one polygon at a time with a single thread is extremely slow for global memory writes. Each polygon write is 48 bytes, so 1000 writes = 48000 bytes of sequential stores from one thread. This serializes memory bandwidth that could be parallelized.

**Recommendation:** Increase workgroup size for select/migrate. Use `@workgroup_size(256)` and have threads cooperatively copy the polygon array when an improvement is found. Broadcast the accept/reject decision via shared memory, then parallelize the copy:

```wgsl
@compute @workgroup_size(256)
fn select_main(...) {
    let chain_id = workgroup_id.x;
    let tid = local_invocation_index;

    // Thread 0 computes fitness + accept/reject
    var do_copy = false;
    if tid == 0u {
        // ... fitness computation ...
        shared_accept = (fitness > current_fitness);
    }
    workgroupBarrier();

    if shared_accept {
        // All 256 threads cooperatively copy polygons
        // Each polygon is 48 bytes = 12 u32s, so copy as u32 array
        // 1000 polygons * 12 words = 12000 words / 256 threads = ~47 words/thread
        for (var i = tid; i < poly_count * 12u; i += 256u) {
            // copy word i from working to chain_states
        }
    }
}
```

**Expected impact:** Moderate. Select runs every iteration and copies are frequent (especially early in evolution when many candidates improve). Parallelizing the copy with 256 threads gives ~256x speedup on the copy portion.

---

## 6. Workgroup Size Tuning for RTX 5090 (Medium Impact)

**Current workgroup sizes:**
- Mutate: 1
- Rasterize: (8, 8, 1) = 64
- Error reduce: (8, 8, 1) = 64
- Select: 1
- Migrate: 1

**RTX 5090 specs:** 32 threads/warp, max 1024 threads/workgroup, max 48 warps/SM (1536 threads/SM).

**Problem with 64-thread workgroups:** 64 = 2 warps. To reach max occupancy of 48 warps/SM, you need 24 concurrent workgroups per SM. Registers and shared memory may limit this. However, 64 is on the small side -- increasing to 128 (4 warps) or 256 (8 warps) can improve instruction-level parallelism within the workgroup and reduce scheduling overhead.

**Recommendation for rasterize/error (or fused pass):** Try `@workgroup_size(16, 16, 1)` = 256 threads. This means dispatching `(W/16, H/16, chain_count)` = `(24, 24, 512)` = 294,912 workgroups. Each workgroup has 8 warps, so you need only 6 concurrent workgroups/SM for full occupancy. Larger workgroups also mean:
- More threads for shared memory polygon prefetching (256 instead of 64)
- Fewer workgroups to schedule (294K vs 1.18M) -- less scheduler overhead
- Better shared memory reduction (256 threads -> still only 8 steps)

**Shared memory needed for 16x16:** If tiled polygon loading is used, `256 * 48 bytes = 12 KB` per workgroup (well within the 64-100 KB/SM limit on Blackwell).

**Expected impact:** 10-30% improvement depending on current register pressure. The reduction in total workgroup count alone saves scheduler overhead.

---

## 7. Error Accumulator Atomics Bottleneck (Medium Impact)

**File:** `src/shaders/error_reduce.wgsl`, line 99
**Current:** Thread 0 of each workgroup does `atomicAdd(&error_accumulators[chain_id], ...)`.

**Problem:** For 512 chains, there are only 512 atomic target addresses. With `(48 * 48) = 2304` workgroups per chain, all 2304 workgroups for the same chain contend on the same atomic u32. Atomic contention on the same cache line serializes and stalls warps.

**Recommendation:** Two-level reduction. Instead of each workgroup atomically adding to a single u32 per chain, use an intermediate buffer of partial sums per workgroup, then run a second small dispatch to sum those partials:

```
// Phase 1: Each workgroup writes its sum to partial_errors[chain_id * num_workgroups + wg_id]
// Phase 2: Small dispatch (one thread per chain) sums the partial_errors for that chain
```

Alternatively, increase the number of accumulator slots per chain (e.g., 4 or 8 slots) and have workgroups hash to different slots, then sum in the select pass. This reduces contention by 4-8x while keeping a single pass.

**Expected impact:** Moderate. Atomic contention is hardware-dependent; NVIDIA L2 atomics are fast, but 2304-way contention per chain is still significant. Reducing to 288-way (8 slots) would help meaningfully.

---

## 8. Batch Size and Command Buffer Overhead (Low-Medium Impact)

**File:** `src/gpu_evolver/mod.rs`, lines 103-171
**Current:** 50 iterations encoded into a single command buffer, each iteration = 4-5 compute passes = ~225 pass dispatches per submission.

**Analysis:** 50 iterations per batch is reasonable. However, after each batch, the CPU does:
1. `queue.submit()` (non-blocking)
2. `copy_buffer_to_buffer` for control flags
3. `device.poll(Maintain::Wait)` -- **blocks CPU until GPU finishes**
4. `map_async` + `recv` -- another CPU stall

This means the CPU is completely idle while the GPU runs, and the GPU is completely idle while the CPU processes the result. There is zero overlap.

**Recommendation: Double-buffering with async polling.** Use two control flag staging buffers. While the GPU runs batch N+1, the CPU reads back results from batch N:

```rust
// Submit batch N+1
encoder_new.copy_buffer_to_buffer(&control_flags_buf, 0, &staging[next], 0, 16);
queue.submit(encoder_new.finish());

// Read back batch N results (already complete)
let flags = read_staging(&staging[current]);
// Process flags...

// Swap buffers
std::mem::swap(&mut current, &mut next);
```

**Expected impact:** Low-medium. If the CPU processing (PNG encoding, WS broadcasting) takes meaningful time, this overlap hides it. The GPU batch is likely the dominant cost, so the absolute gain depends on CPU-side work per batch.

---

## 9. Error Accumulator Reset via CPU Write (Low Impact)

**File:** `src/gpu_evolver/mod.rs`, lines 100-101
**Current:** CPU does `queue.write_buffer()` to zero out `chain_count * 4` bytes of error accumulators before each batch.

**Problem:** `queue.write_buffer` is a host-to-device transfer that may stall the pipeline. For 512 chains this is only 2 KB, but it happens every batch and inserts a pipeline bubble.

**Recommendation:** The `select_main` shader already does `atomicExchange(&error_accumulators[chain_id], 0u)` to reset accumulators. This means the accumulators are already zeroed after select runs. The CPU-side zero write is redundant for iterations 2+ within a batch (the select pass at iteration N resets for iteration N+1). The only issue is the first iteration of a new batch -- but the select pass from the previous batch already zeroed it.

**Action:** Remove the `queue.write_buffer` for error accumulators in `run_batch()`. The `atomicExchange` in `select_main` already handles this. This eliminates one host-to-device transfer per batch.

**Expected impact:** Low. 2 KB transfer is tiny, but removing it simplifies the pipeline and eliminates a potential sync point.

---

## 10. Memory Layout: Struct-of-Arrays vs Array-of-Structs (Speculative, High Effort)

**Current:** `DrawingState` is an Array-of-Structs pattern -- each chain's entire state (48 KB) is contiguous. The `working_states` buffer is `[DrawingState_0, DrawingState_1, ..., DrawingState_511]`.

**Problem for rasterize:** When 64 threads in a workgroup all read polygon `i` from the same chain, they all hit the same 48-byte polygon. This is a broadcast pattern that works OK with L1 cache. However, threads in different workgroups for the same chain also read the same data, creating cache pressure across SMs.

**Observation:** The current AoS layout is actually reasonable for this workload because all threads within a chain read the same polygon data (broadcast), and different chains are fully independent. Converting to SoA would help only if threads within a workgroup accessed different chains (they don't). **No change recommended** for the primary data layout.

---

## Priority Summary

| # | Optimization | Impact | Effort | Dependencies |
|---|-------------|--------|--------|--------------|
| 4 | Fuse rasterize + error_reduce | High | Medium | None |
| 2 | Shared memory polygon prefetch | High | Medium | Works with or without #4 |
| 6 | Workgroup size 256 (16x16) | Medium | Low | Best combined with #2 |
| 5 | Parallel struct copy in select | Medium | Medium | None |
| 3 | Subgroup intrinsics for reduction | Medium | Low | Requires feature check |
| 7 | Multi-slot error accumulators | Medium | Medium | Eliminated if #4 is done |
| 9 | Remove redundant accumulator reset | Low | Trivial | None |
| 8 | Double-buffer control readback | Low-Med | Medium | None |
| 1 | Mutate workgroup size | Low | High | Limited by serial RNG |

**Recommended implementation order:** 9 (trivial), then 4+2+6 together (the big win: fuse passes + shared memory + larger workgroups), then 5, then 3.
