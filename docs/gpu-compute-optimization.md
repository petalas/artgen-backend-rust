# GPU Compute Optimization Analysis

Analysis of the artgen GPU evolution pipeline for compute performance bottlenecks and optimization opportunities. Focused on wgpu compute shaders running on NVIDIA RTX-class hardware (Vulkan backend via WSL2).

**Pipeline overview:** mutate (1 thread/chain) -> rasterize_error (256 threads/tile/chain) -> select (1 thread/chain) -> migrate (1 thread/chain, periodic)

---

## 1. Mutate Shader: Single-Threaded Bottleneck

**File:** `src/shaders/mutate.wgsl`, line 267
**Current:** `@workgroup_size(1)` -- one thread per chain, dispatched as `(active_chains, 1, 1)`

### Problem: Zero parallelism within each chain

The mutate shader runs a single thread per chain. Each thread:
1. Copies up to 1000 polygons (48 bytes each = 48KB) from `chain_states` to `working_states`
2. Loops through all polygons applying per-polygon mutations
3. May loop up to 1000 retry attempts in the `while !is_dirty` loop

With 512 chains, that is 512 workgroups of size 1. On an RTX GPU with 128 SMs, this means each SM gets ~4 workgroups, but each workgroup is a single thread -- the SM's 32-wide warp executes 1 active thread and 31 idle lanes. **Occupancy is effectively 1/32 = ~3%** within each warp.

### Optimization A: Parallelize the polygon copy with a workgroup

The initial copy of polygons (lines 318-322) and the crossover paths (lines 192-265) iterate over up to 1000 polygons sequentially. This could be parallelized:

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let chain_id = gid.x / 64u;  // or use workgroup_id
    // Thread lid copies polygons lid, lid+64, lid+128, ...
    let poly_count = chain_states[chain_id].polygon_count;
    for (var i = lid; i < poly_count; i += 64u) {
        working_states[chain_id].polygons[i] = chain_states[chain_id].polygons[i];
    }
    workgroupBarrier();
    // Thread 0 does mutation logic
    if lid == 0u {
        // ... existing mutation code ...
    }
}
```

**Expected impact:** The copy phase (48KB per chain) is a significant portion of mutate time. Parallelizing it 64x would reduce copy latency from ~48KB/thread to ~750B/thread. The mutation logic itself is inherently serial (sequential RNG), so only the copy benefits.

**Alternative approach:** Skip the copy entirely. Instead of copying `chain_states -> working_states` and then mutating in-place, mutate directly from `chain_states` into `working_states` by reading the source polygon, mutating it in registers, and writing to the destination. This eliminates the copy pass entirely and avoids the need for a barrier. The current code already almost does this -- it reads from `chain_states` and writes to `working_states` -- but it does a bulk copy first and then mutates in the `working_states` buffer. Restructuring to read-mutate-write in a single pass would halve memory bandwidth.

### Optimization B: Eliminate the `while !is_dirty` retry loop

Lines 328-517: The shader retries mutations until at least one fires. With typical per-polygon probabilities around 1/100 to 1/750, and ~150 active polygons, the expected number of attempts before `is_dirty` is set is very low (usually 1). However, in worst-case scenarios (very few polygons, low probabilities), this loop could spin hundreds of times, causing massive warp divergence as some chains finish in 1 iteration while others take 100+.

**Suggestion:** Guarantee at least one mutation fires by always applying one forced mutation (e.g., micro-adjust a random polygon) if nothing fired after the first attempt. This bounds the loop to exactly 1 or 2 iterations and eliminates tail-latency divergence.

### Problem: Scattered memory access pattern in crossover

The crossover functions (`crossover_spatial`, `crossover_uniform`) read from `chain_states[parent_b]` where `parent_b` is a randomly selected chain via tournament selection. Since each thread in a warp picks a different `parent_b`, reads from `chain_states[parent_b].polygons[i]` are scattered across a huge buffer (~24MB for 512 chains). This defeats any L2 cache locality.

**Mitigation is difficult** since parent selection is inherently random, but the impact is bounded by the crossover probability (default 10%).

---

## 2. Rasterize+Error Shader: Well-Structured but Tunable

**File:** `src/shaders/rasterize_error.wgsl`, line 81
**Current:** `@workgroup_size(16, 16, 1)` = 256 threads, dispatched as `((W+15)/16, (H+15)/16, active_chains)`

### Problem: Workgroup size may be suboptimal for modern NVIDIA GPUs

256 threads/workgroup is reasonable, but NVIDIA RTX GPUs execute in warps of 32. With 256 threads that is 8 warps per workgroup. The SM can hold multiple workgroups concurrently (up to register/shared memory limits).

**Shared memory usage:** `shared_polys` = 256 * 48 = 12,288 bytes + `shared_errors` = 256 * 4 = 1,024 bytes = **13,312 bytes total**. NVIDIA SMs have 48-100KB of shared memory (depending on configuration). At 13KB per workgroup, the SM could theoretically hold 3-7 concurrent workgroups, which is good for latency hiding.

**Suggested experiment:** Try `@workgroup_size(8, 8, 1)` = 64 threads with proportionally more workgroups. This would:
- Reduce shared memory per workgroup to ~3.3KB (allowing more concurrent workgroups)
- Reduce polygon tile size from 256 to 64, requiring more tiles but better fitting small polygon counts
- Reduce the warp count per workgroup to 2, potentially improving scheduling flexibility

For a 256x256 image with 16x16 workgroups: 16*16*512 = 131,072 workgroups. With 8x8: 32*32*512 = 524,288 workgroups. Both are far more than enough to saturate the GPU. The key tradeoff is shared memory tile size vs. number of passes over polygons.

### Problem: Redundant workgroupBarrier in reduction

Lines 183-191: The binary reduction has a `workgroupBarrier()` inside the loop for every stride level. On NVIDIA hardware, once the stride drops below 32 (one warp), the barrier is unnecessary because warp threads execute in lockstep. In WGSL, you cannot use subgroup operations without the `subgroups` extension, but you could unroll the last 5 iterations (stride 16, 8, 4, 2, 1) and rely on `workgroupBarrier()` being a no-op for single-warp operations. In practice, naga/SPIR-V compilation likely already handles this, but it is worth verifying.

**Better approach with subgroup operations (wgpu `subgroups` feature, experimental):**
```wgsl
// If subgroup support available:
let subgroup_sum = subgroupAdd(pixel_error);
if subgroupElect() {
    shared_errors[subgroup_id] = subgroup_sum;
}
workgroupBarrier();
// Then reduce only across subgroups (8 entries instead of 256)
```
This would reduce the reduction from 8 barrier+add steps to 1 subgroup intrinsic + 3 barrier+add steps.

### Observation: AABB early-out causes warp divergence

Lines 126-133: The AABB check causes threads within the same warp to diverge -- some threads skip the polygon (AABB miss) while others proceed to the half-space test and blend. This is inherent to the algorithm and already optimized (AABB is cheap), but it means the effective throughput of the inner loop is less than the peak.

For drawings with many small polygons (the common case at high fitness), most threads in a tile will skip most polygons, making the AABB check very effective. The divergence cost is minor compared to the bandwidth savings.

### Problem: Reference image read pattern

Line 166: `reference_image[py * w + px]` -- each thread reads one pixel. Within a 16x16 workgroup, threads in the same warp read 32 consecutive x-coordinates (since the workgroup is laid out x-first). The reference image buffer is in storage (SSBO), not a texture. This means reads go through L2 cache but do not benefit from texture cache spatial locality optimizations. However, since warps read contiguous addresses (32 consecutive u32 values = 128 bytes = one cache line), **the access pattern is actually coalesced** and efficient.

**Suggestion (low priority):** If wgpu ever exposes read-only storage textures for compute, switching the reference image to a `texture_2d<f32>` with a sampler would engage the texture cache's 2D spatial locality, benefiting the Y-direction neighbors. But for the current linear access pattern, the SSBO is fine.

---

## 3. Select Shader: Memory Bandwidth Dominated

**File:** `src/shaders/select.wgsl`, line 79
**Current:** `@workgroup_size(1)`, dispatched as `(active_chains, 1, 1)`

### Problem: Single-threaded full drawing copy on acceptance

Lines 119-121: When a candidate is accepted (`fitness > current_fitness`), the shader copies the entire working state to chain state:
```wgsl
for (var i = 0u; i < pc; i++) {
    chain_states[chain_id].polygons[i] = working_states[chain_id].polygons[i];
}
```

With up to 1000 polygons at 48 bytes each, this is **48KB of sequential memory copies per accepted chain**, executed by a single thread. At typical acceptance rates (maybe 1-5% of chains improve per iteration), this affects a small fraction of chains, but the affected chains stall their entire warp.

### Optimization: Parallelize select with a workgroup

Same pattern as mutate -- use a workgroup of 64 threads and have all threads participate in the copy:

```wgsl
@compute @workgroup_size(64)
fn select_main(...) {
    let chain_id = workgroup_id.x;
    let lid = local_invocation_index;

    // Thread 0 computes fitness and decides accept/reject
    // (store decision in shared memory)
    // All threads copy in parallel if accepted
    if accepted {
        for (var i = lid; i < pc; i += 64u) {
            chain_states[chain_id].polygons[i] = working_states[chain_id].polygons[i];
        }
    }
}
```

**Expected impact:** Reduces per-chain copy latency by 64x when acceptance occurs. Since acceptance is the slow path, this directly reduces tail latency.

### Problem: atomicMax contention on control.best_fitness_bits

Lines 130-135: All 512 chains race on `atomicMax(&control.best_fitness_bits, ...)`. On NVIDIA hardware, atomic operations on the same address serialize through the L2 cache. With 512 threads, this creates a serialization bottleneck. However, since this is a single atomic per chain (not per pixel), the total contention is bounded: ~512 atomic operations, each taking ~30 clock cycles = ~15,360 cycles = negligible at GPU clock speeds.

**Verdict:** Not a meaningful bottleneck.

---

## 4. Migrate Shader: Same Single-Thread Issue

**File:** `src/shaders/select.wgsl`, lines 163, 180
**Current:** `@workgroup_size(1)`, dispatched as `(active_chains, 1, 1)`

### Problem: Full drawing copy in single thread

`migrate_from()` (lines 139-160) copies up to 1000 polygons sequentially when the neighbor is fitter. Same fix as select: use a workgroup for parallel copy.

### Problem: Race condition in ring migration

In intra-island ring migration, chain `i` reads from chain `i+1`, and chain `i+1` reads from chain `i+2`, etc. Since all chains execute concurrently, chain `i+1` might have already been overwritten by chain `i+2`'s data before chain `i` reads it. This is a classic read-after-write hazard in concurrent ring migration.

**However:** In practice, wgpu compute dispatches with `@workgroup_size(1)` and different `global_invocation_id` values have no ordering guarantees, meaning the read/write order is undefined. The current implementation "works" because the writes and reads happen to different chains, and GPU memory model provides eventual coherence within a dispatch. But the correctness depends on the assumption that chain `i`'s read of `chain_states[i+1]` sees the pre-migration state, not a partially-written state from another thread's migration.

**Recommendation:** This is likely safe on current hardware due to SM-level cache coherence, but for correctness, consider double-buffering: migrate into `working_states` as a staging area, then copy back to `chain_states` in a second pass (or swap buffer roles).

---

## 5. Host-Side Dispatch Overhead

**File:** `src/gpu_evolver/mod.rs`, lines 179-267

### Problem: Separate compute passes per iteration within a batch

Each iteration within the batch creates 3-4 separate compute passes (mutate, rasterize_error, select, optionally migrate). With 50 iterations per batch, that is 150-200 compute passes encoded into a single command buffer. Each compute pass has:
- Begin/end pass overhead
- Implicit pipeline barriers between passes (the GPU must ensure all writes from pass N are visible to reads in pass N+1)

**Current behavior is correct:** The barriers between passes ARE necessary because each pass reads the output of the previous pass. The `begin_compute_pass`/`end_compute_pass` boundaries create the necessary `STORAGE_BUFFER -> STORAGE_BUFFER` barriers in Vulkan.

**Suggestion:** Verify that the wgpu/naga compilation does not insert overly conservative full-pipeline barriers. A targeted `storageBarrier()` within a single compute pass would be more efficient than separate passes, but WGSL `storageBarrier()` only synchronizes within a workgroup, not globally. Cross-workgroup synchronization requires separate dispatches (which is what the current separate passes provide).

**This is a fundamental architectural constraint** -- there is no way to do a global memory barrier within a single compute pass in the WebGPU/Vulkan model without separate dispatches.

### Optimization: Reduce error accumulator reset overhead

Lines 165-166: Before each batch, the CPU uploads `active * 4` bytes of zeros to reset the error accumulators:
```rust
let zeros = vec![0u8; active as usize * 4];
p.queue.write_buffer(&p.error_accumulators_buf, 0, &zeros);
```

This is a CPU-side allocation + DMA transfer for every batch (every ~50 iterations). The select shader already resets the error accumulator via `atomicExchange` (line 88 of select.wgsl), so the accumulators should already be zero after the first iteration. **The initial reset before the batch is only needed for the first iteration's rasterize_error pass.**

**Suggestion:** Move the error accumulator reset into the select shader unconditionally (already done via `atomicExchange`), and remove the CPU-side `write_buffer` call. The only issue is the very first iteration of the batch, where the accumulators contain stale data from the previous batch. Since `atomicExchange` in select already resets them, and the rasterize_error pass of iteration 0 writes fresh data via `atomicAdd` starting from the stale value -- this is a bug if the previous batch's select didn't run for all chains. **Actually, looking more carefully:** the `atomicExchange` in select returns the accumulated error and resets to 0. So after select, the accumulators are 0. The next iteration's rasterize_error pass `atomicAdd`s from 0. This is correct. The CPU-side `write_buffer` is redundant if the previous batch completed successfully. It is only necessary for the very first batch (when accumulators contain garbage from buffer creation).

**Fix:** Initialize the error accumulator buffer with zeros at creation (already done -- `BufferDescriptor` without `mapped_at_creation` and no `init` means zeroed on some backends but not guaranteed). Add `mapped_at_creation: true` and explicitly zero it, or use `create_buffer_init` with zeros. Then remove the per-batch `write_buffer` call.

**Impact:** Eliminates one DMA transfer per batch (~2KB for 512 chains). Minor.

---

## 6. Timestamp Query Overhead

**File:** `src/gpu_evolver/mod.rs`, lines 180, 270-277

### Observation: Timestamps only on last iteration

Lines 180: `let ts = if is_last_iter(i) { Some(&p.timestamp_query_set) } else { None };`

This is already well-optimized -- timestamps only on the final iteration of each batch. The overhead of `resolve_query_set` + two `copy_buffer_to_buffer` calls per batch is negligible.

---

## 7. Buffer Layout and Memory Access Patterns

### Problem: Array-of-Structures layout for chain states

`chain_states` is an array of `DrawingState`, where each `DrawingState` is 48,032 bytes. This means chain 0's data is at offset 0, chain 1 at 48,032, chain 2 at 96,064, etc. When the rasterize_error shader reads `working_states[chain_id].polygons[i]`, all 256 threads in a workgroup read from the same chain (same `chain_id`), which is good for locality. The cooperative load into shared memory (`shared_polys[local_idx] = working_states[chain_id].polygons[load_idx]`) reads 256 consecutive polygons = 12,288 bytes, which is contiguous in memory and results in **coalesced reads across all 8 warps**.

**Verdict:** The AoS layout is actually correct for this workload because the rasterize_error shader processes one chain per workgroup, so all threads in the workgroup read from the same chain's contiguous polygon array. A Structure-of-Arrays layout would hurt here.

### Problem: GpuPolygon padding waste

Each `GpuPolygon` is 48 bytes but only uses 40 bytes of data (color + 3 vertices). The 8-byte `_pad` field wastes 16.7% of bandwidth when reading polygon arrays. With 1000 polygons per chain and 512 chains, that is `1000 * 8 * 512 = 4MB` of wasted buffer space and proportional wasted bandwidth.

**However:** The 48-byte stride aligns to 16-byte boundaries (vec4), which is required by WGSL struct alignment rules. Removing the padding would require restructuring the polygon into a different layout (e.g., SoA for polygon fields), which adds complexity. **The 16.7% waste is the cost of correct alignment.**

**Alternative layout (speculative):** Store polygons as separate arrays of `vec4<f32>` (color), `vec2<f32>` (v0), `vec2<f32>` (v1), `vec2<f32>` (v2). This SoA layout eliminates padding and enables better vectorized loads, but it complicates indexing and the cooperative shared memory load pattern. Not recommended without profiling evidence that the rasterize_error shader is bandwidth-bound.

---

## 8. Dispatch Dimension Analysis

### Rasterize+Error dispatch

For a 256x256 image: `(256/16, 256/16, 512) = (16, 16, 512)` = 131,072 workgroups of 256 threads = ~33.5M thread invocations per iteration.

For a 512x512 image: `(32, 32, 512)` = 524,288 workgroups = ~134M thread invocations.

On an RTX 5090 with 170 SMs, each SM can concurrently execute multiple workgroups. With 256 threads/workgroup and 13KB shared memory, an SM can hold ~3 workgroups (limited by shared memory at 48KB default config). That means ~510 concurrent workgroups, processing 131,072 total = ~257 waves for the 256x256 case. This is well-saturated.

### Mutate/Select/Migrate dispatch

512 workgroups of 1 thread each. On 170 SMs, that is ~3 workgroups per SM, but each workgroup is 1 thread (1/32 warp utilization). **Total active threads: 512 out of a potential 170 * 32 * 48 = 261,120 threads** (assuming 48 warps per SM). That is **0.2% occupancy**.

**This is the single biggest performance opportunity.** Even increasing workgroup_size to 32 (one full warp) and having each thread handle one aspect of the chain processing would be a major improvement.

---

## 9. Concrete Recommendations (Priority Ordered)

### P0: Increase mutate/select/migrate workgroup parallelism

**Impact: HIGH (estimated 5-30% total pipeline speedup depending on pass time breakdown)**

The mutate and select shaders spend most of their time copying polygons (up to 48KB per chain). Using a workgroup of 32-64 threads to parallelize these copies would dramatically reduce per-chain latency and increase warp utilization from 3% to near 100%.

Implementation:
1. Change `@workgroup_size(1)` to `@workgroup_size(64)` for mutate, select, and migrate
2. Use `@builtin(local_invocation_index)` to distribute polygon copies across threads
3. Use shared memory or a flag variable for the single-threaded decision logic (mutation/acceptance), then have all threads participate in the copy
4. Update dispatch from `dispatch_workgroups(active, 1, 1)` to `dispatch_workgroups(active, 1, 1)` (same -- workgroup_id.x still identifies the chain)

### P1: Eliminate mutate's bulk copy phase

**Impact: MEDIUM (estimated 5-15% mutate speedup)**

Instead of copying all polygons from `chain_states` to `working_states` and then mutating in-place, restructure the mutation loop to read each polygon from `chain_states`, apply mutations in registers, and write directly to `working_states`. This halves the memory traffic for unmutated polygons (which is the vast majority, since mutation probabilities are low).

### P2: Bound the mutation retry loop

**Impact: LOW-MEDIUM (reduces tail latency, prevents warp divergence)**

Replace the unbounded `while !is_dirty && attempts < 1000u` loop with a bounded approach: after one full pass through mutation probabilities, if nothing fired, force a micro-adjustment on a random polygon. This guarantees termination in at most 2 iterations and eliminates worst-case warp divergence.

### P3: Experiment with rasterize_error workgroup size

**Impact: UNCERTAIN (needs profiling)**

Try `@workgroup_size(8, 8, 1)` (64 threads) with a tile size of 64 polygons. This reduces shared memory usage 4x, potentially allowing more concurrent workgroups per SM. The tradeoff is more tiles (more iterations of the outer loop for >64 polygon drawings). Profile both configurations.

### P4: Use subgroup operations for error reduction (when available)

**Impact: LOW (saves ~5 barriers per workgroup per iteration)**

If/when wgpu's `subgroups` feature stabilizes, replace the binary reduction tree with `subgroupAdd()` + a small shared-memory reduction across subgroups. This eliminates 5 of 8 barrier+add steps in the reduction.

### P5: Remove redundant CPU-side error accumulator reset

**Impact: NEGLIGIBLE (saves one small DMA per batch)**

The `atomicExchange` in select already resets accumulators. The CPU-side `write_buffer` is only needed for the first-ever batch. Initialize the buffer to zero at creation time and remove the per-batch reset.

---

## 10. What NOT to Optimize

1. **Buffer binding / descriptor set overhead:** With only 1 bind group per pass and pre-created bind groups, this is negligible.
2. **Command buffer encoding:** Encoding 150-200 compute passes takes microseconds on the CPU; the GPU execution time dominates.
3. **Reference image layout (SSBO vs texture):** The current linear SSBO read pattern is already coalesced. Texture cache would help inter-warp Y-locality but adds API complexity for marginal gain.
4. **Double-buffering chain_states for migration:** Theoretically more correct, but doubles the largest buffer (~24MB) and the current approach works correctly in practice.
