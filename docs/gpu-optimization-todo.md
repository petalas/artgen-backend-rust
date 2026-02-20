# GPU Evolution Pipeline — Optimization TODO

Consolidated from 5 expert analyses (2026-02-20). Items are tiered by expected impact and implementation effort.

## Already Completed

- [x] Fused rasterize + error_reduce into single compute shader (~20% faster, eliminated 301MB buffer)
- [x] Per-polygon AABB early-out in rasterizer (skips ~97% of pixel tests)
- [x] L1 error metric instead of sqrt(L2)
- [x] Ring-topology migration (replaced catastrophic global-best)
- [x] Increased chains from 64 → 512, iterations/batch from 10 → 50
- [x] Crossover (spatial + uniform) in mutate shader
- [x] Tournament selection within islands
- [x] Island model with configurable island count and migration intervals

---

## Tier 1 — Quick Wins (high impact, low effort)

### 1.1 Consolidate triple staging readback into single poll
**Source:** CPU-GPU hybrid, wgpu/Vulkan
**Files:** `src/gpu_evolver/mod.rs`
Each batch calls `device.poll(Wait)` three times (control flags, timestamps, fitness). All three staging buffers are filled by the same command buffer — map all three simultaneously and poll once. Eliminates 2 unnecessary CPU-GPU sync points per batch. Especially impactful on WSL2 where each fence wait crosses the VM boundary.

### 1.2 Accept neutral mutations (equal fitness)
**Source:** Evolutionary algorithm
**Files:** `src/shaders/select.wgsl`
Selection uses strict `>`, rejecting neutral moves. Polygon art has vast fitness plateaus. Accept equal-fitness candidates with ~50% probability (e.g., `iteration_number % 2 == 0`). One-line change, potentially significant convergence improvement in late stages.

### 1.3 Remove redundant error accumulator reset
**Source:** GPU compute
**Files:** `src/gpu_evolver/mod.rs`
The error accumulator buffer is zeroed via `queue.write_buffer` before every batch, but the select shader already resets it via `atomicExchange` after each iteration. Remove the CPU-side write.

### 1.4 Swap-remove for polygon insert/remove
**Source:** Data structures
**Files:** `src/shaders/mutate.wgsl`
Insert and remove mutations perform O(n) array shifts (up to 48KB of memcpy). Replace with swap-with-last-element + count adjustment for O(1). Z-order impact is negligible since the reorder mutation already randomizes polygon order.

### 1.5 Skip timestamp readback on non-reporting batches
**Source:** CPU-GPU hybrid
**Files:** `src/gpu_evolver/mod.rs`
Timestamp resolve + copy + map runs every batch but is only consumed every ~2 seconds for stats. Only resolve timestamps on reporting batches.

### 1.6 Batch island readbacks into single command buffer
**Source:** CPU-GPU hybrid
**Files:** `src/gpu_evolver/mod.rs`
`build_gpu_stats` issues N separate GPU submissions (one per island). Batch all island readbacks into one command buffer + one staging buffer map. Saves 1-5ms every 2 seconds.

---

## Tier 2 — Major Improvements (high impact, medium effort)

### 2.1 Quantized 16-byte polygon representation
**Source:** Data structures
**Files:** `src/gpu_evolver/buffers.rs`, `src/shaders/*.wgsl`
Pack polygon into a single `vec4<u32>` (16 bytes): color as packed u8x4, vertices as packed u16x2. Shrinks `GpuDrawingState` from 48,032 → 16,032 bytes (67% reduction). Triples cache utilization, triples shared memory tile capacity, cuts memory bandwidth 3x. Unpack cost is ~1 cycle/polygon on modern GPUs.

### 2.2 Reference image as texture instead of storage buffer
**Source:** wgpu/Vulkan
**Files:** `src/gpu_evolver/pipeline.rs`, `src/shaders/rasterize_error.wgsl`
Reference image is stored as `array<u32>` in storage memory. The 16x16 workgroup access pattern has strong 2D spatial locality — `texture_2d<f32>` with `textureLoad` would leverage dedicated texture cache hardware (separate from L1, native 2D tiling). Eliminates manual bit-unpacking. Estimated 10-30% improvement in rasterize_error pass.

### 2.3 Double-buffered staging for CPU/GPU overlap
**Source:** CPU-GPU hybrid, wgpu/Vulkan
**Files:** `src/gpu_evolver/mod.rs`, `src/gpu_evolver/pipeline.rs`
Current flow: submit batch → block for readback → process → submit next. With two staging buffer sets, submit batch N+1 immediately while reading N's results. Estimated 10-30% throughput increase.

### 2.4 Eliminate redundant full-buffer copy in mutate shader
**Source:** GPU compute
**Files:** `src/shaders/mutate.wgsl`
Mutate copies all ~1000 polygons from `chain_states` to `working_states`, then mutates a handful. With low per-polygon mutation probability, >95% of polygons are copied unchanged. Restructure to read-mutate-write in a single pass, halving mutation-phase bandwidth.

### 2.5 Push constants for GpuParams
**Source:** wgpu/Vulkan
**Files:** `src/gpu_evolver/pipeline.rs`, `src/gpu_evolver/mod.rs`, `src/shaders/*.wgsl`
The 128-byte uniform buffer is re-uploaded via `queue.write_buffer()` every batch (allocates staging + copy). 128 bytes fits within Vulkan's minimum push constant guarantee. Eliminates staging allocation, buffer binding slot, and `params_buf` entirely.

### 2.6 Bound the `while !is_dirty` retry loop in mutate
**Source:** GPU compute
**Files:** `src/shaders/mutate.wgsl`
Some chains finish in 1 attempt while others retry hundreds of times, serializing the entire warp. Force a guaranteed micro-mutation (e.g., nudge one vertex by ±1) after the first unsuccessful pass. Bounds execution to at most 2 iterations, eliminates tail-latency spikes.

---

## Tier 3 — Algorithmic & Structural (medium impact, medium effort)

### 3.1 Increase workgroup size for mutate/select/migrate (1 → 64)
**Source:** GPU compute, Data structures
**Files:** `src/shaders/mutate.wgsl`, `src/shaders/select.wgsl`
These shaders all use `@workgroup_size(1)`, wasting 97% of warp lanes. With 512 single-thread workgroups on a 170-SM GPU, occupancy is <0.2%. Even if only thread 0 does work, larger workgroups improve latency hiding. Better: distribute polygon copy work across threads.

### 3.2 Single-mutation mode
**Source:** Evolutionary algorithm
**Files:** `src/shaders/mutate.wgsl`
With N=500 polygons, ~41 per-polygon mutations fire per iteration, drowning signal in noise. A weighted-roulette single-operator-per-iteration mode gives selection cleaner signal. GPU throughput compensates for smaller per-iteration changes.

### 3.3 Add missing mutation operators (scale, rotate, adjacent-swap)
**Source:** Evolutionary algorithm
**Files:** `src/shaders/mutate.wgsl`
No scale (resize around centroid) or rotate operators exist. Reorder uses random swaps when adjacent swaps are far more effective for z-order optimization. These fill gaps in search space exploration.

### 3.4 Adaptive mutation rates
**Source:** Evolutionary algorithm
**Files:** `src/shaders/mutate.wgsl`, `src/shaders/select.wgsl`
All chains use identical static mutation parameters. `iteration_number` is passed to GPU but unused. Implement fitness-proportional intensity (aggressive for low-fitness, fine-grained for high-fitness) and stagnation-triggered mega-mutations.

### 3.5 Perceptually-weighted error (BT.601 luma)
**Source:** Evolutionary algorithm
**Files:** `src/shaders/rasterize_error.wgsl`
L1 treats R, G, B equally but human vision is far more sensitive to green. Apply BT.601 weights (0.299, 0.587, 0.114) to channel differences. 3-line change, improves visual quality at zero performance cost.

### 3.6 Pipeline cache for shader compilation
**Source:** wgpu/Vulkan
**Files:** `src/gpu_evolver/pipeline.rs`
All pipelines use `cache: None`, forcing full WGSL→SPIR-V→ISA recompilation on every launch (0.5-2.5s). wgpu v22 `PipelineCache` can serialize to disk, reducing subsequent startup to ~50ms.

---

## Tier 4 — Ambitious / Future (high impact, high effort)

### 4.1 CPU-GPU co-evolution
**Source:** CPU-GPU hybrid
**Files:** `src/main.rs`, `src/gpu_evolver/mod.rs`, `src/evaluator.rs`
Run CPU worker threads alongside GPU doing broader-search mutations. Periodically inject improved CPU candidates into worst-performing GPU chains via `write_buffer`. Heterogeneous island migration at zero GPU cost.

### 4.2 Subgroup intrinsics for error reduction
**Source:** GPU compute, Data structures
**Files:** `src/shaders/rasterize_error.wgsl`
The shared-memory binary reduction tree uses `workgroupBarrier()` at every step. The final 5 steps (stride ≤ 16) could use `subgroupAdd`, and once wgpu's subgroup feature stabilizes, the entire reduction collapses to `subgroupAdd` + 3 cross-subgroup steps.

### 4.3 Configurable MAX_POLYGONS_PER_IMAGE
**Source:** Data structures
**Files:** `src/settings.rs`, `src/gpu_evolver/buffers.rs`, `src/shaders/*.wgsl`
Fixed at 1000 but many runs use far fewer. Making this configurable (e.g., 256-512) would proportionally shrink per-chain state size and improve all memory-bound passes.

---

## Expert Analysis Docs

Full detailed analysis for each area:
- [GPU Compute Optimization](gpu-compute-optimization.md)
- [Evolutionary Algorithm Design](evolutionary-algorithm.md)
- [CPU-GPU Hybrid Architecture](cpu-gpu-hybrid.md)
- [Data Structures & Algorithms](data-structures-algorithms.md)
- [wgpu/Vulkan Performance](wgpu-vulkan-performance.md)
