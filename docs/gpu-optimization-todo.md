# GPU Optimization TODO — Prioritized Roadmap

Generated from 5 expert analyses. See individual docs for full details:
- [GPU Compute Optimization](gpu-compute-optimization.md)
- [Evolutionary Algorithm Design](evolutionary-algorithm.md)
- [CPU-GPU Hybrid Architecture](cpu-gpu-hybrid.md)
- [Data Structures & Algorithms](data-structures-algorithms.md)
- [wgpu/Vulkan Performance](wgpu-vulkan-performance.md)

---

## Already Completed

- [x] Ring-topology migration replacing global-best replacement (select.wgsl)
- [x] AABB bounding box culling in rasterize (rasterize.wgsl)
- [x] L1 error metric replacing sqrt+Euclidean (error_reduce.wgsl)
- [x] GPU-specific MAX_ERROR_PER_PIXEL = 765.0 separate from CPU constant
- [x] Chain count increased to 512 (auto-capped to adapter limits)
- [x] Batch size increased to 50 iterations per submission
- [x] WSL2 Dozen driver buffer limit handling

---

## Tier 0 — Profiling (Do First)

> Without profiling data, priority ordering is guesswork. ~1 hour of work that pays for itself immediately.

- [ ] **Add GPU timestamp queries** — Enable `Features::TIMESTAMP_QUERY`, wire up `ComputePassTimestampWrites` on each of the 5 passes. Currently every pass sets `timestamp_writes: None`. This reveals exactly which pass dominates and guides all other work. *(wgpu-vulkan agent)*

---

## Tier 1 — GPU Shader Optimizations (Highest Impact)

> These are pure GPU-side changes with the best effort-to-impact ratio. Expected 2-4x throughput improvement combined.

- [ ] **Fuse rasterize + error_reduce into one shader** — The `render_targets` buffer (301 MB) exists solely to pass pixels between these passes. Fusing keeps pixel values in registers, eliminates 602 MB/iteration of bandwidth, removes one dispatch + barrier, and frees 301 MB of VRAM. ~100 lines of shader change. *(Recommended by 3/5 agents — GPU compute, data structures, wgpu-vulkan)*

- [ ] **Shared-memory polygon prefetch in rasterize** — All 64 threads in an 8x8 tile independently read the same polygon array from global memory. Cooperative loading into `var<workgroup>` shared memory reduces L1 cache pressure ~64x on polygon data. *(GPU compute, wgpu-vulkan agents)*

- [ ] **Increase rasterize workgroup size from 8x8 to 16x16** — Current 64-thread workgroups need 24 concurrent per SM for full occupancy. Moving to 256 threads cuts total workgroup count from 1.18M to 295K, reduces scheduler overhead, and enables more efficient shared-memory tiling. *(GPU compute agent)*

- [ ] **Parallelize struct copies in select/migrate** — select.wgsl and migrate use `@workgroup_size(1)`, copying 48 KB with a single thread. Switching to `@workgroup_size(256)` with cooperative copy gives ~256x speedup on the copy portion. *(GPU compute agent)*

- [ ] **Bump mutate/select workgroup_size(1) to workgroup_size(32)** — A workgroup of 1 wastes 31 of 32 NVIDIA warp lanes. Packing 32 chains per workgroup improves SM utilization. Trivial code change. *(GPU compute, wgpu-vulkan agents)*

- [ ] **Remove redundant CPU-side error accumulator zero-write** — The `atomicExchange` in `select_main` already resets each chain's accumulator. The `queue.write_buffer` of zeros at batch start (mod.rs ~line 100) is unnecessary and adds a host-to-device sync point. *(GPU compute agent)*

---

## Tier 2 — CPU-GPU Hybrid Architecture

> Use all available hardware. CPU is completely idle during GPU evolution.

- [ ] **Run CPU worker threads alongside GPU** — Move GPU work to a dedicated thread, spawn `num_cpus - 2` CPU `Evaluator` threads concurrently. CPU throughput is lower but adds search diversity (L2 error, variable-vertex polygons, different mutation operators). *(CPU-GPU hybrid agent)*

- [ ] **Bidirectional CPU↔GPU migration** — `drawing_to_gpu()` and `gpu_to_drawing()` already exist. CPU improvements inject into random GPU chains (set `fitness_bits = 0` to force recompute). GPU global best broadcasts to CPU workers via existing `broadcast::Sender<Drawing>`. *(CPU-GPU hybrid agent)*

- [ ] **Double-buffer command submission** — Currently `device.poll(Maintain::Wait)` blocks CPU while GPU runs, then GPU idles during next batch encoding. Two sets of staging buffers with alternating submission keeps both busy. Expected 10-30% throughput gain. *(CPU-GPU hybrid, wgpu-vulkan agents)*

- [ ] **Async GPU submission for housekeeping overlap** — Split `run_batch` into non-blocking `submit_batch` + deferred `poll_result`. PNG rendering, JSON serialization, and disk I/O execute during GPU compute instead of after. *(CPU-GPU hybrid agent)*

---

## Tier 3 — Evolutionary Algorithm Improvements

> Transform from parallel hill climbing to a proper genetic algorithm. The algorithm changes are orthogonal to performance — they improve solution quality.

- [ ] **Stagnation detection & reinitialization** — Repurpose `_pad0` field in `GpuDrawingState` as `stagnation_counter` (zero layout change). Increment on rejection, reset on acceptance. After threshold (~5000 iterations), reinitialize the chain. Prevents dead chains from wasting GPU cycles. *(Evolutionary algorithm agent)*

- [ ] **Adaptive mutation strength** — Scale mutation deltas by fitness: `scale = 0.1 + 0.9 * (1.0 - fitness/100.0)`. Large exploratory mutations early, fine-grained adjustments late. Zero additional state required. *(Evolutionary algorithm agent)*

- [ ] **Clone-based add-polygon** — Instead of random triangles (near-zero chance of benefit), 50% chance to duplicate an existing polygon with slight perturbation. Leverages accumulated optimization. ~30 lines in mutate.wgsl. *(Evolutionary algorithm agent)*

- [ ] **Region-based crossover** — Split image spatially, take each parent's polygons from their respective region. New `crossover.wgsl` shader dispatched every ~200 iterations, pairing chains `(2i, 2i+1)`. No additional buffers needed. *(Evolutionary algorithm agent)*

- [ ] **Hierarchical island model** — Replace flat ring with 32 islands of 16 chains. Intra-island ring migration every 50 iterations, inter-island migration every 500 iterations. Preserves diversity while propagating good solutions. New entry point in select.wgsl. *(Evolutionary algorithm agent)*

---

## Tier 4 — Data Structure Optimizations

> Reduce memory footprint and bandwidth further. Best done after Tier 1 fuse.

- [ ] **Pack polygon colors as u32** — RGBA stored in 4x f32 (16 bytes) but values are fundamentally 8-bit. Pack into one u32, shrink GpuPolygon from 48 to 32 bytes (33% reduction), save 15.7 MB across all buffers. Unpack with bitwise ops. *(Data structures agent)*

- [ ] **In-place mutation with undo log** — Mutate pass copies 48 KB/chain (23.5 MB total) every iteration even though typically 1-3 polygons change. Mutate in-place, store a 52-byte undo record. Eliminates `working_states` buffer, halves mutate bandwidth. *(Data structures agent)*

---

## Impact Summary

| Tier | Estimated Throughput Gain | Effort | Key Benefit |
|------|--------------------------|--------|-------------|
| 0 | 0% (enables measurement) | ~1 hour | Data-driven decisions |
| 1 | 2-4x | 1-2 days | Pure GPU efficiency |
| 2 | +30-80% on top of Tier 1 | 2-3 days | Use all hardware |
| 3 | Better convergence quality | 2-3 days | Solution quality, not raw speed |
| 4 | +10-20% bandwidth savings | 1-2 days | Memory reduction |
