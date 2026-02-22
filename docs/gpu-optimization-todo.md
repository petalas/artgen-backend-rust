# GPU Evolution Optimization TODO

Synthesized from 6 expert analyses (GPU compute, evolutionary algorithm, CPU-GPU hybrid, data structures, wgpu API, numerical methods). February 2026. Round 2 — focuses on NEW ideas beyond previous optimizations.

**Target hardware:** RTX 5090 (170 SMs, 32-wide warps, 48KB shared memory per SM)

---

## Already Completed

- [x] Fused rasterize + error computation in single pass
- [x] Double-buffered async staging (CPU reads batch N-1 while GPU computes batch N)
- [x] Persistent PCG-32 RNG per offspring slot
- [x] Shared memory polygon tiling (768 polys/tile cooperative prefetch)
- [x] Reference image stored as GPU texture (2D cache-friendly)
- [x] GpuParams via uniform buffer (shared bind group across pipelines)
- [x] Adaptive mutation scale (1.2x accept, pow(0.99,1/lambda) reject, burst on stagnation)
- [x] (1+lambda)-ES with configurable lambda (1-64, power-of-2)
- [x] Island model with intra/inter-island ring migration
- [x] 13+ mutation operators (structural, geometric, vertex, color, crossover)
- [x] Dirty bounding box tracking per offspring
- [x] Tile culling via bin_polygons shader (optional)
- [x] Incremental evaluation mode (optional)
- [x] Half-space edge function with AABB culling
- [x] Vertex quantization (2x u16) and color quantization (pack4x8unorm)
- [x] Vulkan pipeline cache for shader binary reuse
- [x] Double-buffered command encoding
- [x] Timestamp query profiling infrastructure
- [x] Atomic lock-free selection (atomicMax/atomicExchange)
- [x] Degenerate triangle culling on acceptance
- [x] Compute pass boundary consolidation (single pass for non-profiled iterations)

---

## Tier 1 — High Impact, Reasonable Effort

### 1.1 Reorder submit/collect in run_batch() — [7-30% throughput]
**Source:** CPU-GPU Hybrid expert
**Problem:** `finish_pending_readback()` blocks on `device.poll(Wait)` *before* submitting the next batch, creating a GPU idle bubble while CPU does mutex checks, stats, PNG encoding.
**Fix:** Submit batch N+1 first, then block on batch N results. GPU starts immediately while CPU processes.
**Effort:** Medium — restructure `run_batch()` control flow
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 1.2 Fix incremental eval parent re-rasterization — [5-20x on incremental path]
**Source:** GPU Compute + Data Structures experts
**Problem:** With incremental eval enabled, `rasterize_error.wgsl` re-rasterizes the FULL parent drawing at every pixel in the dirty bbox. But `chain_framebuffers` already stores the parent's rendered output — just never updated after acceptance.
**Fix:** (a) Update `chain_framebuffers` in the select shader after acceptance, (b) read cached pixel instead of re-rasterizing parent (O(1) vs O(polygon_count) per pixel).
**Effort:** Medium — shader changes in select + rasterize_error
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 1.3 Tighter dirty bboxes for z-order mutations — [2-7.5x rasterize speedup for affected mutations]
**Source:** Numerical Methods expert
**Problem:** `adjacent_swap`, `remove_polygon`, `swap_colors` mutations return `full_image_bbox()` in mutate shader, forcing full-image re-rasterization even though only 2 polygons' bounding boxes change.
**Fix:** Replace with `merge_bbox(polygon_a_bbox, polygon_b_bbox)`. Since z-order mutations are 20-30% of operations, weighted average rasterize speedup is 2-7.5x under incremental eval.
**Effort:** Low — ~20 lines in mutate.wgsl
**Details:** [numerical-methods-optimization.md](numerical-methods-optimization.md)

### 1.4 Stratified offspring mutation — [15-25% faster convergence]
**Source:** Evolutionary Algorithm expert
**Problem:** With lambda=64, all offspring sample from the same probability distribution. Rare-but-high-impact mutations (add/remove polygon) fire probabilistically once per ~150 iterations instead of every iteration.
**Fix:** Partition offspring into deterministic strata — reserved slots for add_polygon, remove_polygon, structural mutations, etc. Latin hypercube sampling over mutation type space.
**Effort:** Low-Medium — mutation selection logic in mutate.wgsl
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

### 1.5 Color-aware polygon initialization — [2-5x add_polygon acceptance]
**Source:** Evolutionary Algorithm expert
**Problem:** New polygons get random colors, requiring many subsequent color mutations to become useful.
**Fix:** Sample reference image texture at polygon centroid to initialize color. ~15 lines of code — add texture binding to mutate shader.
**Effort:** Low
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

---

## Tier 2 — Medium Impact, Low Effort (Quick Wins)

### 2.1 Skip unconditional polygon writeback in multi-mutation — [~500KB/iter saved]
**Source:** GPU Compute expert
**Problem:** Per-polygon loop in `mutate.wgsl` unpacks, potentially mutates, then repacks and writes back ALL polygons even when nothing changed (~96% are unmodified).
**Fix:** Add `modified` flag, skip writeback for unmodified polygons.
**Effort:** Very low — zero risk
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 2.2 Fix `shared_polys` over-allocation — [improved SM occupancy]
**Source:** Data Structures expert
**Problem:** `shared_polys` is hardcoded to 1536 entries but actual `TILE_CAP = THREAD_COUNT * 3` at default 16x16 uses only 768. Wastes 12KB shared memory per workgroup, reducing SM occupancy.
**Fix:** Change declaration to `array<Polygon, TILE_CAP>`. Trivial.
**Effort:** Trivial
**Details:** [data-structures-algorithms.md](data-structures-algorithms.md)

### 2.3 Submission index tracking for targeted polling — [eliminates spurious waits]
**Source:** wgpu API expert
**Problem:** `device.poll()` uses `submission_index: None`, waiting for ALL pending work including the just-submitted batch.
**Fix:** Capture `queue.submit()` return value, pass to `device.poll()`.
**Effort:** Trivial
**Details:** [wgpu-api-optimization.md](wgpu-api-optimization.md)

### 2.4 Persistent staging buffers — [eliminates per-call allocation]
**Source:** wgpu API expert
**Problem:** `readback_chains` and `evaluate_chain_fitness` create and destroy MAP_READ staging buffers on every call, triggering Vulkan memory management operations.
**Fix:** Pre-allocate persistent staging buffers at pipeline creation time.
**Effort:** Low
**Details:** [wgpu-api-optimization.md](wgpu-api-optimization.md)

### 2.5 Conditional timestamp collection — [eliminate 95% of extra pass transitions]
**Source:** CPU-GPU Hybrid expert
**Problem:** Every batch forces the last iteration into 3 separate compute passes for timestamp profiling instead of the single bulk pass.
**Fix:** Collect timestamps only every Nth batch (e.g., every 20th).
**Effort:** Trivial — 10 minute change
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 2.6 Push constants for GpuParams — [eliminate UBO write + bind group overhead]
**Source:** wgpu API expert
**Problem:** 256-byte GpuParams written via `queue.write_buffer()` + `set_bind_group(1, ...)` called 3-4 times per iteration (192-256 redundant bind group commands per batch).
**Fix:** Use `immediate_size: 256` in pipeline layout. Data lives in command processor registers for zero-latency reads.
**Effort:** Medium — pipeline layout changes across all shaders
**Details:** [wgpu-api-optimization.md](wgpu-api-optimization.md)

---

## Tier 3 — High Impact, High Effort (Architectural)

### 3.1 Coarse-to-fine two-phase screening — [up to 8x rasterize speedup]
**Source:** Evolutionary Algorithm expert
**Problem:** All 64 offspring rasterized at full 512x512 despite 90-95% rejection rate.
**Fix:** Coarse screen at 128x128 for all offspring, full-resolution only for top 4. Reduces total rasterization from 16.8M to ~2M pixels per iteration.
**Effort:** High — new shader pass, intermediate buffers, architectural rework
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

### 3.2 Progressive resolution evaluation — [up to 16x early speedup]
**Source:** Numerical Methods expert
**Problem:** Full-resolution evaluation wasteful in early evolution when fitness is low.
**Fix:** Three-tier schedule: 1/4 res below 60% fitness, 1/2 at 60-80%, full above 80%.
**Effort:** High — reference image mipmaps, dynamic dispatch sizing, fitness-dependent switching
**Details:** [numerical-methods-optimization.md](numerical-methods-optimization.md)

### 3.3 Copy-on-write offspring representation — [50-100x memory reduction]
**Source:** Data Structures expert
**Problem:** Each offspring duplicates full 16032-byte GpuDrawingState even though single-mutation modifies only 1-2 polygons.
**Fix:** Delta representation: (polygon_index, old_polygon, new_polygon) at ~48 bytes per mutation.
**Effort:** Very high — fundamental restructuring of mutate and rasterize shaders
**Details:** [data-structures-algorithms.md](data-structures-algorithms.md)

### 3.4 Error-guided polygon placement — [3-10x add_polygon acceptance at high fitness]
**Source:** Evolutionary Algorithm expert
**Problem:** New polygons land at random positions, <1% hit rate on high-error regions at high fitness.
**Fix:** Lightweight 4x4 per-chain error grid (64 bytes) biases add_polygon origins and mutation targeting toward high-error cells.
**Effort:** Medium-High — error grid computation, sampling logic in mutate shader
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

---

## Tier 4 — Medium Impact, Medium Effort

### 4.1 SoA shared memory layout — [33% more polys/tile]
**Source:** GPU Compute expert
**Problem:** `shared_polys` stores full 16-byte Polygon structs. Color only needed for ~5-10% that pass half-space test.
**Fix:** Store only 12-byte geometry in shared memory, defer 4-byte color read to L2 on hit. Fits 1024 polys/tile vs 768.
**Effort:** Medium
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 4.2 Integer-space AABB culling — [~700M saved FP ops per offspring]
**Source:** GPU Compute expert
**Problem:** AABB cull requires unpacking all 3 vertices to float before checking bounds.
**Fix:** Perform AABB test in u16 pixel coordinates (bit ops on raw packed u32 words) before float unpack. Saves 6 FP ops per rejected polygon (~90% reject rate).
**Effort:** Low-Medium
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 4.3 Parallelize bin_polygons shader — [32-64x binning utilization]
**Source:** GPU Compute + Data Structures experts
**Problem:** `@workgroup_size(1,1,1)` — one thread per offspring. 128 single-thread workgroups across 170 SMs.
**Fix:** Use 64-thread workgroup, parallelize tile count reset. Polygon scatter remains serial for ordering but reset phase is trivially parallelizable.
**Effort:** Medium
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 4.4 Polygon splitting mutation — [adds representational capacity]
**Source:** Evolutionary Algorithm expert
**Problem:** Adding random new triangles is inefficient for fine detail.
**Fix:** Subdivide existing large polygon into 2-3 sub-triangles at centroid. Fitness-neutral (accepted immediately), then evolution differentiates colors.
**Effort:** Medium
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

### 4.5 L2-squared error metric — [5-15% faster perceptual convergence]
**Source:** Numerical Methods expert
**Problem:** L1 error treats all channel deviations equally. L2-squared penalizes concentrated single-channel errors more.
**Fix:** Use `dr*dr+dg*dg+db*db` with right-shift-by-8 scaling to prevent u32 overflow.
**Effort:** Low-Medium
**Details:** [numerical-methods-optimization.md](numerical-methods-optimization.md)

### 4.6 Offload CPU rendering/encoding to worker thread — [15-30% combined with 1.1]
**Source:** CPU-GPU Hybrid expert
**Problem:** Main thread does inline CPU rasterization, PNG encoding, JSON serialization, file I/O between batches.
**Fix:** Bounded-channel worker thread for rendering/encoding tasks.
**Effort:** Medium
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 4.7 Adaptive batch size controller — [auto-tune gpu_batch_iters]
**Source:** CPU-GPU Hybrid expert
**Problem:** Static `gpu_batch_iters` (default 50) — optimal varies enormously with config.
**Fix:** EWMA-based controller targeting ~10ms per batch, auto-tunes dynamically.
**Effort:** Medium
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 4.8 Multi-mutation RNG batching — [30-50% less arithmetic in mutation loop]
**Source:** Data Structures expert
**Problem:** Multi-mutation path draws ~18 random numbers per polygon regardless of whether any mutation fires.
**Fix:** Batch 2-3 RNG calls, use bit extraction, skip inner loop when no mutation probability exceeds threshold.
**Effort:** Medium
**Details:** [data-structures-algorithms.md](data-structures-algorithms.md)

### 4.9 Async non-blocking best-chain readback — [eliminate 0.5-3ms stall]
**Source:** CPU-GPU Hybrid expert
**Problem:** New global best triggers synchronous GPU round-trip that defeats double-buffering.
**Fix:** Piggyback best-chain copy onto next batch's command encoder, read one batch later.
**Effort:** Medium
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

---

## Ruled Out

| Idea | Why |
|------|-----|
| f16 precision for blending | ~0.3-0.5 error/channel after 150 blends creates noise floor drowning micro-adjust improvements |
| Integer (u16) arithmetic | 2.5x more ALU ops — no native u16 in WGSL, division-by-255 approximation overhead |
| SSIM error metric | Requires local neighborhood statistics incompatible with per-pixel accumulation |
| RNG upgrade from PCG-32 | Adequate period and quality for noise-tolerant evolutionary process |

---

## Expert Analysis Documents

| Expert | Document |
|--------|----------|
| GPU Compute | [gpu-compute-optimization.md](gpu-compute-optimization.md) |
| Evolutionary Algorithm | [evolutionary-algorithm.md](evolutionary-algorithm.md) |
| CPU-GPU Hybrid | [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md) |
| Data Structures | [data-structures-algorithms.md](data-structures-algorithms.md) |
| wgpu/Vulkan API | [wgpu-api-optimization.md](wgpu-api-optimization.md) |
| Numerical Methods | [numerical-methods-optimization.md](numerical-methods-optimization.md) |
