# GPU Optimization TODO

Consolidated from 5 expert analyses. Items are grouped by tier (effort/impact).

---

## Tier 1: Quick Wins (trivial changes, high impact)

- [x] **Fix catastrophic migration policy** — Ring-topology migration instead of "all chains clone global best". Each chain compares with right neighbor only. `select.wgsl`
  - Sources: [evolutionary-algorithm.md](evolutionary-algorithm.md) #1, [gpu-compute-optimization.md](gpu-compute-optimization.md)

- [x] **AABB culling in rasterize** — Per-polygon bounding box early-out before edge function evaluation. Skips ~97% of polygon-pixel tests. `rasterize.wgsl`
  - Sources: [gpu-compute-optimization.md](gpu-compute-optimization.md) #2, [data-structures-algorithms.md](data-structures-algorithms.md) #1

- [x] **Drop sqrt in error metric** — Replaced with sum of absolute differences (L1 distance). Avoids expensive per-pixel sqrt. `error_reduce.wgsl`
  - Sources: [gpu-compute-optimization.md](gpu-compute-optimization.md) #4, [data-structures-algorithms.md](data-structures-algorithms.md) #2

- [x] **GPU-specific MAX_ERROR_PER_PIXEL** — Added `GPU_MAX_ERROR_PER_PIXEL = 765.0` (255×3 for L1) separate from CPU's 441.67 (Euclidean). `settings.rs`, `buffers.rs`
  - Sources: [gpu-compute-optimization.md](gpu-compute-optimization.md) #4

- [x] **Increase chain count (64 → 512)** — 8x more parallel evolution chains. ~200 MB total GPU memory at 384x384, well under 32 GB. `settings.rs`
  - Sources: [gpu-compute-optimization.md](gpu-compute-optimization.md) #1, [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #4

- [x] **Increase batch size (10 → 50)** — 5x more iterations per GPU submission. Amortizes CPU↔GPU synchronization. `settings.rs`
  - Sources: [gpu-compute-optimization.md](gpu-compute-optimization.md) #7, [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #6

---

## Tier 2: Moderate Effort (structural changes, significant impact)

- [ ] **Merge rasterize + error_reduce into single pass** — Eliminates `render_targets` buffer (~301 MB at 512 chains), removes one dispatch, keeps pixel color in registers. Highest single ROI change.
  - Sources: [gpu-compute-optimization.md](gpu-compute-optimization.md) #5

- [ ] **Shared memory polygon loading in rasterize** — Cooperative workgroup loading into shared memory. 64 threads load 64 polygons, broadcast from shared mem instead of global. 2-4x memory bandwidth reduction.
  - Sources: [gpu-compute-optimization.md](gpu-compute-optimization.md) #3, [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #3

- [ ] **CPU workers alongside GPU** — Spawn CPU evaluator threads sharing global best via `Arc<RwLock<Drawing>>`. CPU does fine-tuning (high micro_adjust), GPU does exploration. 10-30% improvement rate boost, especially at high fitness.
  - Sources: [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md) #1, #2

- [ ] **PNG encoding offload** — Move rendering + PNG encoding to a dedicated thread. Non-blocking `try_send` from main loop. Eliminates ~200ms stalls.
  - Sources: [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md) #5

- [ ] **Double-buffered readback** — Two staging buffer sets, alternating each batch. GPU never stalls between batches. 20-30% throughput improvement.
  - Sources: [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md) #3, [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #4

- [ ] **Split polygon copy from mutate** — Separate parallel copy pass (workgroup_size(256)) + sequential mutate pass. Removes 48 KB sequential bottleneck per chain.
  - Sources: [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #1c

- [ ] **Timestamp queries for profiling** — Instrument each compute pass to measure actual per-pass GPU time. Informs all further optimization decisions. Request `Features::TIMESTAMP_QUERY`.
  - Sources: [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #9

---

## Tier 3: Algorithmic Improvements (new capabilities)

- [ ] **Heterogeneous mutation strategies** — 4 strategy classes derived from `chain_id % 4`: Explorer (structural), Refiner (micro-adjust), Colorist (color), Default. Zero new buffers.
  - Sources: [evolutionary-algorithm.md](evolutionary-algorithm.md) #4

- [ ] **Adaptive step sizes (1/5th rule)** — Per-chain success rate tracking. Auto-adjust mutation deltas: increase if >20% success, decrease if <20%. Reuse `_pad` fields in `GpuDrawingState`.
  - Sources: [evolutionary-algorithm.md](evolutionary-algorithm.md) #3

- [ ] **Error-guided polygon placement** — Compute error heatmap in error_reduce, bias new polygon placement toward high-error tiles. 3-10x convergence improvement.
  - Sources: [evolutionary-algorithm.md](evolutionary-algorithm.md) #6, [data-structures-algorithms.md](data-structures-algorithms.md) #5

- [ ] **Segment graft crossover** — New compute pass pairing adjacent chains. Graft contiguous polygon subsequence from one chain into another. Preserves ordering.
  - Sources: [evolutionary-algorithm.md](evolutionary-algorithm.md) #2

- [ ] **Stochastic acceptance (SA hybrid)** — Metropolis acceptance for per-chain selection (not global best). Temperature parameter decaying over time. Escapes local optima.
  - Sources: [evolutionary-algorithm.md](evolutionary-algorithm.md) #5

- [ ] **Fix complexity penalty** — Replace multiplicative penalty (punishes good solutions more) with lexicographic selection: minimize error first, minimize polygons as tiebreaker.
  - Sources: [evolutionary-algorithm.md](evolutionary-algorithm.md) #7

- [ ] **Perceptual error metric** — Weighted RGB channels: `2*dr² + 4*dg² + 3*db²`. Green-heavy weighting matches human luminance perception. Zero extra cost.
  - Sources: [data-structures-algorithms.md](data-structures-algorithms.md) #9

- [ ] **Subgroup operations in error_reduce** — `subgroupAdd` eliminates most shared memory barriers. Single PTX instruction vs tree reduction. Request `Features::SUBGROUP`.
  - Sources: [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #8

- [ ] **Pipeline caching** — `PipelineCache` to avoid 100-500ms shader recompilation on restart.
  - Sources: [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) #6

---

## Tier 4: Complex / Architectural (highest potential, most effort)

- [ ] **Dirty-rectangle delta evaluation** — Track mutated polygon, only rasterize/evaluate pixels in old+new AABB union. Requires per-pixel error buffer. 100-1000x for common single-polygon mutations. Falls back to full rasterize for structural mutations.
  - Sources: [data-structures-algorithms.md](data-structures-algorithms.md) #4a, [evolutionary-algorithm.md](evolutionary-algorithm.md) #8

- [ ] **Tile-based polygon binning** — Pre-pass bins polygons into 16x16 tiles by AABB overlap. Rasterize only iterates tile's polygons (3-10 instead of 1000). 10-50x rasterize speedup.
  - Sources: [data-structures-algorithms.md](data-structures-algorithms.md) #3

- [ ] **Multi-scale / coarse-to-fine evolution** — Start at 96x96 or 192x192, switch to full resolution when improvement rate plateaus. 4-16x faster early evolution, 30-50% total time reduction.
  - Sources: [data-structures-algorithms.md](data-structures-algorithms.md) #8

---

## Expert Reports

- [GPU Compute Optimization](gpu-compute-optimization.md) — occupancy, memory bandwidth, shader dispatch
- [Evolutionary Algorithm](evolutionary-algorithm.md) — migration, crossover, adaptive params, selection
- [CPU-GPU Hybrid Architecture](cpu-gpu-hybrid.md) — CPU workers, async pipeline, data sharing
- [Data Structures & Algorithms](data-structures-algorithms.md) — AABB, delta eval, tiling, error metrics
- [wgpu/Vulkan Performance](wgpu-vulkan-performance.md) — wgpu API, profiling, subgroup ops, caching
