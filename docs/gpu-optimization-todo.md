# GPU Optimization TODO

Synthesized from expert analyses: GPU compute, evolutionary algorithm, data structures, and wgpu/Vulkan performance.

**Target hardware:** RTX 5090 (170 SMs, 32-wide warps, 48KB shared memory per SM)

---

## Already Completed

- [x] Fused rasterize+error pass with workgroup reduction
- [x] Tiled polygon prefetch (768-polygon shared-memory tiles)
- [x] Adaptive mutation scale (1.2x accept, pow(0.8, 1/lambda) reject, burst to 5.0 on stagnation)
- [x] Double-buffered async staging readback
- [x] Per-offspring persistent PCG RNG
- [x] Pipeline cache for shader binaries
- [x] Degenerate triangle culling on acceptance
- [x] Variable lambda (1-64 offspring, power-of-2)
- [x] Island model with intra/inter-island ring migration
- [x] Timestamp query profiling (opt-in)
- [x] Workgroup size 16x16 for rasterize, 64 for mutate/select

---

## Tier 1: Quick Wins (Low effort, zero/low risk, immediate payoff)

### 1.1 Drop `sqrt()` from per-pixel error metric
**Sources:** GPU compute, data structures
**Effort:** ~5 lines in rasterize_error.wgsl
**Impact:** Eliminates 262K sqrt calls per offspring per iteration. Squared L2 is monotonically equivalent for fitness comparison — offspring ranking is unchanged.
**Risk:** Near zero for GPU-only. Fitness values will be in different units (squared) but this only affects display/logging, not selection correctness.

### 1.2 Remove redundant alpha clamping in parent-to-offspring copy
**Sources:** GPU compute, wgpu/Vulkan
**Effort:** ~5 lines in mutate.wgsl
**Impact:** Eliminates ~400K unnecessary unpack-clamp-repack operations per batch. Alpha is already clamped by mutation paths.
**Risk:** Near zero — the invariant is maintained by mutation; this is a no-op path.

### 1.3 Push constants for GpuParams (128 bytes)
**Sources:** wgpu/Vulkan
**Effort:** ~30 lines (pipeline layout + shader var type change)
**Impact:** Eliminates staging buffer allocation, DMA transfer, and descriptor binding per pass. Data lives in command processor registers for zero-latency reads.
**Risk:** Low. 128 bytes fits within Vulkan minimum guaranteed push constant size. Requires `Features::PUSH_CONSTANTS`.

### 1.4 Set `min_binding_size` on all bind group layout entries
**Sources:** wgpu/Vulkan
**Effort:** ~20 lines in pipeline.rs
**Impact:** Eliminates per-dispatch CPU-side buffer validation overhead in wgpu.
**Risk:** None.

### 1.5 Parallel min-reduction in select shader
**Sources:** GPU compute
**Effort:** ~30 lines in select.wgsl
**Impact:** Turns O(lambda) serial atomic operations into O(log2(lambda)) parallel steps. With lambda=64, this is 6 steps instead of 64.
**Risk:** Low. Correctness is straightforward with standard parallel reduction pattern.

---

## Tier 2: Medium Effort, High Impact

### 2.1 Compute pass boundary consolidation
**Sources:** GPU compute, wgpu/Vulkan
**Effort:** ~50 lines in mod.rs encoder logic
**Impact:** Current 50-iteration batch creates ~200 separate compute passes (each with implicit Vulkan barriers). Merging 49 non-profiled iterations into a single compute pass with inline pipeline switches cuts barrier overhead significantly.
**Risk:** Low-medium. Must ensure correctness of storage buffer barriers between dispatches within a single pass.

### 2.2 Subgroup operations for error reduction
**Sources:** Data structures, wgpu/Vulkan
**Effort:** ~40 lines in rasterize shader
**Impact:** Replaces 8-step shared-memory binary reduction with `subgroupAdd()`. On RTX 5090 (warp size 32): 1 subgroup intrinsic + ~3 barrier steps instead of 8. Reduces shared memory traffic by ~87%. Rasterize is 50-70% of GPU time, so this directly targets the dominant bottleneck.
**Risk:** Medium. Requires `Features::SUBGROUP` support. Needs fallback path for hardware without subgroup support.

### 2.3 Cooperative parent copy in mutate shader via shared memory
**Sources:** GPU compute
**Effort:** ~40 lines in mutate.wgsl
**Impact:** All lambda offspring read the same 16KB parent independently. Loading once into shared memory eliminates 7/8 of global memory reads at lambda=8.
**Risk:** Low. Standard shared memory pattern. Requires shared memory allocation (~16KB).

### 2.4 Workgroup size tuning
**Sources:** GPU compute, wgpu/Vulkan
**Effort:** ~20 lines + benchmarking
**Impact:** Rasterize 16x16 with 13KB shared memory limits occupancy to ~9 workgroups/SM. Testing 16x8 (128 threads, ~6.4KB shared) could double occupancy. Mutate shader wastes 87.5% of threads at lambda=8; dispatching active*lambda single-thread workgroups would eliminate waste.
**Risk:** Low. Requires empirical benchmarking to find optimal sizes.

### 2.5 Increase default chain count to 128-256
**Sources:** GPU compute
**Effort:** ~5 lines (constant change + validation)
**Impact:** Mutate and select dispatch only 16 workgroups on a 170-SM GPU (<10% utilization). Increasing to 128-256 chains dramatically improves SM utilization during these phases.
**Risk:** Low-medium. More chains = more memory. 256 chains x 16KB = 4MB for chain states. Need to verify SSBO limits for offspring buffer.

---

## Tier 3: Higher Effort, High Reward (Algorithmic / Shader)

### 3.1 Multi-resolution coarse-to-fine evaluation
**Sources:** Evolutionary algorithm
**Effort:** ~80 lines (shader stride logic + GpuParams field)
**Impact:** Evaluating at 1/4 resolution when fitness < 70% gives up to 16x throughput for early iterations. Could reduce wall-clock time to 90% fitness by 40-60%.
**Risk:** Medium. Fitness values at different resolutions aren't directly comparable — needs careful transition logic. May miss fine-grained details during coarse phase.

### 3.2 Hierarchical tile culling (polygon binning)
**Sources:** Data structures, GPU compute
**Effort:** ~150 lines (new compute pass + prefix sum + modified rasterize)
**Impact:** Current brute-force O(pixels * polygons). Binning polygons into 16x16 tiles could reduce inner-loop iterations by ~10x for 500+ small polygons.
**Risk:** Medium-high. Requires prefix-sum pass, per-tile polygon lists, extra buffer allocation. Adds pipeline complexity.

### 3.3 Cauchy-distributed mutation steps
**Sources:** Evolutionary algorithm
**Effort:** ~30 lines in mutate.wgsl
**Impact:** Heavy-tailed distribution for spatial mutations enables escaping basins of attraction. 10-30% convergence speedup in ES literature.
**Risk:** Low-medium. Well-studied technique. Needs tuning of scale parameter.

### 3.4 Fitness-proportional polygon targeting + SA-style acceptance
**Sources:** Evolutionary algorithm
**Effort:** ~40 lines in mutate.wgsl + select.wgsl
**Impact:** Biases mutation toward top-layer polygons (10-20% fewer wasted mutations). SA acceptance enables tunneling through local optima on stagnation.
**Risk:** Medium. SA temperature schedule needs tuning. Over-aggressive acceptance hurts convergence.

### 3.5 Error-guided polygon placement
**Sources:** Evolutionary algorithm
**Effort:** ~50 lines (4x4 quadrant error tracking in chain state + biased placement in mutate)
**Impact:** New polygons placed where error is highest instead of randomly. 3-10x improvement in add_polygon acceptance rate.
**Risk:** Medium. Requires extra per-chain state (64 bytes for 4x4 grid). Error grid must be maintained across iterations.

### 3.6 SoA polygon layout (split color from geometry)
**Sources:** Data structures
**Effort:** ~100 lines (buffer restructuring + all shader changes)
**Impact:** Loading only 12B vertices into shared memory (defer 4B color to hit-only) fits 1024 polygons per tile at same 12KB budget (vs 768 now). 25% more polygons per tile = fewer tiles = fewer barriers.
**Risk:** Medium-high. Touches every shader and buffer layout. Significant refactoring.

### 3.7 Precomputed bounding boxes
**Sources:** Data structures
**Effort:** ~60 lines (precompute pass + modified rasterize)
**Impact:** Avoids redundant vertex unpacking for AABB. Break-even when >50% of polygons are AABB-rejected (common for small triangles).
**Risk:** Low. Extra buffer + lightweight pass.

### 3.8 Polygon splitting on stagnation
**Sources:** Evolutionary algorithm
**Effort:** ~80 lines
**Impact:** Subdivide large triangles into 3 sub-triangles at centroid on stagnation. Introduces new degrees of freedom where they can help.
**Risk:** Medium. Changes polygon count dynamically, complicating state management.

---

## Expert Analysis Documents

| Expert | Document |
|--------|----------|
| GPU Compute | [gpu-compute-optimization.md](gpu-compute-optimization.md) |
| Evolutionary Algorithm | [evolutionary-algorithm.md](evolutionary-algorithm.md) |
| CPU-GPU Hybrid | [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md) |
| Data Structures | [data-structures-algorithms.md](data-structures-algorithms.md) |
| wgpu/Vulkan | [wgpu-vulkan-performance.md](wgpu-vulkan-performance.md) |
