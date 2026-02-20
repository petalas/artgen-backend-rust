# Data Structures & Algorithms: GPU Evolution Pipeline

Performance analysis of the wgpu compute-shader evolution pipeline. Focuses on data layout, memory access patterns, algorithmic efficiency, and concrete optimization opportunities.

**Scope**: GPU path only (`src/gpu_evolver/`, `src/shaders/`). CPU rasterizer is out of scope.

**Already optimized** (not re-discussed): per-polygon AABB early-out, L1 error metric, fused rasterize+error_reduce, 48-byte GpuPolygon alignment.

---

## 1. Memory Layout: AoS vs SoA

### Current: Array of Structures (AoS)

```
// buffers.rs — GpuDrawingState (48,032 bytes per chain)
pub struct GpuDrawingState {
    polygon_count: u32,        // 4 bytes
    fitness_bits: u32,         // 4 bytes
    _pad: [u32; 2],            // 8 bytes
    rng_state: [u32; 4],       // 16 bytes
    polygons: [GpuPolygon; 1000],  // 48,000 bytes
}
```

Each `GpuPolygon` is 48 bytes containing interleaved geometry (v0, v1, v2) and color (RGBA):

```
// buffers.rs — GpuPolygon (48 bytes)
pub struct GpuPolygon {
    color: [f32; 4],   // 16 bytes
    v0: [f32; 2],      // 8 bytes
    v1: [f32; 2],      // 8 bytes
    v2: [f32; 2],      // 8 bytes
    _pad: [f32; 2],    // 8 bytes (wasted)
}
```

### Analysis: Why AoS Is Acceptable Here

The typical argument for SoA on GPU is that threads in a warp/subgroup need the same field from different elements, enabling coalesced loads. However, in this pipeline:

- **Rasterize shader** (`rasterize_error.wgsl` line 116): Each thread loads `shared_polys[local_idx] = working_states[chain_id].polygons[load_idx]` — one thread loads one *entire* polygon cooperatively into shared memory. Then all 256 threads read the same polygon from shared memory (broadcast, not coalesced). The cooperative load is a single 48-byte read per thread — already a single cache line (or two, depending on alignment). SoA would not help because each thread needs all fields of its assigned polygon.

- **Mutate shader** (`mutate.wgsl` line 268): Workgroup size is 1, so there is no cross-thread access pattern to coalesce. Each thread reads/writes its own chain's polygons sequentially. SoA would add complexity for zero benefit.

- **Select/migrate shaders**: Workgroup size 1, copying entire polygons between chains.

**Verdict**: AoS is the correct layout for this access pattern. Do not convert to SoA.

### The Real Waste: 8 Bytes of Padding Per Polygon

Each `GpuPolygon` has `_pad: [f32; 2]` (8 bytes, 16.7% of the struct). Over 1000 polygons, that's **8,000 bytes per chain wasted** (8 KB), or **4 MB across 512 chains**. This padding exists to reach a 48-byte stride, but the reason for 48 is not a WGSL alignment requirement — WGSL only requires structs to align to their largest member (here `vec4<f32>` = 16 bytes), meaning 40 bytes would need padding to 48 (next multiple of 16). The padding is mandatory given the current field layout.

**Optimization opportunity**: If color were stored as `u8x4` packed into a single `u32` (see Section 6), the struct could shrink from 48 to 32 bytes (a power of 2, inherently 16-byte aligned with no padding needed). This would reduce `GpuDrawingState` from 48,032 to 32,032 bytes, a **33% reduction** in per-chain memory, enabling more chains or reducing cache pressure.

---

## 2. Rasterization Algorithm

### Current: Per-Pixel Half-Space Test with Tiled Polygon Prefetch

```wgsl
// rasterize_error.wgsl — workgroup_size(16, 16, 1)
// Dispatch: (W/16, H/16, K) where K = chain count

// Cooperative load: 256 threads load 256 polygons into shared memory
shared_polys[local_idx] = working_states[chain_id].polygons[load_idx];

// Then each thread tests its pixel against all 256 polygons
for (var i = 0u; i < tile_end; i++) {
    let poly = shared_polys[i];
    // AABB test, then 3 edge functions, then alpha blend
}
```

This is a **screen-space parallel** approach: one thread per pixel, iterating over all polygons. The AABB early-out (`rasterize_error.wgsl` lines 126-133) skips ~97% of tests, so the inner loop is efficient for small polygons.

### Alternative: Tile-Based Rasterization (Not Recommended Yet)

A tile-based approach would assign polygon subsets to screen tiles. However:

- Current tile size (16x16 = 256 pixels) already matches one workgroup.
- With AABB early-out, most polygons are skipped for most tiles. A hierarchical structure would help only if polygon counts are very high (>500) AND polygons are large relative to the image.
- The cooperative shared-memory prefetch already ensures polygon data is loaded once and reused by all 256 threads.

**Verdict**: The current algorithm is appropriate. The main bottleneck is likely memory bandwidth for the polygon loads, not ALU. Focus optimization effort on reducing polygon data size (Section 6).

### Potential Improvement: Precomputed AABBs

Currently each pixel thread recomputes the AABB from vertices every time a polygon is tested (`rasterize_error.wgsl` lines 126-129):

```wgsl
let bb_min_x = min(poly.v0.x, min(poly.v1.x, poly.v2.x));
let bb_max_x = max(poly.v0.x, max(poly.v1.x, poly.v2.x));
let bb_min_y = min(poly.v0.y, min(poly.v1.y, poly.v2.y));
let bb_max_y = max(poly.v0.y, max(poly.v1.y, poly.v2.y));
```

That's 8 `min`/`max` operations per polygon per pixel (256 threads x up to 1000 polygons). Since the AABB is the same for all pixels in a workgroup (and indeed for all workgroups in the same chain), it could be computed once during the cooperative load phase and stored in shared memory alongside the polygon.

**Concrete change**: Store precomputed AABB in the `_pad` field of the polygon (or in a separate `shared_aabb` array). During the cooperative load, one thread computes the AABB for each polygon. This saves 8 `min`/`max` ops per pixel per polygon.

However, note: the `_pad` field is `vec2<f32>` (8 bytes) — enough for `bb_min` (2 floats) but not `bb_max` (needs 4 floats total). A separate `shared_aabb: array<vec4<f32>, 256>` (4 KB) would work, bringing total shared memory to 12,288 + 1,024 + 4,096 = 17,408 bytes — well within the 16,384 byte minimum guaranteed by WebGPU. Wait — 17 KB exceeds the 16 KB minimum. Check your adapter's actual limit; many GPUs support 32 KB or 48 KB shared memory per workgroup. If limited to 16 KB, you could reduce the polygon tile size from 256 to 192 to fit both arrays.

**Expected savings**: Marginal per polygon (GPUs are fast at min/max), but it adds up: at 100 active polygons x 256 threads x 8 ops = 204,800 saved operations per workgroup. Probably a 1-3% improvement.

---

## 3. Error Reduction Algorithm

### Current: Binary Tree Reduction in Shared Memory

```wgsl
// rasterize_error.wgsl lines 183-191
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

This is a standard parallel reduction: 8 steps for 256 elements, each step halves active threads. The pattern is textbook-correct with `log2(256) = 8` barriers.

### Issue: Warp Divergence in Final Steps

In the last 5 steps (stride <= 16), only the first 1-32 threads are active. On GPUs with 32-wide warps (NVIDIA) or 64-wide waves (AMD), this means:

- Stride 16: 16/32 or 16/64 threads active (50% or 25% utilization)
- Stride 1: 1/32 or 1/64 threads active (3% or 1.5% utilization)

Each step still pays for a `workgroupBarrier()`. For subgroup sizes >= 32, the last 5 steps could use **subgroup operations** (`subgroupAdd`) instead of shared memory, eliminating barriers and shared memory bank conflicts.

**Concrete change** (requires `enable subgroups;` in WGSL, supported in wgpu with the `SUBGROUP` feature):

```wgsl
// After stride reaches subgroup_size, switch to subgroup reduction
shared_errors[local_idx] = pixel_error;
workgroupBarrier();

// Reduce in shared memory until we reach subgroup-sized chunks
var stride = 128u;
while stride > subgroup_size {
    if local_idx < stride {
        shared_errors[local_idx] += shared_errors[local_idx + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
}

// Final reduction within each subgroup (no barriers needed)
var val = shared_errors[local_idx];
val = subgroupAdd(val);

// Only lane 0 of subgroup 0 writes the result
if local_idx == 0u {
    atomicAdd(&error_accumulators[chain_id], val);
}
```

**Expected savings**: Eliminates ~5 `workgroupBarrier()` calls. Real-world impact is modest (perhaps 2-5%) because the reduction is fast relative to the rasterization loop, but it's a clean win with no downsides.

**Portability note**: `subgroupAdd` is in the WebGPU subgroups proposal and supported in wgpu v22+ behind a feature flag. Falls back to the current approach on unsupported hardware.

---

## 4. Mutate Shader: Single-Threaded Bottleneck

### Current Architecture

```wgsl
// mutate.wgsl line 267
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_id = gid.x;
    // ... copies entire chain state, loops over all polygons
```

Each chain is processed by a **single thread** in a workgroup of size 1. For 512 chains, this dispatches 512 workgroups of 1 thread each. The per-chain work includes:

1. Copy all polygons (up to 1000) from `chain_states` to `working_states` with alpha clamping
2. Mutation loop (up to 1000 attempts until dirty)
3. Per-polygon mutations: iterate all polygons, test ~15 probability checks each

### Problems

1. **GPU occupancy**: Workgroup size 1 means the GPU scheduler has only 1 thread per workgroup to schedule. Modern GPUs have warps/waves of 32-64 threads. A workgroup of 1 means 31-63 lanes are idle per scheduled warp. With 512 chains, only 512 threads are active across the entire GPU — far below the thousands needed for full occupancy.

2. **Memory bandwidth**: The copy phase reads/writes 48,032 bytes per chain, sequentially. With 512 chains, that's ~47 MB of memory traffic. The polygon copy loop (`mutate.wgsl` lines 318-322) writes one polygon per loop iteration — no vectorization.

3. **Serialized polygon iteration**: The mutation loop iterates polygons 0..count sequentially within a single thread. This is inherently serial but mutation order matters (each mutation depends on previous state).

### Optimization Opportunity: Parallelize the Copy Phase

The alpha-clamped copy (`mutate.wgsl` lines 317-322) could be done as a separate compute pass with a larger workgroup, where each thread copies a few polygons. This separates the embarrassingly-parallel copy from the inherently-serial mutation.

However, the real bottleneck is that mutation is sequential per chain by design (each decision depends on RNG state and previous mutations). Increasing workgroup size would not help the mutation loop.

**Practical improvement**: Increase workgroup size to 64 and have only thread 0 do actual work. This improves occupancy for the GPU scheduler even though extra threads are idle — the scheduler can overlap memory latency better with 64 threads per workgroup.

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
    if local_idx != 0u { return; }
    let chain_id = gid.x / 64u;
    // ... rest of mutation logic
}
```

Dispatch becomes `dispatch_workgroups(ceil(active / 64), 1, 1)` with `chain_id` computed from the global invocation ID. This is a common GPU pattern for latency hiding.

**Expected impact**: Moderate. The mutate pass is already relatively cheap compared to rasterization. Profile first to confirm it's a bottleneck.

---

## 5. Select & Migrate: Full-Chain Copy Cost

### Current: Copy All Polygons on Accept

```wgsl
// select.wgsl lines 111-122
if fitness > current_fitness {
    // Copy working → chain_state (header + all polygons)
    chain_states[chain_id].polygon_count = working_states[chain_id].polygon_count;
    chain_states[chain_id].fitness_bits = fitness_bits;
    let pc = working_states[chain_id].polygon_count;
    for (var i = 0u; i < pc; i++) {
        chain_states[chain_id].polygons[i] = working_states[chain_id].polygons[i];
    }
}
```

When a candidate is accepted, the entire polygon array (up to 48,000 bytes) is copied in a loop within a workgroup of size 1. Similarly, migration (`select.wgsl` lines 147-155) copies the full polygon set.

### Optimization: Double-Buffer Pointer Swap

Instead of copying polygon data, maintain two buffers (A and B) per chain and swap which is "current best" vs "working candidate" using a single u32 flag. The select shader would just flip the flag instead of copying 48 KB.

**Trade-off**: This requires restructuring the buffer layout. Currently `chain_states` and `working_states` are separate buffers. A pointer-swap approach would need either:
- An indirection buffer (`chain_active_buffer: array<u32>`) and both buffers accessible from all shaders
- Or a single buffer with 2x the entries and an index per chain

This is a significant architectural change with non-trivial complexity. **Recommended only if profiling shows select/migrate copy is a significant fraction of total time.** Current GPU timestamp profiling (`PassTimings`) already captures per-pass timing — check `select` percentage first.

---

## 6. Compressed Polygon Representation

### Current: f32 Everything (48 bytes per polygon)

```
color: [f32; 4]  = 16 bytes  (RGBA, each 0.0-1.0)
v0:    [f32; 2]  =  8 bytes  (xy, each 0.0-1.0)
v1:    [f32; 2]  =  8 bytes
v2:    [f32; 2]  =  8 bytes
_pad:  [f32; 2]  =  8 bytes
Total:             48 bytes
```

### Proposal: Quantized 32-Byte Polygon

All values are in [0.0, 1.0]. Using `u16` (65,536 steps) gives sub-pixel precision for images up to 32K resolution. Color channels are originally u8 (256 levels) on the CPU side and only converted to f32 for the GPU — a `u8` roundtrip is lossless.

```
color: u8x4 packed as u32    =  4 bytes  (exact match to CPU Color struct)
v0:    [u16; 2] packed as u32 =  4 bytes  (65536 steps, ~0.002% precision)
v1:    [u16; 2] packed as u32 =  4 bytes
v2:    [u16; 2] packed as u32 =  4 bytes
Total:                          16 bytes
```

At 16 bytes, the struct aligns naturally to 16-byte boundaries (WGSL vec4 alignment). With proper padding to 16 bytes, this fits in a single `vec4<u32>`:

```wgsl
struct PackedPolygon {
    data: vec4<u32>,  // [color_rgba8, v0_xy16, v1_xy16, v2_xy16]
}
// Unpack in shader:
fn unpack_polygon(p: PackedPolygon) -> Polygon {
    let c = unpack4x8unorm(p.data.x);  // WGSL built-in
    let v0 = unpack2x16unorm(p.data.y);
    let v1 = unpack2x16unorm(p.data.z);
    let v2 = unpack2x16unorm(p.data.w);
    // ...
}
```

**Impact**:
- Per-chain state: 32 (header) + 1000 * 16 = **16,032 bytes** (down from 48,032 — **67% reduction**)
- 512 chains: **~15.6 MB** total for chain+working states (down from ~47 MB)
- 3x more polygons fit in shared memory tile (768 vs 256 per 12 KB)
- 3x less memory bandwidth for polygon loads
- L1/L2 cache hit rates improve dramatically

**Trade-off**: Unpacking adds ALU work (4 unpack operations per polygon per pixel). But `unpack4x8unorm` and `unpack2x16unorm` are single-cycle instructions on modern GPUs. The memory bandwidth savings vastly outweigh the unpack cost.

**Mutation shader impact**: Mutations currently operate in f32 space. With quantized storage, the mutate shader would unpack to f32, mutate, then repack. This adds ~8 pack/unpack ops per polygon touched — negligible given mutation only touches 1-3 polygons per iteration.

**Precision consideration**: For a 512x512 image, u16 gives 65536/512 = 128 sub-pixel positions per pixel — far more than needed. For color, the CPU side already works in u8, so no precision is lost.

---

## 7. Reducing Per-Chain State to Fit More Chains

### Current Memory Budget

| Component | Per-chain | 512 chains |
|-----------|-----------|------------|
| chain_states (GpuDrawingState) | 48,032 B | 23.4 MB |
| working_states (GpuDrawingState) | 48,032 B | 23.4 MB |
| error_accumulators (u32) | 4 B | 2 KB |
| fitness_packed (u32) | 4 B | 2 KB |
| **Total per-chain** | **96,072 B** | **~46.8 MB** |

Plus reference image (512x512x4 = 1 MB) and control flags (16 B).

### With Quantized Polygons (Section 6)

| Component | Per-chain | 512 chains |
|-----------|-----------|------------|
| chain_states | 16,032 B | 7.8 MB |
| working_states | 16,032 B | 7.8 MB |
| **Total per-chain** | **32,068 B** | **~15.6 MB** |

This alone would allow **~3x more chains** within the same memory budget, or the same chain count with dramatically better cache utilization.

### Further: Reduce MAX_POLYGONS_PER_IMAGE

`MAX_POLYGONS_PER_IMAGE = 1000` (`settings.rs` line 36) allocates space for 1000 polygons per chain regardless of how many are actually used. The `polygon_count` field tracks the actual count. If typical drawings use 100-300 polygons, 70-90% of the polygon array is wasted zeros.

**Dynamic allocation is impractical on GPU** (no malloc), but a **configurable compile-time maximum** (e.g., 256 or 512) that matches actual usage would save significant memory. This is a configuration change, not an algorithm change — just reduce the constant and recompile.

With MAX_POLYGONS = 256 and quantized polygons: 32 + 256*16 = 4,128 bytes per chain. At 512 chains, that's only ~4 MB for both chain+working states.

---

## 8. Shared Memory Utilization

### Current Usage in Rasterize Shader

```wgsl
var<workgroup> shared_polys: array<Polygon, 256>;   // 256 * 48 = 12,288 bytes
var<workgroup> shared_errors: array<u32, 256>;       // 256 * 4  = 1,024 bytes
// Total: 13,312 bytes
```

WebGPU guarantees minimum 16,384 bytes shared memory per workgroup. Most desktop GPUs offer 32-48 KB. Current usage is **13,312 / 16,384 = 81%** of guaranteed minimum.

### With Quantized Polygons

```wgsl
var<workgroup> shared_polys: array<PackedPolygon, 256>;  // 256 * 16 = 4,096 bytes
var<workgroup> shared_errors: array<u32, 256>;           // 1,024 bytes
// Total: 5,120 bytes (31% of minimum)
```

This opens up two options:
1. **Larger tile size**: Load 768 polygons per tile instead of 256, reducing the number of tile iterations by 3x (fewer barriers, better instruction-level parallelism).
2. **More occupancy**: Smaller shared memory footprint means more workgroups can run concurrently on each Compute Unit, improving latency hiding.

---

## 9. Reference Image Optimization

### Current: u32 Per Pixel (4 bytes)

```wgsl
// rasterize_error.wgsl lines 165-169
let reference = reference_image[ref_idx];
let refr = f32(reference & 0xFFu);
let refg = f32((reference >> 8u) & 0xFFu);
let refb = f32((reference >> 16u) & 0xFFu);
```

The reference image is stored as `array<u32>` where each u32 packs RGBA as bytes. At 512x512, this is 1 MB. This is already efficient — no optimization needed here.

However, the reference image access pattern is **coherent** across chains: all chains access the same pixel at the same coordinates. If multiple chains' workgroups land on the same CU, L2 cache reuse is high. This is already implicitly exploited by the dispatch pattern `(wg_x, wg_y, chain_id)`.

---

## 10. Mutation Shader: Polygon Insert/Remove Shifts

### Current: O(n) Array Shifts

```wgsl
// mutate.wgsl lines 362-366 — insert polygon
let insert_idx = rand_u32(&rng, max(count, 1u));
for (var j = count; j > insert_idx; j--) {
    working_states[chain_id].polygons[j] = working_states[chain_id].polygons[j - 1u];
}

// mutate.wgsl lines 374-378 — remove polygon
let remove_idx = rand_u32(&rng, count);
for (var j = remove_idx; j < count - 1u; j++) {
    working_states[chain_id].polygons[j] = working_states[chain_id].polygons[j + 1u];
}
```

Inserting or removing a polygon shifts up to 999 elements (48 bytes each) = up to 47,952 bytes of memory writes. This happens in the single-threaded mutate shader.

### Optimization: Swap-Remove + Append

Instead of maintaining polygon order during insert/remove:

- **Remove**: Swap the removed polygon with the last polygon, then decrement count. O(1) instead of O(n).
- **Insert**: Append at the end (index `count`), then swap with the desired position. O(1) instead of O(n).

**Trade-off**: Polygon order matters for rendering (painter's algorithm — later polygons are drawn on top). Swap-remove changes the z-order of one polygon per operation. However, the `reorder_polygon_prob` mutation already randomly swaps polygons, so the algorithm explicitly explores different orderings. A single reorder per add/remove is likely insignificant compared to the dedicated reorder mutation.

**Alternative**: Use the current shift only occasionally (or for a small fraction of cases) and default to swap-remove. Or mark "order matters" as a separate, less-frequent mutation.

**Expected savings**: Eliminates the worst-case 48 KB memcpy per add/remove. For chains with 500+ polygons, this is meaningful.

---

## 11. Workgroup Barrier Count in Rasterize Shader

### Current: 2 Barriers Per Tile

```wgsl
for (var tile = 0u; tile < tile_count; tile++) {
    // Cooperative load
    shared_polys[local_idx] = working_states[chain_id].polygons[load_idx];
    workgroupBarrier();    // Barrier 1: ensure all loads complete

    // Process all polygons in tile
    for (var i = 0u; i < tile_end; i++) { ... }
    workgroupBarrier();    // Barrier 2: prevent next tile overwriting shared_polys
}
```

For 1000 polygons, there are `ceil(1000/256) = 4` tiles, so **8 barriers** during rasterization plus **8 barriers** for the reduction phase = **16 total barriers**. This is already quite good.

With quantized polygons and a larger tile size (e.g., 768), there would be `ceil(1000/768) = 2` tiles = **4 barriers** for rasterization — a 50% reduction in barrier overhead.

---

## Summary of Recommendations (Priority Order)

| # | Optimization | Effort | Expected Impact | Risk |
|---|-------------|--------|----------------|------|
| 1 | **Quantized 16-byte polygons** (Section 6) | Medium | High (3x less memory, 3x better cache, 3x larger tiles) | Low — precision is sufficient |
| 2 | **Swap-remove for insert/delete** (Section 10) | Low | Medium (eliminates O(n) shifts in mutate) | Low — order already explored by reorder mutation |
| 3 | **Precomputed AABBs in shared memory** (Section 2) | Low | Low-Medium (saves 8 min/max per polygon per pixel) | None |
| 4 | **Subgroup reduction** (Section 3) | Low | Low (eliminates ~5 barriers) | Medium — requires feature detection |
| 5 | **Reduce MAX_POLYGONS_PER_IMAGE** (Section 7) | Trivial | Medium (if actual usage is well below 1000) | Requires checking typical polygon counts |
| 6 | **Increase mutate workgroup size** (Section 4) | Low | Low (better occupancy/latency hiding) | None |
| 7 | **Double-buffer pointer swap** (Section 5) | High | Medium (eliminates 48 KB copy on accept) | Architectural complexity |
