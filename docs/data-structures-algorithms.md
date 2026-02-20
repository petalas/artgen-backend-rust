# Data Structures & Algorithms Analysis

Analysis of GPU compute evolution pipeline. Focus: what is next after the already-completed optimizations (ring migration, AABB culling, L1 error, chain count capping, batch size 50).

---

## 1. GPU Struct Layout & Padding Waste

### Current: GpuPolygon = 48 bytes

```
color: vec4<f32>  (16 bytes)  — offset 0
v0:    vec2<f32>  ( 8 bytes)  — offset 16
v1:    vec2<f32>  ( 8 bytes)  — offset 24
v2:    vec2<f32>  ( 8 bytes)  — offset 32
_pad:  vec2<f32>  ( 8 bytes)  — offset 40  ← WASTE
```

**8 bytes of padding per polygon (16.7% waste).** With 1000 polygons per chain and 512 chains, this is `8 * 1000 * 512 * 2 = 8.2 MB` of wasted GPU memory across chain_states + working_states, and wasted bandwidth every time polygons are read or written.

### Fix: Pack to 40 bytes

Removing `_pad` requires changing the struct stride. WGSL `array<Polygon, 1000>` with a 40-byte struct would still work if the struct alignment is adjusted. However, WGSL requires struct alignment to be a multiple of the largest member's alignment (vec4<f32> = 16). So the struct will be padded to 48 bytes by the WGSL compiler regardless.

**Verdict: No savings possible without restructuring.** The 16-byte alignment of `vec4<f32> color` forces the struct to round up to 48. To fix this, you would need to split color out into a separate array (SoA), or pack color into fewer bytes (see Section 5).

### Current: GpuDrawingState = 48,032 bytes

```
polygon_count:  4 bytes
fitness_bits:   4 bytes
_pad:           8 bytes  ← alignment padding
rng_state:     16 bytes
polygons:   48,000 bytes (1000 * 48)
```

Header overhead is 32 bytes / 48,032 = 0.07%, negligible.

**Total memory for chain states:** `48,032 * 512 * 2 = 47.0 MB` (chain_states + working_states).

---

## 2. Memory Bandwidth Analysis

### Per-iteration memory traffic (512 chains, 384x384 image)

| Pass | Read | Write | Total |
|------|------|-------|-------|
| Mutate | 48,032 * 512 = 23.5 MB (chain_states) | 48,032 * 512 = 23.5 MB (working_states) | 47.0 MB |
| Rasterize | 48,032 * 512 = 23.5 MB (working_states) | 384*384*4*512 = 301.0 MB (render_targets) | 324.5 MB |
| Error Reduce | 301.0 MB (render_targets) + 0.6 MB (reference) | ~0 MB (atomic adds) | 301.6 MB |
| Select | 23.5 MB (working) + 23.5 MB (chain) + ~0 (accum) | 23.5 MB (chain) | 70.5 MB |
| **Total** | | | **~744 MB/iteration** |

At 50 iterations per batch, that is **~37.2 GB per batch submission**. At typical GPU memory bandwidth of 200-500 GB/s, this bounds throughput to roughly 5-13 batches/second (250-650 iterations/sec).

### Key insight: Rasterize + Error Reduce dominate

Rasterize writes 301 MB and Error Reduce reads 301 MB — together they are 81% of total bandwidth. The render_targets buffer is the single biggest bandwidth consumer.

---

## 3. Fused Rasterize + Error: Eliminate render_targets buffer

**The highest-impact optimization available.** Currently:

1. `rasterize.wgsl` composites all polygons, writes packed u32 to `render_targets[pixel_idx]`
2. `error_reduce.wgsl` reads `render_targets[pixel_idx]`, reads `reference_image[ref_idx]`, computes L1 diff

These two passes can be fused into a single pass. Each thread composites all polygons at its pixel (same as now), but instead of writing the result to render_targets, it immediately computes the L1 error against the reference image in registers, then reduces via shared memory and atomicAdd.

**Savings:**
- Eliminates 301 MB write (rasterize) + 301 MB read (error_reduce) = **602 MB/iteration**
- Eliminates the `render_targets` buffer entirely: **301 MB GPU memory freed**
- Reduces dispatch overhead (one fewer compute pass per iteration)
- Total bandwidth drops from ~744 MB to ~142 MB/iteration = **81% reduction**

**Implementation complexity:** Low. The rasterize shader already computes the final pixel color in registers (`r`, `g`, `b`). Instead of packing and writing to `render_targets`, compute the error inline:

```wgsl
// After compositing all polygons (existing code), instead of writing render_targets:
let ref_pixel = reference_image[py * w + px];
let ref_r = f32(ref_pixel & 0xFFu);
let ref_g = f32((ref_pixel >> 8u) & 0xFFu);
let ref_b = f32((ref_pixel >> 16u) & 0xFFu);
let pixel_error = u32(abs(r - ref_r) + abs(g - ref_g) + abs(b - ref_b));
// Then workgroup reduce + atomicAdd (same as current error_reduce.wgsl)
```

**Risk:** None. Mathematically identical. The render_targets buffer is not read by any other pass.

---

## 4. Mutate Pass: Excessive Global Memory Traffic

### Problem: Full state copy every iteration

The mutate shader copies the entire `chain_states[chain_id]` (48 KB) to `working_states[chain_id]`, then applies mutations. Most iterations mutate only 1-3 polygons out of hundreds.

**Current bandwidth:** 47.0 MB per iteration (23.5 MB read + 23.5 MB write).

### Option A: Copy-on-write with dirty tracking

Track which polygon index was mutated. In the select pass, only copy changed polygons back. This is complex and may not save much due to the copy-forward still being needed.

### Option B: In-place mutation with undo

Instead of maintaining chain_states + working_states as separate buffers, mutate chain_states in-place and store a small "undo log" (just the polygon index + old polygon data). If the mutation is rejected, replay the undo. This halves the state buffer memory and eliminates the full copy.

**Savings:** Halves mutate-pass bandwidth (23.5 MB saved), eliminates working_states buffer (23.5 MB GPU memory). The undo log would be tiny: one polygon index (4 bytes) + one GpuPolygon (48 bytes) = 52 bytes per chain = 26 KB total.

**Risk:** Medium. Requires restructuring the select pass. Add/remove polygon mutations modify multiple polygon slots (shift operations), so the undo log needs to handle those cases. Most mutations touch 1 polygon, so the common case is simple.

### Option C: Store only active polygons + metadata

Currently 1000 polygon slots are allocated per chain even though `polygon_count` is typically 150-300. A variable-length representation could save memory, but GPU compute shaders don't handle variable-length structs well. Not recommended.

---

## 5. Smaller Data Types: Quantized Polygon Representation

### Color: f32x4 (16 bytes) vs u8x4 (4 bytes)

Color components are fundamentally 8-bit values (0-255). Storing them as 4x f32 wastes 12 bytes per polygon. Switching to packed u8x4 (a single u32) would:

- Save 12 bytes per polygon = 12 * 1000 * 512 * 2 = **11.7 MB**
- Reduce polygon size from 48 to 36 bytes (but WGSL alignment would round to 36... wait, let's check)

New layout without f32 color:
```
color_packed: u32      ( 4 bytes)
v0: vec2<f32>          ( 8 bytes)
v1: vec2<f32>          ( 8 bytes)
v2: vec2<f32>          ( 8 bytes)
                       = 28 bytes → rounds to 32 (vec2<f32> alignment = 8)
```

That is 32 bytes vs 48 bytes = **33% reduction in polygon data size**. Per-chain state drops from 48,032 to 32,032 bytes.

**Total state memory:** 32,032 * 512 * 2 = 31.3 MB (vs 47.0 MB) = **33% reduction**.

The rasterize shader would unpack: `let r = f32(color_packed & 0xFFu); ...` — trivial ALU cost.

The mutate shader would need to pack/unpack for mutations. Micro-adjustments (+/- 1) become integer add/clamp on bytes — actually cheaper than the current f32 arithmetic.

**Risk:** Low. Integer color arithmetic is well-defined. The main concern is that mutation probabilities and color randomization currently use `rand_f32()` which maps naturally to f32 — but converting `rand_f32() * 255.0` to u8 is trivial.

### Vertex coordinates: f32 vs f16

Vertex positions are normalized 0.0-1.0 with micro-adjustment delta of 0.01 (1% of range). f16 has ~0.1% precision in [0,1], which is sufficient for the 384x384 target (1/384 = 0.26%). However, WGSL f16 support requires the `f16` extension which is not universally available. Not recommended until hardware support is broader.

---

## 6. Rasterization Algorithm Alternatives

### Current: Brute-force per-pixel-per-polygon

Each pixel thread iterates over ALL polygons (up to 1000), performing AABB test + 3 edge functions. For a 384x384 image with 300 polygons:

- 147,456 pixels * 300 polygons = **44.2 million triangle tests per chain**
- 44.2M * 512 chains = **22.6 billion triangle tests per iteration**

The AABB cull helps but still requires loading each polygon's data to check bounds.

### Alternative A: Tile-based polygon binning (recommended)

Pre-bin polygons into screen-space tiles (e.g., 8x8 or 16x16). A separate pass computes which polygons overlap each tile and writes a per-tile polygon list. Rasterize threads only iterate over polygons in their tile.

For typical polygon distributions where most polygons cover <5% of the image, this reduces per-pixel polygon iteration from ~300 to ~15-30 on average, a **10-20x reduction in rasterize ALU**.

**Implementation:** Add a binning pass dispatched as (num_tiles, chain_count, 1) with workgroup_size(1). Each thread tests each polygon's AABB against the tile and appends the polygon index to a per-tile list stored in a buffer. The rasterize pass then reads the tile's polygon list instead of the full polygon array.

**Memory cost:** Per-tile list needs max ~1000 indices * ~50x50 tiles * 512 chains = not feasible at this scale. Alternative: use a bitmask (1000 polygons = 32 u32s per tile per chain). At 48x48 tiles (384/8): `32 * 4 * 48 * 48 * 512 = 144 MB`. Too expensive.

**Simpler variant: coarse Y-sorting.** Sort polygons by AABB min_y and max_y. Each pixel row only iterates polygons whose Y range overlaps. This requires a sorted polygon buffer (already available) and two binary searches per pixel row. Moderate complexity, moderate gains.

**Verdict:** Tile binning is attractive but memory overhead for per-chain tile lists is prohibitive at 512 chains. The fused rasterize+error pass (Section 3) provides better bang-for-buck without algorithmic changes to rasterization.

### Alternative B: Hierarchical rasterization (not recommended for GPU compute)

The CPU path uses 8x8 block hierarchical rasterization (test block corners, fill entire block or test per-pixel). This doesn't map well to GPU compute because the workgroup is already 8x8 — the hierarchy is implicit in the dispatch granularity.

---

## 7. Error Reduction Algorithm

### Current implementation

```wgsl
var<workgroup> shared_errors: array<u32, 64>;  // 8x8 = 64 threads
// Binary reduction: 64 → 32 → 16 → 8 → 4 → 2 → 1
// Thread 0 atomicAdd to per-chain accumulator
```

This is a standard tree reduction. For a 384x384 image, there are `48 * 48 = 2,304` workgroups per chain, each contributing one atomicAdd.

### Analysis

The reduction within each workgroup is efficient (6 steps for 64 threads, no bank conflicts on shared memory). The final atomicAdd bottleneck: 2,304 atomic operations per chain is fine — atomic throughput on modern GPUs is millions/sec.

**One issue:** After the fused rasterize+error optimization (Section 3), the reduction will be part of the rasterize shader. The shared_errors array and barrier-based reduction will add negligible overhead since the threads already need a barrier before the workgroup ends anyway.

### Possible improvement: Two-level reduction

Instead of 2,304 atomicAdds per chain, use a two-level scheme: each workgroup writes to an intermediate buffer, then a second small pass sums those. This trades buffer space for reduced atomic contention. At 2,304 atomics per chain this is not a bottleneck — **not recommended**.

---

## 8. Select/Migrate Pass: Unnecessary Full Copy

### Problem

When a candidate is accepted in `select_main`, the shader copies all `polygon_count` polygons from working_states to chain_states in a serial loop:

```wgsl
for (var i = 0u; i < pc; i++) {
    chain_states[chain_id].polygons[i] = working_states[chain_id].polygons[i];
}
```

With a single thread per chain and 300 polygons * 48 bytes = 14.4 KB per copy, this is a sequential memory operation. Similarly, `migrate_main` copies all polygons from neighbor to self.

### Fix: Pointer swap instead of data copy

Instead of two buffers (chain_states, working_states) with data copying, use an indirection table. Each chain has a "current" index and "working" index into a pool of drawing states. Select just swaps the indices. This eliminates the O(polygon_count) copy entirely.

**Implementation:** Add a `chain_pointers` buffer of `u32[chain_count * 2]` — each chain has (current_idx, working_idx). Mutate reads from `states[current_idx]`, writes to `states[working_idx]`. Select either swaps or keeps the pointers. All shaders index through the pointer table.

**Savings:** Eliminates 14.4 KB * 512 * acceptance_rate per iteration of copy. If acceptance rate is ~10%, saves ~750 KB/iteration. The indirection adds one extra memory read per buffer access (4 bytes) — negligible.

**Risk:** Medium. All four shaders need to be updated to use indirect indexing. The migration pass becomes a pointer swap instead of a data copy (big win for migration). However, the mutate pass still needs to copy data to produce the candidate — the working copy needs to start from the current state.

**Revised approach:** Use double-buffering with ping-pong. Even iterations: chain_states = current, working_states = candidate. Odd iterations: swap roles. Select pass writes a "swap bit" instead of copying data. Mutate pass copies from whichever buffer is "current" based on the swap bit.

This still requires a full copy in mutate, so the savings come only from select and migrate. Net benefit is modest — **low priority**.

---

## 9. Incremental / Delta Evaluation

### The big idea

When a mutation changes only 1 polygon, only the pixels covered by that polygon (and its old position) need to be re-rendered and re-evaluated. Instead of full rasterization + full error computation, do:

1. Record which polygon was mutated and its old AABB
2. For pixels in (old_AABB union new_AABB), re-composite all polygons and recompute error
3. Adjust the total error by the delta

### Analysis

For a small polygon covering 3% of the image: instead of evaluating 147,456 pixels, evaluate ~4,400 pixels. That is **33x fewer pixels** for the rasterize+error pass.

### Why it is hard on GPU

- Must know which polygon changed (need mutation tracking from mutate pass)
- Must re-composite ALL polygons at affected pixels (not just the changed one) because of alpha blending order dependency — polygons on top affect the final color even if they didn't change
- The "affected region" varies per mutation, making workgroup dispatch irregular
- Add/remove polygon mutations affect the entire image (z-order changes)

### Feasible variant: Cached per-polygon error contribution

Not feasible because alpha blending is order-dependent — a polygon's contribution depends on all polygons below it.

### Feasible variant: Scanline-level caching with dirty rectangles

Maintain a per-chain "cached error" value. After mutation, compute error only in the dirty rectangle, subtract old partial error for that rectangle, add new partial error. This requires storing per-tile or per-scanline error subtotals.

**Memory:** Per-chain, 48 tiles * 48 tiles = 2,304 u32 subtotals * 512 chains = 4.5 MB. Feasible.

**Complexity:** High. Requires a pre-pass to compute dirty rectangle, dynamic dispatch for the partial re-render, and bookkeeping for tile error subtotals. The bookkeeping is fragile (rounding errors accumulate over thousands of iterations).

**Verdict:** High reward but high complexity. Best saved for a v2 architecture. The fused rasterize+error (Section 3) should be done first as it is much simpler and still eliminates 81% of bandwidth.

---

## 10. Drawing Representation: Are 1000 Triangles Optimal?

### Current allocation

- `MAX_POLYGONS_PER_IMAGE = 1000`
- Each polygon is a single triangle (3 vertices)
- Typical active count: 150-300 (set by `START_WITH_POLYGONS_PER_IMAGE = 150`, grows via add_polygon mutations)

### Wasted capacity

At 300 active polygons, 700 polygon slots (70%) are allocated but unused. This wastes:
- `700 * 48 * 512 * 2 = 34.4 MB` — memory that is allocated but never read/written

However, the mutate and select passes only iterate up to `polygon_count`, so the unused slots do not consume bandwidth. They only waste allocation (GPU memory address space), which is not the bottleneck.

### Would more polygons help convergence?

1000 triangles can represent complex images well. The bottleneck for art quality is iterations-per-second, not polygon count. Reducing MAX_POLYGONS_PER_IMAGE to 512 would halve state size and proportionally reduce mutate/select bandwidth. But it limits the ultimate detail achievable.

**Recommendation:** Make MAX_POLYGONS_PER_IMAGE configurable (it already effectively is via `max_polygons` param). For faster iteration during early convergence, use 256-512 polygons. For final refinement, switch to 1000. No code change needed — just parameter tuning.

### Would non-triangle primitives help?

Ellipses, rectangles, or Bezier patches could represent some shapes with fewer primitives. But they complicate rasterization (no simple half-space test) and mutation (more parameters per primitive). Triangles are the right choice for GPU compute.

---

## 11. Cache-Friendly Layouts: AoS vs SoA

### Current: Array of Structures (AoS)

```
polygons: array<Polygon, 1000>
// Each Polygon: color(16) + v0(8) + v1(8) + v2(8) + pad(8) = 48 bytes
```

In the rasterize pass, every thread reads the full polygon struct to check AABB, then edge functions, then alpha blend. All 48 bytes are used (except padding), so AoS is fine for rasterize — it loads a contiguous 48-byte cache line per polygon.

### Would SoA help?

SoA layout:
```
colors: array<vec4<f32>, 1000>     // separate buffer
vertices: array<TriVerts, 1000>    // v0, v1, v2 packed
```

In rasterize, we need both color and vertices for each polygon, so SoA would cause two separate memory accesses per polygon instead of one contiguous read. **SoA would be worse for rasterize.**

In mutate, different mutations touch different fields (color-only, vertex-only), so SoA could reduce bandwidth for color-only mutations. But mutate is already a minor contributor (6.3% of total bandwidth).

**Verdict: AoS is correct for this workload.** No change recommended.

---

## 12. Algorithmic Complexity Summary

| Pass | Time Complexity (per chain) | Bottleneck |
|------|----------------------------|------------|
| Mutate | O(polygon_count) | Serial per-polygon loop, 1 thread |
| Rasterize | O(W * H * polygon_count) | Per-pixel polygon iteration |
| Error Reduce | O(W * H) | Per-pixel diff + tree reduction |
| Select | O(polygon_count) | Serial polygon copy (when accepted) |
| Migrate | O(polygon_count) | Serial polygon copy (when adopted) |

**Rasterize dominates** at O(W * H * polygon_count). For 384x384 * 300 = 44.2M operations per chain. All other passes are linear in polygon_count or image size.

---

## 13. Prioritized Recommendations

### Tier 1 — High Impact, Low Risk

1. **Fuse rasterize + error_reduce into a single shader pass** (Section 3)
   - Eliminates render_targets buffer (301 MB GPU memory)
   - Saves 602 MB bandwidth/iteration (81% of total)
   - Implementation: ~100 lines of shader change
   - Risk: None (mathematically identical)

### Tier 2 — Medium Impact, Low-Medium Risk

2. **Pack polygon colors as u32 instead of vec4<f32>** (Section 5)
   - Reduces polygon size from 48 to 32 bytes (33% reduction)
   - Saves 15.7 MB GPU memory across all state buffers
   - Reduces mutate/select/migrate bandwidth proportionally
   - Risk: Low (trivial unpack in rasterize, pack in mutate)

### Tier 3 — Medium Impact, Medium Risk

3. **In-place mutation with undo log** (Section 4, Option B)
   - Eliminates working_states buffer (23.5 MB)
   - Halves mutate-pass bandwidth
   - Risk: Medium (undo logic for add/remove/reorder mutations)

4. **Tile-based polygon binning** (Section 6, Alternative A)
   - Could reduce rasterize ALU 10-20x, but memory overhead is high for 512 chains
   - Only worthwhile after fused rasterize+error (since rasterize ALU becomes the bottleneck once bandwidth is solved)

### Tier 4 — High Impact, High Risk (v2 architecture)

5. **Incremental delta evaluation** (Section 9)
   - 10-30x fewer pixels evaluated per iteration
   - Requires mutation tracking, dirty rectangles, partial error bookkeeping
   - Fragile to rounding errors; high implementation complexity
