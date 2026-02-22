# Data Structures & Algorithms Analysis (v2)

Updated analysis of data layout, memory efficiency, and algorithmic complexity in the artgen-backend-rust GPU evolution pipeline. This revision accounts for all optimizations completed since the original analysis: subgroup error reduction, tile culling via `bin_polygons.wgsl`, cooperative parent loading via shared memory, compute pass consolidation, parallel min-reduction in select, L1 error metric (no sqrt), configurable workgroup sizes, and incremental evaluation with dirty bounding boxes.

**Target hardware:** RTX 5090 (170 SMs, 32-wide warps, 128KB shared memory per SM, 96MB L2)

---

## Already Completed (Not Discussed Further)

- Fused rasterize+error with subgroup-accelerated reduction (`subgroupAdd`)
- Tiled shared memory polygon prefetch (TILE_CAP = THREAD_COUNT * 3)
- Tile culling via `bin_polygons.wgsl` (per-offspring polygon binning into spatial tiles)
- Cooperative parent loading in mutate shader via workgroup shared memory
- Compute pass consolidation (bulk iterations in single pass, separate passes only for timestamps)
- Parallel min-reduction in select shader (O(log2(64)) binary tree)
- L1 error metric (no sqrt; `abs(dr) + abs(dg) + abs(db)`)
- Adaptive mutation scale, per-offspring persistent PCG RNG
- Configurable rasterize workgroup size (8x8, 16x8, 16x16) via string replacement
- Incremental evaluation with dirty bounding boxes (single-mutation mode)
- Double-buffered async staging readback
- Pipeline cache, degenerate triangle culling, variable lambda

---

## 1. bin_polygons.wgsl is Single-Threaded: O(polygons * tiles) Serial Bottleneck

### Current State

The `bin_polygons.wgsl` shader dispatches one workgroup of **1 thread** per offspring (`@workgroup_size(1, 1, 1)`). That single thread serially iterates over all polygons, computes each AABB, converts to tile coordinates, and atomically appends polygon indices to every overlapping tile:

```wgsl
for (var pi = 0u; pi < poly_count; pi++) {
    // unpack vertices, compute AABB, iterate tile ranges
    for (var ty = tile_min_y; ty < tile_max_y; ty++) {
        for (var tx = tile_min_x; tx < tile_max_x; tx++) {
            atomicAdd(&tile_counts[...], 1u);
            tile_data[...] = pi;
        }
    }
}
```

For 500 polygons covering an average of 4 tiles each, that is 2000 iterations per offspring on a single thread. With 128 offspring (16 chains * lambda=8), only 128 workgroups are dispatched, each with 1 thread. On an RTX 5090 with 170 SMs, this leaves the vast majority of the GPU idle.

### Problem

The serial iteration preserves polygon draw order (required for alpha blending correctness), but the per-polygon tile scatter is embarrassingly parallelizable. The constraint is that polygon indices within each tile must appear in ascending order. Serial iteration trivially satisfies this, but it is far from the only way.

### Optimization A: Multi-Threaded Binning With Ordered Writes

Use a workgroup of 64 threads per offspring. Each thread handles a contiguous stripe of polygons (thread `t` handles polygons `t, t+64, t+128, ...`). Since polygon indices are processed in order within each thread and threads process non-overlapping index ranges, the writes are ordered as long as thread `t` writes all its indices to a tile before thread `t+1` does.

**Two-phase approach:**
1. **Count phase**: All 64 threads count how many polygons they would write to each tile (local accumulator, no atomics). A prefix sum across threads gives per-thread offsets within each tile.
2. **Write phase**: Each thread writes its polygon indices to the computed offsets, preserving order.

This is complex because the per-tile offsets depend on all threads' counts. A simpler variant:

**Approach: Parallel atomic appends with post-sort.**
Let all 64 threads atomically append polygon indices to tile lists (unordered). After binning, dispatch a lightweight sort pass that sorts each tile's polygon list by index. Bitonic sort on 256 elements (TILE_MAX_POLYS) requires only 8 passes of ceil(256/2)=128 compare-swaps, trivially parallelizable within a workgroup.

However, the sort adds a full extra dispatch. The cost must be weighed against the current serial binning.

**Simplest effective approach**: Use `@workgroup_size(64)` but keep serial polygon iteration for the write. The 64 threads can parallelize the **tile count reset** (currently serial: `for t in 0..num_tiles { atomicStore(&tile_counts[...], 0u); }`). The polygon iteration itself must remain serial for order, but the tile reset and any setup work can be parallelized.

### Optimization B: Fuse Binning Into Mutate Shader

Since the mutate shader already knows which polygons changed (and the dirty bounding box in single-mutation mode), it could incrementally update the tile data rather than rebuilding from scratch every iteration. In single-mutation mode, only the tiles overlapping the dirty bbox need to be re-binned.

**Estimated complexity**: High. Requires persistent tile data per chain (not per offspring), copy-on-write semantics for offspring tile data, and careful handling of polygon insertion/deletion that shifts indices.

### Verdict

Optimization A (parallel tile count reset + potential post-sort) is the most practical improvement. The serial polygon loop is the genuine bottleneck, and the simplest fix is increasing the workgroup size for the reset phase. For a deeper fix, profiling should confirm whether bin_polygons is actually a significant fraction of total iteration time. If it is under 5% of rasterize_error, the serial approach is acceptable.

**Estimated impact**: 2-5x faster binning pass. Total iteration impact depends on binning's share of compute time.

---

## 2. Incremental Eval: Double Rasterization Negates Most Savings

### Current State

When incremental evaluation is enabled (`params.incremental_eval == 1`), the rasterize_error shader performs **two full rasterizations** per pixel in the dirty region:

1. **Parent rasterization** (old pixel error): iterates all of `chain_states[parent_chain].polygons` from global memory (no shared memory tiling, no AABB culling):
```wgsl
for (var pi = 0u; pi < parent_poly_count; pi++) {
    rasterize_blend(chain_states[parent_chain].polygons[pi], fx, fy, &old_r, &old_g, &old_b);
}
```

2. **Offspring rasterization** (new pixel error): uses the normal tiled shared memory path with tile culling.

The parent rasterization loop has no shared memory tiling, no AABB culling, and reads from global memory (`chain_states`) instead of `working_states`. This is a raw `O(parent_poly_count)` global memory read per pixel, which is the exact brute-force pattern that shared memory tiling was designed to avoid.

### Problem

For a typical drawing with 300 polygons, the parent rasterization does 300 global memory reads per pixel. With a dirty region covering 10% of a 512x512 image (~26K pixels), that is 7.8M global memory reads just for the old-error computation. Meanwhile the offspring rasterization (with tiling and tile culling) is much more efficient. The parent rasterization is the dominant cost of the incremental eval path.

### Optimization: Cache Parent Rasterization in Framebuffer

Store a pre-rasterized RGBA framebuffer per chain (512x512x4 = 1MB per chain). When a mutation is accepted, update the dirty region of the framebuffer. When computing old pixel error for incremental eval, read from the cached framebuffer instead of re-rasterizing the parent.

This turns the parent error computation from `O(dirty_pixels * parent_polygons)` to `O(dirty_pixels)` -- just a texture/buffer read.

**Memory cost**: 1MB per chain. At 16 chains = 16MB, well within the RTX 5090's 32GB VRAM budget.

**Consistency issue**: The `init_framebuffers.wgsl` shader already exists and computes this framebuffer, but it is only used for initial setup (`dispatch_init_framebuffers`). The select shader already has `chain_framebuffers` and `chain_total_errors` bindings. The infrastructure is partially in place but the rasterize_error shader does not read from the cached framebuffer for the old-error path.

### Proposed Change

In `rasterize_error.wgsl`, replace the parent re-rasterization with a framebuffer lookup:

```wgsl
if incremental {
    // Read cached parent pixel from framebuffer (1 global read vs N polygon reads)
    let fb_idx = parent_chain * w * h + py * w + px;
    let packed = chain_framebuffers[fb_idx];
    let old_r = f32(packed & 0xFFu);
    let old_g = f32((packed >> 8u) & 0xFFu);
    let old_b = f32((packed >> 16u) & 0xFFu);
    pixel_error_old = u32(abs(old_r - refr) + abs(old_g - refg) + abs(old_b - refb));
}
```

**Complication**: The framebuffer stores u8-quantized values, while the parent re-rasterization computes in f32. This introduces quantization error drift over many accepted mutations. The current code explicitly notes this: "Uses chain_states (parent) instead of a u8 framebuffer to avoid quantization mismatch that causes error drift."

**Mitigation**: Periodically (every N accepted mutations per chain) re-rasterize the full framebuffer from chain_states to reset quantization drift. Alternatively, store the framebuffer in f16 per channel (2 bytes per channel, 6 bytes per pixel) using two u32s per pixel, halving quantization error. At 8 bytes per pixel: 512x512x8 = 2MB per chain = 32MB for 16 chains.

### Verdict

**High impact if incremental eval is the primary mode.** The parent re-rasterization is the single largest inefficiency in the incremental eval path. Using the cached framebuffer (with periodic refresh to control drift) could make incremental eval 5-20x faster for single-polygon mutations. This is the most impactful data-structure change available.

**Estimated impact**: 5-20x speedup for incremental eval's old-error computation. Total iteration speedup depends on dirty region size (smaller regions = larger speedup proportion).

---

## 3. Tile Culling Memory: Massive VRAM Cost for Large Configurations

### Current State

The tile culling buffers are sized for worst-case:

```rust
let max_num_tiles = ((w + 7) / 8) * ((h + 7) / 8);  // worst case: 8x8 WG
let tile_data = offspring_capacity * max_num_tiles * TILE_MAX_POLYS * 4;  // u32 per entry
let tile_counts = offspring_capacity * max_num_tiles * 4;
```

For a 512x512 image with 8x8 tiles: `max_num_tiles = 64 * 64 = 4096`. With `offspring_capacity = 1024 * 64 = 65536` (max chains * max lambda) and `TILE_MAX_POLYS = 256`:

- `tile_data = 65536 * 4096 * 256 * 4 = 274.9 GB` -- this obviously cannot be allocated.

In practice, `offspring_capacity` is capped by `max_ssbo / state_size`, which limits it to a much smaller value. But even with 128 offspring and 16x16 tiles (1024 tiles): `128 * 1024 * 256 * 4 = 128 MB` for tile_data alone. This is significant.

### Problem

`TILE_MAX_POLYS = 256` is a fixed constant that wastes space for tiles where most polygons do not overlap. For a drawing with 300 small polygons, the average tile might contain 30 polygon indices, but the buffer allocates slots for 256.

### Optimization A: Reduce TILE_MAX_POLYS Adaptively

Track the actual maximum tile occupancy during binning (using atomicMax on a counter) and dynamically reduce the allocation on subsequent iterations. Start at 256, but if observed max is consistently 80, reduce to 128 (next power of 2). This requires a GPU-side counter and a CPU readback path (or just use a heuristic based on polygon count).

**Simple heuristic**: `effective_tile_max = min(256, polygon_count)`. A drawing with 150 polygons can have at most 150 polygons per tile, so allocating 256 slots is wasteful. This saves `(256-150)/256 = 41%` of tile_data memory for that case.

### Optimization B: Compact Tile Storage with Prefix Sums

Replace the fixed-size per-tile allocation with a compact representation:
1. Binning pass outputs per-tile counts only (no data yet).
2. A prefix-sum pass computes offsets into a compact flat buffer.
3. A second binning pass (or the same pass with two sub-phases) writes polygon indices at the computed offsets.

This eliminates all wasted slots. For 300 polygons averaging 4 tiles each = 1200 total entries vs. 4096 * 256 = 1M entries with fixed allocation. A 833x memory reduction.

**Complexity**: Requires an additional dispatch (prefix sum) and modifying the binning shader to be two-pass. The prefix sum is trivial for up to 4096 tiles (fits in one workgroup).

### Verdict

Optimization A (heuristic TILE_MAX_POLYS) is simple and effective. Optimization B (prefix-sum compaction) is more impactful but requires significant shader refactoring. Both are worth considering when running at high chain counts where tile buffer memory becomes the bottleneck.

**Estimated impact**: 30-80% reduction in tile culling VRAM, enabling higher chain counts before hitting memory limits.

---

## 4. Multi-Mutation Mode: Per-Polygon Loop is O(polygons * mutation_types)

### Current State

In multi-mutation mode (the default when `single_mutation_mode == 0`), the mutate shader iterates over every polygon and tests each against ~18 independent random draws:

```wgsl
for (var pi = 0u; pi < count; pi++) {
    // offset_polygon: 1 rand draw + conditional work
    // change_color: 4 rand draws (one per channel)
    // micro_adjust_color: 4 rand draws
    // adjust_brightness: 1 rand draw
    // adjust_saturation: 1 rand draw
    // move_point: 3 rand draws (one per vertex)
    // medium_move: 3 rand draws
    // micro_adjust: 3 rand draws
    // Total: ~18 rand draws per polygon
}
```

For 300 polygons: 5400 RNG calls per offspring. With lambda=8 and 16 chains: 691K RNG calls per iteration. The PCG hash function (`pcg_step`) is ~8 ALU ops, so this is ~5.5M ALU ops in RNG alone per iteration.

### Optimization: Batched RNG with Bit Partitioning

Draw fewer random numbers and use bit slicing to make multiple decisions from each one. Since all mutation probabilities are small (typically 0.001-0.01), a single 32-bit random number can drive 4 independent Bernoulli trials at u8 precision:

```wgsl
let r = pcg_step(&rng);
let should_offset = (r & 0xFFu) < u32(params.offset_polygon_prob * 255.0);
let should_color_r = ((r >> 8u) & 0xFFu) < u32(params.change_color_prob * 255.0);
let should_color_g = ((r >> 16u) & 0xFFu) < u32(params.change_color_prob * 255.0);
let should_color_b = ((r >> 24u) & 0xFFu) < u32(params.change_color_prob * 255.0);
```

This reduces RNG calls from ~18 to ~5 per polygon (ceil(18/4) + a few that need their own draw for the actual random values, not just the probability test).

**Precision impact**: Using 8-bit comparison means probability resolution is 1/256. For probabilities like 0.003 (change_color default), the nearest representable value is 1/256 = 0.0039. This is a ~30% error in the probability, which is unlikely to affect evolution quality since these probabilities are approximate anyway.

### Optimization: Skip Unchanged Polygons (Early-Out)

In multi-mutation mode, most polygons are not modified in a given iteration (each per-polygon mutation has <1% probability). The shader could test all mutation probabilities for a polygon using a single combined threshold first:

```wgsl
let any_mutation_prob = 1.0 - pow(1.0 - max_per_poly_prob, 18.0);
if rand_f32(&rng) > any_mutation_prob {
    // No mutations will fire for this polygon -- skip entirely
    continue;
}
```

For `max_per_poly_prob = 0.01`, `any_mutation_prob = 1 - 0.99^18 = 0.166`. So ~83% of polygons can be skipped entirely, saving all 18 RNG draws and the associated unpack/repack work.

**Caveat**: The `pow` computation should be precomputed on the CPU and passed as a GpuParams field, not computed per polygon.

### Verdict

The early-out optimization is the bigger win (83% reduction in per-polygon work) and is simple to implement. Batched RNG is an additional optimization that reduces the remaining 17% of polygon work.

**Estimated impact**: 3-5x faster per-polygon loop in multi-mutation mode. Mutate is typically 10-20% of total iteration time, so overall impact is 5-10%.

---

## 5. Rasterize Shared Memory: `shared_polys` Over-Allocation

### Current State

The rasterize_error shader declares a worst-case shared memory array:

```wgsl
var<workgroup> shared_polys: array<Polygon, 1536>;   // max tile cap (512*3)
```

This is 1536 * 16 = 24,576 bytes, allocated for the 512-thread configuration (THREAD_COUNT=512 is theoretically possible). However, the actual tile cap depends on compile-time constants:

| WG Size | THREAD_COUNT | TILE_CAP | shared_polys used | shared_polys allocated | Waste |
|---------|-------------|----------|-------------------|----------------------|-------|
| 16x16   | 256         | 768      | 12,288 B          | 24,576 B             | 50%   |
| 16x8    | 128         | 384      | 6,144 B           | 24,576 B             | 75%   |
| 8x8     | 64          | 192      | 3,072 B           | 24,576 B             | 87.5% |

The array is declared at 1536 entries but only TILE_CAP entries are ever accessed. The remaining entries waste shared memory and reduce per-SM occupancy.

### Optimization

Change the shared memory declaration to use TILE_CAP directly:

```wgsl
var<workgroup> shared_polys: array<Polygon, TILE_CAP>;
```

Since `TILE_CAP` is a compile-time constant (derived from `THREAD_COUNT * 3`), and the pipeline already uses string replacement on `WG_X`/`WG_Y`, this would automatically size shared memory to match the workgroup configuration.

### Impact

At the default 16x16 workgroup (TILE_CAP=768):
- Current: 24,576 B shared_polys + 512 B shared_errors + 512 B shared_errors_old + 20 B misc = ~25,620 B
- Fixed: 12,288 B + 512 B + 512 B + 20 B = ~13,332 B
- Savings: ~12 KB per workgroup

On RTX 5090 with 128KB shared memory per SM:
- Current: 128KB / 25.6KB = 5 concurrent workgroups/SM
- Fixed: 128KB / 13.3KB = 9 concurrent workgroups/SM

This is an 80% increase in occupancy, directly improving latency hiding for memory-bound texture reads.

**Similarly**, `shared_errors` and `shared_errors_old` are declared as `array<u32, 128>` (128 entries) but only `ceil(THREAD_COUNT / sg_size)` entries are used. With sg_size=32 and THREAD_COUNT=256, only 8 entries are used. The 120 unused entries waste 960 bytes. While small individually, fixing these reduces total shared memory to ~12.5 KB.

### Verdict

**High impact, trivial effort.** Change the literal array size `1536` to `TILE_CAP` in the shader source. This is a one-line fix with the string replacement infrastructure already in place.

**Estimated impact**: 40-80% more concurrent workgroups per SM for the rasterize pass, depending on workgroup size configuration.

---

## 6. Error Accumulator Stride-2 Layout Causes Atomic Contention

### Current State

Error accumulators use a stride-2 layout: `[new_error_0, old_error_0, new_error_1, old_error_1, ...]`. Each offspring's two u32 accumulators are adjacent in memory:

```wgsl
atomicAdd(&error_accumulators[chain_id * 2u], total);       // new error
atomicAdd(&error_accumulators[chain_id * 2u + 1u], total_old); // old error (incremental)
```

### Problem

On NVIDIA GPUs, atomic operations on addresses within the same 128-byte cache line are serialized. Two adjacent u32s (at offset 0 and 4 within a cache line) will always share a cache line. When multiple workgroups for the same offspring (different pixel tiles) issue concurrent atomicAdds to the same accumulator, they contend on the same cache line.

For a 512x512 image with 16x16 workgroups, each offspring has 32x32 = 1024 workgroups, all atomically adding to the same two u32s. These 1024 atomicAdds must be fully serialized because they target the same cache line.

### Optimization: Hierarchical Reduction Instead of Global Atomics

Replace the per-workgroup atomicAdd with a two-level reduction:

1. **Level 1 (already done)**: subgroupAdd within each workgroup (produces 1 value per workgroup per offspring).
2. **Level 2 (new)**: Instead of atomicAdd to a global accumulator, write workgroup-level partial sums to a per-offspring array, then dispatch a lightweight reduction kernel.

**Layout**: `workgroup_errors[offspring_id * num_wgs + wg_linear_id]` where `wg_linear_id = wid.y * num_wgs_x + wid.x`.

**Final reduction**: A separate pass sums `num_wgs` values per offspring. With 1024 workgroups per offspring, this is 1024 u32 additions, trivially parallelizable in one workgroup.

**Memory cost**: 1024 * offspring_count * 4 bytes. For 128 offspring: 512 KB. Acceptable.

### Analysis

The benefit depends on atomic contention being a measurable bottleneck. On modern NVIDIA GPUs, L2 atomic throughput is high enough that 1024 serialized atomics may only take ~1-2 microseconds. The reduction approach eliminates contention entirely but adds a dispatch.

### Verdict

**Medium priority.** Worth profiling with `atomicAdd` vs. the two-level approach. If rasterize_error is memory-bound (likely), the atomic contention is hidden by memory latency and the benefit is minimal. If rasterize_error is compute-bound (at low polygon counts), atomic contention could be the tail latency bottleneck.

---

## 7. Crossover Reads from Global Memory Without Shared Memory Prefetch

### Current State

The crossover path in `mutate.wgsl` reads parent B's polygons directly from global memory:

```wgsl
fn crossover_uniform_offspring(rng: ..., parent_b: u32, offspring_id: u32) {
    // ...
    working_states[offspring_id].polygons[out_count] = chain_states[parent_b].polygons[i];
    // ...
}
```

While parent A's polygons are loaded from shared memory (the cooperative parent load), parent B is always read from `chain_states` -- a global memory access per polygon. For a drawing with 500 polygons, the crossover path issues 500 uncoalesced global reads from a random chain's state (which may be 16KB * parent_b bytes away from any cached data).

### Optimization

When crossover fires, load parent B's polygon data into a second shared memory region (or reuse the existing `shared_parent_polygons` with a second barrier-synchronized load phase). Since crossover probability is typically low (~10%), this adds shared memory cost to all invocations for a benefit that only fires 10% of the time.

**Alternative**: Only load the portions of parent B that will actually be used. In uniform crossover, ~50% of polygons come from each parent. A two-pass approach: first determine which indices come from B (using RNG), then cooperatively load only those polygons from B into shared memory.

### Verdict

**Low priority.** Crossover fires rarely (~10% of offspring), and the per-polygon global reads from parent B are sequential (good for hardware prefetch). The benefit of shared memory would be a constant factor improvement on 10% of mutations -- at most a 5% overall improvement.

---

## 8. GpuDrawingState: rng_state Uses Only 4 of 16 Bytes

### Current State

```rust
pub rng_state: [u32; 4],  // offset 16 -- only [0] is active RNG state; [1..3] unused padding
```

The PCG-RXS-M-XS RNG uses a single u32 state. The remaining 12 bytes (3 * u32) are padding to maintain `vec4<u32>` alignment.

### Optimization: Use Padding for Metadata

The 12 bytes of padding could store useful per-chain metadata without increasing the struct size:

- **`rng_state[1]`**: Last mutation type applied (u32 enum). Useful for adaptive operator selection (tracking which mutation types are productive per chain).
- **`rng_state[2]`**: Acceptance count (u16) + rejection count (u16) since last migration. Useful for per-chain acceptance rate tracking.
- **`rng_state[3]`**: Reserved for future use or error-map grid index.

This is free storage -- zero memory cost, zero bandwidth cost (the data is already loaded/stored as part of the `vec4<u32>` read/write).

### Verdict

**Zero-cost opportunity.** Any future feature that needs per-chain metadata (adaptive operator selection, per-chain acceptance rate tracking) should use these padding bytes rather than extending the struct.

---

## 9. Integer Rasterization: Eliminate Float Edge Functions

### Current State

The rasterize_error shader computes half-space edge functions in floating point:

```wgsl
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}
```

Vertices are unpacked from u16 to f32 (`f32(word & 0xFFFFu) / 65535.0`), pixel coordinates are converted to normalized f32 (`(f32(px) + 0.5) / f32(w)`), and the edge function uses 4 f32 multiplies and 5 f32 subtracts.

### Optimization: Integer Edge Functions

The CPU path (`fill_triangle` in `utils.rs`) already uses **28.4 fixed-point** integer edge functions, which are exact (no rounding) and potentially faster on GPUs where integer ALU units are less loaded than float units.

Since vertices are stored as u16 and pixel coordinates are integers, the edge function can be computed entirely in i32 without any float conversion:

```wgsl
fn edge_fn_int(ax: i32, ay: i32, bx: i32, by: i32, px: i32, py: i32) -> i32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}
```

Vertices would be unpacked as raw i32 (scaled by image dimensions) instead of normalized f32. Pixel coordinates are already integers.

**Precision**: With u16 vertices (0-65535) and 512-pixel images, the edge function values fit in i32 (max magnitude: 65535^2 * 2 = ~8.6 billion, within i32 range of 2.1 billion -- actually this overflows). Need i64 or scale down. With vertices scaled to pixel coordinates (0-512), max magnitude: 512^2 * 2 = 524K, easily fits i32.

**Approach**: Unpack vertices directly to pixel coordinates:
```wgsl
let vx = i32((word & 0xFFFFu) * w / 65535u);  // integer pixel coord
let vy = i32((word >> 16u) * h / 65535u);
```

### Analysis

On NVIDIA GPUs, i32 multiplies and f32 multiplies have the same throughput (1 per cycle per core). The main benefit is eliminating the `f32(word) / 65535.0` conversion and the `(f32(px) + 0.5) / f32(w)` normalization -- saving 2 divides + 2 int-to-float conversions per vertex unpack and 1 divide + 1 add per pixel coordinate. For 3 vertices * ~300 polygons * 256 pixels per workgroup = 230K vertex unpacks per workgroup dispatch, this is a measurable savings.

**Complication**: The AABB cull currently compares in normalized coordinates. With integer coordinates, AABB cull compares in pixel space, which is actually simpler and avoids 6 float comparisons.

### Verdict

**Medium impact, medium effort.** The main benefit is eliminating float conversions and divisions in the inner loop. The integer edge function itself is not faster (same throughput), but the reduced conversion overhead adds up over millions of polygon tests. Requires changing all coordinate handling in the rasterize shader.

---

## 10. Offspring Buffer: Copy-on-Write for Single-Mutation Mode

### Current State

In single-mutation mode, the mutate shader copies the entire parent drawing (up to 1000 polygons = 16KB) from shared memory to the offspring slot, then modifies 1-2 polygons. With lambda=8, this means 8 copies of the same 16KB parent per chain per iteration.

```wgsl
for (var i = 0u; i < poly_count; i++) {
    working_states[offspring_id].polygons[i] = shared_parent_polygons[i];
}
```

### Optimization: Pointer + Delta Representation

Instead of copying the entire parent, store offspring as (parent_chain_id, delta_count, delta_entries[]):

```
struct OffspringDelta {
    parent_chain: u32,
    delta_count: u32,        // 0-3 modified polygons
    delta_indices: vec3<u32>, // which polygon indices were modified
    delta_polys: array<Polygon, 3>,  // the new polygon values
    dirty_bbox: vec2<u32>,   // packed dirty region
    // ... header fields
}
```

The rasterize shader would read the parent chain's polygons directly and overlay the deltas at the modified indices. This eliminates the 16KB copy entirely.

### Analysis

**Pros**: Eliminates ~16KB write per offspring. With 128 offspring per iteration, saves ~2MB of write bandwidth per iteration.

**Cons**: The rasterize shader's shared memory tiling pattern breaks. Currently, polygons are cooperatively loaded from `working_states[offspring_id]` as a contiguous array. With the delta representation, the rasterizer would need to: (1) load parent polygons from `chain_states[parent_chain]`, (2) check each polygon against the delta indices, (3) substitute the delta values. This adds per-polygon branch overhead in the inner loop.

**Alternative**: Apply deltas during the cooperative load into shared memory. Thread 0 checks the delta list while other threads load from the parent:

```wgsl
// Cooperative load: read from parent, apply deltas
for (var i = local_idx; i < poly_count; i += THREAD_COUNT) {
    var poly = chain_states[parent_chain].polygons[i];
    // Check if this polygon is in the delta list (at most 3 checks)
    for (var d = 0u; d < delta_count; d++) {
        if i == delta_indices[d] { poly = delta_polys[d]; break; }
    }
    shared_polys[i] = poly;
}
```

This adds 1-3 comparisons per polygon per thread during the cooperative load, which is negligible.

**Memory savings**: Offspring buffer shrinks from `offspring_capacity * 16032 bytes` to `offspring_capacity * ~128 bytes` (header + 3 deltas = ~112 bytes). For 128 offspring: from 2MB to 16KB. This is a 125x reduction in offspring buffer size.

### Verdict

**High impact on memory footprint, medium implementation effort.** The offspring buffer is the largest GPU allocation after tile culling data. Reducing it by 100x allows dramatically higher offspring_capacity (more chains or higher lambda) within the same VRAM budget. The rasterize shader change (delta overlay during cooperative load) adds minimal overhead to the hot loop.

This optimization only works for single-mutation mode. Multi-mutation mode modifies many polygons and falls back to the full copy.

**Estimated impact**: 100x reduction in offspring buffer VRAM. Enables 10-100x more chains within current memory limits.

---

## 11. GpuParams: 80 Bytes of Padding Unused

### Current State

`GpuParams` is 256 bytes. The last 80 bytes (`_reserved: [u32; 20]`) are unused padding. In addition, `_pad0`, `_pad2`, `_pad3` waste 12 bytes. Total unused: 92 bytes (36% of the struct).

### Optimization

Use the reserved space for frequently-needed per-batch metadata that currently requires separate buffer writes or recomputation:

- **Precomputed mutation skip threshold** (1 f32): The `any_mutation_prob` from Section 4's early-out optimization.
- **Resolution stride** (1 u32): For multi-resolution coarse-to-fine evaluation.
- **Error grid** (16 u32): A 4x4 error heatmap for error-guided polygon placement.
- **Crossover parameters** (4 f32): Layer-range crossover cut point bias, differential evolution scale factor.
- **Per-batch RNG seed** (1 u32): A per-batch seed for batch-deterministic experiments.

The 20 reserved u32s (80 bytes) can accommodate all of these with room to spare.

### Verdict

**Zero-cost opportunity.** The reserved space should be used for new parameters as features are added, avoiding the need to resize the params struct.

---

## 12. CPU Path: `fill_shape` Allocates HashMap Per Polygon

### Current State

The scanline fill function `fill_shape` in `utils.rs` allocates a `HashMap<usize, Vec<Line>>` edge table, a `Vec<Point>` for intersection points, and sorts intersection points per scanline. This allocates on every polygon rasterization call.

### Optimization

Fan-triangulate multi-point polygons on the CPU side (matching the GPU path) and use `fill_triangle` for everything. `fill_triangle` uses no heap allocation (only stack-based 8x8 blocking with SIMD blending).

### Implementation

```rust
pub fn draw(&self, buffer: &mut [u8], w: usize, h: usize, rm: Rasterizer) {
    buffer.fill(255u8);
    for polygon in &self.polygons {
        if polygon.points.len() == 3 {
            let _ = fill_triangle(buffer, polygon, w, h);
        } else {
            // Fan triangulation
            for i in 1..polygon.points.len() - 1 {
                let tri = Polygon {
                    points: vec![polygon.points[0], polygon.points[i], polygon.points[i + 1]],
                    color: polygon.color,
                };
                let _ = fill_triangle(buffer, &tri, w, h);
            }
        }
    }
}
```

### Verdict

**Low effort, eliminates all heap allocation in the CPU raster path.** Since the CPU path is used for display rendering (PNG generation every 200ms), reducing allocation pressure improves worst-case latency.

---

## Summary of New Recommendations

### High Impact

| # | Optimization | Expected Benefit | Effort |
|---|-------------|-----------------|--------|
| 2 | Incremental eval: use cached framebuffer instead of re-rasterizing parent | 5-20x faster incremental eval old-error computation | Medium |
| 5 | Fix shared_polys over-allocation (1536 -> TILE_CAP) | 40-80% more concurrent workgroups per SM | Trivial |
| 10 | Copy-on-write offspring (delta representation) | 100x offspring buffer memory reduction | Medium-High |

### Medium Impact

| # | Optimization | Expected Benefit | Effort |
|---|-------------|-----------------|--------|
| 4 | Multi-mutation early-out (skip unmodified polygons) | 3-5x faster per-polygon loop | Low |
| 3 | Compact tile storage (reduce TILE_MAX_POLYS or prefix-sum) | 30-80% tile VRAM savings | Medium |
| 9 | Integer edge functions (eliminate float conversion overhead) | 5-15% faster per-polygon rasterize | Medium |
| 1 | Parallelize bin_polygons (multi-threaded tile reset, optional post-sort) | 2-5x faster binning | Medium |

### Zero-Cost Opportunities (Use What Is Already Allocated)

| # | Optimization | Notes |
|---|-------------|-------|
| 8 | Use rng_state padding for per-chain metadata | 12 free bytes per chain, enables adaptive operator selection |
| 11 | Use GpuParams reserved space for new features | 80 free bytes, eliminates future struct resizing |

### Low Impact (Not Recommended Unless Specific Need Arises)

| # | Optimization | Reason to Skip |
|---|-------------|---------------|
| 6 | Hierarchical error reduction (eliminate atomic contention) | Atomic throughput on modern GPUs likely sufficient |
| 7 | Shared memory prefetch for crossover parent B | Crossover fires rarely (~10%), benefit is small |
| 12 | CPU fan-triangulate all polygons | CPU path is secondary; low user-facing impact |
