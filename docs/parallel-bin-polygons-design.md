# Parallel bin_polygons Design Proposal

## Executive Summary

**Recommendation: Do not implement fully parallel scatter. Implement the minimal optimization (parallel tile reset + serial scatter) instead.**

The bin_polygons shader is a minor fraction of total GPU time. The binning pass runs inside the rasterize_error timing window, and GPU profiling from the existing timestamp infrastructure shows mutate + rasterize + select. Binning is bundled with rasterize_error and represents only a small portion of it. The algorithmic complexity of a fully parallel z-order-preserving scatter (counting sort with prefix sum) would add ~150 lines of WGSL, a new intermediate buffer, and meaningful correctness risk, all to save microseconds on a pass that takes <1ms. The simple optimization (parallel reset, serial scatter) captures most of the benefit with minimal risk.

---

## Problem Statement

`bin_polygons.wgsl` runs at `@workgroup_size(1,1,1)` -- one thread per offspring. That thread:
1. Resets tile counts for this offspring (serial loop over up to 1024 tiles)
2. Iterates polygons in z-order, computing AABB-to-tile overlap and writing polygon indices into `tile_data` via `atomicAdd` on `tile_counts`

The serial scatter in step 2 is required for z-order correctness: polygon indices must appear in ascending order within each tile's list because the rasterize shader composites them front-to-back (alpha blending is order-dependent).

With 256 offspring (4 chains x 64 lambda), this launches 256 single-thread workgroups on a 170-SM RTX 5090. Each workgroup wastes 31/32 warp lanes.

---

## Algorithm Options Analysis

### Option A: Parallel Reset + Serial Scatter (Recommended)

Increase workgroup size to 64. Parallelize the tile count reset across 64 threads. Keep the polygon scatter serial (thread 0 only, or all threads after a barrier with thread 0 doing the scatter).

**Tile reset (parallel):**
```
@workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid, @builtin(local_invocation_index) lid) {
    let offspring_id = wid.x;
    // Parallel tile reset: 64 threads, each resets ceil(num_tiles/64) tiles
    for (var t = lid; t < num_tiles; t += 64u) {
        atomicStore(&tile_counts[offspring_id * num_tiles + t], 0u);
    }
    workgroupBarrier();

    // Serial polygon scatter: thread 0 only (preserves z-order trivially)
    if lid == 0u {
        // ... existing serial polygon iteration loop unchanged ...
    }
}
```

**Pros:**
- Trivially correct -- z-order preserved because scatter is still serial
- Parallel reset eliminates 1024 serial atomicStores (now 16 per thread)
- No new buffers, no new passes, no algorithm complexity
- 10 lines changed

**Cons:**
- Only parallelizes the reset phase; scatter remains serial
- 63/64 threads idle during scatter

**Estimated speedup: 5-15% of binning pass time.** The reset loop is O(num_tiles) = O(1024) and dominates when polygon count is low. For high polygon counts (500+), the scatter loop dominates and this optimization has diminishing returns.

---

### Option B: Two-Pass Counting Sort (Full Parallel Scatter)

The classic GPU scatter algorithm that preserves insertion order:

**Pass 1: Count (parallel)**
Each thread processes a stripe of polygons. For each polygon, compute its AABB, determine overlapping tiles, and atomically increment per-tile counts. Z-order does not matter here -- we only need final counts.

**Pass 2: Prefix Sum**
Compute exclusive prefix sum over tile counts to get per-tile write offsets. This tells us where each tile's polygon list starts in the output array.

**Pass 3: Scatter (parallel, z-order preserving)**
Re-iterate polygons in z-order. For each polygon and each tile it overlaps, use the prefix-sum offset + a per-tile running counter to compute the exact write position. Since polygons are iterated in order, and the offset for tile T is deterministic, polygon indices land in z-order within each tile.

BUT: Pass 3 must iterate polygons in order, which means it cannot be parallelized across polygons without breaking z-order. The only parallelism is across tiles within a single polygon's overlap set (typically 1-5 tiles), which is too little to justify the complexity.

**Alternative: Key-Value Sort**
Emit (tile_id, polygon_index) pairs for every polygon-tile intersection, then sort by (tile_id, polygon_index) using a GPU radix sort or bitonic sort. This is fully parallel but:
- Requires a temp buffer of size `polygons * avg_tiles_per_polygon * 8 bytes`
- GPU sort of 5000 pairs (1000 polys * 5 tiles avg) is overkill
- Bitonic sort in WGSL is painful (no dynamic shared memory, manual unrolling)
- Radix sort requires multiple passes with prefix sums

**Pseudocode for counting sort approach:**

```
// === Pass 1: Count ===
@workgroup_size(64, 1, 1)
fn count_pass(wid, lid) {
    offspring_id = wid.x;
    // Parallel tile reset
    for t in lid..num_tiles step 64 { atomicStore(tile_counts[...], 0); }
    workgroupBarrier();

    // Parallel polygon counting: thread t handles polygons t, t+64, t+128...
    for pi in lid..poly_count step 64 {
        poly = working_states[offspring_id].polygons[pi];
        (tile_min_x, tile_min_y, tile_max_x, tile_max_y) = compute_tile_aabb(poly);
        for ty in tile_min_y..tile_max_y {
            for tx in tile_min_x..tile_max_x {
                tile_id = ty * num_tiles_x + tx;
                atomicAdd(&tile_counts[offspring_id * num_tiles + tile_id], 1u);
            }
        }
    }
}

// === CPU/shader: Prefix sum over tile_counts ===
// For each offspring: exclusive_prefix_sum(tile_counts[offspring * num_tiles .. (offspring+1) * num_tiles])
// Store result in tile_offsets buffer

// === Pass 2: Scatter (serial per offspring, parallel across offspring) ===
@workgroup_size(1, 1, 1)   // still serial per offspring for z-order!
fn scatter_pass(wid) {
    offspring_id = wid.x;
    for pi in 0..poly_count {
        poly = working_states[offspring_id].polygons[pi];
        (tile_ranges) = compute_tile_aabb(poly);
        for each tile in ranges {
            offset = tile_offsets[offspring_id * num_tiles + tile_id];
            slot = atomicAdd(&tile_write_counters[offspring_id * num_tiles + tile_id], 1u);
            tile_data[offspring_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS + offset + slot] = pi;
        }
    }
}
```

**Critical observation:** The scatter pass STILL requires serial polygon iteration to preserve z-order. Splitting into two passes does not help with the fundamental ordering constraint. The counting pass can be parallel, but the scatter must be serial. This is identical to Option A with extra complexity.

**Pros:**
- Count phase is fully parallel (useful if count is the bottleneck)
- Could use prefix sum offsets to pack tile_data more tightly (not needed with current TILE_MAX_POLYS cap)

**Cons:**
- Scatter phase is still serial -- no speedup over Option A for the dominant work
- Requires new `tile_offsets` buffer (offspring_capacity * num_tiles * 4 bytes = ~1MB)
- Requires prefix sum implementation (either a separate shader pass or in-workgroup)
- Two dispatches instead of one
- Significantly more complex, more surface area for bugs

**Estimated speedup over Option A: negligible.** The scatter loop (serial polygon iteration + AABB computation + tile writes) is identical in both approaches. The only difference is the count phase is parallel in Option B, but Option A already parallelizes the reset. The count phase in Option B adds work (redundant AABB computation in both passes).

---

### Option C: Parallel Scatter via Per-Polygon Tile Lists + Merge

Pre-compute each polygon's tile list in parallel, then merge lists per-tile in z-order.

**Phase 1:** Each thread computes tiles for its polygon stripe, writes to a per-polygon scratch buffer.
**Phase 2:** For each tile, iterate polygons 0..N in order, check if polygon overlaps this tile (from phase 1 results), and append to tile_data.

Phase 2 inverts the loop: iterate tiles in the outer loop, polygons in the inner loop. This allows parallelism across tiles but still requires serial polygon iteration within each tile.

```
// Phase 2: One thread per tile (parallel across tiles, serial across polygons within each tile)
@workgroup_size(64, 1, 1)
fn scatter_by_tile(wid, lid) {
    offspring_id = wid.x;
    // Each thread handles a different tile
    for tile_id in lid..num_tiles step 64 {
        count = 0u;
        for pi in 0..poly_count {
            if polygon_overlaps_tile(pi, tile_id) {  // needs AABB recompute or lookup
                tile_data[offspring_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS + count] = pi;
                count++;
                if count >= TILE_MAX_POLYS { break; }
            }
        }
        tile_counts[offspring_id * num_tiles + tile_id] = count;  // non-atomic since exclusive
    }
}
```

**This is the only approach that achieves true parallelism while preserving z-order.** Each thread owns a tile exclusively, so no atomics needed. Polygons are iterated in z-order within each tile. Tiles are processed in parallel.

**Pros:**
- Truly parallel: 64 threads process 64 different tiles simultaneously
- Z-order trivially preserved (inner loop iterates polygons 0..N)
- No atomics in the scatter (each thread owns its tile)
- No new buffers needed
- No prefix sum
- Conceptually simple

**Cons:**
- Every thread must check every polygon against its tile = O(poly_count * ceil(num_tiles/64)) total work per thread
- Redundant AABB computation: each polygon's AABB is recomputed by every thread that handles any tile it might overlap
- Total work is O(poly_count * num_tiles / 64), compared to O(poly_count * avg_tiles_per_polygon) in the serial version
- For 1000 polygons and 1024 tiles with 64 threads: each thread checks 1000 * 16 = 16,000 polygon-tile pairs. Serial version does 1000 * 5 = 5,000 polygon-tile pairs total. Parallel version does 3.2x MORE total work.
- However: 64 threads doing 16K each in parallel > 1 thread doing 5K serially, assuming memory bandwidth isn't the bottleneck

**Work amplification analysis:**
- Serial: 1000 polys * 5 tiles avg = 5000 atomicAdd + write operations
- Option C: 1000 polys * 1024 tiles / 64 threads = 16,000 overlap checks per thread, but only 5000 / 64 = ~78 actual writes per thread
- The overlap check is cheap (4 integer comparisons) compared to the global memory write
- Net: 64x parallelism, 3.2x work amplification, for a theoretical ~20x speedup

**WGSL implementation:**

```wgsl
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let offspring_id = wid.x;
    let offspring_count = arrayLength(&working_states);
    if offspring_id >= offspring_count { return; }

    let w = params.image_width;
    let h = params.image_height;
    let num_tiles_x = (w + TILE_W - 1u) / TILE_W;
    let num_tiles_y = (h + TILE_H - 1u) / TILE_H;
    let num_tiles = num_tiles_x * num_tiles_y;
    let poly_count = working_states[offspring_id].polygon_count;

    // Each thread processes a stripe of tiles
    for (var t = lid; t < num_tiles; t += 64u) {
        let tile_x = t % num_tiles_x;
        let tile_y = t / num_tiles_x;

        // Tile pixel bounds
        let tile_px_min_x = tile_x * TILE_W;
        let tile_px_max_x = min(tile_px_min_x + TILE_W, w);
        let tile_px_min_y = tile_y * TILE_H;
        let tile_px_max_y = min(tile_px_min_y + TILE_H, h);

        // Convert to normalized coords for AABB comparison
        let tile_fmin_x = f32(tile_px_min_x) / f32(w);
        let tile_fmax_x = f32(tile_px_max_x) / f32(w);
        let tile_fmin_y = f32(tile_px_min_y) / f32(h);
        let tile_fmax_y = f32(tile_px_max_y) / f32(h);

        var count = 0u;
        let data_base = offspring_id * num_tiles * TILE_MAX_POLYS + t * TILE_MAX_POLYS;

        for (var pi = 0u; pi < poly_count; pi++) {
            let poly = working_states[offspring_id].polygons[pi];
            let v0 = unpack_vertex(poly.data.y);
            let v1 = unpack_vertex(poly.data.z);
            let v2 = unpack_vertex(poly.data.w);

            let bb_min_x = min(v0.x, min(v1.x, v2.x));
            let bb_max_x = max(v0.x, max(v1.x, v2.x));
            let bb_min_y = min(v0.y, min(v1.y, v2.y));
            let bb_max_y = max(v0.y, max(v1.y, v2.y));

            // AABB overlap test
            if bb_max_x >= tile_fmin_x && bb_min_x < tile_fmax_x &&
               bb_max_y >= tile_fmin_y && bb_min_y < tile_fmax_y {
                if count < TILE_MAX_POLYS {
                    tile_data[data_base + count] = pi;
                    count++;
                }
            }
        }
        // Non-atomic store: this thread exclusively owns this tile slot
        atomicStore(&tile_counts[offspring_id * num_tiles + t], count);
    }
}
```

---

### Option D: Shared Memory Cooperative Load + Per-Tile Scatter

Combine cooperative polygon loading (like the rasterize shader already does) with per-tile scatter:

1. Load a batch of 64 polygons into shared memory cooperatively
2. Each thread owns a tile and checks the batch against its tile
3. Repeat for all polygon batches

This amortizes global memory reads across all 64 threads (each polygon loaded once, checked by all threads).

```wgsl
var<workgroup> shared_polys: array<Polygon, 64>;
var<workgroup> shared_poly_base: u32;

@compute @workgroup_size(64, 1, 1)
fn main(...) {
    // Each thread owns ceil(num_tiles/64) tiles
    // For each tile, maintain a write cursor in registers

    let poly_count = working_states[offspring_id].polygon_count;
    let batch_count = (poly_count + 63u) / 64u;

    // Process tiles in round-robin
    for (var t = lid; t < num_tiles; t += 64u) {
        // Reset this tile's count
        atomicStore(&tile_counts[offspring_id * num_tiles + t], 0u);
    }
    workgroupBarrier();

    // Tile ownership: each thread processes tiles lid, lid+64, lid+128...
    // We need to maintain per-tile state across polygon batches.
    // With 1024 tiles / 64 threads = 16 tiles per thread max.
    // Store per-tile counts in registers (array of 16).
    var my_tile_counts: array<u32, 16>;  // NOTE: WGSL has no variable-length arrays
    // ... this approach requires knowing max tiles per thread at compile time

    for (var batch = 0u; batch < batch_count; batch++) {
        let base = batch * 64u;
        // Cooperative load
        if base + lid < poly_count {
            shared_polys[lid] = working_states[offspring_id].polygons[base + lid];
        }
        workgroupBarrier();

        let batch_end = min(64u, poly_count - base);

        // Each thread checks all polygons in batch against its owned tiles
        for (var t_idx = 0u; t_idx < tiles_per_thread; t_idx++) {
            let t = lid + t_idx * 64u;
            if t >= num_tiles { break; }
            // ... check batch against tile t, write to tile_data
        }
        workgroupBarrier();
    }
}
```

**Problem:** Each thread owns multiple tiles (up to 16 at 1024 tiles / 64 threads). The inner loop must check each of the 64 polygons in the batch against each of the 16 tiles per thread = 1024 checks per thread per batch. With 16 batches (1000 polys / 64), that's 16,384 checks per thread. Same total work as Option C but with shared memory benefit for polygon loads.

**Pros:**
- Cooperative load amortizes global memory reads (each polygon loaded once, not 64 times)
- Cache-friendly access pattern for polygon data

**Cons:**
- More complex than Option C
- Requires managing per-tile state across batches (write cursors)
- WGSL `var<workgroup>` arrays must have compile-time-known sizes
- Register pressure from per-tile counts array

**Estimated speedup over Option C:** Modest (polygon loads are 16 bytes each; with 1000 polygons that's 16KB per offspring -- fits in L2 cache either way).

---

## Correctness Proof for Z-Order Preservation

All options that iterate polygons in ascending index order within each tile trivially preserve z-order:

**Options A & B (serial scatter):** The single thread iterates `pi = 0, 1, 2, ..., N-1`. For each polygon, it appends to each overlapping tile. Since `pi` is monotonically increasing and the append slot is allocated by `atomicAdd` (which returns consecutive indices when called by a single thread), the resulting tile list is sorted by polygon index. QED.

**Option C (per-tile parallel):** Each thread owns a tile exclusively and iterates `pi = 0, 1, 2, ..., N-1`, writing to `tile_data[base + count]` with monotonically increasing `count`. Since `pi` increases monotonically and each write goes to the next slot, the resulting tile list is sorted by polygon index. No atomics are needed because tile ownership is exclusive. QED.

**Option D:** Same as Option C but with batched polygon loading. Within each batch, polygons are loaded in order and checked in order. Across batches, the outer loop iterates batches in ascending order. So polygon indices are checked in ascending order per tile. QED.

---

## Buffer Requirements

| Option | New Buffers | Memory Overhead |
|--------|-------------|----------------|
| A (parallel reset + serial scatter) | None | 0 |
| B (counting sort) | `tile_offsets` + `tile_write_counters` | 2 * offspring_capacity * num_tiles * 4 bytes = ~2MB |
| C (per-tile parallel) | None | 0 |
| D (shared memory cooperative) | None | 0 |

---

## Dispatch Configuration

| Option | Workgroup Size | Dispatch | Passes |
|--------|---------------|----------|--------|
| A | (64, 1, 1) | (offspring_count, 1, 1) | 1 |
| B | (64, 1, 1) count + (1, 1, 1) scatter | (offspring_count, 1, 1) + prefix_sum + (offspring_count, 1, 1) | 3 |
| C | (64, 1, 1) | (offspring_count, 1, 1) | 1 |
| D | (64, 1, 1) | (offspring_count, 1, 1) | 1 |

---

## Performance Analysis

### Current Serial Implementation

For N polygons covering T_avg tiles each, with M total tiles:
- Reset: M atomic stores (serial) = O(M)
- Scatter: N * T_avg atomic adds + writes (serial) = O(N * T_avg)
- Total per offspring: O(M + N * T_avg)
- Typical: M=1024, N=500, T_avg=5 -> 1024 + 2500 = 3524 serial operations
- Wall time: ~3524 * ~100ns (global atomic latency) = ~350us per offspring
- With 256 offspring on 170 SMs: ~352us * 256/170 = ~530us total

### Option A (Parallel Reset)

- Reset: M/64 atomic stores (parallel) = O(M/64)
- Scatter: unchanged O(N * T_avg) serial
- Total per offspring: O(M/64 + N * T_avg)
- Typical: 16 + 2500 = 2516 serial operations (1.4x faster per offspring)

### Option C (Per-Tile Parallel)

- Work per thread: N * (M_owned) overlap checks, M_owned = ceil(M/64) = 16
- Total per thread: 500 * 16 = 8000 overlap checks (each = 4 integer compares = cheap)
- But: ~78 writes per thread (5000 total / 64 threads)
- Wall time: dominated by 8000 cheap checks + 78 writes per thread
- Compared to serial: 3524 atomic writes per offspring
- Net: ~8000 checks + 78 non-atomic writes vs 3524 atomic writes
- The integer compare is ~2 cycles; atomic global write is ~100+ cycles
- Option C: 8000 * 2 + 78 * 50 = ~20,000 cycles per thread
- Serial: 3524 * 100 = ~352,000 cycles per thread (1 thread)
- Speedup: ~17x per offspring, but amortized across 64 threads already

### Comparison Against Other Passes

From the `PassTimings` structure, only 3 timestamps are collected:
- `mutate_ns`
- `rasterize_error_ns` (includes bin_polygons when tile culling is on!)
- `select_ns`

bin_polygons is included inside the rasterize_error timing window. Typical timing breakdown (from gpu-optimization-todo.md context):
- mutate: ~5-10% of total
- rasterize+error: ~80-90% of total (binning is a small fraction of this)
- select: ~5-10% of total

At 512x512 with 16x16 tiles, 500 polygons, lambda=64, 4 chains:
- Rasterize (per pixel, per offspring): 500 polygon tests * 262,144 pixels = ~131M ops per offspring
- Binning (per offspring): ~3500 operations
- Ratio: binning is ~0.003% of rasterize work per offspring

Even a 20x speedup on binning (Option C) saves 0.003% * 19/20 = 0.0028% of rasterize pass time. This is unmeasurably small.

---

## Honest Assessment: Is This Worth Doing?

**No, fully parallel scatter (Options B/C/D) is not worth the complexity.**

The binning pass is a negligible fraction of total GPU time. The rasterize_error pass dominates at 80-90%, and within that, binning is <0.01% of the work. Even Option C's theoretical 20x speedup on binning translates to an unmeasurable improvement in end-to-end performance.

**Option A (parallel tile reset) is worth doing as a trivial cleanup.** It is ~10 lines of code, zero risk, and makes the shader less embarrassingly single-threaded. But it should not be a priority.

**The real optimization opportunities are elsewhere:**
1. **Incremental evaluation improvements** (Tier 1.2, 1.3 in gpu-optimization-todo.md): 2-7.5x speedup on the rasterize pass
2. **Coarse-to-fine screening** (Tier 3.1): up to 8x rasterize speedup
3. **Stratified mutations** (Tier 1.4): 15-25% faster convergence
4. **SoA shared memory layout** (Tier 4.1): 33% more polys/tile in rasterize

All of these target the rasterize pass (where 80-90% of time is spent) rather than the binning pass (where <0.01% is spent).

### When Would Parallelizing Binning Matter?

Only if:
1. The image resolution is very high (2048x2048+) AND tile size is very small (4x4), creating 262,144 tiles per offspring -- then reset + scatter becomes significant
2. The polygon count is extremely high (5000+) -- but MAX_POLYGONS_PER_IMAGE is capped at 1000
3. The rasterize pass is somehow eliminated or reduced to near-zero (e.g., via perfect incremental eval) -- then binning becomes a larger fraction

None of these scenarios apply to the current system.

---

## Recommendation

1. **Implement Option A** (parallel tile reset only) as a low-risk cleanup:
   - Change `@workgroup_size(1,1,1)` to `@workgroup_size(64,1,1)`
   - Parallelize the tile reset loop across 64 threads
   - Keep the polygon scatter serial on thread 0
   - ~10 lines of code, zero correctness risk

2. **Do not implement Options B/C/D.** The complexity-to-benefit ratio is extremely unfavorable. The engineering time is better spent on rasterize pass optimizations.

3. **If binning ever becomes a bottleneck** (e.g., after rasterize is optimized 10x via incremental eval), revisit Option C (per-tile parallel scatter) as the simplest fully-parallel approach. It requires no new buffers, no prefix sum, and has a clean correctness argument.
