// Raster worklist: lists the (offspring, tile) pairs rasterize_error must run.
// Dispatch: (offspring_count, 1, 1), after bin_polygons and before
// rasterize_error. Workgroup c tests every tile of offspring c with the
// two workgroup-uniform early returns rasterize_error used to take itself:
//   1. the tile misses the offspring's dirty rectangle;
//   2. the shortcut is enabled and no parent or offspring version of a changed
//      index may cover a pixel of the tile.
// A tile that returned early added nothing to either error sum, so leaving it
// out keeps every offspring's error bit-identical. (Running a tile of the second
// kind would add each pixel's old error to both sums, which select subtracts
// again.) Exact copies (NO_CHANGES) list no index and need no tile. The first
// test is tiles.wgsl's tile_consumed; the second must stay identical to
// bin_polygons.wgsl's.
//
// raster_work layout (u32 words): [0] entry count, [1 + i] entry i =
// offspring * num_tiles + tile_id. update_framebuffers resets the count at the
// end of every iteration; the host resets it around a full evaluation. Entry
// order is unspecified: raster only adds integer sums.
//
// common.wgsl, tiles.wgsl and the generated constants (TILE_W, TILE_H,
// TILE_MAX_POLYS, WORK_COUNT, WORK_ENTRIES) are appended by programs.rs.
const LANES: u32 = 64u;

@group(0) @binding(0) var<storage, read>       working_states: array<DrawingState>;
@group(0) @binding(1) var<storage, read>       chain_states:   array<DrawingState>;
@group(0) @binding(2) var<storage, read>       tile_counts:    array<u32>;
@group(0) @binding(3) var<storage, read_write> raster_work:    array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             params:         Params;

// Must match rasterize_error.wgsl.
const CHANGED_INDICES: u32 = 4u;
const COVER_MARGIN: f32 = 1e-5;

// Same box as rasterize_error's decode_changed.
fn changed_may_meet_tile(poly: Polygon, lo: vec2<f32>, hi: vec2<f32>) -> bool {
    let pv0 = unpack_vertex(poly.data.y);
    let pv1 = unpack_vertex(poly.data.z);
    let pv2 = unpack_vertex(poly.data.w);
    let bb_min_x = min(pv0.x, min(pv1.x, pv2.x));
    let bb_max_x = max(pv0.x, max(pv1.x, pv2.x));
    let bb_min_y = min(pv0.y, min(pv1.y, pv2.y));
    let bb_max_y = max(pv0.y, max(pv1.y, pv2.y));
    return hi.x >= bb_min_x - COVER_MARGIN && lo.x <= bb_max_x + COVER_MARGIN &&
           hi.y >= bb_min_y - COVER_MARGIN && lo.y <= bb_max_y + COVER_MARGIN;
}

// Whether any parent or offspring version of the listed changed indices may
// cover a pixel of the tile. Indices at or beyond a drawing's count are empty.
fn changed_meets_tile(
    changed: vec2<u32>, parent_chain: u32, chain_id: u32, lo: vec2<f32>, hi: vec2<f32>,
) -> bool {
    let parent_count = chain_states[parent_chain].polygon_count;
    let offspring_count = working_states[chain_id].polygon_count;
    var meets = false;
    for (var half = 0u; half < CHANGED_INDICES; half++) {
        let word = select(changed.x, changed.y, half >= 2u);
        let encoded = (word >> (16u * (half & 1u))) & 0xFFFFu;
        if encoded != 0u {
            let idx = encoded - 1u;
            if idx < parent_count {
                meets = meets || changed_may_meet_tile(chain_states[parent_chain].polygons[idx], lo, hi);
            }
            if idx < offspring_count {
                meets = meets || changed_may_meet_tile(working_states[chain_id].polygons[idx], lo, hi);
            }
        }
    }
    return meets;
}

// Whether rasterize_error has work in this tile (the negation of its former
// early returns).
fn tile_needed(chain_id: u32, parent_chain: u32, tile_x: u32, tile_y: u32,
               dirty_min: vec2<u32>, dirty_max: vec2<u32>) -> bool {
    let w = params.image_width;
    let h = params.image_height;
    if !tile_consumed(vec2<u32>(tile_x, tile_y), w, h, dirty_min, dirty_max) {
        return false;
    }
    let tile_min_x = tile_x * TILE_W;
    let tile_min_y = tile_y * TILE_H;
    let tile_max_x = min(tile_min_x + TILE_W, w);
    let tile_max_y = min(tile_min_y + TILE_H, h);
    let grid = tile_grid(w, h);
    let num_tiles_x = grid.x;
    let num_tiles = num_tiles_x * grid.y;
    let tile_id = tile_y * num_tiles_x + tile_x;
    let raw_count = tile_counts[chain_id * num_tiles + tile_id];
    let first = working_states[chain_id].rng_state.y;
    var second = 0u;
    if first != 0u {
        second = working_states[chain_id].rng_state.z;
    }
    let listed = select(2u, CHANGED_INDICES, second != 0u);
    if raw_count + listed > TILE_MAX_POLYS || first == 0u {
        // Shortcut disabled: every pixel blends.
        return true;
    }
    let tile_lo = vec2<f32>(f32(tile_min_x) / f32(w), f32(tile_min_y) / f32(h));
    let tile_hi = vec2<f32>(f32(tile_max_x) / f32(w), f32(tile_max_y) / f32(h));
    return changed_meets_tile(vec2<u32>(first, second), parent_chain, chain_id, tile_lo, tile_hi);
}

@compute @workgroup_size(LANES, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32,
) {
    let chain_id = wid.x;
    if chain_id >= arrayLength(&working_states) {
        return;
    }
    let w = params.image_width;
    let h = params.image_height;
    let parent_chain = chain_id / params.lambda;
    let grid = tile_grid(w, h);
    let num_tiles_x = grid.x;
    let num_tiles_y = grid.y;
    let num_tiles = num_tiles_x * num_tiles_y;

    let dirty_min = unpack_dirty_corner(working_states[chain_id].fitness_bits);
    let dirty_max = unpack_dirty_corner(working_states[chain_id].stagnation_counter);

    // Superset of the consumed tiles; tile_needed applies the exact test.
    // Columns before col_lo end at or before dirty_min.x and columns from col_hi
    // start at or after dirty_max.x (same for rows).
    let col_lo = dirty_min.x / TILE_W;
    let col_hi = min((dirty_max.x + TILE_W - 1u) / TILE_W, num_tiles_x);
    let row_lo = dirty_min.y / TILE_H;
    let row_hi = min((dirty_max.y + TILE_H - 1u) / TILE_H, num_tiles_y);
    if col_hi <= col_lo || row_hi <= row_lo {
        return;
    }
    let cols = col_hi - col_lo;
    let rect_tiles = cols * (row_hi - row_lo);

    // The loop bound is workgroup-uniform, so every subgroup operation runs
    // with all lanes active.
    for (var round = 0u; round < rect_tiles; round += LANES) {
        let i = round + local_idx;
        var needed = false;
        var entry = 0u;
        if i < rect_tiles {
            let tile_x = col_lo + i % cols;
            let tile_y = row_lo + i / cols;
            needed = tile_needed(chain_id, parent_chain, tile_x, tile_y, dirty_min, dirty_max);
            entry = chain_id * num_tiles + tile_y * num_tiles_x + tile_x;
        }
        let flag = select(0u, 1u, needed);
        let n = subgroupAdd(flag);
        let prefix = subgroupExclusiveAdd(flag);
        var base = 0u;
        if sg_inv_id == 0u && n != 0u {
            base = atomicAdd(&raster_work[WORK_COUNT], n);
        }
        base = subgroupBroadcastFirst(base);
        if needed {
            atomicStore(&raster_work[WORK_ENTRIES + base + prefix], entry);
        }
    }
}
