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
// test is tiles.wgsl's tile_consumed, the second offspring_changes.wgsl's
// changed_meets_rect, as in bin_polygons.wgsl.
//
// raster_work layout (u32 words): [0] entry count, [1 + i] entry i =
// offspring * num_tiles + tile_id. update_framebuffers resets the count at the
// end of every iteration; the host resets it around a full evaluation. Entry
// order is unspecified: raster only adds integer sums.
//
// common.wgsl, changes.wgsl, offspring_changes.wgsl, tiles.wgsl and the
// generated constants (TILE_W, TILE_H, TILE_MAX_POLYS, CHANGED_INDICES,
// WORK_COUNT, WORK_ENTRIES) are appended by programs.rs.
const LANES: u32 = 64u;

@group(0) @binding(0) var<storage, read>       working_states: array<DrawingState>;
@group(0) @binding(1) var<storage, read>       chain_states:   array<DrawingState>;
@group(0) @binding(2) var<storage, read>       tile_counts:    array<u32>;
@group(0) @binding(3) var<storage, read_write> raster_work:    array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             params:         Params;

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
    let changed = changes_shortcut(offspring_changed_words(chain_id), raw_count);
    if changed.x == 0u {
        // Shortcut disabled: every pixel blends.
        return true;
    }
    let tile_lo = vec2<f32>(f32(tile_min_x) / f32(w), f32(tile_min_y) / f32(h));
    let tile_hi = vec2<f32>(f32(tile_max_x) / f32(w), f32(tile_max_y) / f32(h));
    return changed_meets_rect(changed, parent_chain, chain_id, tile_lo, tile_hi);
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

    let dirty_min = offspring_dirty_min(chain_id);
    let dirty_max = offspring_dirty_max(chain_id);

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
