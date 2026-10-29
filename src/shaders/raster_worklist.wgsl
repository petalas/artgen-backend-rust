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
// again.) Exact copies (NO_CHANGES) list no index and need no tile. Both tests
// are copied from the former rasterize_error code and must stay identical.
//
// raster_work layout (u32 words): [0] entry count, [1 + i] entry i =
// offspring * num_tiles + tile_id. update_framebuffers resets the count at the
// end of every iteration; the host resets it around a full evaluation. Entry
// order is unspecified: raster only adds integer sums.
//
// pipeline.rs specializes the exact WG_X/WG_Y markers below.
const WG_X: u32 = 16;
const WG_Y: u32 = 16;
const TILE_MAX_POLYS: u32 = 256u;
const LANES: u32 = 64u;
const WORK_COUNT: u32 = 0u;
const WORK_ENTRIES: u32 = 1u;

struct Polygon {
    data: vec4<u32>,   // [color_packed, v0_packed, v1_packed, v2_packed] — 16 bytes
}

struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    mutation_scale: f32,
    stagnation_counter: u32,
    rng_state: vec4<u32>,
    polygons: array<Polygon, 1000>,
}

struct Params {
    image_width: u32,
    image_height: u32,
    max_polygons: u32,
    min_polygons: u32,

    max_error_per_pixel: f32,
    per_point_multiplier: f32,
    iteration_number: u32,
    _pad0: u32,

    add_polygon_prob: f32,
    remove_polygon_prob: f32,
    reorder_polygon_prob: f32,
    offset_polygon_prob: f32,

    move_point_prob: f32,
    micro_adjust_prob: f32,
    change_color_prob: f32,
    adjust_brightness_prob: f32,

    adjust_saturation_prob: f32,
    move_point_max_delta: f32,
    micro_adjust_delta: f32,
    new_point_max_distance: f32,

    offset_polygon_magnitude: f32,
    min_alpha_norm: f32,
    max_alpha_norm: f32,
    crossover_prob: f32,

    // Crossover params
    spatial_crossover_weight: f32,
    tournament_size: u32,
    _pad_ie: u32,
    _pad_tc: u32,

    // Chain count + lambda + padding
    chain_count_param: u32,
    single_mutation_mode: u32,
    lambda: u32,
    adaptive_mutation: u32,

    // New mutation probabilities
    scale_polygon_prob: f32,
    rotate_polygon_prob: f32,
    adjacent_swap_prob: f32,
    merge_polygon_prob: f32,

    clone_polygon_prob: f32,
    medium_move_prob: f32,
    medium_move_delta: f32,
    swap_colors_prob: f32,

    // Merge thresholds + padding
    merge_centroid_threshold: f32,
    merge_color_threshold: f32,
    _pad2: u32,
    _pad3: u32,

    // Reserved padding (vec4[11-15])
    _reserved0: vec4<u32>,
    _reserved1: vec4<u32>,
    _reserved2: vec4<u32>,
    _reserved3: vec4<u32>,
    _reserved4: vec4<u32>,
}

@group(0) @binding(0) var<storage, read>       working_states: array<DrawingState>;
@group(0) @binding(1) var<storage, read>       chain_states:   array<DrawingState>;
@group(0) @binding(2) var<storage, read>       tile_counts:    array<u32>;
@group(0) @binding(3) var<storage, read_write> raster_work:    array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             params:         Params;

// Must match rasterize_error.wgsl.
const CHANGED_INDICES: u32 = 4u;
const COVER_MARGIN: f32 = 1e-5;

fn unpack_vertex(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xFFFFu) / 65535.0, f32(word >> 16u) / 65535.0);
}

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
    let tile_min_x = tile_x * WG_X;
    let tile_min_y = tile_y * WG_Y;
    let tile_max_x = min(tile_min_x + WG_X, w);
    let tile_max_y = min(tile_min_y + WG_Y, h);
    if tile_max_x <= dirty_min.x || tile_min_x >= dirty_max.x ||
       tile_max_y <= dirty_min.y || tile_min_y >= dirty_max.y {
        return false;
    }
    let num_tiles_x = (w + WG_X - 1u) / WG_X;
    let num_tiles_y = (h + WG_Y - 1u) / WG_Y;
    let num_tiles = num_tiles_x * num_tiles_y;
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
    let num_tiles_x = (w + WG_X - 1u) / WG_X;
    let num_tiles_y = (h + WG_Y - 1u) / WG_Y;
    let num_tiles = num_tiles_x * num_tiles_y;

    let bbox_lo = working_states[chain_id].fitness_bits;
    let bbox_hi = working_states[chain_id].stagnation_counter;
    let dirty_min = vec2<u32>(bbox_lo & 0xFFFFu, (bbox_lo >> 16u) & 0xFFFFu);
    let dirty_max = vec2<u32>(bbox_hi & 0xFFFFu, (bbox_hi >> 16u) & 0xFFFFu);

    // Superset of the tiles meeting the dirty rectangle; tile_needed applies
    // the exact test. Columns before col_lo end at or before dirty_min.x and
    // columns from col_hi start at or after dirty_max.x (same for rows).
    let col_lo = dirty_min.x / WG_X;
    let col_hi = min((dirty_max.x + WG_X - 1u) / WG_X, num_tiles_x);
    let row_lo = dirty_min.y / WG_Y;
    let row_hi = min((dirty_max.y + WG_Y - 1u) / WG_Y, num_tiles_y);
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
