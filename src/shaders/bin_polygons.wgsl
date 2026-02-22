// Polygon binning compute shader — assigns each polygon to the spatial tiles it overlaps.
// Dispatch: (offspring_count, 1, 1) — one workgroup per offspring, one thread does the serial iteration.
// Serial fill ensures polygon indices appear in original order (required for alpha blending).
//
// This shader runs between mutate and rasterize_error when tile culling is enabled.

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
    _pad1: u32,
    tile_culling: u32,

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

    // Merge thresholds + integer_aabb toggle
    merge_centroid_threshold: f32,
    merge_color_threshold: f32,
    integer_aabb: u32,
    _pad3: u32,

    // Reserved padding (vec4[11-15])
    _reserved0: vec4<u32>,
    _reserved1: vec4<u32>,
    _reserved2: vec4<u32>,
    _reserved3: vec4<u32>,
    _reserved4: vec4<u32>,
}

fn unpack_vertex(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xFFFFu) / 65535.0, f32(word >> 16u) / 65535.0);
}

// Tile size must match rasterize_error workgroup size.
// These are replaced by pipeline.rs at pipeline creation time (same as rasterize shader).
const TILE_W: u32 = 16;
const TILE_H: u32 = 16;
const TILE_MAX_POLYS: u32 = 256u;

@group(0) @binding(0) var<storage, read>       working_states: array<DrawingState>;
@group(0) @binding(1) var<storage, read_write> tile_data:      array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_counts:    array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             params:         Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let offspring_id = wid.x;
    let offspring_count = arrayLength(&working_states);
    if offspring_id >= offspring_count {
        return;
    }

    let w = params.image_width;
    let h = params.image_height;
    let num_tiles_x = (w + TILE_W - 1u) / TILE_W;
    let num_tiles_y = (h + TILE_H - 1u) / TILE_H;
    let num_tiles = num_tiles_x * num_tiles_y;

    // Reset tile counts for this offspring
    for (var t = 0u; t < num_tiles; t++) {
        atomicStore(&tile_counts[offspring_id * num_tiles + t], 0u);
    }

    let poly_count = working_states[offspring_id].polygon_count;

    // Serial iteration: process polygons in order to preserve alpha blending order
    for (var pi = 0u; pi < poly_count; pi++) {
        let poly = working_states[offspring_id].polygons[pi];

        // Unpack vertices
        let v0 = unpack_vertex(poly.data.y);
        let v1 = unpack_vertex(poly.data.z);
        let v2 = unpack_vertex(poly.data.w);

        // AABB in normalized coords
        let bb_min_x = min(v0.x, min(v1.x, v2.x));
        let bb_max_x = max(v0.x, max(v1.x, v2.x));
        let bb_min_y = min(v0.y, min(v1.y, v2.y));
        let bb_max_y = max(v0.y, max(v1.y, v2.y));

        // Convert to pixel coords
        let px_min_x = u32(bb_min_x * f32(w));
        let px_max_x = min(u32(ceil(bb_max_x * f32(w))), w);
        let px_min_y = u32(bb_min_y * f32(h));
        let px_max_y = min(u32(ceil(bb_max_y * f32(h))), h);

        // Convert to tile coords
        let tile_min_x = px_min_x / TILE_W;
        let tile_max_x = min((px_max_x + TILE_W - 1u) / TILE_W, num_tiles_x);
        let tile_min_y = px_min_y / TILE_H;
        let tile_max_y = min((px_max_y + TILE_H - 1u) / TILE_H, num_tiles_y);

        // Append polygon index to each overlapping tile
        for (var ty = tile_min_y; ty < tile_max_y; ty++) {
            for (var tx = tile_min_x; tx < tile_max_x; tx++) {
                let tile_id = ty * num_tiles_x + tx;
                let tile_global = offspring_id * num_tiles + tile_id;
                let slot = atomicAdd(&tile_counts[tile_global], 1u);
                if slot < TILE_MAX_POLYS {
                    let data_offset = offspring_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS + slot;
                    tile_data[data_offset] = pi;
                }
            }
        }
    }
}
