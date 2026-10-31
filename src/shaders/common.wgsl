// Shared by every bundled stage. gpu_evolver::programs appends this file, after
// the generated constants, to each stage source; stage files do not repeat it.
// The structs mirror gpu_evolver::buffers, and CPU tests compare their layouts.

struct Polygon {
    data: vec4<u32>,   // [color_packed, v0_packed, v1_packed, v2_packed] — 16 bytes
}

// GpuDrawingState. An offspring's change record (changes.wgsl) is its
// fitness_bits, dirty_max and rng_state.yz; parents leave dirty_max zero.
struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    dirty_max: u32,            // offspring: dirty rectangle's max corner
    _pad: u32,
    rng_state: vec4<u32>,      // .x RNG state; .yz offspring changed-index words
    polygons: array<Polygon, POLYGON_CAPACITY>,
}

// GpuParams; 256 bytes.
struct Params {
    // Image size and polygon-count bounds
    image_width: u32,
    image_height: u32,
    max_polygons: u32,
    min_polygons: u32,

    // Population and fitness
    chain_count_param: u32,
    lambda: u32,
    max_error_per_pixel: f32,
    per_point_penalty: f32,

    // Crossover
    crossover_prob: f32,
    tournament_size: u32,
    _pad0: vec2<u32>,

    // Mutation operator weights
    add_polygon_prob: f32,
    remove_polygon_prob: f32,
    reorder_polygon_prob: f32,
    offset_polygon_prob: f32,

    move_point_prob: f32,
    micro_adjust_prob: f32,
    change_color_prob: f32,
    adjust_brightness_prob: f32,

    adjust_saturation_prob: f32,
    scale_polygon_prob: f32,
    rotate_polygon_prob: f32,
    adjacent_swap_prob: f32,

    merge_polygon_prob: f32,
    clone_polygon_prob: f32,
    medium_move_prob: f32,
    swap_colors_prob: f32,

    // Last operator weight, then the alpha range of new and recolored polygons
    micro_color_prob: f32,
    min_alpha_norm: f32,
    max_alpha_norm: f32,
    _pad1: u32,

    // Step sizes, multiplied by the offspring's step_multiplier()
    offset_polygon_magnitude: f32,
    move_point_max_delta: f32,
    medium_move_delta: f32,
    micro_adjust_delta: f32,

    // New vertex spread (add, clone) and merge thresholds
    new_point_max_distance: f32,
    merge_centroid_threshold: f32,
    merge_color_threshold: f32,
    _pad2: u32,

    // Step-size multiplier per rung (vec4[10-11]); mutate's step_multiplier() reads it
    step_multipliers: array<vec4<f32>, 2>,

    // Reserved (vec4[12-15])
    _reserved: array<vec4<u32>, 4>,
}
const_assert STEP_RUNGS == 8u;

// --- Quantized polygon unpack helpers ---

fn unpack_color(p: Polygon) -> vec4<f32> {
    return unpack4x8unorm(p.data.x);
}

fn unpack_vertex(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xFFFFu) / 65535.0, f32(word >> 16u) / 65535.0);
}

// Half-space edge function: positive if point (px,py) is on the left side of edge (ax,ay)->(bx,by)
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}
