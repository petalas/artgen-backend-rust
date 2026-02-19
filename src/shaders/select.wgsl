// Selection + Migration compute shader — one thread per chain
// Two entry points: `select_main` and `migrate_main`

struct Polygon {
    color: vec4<f32>,
    v0: vec2<f32>,
    v1: vec2<f32>,
    v2: vec2<f32>,
    _pad: vec2<f32>,
}

struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    _pad0: u32,
    _pad1: u32,
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
    migration_interval: u32,

    add_polygon_prob: f32,
    remove_polygon_prob: f32,
    reorder_polygon_prob: f32,
    offset_polygon_prob: f32,

    move_point_prob: f32,
    micro_adjust_prob: f32,
    change_color_prob: f32,
    lighten_color_prob: f32,

    darken_color_prob: f32,
    move_point_max_delta: f32,
    micro_adjust_delta: f32,
    new_point_max_distance: f32,

    offset_polygon_magnitude: f32,
    min_alpha_norm: f32,
    max_alpha_norm: f32,
    _params_pad: u32,
}

struct ControlFlags {
    new_best_found: atomic<u32>,
    best_chain_id: atomic<u32>,
    best_fitness_bits: atomic<u32>,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> chain_states:       array<DrawingState>;
@group(0) @binding(1) var<storage, read>       working_states:     array<DrawingState>;
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> control:            ControlFlags;
@group(0) @binding(4) var<uniform>             params:             Params;

@compute @workgroup_size(1)
fn select_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_id = gid.x;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }

    // Read and reset error accumulator
    let total_error = atomicExchange(&error_accumulators[chain_id], 0u);

    // Compute fitness: 100 * (1 - error / max_total_error)
    let w = params.image_width;
    let h = params.image_height;
    let max_total_error = params.max_error_per_pixel * f32(w * h);
    var fitness = 100.0 * (1.0 - f32(total_error) / max_total_error);

    // Complexity penalty: fitness * per_point_multiplier * num_points
    // Each GPU polygon is a triangle = 3 points
    let num_points = working_states[chain_id].polygon_count * 3u;
    fitness -= fitness * params.per_point_multiplier * f32(num_points);

    let fitness_bits = bitcast<u32>(fitness);

    // Compare against chain's current best
    let current_fitness_bits = chain_states[chain_id].fitness_bits;
    let current_fitness = bitcast<f32>(current_fitness_bits);

    if fitness > current_fitness {
        // Accept: copy working → chain_state (keep RNG state from chain_state)
        let saved_rng = chain_states[chain_id].rng_state;
        chain_states[chain_id].polygon_count = working_states[chain_id].polygon_count;
        chain_states[chain_id].fitness_bits = fitness_bits;
        chain_states[chain_id]._pad0 = 0u;
        chain_states[chain_id]._pad1 = 0u;
        chain_states[chain_id].rng_state = saved_rng;

        let pc = working_states[chain_id].polygon_count;
        for (var i = 0u; i < pc; i++) {
            chain_states[chain_id].polygons[i] = working_states[chain_id].polygons[i];
        }
    }

    // Track global best using atomicMax on IEEE 754 bit pattern
    // For positive floats, bit patterns sort the same as float values
    let old_best_bits = atomicMax(&control.best_fitness_bits, fitness_bits);
    if fitness_bits > old_best_bits {
        // We set a new global best
        atomicStore(&control.best_chain_id, chain_id);
        atomicStore(&control.new_best_found, 1u);
    }
}

@compute @workgroup_size(1)
fn migrate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_id = gid.x;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }

    // Read global best chain
    let best_id = atomicLoad(&control.best_chain_id);
    if best_id == chain_id {
        return; // Don't copy to self
    }

    let global_best_fitness = bitcast<f32>(atomicLoad(&control.best_fitness_bits));
    let my_fitness = bitcast<f32>(chain_states[chain_id].fitness_bits);

    // Only adopt if global best is significantly better
    if global_best_fitness > my_fitness {
        // Copy drawing from best chain, but keep our own RNG for diversity
        let saved_rng = chain_states[chain_id].rng_state;

        chain_states[chain_id].polygon_count = chain_states[best_id].polygon_count;
        chain_states[chain_id].fitness_bits = chain_states[best_id].fitness_bits;
        chain_states[chain_id]._pad0 = 0u;
        chain_states[chain_id]._pad1 = 0u;
        chain_states[chain_id].rng_state = saved_rng;

        let pc = chain_states[best_id].polygon_count;
        for (var i = 0u; i < pc; i++) {
            chain_states[chain_id].polygons[i] = chain_states[best_id].polygons[i];
        }
    }
}
