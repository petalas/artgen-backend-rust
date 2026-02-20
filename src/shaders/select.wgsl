// Selection + Migration compute shader — 64 threads per chain (parallel polygon copy)
// Three entry points: `select_main`, `migrate_intra_main`, `migrate_inter_main`

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
    crossover_prob: f32,

    // Crossover & island params
    spatial_crossover_weight: f32,
    tournament_size: u32,
    island_count: u32,
    inter_island_interval: u32,

    // Chain count + lambda + padding
    chain_count_param: u32,
    single_mutation_mode: u32,
    lambda: u32,
    adaptive_mutation: u32,
}

fn unpack_vertex(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xFFFFu) / 65535.0, f32(word >> 16u) / 65535.0);
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
@group(0) @binding(5) var<storage, read_write> fitness_packed:     array<u32>;

// Workgroup-shared variables for communicating decisions from thread 0 to all threads
var<workgroup> shared_accept: u32;
var<workgroup> shared_copy_count: u32;
var<workgroup> shared_best_offspring_id: u32;

/// Compute fitness from total error and polygon count.
fn compute_fitness(total_error: u32, polygon_count: u32) -> f32 {
    let w = params.image_width;
    let h = params.image_height;
    let max_total_error = params.max_error_per_pixel * f32(w * h);
    var fitness = 100.0 * (1.0 - f32(total_error) / max_total_error);
    let num_points = polygon_count * 3u;
    fitness -= fitness * params.per_point_multiplier * f32(num_points);
    return fitness;
}

@compute @workgroup_size(64)
fn select_main(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {
    let chain_id = wid.x;
    let local_id = lid.x;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }

    // Thread 0 does all decision logic
    if local_id == 0u {
        let lambda = params.lambda;

        // Find best offspring among λ candidates
        var best_error = 0xFFFFFFFFu;
        var best_local = 0u;
        for (var i = 0u; i < lambda; i++) {
            let oid = chain_id * lambda + i;
            let err = atomicExchange(&error_accumulators[oid], 0u);
            if err < best_error {
                best_error = err;
                best_local = i;
            }
        }
        let best_offspring_id = chain_id * lambda + best_local;

        let fitness = compute_fitness(best_error, working_states[best_offspring_id].polygon_count);
        let fitness_bits = bitcast<u32>(fitness);

        // Compare against chain's current best
        let current_fitness_bits = chain_states[chain_id].fitness_bits;
        let current_fitness = bitcast<f32>(current_fitness_bits);

        // Accept if strictly better, or with 50% probability if equal (plateau traversal)
        // Use the best offspring's RNG for neutral acceptance
        let dominated = fitness > current_fitness;
        let neutral = fitness == current_fitness && (working_states[best_offspring_id].rng_state.x & 1u) == 1u;
        let should_accept = dominated || neutral;

        if should_accept {
            chain_states[chain_id].polygon_count = working_states[best_offspring_id].polygon_count;
            chain_states[chain_id].fitness_bits = fitness_bits;
            chain_states[chain_id].stagnation_counter = 0u;
            shared_accept = 1u;
            shared_copy_count = working_states[best_offspring_id].polygon_count;
            shared_best_offspring_id = best_offspring_id;

            // Adaptive mutation scale: only update when enabled
            if params.adaptive_mutation == 1u {
                let new_scale = min(chain_states[chain_id].mutation_scale * 1.2, 2.0);
                chain_states[chain_id].mutation_scale = new_scale;
            }
        } else {
            if params.adaptive_mutation == 1u {
                // Gentle decay on rejection
                // pow(0.99, 1/λ) — at 5% acceptance rate, geometric mean ≈ 1.0 (stable)
                let decay = pow(0.99, 1.0 / f32(lambda));
                let new_scale = max(chain_states[chain_id].mutation_scale * decay, 0.2);
                chain_states[chain_id].mutation_scale = new_scale;

                // Stagnation counter
                let stagnation = chain_states[chain_id].stagnation_counter + 1u;
                let stagnation_threshold = 10000u / lambda;
                if stagnation > stagnation_threshold {
                    chain_states[chain_id].mutation_scale = 1.5;
                    chain_states[chain_id].stagnation_counter = 0u;
                } else {
                    chain_states[chain_id].stagnation_counter = stagnation;
                }
            }

            shared_accept = 0u;
        }

        // Export actual chain fitness (after acceptance) for CPU readback
        fitness_packed[chain_id] = chain_states[chain_id].fitness_bits;

        // Track global best using atomicMax on IEEE 754 bit pattern
        let best_bits = chain_states[chain_id].fitness_bits;
        let old_best_bits = atomicMax(&control.best_fitness_bits, best_bits);
        if best_bits > old_best_bits {
            atomicStore(&control.best_chain_id, chain_id);
            atomicStore(&control.new_best_found, 1u);
        }
    }

    workgroupBarrier();

    // ALL threads cooperate on the polygon copy if accepted
    if shared_accept == 1u {
        let count = shared_copy_count;
        let src = shared_best_offspring_id;
        for (var i = local_id; i < count; i += 64u) {
            chain_states[chain_id].polygons[i] = working_states[src].polygons[i];
        }
    }

    workgroupBarrier();

    // Degenerate triangle culling: thread 0 compacts chain_states after copy
    // Only runs on acceptance (rare), so minimal performance impact
    if shared_accept == 1u && local_id == 0u {
        let count = shared_copy_count;
        var write_idx = 0u;
        for (var r = 0u; r < count; r++) {
            let p = chain_states[chain_id].polygons[r];
            let pv0 = unpack_vertex(p.data.y);
            let pv1 = unpack_vertex(p.data.z);
            let pv2 = unpack_vertex(p.data.w);
            // Cross product magnitude (2x area)
            let cross = abs((pv1.x - pv0.x) * (pv2.y - pv0.y) - (pv2.x - pv0.x) * (pv1.y - pv0.y));
            if cross > 0.00001 {
                chain_states[chain_id].polygons[write_idx] = p;
                write_idx++;
            }
        }
        if write_idx < count && write_idx >= params.min_polygons {
            chain_states[chain_id].polygon_count = write_idx;
        }
    }
}

/// Migrate if neighbor is fitter. Shared logic for both intra/inter migration.
/// Uses module-scope shared_accept and shared_copy_count for workgroup communication.
fn migrate_from(chain_id: u32, neighbor_id: u32, local_id: u32) {
    if local_id == 0u {
        let neighbor_fitness = bitcast<f32>(chain_states[neighbor_id].fitness_bits);
        let my_fitness = bitcast<f32>(chain_states[chain_id].fitness_bits);

        if neighbor_fitness > my_fitness {
            // Adopt neighbor's drawing, but keep our own RNG for diversity
            let saved_rng = chain_states[chain_id].rng_state;

            let pc = min(chain_states[neighbor_id].polygon_count, params.max_polygons);
            chain_states[chain_id].polygon_count = pc;
            chain_states[chain_id].fitness_bits = chain_states[neighbor_id].fitness_bits;
            chain_states[chain_id].mutation_scale = chain_states[neighbor_id].mutation_scale;
            chain_states[chain_id].stagnation_counter = 0u;
            chain_states[chain_id].rng_state = saved_rng;
            shared_accept = 1u;
            shared_copy_count = pc;
        } else {
            shared_accept = 0u;
        }

        // Update fitness_packed with (possibly migrated) fitness
        fitness_packed[chain_id] = chain_states[chain_id].fitness_bits;
    }

    workgroupBarrier();

    if shared_accept == 1u {
        let pc = shared_copy_count;
        for (var i = local_id; i < pc; i += 64u) {
            chain_states[chain_id].polygons[i] = chain_states[neighbor_id].polygons[i];
        }
    }
}

/// Intra-island migration: ring within island boundaries.
@compute @workgroup_size(64)
fn migrate_intra_main(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>,
                      @builtin(workgroup_id) wid: vec3<u32>) {
    let chain_id = wid.x;
    let local_id = lid.x;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }

    let island_size = params.chain_count_param / max(params.island_count, 1u);
    let island_start = (chain_id / island_size) * island_size;
    let local_idx = chain_id - island_start;
    let neighbor_id = island_start + (local_idx + 1u) % island_size;

    migrate_from(chain_id, neighbor_id, local_id);
}

/// Inter-island migration: global ring across all chains.
@compute @workgroup_size(64)
fn migrate_inter_main(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>,
                      @builtin(workgroup_id) wid: vec3<u32>) {
    let chain_id = wid.x;
    let local_id = lid.x;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }

    let neighbor_id = (chain_id + 1u) % chain_count;
    migrate_from(chain_id, neighbor_id, local_id);
}
