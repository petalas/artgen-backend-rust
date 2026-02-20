// Mutation compute shader — one thread per chain
// Copies chain_states[i] → working_states[i], then applies probabilistic mutations.
// Loops until at least one mutation fires (is_dirty).

struct Polygon {
    color: vec4<f32>,   // RGBA normalized 0.0–1.0
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
    rng_state: vec4<u32>,  // .xy = state, .zw = increment
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

    // Chain count + padding
    chain_count_param: u32,
    _pad6: u32,
    _pad7: u32,
    _pad8: u32,
}

@group(0) @binding(0) var<storage, read>       chain_states:   array<DrawingState>;
@group(0) @binding(1) var<storage, read_write>  working_states: array<DrawingState>;
@group(0) @binding(2) var<uniform>              params:         Params;

// --- PCG32 RNG ---
// PCG-XSH-RR: high-quality, fast, minimal state
fn pcg_step(state: ptr<function, vec4<u32>>) -> u32 {
    let s64_lo = (*state).x;
    let s64_hi = (*state).y;
    let inc_lo = (*state).z;
    let inc_hi = (*state).w;

    // state = state * 6364136223846793005 + increment
    // 64-bit multiply: 6364136223846793005 = 0x5851F42D4C957F2D
    let mul_lo: u32 = 0x4C957F2Du;
    let mul_hi: u32 = 0x5851F42Du;

    // 64-bit multiply (lo * lo, cross terms, hi * lo)
    let ll = u64_mul_lo(s64_lo, mul_lo);
    let ll_hi = u64_mul_hi(s64_lo, mul_lo);
    let lh = u64_mul_lo(s64_lo, mul_hi);
    let hl = u64_mul_lo(s64_hi, mul_lo);

    let new_lo = ll;
    let new_hi = ll_hi + lh + hl;

    // Add increment
    let add_lo = new_lo + inc_lo;
    let carry = select(0u, 1u, add_lo < new_lo);
    let add_hi = new_hi + inc_hi + carry;

    (*state).x = add_lo;
    (*state).y = add_hi;

    // XSH-RR output function on old 64-bit state (emulated with 32-bit ops)
    // Step 1: 64-bit right shift by 18
    let shifted18_lo = (s64_lo >> 18u) | (s64_hi << 14u);
    let shifted18_hi = s64_hi >> 18u;
    // Step 2: XOR with original state
    let xor_lo = shifted18_lo ^ s64_lo;
    let xor_hi = shifted18_hi ^ s64_hi;
    // Step 3: 64-bit right shift by 27 → take lower 32 bits
    let xorshifted = (xor_lo >> 27u) | (xor_hi << 5u);
    // Step 4: rotation amount = top 5 bits of 64-bit state = hi >> 27
    let rot = s64_hi >> 27u;
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

// Unsigned 32×32 → lower 32 bits
fn u64_mul_lo(a: u32, b: u32) -> u32 {
    return a * b;
}

// Unsigned 32×32 → upper 32 bits (via mulhi trick)
fn u64_mul_hi(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + (ll >> 16u);
    let mid2 = (mid & 0xFFFFu) + hl;

    return hh + (mid >> 16u) + (mid2 >> 16u);
}

// Random f32 in [0, 1)
fn rand_f32(state: ptr<function, vec4<u32>>) -> f32 {
    return f32(pcg_step(state)) / 4294967296.0;
}

// Random f32 in [min, max)
fn rand_f32_range(state: ptr<function, vec4<u32>>, min_val: f32, max_val: f32) -> f32 {
    return min_val + rand_f32(state) * (max_val - min_val);
}

// Random u32 in [0, max)
fn rand_u32(state: ptr<function, vec4<u32>>, max_val: u32) -> u32 {
    return pcg_step(state) % max_val;
}

// --- Crossover functions ---

/// Compute centroid of a triangle (average of 3 vertices).
fn centroid(poly: Polygon) -> vec2<f32> {
    return (poly.v0 + poly.v1 + poly.v2) / 3.0;
}

/// Tournament selection: pick the fittest chain from `tournament_size` random samples within the same island.
/// Returns the chain ID of the winner.
fn tournament_select(rng: ptr<function, vec4<u32>>, chain_id: u32, chain_count: u32) -> u32 {
    let island_size = params.chain_count_param / max(params.island_count, 1u);
    let island_start = (chain_id / island_size) * island_size;

    var best_id = island_start + rand_u32(rng, island_size);
    var best_fitness = bitcast<f32>(chain_states[best_id].fitness_bits);

    for (var t = 1u; t < params.tournament_size; t++) {
        let candidate_id = island_start + rand_u32(rng, island_size);
        let candidate_fitness = bitcast<f32>(chain_states[candidate_id].fitness_bits);
        if candidate_fitness > best_fitness {
            best_id = candidate_id;
            best_fitness = candidate_fitness;
        }
    }
    return best_id;
}

/// Spatial crossover: split space along a random axis at a random position.
/// Polygons on parent A's side come from A, polygons on parent B's side come from B.
fn crossover_spatial(rng: ptr<function, vec4<u32>>, chain_id: u32, parent_b: u32) {
    // Random split: axis (0=x, 1=y) and position [0, 1)
    let use_y_axis = rand_f32(rng) > 0.5;
    let split_pos = rand_f32(rng);

    let count_a = min(chain_states[chain_id].polygon_count, params.max_polygons);
    let count_b = min(chain_states[parent_b].polygon_count, params.max_polygons);

    var out_count = 0u;

    // From parent A: include polygons whose centroid is on A's side of the split
    for (var i = 0u; i < count_a; i++) {
        if out_count >= params.max_polygons { break; }
        let poly = chain_states[chain_id].polygons[i];
        let c = centroid(poly);
        let coord = select(c.x, c.y, use_y_axis);
        if coord < split_pos {
            working_states[chain_id].polygons[out_count] = poly;
            out_count++;
        }
    }

    // From parent B: include polygons whose centroid is on B's side of the split
    for (var i = 0u; i < count_b; i++) {
        if out_count >= params.max_polygons { break; }
        let poly = chain_states[parent_b].polygons[i];
        let c = centroid(poly);
        let coord = select(c.x, c.y, use_y_axis);
        if coord >= split_pos {
            working_states[chain_id].polygons[out_count] = poly;
            out_count++;
        }
    }

    // Guard: never produce an empty drawing
    working_states[chain_id].polygon_count = max(out_count, 1u);
    if out_count == 0u {
        // Copy at least one polygon from parent A
        working_states[chain_id].polygons[0] = chain_states[chain_id].polygons[0];
    }
}

/// Uniform crossover: for each polygon index, randomly pick from parent A or B.
fn crossover_uniform(rng: ptr<function, vec4<u32>>, chain_id: u32, parent_b: u32) {
    let count_a = min(chain_states[chain_id].polygon_count, params.max_polygons);
    let count_b = min(chain_states[parent_b].polygon_count, params.max_polygons);
    let max_count = max(count_a, count_b);

    var out_count = 0u;

    for (var i = 0u; i < max_count; i++) {
        if out_count >= params.max_polygons { break; }
        let have_a = i < count_a;
        let have_b = i < count_b;
        let pick_a = rand_f32(rng) < 0.5;

        if have_a && have_b {
            // Both parents have this index — pick one
            if pick_a {
                working_states[chain_id].polygons[out_count] = chain_states[chain_id].polygons[i];
            } else {
                working_states[chain_id].polygons[out_count] = chain_states[parent_b].polygons[i];
            }
            out_count++;
        } else if have_a {
            // Only parent A — include with 50% probability
            if pick_a {
                working_states[chain_id].polygons[out_count] = chain_states[chain_id].polygons[i];
                out_count++;
            }
        } else if have_b {
            // Only parent B — include with 50% probability
            if !pick_a {
                working_states[chain_id].polygons[out_count] = chain_states[parent_b].polygons[i];
                out_count++;
            }
        }
    }

    // Guard: never produce an empty drawing
    working_states[chain_id].polygon_count = max(out_count, 1u);
    if out_count == 0u {
        working_states[chain_id].polygons[0] = chain_states[chain_id].polygons[0];
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_id = gid.x;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }

    // Load RNG state into registers
    var rng = chain_states[chain_id].rng_state;

    // --- Crossover path ---
    // With probability crossover_prob, produce offspring via crossover instead of mutation.
    let island_size = params.chain_count_param / max(params.island_count, 1u);
    if params.crossover_prob > 0.0 && rand_f32(&rng) < params.crossover_prob && island_size >= 2u {
        let parent_b = tournament_select(&rng, chain_id, chain_count);

        // Initialize working state header
        working_states[chain_id].fitness_bits = chain_states[chain_id].fitness_bits;
        working_states[chain_id]._pad0 = 0u;
        working_states[chain_id]._pad1 = 0u;

        if rand_f32(&rng) < params.spatial_crossover_weight {
            crossover_spatial(&rng, chain_id, parent_b);
        } else {
            crossover_uniform(&rng, chain_id, parent_b);
        }

        // Clamp alphas on crossover offspring
        let offspring_count = working_states[chain_id].polygon_count;
        for (var i = 0u; i < offspring_count; i++) {
            var poly = working_states[chain_id].polygons[i];
            poly.color.w = clamp(poly.color.w, params.min_alpha_norm, params.max_alpha_norm);
            working_states[chain_id].polygons[i] = poly;
        }

        // Save RNG and return — skip mutation loop
        working_states[chain_id].rng_state = rng;
        return;
    }

    // --- Normal mutation path ---

    // Copy current best to working state, clamping to current limits
    let poly_count = min(chain_states[chain_id].polygon_count, params.max_polygons);
    working_states[chain_id].polygon_count = poly_count;
    working_states[chain_id].fitness_bits = chain_states[chain_id].fitness_bits;
    working_states[chain_id]._pad0 = 0u;
    working_states[chain_id]._pad1 = 0u;

    // Copy polygons, clamping alpha to current range
    for (var i = 0u; i < poly_count; i++) {
        var poly = chain_states[chain_id].polygons[i];
        poly.color.w = clamp(poly.color.w, params.min_alpha_norm, params.max_alpha_norm);
        working_states[chain_id].polygons[i] = poly;
    }

    // Mutate until dirty
    var is_dirty = false;
    var attempts = 0u;

    while !is_dirty && attempts < 1000u {
        attempts++;
        var count = working_states[chain_id].polygon_count;

        // --- Drawing-level mutations ---

        // Add polygon
        if rand_f32(&rng) < params.add_polygon_prob && count < params.max_polygons {
            let origin_x = rand_f32(&rng);
            let origin_y = rand_f32(&rng);
            let d = params.new_point_max_distance;

            var new_poly: Polygon;
            new_poly.color = vec4<f32>(
                rand_f32(&rng),
                rand_f32(&rng),
                rand_f32(&rng),
                clamp(rand_f32(&rng), params.min_alpha_norm, params.max_alpha_norm)
            );
            new_poly.v0 = vec2<f32>(
                clamp(rand_f32_range(&rng, origin_x - d, origin_x + d), 0.0, 1.0),
                clamp(rand_f32_range(&rng, origin_y - d, origin_y + d), 0.0, 1.0)
            );
            new_poly.v1 = vec2<f32>(
                clamp(rand_f32_range(&rng, origin_x - d, origin_x + d), 0.0, 1.0),
                clamp(rand_f32_range(&rng, origin_y - d, origin_y + d), 0.0, 1.0)
            );
            new_poly.v2 = vec2<f32>(
                clamp(rand_f32_range(&rng, origin_x - d, origin_x + d), 0.0, 1.0),
                clamp(rand_f32_range(&rng, origin_y - d, origin_y + d), 0.0, 1.0)
            );
            new_poly._pad = vec2<f32>(0.0, 0.0);

            // Append to end (reorder mutation handles z-order)
            working_states[chain_id].polygons[count] = new_poly;
            count++;
            working_states[chain_id].polygon_count = count;
            is_dirty = true;
        }

        // Remove polygon (swap-remove: replace with last element)
        if rand_f32(&rng) < params.remove_polygon_prob && count > params.min_polygons {
            let remove_idx = rand_u32(&rng, count);
            let last_idx = count - 1u;
            if remove_idx != last_idx {
                working_states[chain_id].polygons[remove_idx] = working_states[chain_id].polygons[last_idx];
            }
            count--;
            working_states[chain_id].polygon_count = count;
            is_dirty = true;
        }

        // Reorder (swap two polygons)
        if rand_f32(&rng) < params.reorder_polygon_prob && count >= 2u {
            let i1 = rand_u32(&rng, count);
            var i2 = rand_u32(&rng, count);
            while i1 == i2 {
                i2 = rand_u32(&rng, count);
            }
            let tmp = working_states[chain_id].polygons[i1];
            working_states[chain_id].polygons[i1] = working_states[chain_id].polygons[i2];
            working_states[chain_id].polygons[i2] = tmp;
            is_dirty = true;
        }

        // --- Per-polygon mutations ---
        for (var pi = 0u; pi < count; pi++) {
            var poly = working_states[chain_id].polygons[pi];

            // Offset polygon (move all vertices by same delta)
            if rand_f32(&rng) < params.offset_polygon_prob {
                let dx = rand_f32_range(&rng, -params.offset_polygon_magnitude, params.offset_polygon_magnitude);
                let dy = rand_f32_range(&rng, -params.offset_polygon_magnitude, params.offset_polygon_magnitude);
                poly.v0 = clamp(poly.v0 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
                poly.v1 = clamp(poly.v1 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
                poly.v2 = clamp(poly.v2 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
                is_dirty = true;
            }

            // Mutate color channels independently
            if rand_f32(&rng) < params.change_color_prob {
                poly.color.x = rand_f32(&rng);
                is_dirty = true;
            }
            if rand_f32(&rng) < params.change_color_prob {
                poly.color.y = rand_f32(&rng);
                is_dirty = true;
            }
            if rand_f32(&rng) < params.change_color_prob {
                poly.color.z = rand_f32(&rng);
                is_dirty = true;
            }
            if rand_f32(&rng) < params.change_color_prob {
                poly.color.w = clamp(rand_f32(&rng), params.min_alpha_norm, params.max_alpha_norm);
                is_dirty = true;
            }

            // Micro-adjust color channels (+/- 1/255)
            let color_step = 1.0 / 255.0;
            if rand_f32(&rng) < params.micro_adjust_prob {
                let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5);
                poly.color.x = clamp(poly.color.x + dir, 0.0, 1.0);
                is_dirty = true;
            }
            if rand_f32(&rng) < params.micro_adjust_prob {
                let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5);
                poly.color.y = clamp(poly.color.y + dir, 0.0, 1.0);
                is_dirty = true;
            }
            if rand_f32(&rng) < params.micro_adjust_prob {
                let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5);
                poly.color.z = clamp(poly.color.z + dir, 0.0, 1.0);
                is_dirty = true;
            }
            if rand_f32(&rng) < params.micro_adjust_prob {
                let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5);
                poly.color.w = clamp(poly.color.w + dir, params.min_alpha_norm, params.max_alpha_norm);
                is_dirty = true;
            }

            // Lighten (all RGB +1/255)
            if rand_f32(&rng) < params.lighten_color_prob
                && poly.color.x < 1.0 && poly.color.y < 1.0 && poly.color.z < 1.0 {
                poly.color.x += color_step;
                poly.color.y += color_step;
                poly.color.z += color_step;
                is_dirty = true;
            }

            // Darken (all RGB -1/255)
            if rand_f32(&rng) < params.darken_color_prob
                && poly.color.x > 0.0 && poly.color.y > 0.0 && poly.color.z > 0.0 {
                poly.color.x -= color_step;
                poly.color.y -= color_step;
                poly.color.z -= color_step;
                is_dirty = true;
            }

            // Mutate vertices
            // v0: move point
            if rand_f32(&rng) < params.move_point_prob {
                let d = params.move_point_max_delta;
                poly.v0.x = clamp(rand_f32_range(&rng, poly.v0.x - d, poly.v0.x + d), 0.0, 1.0);
                poly.v0.y = clamp(rand_f32_range(&rng, poly.v0.y - d, poly.v0.y + d), 0.0, 1.0);
                is_dirty = true;
            }
            // v0: micro adjust
            if rand_f32(&rng) < params.micro_adjust_prob {
                let d = params.micro_adjust_delta;
                poly.v0.x = clamp(rand_f32_range(&rng, poly.v0.x - d, poly.v0.x + d), 0.0, 1.0);
                poly.v0.y = clamp(rand_f32_range(&rng, poly.v0.y - d, poly.v0.y + d), 0.0, 1.0);
                is_dirty = true;
            }

            // v1: move point
            if rand_f32(&rng) < params.move_point_prob {
                let d = params.move_point_max_delta;
                poly.v1.x = clamp(rand_f32_range(&rng, poly.v1.x - d, poly.v1.x + d), 0.0, 1.0);
                poly.v1.y = clamp(rand_f32_range(&rng, poly.v1.y - d, poly.v1.y + d), 0.0, 1.0);
                is_dirty = true;
            }
            // v1: micro adjust
            if rand_f32(&rng) < params.micro_adjust_prob {
                let d = params.micro_adjust_delta;
                poly.v1.x = clamp(rand_f32_range(&rng, poly.v1.x - d, poly.v1.x + d), 0.0, 1.0);
                poly.v1.y = clamp(rand_f32_range(&rng, poly.v1.y - d, poly.v1.y + d), 0.0, 1.0);
                is_dirty = true;
            }

            // v2: move point
            if rand_f32(&rng) < params.move_point_prob {
                let d = params.move_point_max_delta;
                poly.v2.x = clamp(rand_f32_range(&rng, poly.v2.x - d, poly.v2.x + d), 0.0, 1.0);
                poly.v2.y = clamp(rand_f32_range(&rng, poly.v2.y - d, poly.v2.y + d), 0.0, 1.0);
                is_dirty = true;
            }
            // v2: micro adjust
            if rand_f32(&rng) < params.micro_adjust_prob {
                let d = params.micro_adjust_delta;
                poly.v2.x = clamp(rand_f32_range(&rng, poly.v2.x - d, poly.v2.x + d), 0.0, 1.0);
                poly.v2.y = clamp(rand_f32_range(&rng, poly.v2.y - d, poly.v2.y + d), 0.0, 1.0);
                is_dirty = true;
            }

            working_states[chain_id].polygons[pi] = poly;
        }
    }

    // Save updated RNG state
    working_states[chain_id].rng_state = rng;
}
