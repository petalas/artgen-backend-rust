// Mutation compute shader — one thread per chain
// Copies chain_states[i] → working_states[i], then applies probabilistic mutations.
// Single-pass: tries all mutations once, then forces a vertex micro-nudge if none fired.

struct Polygon {
    data: vec4<u32>,   // [color_packed, v0_packed, v1_packed, v2_packed] — 16 bytes
}

struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    mutation_scale: f32,       // adaptive mutation scale
    stagnation_counter: u32,   // iterations since last improvement
    rng_state: vec4<u32>,      // .xy = state, .zw = increment
    polygons: array<Polygon, 1000>,
}

// --- Quantized polygon pack/unpack helpers ---

fn unpack_color(p: Polygon) -> vec4<f32> {
    return unpack4x8unorm(p.data.x);
}

fn pack_color(c: vec4<f32>) -> u32 {
    return pack4x8unorm(c);
}

fn unpack_vertex(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xFFFFu) / 65535.0, f32(word >> 16u) / 65535.0);
}

fn pack_vertex(v: vec2<f32>) -> u32 {
    return u32(round(clamp(v.x, 0.0, 1.0) * 65535.0)) | (u32(round(clamp(v.y, 0.0, 1.0) * 65535.0)) << 16u);
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
    let v0 = unpack_vertex(poly.data.y);
    let v1 = unpack_vertex(poly.data.z);
    let v2 = unpack_vertex(poly.data.w);
    return (v0 + v1 + v2) / 3.0;
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

/// Spatial crossover writing to offspring slot.
fn crossover_spatial_offspring(rng: ptr<function, vec4<u32>>, chain_id: u32, parent_b: u32, offspring_id: u32) {
    let use_y_axis = rand_f32(rng) > 0.5;
    let split_pos = rand_f32(rng);
    let count_a = min(chain_states[chain_id].polygon_count, params.max_polygons);
    let count_b = min(chain_states[parent_b].polygon_count, params.max_polygons);
    var out_count = 0u;
    for (var i = 0u; i < count_a; i++) {
        if out_count >= params.max_polygons { break; }
        let poly = chain_states[chain_id].polygons[i];
        let c = centroid(poly);
        let coord = select(c.x, c.y, use_y_axis);
        if coord < split_pos {
            working_states[offspring_id].polygons[out_count] = poly;
            out_count++;
        }
    }
    for (var i = 0u; i < count_b; i++) {
        if out_count >= params.max_polygons { break; }
        let poly = chain_states[parent_b].polygons[i];
        let c = centroid(poly);
        let coord = select(c.x, c.y, use_y_axis);
        if coord >= split_pos {
            working_states[offspring_id].polygons[out_count] = poly;
            out_count++;
        }
    }
    working_states[offspring_id].polygon_count = max(out_count, 1u);
    if out_count == 0u {
        working_states[offspring_id].polygons[0] = chain_states[chain_id].polygons[0];
    }
}

/// Uniform crossover writing to offspring slot.
fn crossover_uniform_offspring(rng: ptr<function, vec4<u32>>, chain_id: u32, parent_b: u32, offspring_id: u32) {
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
            if pick_a {
                working_states[offspring_id].polygons[out_count] = chain_states[chain_id].polygons[i];
            } else {
                working_states[offspring_id].polygons[out_count] = chain_states[parent_b].polygons[i];
            }
            out_count++;
        } else if have_a {
            if pick_a { working_states[offspring_id].polygons[out_count] = chain_states[chain_id].polygons[i]; out_count++; }
        } else if have_b {
            if !pick_a { working_states[offspring_id].polygons[out_count] = chain_states[parent_b].polygons[i]; out_count++; }
        }
    }
    working_states[offspring_id].polygon_count = max(out_count, 1u);
    if out_count == 0u {
        working_states[offspring_id].polygons[0] = chain_states[chain_id].polygons[0];
    }
}

// --- Single-mutation mode (offspring-aware, with adaptive mutation scale) ---
fn single_mutate_offspring(rng: ptr<function, vec4<u32>>, oid: u32, count: ptr<function, u32>, ms: f32) {
    let c = *count;

    let w_add = params.add_polygon_prob;
    let w_remove = params.remove_polygon_prob;
    let w_reorder = params.reorder_polygon_prob;
    let w_scale = params.offset_polygon_prob;
    let w_rotate = params.offset_polygon_prob;
    let w_adjacent_swap = params.reorder_polygon_prob;
    let fc = f32(c);
    let w_offset = params.offset_polygon_prob * fc;
    let w_move_point = params.move_point_prob * fc * 3.0;
    let w_micro_adjust = params.micro_adjust_prob * fc * 3.0;
    let w_change_color = params.change_color_prob * fc * 4.0;
    let w_micro_color = params.micro_adjust_prob * fc * 4.0;
    let w_lighten = params.lighten_color_prob * fc;
    let w_darken = params.darken_color_prob * fc;

    let total = w_add + w_remove + w_reorder + w_scale + w_rotate + w_adjacent_swap + w_offset + w_move_point + w_micro_adjust + w_change_color + w_micro_color + w_lighten + w_darken;

    let r = rand_f32(rng) * total;
    var cumulative = 0.0;

    // Add polygon
    cumulative += w_add;
    if r < cumulative && c < params.max_polygons {
        let origin_x = rand_f32(rng);
        let origin_y = rand_f32(rng);
        let d = params.new_point_max_distance;
        let new_color = vec4<f32>(rand_f32(rng), rand_f32(rng), rand_f32(rng), clamp(rand_f32(rng), params.min_alpha_norm, params.max_alpha_norm));
        let new_v0 = vec2<f32>(clamp(rand_f32_range(rng, origin_x - d, origin_x + d), 0.0, 1.0), clamp(rand_f32_range(rng, origin_y - d, origin_y + d), 0.0, 1.0));
        let new_v1 = vec2<f32>(clamp(rand_f32_range(rng, origin_x - d, origin_x + d), 0.0, 1.0), clamp(rand_f32_range(rng, origin_y - d, origin_y + d), 0.0, 1.0));
        let new_v2 = vec2<f32>(clamp(rand_f32_range(rng, origin_x - d, origin_x + d), 0.0, 1.0), clamp(rand_f32_range(rng, origin_y - d, origin_y + d), 0.0, 1.0));
        var new_poly: Polygon;
        new_poly.data = vec4<u32>(pack_color(new_color), pack_vertex(new_v0), pack_vertex(new_v1), pack_vertex(new_v2));
        working_states[oid].polygons[c] = new_poly;
        *count = c + 1u;
        working_states[oid].polygon_count = c + 1u;
        return;
    }

    cumulative += w_remove;
    if r < cumulative && c > params.min_polygons {
        let remove_idx = rand_u32(rng, c);
        let last_idx = c - 1u;
        if remove_idx != last_idx { working_states[oid].polygons[remove_idx] = working_states[oid].polygons[last_idx]; }
        *count = c - 1u;
        working_states[oid].polygon_count = c - 1u;
        return;
    }

    cumulative += w_reorder;
    if r < cumulative && c >= 2u {
        let i1 = rand_u32(rng, c);
        var i2 = rand_u32(rng, c);
        while i1 == i2 { i2 = rand_u32(rng, c); }
        let tmp = working_states[oid].polygons[i1];
        working_states[oid].polygons[i1] = working_states[oid].polygons[i2];
        working_states[oid].polygons[i2] = tmp;
        return;
    }

    cumulative += w_scale;
    if r < cumulative && c >= 1u {
        let si = rand_u32(rng, c);
        var poly = working_states[oid].polygons[si];
        var sv0 = unpack_vertex(poly.data.y); var sv1 = unpack_vertex(poly.data.z); var sv2 = unpack_vertex(poly.data.w);
        let sc = (sv0 + sv1 + sv2) / 3.0;
        let scale = rand_f32_range(rng, 0.8, 1.2);
        sv0 = clamp(sc + (sv0 - sc) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        sv1 = clamp(sc + (sv1 - sc) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        sv2 = clamp(sc + (sv2 - sc) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        poly.data.y = pack_vertex(sv0); poly.data.z = pack_vertex(sv1); poly.data.w = pack_vertex(sv2);
        working_states[oid].polygons[si] = poly;
        return;
    }

    cumulative += w_rotate;
    if r < cumulative && c >= 1u {
        let ri = rand_u32(rng, c);
        var poly = working_states[oid].polygons[ri];
        var rv0 = unpack_vertex(poly.data.y); var rv1 = unpack_vertex(poly.data.z); var rv2 = unpack_vertex(poly.data.w);
        let rc = (rv0 + rv1 + rv2) / 3.0;
        let angle = rand_f32_range(rng, -0.2618, 0.2618);
        let cos_a = cos(angle); let sin_a = sin(angle);
        let rd0 = rv0 - rc; rv0 = clamp(rc + vec2<f32>(rd0.x * cos_a - rd0.y * sin_a, rd0.x * sin_a + rd0.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        let rd1 = rv1 - rc; rv1 = clamp(rc + vec2<f32>(rd1.x * cos_a - rd1.y * sin_a, rd1.x * sin_a + rd1.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        let rd2 = rv2 - rc; rv2 = clamp(rc + vec2<f32>(rd2.x * cos_a - rd2.y * sin_a, rd2.x * sin_a + rd2.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        poly.data.y = pack_vertex(rv0); poly.data.z = pack_vertex(rv1); poly.data.w = pack_vertex(rv2);
        working_states[oid].polygons[ri] = poly;
        return;
    }

    cumulative += w_adjacent_swap;
    if r < cumulative && c >= 2u {
        let ai = rand_u32(rng, c);
        let aj = select(ai + 1u, ai - 1u, ai == c - 1u);
        let tmp = working_states[oid].polygons[ai];
        working_states[oid].polygons[ai] = working_states[oid].polygons[aj];
        working_states[oid].polygons[aj] = tmp;
        return;
    }

    if c == 0u { return; }
    let pi = rand_u32(rng, c);
    var poly = working_states[oid].polygons[pi];
    var color = unpack_color(poly);
    var v0 = unpack_vertex(poly.data.y);
    var v1 = unpack_vertex(poly.data.z);
    var v2 = unpack_vertex(poly.data.w);

    // Offset polygon (scaled)
    cumulative += w_offset;
    if r < cumulative {
        let mag = params.offset_polygon_magnitude * ms;
        let dx = rand_f32_range(rng, -mag, mag);
        let dy = rand_f32_range(rng, -mag, mag);
        v0 = clamp(v0 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
        v1 = clamp(v1 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
        v2 = clamp(v2 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
    }
    else {
        cumulative += w_move_point;
        if r < cumulative {
            let d = params.move_point_max_delta * ms;
            let vi = rand_u32(rng, 3u);
            if vi == 0u { v0.x = clamp(rand_f32_range(rng, v0.x - d, v0.x + d), 0.0, 1.0); v0.y = clamp(rand_f32_range(rng, v0.y - d, v0.y + d), 0.0, 1.0); }
            else if vi == 1u { v1.x = clamp(rand_f32_range(rng, v1.x - d, v1.x + d), 0.0, 1.0); v1.y = clamp(rand_f32_range(rng, v1.y - d, v1.y + d), 0.0, 1.0); }
            else { v2.x = clamp(rand_f32_range(rng, v2.x - d, v2.x + d), 0.0, 1.0); v2.y = clamp(rand_f32_range(rng, v2.y - d, v2.y + d), 0.0, 1.0); }
        }
        else {
            cumulative += w_micro_adjust;
            if r < cumulative {
                let d = params.micro_adjust_delta * ms;
                let vi = rand_u32(rng, 3u);
                if vi == 0u { v0.x = clamp(rand_f32_range(rng, v0.x - d, v0.x + d), 0.0, 1.0); v0.y = clamp(rand_f32_range(rng, v0.y - d, v0.y + d), 0.0, 1.0); }
                else if vi == 1u { v1.x = clamp(rand_f32_range(rng, v1.x - d, v1.x + d), 0.0, 1.0); v1.y = clamp(rand_f32_range(rng, v1.y - d, v1.y + d), 0.0, 1.0); }
                else { v2.x = clamp(rand_f32_range(rng, v2.x - d, v2.x + d), 0.0, 1.0); v2.y = clamp(rand_f32_range(rng, v2.y - d, v2.y + d), 0.0, 1.0); }
            }
            // Change color channel
            else {
                cumulative += w_change_color;
                if r < cumulative {
                    let ch = rand_u32(rng, 4u);
                    if ch == 0u { color.x = rand_f32(rng); }
                    else if ch == 1u { color.y = rand_f32(rng); }
                    else if ch == 2u { color.z = rand_f32(rng); }
                    else { color.w = clamp(rand_f32(rng), params.min_alpha_norm, params.max_alpha_norm); }
                }
                // Micro-adjust color
                else {
                    cumulative += w_micro_color;
                    if r < cumulative {
                        let ch = rand_u32(rng, 4u);
                        let color_step = 1.0 / 255.0;
                        let dir = select(-color_step, color_step, rand_f32(rng) > 0.5);
                        if ch == 0u { color.x = clamp(color.x + dir, 0.0, 1.0); }
                        else if ch == 1u { color.y = clamp(color.y + dir, 0.0, 1.0); }
                        else if ch == 2u { color.z = clamp(color.z + dir, 0.0, 1.0); }
                        else { color.w = clamp(color.w + dir, params.min_alpha_norm, params.max_alpha_norm); }
                    }
                    // Lighten
                    else {
                        cumulative += w_lighten;
                        if r < cumulative {
                            let color_step = 1.0 / 255.0;
                            color.x = min(color.x + color_step, 1.0);
                            color.y = min(color.y + color_step, 1.0);
                            color.z = min(color.z + color_step, 1.0);
                        }
                        // Darken (fallback)
                        else {
                            let color_step = 1.0 / 255.0;
                            color.x = max(color.x - color_step, 0.0);
                            color.y = max(color.y - color_step, 0.0);
                            color.z = max(color.z - color_step, 0.0);
                        }
                    }
                }
            }
        }
    }

    // Repack and write back
    poly.data = vec4<u32>(pack_color(color), pack_vertex(v0), pack_vertex(v1), pack_vertex(v2));
    working_states[oid].polygons[pi] = poly;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let chain_id = wid.x;
    let offspring_local_idx = lid.x;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }
    // Early exit for threads beyond lambda
    if offspring_local_idx >= params.lambda {
        return;
    }

    // Compute offspring buffer index
    let offspring_id = chain_id * params.lambda + offspring_local_idx;

    // Load per-offspring RNG from working_states (persistent across iterations)
    var rng = working_states[offspring_id].rng_state;

    // Adaptive mutation scale: only for offspring 1..λ-1 when enabled
    // Offspring 0 always uses scale 1.0 so (1+1) behavior is unchanged
    var mutation_scale = 1.0;
    if params.adaptive_mutation == 1u && offspring_local_idx > 0u {
        mutation_scale = chain_states[chain_id].mutation_scale;
    }

    // --- Crossover path ---
    let island_size = params.chain_count_param / max(params.island_count, 1u);
    if params.crossover_prob > 0.0 && rand_f32(&rng) < params.crossover_prob && island_size >= 2u {
        let parent_b = tournament_select(&rng, chain_id, chain_count);

        // Initialize working state header
        working_states[offspring_id].fitness_bits = chain_states[chain_id].fitness_bits;
        working_states[offspring_id].mutation_scale = mutation_scale;
        working_states[offspring_id].stagnation_counter = 0u;

        // Crossover writes to working_states[offspring_id] instead of working_states[chain_id]
        if rand_f32(&rng) < params.spatial_crossover_weight {
            crossover_spatial_offspring(&rng, chain_id, parent_b, offspring_id);
        } else {
            crossover_uniform_offspring(&rng, chain_id, parent_b, offspring_id);
        }

        // Clamp alphas on crossover offspring
        let offspring_count = working_states[offspring_id].polygon_count;
        for (var i = 0u; i < offspring_count; i++) {
            var poly = working_states[offspring_id].polygons[i];
            var color = unpack_color(poly);
            color.w = clamp(color.w, params.min_alpha_norm, params.max_alpha_norm);
            poly.data.x = pack_color(color);
            working_states[offspring_id].polygons[i] = poly;
        }

        // Save RNG to offspring slot
        working_states[offspring_id].rng_state = rng;
        return;
    }

    // --- Normal mutation path ---

    // Copy parent to offspring slot
    let poly_count = min(chain_states[chain_id].polygon_count, params.max_polygons);
    working_states[offspring_id].polygon_count = poly_count;
    working_states[offspring_id].fitness_bits = chain_states[chain_id].fitness_bits;
    working_states[offspring_id].mutation_scale = mutation_scale;
    working_states[offspring_id].stagnation_counter = 0u;

    // Copy polygons, clamping alpha
    for (var i = 0u; i < poly_count; i++) {
        var poly = chain_states[chain_id].polygons[i];
        var color = unpack_color(poly);
        color.w = clamp(color.w, params.min_alpha_norm, params.max_alpha_norm);
        poly.data.x = pack_color(color);
        working_states[offspring_id].polygons[i] = poly;
    }

    // Apply mutation
    var count = working_states[offspring_id].polygon_count;

    if params.single_mutation_mode == 1u {
        single_mutate_offspring(&rng, offspring_id, &count, mutation_scale);
        working_states[offspring_id].rng_state = rng;
        return;
    }

    // Multi-mutation mode
    var is_dirty = false;

    // Add polygon
    if rand_f32(&rng) < params.add_polygon_prob && count < params.max_polygons {
        let origin_x = rand_f32(&rng);
        let origin_y = rand_f32(&rng);
        let d = params.new_point_max_distance;

        let new_color = vec4<f32>(
            rand_f32(&rng),
            rand_f32(&rng),
            rand_f32(&rng),
            clamp(rand_f32(&rng), params.min_alpha_norm, params.max_alpha_norm)
        );
        let new_v0 = vec2<f32>(
            clamp(rand_f32_range(&rng, origin_x - d, origin_x + d), 0.0, 1.0),
            clamp(rand_f32_range(&rng, origin_y - d, origin_y + d), 0.0, 1.0)
        );
        let new_v1 = vec2<f32>(
            clamp(rand_f32_range(&rng, origin_x - d, origin_x + d), 0.0, 1.0),
            clamp(rand_f32_range(&rng, origin_y - d, origin_y + d), 0.0, 1.0)
        );
        let new_v2 = vec2<f32>(
            clamp(rand_f32_range(&rng, origin_x - d, origin_x + d), 0.0, 1.0),
            clamp(rand_f32_range(&rng, origin_y - d, origin_y + d), 0.0, 1.0)
        );

        var new_poly: Polygon;
        new_poly.data = vec4<u32>(
            pack_color(new_color),
            pack_vertex(new_v0),
            pack_vertex(new_v1),
            pack_vertex(new_v2)
        );

        working_states[offspring_id].polygons[count] = new_poly;
        count++;
        working_states[offspring_id].polygon_count = count;
        is_dirty = true;
    }

    // Remove polygon
    if rand_f32(&rng) < params.remove_polygon_prob && count > params.min_polygons {
        let remove_idx = rand_u32(&rng, count);
        let last_idx = count - 1u;
        if remove_idx != last_idx {
            working_states[offspring_id].polygons[remove_idx] = working_states[offspring_id].polygons[last_idx];
        }
        count--;
        working_states[offspring_id].polygon_count = count;
        is_dirty = true;
    }

    // Reorder (swap two)
    if rand_f32(&rng) < params.reorder_polygon_prob && count >= 2u {
        let i1 = rand_u32(&rng, count);
        var i2 = rand_u32(&rng, count);
        while i1 == i2 { i2 = rand_u32(&rng, count); }
        let tmp = working_states[offspring_id].polygons[i1];
        working_states[offspring_id].polygons[i1] = working_states[offspring_id].polygons[i2];
        working_states[offspring_id].polygons[i2] = tmp;
        is_dirty = true;
    }

    // Scale polygon
    if rand_f32(&rng) < params.offset_polygon_prob && count >= 1u {
        let si = rand_u32(&rng, count);
        var poly = working_states[offspring_id].polygons[si];
        var v0 = unpack_vertex(poly.data.y);
        var v1 = unpack_vertex(poly.data.z);
        var v2 = unpack_vertex(poly.data.w);
        let c = (v0 + v1 + v2) / 3.0;
        let scale = rand_f32_range(&rng, 0.8, 1.2);
        v0 = clamp(c + (v0 - c) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        v1 = clamp(c + (v1 - c) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        v2 = clamp(c + (v2 - c) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        poly.data.y = pack_vertex(v0);
        poly.data.z = pack_vertex(v1);
        poly.data.w = pack_vertex(v2);
        working_states[offspring_id].polygons[si] = poly;
        is_dirty = true;
    }

    // Rotate polygon
    if rand_f32(&rng) < params.offset_polygon_prob && count >= 1u {
        let ri = rand_u32(&rng, count);
        var poly = working_states[offspring_id].polygons[ri];
        var v0 = unpack_vertex(poly.data.y);
        var v1 = unpack_vertex(poly.data.z);
        var v2 = unpack_vertex(poly.data.w);
        let c = (v0 + v1 + v2) / 3.0;
        let angle = rand_f32_range(&rng, -0.2618, 0.2618);
        let cos_a = cos(angle);
        let sin_a = sin(angle);
        let d0 = v0 - c;
        v0 = clamp(c + vec2<f32>(d0.x * cos_a - d0.y * sin_a, d0.x * sin_a + d0.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        let d1 = v1 - c;
        v1 = clamp(c + vec2<f32>(d1.x * cos_a - d1.y * sin_a, d1.x * sin_a + d1.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        let d2 = v2 - c;
        v2 = clamp(c + vec2<f32>(d2.x * cos_a - d2.y * sin_a, d2.x * sin_a + d2.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        poly.data.y = pack_vertex(v0);
        poly.data.z = pack_vertex(v1);
        poly.data.w = pack_vertex(v2);
        working_states[offspring_id].polygons[ri] = poly;
        is_dirty = true;
    }

    // Adjacent swap
    if rand_f32(&rng) < params.reorder_polygon_prob && count >= 2u {
        let i1 = rand_u32(&rng, count);
        let i2 = select(i1 + 1u, i1 - 1u, i1 == count - 1u);
        let tmp = working_states[offspring_id].polygons[i1];
        working_states[offspring_id].polygons[i1] = working_states[offspring_id].polygons[i2];
        working_states[offspring_id].polygons[i2] = tmp;
        is_dirty = true;
    }

    // Per-polygon mutations
    for (var pi = 0u; pi < count; pi++) {
        var poly = working_states[offspring_id].polygons[pi];
        var color = unpack_color(poly);
        var v0 = unpack_vertex(poly.data.y);
        var v1 = unpack_vertex(poly.data.z);
        var v2 = unpack_vertex(poly.data.w);

        // Offset polygon (scaled by mutation_scale)
        if rand_f32(&rng) < params.offset_polygon_prob {
            let mag = params.offset_polygon_magnitude * mutation_scale;
            let dx = rand_f32_range(&rng, -mag, mag);
            let dy = rand_f32_range(&rng, -mag, mag);
            v0 = clamp(v0 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
            v1 = clamp(v1 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
            v2 = clamp(v2 + vec2<f32>(dx, dy), vec2<f32>(0.0), vec2<f32>(1.0));
            is_dirty = true;
        }

        // Color mutations (not scaled)
        if rand_f32(&rng) < params.change_color_prob { color.x = rand_f32(&rng); is_dirty = true; }
        if rand_f32(&rng) < params.change_color_prob { color.y = rand_f32(&rng); is_dirty = true; }
        if rand_f32(&rng) < params.change_color_prob { color.z = rand_f32(&rng); is_dirty = true; }
        if rand_f32(&rng) < params.change_color_prob { color.w = clamp(rand_f32(&rng), params.min_alpha_norm, params.max_alpha_norm); is_dirty = true; }

        let color_step = 1.0 / 255.0;
        if rand_f32(&rng) < params.micro_adjust_prob { let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5); color.x = clamp(color.x + dir, 0.0, 1.0); is_dirty = true; }
        if rand_f32(&rng) < params.micro_adjust_prob { let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5); color.y = clamp(color.y + dir, 0.0, 1.0); is_dirty = true; }
        if rand_f32(&rng) < params.micro_adjust_prob { let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5); color.z = clamp(color.z + dir, 0.0, 1.0); is_dirty = true; }
        if rand_f32(&rng) < params.micro_adjust_prob { let dir = select(-color_step, color_step, rand_f32(&rng) > 0.5); color.w = clamp(color.w + dir, params.min_alpha_norm, params.max_alpha_norm); is_dirty = true; }

        if rand_f32(&rng) < params.lighten_color_prob && color.x < 1.0 && color.y < 1.0 && color.z < 1.0 {
            color.x += color_step; color.y += color_step; color.z += color_step; is_dirty = true;
        }
        if rand_f32(&rng) < params.darken_color_prob && color.x > 0.0 && color.y > 0.0 && color.z > 0.0 {
            color.x -= color_step; color.y -= color_step; color.z -= color_step; is_dirty = true;
        }

        // Move point (scaled by mutation_scale)
        let move_d = params.move_point_max_delta * mutation_scale;
        if rand_f32(&rng) < params.move_point_prob {
            v0.x = clamp(rand_f32_range(&rng, v0.x - move_d, v0.x + move_d), 0.0, 1.0);
            v0.y = clamp(rand_f32_range(&rng, v0.y - move_d, v0.y + move_d), 0.0, 1.0);
            is_dirty = true;
        }
        let micro_d = params.micro_adjust_delta * mutation_scale;
        if rand_f32(&rng) < params.micro_adjust_prob {
            v0.x = clamp(rand_f32_range(&rng, v0.x - micro_d, v0.x + micro_d), 0.0, 1.0);
            v0.y = clamp(rand_f32_range(&rng, v0.y - micro_d, v0.y + micro_d), 0.0, 1.0);
            is_dirty = true;
        }
        if rand_f32(&rng) < params.move_point_prob {
            v1.x = clamp(rand_f32_range(&rng, v1.x - move_d, v1.x + move_d), 0.0, 1.0);
            v1.y = clamp(rand_f32_range(&rng, v1.y - move_d, v1.y + move_d), 0.0, 1.0);
            is_dirty = true;
        }
        if rand_f32(&rng) < params.micro_adjust_prob {
            v1.x = clamp(rand_f32_range(&rng, v1.x - micro_d, v1.x + micro_d), 0.0, 1.0);
            v1.y = clamp(rand_f32_range(&rng, v1.y - micro_d, v1.y + micro_d), 0.0, 1.0);
            is_dirty = true;
        }
        if rand_f32(&rng) < params.move_point_prob {
            v2.x = clamp(rand_f32_range(&rng, v2.x - move_d, v2.x + move_d), 0.0, 1.0);
            v2.y = clamp(rand_f32_range(&rng, v2.y - move_d, v2.y + move_d), 0.0, 1.0);
            is_dirty = true;
        }
        if rand_f32(&rng) < params.micro_adjust_prob {
            v2.x = clamp(rand_f32_range(&rng, v2.x - micro_d, v2.x + micro_d), 0.0, 1.0);
            v2.y = clamp(rand_f32_range(&rng, v2.y - micro_d, v2.y + micro_d), 0.0, 1.0);
            is_dirty = true;
        }

        poly.data = vec4<u32>(pack_color(color), pack_vertex(v0), pack_vertex(v1), pack_vertex(v2));
        working_states[offspring_id].polygons[pi] = poly;
    }

    // Fallback micro-nudge if nothing fired
    if !is_dirty {
        let fallback_count = working_states[offspring_id].polygon_count;
        if fallback_count > 0u {
            let target_idx = rand_u32(&rng, fallback_count);
            var poly = working_states[offspring_id].polygons[target_idx];
            let vertex_choice = rand_u32(&rng, 3u);
            let d = params.micro_adjust_delta * mutation_scale;
            if vertex_choice == 0u {
                var v = unpack_vertex(poly.data.y);
                v.x = clamp(rand_f32_range(&rng, v.x - d, v.x + d), 0.0, 1.0);
                v.y = clamp(rand_f32_range(&rng, v.y - d, v.y + d), 0.0, 1.0);
                poly.data.y = pack_vertex(v);
            } else if vertex_choice == 1u {
                var v = unpack_vertex(poly.data.z);
                v.x = clamp(rand_f32_range(&rng, v.x - d, v.x + d), 0.0, 1.0);
                v.y = clamp(rand_f32_range(&rng, v.y - d, v.y + d), 0.0, 1.0);
                poly.data.z = pack_vertex(v);
            } else {
                var v = unpack_vertex(poly.data.w);
                v.x = clamp(rand_f32_range(&rng, v.x - d, v.x + d), 0.0, 1.0);
                v.y = clamp(rand_f32_range(&rng, v.y - d, v.y + d), 0.0, 1.0);
                poly.data.w = pack_vertex(v);
            }
            working_states[offspring_id].polygons[target_idx] = poly;
        }
    }

    // Save per-offspring RNG state
    working_states[offspring_id].rng_state = rng;
}
