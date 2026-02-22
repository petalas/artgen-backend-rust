// Mutation compute shader — one workgroup per chain, one thread per offspring
// Cooperatively loads parent into shared memory, then each thread copies to its
// offspring slot and applies probabilistic mutations.
// Single-pass: tries all mutations once, then forces a vertex micro-nudge if none fired.

struct Polygon {
    data: vec4<u32>,   // [color_packed, v0_packed, v1_packed, v2_packed] — 16 bytes
}

struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    mutation_scale: f32,       // adaptive mutation scale
    stagnation_counter: u32,   // iterations since last improvement
    rng_state: vec4<u32>,      // only .x is active RNG state; .yzw unused padding
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

// --- Dirty bounding box helpers for incremental evaluation ---
// Bbox is packed as two u32s: bbox_lo = (min_x | min_y << 16), bbox_hi = (max_x | max_y << 16)
// Coordinates are in pixel space (u16).

fn polygon_bbox_pixels(poly: Polygon, w: u32, h: u32) -> vec4<u32> {
    let v0 = unpack_vertex(poly.data.y);
    let v1 = unpack_vertex(poly.data.z);
    let v2 = unpack_vertex(poly.data.w);
    let min_x = u32(floor(min(v0.x, min(v1.x, v2.x)) * f32(w)));
    let min_y = u32(floor(min(v0.y, min(v1.y, v2.y)) * f32(h)));
    let max_x = min(u32(ceil(max(v0.x, max(v1.x, v2.x)) * f32(w))), w);
    let max_y = min(u32(ceil(max(v0.y, max(v1.y, v2.y)) * f32(h))), h);
    return vec4<u32>(min_x, min_y, max_x, max_y);
}

fn merge_bbox(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return vec4<u32>(min(a.x, b.x), min(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

fn full_image_bbox() -> vec4<u32> {
    return vec4<u32>(0u, 0u, params.image_width, params.image_height);
}

fn pack_bbox(bb: vec4<u32>) -> vec2<u32> {
    return vec2<u32>(
        (bb.x & 0xFFFFu) | ((bb.y & 0xFFFFu) << 16u),
        (bb.z & 0xFFFFu) | ((bb.w & 0xFFFFu) << 16u),
    );
}

fn write_dirty_bbox(oid: u32, bb: vec4<u32>) {
    let packed = pack_bbox(bb);
    working_states[oid].fitness_bits = packed.x;
    working_states[oid].stagnation_counter = packed.y;
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
    incremental_eval: u32,
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

@group(0) @binding(0) var<storage, read>       chain_states:   array<DrawingState>;
@group(0) @binding(1) var<storage, read_write>  working_states: array<DrawingState>;
var<immediate>                                  params:         Params;

// Shared memory for cooperative parent loading.
// All threads in a workgroup collaboratively load the parent chain's polygon data
// here, then each thread copies from shared memory to its offspring slot.
// This avoids lambda redundant global memory reads of the same ~16KB parent.
var<workgroup> shared_parent_poly_count: u32;
var<workgroup> shared_parent_fitness_bits: u32;
var<workgroup> shared_parent_mutation_scale: f32;
var<workgroup> shared_parent_polygons: array<Polygon, 1000>;

// --- PCG32 RNG ---
// 32-bit PCG hash (PCG-RXS-M-XS): ~8 ALU ops per call, only .x of rng_state used
fn pcg_step(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

// Random f32 in [0, 1)
fn rand_f32(state: ptr<function, u32>) -> f32 {
    return f32(pcg_step(state)) / 4294967296.0;
}

// Random f32 in [min, max)
fn rand_f32_range(state: ptr<function, u32>, min_val: f32, max_val: f32) -> f32 {
    return min_val + rand_f32(state) * (max_val - min_val);
}

// Random u32 in [0, max)
fn rand_u32(state: ptr<function, u32>, max_val: u32) -> u32 {
    return pcg_step(state) % max_val;
}

// --- Crossover ---

/// Tournament selection: pick the fittest chain from `tournament_size` random samples across the whole population.
fn tournament_select(rng: ptr<function, u32>, chain_id: u32, chain_count: u32) -> u32 {
    var best_id = rand_u32(rng, chain_count);
    var best_fitness = bitcast<f32>(chain_states[best_id].fitness_bits);

    for (var t = 1u; t < params.tournament_size; t++) {
        let candidate_id = rand_u32(rng, chain_count);
        let candidate_fitness = bitcast<f32>(chain_states[candidate_id].fitness_bits);
        if candidate_fitness > best_fitness {
            best_id = candidate_id;
            best_fitness = candidate_fitness;
        }
    }
    return best_id;
}

/// Uniform crossover: walk both parents in lockstep by layer index, coin-flip each slot.
/// Preserves z-ordering (alpha compositing order) — no centroid math needed.
/// Parent A from shared memory, parent B from global.
fn crossover_uniform_offspring(rng: ptr<function, u32>, parent_b: u32, offspring_id: u32) {
    let count_a = min(shared_parent_poly_count, params.max_polygons);
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
                working_states[offspring_id].polygons[out_count] = shared_parent_polygons[i];
            } else {
                working_states[offspring_id].polygons[out_count] = chain_states[parent_b].polygons[i];
            }
            out_count++;
        } else if have_a {
            if pick_a { working_states[offspring_id].polygons[out_count] = shared_parent_polygons[i]; out_count++; }
        } else if have_b {
            if !pick_a { working_states[offspring_id].polygons[out_count] = chain_states[parent_b].polygons[i]; out_count++; }
        }
    }
    working_states[offspring_id].polygon_count = max(out_count, 1u);
    if out_count == 0u {
        working_states[offspring_id].polygons[0] = shared_parent_polygons[0];
    }
}

// --- Single-mutation mode (offspring-aware, with adaptive mutation scale) ---
// Returns dirty bbox as vec4<u32>(min_x, min_y, max_x, max_y) in pixel coords.
fn single_mutate_offspring(rng: ptr<function, u32>, oid: u32, count: ptr<function, u32>, ms: f32) -> vec4<u32> {
    let c = *count;
    let w = params.image_width;
    let h = params.image_height;

    let w_add = params.add_polygon_prob;
    let w_remove = params.remove_polygon_prob;
    let w_reorder = params.reorder_polygon_prob;
    let w_scale = params.scale_polygon_prob;
    let w_rotate = params.rotate_polygon_prob;
    let w_adjacent_swap = params.adjacent_swap_prob;
    let w_merge = params.merge_polygon_prob;
    let w_clone = params.clone_polygon_prob;
    let w_swap_colors = params.swap_colors_prob;
    let fc = f32(c);
    let w_offset = params.offset_polygon_prob * fc;
    let w_move_point = params.move_point_prob * fc * 3.0;
    let w_medium_move = params.medium_move_prob * fc * 3.0;
    let w_micro_adjust = params.micro_adjust_prob * fc * 3.0;
    let w_change_color = params.change_color_prob * fc * 4.0;
    let w_micro_color = params.micro_adjust_prob * fc * 4.0;
    let w_brightness = params.adjust_brightness_prob * fc;
    let w_saturation = params.adjust_saturation_prob * fc;

    let total = w_add + w_remove + w_reorder + w_scale + w_rotate + w_adjacent_swap + w_merge + w_clone + w_swap_colors + w_offset + w_move_point + w_medium_move + w_micro_adjust + w_change_color + w_micro_color + w_brightness + w_saturation;

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
        return polygon_bbox_pixels(new_poly, w, h);
    }

    // Remove, reorder, adjacent swap → full image (z-order changes)
    cumulative += w_remove;
    if r < cumulative && c > params.min_polygons {
        let remove_idx = rand_u32(rng, c);
        let last_idx = c - 1u;
        if remove_idx != last_idx { working_states[oid].polygons[remove_idx] = working_states[oid].polygons[last_idx]; }
        *count = c - 1u;
        working_states[oid].polygon_count = c - 1u;
        return full_image_bbox();
    }

    cumulative += w_reorder;
    if r < cumulative && c >= 2u {
        let i1 = rand_u32(rng, c);
        var i2 = rand_u32(rng, c);
        while i1 == i2 { i2 = rand_u32(rng, c); }
        let tmp = working_states[oid].polygons[i1];
        working_states[oid].polygons[i1] = working_states[oid].polygons[i2];
        working_states[oid].polygons[i2] = tmp;
        return full_image_bbox();
    }

    // Scale polygon: union(old bbox, new bbox)
    cumulative += w_scale;
    if r < cumulative && c >= 1u {
        let si = rand_u32(rng, c);
        var poly = working_states[oid].polygons[si];
        let old_bbox = polygon_bbox_pixels(poly, w, h);
        var sv0 = unpack_vertex(poly.data.y); var sv1 = unpack_vertex(poly.data.z); var sv2 = unpack_vertex(poly.data.w);
        let sc = (sv0 + sv1 + sv2) / 3.0;
        let scale = rand_f32_range(rng, 0.8, 1.2);
        sv0 = clamp(sc + (sv0 - sc) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        sv1 = clamp(sc + (sv1 - sc) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        sv2 = clamp(sc + (sv2 - sc) * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        poly.data.y = pack_vertex(sv0); poly.data.z = pack_vertex(sv1); poly.data.w = pack_vertex(sv2);
        working_states[oid].polygons[si] = poly;
        return merge_bbox(old_bbox, polygon_bbox_pixels(poly, w, h));
    }

    // Rotate polygon: union(old bbox, new bbox)
    cumulative += w_rotate;
    if r < cumulative && c >= 1u {
        let ri = rand_u32(rng, c);
        var poly = working_states[oid].polygons[ri];
        let old_bbox = polygon_bbox_pixels(poly, w, h);
        var rv0 = unpack_vertex(poly.data.y); var rv1 = unpack_vertex(poly.data.z); var rv2 = unpack_vertex(poly.data.w);
        let rc = (rv0 + rv1 + rv2) / 3.0;
        let angle = rand_f32_range(rng, -0.2618, 0.2618);
        let cos_a = cos(angle); let sin_a = sin(angle);
        let rd0 = rv0 - rc; rv0 = clamp(rc + vec2<f32>(rd0.x * cos_a - rd0.y * sin_a, rd0.x * sin_a + rd0.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        let rd1 = rv1 - rc; rv1 = clamp(rc + vec2<f32>(rd1.x * cos_a - rd1.y * sin_a, rd1.x * sin_a + rd1.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        let rd2 = rv2 - rc; rv2 = clamp(rc + vec2<f32>(rd2.x * cos_a - rd2.y * sin_a, rd2.x * sin_a + rd2.y * cos_a), vec2<f32>(0.0), vec2<f32>(1.0));
        poly.data.y = pack_vertex(rv0); poly.data.z = pack_vertex(rv1); poly.data.w = pack_vertex(rv2);
        working_states[oid].polygons[ri] = poly;
        return merge_bbox(old_bbox, polygon_bbox_pixels(poly, w, h));
    }

    // Adjacent swap → full image (z-order changes)
    cumulative += w_adjacent_swap;
    if r < cumulative && c >= 2u {
        let ai = rand_u32(rng, c);
        let aj = select(ai + 1u, ai - 1u, ai == c - 1u);
        let tmp = working_states[oid].polygons[ai];
        working_states[oid].polygons[ai] = working_states[oid].polygons[aj];
        working_states[oid].polygons[aj] = tmp;
        return full_image_bbox();
    }

    // Merge polygons: pick two, check centroid proximity + color similarity, remove smaller
    cumulative += w_merge;
    if r < cumulative && c >= 2u && c > params.min_polygons {
        let mi = rand_u32(rng, c);
        var mj = rand_u32(rng, c);
        while mi == mj { mj = rand_u32(rng, c); }
        let pi_poly = working_states[oid].polygons[mi];
        let pj_poly = working_states[oid].polygons[mj];
        let ci_v0 = unpack_vertex(pi_poly.data.y); let ci_v1 = unpack_vertex(pi_poly.data.z); let ci_v2 = unpack_vertex(pi_poly.data.w);
        let cj_v0 = unpack_vertex(pj_poly.data.y); let cj_v1 = unpack_vertex(pj_poly.data.z); let cj_v2 = unpack_vertex(pj_poly.data.w);
        let centroid_i = (ci_v0 + ci_v1 + ci_v2) / 3.0;
        let centroid_j = (cj_v0 + cj_v1 + cj_v2) / 3.0;
        let cdist = abs(centroid_i.x - centroid_j.x) + abs(centroid_i.y - centroid_j.y);
        let color_i = unpack_color(pi_poly);
        let color_j = unpack_color(pj_poly);
        let cdiff = abs(color_i.x - color_j.x) + abs(color_i.y - color_j.y) + abs(color_i.z - color_j.z);
        if cdist < params.merge_centroid_threshold && cdiff < params.merge_color_threshold {
            // Remove the polygon with smaller area (cross product)
            let area_i = abs((ci_v1.x - ci_v0.x) * (ci_v2.y - ci_v0.y) - (ci_v2.x - ci_v0.x) * (ci_v1.y - ci_v0.y));
            let area_j = abs((cj_v1.x - cj_v0.x) * (cj_v2.y - cj_v0.y) - (cj_v2.x - cj_v0.x) * (cj_v1.y - cj_v0.y));
            let remove_idx = select(mi, mj, area_j < area_i);
            let last = c - 1u;
            if remove_idx != last { working_states[oid].polygons[remove_idx] = working_states[oid].polygons[last]; }
            *count = c - 1u;
            working_states[oid].polygon_count = c - 1u;
            return full_image_bbox();
        }
        // Criteria not met — fall through to next mutation
    }

    // Clone + jitter: duplicate a polygon with small perturbation
    cumulative += w_clone;
    if r < cumulative && c < params.max_polygons && c >= 1u {
        let src_idx = rand_u32(rng, c);
        var new_poly = working_states[oid].polygons[src_idx];
        // Jitter position
        let jd = params.new_point_max_distance;
        let jdx = rand_f32_range(rng, -jd, jd);
        let jdy = rand_f32_range(rng, -jd, jd);
        var jv0 = unpack_vertex(new_poly.data.y);
        var jv1 = unpack_vertex(new_poly.data.z);
        var jv2 = unpack_vertex(new_poly.data.w);
        jv0 = clamp(jv0 + vec2<f32>(jdx, jdy), vec2<f32>(0.0), vec2<f32>(1.0));
        jv1 = clamp(jv1 + vec2<f32>(jdx, jdy), vec2<f32>(0.0), vec2<f32>(1.0));
        jv2 = clamp(jv2 + vec2<f32>(jdx, jdy), vec2<f32>(0.0), vec2<f32>(1.0));
        new_poly.data.y = pack_vertex(jv0);
        new_poly.data.z = pack_vertex(jv1);
        new_poly.data.w = pack_vertex(jv2);
        // Jitter color: ±5/255 per RGB channel
        var jcolor = unpack_color(new_poly);
        let cstep = 5.0 / 255.0;
        jcolor.x = clamp(jcolor.x + rand_f32_range(rng, -cstep, cstep), 0.0, 1.0);
        jcolor.y = clamp(jcolor.y + rand_f32_range(rng, -cstep, cstep), 0.0, 1.0);
        jcolor.z = clamp(jcolor.z + rand_f32_range(rng, -cstep, cstep), 0.0, 1.0);
        new_poly.data.x = pack_color(jcolor);
        working_states[oid].polygons[c] = new_poly;
        *count = c + 1u;
        working_states[oid].polygon_count = c + 1u;
        return full_image_bbox();
    }

    // Swap colors between two polygons
    cumulative += w_swap_colors;
    if r < cumulative && c >= 2u {
        let sci = rand_u32(rng, c);
        var scj = rand_u32(rng, c);
        while sci == scj { scj = rand_u32(rng, c); }
        let tmp_color = working_states[oid].polygons[sci].data.x;
        working_states[oid].polygons[sci].data.x = working_states[oid].polygons[scj].data.x;
        working_states[oid].polygons[scj].data.x = tmp_color;
        return full_image_bbox();
    }

    if c == 0u { return full_image_bbox(); }
    let pi = rand_u32(rng, c);
    var poly = working_states[oid].polygons[pi];
    let old_bbox = polygon_bbox_pixels(poly, w, h);
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
            // Medium-range point move
            cumulative += w_medium_move;
            if r < cumulative {
                let d = params.medium_move_delta * ms;
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
                        // Adjust brightness (50/50 lighten/darken)
                        else {
                            cumulative += w_brightness;
                            if r < cumulative {
                                let color_step = 1.0 / 255.0;
                                let brighten = rand_f32(rng) > 0.5;
                                if brighten {
                                    color.x = min(color.x + color_step, 1.0);
                                    color.y = min(color.y + color_step, 1.0);
                                    color.z = min(color.z + color_step, 1.0);
                                } else {
                                    color.x = max(color.x - color_step, 0.0);
                                    color.y = max(color.y - color_step, 0.0);
                                    color.z = max(color.z - color_step, 0.0);
                                }
                            }
                            // Adjust saturation (fallback)
                            else {
                                let avg = (color.x + color.y + color.z) / 3.0;
                                let color_step = 1.0 / 255.0;
                                let saturate = rand_f32(rng) > 0.5;
                                let dx = sign(color.x - avg);
                                let dy = sign(color.y - avg);
                                let dz = sign(color.z - avg);
                                if saturate {
                                    color.x = clamp(color.x + dx * color_step, 0.0, 1.0);
                                    color.y = clamp(color.y + dy * color_step, 0.0, 1.0);
                                    color.z = clamp(color.z + dz * color_step, 0.0, 1.0);
                                } else {
                                    color.x = clamp(color.x - dx * color_step, 0.0, 1.0);
                                    color.y = clamp(color.y - dy * color_step, 0.0, 1.0);
                                    color.z = clamp(color.z - dz * color_step, 0.0, 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Repack and write back
    poly.data = vec4<u32>(pack_color(color), pack_vertex(v0), pack_vertex(v1), pack_vertex(v2));
    working_states[oid].polygons[pi] = poly;
    // For per-polygon mutations (vertex/color changes), dirty region is union of old and new bbox
    return merge_bbox(old_bbox, polygon_bbox_pixels(poly, w, h));
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

    // --- Cooperative parent load into shared memory ---
    // All lambda threads collaborate to load the parent's polygon data once,
    // instead of each thread independently reading ~16KB from global memory.
    let poly_count = min(chain_states[chain_id].polygon_count, params.max_polygons);
    let lambda = params.lambda;

    // Thread 0 loads the scalar header fields
    if offspring_local_idx == 0u {
        shared_parent_poly_count = poly_count;
        shared_parent_fitness_bits = chain_states[chain_id].fitness_bits;
        shared_parent_mutation_scale = chain_states[chain_id].mutation_scale;
    }

    // All threads cooperatively load polygons in stripes
    for (var i = offspring_local_idx; i < poly_count; i += lambda) {
        shared_parent_polygons[i] = chain_states[chain_id].polygons[i];
    }

    workgroupBarrier();
    // --- Parent data is now in shared memory ---

    // Compute offspring buffer index
    let offspring_id = chain_id * lambda + offspring_local_idx;

    // Load per-offspring RNG from working_states (persistent across iterations)
    var rng = working_states[offspring_id].rng_state.x;

    // Adaptive mutation scale: only for offspring 1..λ-1 when enabled
    // Offspring 0 always uses scale 1.0 so (1+1) behavior is unchanged
    var mutation_scale = 1.0;
    if params.adaptive_mutation == 1u && offspring_local_idx > 0u {
        mutation_scale = shared_parent_mutation_scale;
    }

    // --- Crossover path ---
    if params.crossover_prob > 0.0 && rand_f32(&rng) < params.crossover_prob && chain_count >= 2u {
        let parent_b = tournament_select(&rng, chain_id, chain_count);

        // Initialize working state header from shared memory
        working_states[offspring_id].fitness_bits = shared_parent_fitness_bits;
        working_states[offspring_id].mutation_scale = mutation_scale;
        working_states[offspring_id].stagnation_counter = 0u;

        // Uniform crossover: parent A from shared memory, parent B from global
        crossover_uniform_offspring(&rng, parent_b, offspring_id);

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
        if params.incremental_eval == 1u {
            write_dirty_bbox(offspring_id, full_image_bbox());
        }
        working_states[offspring_id].rng_state.x = rng;
        return;
    }

    // --- Normal mutation path ---

    // Copy parent to offspring slot from shared memory (not global)
    working_states[offspring_id].polygon_count = poly_count;
    working_states[offspring_id].fitness_bits = shared_parent_fitness_bits;
    working_states[offspring_id].mutation_scale = mutation_scale;
    working_states[offspring_id].stagnation_counter = 0u;

    // Copy polygons from shared memory
    for (var i = 0u; i < poly_count; i++) {
        working_states[offspring_id].polygons[i] = shared_parent_polygons[i];
    }

    // Apply mutation
    var count = working_states[offspring_id].polygon_count;

    if params.single_mutation_mode == 1u {
        let dirty_bbox = single_mutate_offspring(&rng, offspring_id, &count, mutation_scale);
        if params.incremental_eval == 1u {
            write_dirty_bbox(offspring_id, dirty_bbox);
        }
        working_states[offspring_id].rng_state.x = rng;
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
    if rand_f32(&rng) < params.scale_polygon_prob && count >= 1u {
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
    if rand_f32(&rng) < params.rotate_polygon_prob && count >= 1u {
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
    if rand_f32(&rng) < params.adjacent_swap_prob && count >= 2u {
        let i1 = rand_u32(&rng, count);
        let i2 = select(i1 + 1u, i1 - 1u, i1 == count - 1u);
        let tmp = working_states[offspring_id].polygons[i1];
        working_states[offspring_id].polygons[i1] = working_states[offspring_id].polygons[i2];
        working_states[offspring_id].polygons[i2] = tmp;
        is_dirty = true;
    }

    // Merge polygons
    if rand_f32(&rng) < params.merge_polygon_prob && count >= 2u && count > params.min_polygons {
        let mi = rand_u32(&rng, count);
        var mj = rand_u32(&rng, count);
        while mi == mj { mj = rand_u32(&rng, count); }
        let pi_poly = working_states[offspring_id].polygons[mi];
        let pj_poly = working_states[offspring_id].polygons[mj];
        let ci_v0 = unpack_vertex(pi_poly.data.y); let ci_v1 = unpack_vertex(pi_poly.data.z); let ci_v2 = unpack_vertex(pi_poly.data.w);
        let cj_v0 = unpack_vertex(pj_poly.data.y); let cj_v1 = unpack_vertex(pj_poly.data.z); let cj_v2 = unpack_vertex(pj_poly.data.w);
        let centroid_i = (ci_v0 + ci_v1 + ci_v2) / 3.0;
        let centroid_j = (cj_v0 + cj_v1 + cj_v2) / 3.0;
        let cdist = abs(centroid_i.x - centroid_j.x) + abs(centroid_i.y - centroid_j.y);
        let color_i = unpack_color(pi_poly);
        let color_j = unpack_color(pj_poly);
        let cdiff = abs(color_i.x - color_j.x) + abs(color_i.y - color_j.y) + abs(color_i.z - color_j.z);
        if cdist < params.merge_centroid_threshold && cdiff < params.merge_color_threshold {
            let area_i = abs((ci_v1.x - ci_v0.x) * (ci_v2.y - ci_v0.y) - (ci_v2.x - ci_v0.x) * (ci_v1.y - ci_v0.y));
            let area_j = abs((cj_v1.x - cj_v0.x) * (cj_v2.y - cj_v0.y) - (cj_v2.x - cj_v0.x) * (cj_v1.y - cj_v0.y));
            let remove_idx = select(mi, mj, area_j < area_i);
            let last = count - 1u;
            if remove_idx != last { working_states[offspring_id].polygons[remove_idx] = working_states[offspring_id].polygons[last]; }
            count--;
            working_states[offspring_id].polygon_count = count;
            is_dirty = true;
        }
    }

    // Clone + jitter
    if rand_f32(&rng) < params.clone_polygon_prob && count < params.max_polygons && count >= 1u {
        let src_idx = rand_u32(&rng, count);
        var new_poly = working_states[offspring_id].polygons[src_idx];
        let jd = params.new_point_max_distance;
        let jdx = rand_f32_range(&rng, -jd, jd);
        let jdy = rand_f32_range(&rng, -jd, jd);
        var jv0 = unpack_vertex(new_poly.data.y);
        var jv1 = unpack_vertex(new_poly.data.z);
        var jv2 = unpack_vertex(new_poly.data.w);
        jv0 = clamp(jv0 + vec2<f32>(jdx, jdy), vec2<f32>(0.0), vec2<f32>(1.0));
        jv1 = clamp(jv1 + vec2<f32>(jdx, jdy), vec2<f32>(0.0), vec2<f32>(1.0));
        jv2 = clamp(jv2 + vec2<f32>(jdx, jdy), vec2<f32>(0.0), vec2<f32>(1.0));
        new_poly.data.y = pack_vertex(jv0);
        new_poly.data.z = pack_vertex(jv1);
        new_poly.data.w = pack_vertex(jv2);
        var jcolor = unpack_color(new_poly);
        let cstep = 5.0 / 255.0;
        jcolor.x = clamp(jcolor.x + rand_f32_range(&rng, -cstep, cstep), 0.0, 1.0);
        jcolor.y = clamp(jcolor.y + rand_f32_range(&rng, -cstep, cstep), 0.0, 1.0);
        jcolor.z = clamp(jcolor.z + rand_f32_range(&rng, -cstep, cstep), 0.0, 1.0);
        new_poly.data.x = pack_color(jcolor);
        working_states[offspring_id].polygons[count] = new_poly;
        count++;
        working_states[offspring_id].polygon_count = count;
        is_dirty = true;
    }

    // Swap colors
    if rand_f32(&rng) < params.swap_colors_prob && count >= 2u {
        let sci = rand_u32(&rng, count);
        var scj = rand_u32(&rng, count);
        while sci == scj { scj = rand_u32(&rng, count); }
        let tmp_color = working_states[offspring_id].polygons[sci].data.x;
        working_states[offspring_id].polygons[sci].data.x = working_states[offspring_id].polygons[scj].data.x;
        working_states[offspring_id].polygons[scj].data.x = tmp_color;
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

        if rand_f32(&rng) < params.adjust_brightness_prob {
            let brighten = rand_f32(&rng) > 0.5;
            if brighten {
                color.x = min(color.x + color_step, 1.0); color.y = min(color.y + color_step, 1.0); color.z = min(color.z + color_step, 1.0);
            } else {
                color.x = max(color.x - color_step, 0.0); color.y = max(color.y - color_step, 0.0); color.z = max(color.z - color_step, 0.0);
            }
            is_dirty = true;
        }
        if rand_f32(&rng) < params.adjust_saturation_prob {
            let avg = (color.x + color.y + color.z) / 3.0;
            let sat = rand_f32(&rng) > 0.5;
            let sdx = sign(color.x - avg); let sdy = sign(color.y - avg); let sdz = sign(color.z - avg);
            if sat {
                color.x = clamp(color.x + sdx * color_step, 0.0, 1.0); color.y = clamp(color.y + sdy * color_step, 0.0, 1.0); color.z = clamp(color.z + sdz * color_step, 0.0, 1.0);
            } else {
                color.x = clamp(color.x - sdx * color_step, 0.0, 1.0); color.y = clamp(color.y - sdy * color_step, 0.0, 1.0); color.z = clamp(color.z - sdz * color_step, 0.0, 1.0);
            }
            is_dirty = true;
        }

        // Move point (scaled by mutation_scale)
        let move_d = params.move_point_max_delta * mutation_scale;
        let medium_d = params.medium_move_delta * mutation_scale;
        let micro_d = params.micro_adjust_delta * mutation_scale;
        if rand_f32(&rng) < params.move_point_prob {
            v0.x = clamp(rand_f32_range(&rng, v0.x - move_d, v0.x + move_d), 0.0, 1.0);
            v0.y = clamp(rand_f32_range(&rng, v0.y - move_d, v0.y + move_d), 0.0, 1.0);
            is_dirty = true;
        }
        if rand_f32(&rng) < params.medium_move_prob {
            v0.x = clamp(rand_f32_range(&rng, v0.x - medium_d, v0.x + medium_d), 0.0, 1.0);
            v0.y = clamp(rand_f32_range(&rng, v0.y - medium_d, v0.y + medium_d), 0.0, 1.0);
            is_dirty = true;
        }
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
        if rand_f32(&rng) < params.medium_move_prob {
            v1.x = clamp(rand_f32_range(&rng, v1.x - medium_d, v1.x + medium_d), 0.0, 1.0);
            v1.y = clamp(rand_f32_range(&rng, v1.y - medium_d, v1.y + medium_d), 0.0, 1.0);
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
        if rand_f32(&rng) < params.medium_move_prob {
            v2.x = clamp(rand_f32_range(&rng, v2.x - medium_d, v2.x + medium_d), 0.0, 1.0);
            v2.y = clamp(rand_f32_range(&rng, v2.y - medium_d, v2.y + medium_d), 0.0, 1.0);
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

    // Multi-mutation mode always dirties the full image
    if params.incremental_eval == 1u {
        write_dirty_bbox(offspring_id, full_image_bbox());
    }

    // Save per-offspring RNG state
    working_states[offspring_id].rng_state.x = rng;
}
