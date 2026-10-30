// Mutation compute shader — split each chain across logical offspring groups.
// Physical lanes cooperatively copy the parent's polygons into the child.
// Compact leaders apply the original mutations after both barriers.
// Single-pass: tries all mutations once, then forces a vertex micro-nudge if none fired.

// common.wgsl, changes.wgsl and the generated constants (POLYGON_CAPACITY,
// CHANGED_INDICES, TILE_MAX_POLYS, OPERATOR_COUNT and the OP_* operator codes)
// are appended by programs.rs.

// --- Quantized polygon pack helpers ---

fn pack_color(c: vec4<f32>) -> u32 {
    return pack4x8unorm(c);
}

fn pack_vertex(v: vec2<f32>) -> u32 {
    return u32(round(clamp(v.x, 0.0, 1.0) * 65535.0)) | (u32(round(clamp(v.y, 0.0, 1.0) * 65535.0)) << 16u);
}

// --- Dirty bounding box helpers for incremental evaluation ---
// vec4<u32>(min_x, min_y, max_x, max_y) in pixels; write_dirty_box packs it.

fn polygon_bbox_pixels(poly: Polygon, w: u32, h: u32) -> vec4<u32> {
    let v0 = unpack_vertex(poly.data.y);
    let v1 = unpack_vertex(poly.data.z);
    let v2 = unpack_vertex(poly.data.w);
    // +1 padding on max to guard against float→int boundary rounding
    let min_x = u32(floor(min(v0.x, min(v1.x, v2.x)) * f32(w)));
    let min_y = u32(floor(min(v0.y, min(v1.y, v2.y)) * f32(h)));
    let max_x = min(u32(ceil(max(v0.x, max(v1.x, v2.x)) * f32(w))) + 1u, w);
    let max_y = min(u32(ceil(max(v0.y, max(v1.y, v2.y)) * f32(h))) + 1u, h);
    return vec4<u32>(min_x, min_y, max_x, max_y);
}

fn merge_bbox(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return vec4<u32>(min(a.x, b.x), min(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

fn full_image_bbox() -> vec4<u32> {
    return vec4<u32>(0u, 0u, params.image_width, params.image_height);
}

// --- Offspring change record (changes.wgsl) ---
// mutate is its only writer; offspring_changes.wgsl reads it.

fn write_dirty_box(oid: u32, bb: vec4<u32>) {
    let packed = pack_dirty_box(bb);
    working_states[oid].fitness_bits = packed.x;
    working_states[oid].stagnation_counter = packed.y;
}

fn write_changed_words(oid: u32, words: vec2<u32>) {
    working_states[oid].rng_state.y = words.x;
    working_states[oid].rng_state.z = words.y;
}

@group(0) @binding(0) var<storage, read>       chain_states:   array<DrawingState>;
@group(0) @binding(1) var<storage, read_write>  working_states: array<DrawingState>;
// The operator that produced each offspring (OP_*), with rung + 1 from
// OP_RUNG_SHIFT up when the operator scales with the step, for select's
// operator stats. Outside the working states, so canonical state never sees it.
@group(0) @binding(2) var<storage, read_write>  offspring_ops:  array<u32>;
@group(1) @binding(0) var<uniform>              params:         Params;

// The parent's header, broadcast to the workgroup. Its polygons are read from
// chain_states, which no stage writes during mutate: each workgroup serves one
// offspring, so a workgroup copy of the parent (16 bytes per POLYGON_CAPACITY
// slot) would be read about once and cost more to fill than it saves.
var<workgroup> shared_parent_poly_count: u32;
var<workgroup> shared_parent_fitness_bits: u32;
var<workgroup> shared_parent_mutation_scale: f32;

// Cooperative crossover: lane 0 decides and broadcasts; tail keep bits (one per
// polygon slot) give each kept tail slot its output rank.
const CROSSOVER_KEEP_WORDS: u32 = (POLYGON_CAPACITY + 31u) / 32u;
var<workgroup> shared_crossover_parent_b: u32;
var<workgroup> shared_crossover_rng: u32;
var<workgroup> shared_crossover_flag: u32;
var<workgroup> shared_crossover_keep: array<atomic<u32>, CROSSOVER_KEEP_WORDS>;
// Crossover slots below min(count_a, offspring count) that differ from parent A.
// Only the first CHANGED_INDICES are kept; a larger count disables the
// rasterizer shortcut.
var<workgroup> shared_crossover_changed_count: atomic<u32>;
var<workgroup> shared_crossover_changed: array<u32, CHANGED_INDICES>;
// Union of the pixel bounds (polygon_bbox_pixels) of both versions of every
// changed slot: min x, min y, max x, max y. Unchanged slots cover the same
// pixels in the same order, so the offspring differs from parent A only inside
// it, and every changed polygon's binner bounds lie inside it.
var<workgroup> shared_crossover_box: array<atomic<u32>, 4>;

// --- PCG32 RNG ---
// 32-bit PCG hash (PCG-RXS-M-XS): ~8 ALU ops per call; the state is rng_state.x
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

/// LCG jump-ahead: returns (mul, add) such that `n` calls of `pcg_step` map
/// state s to mul * s + add (mod 2^32). Powers of one LCG commute, so the
/// square-and-multiply order does not matter.
fn pcg_jump(n: u32) -> vec2<u32> {
    var acc_mul = 1u;
    var acc_add = 0u;
    var cur_mul = 747796405u;
    var cur_add = 2891336453u;
    var k = n;
    while k != 0u {
        if (k & 1u) != 0u {
            acc_mul = acc_mul * cur_mul;
            acc_add = acc_add * cur_mul + cur_add;
        }
        cur_add = cur_add * cur_mul + cur_add;
        cur_mul = cur_mul * cur_mul;
        k = k >> 1u;
    }
    return vec2<u32>(acc_mul, acc_add);
}

/// Crossover offspring polygons are stored with their alpha clamped.
fn clamp_polygon_alpha(poly_in: Polygon) -> Polygon {
    var poly = poly_in;
    var color = unpack_color(poly);
    color.w = clamp(color.w, params.min_alpha_norm, params.max_alpha_norm);
    poly.data.x = pack_color(color);
    return poly;
}

/// Write crossover slot `i` and note it if it differs from parent A's slot `i`.
/// Chains descend from one seed, so most slots equal parent A's.
fn write_crossover_slot(offspring_id: u32, i: u32, parent_a: u32, count_a: u32, source: Polygon) {
    let poly = clamp_polygon_alpha(source);
    working_states[offspring_id].polygons[i] = poly;
    if i >= count_a {
        merge_crossover_box(poly);
    } else if any(poly.data != chain_states[parent_a].polygons[i].data) {
        merge_crossover_box(poly);
        merge_crossover_box(chain_states[parent_a].polygons[i]);
        let k = atomicAdd(&shared_crossover_changed_count, 1u);
        if k < CHANGED_INDICES {
            shared_crossover_changed[k] = i;
        }
    }
}

fn merge_crossover_box(poly: Polygon) {
    let bb = polygon_bbox_pixels(poly, params.image_width, params.image_height);
    atomicMin(&shared_crossover_box[0], bb.x);
    atomicMin(&shared_crossover_box[1], bb.y);
    atomicMax(&shared_crossover_box[2], bb.z);
    atomicMax(&shared_crossover_box[3], bb.w);
}

/// Lane 0 only, after every slot is written. Slots in [min(count_a, count_o),
/// max(count_a, count_o)) exist in only one of the two drawings and are changed
/// too. Returns the changed-index words (changes.wgsl), or unknown when more
/// than CHANGED_INDICES slots changed or the empty-offspring fallback fired.
fn crossover_changed_words(count_a: u32, out_count: u32) -> vec2<u32> {
    let listed = atomicLoad(&shared_crossover_changed_count);
    let lo = min(count_a, out_count);
    let hi = max(count_a, out_count);
    let total = listed + (hi - lo);
    if out_count == 0u || total > CHANGED_INDICES {
        return vec2<u32>(0u, 0u);
    }
    if total == 0u {
        return vec2<u32>(NO_CHANGES, 0u);
    }
    var indices = array<u32, 4>(NO_CHANGED_INDEX, NO_CHANGED_INDEX, NO_CHANGED_INDEX, NO_CHANGED_INDEX);
    for (var k = 0u; k < listed; k++) {
        indices[k] = shared_crossover_changed[k];
    }
    for (var j = lo; j < hi; j++) {
        indices[listed + j - lo] = j;
    }
    return vec2<u32>(changed_indices(indices[0], indices[1]),
                     select(0u, changed_indices(indices[2], indices[3]), total > 2u));
}

/// Uniform crossover: walk both parents in lockstep by layer index, coin-flip each slot.
/// Preserves z-ordering (alpha compositing order) — no centroid math needed.
/// Parent A is the offspring's own chain.
///
/// Cooperative: every lane of the offspring's workgroup calls this in uniform
/// control flow. Slot i consumes exactly the i-th draw after `base_rng` (one
/// draw per slot of max(count_a, count_b), as in the serial walk, whose
/// `out_count >= max_polygons` break cannot fire because out_count <= i <
/// max_count <= max_polygons). Lane l starts at draw l and jumps by the
/// workgroup width. Slots below min(count_a, count_b) always emit at index i.
/// Tail slots emit at min_count + (number of kept tail slots before them),
/// found from a workgroup bitmask after a barrier. Alpha clamping is fused into
/// each write. Returns the RNG state after max_count draws (valid on lane 0).
fn crossover_uniform_offspring(base_rng: u32, parent_a: u32, parent_b: u32, offspring_id: u32, lane: u32) -> u32 {
    let count_a = min(shared_parent_poly_count, params.max_polygons);
    let count_b = min(chain_states[parent_b].polygon_count, params.max_polygons);
    let max_count = max(count_a, count_b);
    let min_count = min(count_a, count_b);
    let a_is_longer = count_a > count_b;

    let start = pcg_jump(lane);
    let stride = pcg_jump(PHYSICAL_WG_SIZE);
    var state = start.x * base_rng + start.y;
    for (var i = lane; i < max_count; i += PHYSICAL_WG_SIZE) {
        var draw_state = state;
        let pick_a = rand_f32(&draw_state) < 0.5;
        state = stride.x * state + stride.y;
        if i < min_count {
            var poly: Polygon;
            if pick_a {
                poly = chain_states[parent_a].polygons[i];
            } else {
                poly = chain_states[parent_b].polygons[i];
            }
            write_crossover_slot(offspring_id, i, parent_a, count_a, poly);
        } else if pick_a == a_is_longer {
            atomicOr(&shared_crossover_keep[i >> 5u], 1u << (i & 31u));
        }
    }
    workgroupBarrier();

    let first_word = min_count >> 5u;
    let tail_start = (min_count / PHYSICAL_WG_SIZE) * PHYSICAL_WG_SIZE + lane;
    for (var i = tail_start; i < max_count; i += PHYSICAL_WG_SIZE) {
        if i < min_count { continue; }
        let word = i >> 5u;
        let bit = 1u << (i & 31u);
        let bits = atomicLoad(&shared_crossover_keep[word]);
        if (bits & bit) == 0u { continue; }
        var rank = countOneBits(bits & (bit - 1u));
        for (var w = first_word; w < word; w++) {
            rank += countOneBits(atomicLoad(&shared_crossover_keep[w]));
        }
        var poly: Polygon;
        if a_is_longer {
            poly = chain_states[parent_a].polygons[i];
        } else {
            poly = chain_states[parent_b].polygons[i];
        }
        write_crossover_slot(offspring_id, min_count + rank, parent_a, count_a, poly);
    }
    // Publishes every lane's changed-slot notes and keep bits.
    workgroupBarrier();

    var out_count = min_count;
    let last_word = (max_count + 31u) >> 5u;
    for (var w = first_word; w < last_word; w++) {
        out_count += countOneBits(atomicLoad(&shared_crossover_keep[w]));
    }
    // Parent A's slots past the offspring's count are removed, including any
    // beyond a search cap lowered below its count.
    let stored_a = chain_states[parent_a].polygon_count;
    for (var i = out_count + lane; i < stored_a; i += PHYSICAL_WG_SIZE) {
        merge_crossover_box(chain_states[parent_a].polygons[i]);
    }
    workgroupBarrier();

    if lane == 0u {
        working_states[offspring_id].polygon_count = max(out_count, 1u);
        if out_count == 0u {
            // Slot 0 of an empty parent A: the zero polygon, as a zeroed
            // workgroup copy of the parent used to hold (slots past a drawing's
            // count may hold stale polygons).
            var first = Polygon(vec4<u32>(0u));
            if count_a > 0u {
                first = chain_states[parent_a].polygons[0];
            }
            working_states[offspring_id].polygons[0] = clamp_polygon_alpha(first);
        }
        write_changed_words(offspring_id, crossover_changed_words(count_a, out_count));
        write_dirty_box(offspring_id, crossover_dirty_box(out_count));
    }
    let end = pcg_jump(max_count);
    return end.x * base_rng + end.y;
}

/// Lane 0, after crossover_uniform_offspring: the changed slots' pixel bounds;
/// an empty rectangle for an exact copy of parent A (no tile is consumed), and
/// the full image for the empty-offspring fallback, whose slot 0 is not tracked.
fn crossover_dirty_box(out_count: u32) -> vec4<u32> {
    if out_count == 0u {
        return full_image_bbox();
    }
    let bb = vec4<u32>(atomicLoad(&shared_crossover_box[0]), atomicLoad(&shared_crossover_box[1]),
                       atomicLoad(&shared_crossover_box[2]), atomicLoad(&shared_crossover_box[3]));
    if bb.x >= bb.z || bb.y >= bb.w {
        return vec4<u32>(0u);
    }
    return bb;
}

// --- Mutation: exactly one operator per offspring, picked by weighted roulette
// (offspring-aware, with adaptive mutation scale) ---
// `ms` multiplies offset, point-move and micro vertex step sizes: the adaptive
// scale times the offspring's step rung multiplier (op_scales_with_step).
// Returns dirty bbox as vec4<u32>(min_x, min_y, max_x, max_y) in pixel coords.
// `op` receives the branch that ran (OP_*); a pick whose guard fails falls
// through to a later branch, which is the one recorded.
fn mutate_offspring(rng: ptr<function, u32>, oid: u32, count: ptr<function, u32>, ms: f32, changed: ptr<function, u32>, op: ptr<function, u32>) -> vec4<u32> {
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
        *changed = changed_indices(c, NO_CHANGED_INDEX);
        *op = OP_ADD;
        return polygon_bbox_pixels(new_poly, w, h);
    }

    // Remove polygon — dirty region covers both the removed polygon and the polygon
    // that swap-fills its slot (which changes z-order, affecting alpha blend results)
    cumulative += w_remove;
    if r < cumulative && c > params.min_polygons {
        let remove_idx = rand_u32(rng, c);
        let removed_bbox = polygon_bbox_pixels(working_states[oid].polygons[remove_idx], w, h);
        let last_idx = c - 1u;
        var dirty = removed_bbox;
        if remove_idx != last_idx {
            dirty = merge_bbox(dirty, polygon_bbox_pixels(working_states[oid].polygons[last_idx], w, h));
            working_states[oid].polygons[remove_idx] = working_states[oid].polygons[last_idx];
        }
        *count = c - 1u;
        working_states[oid].polygon_count = c - 1u;
        *changed = changed_indices(remove_idx, last_idx);
        *op = OP_REMOVE;
        return dirty;
    }

    // Reorder (swap two) — dirty region is union of both polygons' bboxes
    cumulative += w_reorder;
    if r < cumulative && c >= 2u {
        let i1 = rand_u32(rng, c);
        var i2 = rand_u32(rng, c);
        while i1 == i2 { i2 = rand_u32(rng, c); }
        let bbox1 = polygon_bbox_pixels(working_states[oid].polygons[i1], w, h);
        let bbox2 = polygon_bbox_pixels(working_states[oid].polygons[i2], w, h);
        let tmp = working_states[oid].polygons[i1];
        working_states[oid].polygons[i1] = working_states[oid].polygons[i2];
        working_states[oid].polygons[i2] = tmp;
        *changed = changed_indices(i1, i2);
        *op = OP_REORDER;
        return merge_bbox(bbox1, bbox2);
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
        *changed = changed_indices(si, NO_CHANGED_INDEX);
        *op = OP_SCALE;
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
        *changed = changed_indices(ri, NO_CHANGED_INDEX);
        *op = OP_ROTATE;
        return merge_bbox(old_bbox, polygon_bbox_pixels(poly, w, h));
    }

    // Adjacent swap — dirty region is union of both polygons' bboxes
    cumulative += w_adjacent_swap;
    if r < cumulative && c >= 2u {
        let ai = rand_u32(rng, c);
        let aj = select(ai + 1u, ai - 1u, ai == c - 1u);
        let bbox_ai = polygon_bbox_pixels(working_states[oid].polygons[ai], w, h);
        let bbox_aj = polygon_bbox_pixels(working_states[oid].polygons[aj], w, h);
        let tmp = working_states[oid].polygons[ai];
        working_states[oid].polygons[ai] = working_states[oid].polygons[aj];
        working_states[oid].polygons[aj] = tmp;
        *changed = changed_indices(ai, aj);
        *op = OP_ADJACENT_SWAP;
        return merge_bbox(bbox_ai, bbox_aj);
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
            let removed_bbox = polygon_bbox_pixels(working_states[oid].polygons[remove_idx], w, h);
            let last = c - 1u;
            var dirty = removed_bbox;
            if remove_idx != last {
                dirty = merge_bbox(dirty, polygon_bbox_pixels(working_states[oid].polygons[last], w, h));
                working_states[oid].polygons[remove_idx] = working_states[oid].polygons[last];
            }
            *count = c - 1u;
            working_states[oid].polygon_count = c - 1u;
            *changed = changed_indices(remove_idx, last);
            *op = OP_MERGE;
            return dirty;
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
        *changed = changed_indices(c, NO_CHANGED_INDEX);
        *op = OP_CLONE;
        return polygon_bbox_pixels(new_poly, w, h);
    }

    // Swap colors — dirty region is union of both polygons' bboxes
    cumulative += w_swap_colors;
    if r < cumulative && c >= 2u {
        let sci = rand_u32(rng, c);
        var scj = rand_u32(rng, c);
        while sci == scj { scj = rand_u32(rng, c); }
        let bbox_sci = polygon_bbox_pixels(working_states[oid].polygons[sci], w, h);
        let bbox_scj = polygon_bbox_pixels(working_states[oid].polygons[scj], w, h);
        let tmp_color = working_states[oid].polygons[sci].data.x;
        working_states[oid].polygons[sci].data.x = working_states[oid].polygons[scj].data.x;
        working_states[oid].polygons[scj].data.x = tmp_color;
        *changed = changed_indices(sci, scj);
        *op = OP_SWAP_COLORS;
        return merge_bbox(bbox_sci, bbox_scj);
    }

    // An empty parent whose pick failed its guard: evaluated unchanged.
    if c == 0u {
        *op = OP_UNCHANGED;
        return full_image_bbox();
    }
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
        *op = OP_OFFSET;
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
            *op = OP_MOVE_POINT;
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
                *op = OP_MEDIUM_MOVE;
                let d = params.medium_move_delta * ms;
                let vi = rand_u32(rng, 3u);
                if vi == 0u { v0.x = clamp(rand_f32_range(rng, v0.x - d, v0.x + d), 0.0, 1.0); v0.y = clamp(rand_f32_range(rng, v0.y - d, v0.y + d), 0.0, 1.0); }
                else if vi == 1u { v1.x = clamp(rand_f32_range(rng, v1.x - d, v1.x + d), 0.0, 1.0); v1.y = clamp(rand_f32_range(rng, v1.y - d, v1.y + d), 0.0, 1.0); }
                else { v2.x = clamp(rand_f32_range(rng, v2.x - d, v2.x + d), 0.0, 1.0); v2.y = clamp(rand_f32_range(rng, v2.y - d, v2.y + d), 0.0, 1.0); }
            }
            else {
                cumulative += w_micro_adjust;
                if r < cumulative {
                    *op = OP_MICRO_MOVE;
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
                        *op = OP_CHANGE_COLOR;
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
                            *op = OP_MICRO_COLOR;
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
                                *op = OP_BRIGHTNESS;
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
                                *op = OP_SATURATION;
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
    *changed = changed_indices(pi, NO_CHANGED_INDEX);
    // For per-polygon mutations (vertex/color changes), dirty region is union of old and new bbox
    return merge_bbox(old_bbox, polygon_bbox_pixels(poly, w, h));
}

// Step-size multiplier of `rung` (< STEP_RUNGS); see MutationParams::step_multipliers.
fn step_multiplier(rung: u32) -> f32 {
    return params.step_multipliers[rung / 4u][rung % 4u];
}

// Source metadata must match the host's offspring dispatch group size.
const OFFSPRING_GROUP_SIZE: u32 = 1u;
const PHYSICAL_WG_SIZE: u32 = 64u;
const COPY_LANES: u32 = 64u;
// Cooperative crossover gives the whole workgroup to one offspring.
const_assert OFFSPRING_GROUP_SIZE == 1u;

// Copy roles are independent of the compact mutation leaders.
fn copy_parent_prefix(chain_id: u32, group_base: u32, local_idx: u32,
                      poly_count: u32, lambda: u32) {
    let copy_child = local_idx / COPY_LANES;
    let copy_rank = local_idx % COPY_LANES;
    let logical_child = group_base + copy_child;
    if copy_child < OFFSPRING_GROUP_SIZE && logical_child < lambda {
        let copy_oid = chain_id * lambda + logical_child;
        for (var i = copy_rank; i < poly_count; i += COPY_LANES) {
            working_states[copy_oid].polygons[i] = chain_states[chain_id].polygons[i];
        }
    }
}

@compute @workgroup_size(PHYSICAL_WG_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let chain_id = wid.x;
    let group_base = wid.y * OFFSPRING_GROUP_SIZE;
    let offspring_local_idx = group_base + lid.x;
    let chain_count = arrayLength(&chain_states);
    let lambda = params.lambda;
    var poly_count = 0u;

    // All physical lanes reach both barriers, including invalid chains/children.
    if chain_id < chain_count {
        poly_count = min(chain_states[chain_id].polygon_count, params.max_polygons);
        if lid.x == 0u {
            shared_parent_poly_count = poly_count;
            shared_parent_fitness_bits = chain_states[chain_id].fitness_bits;
            shared_parent_mutation_scale = chain_states[chain_id].mutation_scale;
        }
    }

    workgroupBarrier();
    if chain_id < chain_count {
        copy_parent_prefix(chain_id, group_base, lid.x, poly_count, lambda);
    }
    storageBarrier();

    // One offspring per workgroup (OFFSPRING_GROUP_SIZE == 1), so the leader's
    // validity and crossover decision are workgroup-uniform.
    let group_valid = chain_id < chain_count && group_base < lambda;

    // Compute offspring buffer index (leader lane's logical offspring)
    let offspring_id = chain_id * lambda + group_base;

    var rng = 0u;
    var mutation_scale = 1.0;
    for (var w = lid.x; w < CROSSOVER_KEEP_WORDS; w += PHYSICAL_WG_SIZE) {
        atomicStore(&shared_crossover_keep[w], 0u);
    }
    if lid.x == 0u {
        atomicStore(&shared_crossover_changed_count, 0u);
        atomicStore(&shared_crossover_box[0], 0xFFFFFFFFu);
        atomicStore(&shared_crossover_box[1], 0xFFFFFFFFu);
        atomicStore(&shared_crossover_box[2], 0u);
        atomicStore(&shared_crossover_box[3], 0u);
        var crossover = 0u;
        if group_valid {
            // Load per-offspring RNG from working_states (persistent across iterations)
            rng = working_states[offspring_id].rng_state.x;

            // Adaptive mutation scale: only for offspring 1..λ-1 when enabled
            // Offspring 0 always uses scale 1.0 so (1+1) behavior is unchanged
            if params.adaptive_mutation == 1u && offspring_local_idx > 0u {
                mutation_scale = shared_parent_mutation_scale;
            }

            if params.crossover_prob > 0.0 && rand_f32(&rng) < params.crossover_prob && params.chain_count_param >= 2u {
                shared_crossover_parent_b = tournament_select(&rng, chain_id, params.chain_count_param);
                shared_crossover_rng = rng;
                crossover = 1u;

                // Initialize working state header from the broadcast parent header
                working_states[offspring_id].fitness_bits = shared_parent_fitness_bits;
                working_states[offspring_id].mutation_scale = mutation_scale;
                working_states[offspring_id].stagnation_counter = 0u;
            }
        }
        shared_crossover_flag = crossover;
    }

    // --- Crossover path (all lanes; the uniform load also publishes the keep-bit reset) ---
    if workgroupUniformLoad(&shared_crossover_flag) != 0u {
        // Uniform crossover: parent A is this chain, parent B its tournament winner
        let crossover_rng = crossover_uniform_offspring(
            shared_crossover_rng, chain_id, shared_crossover_parent_b, offspring_id, lid.x);
        if lid.x == 0u {
            working_states[offspring_id].rng_state.x = crossover_rng;
            offspring_ops[offspring_id] = OP_CROSSOVER;
        }
        return;
    }

    // Only the leader continues, with the original logical offspring index.
    if !group_valid || lid.x >= OFFSPRING_GROUP_SIZE {
        return;
    }

    // --- Normal mutation path ---

    // The parent's header; copy_parent_prefix copied its polygons
    working_states[offspring_id].polygon_count = poly_count;
    working_states[offspring_id].fitness_bits = shared_parent_fitness_bits;
    working_states[offspring_id].mutation_scale = mutation_scale;
    working_states[offspring_id].stagnation_counter = 0u;

    // Apply exactly one mutation. The working state keeps the adaptive scale
    // (select reads the chain's own scale, never an offspring's); only the
    // sizes inside mutate_offspring see the rung's multiplier, which is
    // exactly 1.0 on every rung without a spread.
    let rung = offspring_local_idx % STEP_RUNGS;
    var count = working_states[offspring_id].polygon_count;
    var changed = 0u;
    var op = OP_UNCHANGED;
    let step = mutation_scale * step_multiplier(rung);
    let dirty_bbox = mutate_offspring(&rng, offspring_id, &count, step, &changed, &op);
    write_dirty_box(offspring_id, dirty_bbox);
    working_states[offspring_id].rng_state.x = rng;
    write_changed_words(offspring_id, vec2<u32>(changed, 0u));
    offspring_ops[offspring_id] = op | select(0u, (rung + 1u) << OP_RUNG_SHIFT, op_scales_with_step(op));
}
