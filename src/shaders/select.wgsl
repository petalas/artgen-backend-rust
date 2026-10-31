// Selection compute shader — 64 threads per chain (parallel polygon copy)
// Entry point: `select_main`

// common.wgsl, changes.wgsl, offspring_changes.wgsl and the generated constants
// (POLYGON_CAPACITY, CHANGED_INDICES, TILE_MAX_POLYS, TILE_OVERFLOW_ERROR,
// OPERATOR_COUNT and the OP_* operator codes) are appended by programs.rs.

// Reduction value of an offspring that must not be accepted: rasterize_error
// set TILE_OVERFLOW_ERROR because a tile list it evaluated was truncated. Real
// totals stay below 2^32 - 1, so it also loses to every evaluated offspring.
const REJECTED_ERROR: u32 = 0xFFFFFFFFu;

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
@group(1) @binding(0) var<uniform>             params:             Params;
@group(0) @binding(4) var<storage, read_write> fitness_packed:     array<u32>;
@group(0) @binding(5) var<storage, read_write> chain_update_flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> chain_total_errors: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> chain_changes:      array<ChainChange>;
// Operator stats (gpu_evolver::operators): mutate's code (and step rung + 1 above
// OP_RUNG_SHIFT) per offspring, and cumulative counters [offered | improved |
// neutral], OPERATOR_COUNT each, then the same three blocks of STEP_RUNGS, then
// the improvements' summed fitness gain (signed, in error units: the penalized_error drop) per
// operator and per rung.
@group(0) @binding(8) var<storage, read>       offspring_ops:      array<u32>;
@group(0) @binding(9) var<storage, read_write> operator_stats:     array<atomic<u32>>;

// Workgroup-shared variables for communicating decisions from thread 0 to all threads
var<workgroup> shared_accept: u32;
var<workgroup> shared_copy_count: u32;
var<workgroup> shared_best_offspring_id: u32;

// Min-reduction over the offspring's penalized errors (see penalized_error),
// carrying the local offspring index; raw_err keeps each lane's pixel error.
var<workgroup> reduction_err: array<u32, 64>;
var<workgroup> reduction_idx: array<u32, 64>;
var<workgroup> raw_err: array<u32, 64>;

// This chain's offspring per operator code, then per step rung, added to
// operator_stats once.
var<workgroup> offered_ops: array<atomic<u32>, OPERATOR_COUNT + STEP_RUNGS>;
const_assert OPERATOR_COUNT + STEP_RUNGS <= 64u;
const RUNG_STATS: u32 = 3u * OPERATOR_COUNT;
const GAIN_STATS: u32 = RUNG_STATS + 3u * STEP_RUNGS;

// Add one accepted winner (block 1 improved, 2 neutral) from its offspring_ops
// word: under its operator, and under its rung when it has one. An improvement
// also adds its gain, as two's complement, to the gain blocks.
fn count_winner(block: u32, word: u32, gain: i32) {
    let code = word & OP_CODE_MASK;
    atomicAdd(&operator_stats[block * OPERATOR_COUNT + code], 1u);
    let rung_tag = word >> OP_RUNG_SHIFT;
    if rung_tag != 0u {
        atomicAdd(&operator_stats[RUNG_STATS + block * STEP_RUNGS + rung_tag - 1u], 1u);
    }
    if block == 1u {
        atomicAdd(&operator_stats[GAIN_STATS + code], bitcast<u32>(gain));
        if rung_tag != 0u {
            atomicAdd(&operator_stats[GAIN_STATS + OPERATOR_COUNT + rung_tag - 1u], bitcast<u32>(gain));
        }
    }
}

// Fraction of the fitness that n triangles remove at penalty exponent k > 0:
// P(n) = 3 p N0 / (k + 1) * (n / N0)^(k + 1), N0 = PENALTY_REFERENCE_TRIANGLES,
// so the next triangle costs 3 p (n / N0)^k. PointPenalty mirrors it on the CPU
// in this operation order. k == 0 (the flat P(n) = 3 p n) never calls this:
// its callers keep their historical arithmetic so scores keep every bit.
fn penalty_fraction(polygons: u32) -> f32 {
    if polygons == 0u {
        return 0.0;
    }
    let power = params.penalty_exponent + 1.0;
    return 3.0 * params.per_point_penalty * PENALTY_REFERENCE_TRIANGLES / power
        * pow(f32(polygons) / PENALTY_REFERENCE_TRIANGLES, power);
}

// The error with the point penalty added, in whole error units, which every
// selection decision compares. fitness = 100 (M - E) / M * (1 - P(n)), M =
// max_total_error, equals 100 (1 - (E + (M - E) P(n)) / M), so ordering by
// E + round((M - E) P(n)) orders by fitness, exactly where the f32 fitness
// (one ulp is about 9 error units near 99%) rounds differences away. Below
// 2^32 because P(n) < 1 and M fits u32. At k == 0, P(n) = 3 p n.
fn penalized_error(error: u32, polygons: u32) -> u32 {
    let max_total_error = params.max_error_per_pixel * f32(params.image_width * params.image_height);
    var fraction: f32;
    if params.penalty_exponent == 0.0 {
        fraction = 3.0 * params.per_point_penalty * f32(polygons);
    } else {
        fraction = penalty_fraction(polygons);
    }
    return error + u32(round((max_total_error - f32(error)) * fraction));
}

// Unchanged-pixel note for update_framebuffers (ChainChange in changes.wgsl):
// the winner's changed-index words and the parent versions of those indices.
// Must run before the winner's polygons overwrite the parent's.
fn record_chain_change(chain_id: u32, oid: u32, parent_count: u32) {
    let changed = offspring_changed_words(oid);
    var parent_mask = 0u;
    for (var k = 0u; k < CHANGED_INDICES; k++) {
        let encoded = changed_half(changed, k);
        if encoded != 0u && encoded - 1u < parent_count {
            chain_changes[chain_id].parents[k] = chain_states[chain_id].polygons[encoded - 1u];
            parent_mask |= 1u << k;
        }
    }
    chain_changes[chain_id].changed = changed;
    chain_changes[chain_id].parent_mask = parent_mask;
}

/// Compute fitness from total error and polygon count.
fn compute_fitness(total_error: u32, polygon_count: u32) -> f32 {
    let w = params.image_width;
    let h = params.image_height;
    let max_total_error = params.max_error_per_pixel * f32(w * h);
    var fitness = 100.0 * (1.0 - f32(total_error) / max_total_error);
    if params.penalty_exponent == 0.0 {
        let num_points = polygon_count * 3u;
        fitness -= fitness * params.per_point_penalty * f32(num_points);
    } else {
        fitness -= fitness * penalty_fraction(polygon_count);
    }
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

    // --- Parallel min-reduction to find best offspring among λ candidates ---
    let lambda = params.lambda;
    if local_id < OPERATOR_COUNT + STEP_RUNGS {
        atomicStore(&offered_ops[local_id], 0u);
    }
    var op = 0u;

    // Phase 1: Each thread loads its error value (or sentinel if beyond lambda)
    // Threads 0..lambda-1 each read one error accumulator via atomicExchange (resets to 0)
    // Threads lambda..63 load the REJECTED_ERROR sentinel so they lose all comparisons
    // Error accumulators are stride-2: [new_error, old_error] per offspring
    // Incremental: final_error = parent_total - old_dirty + new_dirty
    if local_id < lambda {
        let oid = chain_id * lambda + local_id;
        let new_err = atomicExchange(&error_accumulators[oid * 2u], 0u);
        let old_err = atomicExchange(&error_accumulators[oid * 2u + 1u], 0u);
        let parent_total = atomicLoad(&chain_total_errors[chain_id]);
        // Saturating subtraction to avoid underflow
        let base = select(parent_total - old_err, 0u, old_err > parent_total);
        let overflowed = (new_err & TILE_OVERFLOW_ERROR) != 0u;
        let error = base + new_err;
        raw_err[local_id] = error;
        reduction_err[local_id] = select(penalized_error(error, working_states[oid].polygon_count),
            REJECTED_ERROR, overflowed);
        reduction_idx[local_id] = local_id;
        op = offspring_ops[oid];
    } else {
        reduction_err[local_id] = REJECTED_ERROR;
        reduction_idx[local_id] = 0xFFFFFFFFu;
    }

    workgroupBarrier();
    // The reduction's barriers order these before the totals are read.
    if local_id < lambda {
        atomicAdd(&offered_ops[op & OP_CODE_MASK], 1u);
        let rung_tag = op >> OP_RUNG_SHIFT;
        if rung_tag != 0u {
            atomicAdd(&offered_ops[OPERATOR_COUNT + rung_tag - 1u], 1u);
        }
    }

    // Phase 2: Binary tree min-reduction (O(log2(64)) = 6 steps)
    // At each step, the active thread compares its value with the one at offset `stride`
    // and keeps the smaller one (preferring the lower index on ties)
    for (var stride = 32u; stride >= 1u; stride >>= 1u) {
        if local_id < stride {
            let other_err = reduction_err[local_id + stride];
            let my_err = reduction_err[local_id];
            if other_err < my_err {
                reduction_err[local_id] = other_err;
                reduction_idx[local_id] = reduction_idx[local_id + stride];
            }
        }
        workgroupBarrier();
    }

    // Phase 3: Thread 0 reads the reduction result and does all decision logic
    if local_id == 0u {
        let best_penalized = reduction_err[0];
        let best_local = reduction_idx[0];
        let best_offspring_id = chain_id * lambda + best_local;
        let best_error = raw_err[best_local];
        let best_polygons = working_states[best_offspring_id].polygon_count;

        // Compare penalized errors; the f32 fitness is only reported.
        let parent_error = atomicLoad(&chain_total_errors[chain_id]);
        let parent_polygons = chain_states[chain_id].polygon_count;
        let parent_penalized = penalized_error(parent_error, parent_polygons);
        chain_states[chain_id].fitness_bits = bitcast<u32>(compute_fitness(parent_error, parent_polygons));

        // Accept if strictly better, or with 50% probability if equal (plateau traversal)
        // Use the best offspring's RNG for neutral acceptance. An exact copy of
        // the parent (a crossover that kept every slot: NO_CHANGES) is never
        // accepted; it would leave the chain as it is at the cost of the
        // polygon copy and framebuffer update.
        // Every offspring was rejected when the minimum is REJECTED_ERROR.
        let dominated = best_penalized < parent_penalized;
        let copy = offspring_changed_words(best_offspring_id).x == NO_CHANGES;
        let neutral = best_penalized == parent_penalized && !copy
            && (working_states[best_offspring_id].rng_state.x & 1u) == 1u;
        let should_accept = best_penalized != REJECTED_ERROR && (dominated || neutral);

        if should_accept {
            // Improved, else neutral: operator_stats' second or third block.
            // The gain is in error units: the penalized error's drop.
            let gain = i32(parent_penalized) - i32(best_penalized);
            count_winner(select(2u, 1u, dominated), offspring_ops[best_offspring_id], gain);
            record_chain_change(chain_id, best_offspring_id, parent_polygons);
            chain_states[chain_id].polygon_count = best_polygons;
            chain_states[chain_id].fitness_bits = bitcast<u32>(compute_fitness(best_error, best_polygons));
            shared_accept = 1u;
            shared_copy_count = best_polygons;
            shared_best_offspring_id = best_offspring_id;

            // Store total error + write update flags for update_framebuffers
            atomicStore(&chain_total_errors[chain_id], best_error);
            let dirty = offspring_dirty_words(best_offspring_id);
            chain_update_flags[chain_id * 4u] = 1u;       // accepted
            chain_update_flags[chain_id * 4u + 1u] = dirty.x;
            chain_update_flags[chain_id * 4u + 2u] = dirty.y;
            // Winner whose tile lists update_framebuffers reuses; outside canonical state.
            chain_update_flags[chain_id * 4u + 3u] = best_offspring_id;
        } else {
            shared_accept = 0u;
            chain_update_flags[chain_id * 4u] = 0u;  // not accepted
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

    if local_id < OPERATOR_COUNT + STEP_RUNGS {
        let offered = atomicLoad(&offered_ops[local_id]);
        if offered != 0u {
            // Operator codes index the first block; rungs follow at RUNG_STATS.
            let rung = local_id >= OPERATOR_COUNT;
            let word = select(local_id, RUNG_STATS + local_id - OPERATOR_COUNT, rung);
            atomicAdd(&operator_stats[word], offered);
        }
    }

    // Copy the evaluated candidate unchanged. Even a tiny triangle can cover a
    // pixel; unscored cleanup would invalidate cached error and point penalty.
    // ALL threads cooperate on the polygon copy if accepted.
    if shared_accept == 1u {
        let count = shared_copy_count;
        let src = shared_best_offspring_id;
        for (var i = local_id; i < count; i += 64u) {
            chain_states[chain_id].polygons[i] = working_states[src].polygons[i];
        }
    }

}

