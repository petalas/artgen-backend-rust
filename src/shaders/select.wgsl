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
// Operator stats (gpu_evolver::operators): mutate's code per offspring, and
// cumulative counters [offered | improved | neutral], OPERATOR_COUNT each.
@group(0) @binding(8) var<storage, read>       offspring_ops:      array<u32>;
@group(0) @binding(9) var<storage, read_write> operator_stats:     array<atomic<u32>>;

// Workgroup-shared variables for communicating decisions from thread 0 to all threads
var<workgroup> shared_accept: u32;
var<workgroup> shared_copy_count: u32;
var<workgroup> shared_best_offspring_id: u32;

// Shared memory for parallel min-reduction across offspring errors
// Each entry holds (error, local_offspring_index) packed so min on error also selects the index
var<workgroup> reduction_err: array<u32, 64>;
var<workgroup> reduction_idx: array<u32, 64>;

// This chain's offspring per operator code, added to operator_stats once.
var<workgroup> offered_ops: array<atomic<u32>, OPERATOR_COUNT>;
const_assert OPERATOR_COUNT <= 64u;

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
    let num_points = polygon_count * 3u;
    fitness -= fitness * params.per_point_penalty * f32(num_points);
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
    if local_id < OPERATOR_COUNT {
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
        reduction_err[local_id] = select(base + new_err, REJECTED_ERROR, overflowed);
        reduction_idx[local_id] = local_id;
        op = offspring_ops[oid];
    } else {
        reduction_err[local_id] = REJECTED_ERROR;
        reduction_idx[local_id] = 0xFFFFFFFFu;
    }

    workgroupBarrier();
    // The reduction's barriers order these before the totals are read.
    if local_id < lambda {
        atomicAdd(&offered_ops[op], 1u);
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
        let best_error = reduction_err[0];
        let best_local = reduction_idx[0];
        let best_offspring_id = chain_id * lambda + best_local;

        let fitness = compute_fitness(best_error, working_states[best_offspring_id].polygon_count);
        let fitness_bits = bitcast<u32>(fitness);

        // Compare against chain's current best.
        // Recompute parent fitness from chain_total_errors (quantized precision)
        // so it's consistent with offspring error computation.
        let parent_error = atomicLoad(&chain_total_errors[chain_id]);
        let current_fitness = compute_fitness(parent_error, chain_states[chain_id].polygon_count);
        chain_states[chain_id].fitness_bits = bitcast<u32>(current_fitness);

        // Accept if strictly better, or with 50% probability if equal (plateau traversal)
        // Use the best offspring's RNG for neutral acceptance
        let dominated = fitness > current_fitness;
        let neutral = fitness == current_fitness && (working_states[best_offspring_id].rng_state.x & 1u) == 1u;
        // Every offspring was rejected when the minimum is REJECTED_ERROR.
        let should_accept = best_error != REJECTED_ERROR && (dominated || neutral);

        if should_accept {
            // Improved, else neutral: operator_stats' second or third block.
            let block = select(2u, 1u, dominated);
            atomicAdd(&operator_stats[block * OPERATOR_COUNT + offspring_ops[best_offspring_id]], 1u);
            record_chain_change(chain_id, best_offspring_id, chain_states[chain_id].polygon_count);
            chain_states[chain_id].polygon_count = working_states[best_offspring_id].polygon_count;
            chain_states[chain_id].fitness_bits = fitness_bits;
            chain_states[chain_id].stagnation_counter = 0u;
            shared_accept = 1u;
            shared_copy_count = working_states[best_offspring_id].polygon_count;
            shared_best_offspring_id = best_offspring_id;

            // Store total error + write update flags for update_framebuffers
            atomicStore(&chain_total_errors[chain_id], best_error);
            let dirty = offspring_dirty_words(best_offspring_id);
            chain_update_flags[chain_id * 4u] = 1u;       // accepted
            chain_update_flags[chain_id * 4u + 1u] = dirty.x;
            chain_update_flags[chain_id * 4u + 2u] = dirty.y;
            // Winner whose tile lists update_framebuffers reuses; outside canonical state.
            chain_update_flags[chain_id * 4u + 3u] = best_offspring_id;

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

    if local_id < OPERATOR_COUNT {
        let offered = atomicLoad(&offered_ops[local_id]);
        if offered != 0u {
            atomicAdd(&operator_stats[local_id], offered);
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

