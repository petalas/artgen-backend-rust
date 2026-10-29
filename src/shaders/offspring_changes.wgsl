// Readers of the offspring change record (changes.wgsl), for the stages that
// read it: select, bin_polygons, raster_worklist and rasterize_error. Each binds
// working_states and chain_states. gpu_evolver::programs appends this file
// after changes.wgsl. mutate is the only writer (write_dirty_box and
// write_changed_words there).

// The dirty rectangle's corners, packed as in pack_dirty_box.
fn offspring_dirty_words(offspring: u32) -> vec2<u32> {
    return vec2<u32>(working_states[offspring].fitness_bits,
                     working_states[offspring].stagnation_counter);
}

fn offspring_dirty_min(offspring: u32) -> vec2<u32> {
    return unpack_dirty_corner(working_states[offspring].fitness_bits);
}

fn offspring_dirty_max(offspring: u32) -> vec2<u32> {
    return unpack_dirty_corner(working_states[offspring].stagnation_counter);
}

// The changed-index words; the second is zero when the first is unknown.
fn offspring_changed_words(offspring: u32) -> vec2<u32> {
    let first = working_states[offspring].rng_state.y;
    var second = 0u;
    if first != 0u {
        second = working_states[offspring].rng_state.z;
    }
    return vec2<u32>(first, second);
}

// Whether any parent or offspring version of the indices in `words` (from
// changes_shortcut) may cover a pixel of the normalized rectangle [lo, hi].
// Indices at or beyond a drawing's count are empty.
fn changed_meets_rect(
    words: vec2<u32>, parent_chain: u32, offspring: u32, lo: vec2<f32>, hi: vec2<f32>,
) -> bool {
    let parent_count = chain_states[parent_chain].polygon_count;
    let offspring_count = working_states[offspring].polygon_count;
    var meets = false;
    for (var half = 0u; half < CHANGED_INDICES; half++) {
        let encoded = changed_half(words, half);
        if encoded != 0u {
            let idx = encoded - 1u;
            if idx < parent_count {
                meets = meets || changed_may_meet_rect(chain_states[parent_chain].polygons[idx], lo, hi);
            }
            if idx < offspring_count {
                meets = meets || changed_may_meet_rect(working_states[offspring].polygons[idx], lo, hi);
            }
        }
    }
    return meets;
}
