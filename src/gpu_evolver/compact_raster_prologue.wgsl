    let w = params.image_width;
    let h = params.image_height;
    let lambda = params.lambda;
    let count = params.chain_count_param * lambda;
    let q = wid.x + grid.x * wid.y;
    // Workgroup-uniform padding return, including partial final dispatch rows.
    if q >= compact_prefix[count] {
        return;
    }
    if local_idx == 0u {
        var lower = 0u;
        var upper = count;
        // Strict upper bound skips duplicate prefixes. At most 32 u32 steps.
        for (var step = 0u; step < 32u; step++) {
            if lower >= upper {
                break;
            }
            let middle = lower + (upper - lower) / 2u;
            if compact_prefix[middle + 1u] > q {
                upper = middle;
            } else {
                lower = middle + 1u;
            }
        }
        let bbox_lo = working_states[lower].fitness_bits;
        let bbox_hi = working_states[lower].stagnation_counter;
        shared_dirty_min_x = bbox_lo & 0xFFFFu;
        shared_dirty_min_y = (bbox_lo >> 16u) & 0xFFFFu;
        shared_dirty_max_x = bbox_hi & 0xFFFFu;
        shared_dirty_max_y = (bbox_hi >> 16u) & 0xFFFFu;
        let start_x = shared_dirty_min_x / WG_X;
        let start_y = shared_dirty_min_y / WG_Y;
        let end_x = min((shared_dirty_max_x + WG_X - 1u) / WG_X, (w + WG_X - 1u) / WG_X);
        let local_tile = q - compact_prefix[lower];
        let rect_width = end_x - start_x;
        shared_compact_chain = lower;
        shared_compact_x = start_x + local_tile % rect_width;
        shared_compact_y = start_y + local_tile / rect_width;

        let tile_min_x = shared_compact_x * WG_X;
        let tile_min_y = shared_compact_y * WG_Y;
        let tile_max_x = min(tile_min_x + WG_X, w);
        let tile_max_y = min(tile_min_y + WG_Y, h);
        let no_overlap = tile_max_x <= shared_dirty_min_x || tile_min_x >= shared_dirty_max_x ||
                         tile_max_y <= shared_dirty_min_y || tile_min_y >= shared_dirty_max_y;
        shared_skip_tile = select(0u, 1u, no_overlap);
    }

    workgroupBarrier();
    if shared_skip_tile == 1u {
        return;
    }

    // Mapping is published at the original metadata barrier, without another barrier.
    let chain_id = shared_compact_chain;
    let px = shared_compact_x * WG_X + local_idx % WG_X;
    let py = shared_compact_y * WG_Y + local_idx / WG_X;
    let parent_chain = chain_id / lambda;
