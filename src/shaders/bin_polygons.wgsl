// Polygon binning compute shader — assigns each polygon to the spatial tiles it overlaps.
// Dispatch: (offspring_count, BIN_ROW_BANDS, 1). Workgroup (offspring, band) bins
// only the offspring's consumed tiles (those meeting its dirty rectangle) in its
// band of tile rows. It computes tile bounds cooperatively, then one polygon
// bitmask per consumed tile column and row. A tile's list is its column mask AND
// its row mask, walked in ascending polygon order as alpha blending requires.
//
// This shader runs between mutate and rasterize_error. It also appends the
// band's consumed tiles that rasterize_error must run to raster_work (the
// consumed-tile test is in tiles.wgsl; the changed-polygon test is
// offspring_changes.wgsl's changed_meets_rect).

// common.wgsl, changes.wgsl, offspring_changes.wgsl, tiles.wgsl and the
// generated constants (TILE_W, TILE_H, TILE_MAX_POLYS, CHANGED_INDICES,
// WORK_COUNT, WORK_ENTRIES) are appended by programs.rs.

@group(0) @binding(0) var<storage, read>       working_states: array<DrawingState>;
@group(0) @binding(1) var<storage, read_write> tile_data:      array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_counts:    array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       chain_states:   array<DrawingState>;
@group(0) @binding(4) var<storage, read_write> raster_work:    array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             params:         Params;

// Shared scratch, 16,320 bytes, below the 16,352-byte downlevel limit.
// Per chunk of 32 * chunk_words polygons it holds packed tile-column bounds
// [0, chunk), packed tile-row bounds [chunk, 2 * chunk), then one membership
// word per (word, line) at mask_base + word * line_count + line. Lines are the
// consumed tile columns followed by the band's consumed tile rows; bit b of word
// k marks polygon chunk_base + 32k + b. A tile's polygons are its column mask AND
// its row mask. Every word read is written first, so the pipeline skips zeroing.
const SCRATCH_WORDS: u32 = 4080u;
var<workgroup> scratch: array<u32, SCRATCH_WORDS>;

fn polygon_tile_bounds(poly: Polygon, w: u32, h: u32, num_tiles_x: u32, num_tiles_y: u32) -> vec4<u32> {
    // Unpack vertices
    let v0 = unpack_vertex(poly.data.y);
    let v1 = unpack_vertex(poly.data.z);
    let v2 = unpack_vertex(poly.data.w);

    // AABB in normalized coords
    let bb_min_x = min(v0.x, min(v1.x, v2.x));
    let bb_max_x = max(v0.x, max(v1.x, v2.x));
    let bb_min_y = min(v0.y, min(v1.y, v2.y));
    let bb_max_y = max(v0.y, max(v1.y, v2.y));

    // Convert to pixel coords
    let px_min_x = u32(bb_min_x * f32(w));
    let px_max_x = min(u32(ceil(bb_max_x * f32(w))), w);
    let px_min_y = u32(bb_min_y * f32(h));
    let px_max_y = min(u32(ceil(bb_max_y * f32(h))), h);

    // Convert to tile coords
    let tile_min_x = px_min_x / TILE_W;
    let tile_max_x = min((px_max_x + TILE_W - 1u) / TILE_W, num_tiles_x);
    let tile_min_y = px_min_y / TILE_H;
    let tile_max_y = min((px_max_y + TILE_H - 1u) / TILE_H, num_tiles_y);
    return vec4<u32>(tile_min_x, tile_max_x, tile_min_y, tile_max_y);
}

@compute @workgroup_size(512, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32,
) {
    let offspring_id = wid.x;
    let offspring_count = arrayLength(&working_states);
    if offspring_id >= offspring_count {
        return;
    }

    let w = params.image_width;
    let h = params.image_height;
    let grid = tile_grid(w, h);
    let num_tiles_x = grid.x;
    let num_tiles_y = grid.y;
    let num_tiles = num_tiles_x * num_tiles_y;

    // Workgroup y owns one band of whole tile rows; any y count is valid.
    let band_rows = (num_tiles_y + groups.y - 1u) / groups.y;
    let row_lo = wid.y * band_rows;
    let row_hi = min(row_lo + band_rows, num_tiles_y);

    // Consumers read only consumed tiles (tiles.wgsl): rasterize_error, and
    // update_framebuffers for the selected winner. Other tiles keep stale counts
    // and indices.
    let dirty_min = offspring_dirty_min(offspring_id);
    let dirty_max = offspring_dirty_max(offspring_id);

    // Consumed rows and columns are contiguous; narrow the band to them.
    var first_row = row_lo;
    while first_row < row_hi && !tile_row_consumed(first_row, h, dirty_min, dirty_max) {
        first_row++;
    }
    var last_row = first_row;
    while last_row < row_hi && tile_row_consumed(last_row, h, dirty_min, dirty_max) {
        last_row++;
    }
    var first_col = 0u;
    while first_col < num_tiles_x && !tile_column_consumed(first_col, w, dirty_min, dirty_max) {
        first_col++;
    }
    var last_col = first_col;
    while last_col < num_tiles_x && tile_column_consumed(last_col, w, dirty_min, dirty_max) {
        last_col++;
    }
    let rows = last_row - first_row;
    let cols = last_col - first_col;
    if rows == 0u || cols == 0u {
        return;
    }

    let poly_count = working_states[offspring_id].polygon_count;
    let line_count = cols + rows;
    let band_tiles = cols * rows;

    // Every chunk's bounds plus masks fit: chunk_words * (64 + line_count) words.
    // Supported images give line_count <= 512, so chunk_words >= 7.
    let chunk_words = min(32u, SCRATCH_WORDS / (64u + line_count));
    let chunk_capacity = chunk_words * 32u;
    let row_base = chunk_capacity;
    let mask_base = 2u * chunk_capacity;
    // At least one chunk, so an empty drawing still stores zero counts.
    let chunk_count = max(1u, (poly_count + chunk_capacity - 1u) / chunk_capacity);
    // With one chunk, several lanes split one tile's words. Otherwise one lane
    // owns a tile across chunks and carries its count through tile_counts.
    let tile_lanes = select(1u, clamp(512u / band_tiles, 1u, 32u), chunk_count == 1u);

    for (var chunk = 0u; chunk < chunk_count; chunk++) {
        let chunk_base = chunk * chunk_capacity;
        let chunk_len = min(chunk_capacity, poly_count - min(poly_count, chunk_base));
        let used_words = (chunk_len + 31u) / 32u;

        // Each bound slot has exactly one writer, including partial lane batches.
        for (var i = local_idx; i < chunk_len; i += 512u) {
            let poly = working_states[offspring_id].polygons[chunk_base + i];
            let bounds = polygon_tile_bounds(poly, w, h, num_tiles_x, num_tiles_y);
            scratch[i] = bounds.x | (bounds.y << 16u);
            scratch[row_base + i] = bounds.z | (bounds.w << 16u);
        }
        workgroupBarrier();

        // Each (word, line) mask has exactly one writer. Lines are the consumed
        // columns, then the consumed rows of this band. Neighbouring lanes share
        // the word, so they read the same bound words.
        let task_count = used_words * line_count;
        for (var task = local_idx; task < task_count; task += 512u) {
            let word = task / line_count;
            let line = task % line_count;
            let is_row = line >= cols;
            let coord = select(first_col + line, first_row + line - cols, is_row);
            let bound_base = select(0u, row_base, is_row) + word * 32u;
            let bits = min(32u, chunk_len - word * 32u);
            var mask = 0u;
            for (var b = 0u; b < bits; b++) {
                let packed = scratch[bound_base + b];
                // Same half-open membership as the per-axis tile test.
                if coord >= (packed & 0xFFFFu) && coord < (packed >> 16u) {
                    mask |= 1u << b;
                }
            }
            scratch[mask_base + word * line_count + line] = mask;
        }
        workgroupBarrier();

        // tile_lanes lanes split each consumed tile's words; a lane's first slot is
        // the population of the tile's earlier words, so ascending words and bits
        // keep polygon order. The part-0 lane stores the uncapped count.
        let words_per_lane = (used_words + tile_lanes - 1u) / tile_lanes;
        for (var task = local_idx; task < band_tiles * tile_lanes; task += 512u) {
            let band_tile = task / tile_lanes;
            let part = task % tile_lanes;
            let col = band_tile % cols;
            let row = band_tile / cols;
            let tile_id = (first_row + row) * num_tiles_x + first_col + col;
            let tile_global = offspring_id * num_tiles + tile_id;
            let col_line = mask_base + col;
            let row_line = mask_base + cols + row;
            let word_lo = min(part * words_per_lane, used_words);
            let word_hi = min(word_lo + words_per_lane, used_words);

            var raw_count = 0u;
            if chunk > 0u {
                raw_count = atomicLoad(&tile_counts[tile_global]);
            }
            for (var word = 0u; word < word_lo; word++) {
                let offset = word * line_count;
                raw_count += countOneBits(scratch[col_line + offset] & scratch[row_line + offset]);
            }
            let data_base = offspring_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS;
            for (var word = word_lo; word < word_hi; word++) {
                let offset = word * line_count;
                var mask = scratch[col_line + offset] & scratch[row_line + offset];
                while mask != 0u {
                    let pi = chunk_base + word * 32u + countTrailingZeros(mask);
                    mask &= mask - 1u;
                    let slot = raw_count;
                    raw_count += 1u;
                    if slot < TILE_MAX_POLYS {
                        tile_data[data_base + slot] = pi;
                    }
                }
            }
            if part == 0u {
                for (var word = word_hi; word < used_words; word++) {
                    let offset = word * line_count;
                    raw_count += countOneBits(scratch[col_line + offset] & scratch[row_line + offset]);
                }
                atomicStore(&tile_counts[tile_global], raw_count);
            }
        }
        // The next chunk overwrites bounds and masks.
        workgroupBarrier();
    }

    // Raster worklist. A consumed tile is skipped only when the shortcut is on
    // and no changed version may meet it: rasterize_error used to return early
    // there, adding nothing to the error sums. Final counts are read back after
    // the storage barrier; the loop bound is uniform, so subgroup operations run
    // with all lanes active.
    storageBarrier();
    let parent_chain = offspring_id / params.lambda;
    let changed = offspring_changed_words(offspring_id);
    for (var round = 0u; round < band_tiles; round += 512u) {
        let band_tile = round + local_idx;
        var needed = false;
        var entry = 0u;
        if band_tile < band_tiles {
            let tile_x = first_col + band_tile % cols;
            let tile_y = first_row + band_tile / cols;
            entry = offspring_id * num_tiles + tile_y * num_tiles_x + tile_x;
            let shortcut = changes_shortcut(changed, atomicLoad(&tile_counts[entry]));
            needed = shortcut.x == 0u;
            if !needed {
                let tile_min_x = tile_x * TILE_W;
                let tile_min_y = tile_y * TILE_H;
                let tile_max_x = min(tile_min_x + TILE_W, w);
                let tile_max_y = min(tile_min_y + TILE_H, h);
                let tile_lo = vec2<f32>(f32(tile_min_x) / f32(w), f32(tile_min_y) / f32(h));
                let tile_hi = vec2<f32>(f32(tile_max_x) / f32(w), f32(tile_max_y) / f32(h));
                needed = changed_meets_rect(shortcut, parent_chain, offspring_id, tile_lo, tile_hi);
            }
        }
        let flag = select(0u, 1u, needed);
        let n = subgroupAdd(flag);
        let prefix = subgroupExclusiveAdd(flag);
        var base = 0u;
        if sg_inv_id == 0u && n != 0u {
            base = atomicAdd(&raster_work[WORK_COUNT], n);
        }
        base = subgroupBroadcastFirst(base);
        let slot = WORK_ENTRIES + base + prefix;
        if needed && slot < arrayLength(&raster_work) {
            atomicStore(&raster_work[slot], entry);
        }
    }
}
