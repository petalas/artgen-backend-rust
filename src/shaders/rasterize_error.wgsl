// Paired pixels per physical lane, same logical raster tile.
// Dispatch: (n, 1, 1) for a fixed n (pipeline.rs). bin_polygons lists the
// (offspring, tile) pairs that have work; workgroup g rasterizes entries g,
// g + n, g + 2n, ... The list omits tiles outside the dirty rectangle and tiles no
// changed polygon meets, which added nothing to the error sums, so far fewer
// workgroups launch than the former (tiles_x, tiles_y, offspring) grid.
// 1D workgroup layout is retained for subgroup intrinsics.
// Each lane owns ordinal local_idx + k*PHYSICAL_LANES, within TILE_W*TILE_H.
// Each pixel retains C's f32 rasterization followed by integer byte-L1.
// Integer lane totals feed C's unchanged dense actual-subgroup-ID reduction.
// Cooperative decoded-cache loading uses every physical lane, even at image edges.
//
// common.wgsl, changes.wgsl, offspring_changes.wgsl, tiles.wgsl and the
// generated constants (TILE_W, TILE_H, TILE_MAX_POLYS, CHANGED_INDICES,
// TILE_OVERFLOW_ERROR, WORK_COUNT, WORK_ENTRIES) are appended by programs.rs.
// Supported logical tiles: 32x16, 16x16, 32x8, 16x8, 8x8.
// Shared polygon capacity remains the clamped 256-polygon limit.
const THREAD_COUNT: u32 = TILE_W * TILE_H;
const PHYSICAL_LANES: u32 = min(256u, THREAD_COUNT);

// Decoded cache layout: vec2<f32> alignment 8, scalar alignment 4.
struct DecodedPolygon {
    pv0: vec2<f32>,  // offset 0
    pv1: vec2<f32>,  // offset 8
    pv2: vec2<f32>,  // offset 16
    bb_min_x: f32,   // offset 24
    bb_max_x: f32,   // offset 28
    bb_min_y: f32,   // offset 32
    bb_max_y: f32,   // offset 36
    color_packed: u32, // offset 40; trailing padding 44..47; stride 48
}

@group(0) @binding(0) var<storage, read>       working_states:     array<DrawingState>;
@group(0) @binding(1)                          var reference_image: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       tile_data:          array<u32>;
@group(0) @binding(4) var<storage, read>       tile_counts_buf:    array<u32>;
@group(0) @binding(5) var<storage, read>       chain_states:       array<DrawingState>;
@group(0) @binding(6) var<storage, read>       chain_framebuffers: array<u32>;
@group(0) @binding(7) var<storage, read>       raster_work:        array<u32>;
@group(1) @binding(0) var<uniform>             params:             Params;

// Shared polygon capacity follows the tile polygon clamp.
// shared_polys: 256 decoded polygons, stride 48, 12288 bytes
// shared_errors: C's 128 slots retained; indexed by actual dense subgroup ID
var<workgroup> shared_polys: array<DecodedPolygon, TILE_MAX_POLYS>;
var<workgroup> shared_errors: array<u32, 128>;      // only num_subgroups entries are used
var<workgroup> shared_errors_old: array<u32, 128>;  // old errors for incremental eval

// Unchanged-pixel shortcut. mutate.wgsl lists every polygon index a
// mutated or crossover offspring changed in its change record
// (changes.wgsl). Slots hold the parent and offspring versions of up to
// CHANGED_INDICES indices. A pixel that none of them can cover has the same
// covering polygon sequence in parent and offspring, so its offspring colour is
// the cached parent framebuffer pixel and its new error equals its old error.
// Crossovers with more changed indices record unknown and keep the full blend
// loop.
var<workgroup> shared_changed: array<ChangedPolygon, CHANGED_SLOTS>;

/// Rasterize a polygon and alpha-blend if pixel is inside.
fn rasterize_blend(poly: DecodedPolygon, fx: f32, fy: f32, r: ptr<function, f32>, g: ptr<function, f32>, b: ptr<function, f32>) {
    // Read vertices decoded by the cooperative loader
    let pv0 = poly.pv0;
    let pv1 = poly.pv1;
    let pv2 = poly.pv2;

    // AABB culling: skip polygons whose bounding box doesn't contain this pixel
    let bb_min_x = poly.bb_min_x;
    let bb_max_x = poly.bb_max_x;
    let bb_min_y = poly.bb_min_y;
    let bb_max_y = poly.bb_max_y;

    if fx < bb_min_x || fx > bb_max_x || fy < bb_min_y || fy > bb_max_y {
        return;
    }

    // Half-space triangle test (3 edge evaluations)
    let e0 = edge_fn(pv0.x, pv0.y, pv1.x, pv1.y, fx, fy);
    let e1 = edge_fn(pv1.x, pv1.y, pv2.x, pv2.y, fx, fy);
    let e2 = edge_fn(pv2.x, pv2.y, pv0.x, pv0.y, fx, fy);

    // Inside if all same sign (handle both CW and CCW winding)
    let all_pos = e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0;
    let all_neg = e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0;

    if all_pos || all_neg {
        // Unpack color and alpha blend: out = src * alpha + dst * (1 - alpha)
        let pcolor = unpack_color(Polygon(vec4<u32>(poly.color_packed, 0u, 0u, 0u)));
        let alpha = pcolor.w;
        let inv_alpha = 1.0 - alpha;
        let src_r = pcolor.x * 255.0;
        let src_g = pcolor.y * 255.0;
        let src_b = pcolor.z * 255.0;
        *r = *r * inv_alpha + src_r * alpha;
        *g = *g * inv_alpha + src_g * alpha;
        *b = *b * inv_alpha + src_b * alpha;
    }
}

// Tests the first `slots` entries of shared_changed (workgroup-uniform).
fn may_change(fx: f32, fy: f32, slots: u32) -> bool {
    var touched = false;
    for (var k = 0u; k < slots; k++) {
        touched = touched || may_cover(shared_changed[k], fx, fy);
    }
    return touched;
}

// No barriers or cross-pixel floating-point state are permitted in this helper.
fn pixel_errors(
    px: u32, py: u32, w: u32, h: u32,
    parent_chain: u32, tile_poly_count: u32, valid_pixel: bool, shortcut_slots: u32,
) -> vec2<u32> {
    if !valid_pixel {
        return vec2<u32>(0u, 0u);
    }
    var pixel_error = 0u;
    var pixel_error_old = 0u;

    var fx = 0.0;
    var fy = 0.0;
    var r = 255.0;
    var g = 255.0;
    var b = 255.0;
    var refr = 0.0;
    var refg = 0.0;
    var refb = 0.0;

    fx = (f32(px) + 0.5) / f32(w);
    fy = (f32(py) + 0.5) / f32(h);

    let ref_color = textureLoad(reference_image, vec2<i32>(i32(px), i32(py)), 0);
    // Match init_framebuffers: L1 compares exact reference bytes, not the
    // approximate UNORM reconstruction whose error would be truncated.
    refr = round(ref_color.x * 255.0);
    refg = round(ref_color.y * 255.0);
    refb = round(ref_color.z * 255.0);

    // Read cached parent pixel from chain_framebuffers (O(1) per pixel).
    let fb_idx = parent_chain * w * h + py * w + px;
    let packed = chain_framebuffers[fb_idx];
    let old_ri = f32(packed & 0xFFu);
    let old_gi = f32((packed >> 8u) & 0xFFu);
    let old_bi = f32((packed >> 16u) & 0xFFu);
    pixel_error_old = u32(abs(old_ri - refr) + abs(old_gi - refg) + abs(old_bi - refb));
    if shortcut_slots != 0u && !may_change(fx, fy, shortcut_slots) {
        return vec2<u32>(pixel_error_old, pixel_error_old);
    }

    for (var i = 0u; i < tile_poly_count; i++) {
        rasterize_blend(shared_polys[i], fx, fy, &r, &g, &b);
    }

    // L1 error: quantize to u8 integers (floor) so error matches
    // framebuffer precision (pack_fb_pixel truncates to u8).
    let qr = floor(clamp(r, 0.0, 255.0));
    let qg = floor(clamp(g, 0.0, 255.0));
    let qb = floor(clamp(b, 0.0, 255.0));
    pixel_error = u32(abs(qr - refr) + abs(qg - refg) + abs(qb - refb));

    return vec2<u32>(pixel_error, pixel_error_old);
}

// Private state for one pixel. No color decoding or edge arithmetic moves here.
struct RasterPixelState {
    valid_pixel: bool,
    fx: f32,
    fy: f32,
    r: f32,
    g: f32,
    b: f32,
    refr: f32,
    refg: f32,
    refb: f32,
    pixel_error_old: u32,
}

fn paired_pixel_state(
    px: u32, py: u32, w: u32, h: u32,
    parent_chain: u32, valid_pixel: bool,
) -> RasterPixelState {
    if !valid_pixel {
        var invalid: RasterPixelState;
        return invalid;
    }
    var pixel_error_old = 0u;

    var fx = 0.0;
    var fy = 0.0;
    var r = 255.0;
    var g = 255.0;
    var b = 255.0;
    var refr = 0.0;
    var refg = 0.0;
    var refb = 0.0;

    fx = (f32(px) + 0.5) / f32(w);
    fy = (f32(py) + 0.5) / f32(h);

    let ref_color = textureLoad(reference_image, vec2<i32>(i32(px), i32(py)), 0);
    // Match init_framebuffers: L1 compares exact reference bytes, not the
    // approximate UNORM reconstruction whose error would be truncated.
    refr = round(ref_color.x * 255.0);
    refg = round(ref_color.y * 255.0);
    refb = round(ref_color.z * 255.0);

    // Read cached parent pixel from chain_framebuffers (O(1) per pixel).
    let fb_idx = parent_chain * w * h + py * w + px;
    let packed = chain_framebuffers[fb_idx];
    let old_ri = f32(packed & 0xFFu);
    let old_gi = f32((packed >> 8u) & 0xFFu);
    let old_bi = f32((packed >> 16u) & 0xFFu);
    pixel_error_old = u32(abs(old_ri - refr) + abs(old_gi - refg) + abs(old_bi - refb));

    return RasterPixelState(valid_pixel, fx, fy, r, g, b, refr, refg, refb, pixel_error_old);
}

fn paired_pixel_errors(pixel: RasterPixelState) -> vec2<u32> {
    if !pixel.valid_pixel {
        return vec2<u32>(0u, 0u);
    }
    var pixel_error = 0u;
    let r = pixel.r;
    let g = pixel.g;
    let b = pixel.b;
    let refr = pixel.refr;
    let refg = pixel.refg;
    let refb = pixel.refb;
    let pixel_error_old = pixel.pixel_error_old;

    // L1 error: quantize to u8 integers (floor) so error matches
    // framebuffer precision (pack_fb_pixel truncates to u8).
    let qr = floor(clamp(r, 0.0, 255.0));
    let qg = floor(clamp(g, 0.0, 255.0));
    let qb = floor(clamp(b, 0.0, 255.0));
    pixel_error = u32(abs(qr - refr) + abs(qg - refg) + abs(qb - refb));

    return vec2<u32>(pixel_error, pixel_error_old);
}

// A valid pixel that skipped blending keeps its parent colour and old error.
fn kept_or_blended_errors(pixel: RasterPixelState, blended: bool) -> vec2<u32> {
    if pixel.valid_pixel && !blended {
        return vec2<u32>(pixel.pixel_error_old, pixel.pixel_error_old);
    }
    return paired_pixel_errors(pixel);
}

// Used only when the supported logical tile has two ordinals per physical lane.
fn pixel_pair_errors(
    wid: vec3<u32>, local_idx: u32, w: u32, h: u32,
    parent_chain: u32, tile_poly_count: u32, valid_chain: bool, shortcut_slots: u32,
) -> vec2<u32> {
    let ordinal0 = local_idx;
    let ordinal1 = local_idx + PHYSICAL_LANES;
    let px0 = wid.x * TILE_W + ordinal0 % TILE_W;
    let py0 = wid.y * TILE_H + ordinal0 / TILE_W;
    let px1 = wid.x * TILE_W + ordinal1 % TILE_W;
    let py1 = wid.y * TILE_H + ordinal1 / TILE_W;
    let valid0 = ordinal0 < THREAD_COUNT && px0 < w && py0 < h && valid_chain;
    let valid1 = ordinal1 < THREAD_COUNT && px1 < w && py1 < h && valid_chain;
    var pixel0 = paired_pixel_state(px0, py0, w, h, parent_chain, valid0);
    var pixel1 = paired_pixel_state(px1, py1, w, h, parent_chain, valid1);
    if !valid0 && !valid1 {
        return vec2<u32>(0u, 0u);
    }
    let shortcut = shortcut_slots != 0u;
    let blend0 = valid0 && (!shortcut || may_change(pixel0.fx, pixel0.fy, shortcut_slots));
    let blend1 = valid1 && (!shortcut || may_change(pixel1.fx, pixel1.fy, shortcut_slots));
    let blend_count = select(0u, tile_poly_count, blend0 || blend1);

    var r0 = pixel0.r;
    var g0 = pixel0.g;
    var b0 = pixel0.b;
    var r1 = pixel1.r;
    var g1 = pixel1.g;
    var b1 = pixel1.b;

    for (var i = 0u; i < blend_count; i++) {
        let poly = shared_polys[i];
        if blend0 {
            rasterize_blend(poly, pixel0.fx, pixel0.fy, &r0, &g0, &b0);
        }
        if blend1 {
            rasterize_blend(poly, pixel1.fx, pixel1.fy, &r1, &g1, &b1);
        }
    }

    pixel0.r = r0;
    pixel0.g = g0;
    pixel0.b = b0;
    pixel1.r = r1;
    pixel1.g = g1;
    pixel1.b = b1;

    // Only the already-integer errors cross pixels; each component is at most 1530.
    return kept_or_blended_errors(pixel0, blend0) + kept_or_blended_errors(pixel1, blend1);
}

@compute @workgroup_size(PHYSICAL_LANES, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) group_count: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32,
    @builtin(subgroup_id) sg_idx: u32,
    @builtin(num_subgroups) num_subgroups: u32,
) {
    // The bound is a uniform read-only storage word, so every barrier in
    // raster_tile is reached by the whole workgroup. Shared memory needs no
    // barrier between entries: the next entry's shared writes follow the
    // previous entry's final barrier (readers done), and its subgroup leaders
    // write shared_errors only after its first barrier, which thread 0 reaches
    // after summing the previous entry.
    let count = raster_work[WORK_COUNT];
    for (var work_index = group_id.x; work_index < count; work_index += group_count.x) {
        raster_tile(raster_work[WORK_ENTRIES + work_index], local_idx, sg_inv_id, sg_idx, num_subgroups);
    }
}

// One worklist entry: offspring * num_tiles + tile_id.
fn raster_tile(entry: u32, local_idx: u32, sg_inv_id: u32, sg_idx: u32, num_subgroups: u32) {
    let w = params.image_width;
    let h = params.image_height;
    let grid = tile_grid(w, h);
    let num_tiles_x = grid.x;
    let num_tiles = num_tiles_x * grid.y;
    let tile_id = entry % num_tiles;
    // (tile x, tile y, offspring) as in the former direct dispatch.
    let wid = vec3<u32>(tile_id % num_tiles_x, tile_id / num_tiles_x, entry / num_tiles);
    let chain_id = wid.z;

    let lambda = params.lambda;
    let parent_chain = chain_id / lambda;

    // Tile metadata — workgroup-uniform, computed by all threads.
    // Must be outside the pixel bounds check so all threads (including those in
    // partial tiles with px >= w) participate in shared memory loads and barriers.
    let chain_count = arrayLength(&working_states);

    var tile_poly_count = 0u;
    var tile_data_base = 0u;
    var changed = vec2<u32>(0u, 0u);
    // A list longer than TILE_MAX_POLYS lost its last polygons, so this tile's
    // error is not the offspring's. Such a tile is always listed: its shortcut
    // is off (changes_shortcut). Flagging it makes select reject the offspring.
    var overflowed = false;
    if chain_id < chain_count {
        let tile_global = chain_id * num_tiles + tile_id;
        let raw_count = tile_counts_buf[tile_global];
        overflowed = raw_count > TILE_MAX_POLYS;
        tile_poly_count = min(raw_count, TILE_MAX_POLYS);
        tile_data_base = chain_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS;
        // Neither the parent's nor the offspring's list may be truncated.
        changed = changes_shortcut(offspring_changed_words(chain_id), raw_count);
    }
    // Slots to test per pixel; zero disables the shortcut.
    let shortcut_slots = changed_slot_count(changed);

    // Slot k holds index half k/2, from the parent (even k) or offspring (odd k).
    // An index at or beyond the drawing's count (including NO_CHANGES's 0xFFFE)
    // leaves the slot empty.
    if local_idx < CHANGED_SLOTS {
        var decoded = no_changed_polygon();
        let encoded = changed_half(changed, local_idx >> 1u);
        if encoded != 0u {
            let idx = encoded - 1u;
            if (local_idx & 1u) == 0u {
                if idx < chain_states[parent_chain].polygon_count {
                    decoded = decode_changed(chain_states[parent_chain].polygons[idx]);
                }
            } else if idx < working_states[chain_id].polygon_count {
                decoded = decode_changed(working_states[chain_id].polygons[idx]);
            }
        }
        shared_changed[local_idx] = decoded;
    }

    // Tile-culled rasterization: ALL threads participate in shared memory loads
    // (critical for partial tiles where out-of-bounds threads are responsible
    // for loading polygon slots that in-bounds threads need for rasterization).
    for (var slot = local_idx; slot < tile_poly_count; slot += PHYSICAL_LANES) {
        let poly_idx = tile_data[tile_data_base + slot];
        let poly = working_states[chain_id].polygons[poly_idx];
        let pv0 = unpack_vertex(poly.data.y);
        let pv1 = unpack_vertex(poly.data.z);
        let pv2 = unpack_vertex(poly.data.w);

        let bb_min_x = min(pv0.x, min(pv1.x, pv2.x));
        let bb_max_x = max(pv0.x, max(pv1.x, pv2.x));
        let bb_min_y = min(pv0.y, min(pv1.y, pv2.y));
        let bb_max_y = max(pv0.y, max(pv1.y, pv2.y));
        shared_polys[slot] = DecodedPolygon(
            pv0, pv1, pv2, bb_min_x, bb_max_x, bb_min_y, bb_max_y, poly.data.x);
    }
    workgroupBarrier();

    // Each ordinal belongs to one physical lane; only integer L1 crosses pixels.
    var pixel_error = 0u;
    var pixel_error_old = 0u;
    if THREAD_COUNT == PHYSICAL_LANES {
        for (var ordinal = local_idx; ordinal < THREAD_COUNT; ordinal += PHYSICAL_LANES) {
            let px = wid.x * TILE_W + ordinal % TILE_W;
            let py = wid.y * TILE_H + ordinal / TILE_W;
            let valid_pixel = px < w && py < h && chain_id < chain_count;
            let errors = pixel_errors(
                px, py, w, h, parent_chain, tile_poly_count, valid_pixel, shortcut_slots);
            pixel_error += errors.x;
            pixel_error_old += errors.y;
        }
    } else {
        let errors = pixel_pair_errors(
            wid, local_idx, w, h, parent_chain, tile_poly_count, chain_id < chain_count, shortcut_slots);
        pixel_error = errors.x;
        pixel_error_old = errors.y;
    }

    // Subgroup leaders write distinct, dense slots before thread 0 sums them.
    // Subgroup IDs cannot be inferred from local invocation indices.
    let sg_sum = subgroupAdd(pixel_error);
    if sg_inv_id == 0u {
        shared_errors[sg_idx] = sg_sum;
    }

    let sg_sum_old = subgroupAdd(pixel_error_old);
    if sg_inv_id == 0u {
        shared_errors_old[sg_idx] = sg_sum_old;
    }

    workgroupBarrier();

    // Thread 0 sums across subgroups and atomicAdds to chain's accumulator (stride-2)
    if local_idx == 0u {
        var total = 0u;
        var total_old = 0u;
        for (var i = 0u; i < num_subgroups; i++) {
            total += shared_errors[i];
            total_old += shared_errors_old[i];
        }
        atomicAdd(&error_accumulators[chain_id * 2u], total);
        atomicAdd(&error_accumulators[chain_id * 2u + 1u], total_old);
        // The sums stay below the flag bit (execution::error_sum_fits), so the
        // order of these atomics does not matter.
        if overflowed {
            atomicOr(&error_accumulators[chain_id * 2u], TILE_OVERFLOW_ERROR);
        }
    }
}
