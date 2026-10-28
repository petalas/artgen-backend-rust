// SOURCE-ONLY PROTOTYPE: paired pixels per physical lane, same logical raster tile.
// Dispatch and binning remain (ceil(W/WG_X), ceil(H/WG_Y), K*lambda).
// 1D workgroup layout is retained for subgroup intrinsics.
// Each lane owns ordinal local_idx + k*PHYSICAL_LANES, within WG_X*WG_Y.
// Each pixel retains C's f32 rasterization followed by integer byte-L1.
// Integer lane totals feed C's unchanged dense actual-subgroup-ID reduction.
// Cooperative decoded-cache loading uses every physical lane, even at image edges.
//
// pipeline.rs specializes the exact WG_X/WG_Y markers below.
// Supported logical tiles: 32x16, 16x16, 32x8, 16x8, 8x8.
// Shared polygon capacity remains the clamped 256-polygon limit.
const WG_X: u32 = 16;
const WG_Y: u32 = 16;
const THREAD_COUNT: u32 = WG_X * WG_Y;
const PHYSICAL_LANES: u32 = min(256u, THREAD_COUNT);
const TILE_MAX_POLYS: u32 = 256u;

struct Polygon {
    data: vec4<u32>,   // [color_packed, v0_packed, v1_packed, v2_packed] — 16 bytes
}

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

struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    mutation_scale: f32,
    stagnation_counter: u32,
    rng_state: vec4<u32>,
    polygons: array<Polygon, 1000>,
}

// --- Quantized polygon unpack helpers ---

fn unpack_color(p: Polygon) -> vec4<f32> {
    return unpack4x8unorm(p.data.x);
}

fn unpack_vertex(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xFFFFu) / 65535.0, f32(word >> 16u) / 65535.0);
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
    _pad_ie: u32,
    _pad_tc: u32,

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

@group(0) @binding(0) var<storage, read>       working_states:     array<DrawingState>;
@group(0) @binding(1)                          var reference_image: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       tile_data:          array<u32>;
@group(0) @binding(4) var<storage, read>       tile_counts_buf:    array<u32>;
@group(0) @binding(5) var<storage, read>       chain_states:       array<DrawingState>;
@group(0) @binding(6) var<storage, read>       chain_framebuffers: array<u32>;
@group(1) @binding(0) var<uniform>             params:             Params;

// Shared polygon capacity follows the tile polygon clamp.
// shared_polys: 256 decoded polygons, stride 48, 12288 bytes
// shared_errors: C's 128 slots retained; indexed by actual dense subgroup ID
var<workgroup> shared_polys: array<DecodedPolygon, TILE_MAX_POLYS>;
var<workgroup> shared_errors: array<u32, 128>;      // only num_subgroups entries are used
var<workgroup> shared_errors_old: array<u32, 128>;  // old errors for incremental eval

// Unchanged-pixel shortcut. mutate.wgsl lists every polygon index a
// single-mutation or crossover offspring changed in rng_state.y and .z (see
// changed_indices there). Slots hold the parent and offspring versions of up to
// four indices. A pixel that none of them can cover has the same covering
// polygon sequence in parent and offspring, so its offspring colour is the
// cached parent framebuffer pixel and its new error equals its old error.
// Multi-mutation offspring, and crossovers with more changed indices, store
// zero and keep the full blend loop.
const CHANGED_INDICES: u32 = 4u;
const CHANGED_SLOTS: u32 = 2u * CHANGED_INDICES;
// Conservative coverage margin in normalized units (a pixel is ~2.6e-3 at 384 px).
// It absorbs any rounding difference from the blend loop's own coverage test.
const COVER_MARGIN: f32 = 1e-5;
var<workgroup> shared_changed: array<DecodedPolygon, CHANGED_SLOTS>;

// Half-space edge function: positive if point (px,py) is on the left side of edge (ax,ay)->(bx,by)
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

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

fn decode_changed(poly: Polygon) -> DecodedPolygon {
    let pv0 = unpack_vertex(poly.data.y);
    let pv1 = unpack_vertex(poly.data.z);
    let pv2 = unpack_vertex(poly.data.w);
    return DecodedPolygon(
        pv0, pv1, pv2,
        min(pv0.x, min(pv1.x, pv2.x)), max(pv0.x, max(pv1.x, pv2.x)),
        min(pv0.y, min(pv1.y, pv2.y)), max(pv0.y, max(pv1.y, pv2.y)),
        poly.data.x);
}

// Empty slots have an inverted box outside the unit square.
fn no_changed_polygon() -> DecodedPolygon {
    return DecodedPolygon(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), 2.0, -1.0, 2.0, -1.0, 0u);
}

// Superset of rasterize_blend's coverage test: false only when the pixel is
// outside the polygon by more than COVER_MARGIN.
fn may_cover(poly: DecodedPolygon, fx: f32, fy: f32) -> bool {
    if fx < poly.bb_min_x - COVER_MARGIN || fx > poly.bb_max_x + COVER_MARGIN ||
       fy < poly.bb_min_y - COVER_MARGIN || fy > poly.bb_max_y + COVER_MARGIN {
        return false;
    }
    let e0 = edge_fn(poly.pv0.x, poly.pv0.y, poly.pv1.x, poly.pv1.y, fx, fy);
    let e1 = edge_fn(poly.pv1.x, poly.pv1.y, poly.pv2.x, poly.pv2.y, fx, fy);
    let e2 = edge_fn(poly.pv2.x, poly.pv2.y, poly.pv0.x, poly.pv0.y, fx, fy);
    let all_pos = e0 >= -COVER_MARGIN && e1 >= -COVER_MARGIN && e2 >= -COVER_MARGIN;
    let all_neg = e0 <= COVER_MARGIN && e1 <= COVER_MARGIN && e2 <= COVER_MARGIN;
    return all_pos || all_neg;
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
    let px0 = wid.x * WG_X + ordinal0 % WG_X;
    let py0 = wid.y * WG_Y + ordinal0 / WG_X;
    let px1 = wid.x * WG_X + ordinal1 % WG_X;
    let py1 = wid.y * WG_Y + ordinal1 / WG_X;
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
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32,
    @builtin(subgroup_id) sg_idx: u32,
    @builtin(num_subgroups) num_subgroups: u32,
) {
    let chain_id = wid.z;

    let w = params.image_width;
    let h = params.image_height;
    let lambda = params.lambda;
    let parent_chain = chain_id / lambda;

    // --- Dirty bbox tile skip ---
    // Every lane reads the same header words from read-only storage, so the
    // skip is workgroup-uniform and returns before any shared memory or barrier.
    // Workgroups fully outside the dirty region (most of the grid) do nothing else.
    let bbox_lo = working_states[chain_id].fitness_bits;
    let bbox_hi = working_states[chain_id].stagnation_counter;
    let dirty_min_x = bbox_lo & 0xFFFFu;
    let dirty_min_y = (bbox_lo >> 16u) & 0xFFFFu;
    let dirty_max_x = bbox_hi & 0xFFFFu;
    let dirty_max_y = (bbox_hi >> 16u) & 0xFFFFu;
    let tile_min_x = wid.x * WG_X;
    let tile_min_y = wid.y * WG_Y;
    let tile_max_x = min(tile_min_x + WG_X, w);
    let tile_max_y = min(tile_min_y + WG_Y, h);
    if tile_max_x <= dirty_min_x || tile_min_x >= dirty_max_x ||
       tile_max_y <= dirty_min_y || tile_min_y >= dirty_max_y {
        return;
    }

    // Tile metadata — workgroup-uniform, computed by all threads.
    // Must be outside the pixel bounds check so all threads (including those in
    // partial tiles with px >= w) participate in shared memory loads and barriers.
    let chain_count = arrayLength(&working_states);
    let num_tiles_x = (w + WG_X - 1u) / WG_X;
    let num_tiles_y = (h + WG_Y - 1u) / WG_Y;
    let num_tiles = num_tiles_x * num_tiles_y;
    let tile_id = wid.y * num_tiles_x + wid.x;

    var tile_poly_count = 0u;
    var tile_data_base = 0u;
    var changed = vec2<u32>(0u, 0u);
    if chain_id < chain_count {
        let tile_global = chain_id * num_tiles + tile_id;
        let raw_count = tile_counts_buf[tile_global];
        tile_poly_count = min(raw_count, TILE_MAX_POLYS);
        tile_data_base = chain_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS;
        let first = working_states[chain_id].rng_state.y;
        var second = 0u;
        if first != 0u {
            second = working_states[chain_id].rng_state.z;
        }
        // Neither the parent's nor the offspring's list may be truncated: each
        // changed index moves at most one polygon in or out of this tile.
        let listed = select(2u, CHANGED_INDICES, second != 0u);
        if raw_count + listed <= TILE_MAX_POLYS {
            changed = vec2<u32>(first, second);
        }
    }
    // Slots to test per pixel; zero disables the shortcut.
    let shortcut_slots = select(0u, select(4u, CHANGED_SLOTS, changed.y != 0u), changed.x != 0u);

    // Slot k holds index half k/2, from the parent (even k) or offspring (odd k).
    // Halves 0..1 are in .y, 2..3 in .z. An index at or beyond the drawing's
    // count (including NO_CHANGES's 0xFFFE) leaves the slot empty.
    if local_idx < CHANGED_SLOTS {
        var decoded = no_changed_polygon();
        let half = local_idx >> 1u;
        let word = select(changed.x, changed.y, half >= 2u);
        let encoded = (word >> (16u * (half & 1u))) & 0xFFFFu;
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
            let px = wid.x * WG_X + ordinal % WG_X;
            let py = wid.y * WG_Y + ordinal / WG_X;
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
    }
}
