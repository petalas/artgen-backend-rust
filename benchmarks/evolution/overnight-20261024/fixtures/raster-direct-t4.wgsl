// CACHE-ONLY PROTOTYPE: serial adjacent original raster tiles per workgroup.
// Raster dispatch: (ceil(ceil(W/WG_X)/T), ceil(H/WG_Y), K*lambda).
// Binning, tile buffers, original strides, and framebuffer update grids stay unchanged.
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
const RASTER_TILES_PER_GROUP: u32 = 4u;

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
var<workgroup> shared_skip_mask: u32; // lane 0 writes once, then immutable

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

// No barriers or cross-pixel floating-point state are permitted in this helper.
fn pixel_errors(
    px: u32, py: u32, w: u32, h: u32,
    parent_chain: u32, tile_poly_count: u32, valid_pixel: bool,
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

    let chain_count = arrayLength(&working_states);
    let num_tiles_x = (w + WG_X - 1u) / WG_X;
    let num_tiles_y = (h + WG_Y - 1u) / WG_Y;
    let num_tiles = num_tiles_x * num_tiles_y;

    // Padding is rejected before tile metadata. Preserve the strict AE1 predicate,
    // including zero-width and inverted dirty boxes, without endpoint normalization.
    if local_idx == 0u {
        let bbox_lo = working_states[chain_id].fitness_bits;
        let bbox_hi = working_states[chain_id].stagnation_counter;
        let dirty_min_x = bbox_lo & 0xFFFFu;
        let dirty_min_y = (bbox_lo >> 16u) & 0xFFFFu;
        let dirty_max_x = bbox_hi & 0xFFFFu;
        let dirty_max_y = (bbox_hi >> 16u) & 0xFFFFu;
        var mask = 0u;
        for (var s = 0u; s < RASTER_TILES_PER_GROUP; s++) {
            let mask_tile_x = wid.x * RASTER_TILES_PER_GROUP + s;
            if mask_tile_x >= num_tiles_x {
                mask |= 1u << s;
                continue;
            }
            let tile_min_x = mask_tile_x * WG_X;
            let tile_min_y = wid.y * WG_Y;
            let tile_max_x = min(tile_min_x + WG_X, w);
            let tile_max_y = min(tile_min_y + WG_Y, h);
            let no_overlap = tile_max_x <= dirty_min_x || tile_min_x >= dirty_max_x ||
                             tile_max_y <= dirty_min_y || tile_min_y >= dirty_max_y;
            if no_overlap {
                mask |= 1u << s;
            }
        }
        shared_skip_mask = mask;
    }
    let skip_mask = workgroupUniformLoad(&shared_skip_mask);
    if skip_mask == (1u << RASTER_TILES_PER_GROUP) - 1u {
        return;
    }

    for (var tile_offset = 0u; tile_offset < RASTER_TILES_PER_GROUP; tile_offset++) {
        if (skip_mask & (1u << tile_offset)) != 0u {
            continue;
        }
        let tile_x = wid.x * RASTER_TILES_PER_GROUP + tile_offset;
        let tile_id = wid.y * num_tiles_x + tile_x;
        var tile_poly_count = 0u;
        var tile_data_base = 0u;
        if chain_id < chain_count {
            let tile_global = chain_id * num_tiles + tile_id;
            tile_poly_count = min(tile_counts_buf[tile_global], TILE_MAX_POLYS);
            tile_data_base = chain_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS;
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
        for (var ordinal = local_idx; ordinal < THREAD_COUNT; ordinal += PHYSICAL_LANES) {
            let px = tile_x * WG_X + ordinal % WG_X;
            let py = wid.y * WG_Y + ordinal / WG_X;
            let valid_pixel = px < w && py < h && chain_id < chain_count;
            let errors = pixel_errors(px, py, w, h, parent_chain, tile_poly_count, valid_pixel);
            pixel_error += errors.x;
            pixel_error_old += errors.y;
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
        // The final reduction barrier completes every previous polygon read before
        // the next cache write. The next cache-load barrier waits for lane 0's
        // previous error reads before any next-tile error-array write. Skipped
        // iterations write nothing, and skip_mask never changes. No end barrier.
    }
}
