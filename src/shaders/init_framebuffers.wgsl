// Initialize chain framebuffers: full rasterize from chain_states → framebuffers + total errors.
// Dispatch: (W/WG_X, H/WG_Y, active_chains) — one thread per pixel per chain.
// Uses 1D workgroup layout for subgroup intrinsics (naga constraint).
//
// Workgroup size is configurable — pipeline.rs uses string replacement on WG_X/WG_Y.

const WG_X: u32 = 16;
const WG_Y: u32 = 16;
const THREAD_COUNT: u32 = WG_X * WG_Y;

struct Polygon {
    data: vec4<u32>,   // [color_packed, v0_packed, v1_packed, v2_packed] — 16 bytes
}

struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    mutation_scale: f32,
    stagnation_counter: u32,
    rng_state: vec4<u32>,
    polygons: array<Polygon, 1000>,
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

fn unpack_color(p: Polygon) -> vec4<f32> {
    return unpack4x8unorm(p.data.x);
}

fn unpack_vertex(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xFFFFu) / 65535.0, f32(word >> 16u) / 65535.0);
}

fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

fn pack_fb_pixel(r: f32, g: f32, b: f32) -> u32 {
    let ri = u32(clamp(r, 0.0, 255.0));
    let gi = u32(clamp(g, 0.0, 255.0));
    let bi = u32(clamp(b, 0.0, 255.0));
    return ri | (gi << 8u) | (bi << 16u) | (255u << 24u);
}

@group(0) @binding(0) var<storage, read>       chain_states:       array<DrawingState>;
@group(0) @binding(1)                          var reference_image: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> chain_framebuffers: array<u32>;
@group(0) @binding(3) var<storage, read_write> chain_total_errors: array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             params:             Params;

// Shared memory for polygon tiling (same approach as rasterize_error)
const TILE_CAP: u32 = THREAD_COUNT * 3u;
const LOADS_PER_THREAD: u32 = 3u;
var<workgroup> shared_polys: array<Polygon, TILE_CAP>;
var<workgroup> shared_errors: array<u32, 128>;

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let local_x = local_idx % WG_X;
    let local_y = local_idx / WG_X;
    let px = wid.x * WG_X + local_x;
    let py = wid.y * WG_Y + local_y;
    let chain_id = wid.z;

    let w = params.image_width;
    let h = params.image_height;

    var pixel_error = 0u;
    var packed_pixel = pack_fb_pixel(255.0, 255.0, 255.0);  // white background

    if px < w && py < h {
        let chain_count = arrayLength(&chain_states);
        if chain_id < chain_count {
            let fx = (f32(px) + 0.5) / f32(w);
            let fy = (f32(py) + 0.5) / f32(h);

            var r = 255.0;
            var g = 255.0;
            var b = 255.0;

            // Brute-force rasterize with shared memory tiling
            let poly_count = chain_states[chain_id].polygon_count;
            let tile_count = (poly_count + TILE_CAP - 1u) / TILE_CAP;

            for (var tile = 0u; tile < tile_count; tile++) {
                let tile_base = tile * TILE_CAP;
                let tile_end = min(TILE_CAP, poly_count - tile_base);

                for (var load_pass = 0u; load_pass < LOADS_PER_THREAD; load_pass++) {
                    let slot = local_idx + load_pass * THREAD_COUNT;
                    if slot < tile_end {
                        shared_polys[slot] = chain_states[chain_id].polygons[tile_base + slot];
                    }
                }
                workgroupBarrier();

                for (var i = 0u; i < tile_end; i++) {
                    let poly = shared_polys[i];

                    let pv0 = unpack_vertex(poly.data.y);
                    let pv1 = unpack_vertex(poly.data.z);
                    let pv2 = unpack_vertex(poly.data.w);

                    // AABB culling
                    let bb_min_x = min(pv0.x, min(pv1.x, pv2.x));
                    let bb_max_x = max(pv0.x, max(pv1.x, pv2.x));
                    let bb_min_y = min(pv0.y, min(pv1.y, pv2.y));
                    let bb_max_y = max(pv0.y, max(pv1.y, pv2.y));

                    if fx >= bb_min_x && fx <= bb_max_x && fy >= bb_min_y && fy <= bb_max_y {
                        let e0 = edge_fn(pv0.x, pv0.y, pv1.x, pv1.y, fx, fy);
                        let e1 = edge_fn(pv1.x, pv1.y, pv2.x, pv2.y, fx, fy);
                        let e2 = edge_fn(pv2.x, pv2.y, pv0.x, pv0.y, fx, fy);
                        let all_pos = e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0;
                        let all_neg = e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0;
                        if all_pos || all_neg {
                            let pcolor = unpack_color(poly);
                            let alpha = pcolor.w;
                            let inv_alpha = 1.0 - alpha;
                            r = r * inv_alpha + pcolor.x * 255.0 * alpha;
                            g = g * inv_alpha + pcolor.y * 255.0 * alpha;
                            b = b * inv_alpha + pcolor.z * 255.0 * alpha;
                        }
                    }
                }
                workgroupBarrier();
            }

            let ri = clamp(r, 0.0, 255.0);
            let gi = clamp(g, 0.0, 255.0);
            let bi = clamp(b, 0.0, 255.0);

            // Write framebuffer pixel
            packed_pixel = pack_fb_pixel(ri, gi, bi);
            let fb_idx = chain_id * w * h + py * w + px;
            chain_framebuffers[fb_idx] = packed_pixel;

            // Compute error from quantized pixel values (matching what rasterize_error
            // reads from chain_framebuffers) to avoid drift in incremental total error.
            let qi = f32(packed_pixel & 0xFFu);
            let qg = f32((packed_pixel >> 8u) & 0xFFu);
            let qb = f32((packed_pixel >> 16u) & 0xFFu);
            let ref_color = textureLoad(reference_image, vec2<i32>(i32(px), i32(py)), 0);
            let refr = ref_color.x * 255.0;
            let refg = ref_color.y * 255.0;
            let refb = ref_color.z * 255.0;
            pixel_error = u32(abs(qi - refr) + abs(qg - refg) + abs(qb - refb));
        }
    }

    // Subgroup-accelerated reduction for total error
    let sg_sum = subgroupAdd(pixel_error);
    let sg_idx = local_idx / sg_size;
    if sg_inv_id == 0u {
        shared_errors[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    if local_idx == 0u {
        let num_subgroups = (THREAD_COUNT + sg_size - 1u) / sg_size;
        var total = 0u;
        for (var i = 0u; i < num_subgroups; i++) {
            total += shared_errors[i];
        }
        atomicAdd(&chain_total_errors[chain_id], total);
    }
}
