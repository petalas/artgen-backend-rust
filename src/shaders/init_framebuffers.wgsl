// Initialize chain framebuffers: full rasterize from chain_states → framebuffers + total errors.
// Dispatch: (W/TILE_W, H/TILE_H, active_chains) — one thread per pixel per chain.
// Uses 1D workgroup layout for subgroup intrinsics (naga constraint).
//
// common.wgsl, tiles.wgsl and the generated constants (TILE_W, TILE_H) are
// appended by programs.rs.

const THREAD_COUNT: u32 = TILE_W * TILE_H;

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
    @builtin(subgroup_id) sg_idx: u32,
    @builtin(num_subgroups) num_subgroups: u32,
) {
    let local_x = local_idx % TILE_W;
    let local_y = local_idx / TILE_W;
    let px = wid.x * TILE_W + local_x;
    let py = wid.y * TILE_H + local_y;
    let chain_id = wid.z;

    let w = params.image_width;
    let h = params.image_height;

    var pixel_error = 0u;
    var packed_pixel = pack_fb_pixel(255.0, 255.0, 255.0);  // white background

    // Polygon count — workgroup-uniform, must be outside pixel bounds check so all
    // threads (including partial-tile out-of-bounds threads) participate in shared
    // memory loads and barriers.
    let chain_count = arrayLength(&chain_states);
    var poly_count = 0u;
    if chain_id < chain_count {
        poly_count = chain_states[chain_id].polygon_count;
    }
    let tile_count = (poly_count + TILE_CAP - 1u) / TILE_CAP;

    let valid_pixel = px < w && py < h && chain_id < chain_count;
    var fx = 0.0;
    var fy = 0.0;
    var r = 255.0;
    var g = 255.0;
    var b = 255.0;

    if valid_pixel {
        fx = (f32(px) + 0.5) / f32(w);
        fy = (f32(py) + 0.5) / f32(h);
    }

    // Brute-force rasterize with shared memory tiling — ALL threads participate
    // in loading (critical for partial tiles where out-of-bounds threads are
    // responsible for loading polygon slots that in-bounds threads need).
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

        if valid_pixel {
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
        }
        workgroupBarrier();
    }

    if valid_pixel {
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
        // Recover the Rgba8Unorm source bytes before L1 subtraction. A slightly
        // noninteger reconstruction can otherwise lose one when cast to u32.
        let refr = round(ref_color.x * 255.0);
        let refg = round(ref_color.y * 255.0);
        let refb = round(ref_color.z * 255.0);
        pixel_error = u32(abs(qi - refr) + abs(qg - refg) + abs(qb - refb));
    }

    // Use the actual subgroup ID so every consumed slot has exactly one writer.
    let sg_sum = subgroupAdd(pixel_error);
    if sg_inv_id == 0u {
        shared_errors[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    if local_idx == 0u {
        var total = 0u;
        for (var i = 0u; i < num_subgroups; i++) {
            total += shared_errors[i];
        }
        atomicAdd(&chain_total_errors[chain_id], total);
    }
}
