// Fused rasterize + error compute shader — one thread per pixel per offspring
// Dispatch: (W/WG_X, H/WG_Y, K*λ) workgroups of size (THREAD_COUNT, 1, 1)
// 1D workgroup layout enables subgroup intrinsics for the error reduction
// (naga 22 rejects subgroup builtins on multi-dimensional workgroups).
// Pixel coordinates are derived from workgroup_id + local_invocation_index.
//
// Each thread rasterizes all polygons at its pixel, computes L1 error against reference,
// then subgroupAdd reduces within each warp/wave and thread 0 sums across subgroups.
// Polygons are cooperatively loaded into shared memory in tiles (scaled to thread count).
//
// Workgroup size is configurable. pipeline.rs uses string replacement on the
// WG_X/WG_Y constants below when creating non-default pipeline variants.
// Supported configurations: 16x16 (256 threads), 16x8 (128 threads), 8x8 (64 threads).
// Tile capacity and shared memory arrays scale with WG_X * WG_Y.
const WG_X: u32 = 16;
const WG_Y: u32 = 16;
const THREAD_COUNT: u32 = WG_X * WG_Y;
const TILE_CAP: u32 = THREAD_COUNT * 3u;
const LOADS_PER_THREAD: u32 = 3u;

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
    lighten_color_prob: f32,

    darken_color_prob: f32,
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
    _pad1: u32,
    _pad2: u32,

    // Chain count + lambda + padding
    chain_count_param: u32,
    single_mutation_mode: u32,
    lambda: u32,
    adaptive_mutation: u32,
}

@group(0) @binding(0) var<storage, read>       working_states:     array<DrawingState>;
@group(0) @binding(1)                          var reference_image: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>;
var<immediate>                                 params:             Params;

// Shared memory arrays sized to thread count.
// shared_polys: TILE_CAP polygons (THREAD_COUNT * 3 * 16 bytes)
// shared_errors: one u32 per subgroup for cross-subgroup reduction (max 256/4 = 64 subgroups)
var<workgroup> shared_polys: array<Polygon, 1536>;   // max tile cap (512*3) — only TILE_CAP entries used
var<workgroup> shared_errors: array<u32, 128>;      // max subgroups — only ceil(THREAD_COUNT/sg_size) used

// Half-space edge function: positive if point (px,py) is on the left side of edge (ax,ay)->(bx,by)
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    // Derive 2D pixel coordinates from 1D thread index + workgroup ID
    let local_x = local_idx % WG_X;
    let local_y = local_idx / WG_X;
    let px = wid.x * WG_X + local_x;
    let py = wid.y * WG_Y + local_y;
    let chain_id = wid.z;

    let w = params.image_width;
    let h = params.image_height;

    var pixel_error = 0u;

    if px < w && py < h {
        let chain_count = arrayLength(&working_states);
        if chain_id < chain_count {
            // Pixel center in normalized coordinates
            let fx = (f32(px) + 0.5) / f32(w);
            let fy = (f32(py) + 0.5) / f32(h);

            // Start with white background, accumulate in registers
            var r = 255.0;
            var g = 255.0;
            var b = 255.0;

            let poly_count = working_states[chain_id].polygon_count;
            let tile_count = (poly_count + TILE_CAP - 1u) / TILE_CAP;

            for (var tile = 0u; tile < tile_count; tile++) {
                let tile_base = tile * TILE_CAP;
                let tile_end = min(TILE_CAP, poly_count - tile_base);

                // Cooperative load: each thread loads LOADS_PER_THREAD polygons into shared memory
                for (var load_pass = 0u; load_pass < LOADS_PER_THREAD; load_pass++) {
                    let slot = local_idx + load_pass * THREAD_COUNT;
                    if slot < tile_end {
                        shared_polys[slot] = working_states[chain_id].polygons[tile_base + slot];
                    }
                }
                workgroupBarrier();

                // Each thread tests its pixel against all polygons in this tile
                for (var i = 0u; i < tile_end; i++) {
                    let poly = shared_polys[i];

                    // Unpack vertices
                    let pv0 = unpack_vertex(poly.data.y);
                    let pv1 = unpack_vertex(poly.data.z);
                    let pv2 = unpack_vertex(poly.data.w);

                    // AABB culling: skip polygons whose bounding box doesn't contain this pixel
                    let bb_min_x = min(pv0.x, min(pv1.x, pv2.x));
                    let bb_max_x = max(pv0.x, max(pv1.x, pv2.x));
                    let bb_min_y = min(pv0.y, min(pv1.y, pv2.y));
                    let bb_max_y = max(pv0.y, max(pv1.y, pv2.y));

                    if fx < bb_min_x || fx > bb_max_x || fy < bb_min_y || fy > bb_max_y {
                        continue;
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
                        let pcolor = unpack_color(poly);
                        let alpha = pcolor.w;
                        let inv_alpha = 1.0 - alpha;
                        let src_r = pcolor.x * 255.0;
                        let src_g = pcolor.y * 255.0;
                        let src_b = pcolor.z * 255.0;
                        r = r * inv_alpha + src_r * alpha;
                        g = g * inv_alpha + src_g * alpha;
                        b = b * inv_alpha + src_b * alpha;
                    }
                }
                workgroupBarrier();
            }

            // Clamp rendered values
            let ri = clamp(r, 0.0, 255.0);
            let gi = clamp(g, 0.0, 255.0);
            let bi = clamp(b, 0.0, 255.0);

            // Load reference pixel from texture (Rgba8Unorm: automatically [0,1] float)
            let ref_color = textureLoad(reference_image, vec2<i32>(i32(px), i32(py)), 0);
            let refr = ref_color.x * 255.0;
            let refg = ref_color.y * 255.0;
            let refb = ref_color.z * 255.0;

            // L1 error: Manhattan distance in RGB space
            let dr = ri - refr;
            let dg = gi - refg;
            let db = bi - refb;
            pixel_error = u32(abs(dr) + abs(dg) + abs(db));
        }
    }

    // Subgroup-accelerated reduction: subgroupAdd within each warp/wave,
    // then thread 0 sums across subgroups via shared memory.
    // Replaces 8-step LDS tree reduction (8 barriers) with 1 barrier.
    let sg_sum = subgroupAdd(pixel_error);
    let sg_idx = local_idx / sg_size;

    if sg_inv_id == 0u {
        shared_errors[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    // Thread 0 sums across subgroups and atomicAdds to chain's accumulator
    if local_idx == 0u {
        let num_subgroups = (THREAD_COUNT + sg_size - 1u) / sg_size;
        var total = 0u;
        for (var i = 0u; i < num_subgroups; i++) {
            total += shared_errors[i];
        }
        atomicAdd(&error_accumulators[chain_id], total);
    }
}
