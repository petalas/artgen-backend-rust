// Fused rasterize + error compute shader — one thread per pixel per offspring
// Dispatch: (W/16, H/16, K*λ) workgroups of size (16, 16, 1)
// Each thread rasterizes all polygons at its pixel, computes L2 error against reference,
// then workgroup-reduces the error and thread 0 atomicAdds to per-offspring accumulator.
// Polygons are cooperatively loaded into shared memory in tiles of 768 (16 bytes each).

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
    migration_interval: u32,

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

    // Crossover & island params
    spatial_crossover_weight: f32,
    tournament_size: u32,
    island_count: u32,
    inter_island_interval: u32,

    // Chain count + lambda + padding
    chain_count_param: u32,
    single_mutation_mode: u32,
    lambda: u32,
    _pad8: u32,
}

@group(0) @binding(0) var<storage, read>       working_states:     array<DrawingState>;
@group(0) @binding(1)                          var reference_image: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             params:             Params;

var<workgroup> shared_polys: array<Polygon, 768>;   // 768 × 16 = 12,288 bytes
var<workgroup> shared_errors: array<u32, 256>;      // 16×16 = 256 threads

// Half-space edge function: positive if point (px,py) is on the left side of edge (ax,ay)->(bx,by)
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let px = gid.x;
    let py = gid.y;
    let chain_id = gid.z;

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
            let tile_cap = 768u;
            let tile_count = (poly_count + tile_cap - 1u) / tile_cap;

            for (var tile = 0u; tile < tile_count; tile++) {
                let tile_base = tile * tile_cap;
                let tile_end = min(tile_cap, poly_count - tile_base);

                // Cooperative load: each thread loads multiple polygons into shared memory
                // 256 threads loading up to 768 polygons = 3 loads per thread
                for (var load_pass = 0u; load_pass < 3u; load_pass++) {
                    let slot = local_idx + load_pass * 256u;
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

            // L2 error: Euclidean distance in RGB space
            let dr = ri - refr;
            let dg = gi - refg;
            let db = bi - refb;
            pixel_error = u32(sqrt(dr * dr + dg * dg + db * db));
        }
    }

    // Store in shared memory for workgroup reduction
    shared_errors[local_idx] = pixel_error;
    workgroupBarrier();

    // Binary reduction: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    var stride = 128u;
    while stride > 0u {
        if local_idx < stride {
            shared_errors[local_idx] += shared_errors[local_idx + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // Thread 0 adds workgroup sum to chain's accumulator
    if local_idx == 0u {
        atomicAdd(&error_accumulators[chain_id], shared_errors[0]);
    }
}
