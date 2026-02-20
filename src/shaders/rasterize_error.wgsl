// Fused rasterize + error compute shader — one thread per pixel per chain
// Dispatch: (W/16, H/16, K) workgroups of size (16, 16, 1)
// Each thread rasterizes all polygons at its pixel, computes L1 error against reference,
// then workgroup-reduces the error and thread 0 atomicAdds to per-chain accumulator.
// Polygons are cooperatively loaded into shared memory in tiles of 256.

struct Polygon {
    color: vec4<f32>,
    v0: vec2<f32>,
    v1: vec2<f32>,
    v2: vec2<f32>,
    _pad: vec2<f32>,
}

struct DrawingState {
    polygon_count: u32,
    fitness_bits: u32,
    _pad0: u32,
    _pad1: u32,
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

    // Chain count + padding
    chain_count_param: u32,
    _pad6: u32,
    _pad7: u32,
    _pad8: u32,
}

@group(0) @binding(0) var<storage, read>       working_states:     array<DrawingState>;
@group(0) @binding(1) var<storage, read>       reference_image:    array<u32>;
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             params:             Params;

var<workgroup> shared_polys: array<Polygon, 256>;   // 256 × 48 = 12,288 bytes
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
            let tile_count = (poly_count + 255u) / 256u;

            for (var tile = 0u; tile < tile_count; tile++) {
                let tile_base = tile * 256u;
                let load_idx = tile_base + local_idx;

                // Cooperative load: each thread loads one polygon into shared memory
                if load_idx < poly_count {
                    shared_polys[local_idx] = working_states[chain_id].polygons[load_idx];
                }
                workgroupBarrier();

                // Each thread tests its pixel against all polygons in this tile
                let tile_end = min(256u, poly_count - tile_base);
                for (var i = 0u; i < tile_end; i++) {
                    let poly = shared_polys[i];

                    // AABB culling: skip polygons whose bounding box doesn't contain this pixel
                    let bb_min_x = min(poly.v0.x, min(poly.v1.x, poly.v2.x));
                    let bb_max_x = max(poly.v0.x, max(poly.v1.x, poly.v2.x));
                    let bb_min_y = min(poly.v0.y, min(poly.v1.y, poly.v2.y));
                    let bb_max_y = max(poly.v0.y, max(poly.v1.y, poly.v2.y));

                    if fx < bb_min_x || fx > bb_max_x || fy < bb_min_y || fy > bb_max_y {
                        continue;
                    }

                    // Half-space triangle test (3 edge evaluations)
                    let e0 = edge_fn(poly.v0.x, poly.v0.y, poly.v1.x, poly.v1.y, fx, fy);
                    let e1 = edge_fn(poly.v1.x, poly.v1.y, poly.v2.x, poly.v2.y, fx, fy);
                    let e2 = edge_fn(poly.v2.x, poly.v2.y, poly.v0.x, poly.v0.y, fx, fy);

                    // Inside if all same sign (handle both CW and CCW winding)
                    let all_pos = e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0;
                    let all_neg = e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0;

                    if all_pos || all_neg {
                        // Alpha blend: out = src * alpha + dst * (1 - alpha)
                        let alpha = poly.color.w;
                        let inv_alpha = 1.0 - alpha;
                        let src_r = poly.color.x * 255.0;
                        let src_g = poly.color.y * 255.0;
                        let src_b = poly.color.z * 255.0;
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

            // Unpack reference pixel
            let ref_idx = py * w + px;
            let reference = reference_image[ref_idx];
            let refr = f32(reference & 0xFFu);
            let refg = f32((reference >> 8u) & 0xFFu);
            let refb = f32((reference >> 16u) & 0xFFu);

            // L1 error: sum of absolute differences
            let dr = abs(ri - refr);
            let dg = abs(gi - refg);
            let db = abs(bi - refb);
            pixel_error = u32(dr + dg + db);
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
