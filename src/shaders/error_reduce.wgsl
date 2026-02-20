// Error computation + workgroup reduction compute shader
// Dispatch: (W/8, H/8, K) workgroups of size (8, 8, 1)
// Each thread computes per-pixel RGB error, then workgroup reduces via shared memory,
// thread 0 does atomicAdd to per-chain error accumulator.

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
    _params_pad: u32,
}

@group(0) @binding(0) var<storage, read>       render_targets:     array<u32>;   // packed RGBA per pixel per chain
@group(0) @binding(1) var<storage, read>       reference_image:    array<u32>;   // packed RGBA reference
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>; // one per chain
@group(0) @binding(3) var<uniform>             params:             Params;

var<workgroup> shared_errors: array<u32, 64>;  // 8x8 = 64 threads

@compute @workgroup_size(8, 8, 1)
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
        let pixel_idx = chain_id * w * h + py * w + px;
        let ref_idx = py * w + px;

        // Unpack rendered pixel
        let rendered = render_targets[pixel_idx];
        let rr = f32(rendered & 0xFFu);
        let rg = f32((rendered >> 8u) & 0xFFu);
        let rb = f32((rendered >> 16u) & 0xFFu);

        // Unpack reference pixel
        let reference = reference_image[ref_idx];
        let refr = f32(reference & 0xFFu);
        let refg = f32((reference >> 8u) & 0xFFu);
        let refb = f32((reference >> 16u) & 0xFFu);

        // Sum of absolute differences (L1 distance) — avoids expensive sqrt,
        // same selection ordering properties. Max per pixel = 255 * 3 = 765.
        let dr = abs(rr - refr);
        let dg = abs(rg - refg);
        let db = abs(rb - refb);
        pixel_error = u32(dr + dg + db);
    }

    // Store in shared memory for workgroup reduction
    shared_errors[local_idx] = pixel_error;
    workgroupBarrier();

    // Binary reduction: 64 → 32 → 16 → 8 → 4 → 2 → 1
    var stride = 32u;
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
