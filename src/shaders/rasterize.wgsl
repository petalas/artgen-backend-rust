// Rasterize compute shader — one thread per pixel per chain
// Dispatch: (W/8, H/8, K) workgroups of size (8, 8, 1)
// Each thread composites all polygons at its pixel, writes packed RGBA u32

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
    _params_pad: u32,
}

@group(0) @binding(0) var<storage, read>       working_states: array<DrawingState>;
@group(0) @binding(1) var<storage, read_write>  render_targets: array<u32>;
@group(0) @binding(2) var<uniform>              params:         Params;

// Half-space edge function: positive if point (px,py) is on the left side of edge (ax,ay)→(bx,by)
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let chain_id = gid.z;

    let w = params.image_width;
    let h = params.image_height;

    if px >= w || py >= h {
        return;
    }

    let chain_count = arrayLength(&working_states);
    if chain_id >= chain_count {
        return;
    }

    // Pixel center in normalized coordinates
    let fx = (f32(px) + 0.5) / f32(w);
    let fy = (f32(py) + 0.5) / f32(h);

    // Start with white background, accumulate in registers
    var r = 255.0;
    var g = 255.0;
    var b = 255.0;

    let poly_count = working_states[chain_id].polygon_count;

    for (var i = 0u; i < poly_count; i++) {
        let poly = working_states[chain_id].polygons[i];

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

    // Pack RGBA u32 (matching CPU ABGR8888 / Rgba8Unorm layout: R in low byte)
    let ri = u32(clamp(r, 0.0, 255.0));
    let gi = u32(clamp(g, 0.0, 255.0));
    let bi = u32(clamp(b, 0.0, 255.0));
    let packed = ri | (gi << 8u) | (bi << 16u) | (255u << 24u);

    let pixel_idx = chain_id * w * h + py * w + px;
    render_targets[pixel_idx] = packed;
}
