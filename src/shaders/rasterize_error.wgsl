// Fused rasterize + error compute shader — one thread per pixel per offspring
// Dispatch: (W/WG_X, H/WG_Y, K*lambda) workgroups of size (THREAD_COUNT, 1, 1)
// 1D workgroup layout required for subgroup intrinsics (naga constraint).
// Pixel coordinates are derived from workgroup_id + local_invocation_index.
//
// Each thread rasterizes all polygons at its pixel, computes L1 error against reference,
// then subgroupAdd reduces within each warp/wave and thread 0 sums across subgroups.
// Polygons are cooperatively loaded into shared memory in tiles (scaled to thread count).
//
// When tile culling is enabled (params.tile_culling == 1), only polygons that overlap
// this workgroup's spatial tile are loaded from tile_data/tile_counts instead of iterating
// all polygons. The binning pass must run before this shader.
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
const TILE_MAX_POLYS: u32 = 256u;

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
    incremental_eval: u32,
    tile_culling: u32,

    // Chain count + lambda + padding
    chain_count_param: u32,
    single_mutation_mode: u32,
    lambda: u32,
    adaptive_mutation: u32,
}

@group(0) @binding(0) var<storage, read>       working_states:     array<DrawingState>;
@group(0) @binding(1)                          var reference_image: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> error_accumulators: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       tile_data:          array<u32>;
@group(0) @binding(4) var<storage, read>       tile_counts_buf:    array<u32>;
@group(0) @binding(5) var<storage, read>       chain_states:       array<DrawingState>;
var<immediate>                                 params:             Params;

// Shared memory arrays sized to thread count.
// shared_polys: TILE_CAP polygons (THREAD_COUNT * 3 * 16 bytes)
// shared_errors: one u32 per subgroup for cross-subgroup reduction (max 256/4 = 64 subgroups)
var<workgroup> shared_polys: array<Polygon, 1536>;   // max tile cap (512*3) — only TILE_CAP entries used
var<workgroup> shared_errors: array<u32, 128>;      // max subgroups — only ceil(THREAD_COUNT/sg_size) used
var<workgroup> shared_errors_old: array<u32, 128>;  // old errors for incremental eval
var<workgroup> shared_skip_tile: u32;               // set by thread 0 if tile is outside dirty bbox
// Dirty bbox unpacked from offspring header (set by thread 0, read after barrier)
var<workgroup> shared_dirty_min_x: u32;
var<workgroup> shared_dirty_min_y: u32;
var<workgroup> shared_dirty_max_x: u32;
var<workgroup> shared_dirty_max_y: u32;

// Half-space edge function: positive if point (px,py) is on the left side of edge (ax,ay)->(bx,by)
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

/// Rasterize a polygon and alpha-blend if pixel is inside.
fn rasterize_blend(poly: Polygon, fx: f32, fy: f32, r: ptr<function, f32>, g: ptr<function, f32>, b: ptr<function, f32>) {
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
        let pcolor = unpack_color(poly);
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
    let lambda = params.lambda;
    let incremental = params.incremental_eval == 1u;

    // --- Incremental eval: dirty bbox tile skip ---
    // Thread 0 reads the dirty bbox from offspring header and checks if this tile overlaps.
    // Parent chain for this offspring: chain_id / lambda
    let parent_chain = chain_id / lambda;

    if incremental && local_idx == 0u {
        // Unpack dirty bbox from offspring header
        let bbox_lo = working_states[chain_id].fitness_bits;
        let bbox_hi = working_states[chain_id].stagnation_counter;
        shared_dirty_min_x = bbox_lo & 0xFFFFu;
        shared_dirty_min_y = (bbox_lo >> 16u) & 0xFFFFu;
        shared_dirty_max_x = bbox_hi & 0xFFFFu;
        shared_dirty_max_y = (bbox_hi >> 16u) & 0xFFFFu;

        // This tile's pixel range
        let tile_min_x = wid.x * WG_X;
        let tile_min_y = wid.y * WG_Y;
        let tile_max_x = min(tile_min_x + WG_X, w);
        let tile_max_y = min(tile_min_y + WG_Y, h);

        // Check overlap: skip if tile and dirty bbox don't intersect
        let no_overlap = tile_max_x <= shared_dirty_min_x || tile_min_x >= shared_dirty_max_x ||
                         tile_max_y <= shared_dirty_min_y || tile_min_y >= shared_dirty_max_y;
        shared_skip_tile = select(0u, 1u, no_overlap);
    }

    if incremental {
        workgroupBarrier();
        if shared_skip_tile == 1u {
            return;
        }
    }

    var pixel_error = 0u;
    var pixel_error_old = 0u;

    if px < w && py < h {
        let chain_count = arrayLength(&working_states);
        if chain_id < chain_count {
            // Pixel center in normalized coordinates
            let fx = (f32(px) + 0.5) / f32(w);
            let fy = (f32(py) + 0.5) / f32(h);

            // Load reference pixel from texture (Rgba8Unorm: automatically [0,1] float)
            let ref_color = textureLoad(reference_image, vec2<i32>(i32(px), i32(py)), 0);
            let refr = ref_color.x * 255.0;
            let refg = ref_color.y * 255.0;
            let refb = ref_color.z * 255.0;

            // --- Incremental: re-rasterize parent for old pixel error ---
            // Uses chain_states (parent) instead of a u8 framebuffer to avoid
            // quantization mismatch that causes error drift.
            if incremental {
                var old_r = 255.0;
                var old_g = 255.0;
                var old_b = 255.0;
                let parent_poly_count = chain_states[parent_chain].polygon_count;
                for (var pi = 0u; pi < parent_poly_count; pi++) {
                    rasterize_blend(chain_states[parent_chain].polygons[pi], fx, fy, &old_r, &old_g, &old_b);
                }
                let old_ri = clamp(old_r, 0.0, 255.0);
                let old_gi = clamp(old_g, 0.0, 255.0);
                let old_bi = clamp(old_b, 0.0, 255.0);
                pixel_error_old = u32(abs(old_ri - refr) + abs(old_gi - refg) + abs(old_bi - refb));
            }

            // Start with white background, accumulate in registers
            var r = 255.0;
            var g = 255.0;
            var b = 255.0;

            if params.tile_culling == 1u {
                // --- Tile-culled path ---
                // This workgroup's tile coordinates
                let tile_x = wid.x;
                let tile_y = wid.y;
                let num_tiles_x = (w + WG_X - 1u) / WG_X;
                let num_tiles_y = (h + WG_Y - 1u) / WG_Y;
                let num_tiles = num_tiles_x * num_tiles_y;
                let tile_id = tile_y * num_tiles_x + tile_x;

                // Read tile polygon count (non-atomic read — binning pass is complete)
                let tile_global = chain_id * num_tiles + tile_id;
                let tile_poly_count = min(tile_counts_buf[tile_global], TILE_MAX_POLYS);

                // Tile data offset for this offspring's tile
                let tile_data_base = chain_id * num_tiles * TILE_MAX_POLYS + tile_id * TILE_MAX_POLYS;

                // Process tile polygons in shared-memory tiles (same tiling as brute force)
                let sm_tile_count = (tile_poly_count + TILE_CAP - 1u) / TILE_CAP;

                for (var sm_tile = 0u; sm_tile < sm_tile_count; sm_tile++) {
                    let sm_tile_base = sm_tile * TILE_CAP;
                    let sm_tile_end = min(TILE_CAP, tile_poly_count - sm_tile_base);

                    // Cooperative load: each thread loads LOADS_PER_THREAD polygons via tile index
                    for (var load_pass = 0u; load_pass < LOADS_PER_THREAD; load_pass++) {
                        let slot = local_idx + load_pass * THREAD_COUNT;
                        if slot < sm_tile_end {
                            let poly_idx = tile_data[tile_data_base + sm_tile_base + slot];
                            shared_polys[slot] = working_states[chain_id].polygons[poly_idx];
                        }
                    }
                    workgroupBarrier();

                    // Each thread tests its pixel against polygons in this shared memory tile
                    for (var i = 0u; i < sm_tile_end; i++) {
                        rasterize_blend(shared_polys[i], fx, fy, &r, &g, &b);
                    }
                    workgroupBarrier();
                }
            } else {
                // --- Brute-force path (original) ---
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
                        rasterize_blend(shared_polys[i], fx, fy, &r, &g, &b);
                    }
                    workgroupBarrier();
                }
            }

            // Clamp rendered values
            let ri = clamp(r, 0.0, 255.0);
            let gi = clamp(g, 0.0, 255.0);
            let bi = clamp(b, 0.0, 255.0);

            // L1 error: Manhattan distance in RGB space
            let dr = ri - refr;
            let dg = gi - refg;
            let db = bi - refb;
            pixel_error = u32(abs(dr) + abs(dg) + abs(db));
        }
    }

    // Subgroup-accelerated reduction: subgroupAdd within each warp/wave,
    // then thread 0 sums across subgroups via shared memory.
    let sg_sum = subgroupAdd(pixel_error);
    let sg_idx = local_idx / sg_size;

    if sg_inv_id == 0u {
        shared_errors[sg_idx] = sg_sum;
    }

    // Also reduce old errors for incremental eval
    if incremental {
        let sg_sum_old = subgroupAdd(pixel_error_old);
        if sg_inv_id == 0u {
            shared_errors_old[sg_idx] = sg_sum_old;
        }
    }

    workgroupBarrier();

    // Thread 0 sums across subgroups and atomicAdds to chain's accumulator (stride-2)
    if local_idx == 0u {
        let num_subgroups = (THREAD_COUNT + sg_size - 1u) / sg_size;
        var total = 0u;
        for (var i = 0u; i < num_subgroups; i++) {
            total += shared_errors[i];
        }
        atomicAdd(&error_accumulators[chain_id * 2u], total);

        if incremental {
            var total_old = 0u;
            for (var i = 0u; i < num_subgroups; i++) {
                total_old += shared_errors_old[i];
            }
            atomicAdd(&error_accumulators[chain_id * 2u + 1u], total_old);
        }
    }
}
