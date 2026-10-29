// Update chain framebuffers after select acceptance: re-rasterize dirty bbox region.
// Dispatch: (W/TILE_W, H/TILE_H, active_chains) — same parallelism as init_framebuffers.
// Chains that didn't accept → all workgroups early-exit (reading 1 flag from global memory).
// Workgroups on tiles that are not consumed (tiles.wgsl) → early-exit.
// Remaining workgroups: parallel rasterize into chain_framebuffers.
// No error computation — total_error was already set by select.
// Polygons come from the winning offspring's tile list built by bin_polygons this
// iteration (select records the offspring in chain_update_flags[4c+3]). The list
// holds, in drawing order, every polygon whose AABB can reach a pixel centre in
// this tile, so blending the same polygons in the same order is bit-identical to
// scanning all polygons. An overflowed list falls back to the full scan.
//
// common.wgsl, tiles.wgsl and the generated constants (TILE_W, TILE_H,
// TILE_MAX_POLYS, WORK_COUNT) are appended by programs.rs.

const THREAD_COUNT: u32 = TILE_W * TILE_H;

@group(0) @binding(0) var<storage, read>       chain_states:       array<DrawingState>;
@group(0) @binding(1) var<storage, read_write> chain_framebuffers: array<u32>;
@group(0) @binding(2) var<storage, read>       chain_update_flags: array<u32>;
@group(0) @binding(3) var<storage, read>       tile_data:          array<u32>;
@group(0) @binding(4) var<storage, read>       tile_counts:        array<u32>;
// The raster worklist's entry count is word WORK_COUNT. This is the last stage
// of an iteration, so it empties the list for the next one.
@group(0) @binding(5) var<storage, read_write> raster_work:        array<u32>;
@group(1) @binding(0) var<uniform>             params:             Params;

// Shared memory: acceptance flag + dirty bbox (read by thread 0, broadcast to workgroup)
var<workgroup> shared_accepted: u32;
var<workgroup> shared_dirty_min_x: u32;
var<workgroup> shared_dirty_min_y: u32;
var<workgroup> shared_dirty_max_x: u32;
var<workgroup> shared_dirty_max_y: u32;
var<workgroup> shared_offspring_id: u32;

// Shared memory for polygon tiling (same approach as init_framebuffers)
const TILE_CAP: u32 = THREAD_COUNT * 3u;
const LOADS_PER_THREAD: u32 = 3u;
var<workgroup> shared_polys: array<Polygon, TILE_CAP>;

// Unchanged-pixel skip, the same argument as rasterize_error's shortcut. select
// copies the winner's changed indices to the chain's rng_state.y/.z and keeps
// the parent versions in polygons[CHANGED_STASH_BASE + k] (rng_state.w bit k
// marks a present one). A pixel that neither version of any changed index can
// cover has the same covering polygon sequence before and after, so its
// framebuffer value is already the winner's: it is neither blended nor written.
// Unknown indices, or a tile list that rasterize_error could not shortcut, keep
// the full loop.
const CHANGED_INDICES: u32 = 4u;
const CHANGED_SLOTS: u32 = 2u * CHANGED_INDICES;
const CHANGED_STASH_BASE: u32 = 1000u - CHANGED_INDICES;
// Must match rasterize_error.wgsl.
const COVER_MARGIN: f32 = 1e-5;

struct ChangedPolygon {
    pv0: vec2<f32>,
    pv1: vec2<f32>,
    pv2: vec2<f32>,
    bb_min: vec2<f32>,
    bb_max: vec2<f32>,
}

var<workgroup> shared_changed: array<ChangedPolygon, CHANGED_SLOTS>;
var<workgroup> shared_touched: atomic<u32>;
var<workgroup> shared_any_touched: u32;

fn decode_changed(poly: Polygon) -> ChangedPolygon {
    let pv0 = unpack_vertex(poly.data.y);
    let pv1 = unpack_vertex(poly.data.z);
    let pv2 = unpack_vertex(poly.data.w);
    return ChangedPolygon(pv0, pv1, pv2, min(pv0, min(pv1, pv2)), max(pv0, max(pv1, pv2)));
}

// Empty slots have an inverted box outside the unit square.
fn no_changed_polygon() -> ChangedPolygon {
    return ChangedPolygon(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(2.0), vec2<f32>(-1.0));
}

// Superset of the blend loop's coverage test: false only when the pixel is
// outside the polygon by more than COVER_MARGIN.
fn may_cover(poly: ChangedPolygon, fx: f32, fy: f32) -> bool {
    if fx < poly.bb_min.x - COVER_MARGIN || fx > poly.bb_max.x + COVER_MARGIN ||
       fy < poly.bb_min.y - COVER_MARGIN || fy > poly.bb_max.y + COVER_MARGIN {
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

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    if all(wid == vec3<u32>(0u)) && local_idx == 0u {
        raster_work[WORK_COUNT] = 0u;
    }

    let chain_id = wid.z;
    let chain_count = arrayLength(&chain_states);
    if chain_id >= chain_count {
        return;
    }

    // Thread 0: read acceptance flag and dirty bbox, broadcast to workgroup
    if local_idx == 0u {
        atomicStore(&shared_touched, 0u);
        shared_accepted = chain_update_flags[chain_id * 4u];
        if shared_accepted == 1u {
            let dirty_min = unpack_dirty_corner(chain_update_flags[chain_id * 4u + 1u]);
            let dirty_max = unpack_dirty_corner(chain_update_flags[chain_id * 4u + 2u]);
            shared_dirty_min_x = dirty_min.x;
            shared_dirty_min_y = dirty_min.y;
            shared_dirty_max_x = dirty_max.x;
            shared_dirty_max_y = dirty_max.y;
            shared_offspring_id = chain_update_flags[chain_id * 4u + 3u];
        }
    }
    workgroupBarrier();

    // Early-exit: chain didn't accept — nothing to update
    if shared_accepted != 1u {
        return;
    }

    let w = params.image_width;
    let h = params.image_height;

    let dmin_x = shared_dirty_min_x;
    let dmin_y = shared_dirty_min_y;
    let dmax_x = min(shared_dirty_max_x, w);
    let dmax_y = min(shared_dirty_max_y, h);

    // Consumed-tile skip (uniform across workgroup → all threads exit together)
    if !tile_consumed(wid.xy, w, h, vec2<u32>(dmin_x, dmin_y), vec2<u32>(dmax_x, dmax_y)) {
        return;
    }

    let local_x = local_idx % TILE_W;
    let local_y = local_idx / TILE_W;
    let px = wid.x * TILE_W + local_x;
    let py = wid.y * TILE_H + local_y;

    let fx = (f32(px) + 0.5) / f32(w);
    let fy = (f32(py) + 0.5) / f32(h);

    // Workgroup-uniform polygon source: the winner's tile list, or every polygon
    // when the list overflowed TILE_MAX_POLYS.
    let grid = tile_grid(w, h);
    let num_tiles_x = grid.x;
    let num_tiles = num_tiles_x * grid.y;
    let bin_tile = shared_offspring_id * num_tiles + wid.y * num_tiles_x + wid.x;
    let listed_count = tile_counts[bin_tile];
    let use_list = listed_count <= TILE_MAX_POLYS;

    // Changed-index guards mirror rasterize_error: known indices, and neither the
    // parent's nor the winner's tile list truncated.
    let first = chain_states[chain_id].rng_state.y;
    var second = 0u;
    if first != 0u {
        second = chain_states[chain_id].rng_state.z;
    }
    let listed = select(2u, CHANGED_INDICES, second != 0u);
    let shortcut = first != 0u && listed_count + listed <= TILE_MAX_POLYS;
    let shortcut_slots = select(0u, select(4u, CHANGED_SLOTS, second != 0u), shortcut);

    // Slot k holds index half k/2, from the parent stash (even k) or the winner
    // now in chain_states (odd k).
    if local_idx < CHANGED_SLOTS {
        var decoded = no_changed_polygon();
        let half = local_idx >> 1u;
        let word = select(first, second, half >= 2u);
        let encoded = (word >> (16u * (half & 1u))) & 0xFFFFu;
        if shortcut && encoded != 0u {
            if (local_idx & 1u) == 0u {
                if (chain_states[chain_id].rng_state.w & (1u << half)) != 0u {
                    decoded = decode_changed(chain_states[chain_id].polygons[CHANGED_STASH_BASE + half]);
                }
            } else if encoded - 1u < chain_states[chain_id].polygon_count {
                decoded = decode_changed(chain_states[chain_id].polygons[encoded - 1u]);
            }
        }
        shared_changed[local_idx] = decoded;
    }
    workgroupBarrier();

    // Per-pixel dirty bbox clip: all threads participate in tiling loop,
    // but only changed pixels in the dirty region rasterize and write.
    let in_dirty = px < w && py < h && px >= dmin_x && px < dmax_x && py >= dmin_y && py < dmax_y;
    let touched = in_dirty && (shortcut_slots == 0u || may_change(fx, fy, shortcut_slots));
    if touched {
        atomicOr(&shared_touched, 1u);
    }
    workgroupBarrier();
    if local_idx == 0u {
        shared_any_touched = atomicLoad(&shared_touched);
    }
    // Tiles with no changed pixel (all tiles for an exact-copy winner) stop here.
    if workgroupUniformLoad(&shared_any_touched) == 0u {
        return;
    }

    var r = 255.0;
    var g = 255.0;
    var b = 255.0;
    let list_base = bin_tile * TILE_MAX_POLYS;
    let poly_count = select(chain_states[chain_id].polygon_count, listed_count, use_list);
    let tile_count = (poly_count + TILE_CAP - 1u) / TILE_CAP;

    for (var tile = 0u; tile < tile_count; tile++) {
        let tile_base = tile * TILE_CAP;
        let tile_end = min(TILE_CAP, poly_count - tile_base);

        // ALL threads cooperate on shared memory loads
        for (var load_pass = 0u; load_pass < LOADS_PER_THREAD; load_pass++) {
            let slot = local_idx + load_pass * THREAD_COUNT;
            if slot < tile_end {
                var poly_idx = tile_base + slot;
                if use_list {
                    poly_idx = tile_data[list_base + poly_idx];
                }
                shared_polys[slot] = chain_states[chain_id].polygons[poly_idx];
            }
        }
        workgroupBarrier();

        // Only changed dirty-region pixels rasterize
        if touched {
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

    // Write updated pixel to chain framebuffer (quantized via pack_fb_pixel)
    if touched {
        let fb_idx = chain_id * w * h + py * w + px;
        chain_framebuffers[fb_idx] = pack_fb_pixel(r, g, b);
    }
}
