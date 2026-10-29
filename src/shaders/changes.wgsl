// Change records, shared by every bundled stage. gpu_evolver::programs appends
// this file after common.wgsl; gpu_evolver::changes mirrors the encoding and
// CPU tests compare its constants and the ChainChange layout. CHANGED_INDICES
// and TILE_MAX_POLYS are generated constants.
//
// A change record says which polygons an evaluated drawing changed relative to
// the drawing whose pixels are cached. Performance shortcuts skip the pixels
// that no changed polygon can cover; an unknown record turns them off.
//
// The offspring record (written by mutate; read by bin_polygons,
// raster_worklist, rasterize_error and select through offspring_changes.wgsl)
// lives in otherwise unused words of the working state, which historical
// shader variants read and write in place:
//   fitness_bits, stagnation_counter: the dirty rectangle's min and max
//     corners, x | y << 16 pixels, max exclusive;
//   rng_state.y, rng_state.z: the changed-index words.
// The chain record (ChainChange, one per chain in chain_changes) is written by
// select when it accepts a winner and read by update_framebuffers in the same
// iteration. Parent states carry no change record.

// --- Dirty rectangle ---

// vec4<u32>(min_x, min_y, max_x, max_y) in pixels, packed as two words.
fn pack_dirty_box(bb: vec4<u32>) -> vec2<u32> {
    return vec2<u32>(
        (bb.x & 0xFFFFu) | ((bb.y & 0xFFFFu) << 16u),
        (bb.z & 0xFFFFu) | ((bb.w & 0xFFFFu) << 16u),
    );
}

// A dirty-rectangle corner packed as x | y << 16, in pixels.
fn unpack_dirty_corner(word: u32) -> vec2<u32> {
    return vec2<u32>(word & 0xFFFFu, word >> 16u);
}

// --- Changed-index words ---
// Two words, each (first + 1) | (second + 1) << 16, list up to CHANGED_INDICES
// polygon indices (halves 0 and 1 in .x, 2 and 3 in .y). A zero half lists
// nothing. A zero first word means unknown: every pixel blends. The second word
// is meaningful only when the first is non-zero. NO_CHANGES (both halves name
// index 0xFFFE, beyond any drawing) means no index changed. Every changed index
// is listed, so a pixel covered by none of the listed indices' versions before
// and after the change keeps its cached colour.
const NO_CHANGED_INDEX: u32 = 0xFFFFu;
const NO_CHANGES: u32 = 0xFFFFFFFFu;
// One parent and one changed version per index.
const CHANGED_SLOTS: u32 = 2u * CHANGED_INDICES;

// NO_CHANGED_INDEX as `second` wraps to an empty half. `first` must be an index.
fn changed_indices(first: u32, second: u32) -> u32 {
    return (first + 1u) | ((second + 1u) << 16u);
}

// Index half k of the words: its index + 1, or zero when empty.
fn changed_half(words: vec2<u32>, k: u32) -> u32 {
    let word = select(words.x, words.y, k >= 2u);
    return (word >> (16u * (k & 1u))) & 0xFFFFu;
}

// The words a stage may use for a tile whose list holds `raw_count` polygons
// (uncapped), or unknown. Neither the parent's nor the changed drawing's list may
// be truncated: each changed index moves at most one polygon in or out of a
// tile, so a list needs room for one more polygon per listed index.
fn changes_shortcut(words: vec2<u32>, raw_count: u32) -> vec2<u32> {
    let listed = select(2u, CHANGED_INDICES, words.y != 0u);
    if words.x == 0u || raw_count + listed > TILE_MAX_POLYS {
        return vec2<u32>(0u, 0u);
    }
    return words;
}

// Slots to test per pixel for words from changes_shortcut; zero disables the
// shortcut.
fn changed_slot_count(words: vec2<u32>) -> u32 {
    return select(0u, select(4u, CHANGED_SLOTS, words.y != 0u), words.x != 0u);
}

// --- Coverage of changed polygons ---

// Conservative coverage margin in normalized units (a pixel is ~2.6e-3 at 384
// px). It absorbs any rounding difference from the blend loops' own coverage
// test.
const COVER_MARGIN: f32 = 1e-5;

// Vertices and bounding box of a changed polygon version.
struct ChangedPolygon {
    pv0: vec2<f32>,
    pv1: vec2<f32>,
    pv2: vec2<f32>,
    bb_min: vec2<f32>,
    bb_max: vec2<f32>,
}

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

// Superset of the blend loops' coverage test: false only when the pixel is
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

// Whether the polygon's box, widened by COVER_MARGIN, meets the normalized
// rectangle [lo, hi]: a superset of may_cover for every pixel centre inside it.
fn changed_may_meet_rect(poly: Polygon, lo: vec2<f32>, hi: vec2<f32>) -> bool {
    let changed = decode_changed(poly);
    return hi.x >= changed.bb_min.x - COVER_MARGIN && lo.x <= changed.bb_max.x + COVER_MARGIN &&
           hi.y >= changed.bb_min.y - COVER_MARGIN && lo.y <= changed.bb_max.y + COVER_MARGIN;
}

// --- Chain record ---

// gpu_evolver::changes::GpuChainChange; one per chain in chain_changes. select
// writes it when it accepts a winner, before the winner's polygons replace the
// parent's; update_framebuffers reads it in the same iteration. All zeros is
// unknown.
struct ChainChange {
    changed: vec2<u32>,   // the winner's changed-index words
    parent_mask: u32,     // bit k: parents[k] holds the parent version of index half k
    _pad: u32,
    parents: array<Polygon, CHANGED_INDICES>,
}
