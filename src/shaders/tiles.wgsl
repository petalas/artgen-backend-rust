// Shared by the tile-group stages (rasterize_error, bin_polygons,
// raster_worklist, init_framebuffers, update_framebuffers). programs.rs appends
// it after common.wgsl and changes.wgsl. TILE_W x TILE_H is the installed raster
// tile, from the generated constants.

// Tiles across and down a w x h image, counting partial edge tiles.
fn tile_grid(w: u32, h: u32) -> vec2<u32> {
    return vec2<u32>((w + TILE_W - 1u) / TILE_W, (h + TILE_H - 1u) / TILE_H);
}

// --- Consumed tiles ---
// A tile is consumed for an offspring (or a chain's update) when its pixels,
// clipped to the image, meet the dirty rectangle [dirty_min, dirty_max). The
// binner writes tile lists only for consumed tiles; other tiles keep stale
// words. rasterize_error (through the raster worklist) and update_framebuffers
// read only consumed tiles, so every reader must use these functions.

// Whether tile `coord` of `size` pixels along an axis of `extent` pixels meets
// the dirty span [lo, hi) on that axis.
fn tile_span_consumed(coord: u32, size: u32, extent: u32, lo: u32, hi: u32) -> bool {
    let tile_min = coord * size;
    let tile_max = min(tile_min + size, extent);
    return tile_max > lo && tile_min < hi;
}

fn tile_column_consumed(col: u32, w: u32, dirty_min: vec2<u32>, dirty_max: vec2<u32>) -> bool {
    return tile_span_consumed(col, TILE_W, w, dirty_min.x, dirty_max.x);
}

fn tile_row_consumed(row: u32, h: u32, dirty_min: vec2<u32>, dirty_max: vec2<u32>) -> bool {
    return tile_span_consumed(row, TILE_H, h, dirty_min.y, dirty_max.y);
}

fn tile_consumed(tile: vec2<u32>, w: u32, h: u32, dirty_min: vec2<u32>, dirty_max: vec2<u32>) -> bool {
    return tile_column_consumed(tile.x, w, dirty_min, dirty_max) &&
           tile_row_consumed(tile.y, h, dirty_min, dirty_max);
}

// Framebuffer word: RGB truncated to bytes, opaque alpha.
fn pack_fb_pixel(r: f32, g: f32, b: f32) -> u32 {
    let ri = u32(clamp(r, 0.0, 255.0));
    let gi = u32(clamp(g, 0.0, 255.0));
    let bi = u32(clamp(b, 0.0, 255.0));
    return ri | (gi << 8u) | (bi << 16u) | (255u << 24u);
}
