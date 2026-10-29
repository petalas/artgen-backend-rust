//! Fit a drawing to the raster's tile lists before it becomes a parent.
//!
//! `bin_polygons` lists, per raster tile and in ascending polygon order, every
//! polygon whose pixel bounding box meets the tile, and keeps at most
//! `TILE_MAX_POLYS` of them. `rasterize_error` evaluates only that prefix, and
//! `select` rejects an offspring whose evaluated tile lost polygons. A parent
//! uploaded from the host (application start, reinitialization, seeded
//! benchmark preparation) is not an offspring, so a drawing whose tile list
//! overflows would be scored without its late polygons while its framebuffer
//! and the delivered drawing keep them.
//!
//! `fit_tile_lists` removes every polygon that falls past the first
//! `TILE_MAX_POLYS` entries of some tile's list in the uploaded order. Every
//! polygon it keeps was rendered by the raster in each tile it meets, and every
//! remaining tile list fits, so the raster scores the trimmed drawing exactly.

use crate::settings::{MAX_POLYGONS_PER_IMAGE, TILE_MAX_POLYS};

use super::buffers::{GpuDrawingState, GpuPolygon};

/// Slack added to each side of a polygon's pixel bounds, in 1/65535 px (about
/// 1e-3 px). Bounds here are exact: a vertex word k lies at k * extent / 65535
/// px. The binner decodes and scales in f32, which can differ by a few ulps
/// (well under 1e-3 px for images up to 1024 px). With the slack, each list
/// built here contains the binner's list for the same tile, so a polygon's
/// position here is at least its position there.
const BOUND_SLACK: u64 = 66;

/// Remove the polygons of `state` that some tile list of a `tile`-pixel raster
/// over `image` holds past position `TILE_MAX_POLYS`, keeping the order of the
/// rest. Returns the number removed. Freed polygon slots are zeroed, as
/// `drawing_to_gpu` leaves unused slots.
///
/// # Panics
/// Panics if an image or tile dimension is zero, or the polygon count exceeds
/// the state's capacity.
pub fn fit_tile_lists(state: &mut GpuDrawingState, image: [u32; 2], tile: [u32; 2]) -> u32 {
    assert!(
        image.iter().chain(&tile).all(|&n| n > 0),
        "image {image:?} and tile {tile:?} must be nonempty"
    );
    let count = state.polygon_count as usize;
    assert!(count <= MAX_POLYGONS_PER_IMAGE, "{count} polygons exceed the state");
    let grid = [image[0].div_ceil(tile[0]), image[1].div_ceil(tile[1])];
    // Bounded: grid tiles (at most image pixels) and one flag per polygon.
    let mut occupancy = vec![0_u32; grid[0] as usize * grid[1] as usize];
    let mut keep = vec![true; count];
    for (polygon, kept) in state.polygons[..count].iter().zip(&mut keep) {
        let (lo, hi) = tile_span(polygon, image, tile, grid);
        for ty in lo[1]..hi[1] {
            for tx in lo[0]..hi[0] {
                let entries = &mut occupancy[(ty * grid[0] + tx) as usize];
                *entries += 1;
                *kept &= *entries <= TILE_MAX_POLYS;
            }
        }
    }
    let mut kept_count = 0;
    for index in 0..count {
        if keep[index] {
            state.polygons[kept_count] = state.polygons[index];
            kept_count += 1;
        }
    }
    state.polygons[kept_count..count].fill(GpuPolygon { data: [0; 4] });
    state.polygon_count = u32::try_from(kept_count).expect("kept count fits the state");
    u32::try_from(count - kept_count).expect("removed count fits the state")
}

/// The half-open tile ranges `[lo, hi)` per axis that `bin_polygons`'s
/// `polygon_tile_bounds` gives the polygon, widened by `BOUND_SLACK`.
fn tile_span(
    polygon: &GpuPolygon,
    image: [u32; 2],
    tile: [u32; 2],
    grid: [u32; 2],
) -> ([u32; 2], [u32; 2]) {
    let vertices = [polygon.data[1], polygon.data[2], polygon.data[3]];
    let mut lo = [0; 2];
    let mut hi = [0; 2];
    for axis in 0..2 {
        let coordinate = |word: u32| u64::from((word >> (16 * axis)) & 0xFFFF);
        let min = vertices.map(coordinate).into_iter().min().expect("three vertices");
        let max = vertices.map(coordinate).into_iter().max().expect("three vertices");
        // Pixel positions times 65535; both bounds lie in [0, extent].
        let extent = u64::from(image[axis]);
        let pixel_min = (min * extent).saturating_sub(BOUND_SLACK) / 65535;
        let pixel_max = (max * extent + BOUND_SLACK).div_ceil(65535).min(extent);
        let bound = |pixels: u64| u32::try_from(pixels).expect("pixel bound fits the image");
        lo[axis] = bound(pixel_min) / tile[axis];
        hi[axis] = bound(pixel_max).div_ceil(tile[axis]).min(grid[axis]);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;

    use super::{fit_tile_lists, GpuDrawingState, GpuPolygon};
    use crate::benchmark::reference;

    const IMAGE: [u32; 2] = [64, 32];
    const TILE: [u32; 2] = [16, 16];

    fn vertex(x: u16, y: u16) -> u32 {
        u32::from(x) | (u32::from(y) << 16)
    }

    /// A vertex word at `percent` of the axis.
    fn pct(percent: u32) -> u16 {
        u16::try_from(percent * 65535 / 100).unwrap()
    }

    /// A triangle spanning vertex words x in [x0, x1] and the full height.
    fn triangle(x0: u16, x1: u16, color: u32) -> GpuPolygon {
        GpuPolygon {
            data: [color, vertex(x0, 0), vertex(x1, 0), vertex(x0, 65535)],
        }
    }

    fn state(polygons: &[GpuPolygon]) -> GpuDrawingState {
        let mut state = GpuDrawingState::zeroed();
        state.polygons[..polygons.len()].copy_from_slice(polygons);
        state.polygon_count = u32::try_from(polygons.len()).unwrap();
        state
    }

    fn colors(state: &GpuDrawingState) -> Vec<u32> {
        state.polygons[..state.polygon_count as usize].iter().map(|p| p.data[0]).collect()
    }

    fn overflowed_tiles(state: &GpuDrawingState) -> u32 {
        let polygons: Vec<[u32; 4]> =
            state.polygons[..state.polygon_count as usize].iter().map(|p| p.data).collect();
        reference::complexity(&polygons, IMAGE[0], IMAGE[1], TILE).unwrap().overflowed_tiles
    }

    #[test]
    fn drawings_whose_lists_fit_are_unchanged() {
        // 256 polygons in the first tile column fill its lists exactly.
        let polygons: Vec<_> = (0..256).map(|i| triangle(0, pct(20), i)).collect();
        let original = state(&polygons);
        let mut fitted = original;
        assert_eq!(fit_tile_lists(&mut fitted, IMAGE, TILE), 0);
        assert_eq!(bytemuck::bytes_of(&fitted), bytemuck::bytes_of(&original));
    }

    #[test]
    fn polygons_past_a_full_list_are_removed_and_the_rest_keep_their_order() {
        // Indices 0..300 fill tile column 0. Index 300 meets only columns 2..3
        // and stays; index 301 also meets column 0, past its 256th entry, and
        // goes even though columns 1..3 have room.
        let mut polygons: Vec<_> = (0..300).map(|i| triangle(0, pct(20), i)).collect();
        polygons.push(triangle(pct(60), pct(90), 300));
        polygons.push(triangle(pct(10), pct(90), 301));
        let mut fitted = state(&polygons);
        assert_eq!(overflowed_tiles(&fitted), 2);
        assert_eq!(fit_tile_lists(&mut fitted, IMAGE, TILE), 45);
        let expected: Vec<u32> = (0..256).chain([300]).collect();
        assert_eq!(colors(&fitted), expected);
        assert!(fitted.polygons[257..302].iter().all(|p| p.data == [0; 4]));
        // The independent oracle agrees no list overflows, and a second fit
        // removes nothing.
        assert_eq!(overflowed_tiles(&fitted), 0);
        assert_eq!(fit_tile_lists(&mut fitted, IMAGE, TILE), 0);
    }

    #[test]
    fn a_polygon_is_removed_when_any_of_its_lists_overflows() {
        // Column 3 is full from indices 0..256. Index 256 spans columns 0..3:
        // it is entry 1 of columns 0..2 but entry 257 of column 3.
        let mut polygons: Vec<_> = (0..256).map(|i| triangle(pct(80), pct(100), i)).collect();
        polygons.push(triangle(0, pct(100), 256));
        polygons.push(triangle(0, pct(70), 257));
        let mut fitted = state(&polygons);
        assert_eq!(fit_tile_lists(&mut fitted, IMAGE, TILE), 1);
        let expected: Vec<u32> = (0..256).chain([257]).collect();
        assert_eq!(colors(&fitted), expected);
    }

    #[test]
    fn a_left_edge_within_the_slack_of_a_tile_boundary_counts_in_both_tiles() {
        // Vertex x = 16384/65535 is 16.00024 px: the binner starts the polygon
        // in column 1, the slack also counts it in column 0. Column 0 is full,
        // so the fit conservatively removes a polygon whose exact lists fit.
        let mut polygons: Vec<_> = (0..256).map(|i| triangle(0, pct(20), i)).collect();
        let left = 16384;
        assert!((16 * 65535..16 * 65535 + 66).contains(&(u64::from(left) * 64)));
        polygons.push(triangle(left, pct(50), 256));
        let mut fitted = state(&polygons);
        assert_eq!(overflowed_tiles(&fitted), 0);
        assert_eq!(fit_tile_lists(&mut fitted, IMAGE, TILE), 1);
        assert_eq!(fitted.polygon_count, 256);
    }
}
