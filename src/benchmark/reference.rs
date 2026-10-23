//! Independent scalar oracle for the packed compute-renderer contract.
//!
//! This deliberately does not call the drawing renderer or GPU helpers. Vertices
//! use f32 division by 65535; pixel centers, inclusive edge tests, and each blend
//! use separate f32 operations in shader order. Channels are floored only once.
//! No top-left rule or degenerate culling is applied. GPU compiler contraction or
//! UNORM conversion can disagree with these operations; callers must report the
//! actual pixel and integer-error differences rather than assume a tolerance.
//!
//! Both public functions allocate bounded diagnostic buffers. Run them outside
//! the engine's measured interval. Rendering visits every layer, even when the
//! GPU's 256-entry tile bins would overflow.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

const MAX_DIMENSION: u32 = 1024;
const MAX_TRIANGLES: usize = 1000;
const TILE_CAPACITY: u32 = 256;

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceEvaluation {
    /// R in bits 0..8, G in 8..16, B in 16..24, and alpha 255 in 24..32.
    pub pixels: Vec<u32>,
    /// Exact RGB byte L1 sum; reference alpha is ignored.
    pub raw_error: u64,
    /// Unpenalized similarity as a percentage, computed in f64.
    pub raw_similarity: f64,
    /// Percentage with the production per-vertex penalty, computed in f32.
    pub fitness: f32,
}

/// Normalized areas. Empty distributions contain zeros. The median averages
/// the two middle values for even counts; p95 uses the nearest-rank definition.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AreaDistribution {
    pub min: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

/// Counts include empty tiles, transparent triangles, and degenerate triangles.
/// Both p50 and p95 use nearest rank, so occupancy percentiles remain integers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileOverlapDistribution {
    pub min: u32,
    pub p50: u32,
    pub p95: u32,
    pub max: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Complexity {
    pub triangles: u32,
    pub vertices: u32,
    pub triangle_area: AreaDistribution,
    pub bbox_area: AreaDistribution,
    pub tile_count: u32,
    pub tile_overlaps: TileOverlapDistribution,
    /// Tiles with more than 256 entries, counted before GPU truncation.
    pub overflowed_tiles: u32,
}

struct Triangle {
    vertices: [[f32; 2]; 3],
    min: [f32; 2],
    max: [f32; 2],
    source: [f32; 3],
    alpha: f32,
}

impl Triangle {
    fn decode(packed: &[u32; 4]) -> Self {
        let vertices = [vertex(packed[1]), vertex(packed[2]), vertex(packed[3])];
        let [a, b, c] = vertices;
        let min = [a[0].min(b[0].min(c[0])), a[1].min(b[1].min(c[1]))];
        let max = [a[0].max(b[0].max(c[0])), a[1].max(b[1].max(c[1]))];
        // Keep unpack4x8unorm's division and the shader's *255 round trip.
        let color = packed[0];
        let source = [
            (color & 255) as f32 / 255.0 * 255.0,
            ((color >> 8) & 255) as f32 / 255.0 * 255.0,
            ((color >> 16) & 255) as f32 / 255.0 * 255.0,
        ];
        Self {
            vertices,
            min,
            max,
            source,
            alpha: (color >> 24) as f32 / 255.0,
        }
    }

    fn covers(&self, p: [f32; 2]) -> bool {
        if p[0] < self.min[0] || p[0] > self.max[0] || p[1] < self.min[1] || p[1] > self.max[1] {
            return false;
        }
        let [a, b, c] = self.vertices;
        let e0 = edge(a, b, p);
        let e1 = edge(b, c, p);
        let e2 = edge(c, a, p);
        (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
    }

    fn areas(&self) -> (f64, f64) {
        // Metrics describe decoded shader geometry, with f64 area arithmetic.
        let [a, b, c] = self.vertices.map(|p| p.map(f64::from));
        let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        let dx = f64::from(self.max[0]) - f64::from(self.min[0]);
        let dy = f64::from(self.max[1]) - f64::from(self.min[1]);
        (twice_area.abs() * 0.5, dx * dy)
    }
}

/// Render and score packed triangles against exactly width * height RGBA bytes.
/// Rejects empty/oversized dimensions and drawings exceeding 1000 triangles.
/// # Errors
/// Rejects dimensions, reference lengths or triangle counts outside the bounds.
pub fn evaluate(
    polygons: &[[u32; 4]],
    reference: &[u8],
    width: u32,
    height: u32,
) -> Result<ReferenceEvaluation> {
    let pixel_count = validate(polygons, width, height)?;
    ensure!(
        reference.len() == pixel_count * 4,
        "reference must contain exactly width * height RGBA bytes"
    );
    let triangles: Vec<_> = polygons.iter().map(Triangle::decode).collect();
    let mut pixels = vec![0; pixel_count];
    let mut raw_error = 0_u64;
    // At most 1024*1024 pixels, each visiting at most 1000 triangles.
    for y in 0..height {
        for x in 0..width {
            let p = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];
            let rgb = render_pixel(&triangles, p);
            let index = (y * width + x) as usize;
            pixels[index] = u32::from_le_bytes([rgb[0], rgb[1], rgb[2], 255]);
            for channel in 0..3 {
                raw_error += u64::from(rgb[channel].abs_diff(reference[index * 4 + channel]));
            }
        }
    }
    let max_error = 765_u64 * pixel_count as u64;
    debug_assert!(raw_error <= max_error);
    let raw_similarity = 100.0 * (1.0 - raw_error as f64 / max_error as f64);
    // Preserve compute_fitness's operation order and production parameters.
    let mut fitness = 100.0
        * (1.0 - raw_error as f32 / (crate::settings::MAX_ERROR_PER_PIXEL * pixel_count as f32));
    fitness -= fitness * crate::settings::PER_POINT_MULTIPLIER * (polygons.len() * 3) as f32;
    Ok(ReferenceEvaluation {
        pixels,
        raw_error,
        raw_similarity,
        fitness,
    })
}

/// Describe normalized geometry and the exact `bin_polygons.wgsl` AABB scatter.
/// Workgroup axes must be 1..=1024. This diagnostic accepts arbitrary bounded
/// tile shapes; the trial owner separately validates GPU-supported workgroups.
/// # Errors
/// Rejects invalid dimensions, tile shapes and excessive triangle counts.
pub fn complexity(
    polygons: &[[u32; 4]],
    width: u32,
    height: u32,
    workgroup: [u32; 2],
) -> Result<Complexity> {
    validate(polygons, width, height)?;
    ensure!(
        workgroup
            .into_iter()
            .all(|n| (1..=MAX_DIMENSION).contains(&n)),
        "workgroup axes must be in 1..=1024"
    );
    let tiles_x = width.div_ceil(workgroup[0]);
    let tiles_y = height.div_ceil(workgroup[1]);
    let tile_count = tiles_x * tiles_y;
    let mut overlaps = vec![0_u32; tile_count as usize];
    let mut triangle_areas = Vec::with_capacity(polygons.len());
    let mut bbox_areas = Vec::with_capacity(polygons.len());
    // Allocate all scratch before visiting at most 1000 * 1024*1024 tile slots.
    for polygon in polygons {
        let triangle = Triangle::decode(polygon);
        let (area, bbox_area) = triangle.areas();
        triangle_areas.push(area);
        bbox_areas.push(bbox_area);
        let (lo, hi) = bin_bounds(&triangle, [width, height], workgroup);
        for ty in lo[1]..hi[1] {
            for tx in lo[0]..hi[0] {
                overlaps[(ty * tiles_x + tx) as usize] += 1;
            }
        }
    }
    let overflowed_tiles = overlaps.iter().filter(|&&n| n > TILE_CAPACITY).count() as u32;
    overlaps.sort_unstable();
    debug_assert!(overlaps.last().is_some_and(|&n| n <= polygons.len() as u32));
    Ok(Complexity {
        triangles: polygons.len() as u32,
        vertices: (polygons.len() * 3) as u32,
        triangle_area: area_distribution(&mut triangle_areas),
        bbox_area: area_distribution(&mut bbox_areas),
        tile_count,
        tile_overlaps: TileOverlapDistribution {
            min: overlaps[0],
            p50: overlaps[rank_index(overlaps.len(), 50)],
            p95: overlaps[rank_index(overlaps.len(), 95)],
            max: overlaps[overlaps.len() - 1],
        },
        overflowed_tiles,
    })
}

fn validate(polygons: &[[u32; 4]], width: u32, height: u32) -> Result<usize> {
    ensure!(
        (1..=MAX_DIMENSION).contains(&width) && (1..=MAX_DIMENSION).contains(&height),
        "image dimensions must be in 1..=1024"
    );
    ensure!(
        polygons.len() <= MAX_TRIANGLES,
        "drawing exceeds 1000 triangles"
    );
    Ok((width * height) as usize)
}

fn vertex(word: u32) -> [f32; 2] {
    [
        (word & 65535) as f32 / 65535.0,
        (word >> 16) as f32 / 65535.0,
    ]
}

fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn render_pixel(triangles: &[Triangle], p: [f32; 2]) -> [u8; 3] {
    let mut rgb = [255.0_f32; 3];
    for triangle in triangles {
        if triangle.covers(p) {
            let inv_alpha = 1.0 - triangle.alpha;
            for (channel, source) in rgb.iter_mut().zip(triangle.source) {
                *channel = *channel * inv_alpha + source * triangle.alpha;
            }
        }
    }
    rgb.map(|channel| channel.clamp(0.0, 255.0).floor() as u8)
}

fn bin_bounds(
    triangle: &Triangle,
    dimensions: [u32; 2],
    workgroup: [u32; 2],
) -> ([u32; 2], [u32; 2]) {
    let mut lo = [0; 2];
    let mut hi = [0; 2];
    for axis in 0..2 {
        let extent = dimensions[axis];
        let pixel_min = (triangle.min[axis] * extent as f32) as u32;
        let pixel_max = ((triangle.max[axis] * extent as f32).ceil() as u32).min(extent);
        lo[axis] = pixel_min / workgroup[axis];
        hi[axis] = pixel_max
            .div_ceil(workgroup[axis])
            .min(extent.div_ceil(workgroup[axis]));
        debug_assert!(lo[axis] <= hi[axis]);
    }
    (lo, hi)
}

fn area_distribution(values: &mut [f64]) -> AreaDistribution {
    if values.is_empty() {
        return AreaDistribution::default();
    }
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    };
    AreaDistribution {
        min: values[0],
        median,
        p95: values[rank_index(values.len(), 95)],
        max: values[values.len() - 1],
    }
}

fn rank_index(count: usize, percentile: usize) -> usize {
    debug_assert!(count > 0 && (1..=100).contains(&percentile));
    (count * percentile).div_ceil(100) - 1
}

#[cfg(test)]
mod tests {
    use super::{complexity, evaluate, AreaDistribution};

    fn triangle(color: [u8; 4], vertices: [[u16; 2]; 3]) -> [u32; 4] {
        let packed = vertices.map(|[x, y]| u32::from(x) | (u32::from(y) << 16));
        [u32::from_le_bytes(color), packed[0], packed[1], packed[2]]
    }

    fn square(color: [u8; 4]) -> [[u32; 4]; 2] {
        [
            triangle(color, [[0, 0], [65535, 0], [65535, 65535]]),
            triangle(color, [[0, 0], [65535, 65535], [0, 65535]]),
        ]
    }

    #[test]
    fn empty_drawing_is_white_and_ignores_reference_alpha() {
        let black = evaluate(&[], &[0, 0, 0, 0, 0, 0, 0, 255], 2, 1).unwrap();
        assert_eq!(black.pixels, [0xffff_ffff; 2]);
        assert_eq!(black.raw_error, 1530);
        assert_eq!(black.raw_similarity, 0.0);
        assert_eq!(black.fitness, 0.0);
        let white = evaluate(&[], &[255, 255, 255, 0], 1, 1).unwrap();
        assert_eq!(white.raw_error, 0);
        assert_eq!(white.raw_similarity, 100.0);
        assert_eq!(white.fitness, 100.0);
    }

    #[test]
    fn two_opaque_triangles_fill_a_square_and_pay_for_six_vertices() {
        let polygons = square([0, 0, 0, 255]);
        let result = evaluate(&polygons, &[0; 16], 2, 2).unwrap();
        assert_eq!(result.pixels, [0xff00_0000; 4]);
        assert_eq!(result.raw_error, 0);
        assert_eq!(result.raw_similarity, 100.0);
        assert_eq!(result.fitness, 99.99988);
        let metrics = complexity(&polygons, 2, 2, [8, 8]).unwrap();
        assert_eq!((metrics.triangles, metrics.vertices), (2, 6));
        assert_eq!(
            metrics.triangle_area,
            AreaDistribution {
                min: 0.5,
                median: 0.5,
                p95: 0.5,
                max: 0.5,
            }
        );
        assert_eq!(metrics.bbox_area.min, 1.0);
        assert_eq!(metrics.tile_overlaps.max, 2);
    }

    #[test]
    fn packed_rgb_channels_and_raw_l1_use_exact_reference_bytes() {
        let polygon = square([17, 34, 51, 255])[0];
        let result = evaluate(&[polygon], &[16, 36, 50, 0], 1, 1).unwrap();
        assert_eq!(result.pixels, [0xff33_2211]);
        assert_eq!(result.raw_error, 4);
        assert_eq!(result.raw_similarity, 99.47712418300654);
    }

    #[test]
    fn quantized_vertices_bracket_the_normalized_pixel_center() {
        let left = triangle([0, 0, 0, 255], [[0, 0], [32767, 0], [32767, 65535]]);
        let right = triangle([0, 0, 0, 255], [[0, 0], [32768, 0], [32768, 65535]]);
        assert_eq!(
            evaluate(&[left], &[0; 4], 1, 1).unwrap().pixels,
            [0xffff_ffff]
        );
        assert_eq!(
            evaluate(&[right], &[0; 4], 1, 1).unwrap().pixels,
            [0xff00_0000]
        );
    }

    #[test]
    fn half_alpha_preserves_f32_rounding_and_shared_edges_blend_twice() {
        // 128/255 is rounded upward in f32. White*(1-alpha) is 126.99999,
        // so a single layer floors to 126. Two layers floor to 63.
        let polygons = square([0, 0, 0, 128]);
        let result = evaluate(&polygons, &[0; 16], 2, 2).unwrap();
        assert_eq!(
            result.pixels,
            [0xff3f_3f3f, 0xff7e_7e7e, 0xff7e_7e7e, 0xff3f_3f3f]
        );
        assert_eq!(result.raw_error, 1134);
        let reversed = polygons.map(|[color, a, b, c]| [color, c, b, a]);
        assert_eq!(evaluate(&reversed, &[0; 16], 2, 2).unwrap(), result);
    }

    #[test]
    fn layered_blends_quantize_only_at_the_end_and_keep_order() {
        let mut polygons = [square([0, 0, 0, 64])[0]; 2];
        // This pixel is on the diagonal and each triangle covers it once.
        let result = evaluate(&polygons, &[0; 4], 1, 1).unwrap();
        assert_eq!(result.pixels, [0xff8f_8f8f]);
        polygons[1][0] = u32::from_le_bytes([255, 0, 0, 255]);
        let red_last = evaluate(&polygons, &[255, 0, 0, 9], 1, 1).unwrap();
        assert_eq!(red_last.pixels, [0xff00_00ff]);
        assert_eq!(red_last.raw_error, 0);
        polygons.reverse();
        assert_eq!(
            evaluate(&polygons, &[0; 4], 1, 1).unwrap().pixels,
            [0xff00_00bf]
        );
    }

    #[test]
    fn transparent_layers_and_layers_after_bin_capacity_remain_visible() {
        let mut polygons = vec![square([0, 255, 0, 0])[0]; 257];
        assert_eq!(
            evaluate(&polygons, &[0; 4], 1, 1).unwrap().pixels,
            [0xffff_ffff]
        );
        polygons[256][0] = u32::from_le_bytes([255, 0, 0, 255]);
        let result = evaluate(&polygons, &[255, 0, 0, 255], 1, 1).unwrap();
        assert_eq!(result.pixels, [0xff00_00ff]);
        assert_eq!(result.raw_error, 0);
    }

    #[test]
    fn bin_capacity_diagnostics_count_255_256_and_257_overlapping_entries() {
        for count in [255, 256, 257] {
            let polygons = vec![square([0, 0, 0, 0])[0]; count];
            let metrics = complexity(&polygons, 17, 19, [8, 8]).unwrap();
            assert_eq!(metrics.tile_count, 9);
            assert_eq!(metrics.tile_overlaps.min, count as u32);
            assert_eq!(metrics.tile_overlaps.p50, count as u32);
            assert_eq!(metrics.tile_overlaps.p95, count as u32);
            assert_eq!(metrics.tile_overlaps.max, count as u32);
            assert_eq!(metrics.overflowed_tiles, if count == 257 { 9 } else { 0 });
        }
    }

    #[test]
    fn aabb_bins_include_empty_tiles_and_match_exclusive_upper_bounds() {
        // 21845/65535 * 24 rounds to exactly 8. The right bound is exclusive.
        let left = triangle([0; 4], [[0, 0], [21845, 0], [21845, 65535]]);
        let metrics = complexity(&[left], 24, 16, [8, 8]).unwrap();
        assert_eq!(metrics.tile_count, 6);
        assert_eq!(metrics.tile_overlaps.min, 0);
        assert_eq!(metrics.tile_overlaps.p50, 0);
        assert_eq!(metrics.tile_overlaps.p95, 1);
        assert_eq!(metrics.tile_overlaps.max, 1);
        let boundary_line = triangle([0; 4], [[21845, 0], [21845, 65535], [21845, 0]]);
        assert_eq!(
            complexity(&[boundary_line], 24, 16, [8, 8])
                .unwrap()
                .tile_overlaps
                .max,
            0
        );
        let interior_point = triangle([0; 4], [[32768, 32768]; 3]);
        assert_eq!(
            complexity(&[interior_point], 16, 16, [8, 8])
                .unwrap()
                .tile_overlaps
                .max,
            1
        );
        let corner_point = triangle([0; 4], [[65535, 65535]; 3]);
        assert_eq!(
            complexity(&[corner_point], 16, 16, [8, 8])
                .unwrap()
                .tile_overlaps
                .max,
            0
        );
    }

    #[test]
    fn degenerate_diagonal_covers_only_centers_inside_its_aabb() {
        let line = triangle(
            [0, 0, 0, 255],
            [[21845, 21845], [65535, 65535], [65535, 65535]],
        );
        let result = evaluate(&[line], &[0; 36], 3, 3).unwrap();
        assert_eq!(
            result.pixels,
            [
                0xffff_ffff,
                0xffff_ffff,
                0xffff_ffff,
                0xffff_ffff,
                0xff00_0000,
                0xffff_ffff,
                0xffff_ffff,
                0xffff_ffff,
                0xff00_0000,
            ]
        );
    }

    #[test]
    fn empty_complexity_is_finite_and_limits_do_not_truncate() {
        let metrics = complexity(&[], 1024, 1024, [1024, 1024]).unwrap();
        assert_eq!(
            (metrics.triangles, metrics.vertices, metrics.tile_count),
            (0, 0, 1)
        );
        assert_eq!(metrics.triangle_area, AreaDistribution::default());
        assert_eq!(metrics.bbox_area, AreaDistribution::default());
        assert_eq!(metrics.tile_overlaps.max, 0);
        assert_eq!(metrics.overflowed_tiles, 0);
        let polygons = vec![square([0; 4])[0]; 1000];
        assert_eq!(complexity(&polygons, 1, 1, [8, 8]).unwrap().vertices, 3000);
        assert!(evaluate(&polygons, &[0; 4], 1, 1).is_ok());
        let excess = vec![square([0; 4])[0]; 1001];
        assert!(evaluate(&excess, &[0; 4], 1, 1).is_err());
        assert!(complexity(&excess, 1, 1, [8, 8]).is_err());
    }

    #[test]
    fn area_medians_and_serialized_complexity_preserve_the_report() {
        let polygons = [square([0; 4])[0], triangle([0; 4], [[0, 0]; 3])];
        let metrics = complexity(&polygons, 17, 19, [8, 8]).unwrap();
        assert_eq!(
            metrics.triangle_area,
            AreaDistribution {
                min: 0.0,
                median: 0.25,
                p95: 0.5,
                max: 0.5,
            }
        );
        assert_eq!(
            metrics.bbox_area,
            AreaDistribution {
                min: 0.0,
                median: 0.5,
                p95: 1.0,
                max: 1.0,
            }
        );
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: super::Complexity = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, metrics);
    }

    #[test]
    fn invalid_dimensions_reference_lengths_and_workgroups_are_rejected() {
        for dimensions in [[0, 1], [1, 0], [1025, 1], [1, 1025], [u32::MAX, u32::MAX]] {
            assert!(evaluate(&[], &[], dimensions[0], dimensions[1]).is_err());
            assert!(complexity(&[], dimensions[0], dimensions[1], [8, 8]).is_err());
        }
        for length in [0, 3, 5, 8] {
            assert!(evaluate(&[], &vec![0; length], 1, 1).is_err());
        }
        for workgroup in [[0, 8], [8, 0], [1025, 8], [8, u32::MAX]] {
            assert!(complexity(&[], 1, 1, workgroup).is_err());
        }
    }
}
