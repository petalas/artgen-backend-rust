//! A separate, conservative check for pixels near floating-point triangle edges.
//!
//! The nominal scalar renderer remains unchanged. WGSL permits 2.5-ULP division,
//! 3-ULP UNORM unpacking and contracted arithmetic. Propagate outward f32 bounds
//! through the same edge/blend formula instead of ignoring boundary mismatches.
//! See <https://www.w3.org/TR/WGSL/#floating-point-accuracy>.
//! This is a diagnostic envelope, not a proof of portable bitwise rendering.
use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn exact(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    fn rounded(lo: f64, hi: f64, ulps: u32) -> Self {
        let (mut low, mut high) = (lo as f32, hi as f32);
        for _ in 0..ulps {
            low = low.next_down();
            high = high.next_up();
        }
        Self {
            lo: f64::from(low),
            hi: f64::from(high),
        }
    }

    fn ratio(numerator: u32, denominator: u32) -> Self {
        let value = f64::from(numerator) / f64::from(denominator);
        Self::rounded(value, value, 3)
    }

    fn add(self, other: Self) -> Self {
        Self::rounded(self.lo + other.lo, self.hi + other.hi, 1)
    }

    fn sub(self, other: Self) -> Self {
        Self::rounded(self.lo - other.hi, self.hi - other.lo, 1)
    }

    fn mul(self, other: Self) -> Self {
        let values = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::rounded(low, high, 1)
    }

    fn union(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn clamp(self, low: f64, high: f64) -> Self {
        Self {
            lo: self.lo.clamp(low, high),
            hi: self.hi.clamp(low, high),
        }
    }
}

#[derive(Clone, Copy)]
enum Coverage {
    Outside,
    Inside,
    Uncertain,
}

fn vertex(word: u32) -> [Interval; 2] {
    [
        Interval::ratio(word & 65535, 65535),
        Interval::ratio(word >> 16, 65535),
    ]
}

fn edge(a: [Interval; 2], b: [Interval; 2], p: [Interval; 2]) -> Interval {
    p[0].sub(a[0])
        .mul(b[1].sub(a[1]))
        .sub(p[1].sub(a[1]).mul(b[0].sub(a[0])))
}

fn coverage(vertices: [[Interval; 2]; 3], p: [Interval; 2]) -> Coverage {
    let [a, b, c] = vertices;
    let mut definite_aabb = true;
    for axis in 0..2 {
        let min_lo = a[axis].lo.min(b[axis].lo).min(c[axis].lo);
        let min_hi = a[axis].hi.min(b[axis].hi).min(c[axis].hi);
        let max_lo = a[axis].lo.max(b[axis].lo).max(c[axis].lo);
        let max_hi = a[axis].hi.max(b[axis].hi).max(c[axis].hi);
        if p[axis].hi < min_lo || p[axis].lo > max_hi {
            return Coverage::Outside;
        }
        definite_aabb &= p[axis].lo >= min_hi && p[axis].hi <= max_lo;
    }
    let edges = [edge(a, b, p), edge(b, c, p), edge(c, a, p)];
    let possible = edges.iter().all(|e| e.hi >= 0.0) || edges.iter().all(|e| e.lo <= 0.0);
    if !possible {
        return Coverage::Outside;
    }
    let definite = edges.iter().all(|e| e.lo >= 0.0) || edges.iter().all(|e| e.hi <= 0.0);
    if definite && definite_aabb {
        Coverage::Inside
    } else {
        Coverage::Uncertain
    }
}

fn pixel_bounds(polygons: &[[u32; 4]], width: u32, height: u32, index: u32) -> [Interval; 3] {
    let p = [
        Interval::ratio(2 * (index % width) + 1, 2 * width),
        Interval::ratio(2 * (index / width) + 1, 2 * height),
    ];
    let mut color = [Interval::exact(255.0); 3];
    for polygon in polygons {
        let kind = coverage(
            [vertex(polygon[1]), vertex(polygon[2]), vertex(polygon[3])],
            p,
        );
        if matches!(kind, Coverage::Outside) {
            continue;
        }
        let alpha = Interval::ratio(polygon[0] >> 24, 255).clamp(0.0, 1.0);
        let inverse = Interval::exact(1.0).sub(alpha);
        for channel in 0..3 {
            let source = Interval::ratio((polygon[0] >> (channel * 8)) & 255, 255)
                .mul(Interval::exact(255.0));
            let blended = color[channel].mul(inverse).add(source.mul(alpha));
            color[channel] = match kind {
                Coverage::Inside => blended,
                Coverage::Uncertain => color[channel].union(blended),
                Coverage::Outside => unreachable!("outside coverage skipped above"),
            };
        }
    }
    color.map(|c| c.clamp(0.0, 255.0))
}

#[derive(Default, Serialize)]
pub struct PixelAudit {
    pub outside_nominal_quantum: u32,
    pub unexplained_pixels: u32,
    pub widest_channel_envelope: u32,
}

/// Check large nominal differences using a per-pixel arithmetic envelope. This
/// never skips a differing pixel or accepts malformed framebuffer alpha.
/// # Errors
/// Rejects invalid image dimensions, triangle counts and framebuffer lengths.
pub fn audit(
    polygons: &[[u32; 4]],
    actual: &[u32],
    nominal: &[u32],
    width: u32,
    height: u32,
) -> Result<PixelAudit> {
    ensure!(
        (1..=1024).contains(&width) && (1..=1024).contains(&height),
        "invalid dimensions"
    );
    ensure!(
        polygons.len() <= crate::settings::POLYGON_CAPACITY,
        "too many triangles"
    );
    ensure!(
        actual.len() == (width * height) as usize && actual.len() == nominal.len(),
        "framebuffer length mismatch"
    );
    let mut result = PixelAudit::default();
    for (index, (&pixel, &expected)) in actual.iter().zip(nominal).enumerate() {
        if pixel >> 24 != 255 {
            result.unexplained_pixels += 1;
            continue;
        }
        let large =
            (0..3).any(|c| ((pixel >> (c * 8)) & 255).abs_diff((expected >> (c * 8)) & 255) > 1);
        if !large {
            continue;
        }
        result.outside_nominal_quantum += 1;
        let bounds = pixel_bounds(polygons, width, height, index as u32);
        if !contains_pixel(pixel, bounds, &mut result.widest_channel_envelope) {
            result.unexplained_pixels += 1;
        }
    }
    Ok(result)
}

fn contains_pixel(pixel: u32, bounds: [Interval; 3], widest: &mut u32) -> bool {
    let mut contained = true;
    for (channel, bound) in bounds.iter().enumerate() {
        let (low, high) = (bound.lo.floor() as u32, bound.hi.floor() as u32);
        *widest = (*widest).max(high - low);
        contained &= (low..=high).contains(&((pixel >> (channel * 8)) & 255));
    }
    contained
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_interior_pixel_cannot_be_arbitrarily_changed() {
        let triangle = [0xff00_0000, 0, 65535, 65535 << 16];
        let nominal = vec![0xff00_0000; 16];
        let mut actual = nominal.clone();
        actual[0] = 0xff80_8080;
        let result = audit(&[triangle], &actual, &nominal, 4, 4).unwrap();
        assert_eq!(result.unexplained_pixels, 1);
    }
    #[test]
    fn exact_shared_edge_reports_bounded_coverage_ambiguity() {
        let triangle = [0xff00_0000, 0, 65535, 65535 << 16];
        let actual = [0xffff_ffff];
        let nominal = [0xff00_0000];
        let result = audit(&[triangle], &actual, &nominal, 1, 1).unwrap();
        assert_eq!(result.outside_nominal_quantum, 1);
        assert_eq!(result.unexplained_pixels, 0);
    }
    #[test]
    fn blank_canvas_and_alpha_corruption_do_not_get_an_edge_exception() {
        let result = audit(&[], &[0xff00_0000], &[0xffff_ffff], 1, 1).unwrap();
        assert_eq!(result.unexplained_pixels, 1);
        let result = audit(&[], &[0x00ff_ffff], &[0xffff_ffff], 1, 1).unwrap();
        assert_eq!(result.unexplained_pixels, 1);
    }
}
