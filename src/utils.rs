use std::{collections::HashMap, time::Duration};

use rand::Rng;

use crate::{
    evaluator::EvaluatorPayload,
    models::{
        color::Color,
        line::Line,
        point::{FixedPoint, Point},
        polygon::Polygon,
    },
};
use std::simd::num::SimdUint;
use std::simd::{u16x4, u8x4, Simd};

pub struct ImageDimensions {
    pub width: usize,
    pub height: usize,
}

pub fn randomu8() -> u8 {
    rand::thread_rng().gen::<u8>()
}

pub fn randomf32() -> f32 {
    rand::thread_rng().gen::<f32>()
}

pub fn randomf32_clamped(min: f32, max: f32) -> f32 {
    return rand::thread_rng().gen_range(min..max);
}

pub fn calculate_aspect_ratio_fit(
    src_width: usize,
    src_height: usize,
    max_w: usize,
    max_h: usize,
) -> ImageDimensions {
    let w = src_width as f32;
    let h = src_height as f32;
    let ratio: f32 = (max_w as f32 / w as f32).min(max_h as f32 / h as f32);
    ImageDimensions {
        width: (w * ratio).round() as usize,
        height: (h * ratio).round() as usize,
    }
}

// based on scanline fill
// https://www.cs.ucdavis.edu/~ma/ECS175_S00/Notes/0413_a.pdf
pub fn fill_shape(buffer: &mut Vec<u8>, polygon: &Polygon, w: usize, h: usize) {
    let points = &polygon.points;
    let pixel_coords: Vec<Point> = points.iter().map(|p| p.translate(w, h)).collect();
    let sides = sides(&pixel_coords);
    let points_inside: Vec<Point> = get_points_inside(sides);
    points_inside.iter().for_each(|p| {
        let x = p.x as usize;
        let y = p.y as usize;
        let idx = 4 * y * w + 4 * x;
        assert!(buffer.len() > idx);
        fill_pixel(buffer, idx as usize, &polygon.color);
    });
}

// Based on https://web.archive.org/web/20050408192410/http://sw-shader.sourceforge.net/rasterizer.html
// TODO: optimize it by using SIMD to process multiple pixels at once
// Note: we want to use "portable SIMD" so we compile for any target
// https://doc.rust-lang.org/std/simd/index.html
pub fn fill_triangle(
    buffer: &mut Vec<u8>,
    polygon: &Polygon,
    w: usize,
    h: usize,
) -> Result<(), &'static str> {
    let points = &polygon.points;
    assert_eq!(points.len(), 3);
    // Initialize the vector with capacity and collect the results
    let mut v: Vec<FixedPoint> = points
        .iter()
        .map(|p| p.translate_to_fixed(w, h))
        .collect::<Result<Vec<_>, _>>()?;

    // ensure correct orientation, swap two points around if needed
    let orient = orient_2d(&v[0], &v[1], &v[2]);
    if orient == 0 {
        return Ok(()); // degenerate triangle
    }

    if orient > 0 {
        v.swap(0, 1);
    }

    // 28.4 fixed-point coordinates
    // << 4 is basically multiplying by 16
    let Y1 = v[0].y << 4;
    let Y2 = v[1].y << 4;
    let Y3 = v[2].y << 4;
    let X1 = v[0].x << 4;
    let X2 = v[1].x << 4;
    let X3 = v[2].x << 4;

    // Deltas
    let DX12 = X1 - X2;
    let DX23 = X2 - X3;
    let DX31 = X3 - X1;

    let DY12 = Y1 - Y2;
    let DY23 = Y2 - Y3;
    let DY31 = Y3 - Y1;

    // Fixed-point deltas
    let FDX12 = DX12 << 4;
    let FDX23 = DX23 << 4;
    let FDX31 = DX31 << 4;

    let FDY12 = DY12 << 4;
    let FDY23 = DY23 << 4;
    let FDY31 = DY31 << 4;

    // Bounding rectangle
    let mut minx = (i32::min(i32::min(X1, X2), X3) + 0xF) >> 4;
    let maxx = (i32::max(i32::max(X1, X2), X3) + 0xF) >> 4;
    let mut miny = (i32::min(i32::min(Y1, Y2), Y3) + 0xF) >> 4;
    let maxy = (i32::max(i32::max(Y1, Y2), Y3) + 0xF) >> 4;

    // println!("minx={}, maxx={}, miny={}, maxy={}", minx, maxx, miny, maxy);

    // Block size, standard 8x8 (must be power of two)
    let q = 8;

    // Start in corner of 8x8 block
    minx &= !(q - 1);
    miny &= !(q - 1);

    // println!("minx {}, miny {}", minx, miny);

    // Constant part of half-edge functions
    let mut C1 = DY12 * X1 - DX12 * Y1;
    let mut C2 = DY23 * X2 - DX23 * Y2;
    let mut C3 = DY31 * X3 - DX31 * Y3;

    // Correct for fill convention
    if DY12 < 0 || (DY12 == 0 && DX12 > 0) {
        C1 += 1
    };
    if DY23 < 0 || (DY23 == 0 && DX23 > 0) {
        C2 += 1
    };
    if DY31 < 0 || (DY31 == 0 && DX31 > 0) {
        C3 += 1
    };

    // loop through blocks
    for y in (miny..maxy).step_by(q as usize) {
        for x in (minx..maxx).step_by(q as usize) {
            // Corners of block
            let x0 = x << 4;
            let x1 = (x + q - 1) << 4;
            let y0 = y << 4;
            let y1 = (y + q - 1) << 4;

            // Evaluate half-space functions
            let a00 = (C1 + DX12 * y0 - DY12 * x0 > 0) as i32;
            let a10 = (C1 + DX12 * y0 - DY12 * x1 > 0) as i32;
            let a01 = (C1 + DX12 * y1 - DY12 * x0 > 0) as i32;
            let a11 = (C1 + DX12 * y1 - DY12 * x1 > 0) as i32;
            let a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3);

            let b00 = (C2 + DX23 * y0 - DY23 * x0 > 0) as i32;
            let b10 = (C2 + DX23 * y0 - DY23 * x1 > 0) as i32;
            let b01 = (C2 + DX23 * y1 - DY23 * x0 > 0) as i32;
            let b11 = (C2 + DX23 * y1 - DY23 * x1 > 0) as i32;
            let b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);

            let c00 = (C3 + DX31 * y0 - DY31 * x0 > 0) as i32;
            let c10 = (C3 + DX31 * y0 - DY31 * x1 > 0) as i32;
            let c01 = (C3 + DX31 * y1 - DY31 * x0 > 0) as i32;
            let c11 = (C3 + DX31 * y1 - DY31 * x1 > 0) as i32;
            let c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);

            // Skip block when outside an edge
            if a == 0x0 || b == 0x0 || c == 0x0 {
                continue;
            }

            // Accept whole block when totally covered
            if a == 0xF && b == 0xF && c == 0xF {
                for iy in y..(y + q) {
                    for ix in x..(x + q) {
                        let idx = (iy as usize * w + ix as usize) * 4;
                        blend_simd(
                            (&mut buffer[idx..idx + 4]).try_into().unwrap(),
                            &polygon.color.into(),
                        );
                    }
                }
            } else {
                let mut CY1 = C1 + DX12 * y0 - DY12 * x0;
                let mut CY2 = C2 + DX23 * y0 - DY23 * x0;
                let mut CY3 = C3 + DX31 * y0 - DY31 * x0;

                for iy in y..(y + q) {
                    let mut CX1 = CY1;
                    let mut CX2 = CY2;
                    let mut CX3 = CY3;

                    for ix in x..(x + q) {
                        if CX1 >= 0 && CX2 >= 0 && CX3 >= 0 {
                            let idx = (iy as usize * w + ix as usize) * 4;
                            blend_simd(
                                (&mut buffer[idx..idx + 4]).try_into().unwrap(),
                                &polygon.color.into(),
                            );
                        }
                        CX1 -= FDY12;
                        CX2 -= FDY23;
                        CX3 -= FDY31;
                    }

                    CY1 += FDX12;
                    CY2 += FDX23;
                    CY3 += FDX31;
                }
            }
        }
    }

    Ok(())
}

// keeping it around for benchmarks
pub fn fill_pixel(buffer: &mut Vec<u8>, index: usize, color: &Color) {
    assert!(buffer.len() > index + 3);
    let a = color.a as f32 / 255.0;
    let inv_a = 1.0 - a;
    buffer[index] = ((buffer[index] as f32 * inv_a) + (color.r as f32 * a)) as u8;
    buffer[index + 1] = ((buffer[index + 1] as f32 * inv_a) + (color.g as f32 * a)) as u8;
    buffer[index + 2] = ((buffer[index + 2] as f32 * inv_a) + (color.b as f32 * a)) as u8;
    buffer[index + 3] = u8::max(buffer[index + 3], color.a);
}

// keeping it around for benchmarks
pub fn blend(base: &mut [u8; 4], fill_color: &[u8; 4]) {
    // Extract the alpha value and calculate the blending factors
    let alpha = fill_color[3] as f32 / 255.0;
    let inv_alpha = 1.0 - alpha;

    // Calculate the blended values for r, g, b
    base[0] = ((base[0] as f32 * inv_alpha) + (fill_color[0] as f32 * alpha)) as u8;
    base[1] = ((base[1] as f32 * inv_alpha) + (fill_color[1] as f32 * alpha)) as u8;
    base[2] = ((base[2] as f32 * inv_alpha) + (fill_color[2] as f32 * alpha)) as u8;
    base[3] = u8::max(base[3], fill_color[3]);
}

// Color blending using SIMD, equivalent to:
// let alpha = fill_color[3] as f32 / 255.0;
// let inv_alpha = 1.0 - alpha;
// base[0] = ((base[0] as f32 * inv_alpha) + (fill_color[0] as f32 * alpha)) as u8;
// base[1] = ((base[1] as f32 * inv_alpha) + (fill_color[1] as f32 * alpha)) as u8;
// base[2] = ((base[2] as f32 * inv_alpha) + (fill_color[2] as f32 * alpha)) as u8;
// base[3] = u8::max(base[3], fill_color[3]);
pub fn blend_simd(base: &mut [u8; 4], fill_color: &[u8; 4]) {
    // println!("Before inside blend_simd: {:?}", &base);
    let original_max_alpha = base[3].max(fill_color[3]); // keep it around because it gets overwritten by the SIMD stuff
    let base_simd: u16x4 = u8x4::from_slice(base).cast();
    let fill_color_simd: u16x4 = u8x4::from_slice(fill_color).cast();
    let alpha_simd = u16x4::splat(fill_color[3] as u16);
    const MAX: u16x4 = u16x4::from_array([255, 255, 255, 255]);
    let inv_a_simd = MAX - alpha_simd;
    let blend = (base_simd * inv_a_simd + fill_color_simd * alpha_simd) / MAX;
    blend.cast().copy_to_slice(base);
    base[3] = original_max_alpha;
    // println!("After inside blend_simd: {:?}", &base);
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
pub fn orient_2d(a: &FixedPoint, b: &FixedPoint, c: &FixedPoint) -> i32 {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

/// Returns all lines looping back to the first one
/// [A, B, C] -> [AB, BC, CA]
fn sides(points: &Vec<Point>) -> Vec<Line> {
    let l = points.len();
    assert!(l > 1);
    (0..l)
        .map(|i| Line::new(&points[i], &points[(i + 1) % l]))
        .collect()
}

// https://www.cs.ucdavis.edu/~ma/ECS175_S00/Notes/0413_a.pdf
fn get_points_inside(sides: Vec<Line>) -> Vec<Point> {
    let mut points: Vec<Point> = vec![];

    let br = get_bounding_rect(&sides);
    if br.min_y >= br.max_y || br.min_x >= br.max_x {
        return points;
    }

    let mut edge_table: HashMap<usize, Vec<Line>> = HashMap::new();
    sides.into_iter().for_each(|l| {
        let y = l.min_y;
        let list = edge_table.entry(y).or_default();
        list.push(l);
        list.sort_by(|a, b| {
            a.x_of_min_y
                .partial_cmp(&b.x_of_min_y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    });

    let mut active_edges: Vec<Line> = vec![];

    for row in br.min_y..=br.max_y {
        let mut intersection_points: Vec<Point> = vec![];

        // add new edges if we've reached their height
        if edge_table.contains_key(&row) {
            active_edges.append(&mut edge_table.remove(&row).unwrap());
        }

        active_edges.iter_mut().for_each(|line| {
            // https://web.cs.ucdavis.edu/~ma/ECS175_S00/Notes/0411_b.pdf
            // edge case: if at ymax of edge, don't count the point
            if line.max_y != row {
                intersection_points.push(Point {
                    x: line.next_x_to_fill,
                    y: row as f32,
                });
            }
            // TODO: change x = x+ dx/dy to integer arithmetic
            // https://www.cs.ucdavis.edu/~ma/ECS175_S00/Notes/0413_a.pdf
            line.next_x_to_fill = line.next_x_to_fill + line.inverse_slope;
        });

        intersection_points.sort(); // dedup only works on sequential items

        if intersection_points.len() > 1 {
            // console::log_1(&format!("intersection_points {:?}", intersection_points).into());
            let mut i = 0;
            while i < intersection_points.len() - 1 {
                points.append(&mut points_between(
                    &intersection_points[i],
                    &intersection_points[i + 1],
                ));
                i += 2;
            }
        }

        active_edges.retain(|edge| edge.max_y > row);
    }

    points
}

struct BoundingRect {
    max_x: usize,
    max_y: usize,
    min_x: usize,
    min_y: usize,
}

fn get_bounding_rect(sides: &Vec<Line>) -> BoundingRect {
    let max_x = sides
        .iter()
        .map(|s| s.max_x)
        .max()
        .expect("Expected valid max_x");

    let max_y = sides
        .iter()
        .map(|s| s.max_y)
        .max()
        .expect("Expected valid max_y");

    let min_x = sides
        .iter()
        .map(|s| s.min_x)
        .min()
        .expect("Expected valid min_y");

    let min_y = sides
        .iter()
        .map(|s| s.min_y)
        .min()
        .expect("Expected valid min_y");

    return BoundingRect {
        max_x: max_x,
        max_y: max_y,
        min_x: min_x,
        min_y: min_y,
    };
}

fn points_between(a: &Point, b: &Point) -> Vec<Point> {
    // console::log_1(&format!("points_between {:?} <--> {:?}", a, b).into());
    assert_eq!(a.y, b.y);
    if a.x == b.x {
        return vec![a.clone()];
    }
    if a.x - b.x == 1.0 {
        return vec![a.clone(), b.clone()];
    }

    let min = a.x.min(b.x);
    let max = a.x.max(b.x);

    let p = (min as usize..max as usize)
        .map(|x| Point {
            x: x as f32,
            y: a.y,
        })
        .collect();

    // console::log_1(&format!("{:?}", p).into());
    return p;
}

pub fn translate_coord(number: f32) -> f32 {
    return number * 2.0 - 1.0;
}

pub fn translate_color(color: u8) -> f32 {
    color as f32 / 255.0
}

pub fn print_stats(stats: EvaluatorPayload, real_elapsed: Duration) {
    let t = stats.elapsed;
    if t == 0 || real_elapsed.as_millis() == 0 {
        return;
    }

    let e = stats.evaluations;
    let m = stats.mutations;

    let total_sec = t as f64 / 1000.0;
    let total_min = total_sec / 60.0;
    let total_h = total_min / 60.0;

    let real_ms = real_elapsed.as_millis() as f64;
    let real_sec = real_ms / 1000.0;
    let real_min = real_sec / 60.0;
    let real_h = real_min / 60.0;

    let eval_rate = e as f64 / (real_ms as f64 / 1000.0);
    let mut_rate = m as f64 / (real_ms as f64 / 1000.0);
    let speedup = t as f64 / real_elapsed.as_millis() as f64;

    let time = if real_h > 1.0 {
        format!("{:4.1}h", real_h)
    } else {
        if real_min > 1.0 {
            format!("{:4.1}m", real_min)
        } else {
            format!("{:4.1}s", real_sec)
        }
    };

    let total_time = if total_h > 1.0 {
        format!("{:4.1}h", total_h)
    } else {
        if total_min > 1.0 {
            format!("{:4.1}m", total_min)
        } else {
            format!("{:4.1}s", total_sec)
        }
    };

    println!(
        "Elapsed time: {} | {} total => {:.2}x speedup | evaluations: {:<10} ~{:5.0}/s |  mutations: {:<10} ~{:6.0}/s | best => {:3.4}",
        time, total_time, speedup, e, eval_rate, m, mut_rate, stats.best.fitness
    );
}

#[cfg(test)]
mod tests {

    use super::*;

    /// Asserts that two slices are approximately equal, allowing each element to be off by 1.
    pub fn assert_approx_eq(expected: &[u8], actual: &[u8]) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "Slices have different lengths"
        );

        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (*e as i16 - *a as i16).abs() <= 1, // i16 to avoid underflow
                "Values at index {} are not approximately equal: expected {}, got {}",
                i,
                e,
                a
            );
        }
    }

    #[test]
    fn test_fill_pixel_combinations() {
        let test_vals = [0, 128, 255];
        let mut test_cases = vec![];

        // Generate test cases
        for &r in &test_vals {
            for &g in &test_vals {
                for &b in &test_vals {
                    for &a in &test_vals {
                        let color = Color { r, g, b, a };

                        // Generate different starting buffers
                        let initial_buffers = vec![
                            vec![0, 0, 0, 0],
                            vec![128, 128, 128, 128],
                            vec![255, 255, 255, 255],
                            vec![r, g, b, a],
                        ];

                        for buf in initial_buffers {
                            test_cases.push((buf, color));
                        }
                    }
                }
            }
        }

        // Run test cases
        for (mut buf, color) in test_cases {
            // TODO: cleanup
            // println!("testing {:?} <-- {:?}", buf, color);
            let original_base = buf.clone(); // for logs

            // clone before blending to avoid side effects
            // these ones are to test blend which is the same as fill_pixel but takes [u8; 4]
            let mut blend_buf = buf.clone();
            let mut blend_base: [u8; 4] = Default::default();
            blend_base.copy_from_slice(&blend_buf[0..4]);
            let blend_color: [u8; 4] = [color.r, color.g, color.b, color.a];

            let idx = 0;

            // Calculate expected values
            let alpha = color.a as f32 / 255.0;
            let inv_a = 1.0 - alpha;
            let exp_r = ((buf[0] as f32 * inv_a) + (color.r as f32 * alpha)) as u8;
            let exp_g = ((buf[1] as f32 * inv_a) + (color.g as f32 * alpha)) as u8;
            let exp_b = ((buf[2] as f32 * inv_a) + (color.b as f32 * alpha)) as u8;
            let exp_a = u8::max(buf[3], color.a);

            let exp_buf = vec![exp_r, exp_g, exp_b, exp_a];

            // Act
            fill_pixel(&mut buf, idx, &color);

            // Assert
            assert_approx_eq(buf.as_slice(), exp_buf.as_slice());

            // TODO: cleanup, test both
            // test blend here too
            // blend(&mut blend_base, &blend_color);
            blend_simd(&mut blend_base, &blend_color);
            assert_approx_eq(
                blend_base.as_slice(),
                [exp_r, exp_g, exp_b, exp_a].as_slice(),
            );
        }
    }

    // helper for other tests
    fn assert_pixel_eq(buffer: &[u8], x: usize, y: usize, width: usize, expected: Color) {
        let idx = 4 * (y * width + x);
        assert_eq!(
            buffer[idx], expected.r,
            "Red channel mismatch at ({}, {})",
            x, y
        );
        assert_eq!(
            buffer[idx + 1],
            expected.g,
            "Green channel mismatch at ({}, {})",
            x,
            y
        );
        assert_eq!(
            buffer[idx + 2],
            expected.b,
            "Blue channel mismatch at ({}, {})",
            x,
            y
        );
        assert_eq!(
            buffer[idx + 3],
            expected.a,
            "Alpha channel mismatch at ({}, {})",
            x,
            y
        );
    }

    #[test]
    fn test_fill_triangle_basic() {
        let mut buffer = vec![0; 16 * 16 * 4]; // 16x16 image with RGBA channels
        let red_color = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };

        // First triangle covering the top-left half
        let polygon1 = Polygon {
            points: vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 0.0, y: 1.0 },
            ],
            color: red_color,
        };
        fill_triangle(&mut buffer, &polygon1, 16, 16);

        // Second triangle covering the bottom-right half
        let polygon2 = Polygon {
            points: vec![
                Point { x: 1.0, y: 0.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 0.0, y: 1.0 },
            ],
            color: red_color,
        };
        fill_triangle(&mut buffer, &polygon2, 16, 16);

        // Check all pixels to ensure they are filled with red
        for y in 0..16 {
            for x in 0..16 {
                assert_pixel_eq(&buffer, x, y, 16, red_color);
            }
        }
    }

    #[test]
    fn test_fill_triangle_out_of_bounds() {
        let mut buffer = vec![0u8; 16 * 16 * 4]; // 16x16 image with RGBA channels
        let polygon = Polygon {
            points: vec![
                Point { x: -1.0, y: -1.0 },
                Point { x: -0.5, y: -1.0 },
                Point { x: -1.0, y: -0.5 },
            ],
            color: Color {
                r: 0,
                g: 255,
                b: 0,
                a: 255,
            }, // Green color
        };
        // Attempt to fill the triangle and handle out-of-bounds points
        if let Err(_) = fill_triangle(&mut buffer, &polygon, 16, 16) {
            // Check that no pixels are filled since the triangle is out of bounds
            assert!(buffer.iter().all(|&pixel| pixel == 0u8));
        }
    }
}
