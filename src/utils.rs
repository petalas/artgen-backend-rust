use std::collections::HashMap;

use rand::Rng;

use crate::models::{color::Color, line::Line, point::Point, polygon::Polygon};

pub struct ImageDimensions {
    pub width: usize,
    pub height: usize,
}

pub fn randomu8() -> u8 {
    rand::thread_rng().gen::<u8>()
}

pub fn randomf64() -> f64 {
    rand::thread_rng().gen::<f64>()
}

pub fn randomf64_clamped(min: f64, max: f64) -> f64 {
    return rand::thread_rng().gen_range(min..max);
}

pub fn calculate_aspect_ratio_fit(
    src_width: usize,
    src_height: usize,
    max_w: usize,
    max_h: usize,
) -> ImageDimensions {
    let w = src_width as f64;
    let h = src_height as f64;
    let ratio: f64 = (max_w as f64 / w as f64).min(max_h as f64 / h as f64);
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
        fill_pixel(buffer, idx, &polygon.color);
    });
}

// Based on https://web.archive.org/web/20050408192410/http://sw-shader.sourceforge.net/rasterizer.html
// Had to do additional checks otherwise it doesn't work for all triangles
// Another solution (maybe?) would be to sort points clockwise.
pub fn fill_triangle(buffer: &mut Vec<u8>, polygon: &Polygon, w: usize, h: usize) {
    let points = &polygon.points;
    let v: Vec<Point> = points.iter().map(|p| p.translate(w, h)).collect();

    // 28.4 fixed-point coordinates
    let Y1 = (v[0].y * 16f64).round() as i32;
    let Y2 = (v[1].y * 16f64).round() as i32;
    let Y3 = (v[2].y * 16f64).round() as i32;

    let X1 = (v[0].x * 16f64).round() as i32;
    let X2 = (v[1].x * 16f64).round() as i32;
    let X3 = (v[2].x * 16f64).round() as i32;

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

            let a002 = (C1 + DX12 * y0 - DY12 * x0 < 0) as i32;
            let a102 = (C1 + DX12 * y0 - DY12 * x1 < 0) as i32;
            let a012 = (C1 + DX12 * y1 - DY12 * x0 < 0) as i32;
            let a112 = (C1 + DX12 * y1 - DY12 * x1 < 0) as i32;
            let a2 = (a002 << 0) | (a102 << 1) | (a012 << 2) | (a112 << 3);

            let b00 = (C2 + DX23 * y0 - DY23 * x0 > 0) as i32;
            let b10 = (C2 + DX23 * y0 - DY23 * x1 > 0) as i32;
            let b01 = (C2 + DX23 * y1 - DY23 * x0 > 0) as i32;
            let b11 = (C2 + DX23 * y1 - DY23 * x1 > 0) as i32;
            let b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);

            let b002 = (C2 + DX23 * y0 - DY23 * x0 < 0) as i32;
            let b102 = (C2 + DX23 * y0 - DY23 * x1 < 0) as i32;
            let b012 = (C2 + DX23 * y1 - DY23 * x0 < 0) as i32;
            let b112 = (C2 + DX23 * y1 - DY23 * x1 < 0) as i32;
            let b2 = (b002 << 0) | (b102 << 1) | (b012 << 2) | (b112 << 3);

            let c00 = (C3 + DX31 * y0 - DY31 * x0 > 0) as i32;
            let c10 = (C3 + DX31 * y0 - DY31 * x1 > 0) as i32;
            let c01 = (C3 + DX31 * y1 - DY31 * x0 > 0) as i32;
            let c11 = (C3 + DX31 * y1 - DY31 * x1 > 0) as i32;
            let c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);

            let c002 = (C3 + DX31 * y0 - DY31 * x0 < 0) as i32;
            let c102 = (C3 + DX31 * y0 - DY31 * x1 < 0) as i32;
            let c012 = (C3 + DX31 * y1 - DY31 * x0 < 0) as i32;
            let c112 = (C3 + DX31 * y1 - DY31 * x1 < 0) as i32;
            let c2 = (c002 << 0) | (c102 << 1) | (c012 << 2) | (c112 << 3);

            // Skip block when outside an edge
            // TODO optimization? (should only have to check one set of these)
            if (a == 0x0 || b == 0x0 || c == 0x0) && (a2 == 0x0 || b2 == 0x0 || c2 == 0x0) {
                continue;
            }

            // Accept whole block when totally covered
            // TODO optimization? (should only have to check one set of these)
            if (a == 0xF && b == 0xF && c == 0xF) || (a2 == 0xF && b2 == 0xF && c2 == 0xF) {
                for iy in y..(y + q) {
                    for ix in x..(x + q) {
                        let idx = 4 * iy * w as i32 + 4 * ix;
                        fill_pixel(buffer, idx as usize, &polygon.color);
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
                        // TODO optimization? (should only have to check one set of these)
                        if (CX1 >= 0 && CX2 >= 0 && CX3 >= 0) || (CX1 <= 0 && CX2 <= 0 && CX3 <= 0)
                        {
                            let idx = 4 * iy * w as i32 + 4 * ix;
                            fill_pixel(buffer, idx as usize, &&polygon.color);
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
}

fn fill_pixel(buffer: &mut Vec<u8>, index: usize, color: &Color) {
    if buffer.len() <= index {
        return;
    }
    let a = color.a as f64 / 255.0;
    let b = 1.0 - a;
    buffer[index] = ((buffer[index] as f64 * b) + (color.r as f64 * a)).round() as u8;
    buffer[index + 1] = ((buffer[index + 1] as f64 * b) + (color.g as f64 * a)).round() as u8;
    buffer[index + 2] = ((buffer[index + 2] as f64 * b) + (color.b as f64 * a)).round() as u8;
    buffer[index + 3] = u8::max(buffer[index + 3], color.a);
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
                    y: row as f64,
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
            x: x as f64,
            y: a.y,
        })
        .collect();

    // console::log_1(&format!("{:?}", p).into());
    return p;
}
