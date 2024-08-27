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

pub fn fill_shape_2(buffer: &mut Vec<u8>, polygon: &Polygon, w: usize, h: usize) {
    let points = &polygon.points;
    let v: Vec<Point> = points.iter().map(|p| p.translate(w, h)).collect();

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
    let minx = (i32::min(i32::min(X1, X2), X3) + 0xF) >> 4;
    let maxx = (i32::max(i32::max(X1, X2), X3) + 0xF) >> 4;
    let miny = (i32::min(i32::min(Y1, Y2), Y3) + 0xF) >> 4;
    let maxy = (i32::max(i32::max(Y1, Y2), Y3) + 0xF) >> 4;

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

    let mut CY1 = C1 + DX12 * (miny << 4) - DY12 * (minx << 4);
    let mut CY2 = C2 + DX23 * (miny << 4) - DY23 * (minx << 4);
    let mut CY3 = C3 + DX31 * (miny << 4) - DY31 * (minx << 4);

    // Scan through bounding rectangle
    for y in miny as usize..maxy as usize {
        // Start value for horizontal scan
        let mut CX1 = CY1;
        let mut CX2 = CY2;
        let mut CX3 = CY3;

        for x in minx as usize..maxx as usize {
            if (CX1 > 0 && CX2 > 0 && CX3 > 0) || (CX1 < 0 && CX2 < 0 && CX3 < 0) {
                let idx = 4 * y * w + 4 * x;
                fill_pixel(buffer, idx, &polygon.color);
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
