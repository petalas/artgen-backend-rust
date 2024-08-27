use std::collections::HashMap;

use rand::Rng;

use crate::models::{color::Color, line::Line, point::Point};

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
pub fn fill_shape(buffer: &mut Vec<u8>, points: &Vec<Point>, color: &Color, w: usize, h: usize) {
    let pixel_coords: Vec<Point> = points.iter().map(|p| p.translate(w, h)).collect();
    let sides = sides(&pixel_coords);
    let points_inside: Vec<Point> = get_points_inside(sides);
    points_inside.iter().for_each(|p| {
        let x = p.x as usize;
        let y = p.y as usize;
        let idx = 4 * y * w + 4 * x;
        fill_pixel(buffer, idx, color);
    });
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
