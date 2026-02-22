use std::{fs::File, io::BufReader, path::Path};

use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::{
    engine::{Rasterizer, Vertex},
    settings::{
        ADD_POLYGON_PROB, ADJACENT_SWAP_PROB, CLONE_POLYGON_PROB, MAX_POLYGONS_PER_IMAGE,
        MERGE_POLYGON_PROB, MIN_POLYGONS_PER_IMAGE, NEW_POINT_MAX_DISTANCE, REMOVE_POLYGON_PROB,
        REORDER_POLYGON_PROB, START_WITH_POLYGONS_PER_IMAGE, SWAP_COLORS_PROB,
    },
    utils::{fill_shape, fill_triangle, randomf32, randomf32_clamped, translate_color, translate_coord},
};

use super::polygon::Polygon;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Drawing {
    pub polygons: Vec<Polygon>,
    pub is_dirty: bool,
    pub fitness: f32,
}

impl Drawing {
    pub fn draw(&self, buffer: &mut [u8], w: usize, h: usize, rm: Rasterizer) {
        if rm == Rasterizer::GPU {
            panic!("should have been drawn on the GPU");
        }

        // start with white background
        buffer.fill(255u8);

        for polygon in &self.polygons {
            if rm == Rasterizer::Scanline || polygon.points.len() > 3 {
                fill_shape(buffer, polygon, w, h);
            } else {
                let _ = fill_triangle(buffer, polygon, w, h);
            }
        }
    }

    pub fn num_points(&self) -> usize {
        self.polygons
            .iter()
            .fold(0, |sum, polygon| sum + polygon.num_points())
    }

    pub fn new_random() -> Drawing {
        Drawing {
            polygons: (0..START_WITH_POLYGONS_PER_IMAGE)
                .map(|_| Polygon::new_random())
                .collect(),
            is_dirty: true,
            fitness: 0.0,
        }
    }

    /// Create a random drawing respecting the given polygon count limit.
    pub fn new_random_capped(max_polygons: usize) -> Drawing {
        let count = START_WITH_POLYGONS_PER_IMAGE.min(max_polygons);
        Drawing {
            polygons: (0..count)
                .map(|_| Polygon::new_random())
                .collect(),
            is_dirty: true,
            fitness: 0.0,
        }
    }

    pub fn mutate(&mut self) {
        if randomf32() < ADD_POLYGON_PROB && self.add_polygon() {
            self.is_dirty = true;
        }

        if randomf32() < REMOVE_POLYGON_PROB && self.remove_polygon() {
            self.is_dirty = true;
        }

        if randomf32() < REORDER_POLYGON_PROB && self.reorder_polygons() {
            self.is_dirty = true;
        }

        if randomf32() < ADJACENT_SWAP_PROB && self.adjacent_swap() {
            self.is_dirty = true;
        }

        if randomf32() < MERGE_POLYGON_PROB && self.merge_polygons() {
            self.is_dirty = true;
        }

        if randomf32() < CLONE_POLYGON_PROB && self.clone_jitter() {
            self.is_dirty = true;
        }

        if randomf32() < SWAP_COLORS_PROB && self.swap_colors() {
            self.is_dirty = true;
        }

        let mut internal_mutation_happened = false;
        self.polygons.iter_mut().for_each(|p| {
            internal_mutation_happened = p.mutate();
        });

        if internal_mutation_happened {
            self.is_dirty = true;
        }
    }

    pub fn add_polygon(&mut self) -> bool {
        if self.polygons.len() >= MAX_POLYGONS_PER_IMAGE {
            return false;
        }
        let polygon = Polygon::new_random();
        let index = rand::rng().random_range(0..self.polygons.len() - 1);
        self.polygons.insert(index, polygon);
        true
    }

    pub fn remove_polygon(&mut self) -> bool {
        if self.polygons.is_empty() {
            return false;
        }
        if self.polygons.len() <= MIN_POLYGONS_PER_IMAGE {
            return false;
        }
        let index = rand::rng().random_range(0..self.polygons.len() - 1);
        self.polygons.remove(index);
        true
    }

    pub fn reorder_polygons(&mut self) -> bool {
        let l = self.polygons.len();
        if l < 2 {
            return false;
        }
        let i1 = rand::rng().random_range(0..l);
        let mut i2 = rand::rng().random_range(0..l);
        while i1 == i2 {
            i2 = rand::rng().random_range(0..l);
        }
        self.polygons.swap(i1, i2);
        true
    }

    pub fn adjacent_swap(&mut self) -> bool {
        let l = self.polygons.len();
        if l < 2 {
            return false;
        }
        let i = rand::rng().random_range(0..l);
        let j = if i == l - 1 { i - 1 } else { i + 1 };
        self.polygons.swap(i, j);
        true
    }

    pub fn merge_polygons(&mut self) -> bool {
        let l = self.polygons.len();
        if l < 2 || l <= MIN_POLYGONS_PER_IMAGE {
            return false;
        }
        let i = rand::rng().random_range(0..l);
        let mut j = rand::rng().random_range(0..l);
        while i == j {
            j = rand::rng().random_range(0..l);
        }

        // Check if centroids are close and colors are similar
        let centroid_a = polygon_centroid(&self.polygons[i]);
        let centroid_b = polygon_centroid(&self.polygons[j]);
        let dist = (centroid_a.0 - centroid_b.0).abs() + (centroid_a.1 - centroid_b.1).abs();
        if dist > 0.15 {
            return false;
        }

        let ca = &self.polygons[i].color;
        let cb = &self.polygons[j].color;
        let color_dist = (ca.r as i32 - cb.r as i32).abs()
            + (ca.g as i32 - cb.g as i32).abs()
            + (ca.b as i32 - cb.b as i32).abs();
        if color_dist > 50 {
            return false;
        }

        // Remove the polygon with smaller area
        let area_i = polygon_area(&self.polygons[i]);
        let area_j = polygon_area(&self.polygons[j]);
        let remove = if area_i < area_j { i } else { j };
        self.polygons.swap_remove(remove);
        true
    }

    pub fn clone_jitter(&mut self) -> bool {
        if self.polygons.is_empty() || self.polygons.len() >= MAX_POLYGONS_PER_IMAGE {
            return false;
        }
        let src = rand::rng().random_range(0..self.polygons.len());
        let mut clone = self.polygons[src].clone();

        // Jitter position: small offset to all points
        let d = NEW_POINT_MAX_DISTANCE;
        let dx = randomf32_clamped(-d, d);
        let dy = randomf32_clamped(-d, d);
        for p in &mut clone.points {
            p.x = (p.x + dx).clamp(0.0, 1.0);
            p.y = (p.y + dy).clamp(0.0, 1.0);
        }

        // Jitter color: ±5 per channel
        clone.color.r = (clone.color.r as i16 + rand::rng().random_range(-5..=5)).clamp(0, 255) as u8;
        clone.color.g = (clone.color.g as i16 + rand::rng().random_range(-5..=5)).clamp(0, 255) as u8;
        clone.color.b = (clone.color.b as i16 + rand::rng().random_range(-5..=5)).clamp(0, 255) as u8;

        // Insert near the source in z-order
        let insert_at = (src + 1).min(self.polygons.len());
        self.polygons.insert(insert_at, clone);
        true
    }

    pub fn swap_colors(&mut self) -> bool {
        let l = self.polygons.len();
        if l < 2 {
            return false;
        }
        let i = rand::rng().random_range(0..l);
        let mut j = rand::rng().random_range(0..l);
        while i == j {
            j = rand::rng().random_range(0..l);
        }
        let tmp = self.polygons[i].color;
        self.polygons[i].color = self.polygons[j].color;
        self.polygons[j].color = tmp;
        true
    }

    pub fn from_file(path: &str) -> Self {
        let file = BufReader::new(File::open(Path::new(path)).expect("Failed to open file"));
        serde_json::from_reader(file).unwrap_or_else(|_| panic!("Failed to read file: {}", path))
    }

    pub fn to_file(&self, path: &str) {
        let file = match File::create(Path::new(path)) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[Drawing] Failed to create file '{}': {}", path, e);
                return;
            }
        };
        if let Err(e) = serde_json::to_writer(file, &self) {
            eprintln!("[Drawing] Failed to write to '{}': {}", path, e);
        }
    }

    // for gpu rendering
    pub fn to_vertices(&self) -> Vec<Vertex> {
        // FIXME: hacky workaround --> 2 white triangles as background seems to fix blending issues
        let mut background = vec![
            Vertex {
                position: [
                    translate_coord(0.0f32),
                    translate_coord(0.0f32),
                    0.0f32,
                    1.0f32,
                ],
                color: [1.0f32, 1.0f32, 1.0f32, 1.0f32],
            },
            Vertex {
                position: [
                    translate_coord(1.0f32),
                    translate_coord(0.0f32),
                    0.0f32,
                    1.0f32,
                ],
                color: [1.0f32, 1.0f32, 1.0f32, 1.0f32],
            },
            Vertex {
                position: [
                    translate_coord(1.0f32),
                    translate_coord(1.0f32),
                    0.0f32,
                    1.0f32,
                ],
                color: [1.0f32, 1.0f32, 1.0f32, 1.0f32],
            },
            Vertex {
                position: [
                    translate_coord(0.0f32),
                    translate_coord(0.0f32),
                    0.0f32,
                    1.0f32,
                ],
                color: [1.0f32, 1.0f32, 1.0f32, 1.0f32],
            },
            Vertex {
                position: [
                    translate_coord(0.0f32),
                    translate_coord(1.0f32),
                    0.0f32,
                    1.0f32,
                ],
                color: [1.0f32, 1.0f32, 1.0f32, 1.0f32],
            },
            Vertex {
                position: [
                    translate_coord(1.0f32),
                    translate_coord(1.0f32),
                    0.0f32,
                    1.0f32,
                ],
                color: [1.0f32, 1.0f32, 1.0f32, 1.0f32],
            },
        ];

        let vert: Vec<Vertex> = self
            .clone()
            .polygons
            .into_iter()
            .flat_map(|pp| {
                let arr: Vec<Vertex> = pp
                    .points
                    .into_iter()
                    .map(|p| Vertex {
                        position: [
                            translate_coord(p.x),
                            translate_coord(1.0 - p.y),
                            0.0f32,
                            1.0f32,
                        ],
                        color: [
                            translate_color(pp.color.r),
                            translate_color(pp.color.g),
                            translate_color(pp.color.b),
                            translate_color(pp.color.a),
                        ],
                    })
                    .collect();
                arr
            })
            .collect();

        background.extend(vert);
        background
    }
}

fn polygon_centroid(p: &Polygon) -> (f32, f32) {
    let n = p.points.len() as f32;
    let cx = p.points.iter().map(|pt| pt.x).sum::<f32>() / n;
    let cy = p.points.iter().map(|pt| pt.y).sum::<f32>() / n;
    (cx, cy)
}

fn polygon_area(p: &Polygon) -> f32 {
    // Shoelace formula for arbitrary polygon area
    let pts = &p.points;
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].x * pts[j].y;
        area -= pts[j].x * pts[i].y;
    }
    area.abs() / 2.0
}

impl From<String> for Drawing {
    fn from(json: String) -> Self {
        serde_json::from_str(&json).unwrap_or_else(|_| panic!("Expected deserializable Drawing.\n{}", json))
    }
}

impl Default for Drawing {
    fn default() -> Self {
        Self::new_random()
    }
}
