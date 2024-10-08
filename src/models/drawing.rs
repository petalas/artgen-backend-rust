use std::{fs::File, io::BufReader, path::Path};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    engine::{Rasterizer, Vertex},
    settings::{
        ADD_POLYGON_PROB, MAX_POLYGONS_PER_IMAGE, MIN_POLYGONS_PER_IMAGE, REMOVE_POLYGON_PROB,
        REORDER_POLYGON_PROB, START_WITH_POLYGONS_PER_IMAGE,
    },
    utils::{fill_shape, fill_triangle, randomf32, translate_color, translate_coord},
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
    pub fn draw(&self, buffer: &mut Vec<u8>, w: usize, h: usize, rm: Rasterizer) {
        if rm == Rasterizer::GPU {
            panic!("should have been drawn on the GPU");
        }

        // start with white background
        buffer.fill(255u8);

        for polygon in &self.polygons {
            if rm == Rasterizer::Scanline || polygon.points.len() > 3 {
                fill_shape(buffer, &polygon, w, h);
            } else {
                fill_triangle(buffer, &polygon, w, h);
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

    pub fn mutate(&mut self) {
        if randomf32() < ADD_POLYGON_PROB {
            if self.add_polygon() {
                self.is_dirty = true;
            }
        }

        if randomf32() < REMOVE_POLYGON_PROB {
            if self.remove_polygon() {
                self.is_dirty = true;
            }
        }

        if randomf32() < REORDER_POLYGON_PROB {
            if self.reorder_polygons() {
                self.is_dirty = true;
            }
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
        let index = rand::thread_rng().gen_range(0..self.polygons.len() - 1);
        self.polygons.insert(index, polygon);
        return true;
    }

    pub fn remove_polygon(&mut self) -> bool {
        if self.polygons.len() < 1 {
            return false;
        }
        if self.polygons.len() <= MIN_POLYGONS_PER_IMAGE {
            return false;
        }
        let index = rand::thread_rng().gen_range(0..self.polygons.len() - 1);
        self.polygons.remove(index);
        return true;
    }

    pub fn reorder_polygons(&mut self) -> bool {
        let l = self.polygons.len();
        if self.polygons.len() < 2 {
            return false;
        }
        let i1 = rand::thread_rng().gen_range(0..l - 1);
        let mut i2 = rand::thread_rng().gen_range(0..l - 1);
        while i1 == i2 {
            i2 = rand::thread_rng().gen_range(0..l - 1);
        }
        self.polygons.swap(i1, i2);
        return true;
    }

    pub fn from_file(path: &str) -> Self {
        let file = BufReader::new(File::open(&Path::new(&path)).expect("Failed to open file"));
        return serde_json::from_reader(file).expect(format!("Failed to read file: {}", path).as_str());
    }

    pub fn to_file(&self, path: &str) {
        // TODO: change it so that if the file already exists, it overwrites it
        let file = File::create(&Path::new(path)).expect("Failed to create file");
        serde_json::to_writer(file, &self).expect("Failed to write to file");
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
            .map(|pp| {
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
            .flatten()
            .collect();

        background.extend(vert);
        background
    }
}

impl From<String> for Drawing {
    fn from(json: String) -> Self {
        serde_json::from_str(&json).expect(&format!("Expected deserializable Drawing.\n{}", json))
    }
}

impl Default for Drawing {
    fn default() -> Self {
        Self::new_random()
    }
}
