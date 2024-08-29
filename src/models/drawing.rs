use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    engine::Rasterizer, settings::{
        ADD_POLYGON_PROB, MAX_POLYGONS_PER_IMAGE, MIN_POLYGONS_PER_IMAGE, REMOVE_POLYGON_PROB,
        REORDER_POLYGON_PROB, START_WITH_POLYGONS_PER_IMAGE,
    }, utils::{fill_shape, fill_triangle, randomf32}
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
        // start with white background
        buffer.into_iter().for_each(|item| *item = 255u8);

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
}

impl From<String> for Drawing {
    fn from(json: String) -> Self {
        serde_json::from_str(&json).expect(&format!("Expected deserializable Drawing.\n{}", json))
    }
}
