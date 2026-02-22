use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::{
    settings::{
        MIN_POINTS_PER_POLYGON, NEW_POINT_MAX_DISTANCE, OFFSET_POLYGON_MAGNITUDE,
        OFFSET_POLYGON_PROBABILITY, REMOVE_POINT_PROBABILITY, ROTATE_POLYGON_PROB,
        SCALE_POLYGON_PROB,
    },
    utils::randomf32_clamped,
};

use super::{color::Color, point::Point};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Polygon {
    pub points: Vec<Point>,
    pub color: Color,
}

impl Polygon {
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn new_random() -> Polygon {
        let origin: Point = Point::new_random();
        let d = NEW_POINT_MAX_DISTANCE;
        let points = (0..3)
            .map(|_| {
                let x = randomf32_clamped(origin.x - d, origin.x + d).clamp(0.0, 1.0);
                let y = randomf32_clamped(origin.y - d, origin.y + d).clamp(0.0, 1.0);
                Point { x, y }
            })
            .collect();
        Polygon {
            points,
            color: Color::new_random(),
        }
    }

    fn offset_polygon(&mut self) -> bool {
        if self.points.len() < 3 {
            return false;
        }

        let x_offset = randomf32_clamped(-OFFSET_POLYGON_MAGNITUDE, OFFSET_POLYGON_MAGNITUDE);
        let y_offset = randomf32_clamped(-OFFSET_POLYGON_MAGNITUDE, OFFSET_POLYGON_MAGNITUDE);
        self.points
            .iter_mut()
            .for_each(|point| point.offset(x_offset, y_offset));

        true
    }

    fn scale_polygon(&mut self) -> bool {
        if self.points.len() < 3 {
            return false;
        }
        let n = self.points.len() as f32;
        let cx = self.points.iter().map(|p| p.x).sum::<f32>() / n;
        let cy = self.points.iter().map(|p| p.y).sum::<f32>() / n;
        let scale = randomf32_clamped(0.8, 1.2);
        for p in &mut self.points {
            p.x = (cx + (p.x - cx) * scale).clamp(0.0, 1.0);
            p.y = (cy + (p.y - cy) * scale).clamp(0.0, 1.0);
        }
        true
    }

    fn rotate_polygon(&mut self) -> bool {
        if self.points.len() < 3 {
            return false;
        }
        let n = self.points.len() as f32;
        let cx = self.points.iter().map(|p| p.x).sum::<f32>() / n;
        let cy = self.points.iter().map(|p| p.y).sum::<f32>() / n;
        let angle = randomf32_clamped(-0.2618, 0.2618); // ±15°
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        for p in &mut self.points {
            let dx = p.x - cx;
            let dy = p.y - cy;
            p.x = (cx + dx * cos_a - dy * sin_a).clamp(0.0, 1.0);
            p.y = (cy + dx * sin_a + dy * cos_a).clamp(0.0, 1.0);
        }
        true
    }

    fn remove_point(&mut self) -> bool {
        let n = self.points.len();
        if n <= MIN_POINTS_PER_POLYGON {
            return false;
        }
        let i = rand::rng().random_range(0..(n - 1));
        self.points.remove(i);
        true
    }

    pub fn mutate(&mut self) -> bool {
        let mut mutated = false;
        if rand::rng().random::<f32>() < OFFSET_POLYGON_PROBABILITY
            && self.offset_polygon()
        {
            mutated = true;
        }
        if rand::rng().random::<f32>() < SCALE_POLYGON_PROB
            && self.scale_polygon()
        {
            mutated = true;
        }
        if rand::rng().random::<f32>() < ROTATE_POLYGON_PROB
            && self.rotate_polygon()
        {
            mutated = true;
        }
        if rand::rng().random::<f32>() < REMOVE_POINT_PROBABILITY
            && self.remove_point()
        {
            mutated = true;
        }

        if self.color.mutate() {
            mutated = true
        }

        self.points.iter_mut().for_each(|p| {
            if p.mutate() {
                mutated = true;
            }
        });

        mutated
    }
}

impl Default for Polygon {
    fn default() -> Self {
        Self::new_random()
    }
}