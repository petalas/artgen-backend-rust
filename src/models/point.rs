use serde::{Deserialize, Serialize};

use crate::{
    settings::{
        MICRO_ADJUSTMENT_DELTA, MICRO_ADJUSTMENT_PROBABILITY, MOVE_POINT_MAX_DELTA,
        MOVE_POINT_PROBABILITY,
    },
    utils::{randomf32, randomf32_clamped},
};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new_random() -> Point {
        Point {
            x: randomf32(),
            y: randomf32(),
        }
    }

    pub fn offset(&mut self, x_offset: f32, y_offset: f32) {
        self.x = (self.x + x_offset).clamp(0.0, 1.0);
        self.y = (self.y + y_offset).clamp(0.0, 1.0);
    }

    pub fn mutate(&mut self) -> bool {
        let mut mutated = false;
        if randomf32() < MOVE_POINT_PROBABILITY {
            let d = MOVE_POINT_MAX_DELTA;
            self.x = randomf32_clamped(self.x - d, self.x + d).clamp(0.0, 1.0);
            self.y = randomf32_clamped(self.y - d, self.y + d).clamp(0.0, 1.0);
            mutated = true;
        }

        if randomf32() < MICRO_ADJUSTMENT_PROBABILITY {
            let d = MICRO_ADJUSTMENT_DELTA;
            self.x = randomf32_clamped(self.x - d, self.x + d).clamp(0.0, 1.0);
            self.y = randomf32_clamped(self.y - d, self.y + d).clamp(0.0, 1.0);
            mutated = true;
        }
        mutated
    }

    pub fn translate(&self, w: usize, h: usize) -> Point {
        let x = (self.x * w as f32).round();
        let y = (self.y * h as f32).round();
        assert!(x >= 0.0 && x <= w as f32 && y >= 0.0 && y <= h as f32);
        return Point { x, y };
    }
}

impl Eq for Point {}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.x < other.x {
            return std::cmp::Ordering::Less;
        }
        if self.y < other.y {
            return std::cmp::Ordering::Less;
        }
        return std::cmp::Ordering::Equal;
    }
}
