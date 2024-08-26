use serde::{Deserialize, Serialize};

use crate::{
    settings::{
        MICRO_ADJUSTMENT_DELTA, MICRO_ADJUSTMENT_PROBABILITY, MOVE_POINT_MAX_DELTA,
        MOVE_POINT_PROBABILITY,
    },
    utils::{randomf64, randomf64_clamped},
};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new_random() -> Point {
        Point {
            x: randomf64(),
            y: randomf64(),
        }
    }

    pub fn offset(&mut self, x_offset: f64, y_offset: f64) {
        self.x = (self.x + x_offset).clamp(0.0, 1.0);
        self.y = (self.y + y_offset).clamp(0.0, 1.0);
    }

    pub fn mutate(&mut self) -> bool {
        let mut mutated = false;
        if randomf64() < MOVE_POINT_PROBABILITY {
            let d = MOVE_POINT_MAX_DELTA;
            self.x = randomf64_clamped(self.x - d, self.x + d).clamp(0.0, 1.0);
            self.y = randomf64_clamped(self.y - d, self.y + d).clamp(0.0, 1.0);
            mutated = true;
        }

        if randomf64() < MICRO_ADJUSTMENT_PROBABILITY {
            let d = MICRO_ADJUSTMENT_DELTA;
            self.x = randomf64_clamped(self.x - d, self.x + d).clamp(0.0, 1.0);
            self.y = randomf64_clamped(self.y - d, self.y + d).clamp(0.0, 1.0);
            mutated = true;
        }
        mutated
    }

    pub fn translate(&self, w: usize, h: usize) -> Point {
        let x = (self.x * w as f64).round();
        let y = (self.y * h as f64).round();
        assert!(x >= 0.0 && x <= w as f64 && y >= 0.0 && y <= h as f64);
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
