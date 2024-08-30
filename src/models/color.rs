use crate::settings::{
    CHANGE_COLOR_PROB, DARKEN_COLOR_PROB, LIGHTEN_COLOR_PROB, MAX_ALPHA,
    MICRO_ADJUSTMENT_PROBABILITY, MIN_ALPHA,
};
use crate::utils::{randomf32, randomu8};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[repr(C)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

pub const BLACK: Color = Color {
    r: 0,
    g: 0,
    b: 0,
    a: 0,
};

pub const WHITE: Color = Color {
    r: 255,
    g: 255,
    b: 255,
    a: 255,
};

pub const RED: Color = Color {
    r: 255,
    g: 0,
    b: 0,
    a: 255,
};

impl Color {
    pub fn new_random() -> Color {
        Color {
            r: randomu8(),
            g: randomu8(),
            b: randomu8(),
            a: randomu8().clamp(MIN_ALPHA, MAX_ALPHA),
        }
    }

    pub fn mutate(&mut self) -> bool {
        let mut mutation_happened = false;

        if randomf32() < CHANGE_COLOR_PROB {
            self.r = randomu8();
            mutation_happened = true;
        }
        if randomf32() < CHANGE_COLOR_PROB {
            self.g = randomu8();
            mutation_happened = true;
        }
        if randomf32() < CHANGE_COLOR_PROB {
            self.b = randomu8();
            mutation_happened = true;
        }
        if randomf32() < CHANGE_COLOR_PROB {
            self.a = randomu8().clamp(MIN_ALPHA, MAX_ALPHA);
            mutation_happened = true;
        }

        //// same but micro adjustments
        if randomf32() < MICRO_ADJUSTMENT_PROBABILITY {
            self.r = Color::micro_adjust(self.r);
            mutation_happened = true;
        }
        if randomf32() < MICRO_ADJUSTMENT_PROBABILITY {
            self.g = Color::micro_adjust(self.g);
            mutation_happened = true;
        }
        if randomf32() < MICRO_ADJUSTMENT_PROBABILITY {
            self.b = Color::micro_adjust(self.b);
            mutation_happened = true;
        }
        if randomf32() < MICRO_ADJUSTMENT_PROBABILITY {
            self.a = Color::micro_adjust(self.a).clamp(MIN_ALPHA, MAX_ALPHA);
            mutation_happened = true;
        }
        ////
        if randomf32() < LIGHTEN_COLOR_PROB {
            if self.r < u8::MAX && self.g < u8::MAX && self.b < u8::MAX {
                self.r += 1;
                self.g += 1;
                self.b += 1;
                mutation_happened = true;
            }
        }
        if randomf32() < DARKEN_COLOR_PROB {
            if self.r > u8::MIN && self.g > u8::MIN && self.b > u8::MIN {
                self.r -= 1;
                self.g -= 1;
                self.b -= 1;
                mutation_happened = true;
            }
        }

        mutation_happened
    }

    // increment or decrement with 50% chance while avoiding overflows and underflows
    fn micro_adjust(mut val: u8) -> u8 {
        val = if randomf32() > 0.5 {
            if val < u8::MAX {
                val + 1
            } else {
                val - 1
            }
        } else {
            if val > u8::MIN {
                val - 1
            } else {
                val + 1
            }
        };
        val
    }
}

impl From<&[u8]> for Color {
    fn from(value: &[u8]) -> Self {
        Color {
            r: value[0],
            g: value[1],
            b: value[2],
            a: value[3],
        }
    }
}
