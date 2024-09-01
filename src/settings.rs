pub const DEBUG_TIMERS: bool = false;

pub const MAX_IMAGE_WIDTH: usize = 384;
pub const MAX_IMAGE_HEIGHT: usize = 384;

pub const FPS_TARGET :usize = 15;
pub const TARGET_FRAMETIME :usize = (1000.0 / FPS_TARGET as f32) as usize;

pub const MAX_ERROR_PER_PIXEL: f32 = 441.6729559300637; // Math::sqrt(255.0 * 255.0 * 3.0);
pub const PER_POINT_MULTIPLIER: f32 = 1.0 / 5000000.0;
pub const MIN_ALPHA: u8 = 10;
pub const MAX_ALPHA: u8 = 65;
pub const ADD_POLYGON_PROB: f32 = 1.0 / 50.0;
pub const REMOVE_POLYGON_PROB: f32 = 1.0 / 1500.0;
pub const REORDER_POLYGON_PROB: f32 = 1.0 / 500.0;
pub const OFFSET_POLYGON_PROBABILITY: f32 = 1.0 / 500.0;
pub const MOVE_POINT_PROBABILITY: f32 = 1.0 / 500.0;
pub const REMOVE_POINT_PROBABILITY: f32 = 1.0 / 500.0;
pub const MICRO_ADJUSTMENT_PROBABILITY: f32 = 1.0 / 100.0; // move points or shift polygons by just few pixels (useful at higher fitness levels)
pub const CHANGE_COLOR_PROB: f32 = 1.0 / 750.0;
pub const LIGHTEN_COLOR_PROB: f32 = 1.0 / 750.0;
pub const DARKEN_COLOR_PROB: f32 = 1.0 / 750.0;
pub const MOVE_POINT_MAX_DELTA: f32 = 0.1;
pub const MICRO_ADJUSTMENT_DELTA: f32 = 0.01;
pub const NEW_POINT_MAX_DISTANCE: f32 = 0.03;
pub const OFFSET_POLYGON_MAGNITUDE: f32 = 0.1;
pub const MIN_POINTS_PER_POLYGON: usize = 3;
pub const MAX_POLYGONS_PER_IMAGE: usize = 1000;
pub const MIN_POLYGONS_PER_IMAGE: usize = 1;
pub const START_WITH_POLYGONS_PER_IMAGE: usize = 150; // can not go below 2 using range 0..START_WITH_POLYGONS_PER_IMAGE