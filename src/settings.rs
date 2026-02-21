// Re-export shared settings so existing `use crate::settings::*` keeps working
pub use artgen_shared::settings::*;

// Backend-only settings
pub const DEBUG_TIMERS: bool = false;

pub const MIN_IMAGE_WIDTH: usize = 256;
pub const MIN_IMAGE_HEIGHT: usize = 256;

pub const MAX_IMAGE_WIDTH: usize = 512;
pub const MAX_IMAGE_HEIGHT: usize = 512;

pub const DISPLAY_W: u32 = 1024;
pub const DISPLAY_H: u32 = 1024;

pub const FPS_TARGET: u64 = 30;
pub const TARGET_FRAMETIME: u64 = (1000.0 / FPS_TARGET as f32) as u64;

pub const MAX_ERROR_PER_PIXEL: f32 = 441.67297; // sqrt(255.0 * 255.0 * 3.0) — L2 distance
pub const GPU_MAX_ERROR_PER_PIXEL: f32 = 765.0; // L1 distance: abs(dr) + abs(dg) + abs(db), max = 255 * 3
pub const PER_POINT_MULTIPLIER: f32 = 1.0 / 5000000.0;

pub const MIN_POINTS_PER_POLYGON: usize = 3;
pub const START_WITH_POLYGONS_PER_IMAGE: usize = 150;

// Tile culling settings
pub const TILE_MAX_POLYS: u32 = 256; // max polygon indices per tile per offspring
