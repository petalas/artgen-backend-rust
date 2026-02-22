// Mutation probability defaults
pub const ADD_POLYGON_PROB: f32 = 0.5;
pub const REMOVE_POLYGON_PROB: f32 = 0.000862;
pub const REORDER_POLYGON_PROB: f32 = 0.002;
pub const OFFSET_POLYGON_PROBABILITY: f32 = 0.005650;
pub const MOVE_POINT_PROBABILITY: f32 = 0.000294;
pub const REMOVE_POINT_PROBABILITY: f32 = 0.002;
pub const MICRO_ADJUSTMENT_PROBABILITY: f32 = 0.002786;
pub const CHANGE_COLOR_PROB: f32 = 0.000473;
pub const ADJUST_BRIGHTNESS_PROB: f32 = 0.000473;
pub const ADJUST_SATURATION_PROB: f32 = 0.001333;

pub const SCALE_POLYGON_PROB: f32 = 0.005650;
pub const ROTATE_POLYGON_PROB: f32 = 0.002;
pub const ADJACENT_SWAP_PROB: f32 = 0.002;
pub const MERGE_POLYGON_PROB: f32 = 0.000667;
pub const CLONE_POLYGON_PROB: f32 = 0.071429;
pub const MEDIUM_MOVE_PROBABILITY: f32 = 0.004;
pub const SWAP_COLORS_PROB: f32 = 0.005650;

// Merge mutation thresholds
pub const MERGE_CENTROID_THRESHOLD: f32 = 0.15;
pub const MERGE_COLOR_THRESHOLD: f32 = 0.20;

// Mutation delta defaults
pub const MOVE_POINT_MAX_DELTA: f32 = 0.113;
pub const MEDIUM_MOVE_DELTA: f32 = 0.03;
pub const MICRO_ADJUSTMENT_DELTA: f32 = 0.001;
pub const NEW_POINT_MAX_DISTANCE: f32 = 0.005;
pub const OFFSET_POLYGON_MAGNITUDE: f32 = 0.010;

// Alpha range
pub const MIN_ALPHA: u8 = 10;
pub const MAX_ALPHA: u8 = 65;

// Polygon count limits
pub const MAX_POLYGONS_PER_IMAGE: usize = 1000;
pub const MIN_POLYGONS_PER_IMAGE: usize = 1;

// GPU evolution settings
pub const GPU_MAX_CHAIN_COUNT: u32 = 1024;
pub const GPU_DEFAULT_CHAIN_COUNT: u32 = 4;
pub const GPU_DEFAULT_LAMBDA: u32 = 64;
pub const GPU_MAX_LAMBDA: u32 = 64;
pub const GPU_DEFAULT_BATCH_ITERS: u32 = 64;
pub const GPU_MAX_BATCH_ITERS: u32 = 4096;

// Crossover defaults
pub const CROSSOVER_PROB: f32 = 0.2;
pub const SPATIAL_CROSSOVER_WEIGHT: f32 = 0.66;
pub const TOURNAMENT_SIZE: u32 = 3;

// Mutation mode defaults
pub const SINGLE_MUTATION_MODE: bool = true;
pub const ADAPTIVE_MUTATION: bool = false;

// Integer AABB: early rejection using packed u32 vertex data before float unpack
pub const INTEGER_AABB: bool = true;

// Rasterize workgroup size defaults
pub const RASTERIZE_WG_X_DEFAULT: u32 = 32;
pub const RASTERIZE_WG_Y_DEFAULT: u32 = 16;
