// Mutation probability defaults
pub const ADD_POLYGON_PROB: f32 = 1.0 / 50.0;
pub const REMOVE_POLYGON_PROB: f32 = 1.0 / 1500.0;
pub const REORDER_POLYGON_PROB: f32 = 1.0 / 500.0;
pub const OFFSET_POLYGON_PROBABILITY: f32 = 1.0 / 500.0;
pub const MOVE_POINT_PROBABILITY: f32 = 1.0 / 500.0;
pub const REMOVE_POINT_PROBABILITY: f32 = 1.0 / 500.0;
pub const MICRO_ADJUSTMENT_PROBABILITY: f32 = 1.0 / 100.0;
pub const CHANGE_COLOR_PROB: f32 = 1.0 / 750.0;
pub const LIGHTEN_COLOR_PROB: f32 = 1.0 / 750.0;
pub const DARKEN_COLOR_PROB: f32 = 1.0 / 750.0;

// Mutation delta defaults
pub const MOVE_POINT_MAX_DELTA: f32 = 0.1;
pub const MICRO_ADJUSTMENT_DELTA: f32 = 0.01;
pub const NEW_POINT_MAX_DISTANCE: f32 = 0.03;
pub const OFFSET_POLYGON_MAGNITUDE: f32 = 0.1;

// Alpha range
pub const MIN_ALPHA: u8 = 10;
pub const MAX_ALPHA: u8 = 65;

// Polygon count limits
pub const MAX_POLYGONS_PER_IMAGE: usize = 1000;
pub const MIN_POLYGONS_PER_IMAGE: usize = 1;

// GPU evolution settings
pub const GPU_MAX_CHAIN_COUNT: u32 = 1024;
pub const GPU_DEFAULT_CHAIN_COUNT: u32 = 16;
pub const GPU_DEFAULT_LAMBDA: u32 = 8;
pub const GPU_MAX_LAMBDA: u32 = 64;
pub const GPU_DEFAULT_BATCH_ITERS: u32 = 50;
pub const GPU_MAX_BATCH_ITERS: u32 = 4096;

// Crossover defaults
pub const CROSSOVER_PROB: f32 = 0.1;
pub const SPATIAL_CROSSOVER_WEIGHT: f32 = 0.7;
pub const TOURNAMENT_SIZE: u32 = 3;

// Mutation mode defaults
pub const SINGLE_MUTATION_MODE: bool = true;
pub const ADAPTIVE_MUTATION: bool = true;

// Rasterize workgroup size defaults
pub const RASTERIZE_WG_X_DEFAULT: u32 = 16;
pub const RASTERIZE_WG_Y_DEFAULT: u32 = 16;
