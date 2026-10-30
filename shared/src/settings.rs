// Mutation probability defaults
pub const ADD_POLYGON_PROB: f32 = 0.5;
pub const REMOVE_POLYGON_PROB: f32 = 0.000862;
pub const REORDER_POLYGON_PROB: f32 = 0.002;
pub const OFFSET_POLYGON_PROBABILITY: f32 = 0.000565;
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
pub const MEDIUM_MOVE_PROBABILITY: f32 = 0.001;
pub const SWAP_COLORS_PROB: f32 = 0.005650;

// Merge mutation thresholds
pub const MERGE_CENTROID_THRESHOLD: f32 = 0.15;
pub const MERGE_COLOR_THRESHOLD: f32 = 0.20;

// Mutation delta defaults
pub const MOVE_POINT_MAX_DELTA: f32 = 0.113;
pub const MEDIUM_MOVE_DELTA: f32 = 0.03;
pub const MICRO_ADJUSTMENT_DELTA: f32 = 0.002;
pub const NEW_POINT_MAX_DISTANCE: f32 = 0.005;
pub const OFFSET_POLYGON_MAGNITUDE: f32 = 0.010;

// Alpha range
pub const MIN_ALPHA: u8 = 10;
pub const MAX_ALPHA: u8 = 65;

// Polygon count limits
/// Triangles one GPU drawing state can store: the compile-time stride of
/// `GpuDrawingState` and WGSL `DrawingState` (16 bytes per triangle), and the
/// largest `max_polygons` that `MutationParams::sanitize` accepts.
pub const POLYGON_CAPACITY: usize = 2048;
/// The default runtime search cap, `MutationParams::max_polygons`. Mutation adds
/// no triangle beyond it. At most `POLYGON_CAPACITY`.
pub const DEFAULT_MAX_POLYGONS: u32 = 2000;
pub const MIN_POLYGONS_PER_IMAGE: usize = 1;
const _: () = assert!(DEFAULT_MAX_POLYGONS as usize <= POLYGON_CAPACITY);

// Fitness penalty
/// Default `MutationParams::per_point_penalty`: fitness is multiplied by
/// `1 - penalty * points`, three points per triangle.
pub const PER_POINT_PENALTY: f32 = 1.0 / 5_000_000.0;
/// Largest accepted `per_point_penalty`. At `POLYGON_CAPACITY` triangles it
/// removes 61% of the fitness, so fitness stays positive.
pub const MAX_PER_POINT_PENALTY: f32 = 1.0e-4;

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

// Adaptive mutation default
pub const ADAPTIVE_MUTATION: bool = false;

// Step spread
/// Step-size rungs: offspring with local index `i` scale offset, point-move and
/// micro vertex step sizes by rung `i % STEP_RUNGS`'s multiplier, log-spaced
/// from `MutationParams::step_spread_min` to `step_spread_max`.
pub const STEP_RUNGS: usize = 8;
/// Default `step_spread_min` and `step_spread_max`: every rung is 1, no spread.
pub const STEP_SPREAD_DEFAULT: f32 = 1.0;
/// `sanitize` keeps both spread ends within [1 / STEP_SPREAD_LIMIT, STEP_SPREAD_LIMIT].
pub const STEP_SPREAD_LIMIT: f32 = 16.0;

// Rasterize workgroup size defaults
pub const RASTERIZE_WG_X_DEFAULT: u32 = 32;
pub const RASTERIZE_WG_Y_DEFAULT: u32 = 16;
