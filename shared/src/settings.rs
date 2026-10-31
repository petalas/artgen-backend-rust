// Mutation probability defaults
pub const ADD_POLYGON_PROB: f32 = 0.5;
// Off: crossover pairs chains slot by slot, and an accepted removal shifts every
// slot above it, so the chain no longer lines up with the others (late-game
// evaluations halved and quality fell once selection let removals win;
// docs/quality-benchmark.md). Was 0.000862.
pub const REMOVE_POLYGON_PROB: f32 = 0.0;
pub const REORDER_POLYGON_PROB: f32 = 0.002;
pub const OFFSET_POLYGON_PROBABILITY: f32 = 0.000565;
pub const MOVE_POINT_PROBABILITY: f32 = 0.000294;
pub const REMOVE_POINT_PROBABILITY: f32 = 0.002;
// Micro color, saturation and brightness cut, their weight moved to micro moves
// and color changes (docs/quality-benchmark.md, operator weight A/B).
pub const MICRO_ADJUSTMENT_PROBABILITY: f32 = 0.00433;
pub const MICRO_COLOR_PROBABILITY: f32 = 0.000697;
// Micro color's weight when settings predate the field: it shared
// micro_adjust_prob's value, 0.002786 by default, until they were split.
pub const LEGACY_MICRO_COLOR_PROBABILITY: f32 = 0.002786;
pub const CHANGE_COLOR_PROB: f32 = 0.001631;
pub const ADJUST_BRIGHTNESS_PROB: f32 = 0.000237;
pub const ADJUST_SATURATION_PROB: f32 = 0.000667;

pub const SCALE_POLYGON_PROB: f32 = 0.005650;
pub const ROTATE_POLYGON_PROB: f32 = 0.002;
pub const ADJACENT_SWAP_PROB: f32 = 0.002;
// Off for the reason REMOVE_POLYGON_PROB is. Was 0.000667.
pub const MERGE_POLYGON_PROB: f32 = 0.0;
pub const CLONE_POLYGON_PROB: f32 = 0.071429;
pub const MEDIUM_MOVE_PROBABILITY: f32 = 0.001;
pub const SWAP_COLORS_PROB: f32 = 0.005650;
// Absorb (delete a triangle into a nearby larger one's colour) is off by default.
pub const ABSORB_POLYGON_PROB: f32 = 0.0;

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
/// Default `MutationParams::penalty_exponent`: every triangle costs the same,
/// `3 * per_point_penalty`.
pub const PENALTY_EXPONENT: f32 = 0.0;
/// Triangle count at which a scaled penalty's next triangle costs exactly
/// `3 * per_point_penalty`. With exponent `k` the next triangle costs
/// `3 p (n / N0)^k`, so `n` triangles remove the fraction
/// `3 p N0 / (k + 1) * (n / N0)^(k + 1)` of the fitness.
pub const PENALTY_REFERENCE_TRIANGLES: u32 = 1000;
/// Largest accepted `penalty_exponent`. With `MAX_PER_POINT_PENALTY` the
/// fraction at `POLYGON_CAPACITY` triangles grows with the exponent above
/// about 0.4; at 2 it removes 86% of the fitness, so fitness stays positive.
pub const MAX_PENALTY_EXPONENT: f32 = 2.0;

// GPU evolution settings
pub const GPU_MAX_CHAIN_COUNT: u32 = 1024;
pub const GPU_DEFAULT_CHAIN_COUNT: u32 = 4;
pub const GPU_DEFAULT_LAMBDA: u32 = 64;

// Longest side, in pixels, the engine resizes a project's original image to
// unless the viewer picks another resolution.
pub const DEFAULT_TARGET_RESOLUTION: u32 = 512;
pub const GPU_MAX_LAMBDA: u32 = 64;
pub const GPU_DEFAULT_BATCH_ITERS: u32 = 64;
pub const GPU_MAX_BATCH_ITERS: u32 = 4096;

// Crossover defaults
pub const CROSSOVER_PROB: f32 = 0.2;
pub const TOURNAMENT_SIZE: u32 = 3;

// Step spread
/// Step-size rungs: offspring with local index `i` scale offset, point-move and
/// micro vertex step sizes by rung `i % STEP_RUNGS`'s multiplier, log-spaced
/// from `MutationParams::step_spread_min` to `step_spread_max`.
pub const STEP_RUNGS: usize = 8;
/// A spread end of 1 on both sides: every rung is 1, no spread. Serialized
/// params omit an end at this value and read a missing end as it, so records,
/// fixtures and saved settings from before the step spread keep their meaning.
pub const STEP_SPREAD_NONE: f32 = 1.0;
/// Default spread: rungs from 0.25x to 4x (docs/quality-benchmark.md, step spread).
pub const STEP_SPREAD_MIN_DEFAULT: f32 = 0.25;
pub const STEP_SPREAD_MAX_DEFAULT: f32 = 4.0;
/// `sanitize` keeps both spread ends within [1 / STEP_SPREAD_LIMIT, STEP_SPREAD_LIMIT].
pub const STEP_SPREAD_LIMIT: f32 = 16.0;

// Rasterize workgroup size defaults
pub const RASTERIZE_WG_X_DEFAULT: u32 = 32;
pub const RASTERIZE_WG_Y_DEFAULT: u32 = 16;
