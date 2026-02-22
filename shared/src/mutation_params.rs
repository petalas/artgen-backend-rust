use serde::{Deserialize, Serialize};

use crate::settings;

fn default_rasterize_wg() -> [u32; 2] {
    [settings::RASTERIZE_WG_X_DEFAULT, settings::RASTERIZE_WG_Y_DEFAULT]
}

fn default_gpu_batch_iters() -> u32 {
    settings::GPU_DEFAULT_BATCH_ITERS
}

fn default_scale_polygon_prob() -> f32 { settings::SCALE_POLYGON_PROB }
fn default_rotate_polygon_prob() -> f32 { settings::ROTATE_POLYGON_PROB }
fn default_adjacent_swap_prob() -> f32 { settings::ADJACENT_SWAP_PROB }
fn default_merge_polygon_prob() -> f32 { settings::MERGE_POLYGON_PROB }
fn default_clone_polygon_prob() -> f32 { settings::CLONE_POLYGON_PROB }
fn default_medium_move_prob() -> f32 { settings::MEDIUM_MOVE_PROBABILITY }
fn default_swap_colors_prob() -> f32 { settings::SWAP_COLORS_PROB }
fn default_medium_move_delta() -> f32 { settings::MEDIUM_MOVE_DELTA }
fn default_merge_centroid_threshold() -> f32 { settings::MERGE_CENTROID_THRESHOLD }
fn default_merge_color_threshold() -> f32 { settings::MERGE_COLOR_THRESHOLD }
fn default_integer_aabb() -> bool { settings::INTEGER_AABB }


/// Runtime-configurable mutation parameters.
/// Sent over WebSocket as JSON (camelCase) and used to build `GpuParams`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MutationParams {
    // Probabilities (0.0–1.0)
    pub add_polygon_prob: f32,
    pub remove_polygon_prob: f32,
    pub reorder_polygon_prob: f32,
    pub offset_polygon_prob: f32,
    pub move_point_prob: f32,
    pub remove_point_prob: f32,
    pub micro_adjust_prob: f32,
    pub change_color_prob: f32,
    pub adjust_brightness_prob: f32,
    pub adjust_saturation_prob: f32,
    #[serde(default = "default_scale_polygon_prob")]
    pub scale_polygon_prob: f32,
    #[serde(default = "default_rotate_polygon_prob")]
    pub rotate_polygon_prob: f32,
    #[serde(default = "default_adjacent_swap_prob")]
    pub adjacent_swap_prob: f32,
    #[serde(default = "default_merge_polygon_prob")]
    pub merge_polygon_prob: f32,
    #[serde(default = "default_clone_polygon_prob")]
    pub clone_polygon_prob: f32,
    #[serde(default = "default_medium_move_prob")]
    pub medium_move_prob: f32,
    #[serde(default = "default_swap_colors_prob")]
    pub swap_colors_prob: f32,

    // Deltas / magnitudes
    pub move_point_max_delta: f32,
    pub micro_adjust_delta: f32,
    pub new_point_max_distance: f32,
    pub offset_polygon_magnitude: f32,
    #[serde(default = "default_medium_move_delta")]
    pub medium_move_delta: f32,

    // Merge thresholds
    #[serde(default = "default_merge_centroid_threshold")]
    pub merge_centroid_threshold: f32,
    #[serde(default = "default_merge_color_threshold")]
    pub merge_color_threshold: f32,

    // Alpha range
    pub min_alpha: u8,
    pub max_alpha: u8,

    // Polygon count limits
    pub min_polygons: u32,
    pub max_polygons: u32,

    // Crossover parameters
    pub crossover_prob: f32,
    pub spatial_crossover_weight: f32,
    pub tournament_size: u32,

    // Chain count (runtime-configurable, capped to GPU buffer allocation)
    pub chain_count: u32,

    // Lambda: offspring per chain per iteration (1+λ)-ES
    pub lambda: u32,

    // Mutation mode
    pub single_mutation_mode: bool,

    // Adaptive mutation scale (only applies to offspring indices 1..λ-1)
    pub adaptive_mutation: bool,

    // Rasterize workgroup size: [wg_x, wg_y] (e.g. [16,16], [16,8], [8,8])
    // Affects SM occupancy vs shared memory tradeoff. Default: [16,16] = 256 threads.
    #[serde(default = "default_rasterize_wg")]
    pub rasterize_wg: [u32; 2],

    // GPU batch iterations: number of mutate->rasterize->select cycles per GPU submission.
    // Higher values reduce CPU<->GPU round-trip overhead but delay readback/progress reporting.
    #[serde(default = "default_gpu_batch_iters")]
    pub gpu_batch_iters: u32,

    // Tile culling: spatial binning optimization for the rasterize pass.
    // When enabled, a binning pass assigns each polygon to the tiles it overlaps,
    // and the rasterize shader only processes polygons in its tile's list.
    #[serde(default)]
    pub tile_culling: bool,

    // Incremental evaluation: cache the rasterized framebuffer per chain and only
    // re-rasterize the dirty region (bounding box of the changed polygon).
    // Only effective in single_mutation_mode.
    #[serde(default)]
    pub incremental_eval: bool,

    // Integer AABB: early rejection using packed u32 vertex data before float unpack.
    // Skips the expensive float unpack for polygons that fail AABB in integer pixel space.
    #[serde(default = "default_integer_aabb")]
    pub integer_aabb: bool,
}

impl MutationParams {
    /// Enforce constraints: min <= max, values within valid bounds.
    pub fn sanitize(&mut self) {
        // Alpha: min <= max, both within 0..=255 (guaranteed by u8)
        if self.min_alpha > self.max_alpha {
            self.min_alpha = self.max_alpha;
        }

        // Polygons: min <= max, max capped at buffer limit (1000)
        self.max_polygons = self.max_polygons.min(settings::MAX_POLYGONS_PER_IMAGE as u32);
        self.min_polygons = self.min_polygons.max(1).min(self.max_polygons);

        // Probabilities: clamp to [0, 1]
        self.add_polygon_prob = self.add_polygon_prob.clamp(0.0, 1.0);
        self.remove_polygon_prob = self.remove_polygon_prob.clamp(0.0, 1.0);
        self.reorder_polygon_prob = self.reorder_polygon_prob.clamp(0.0, 1.0);
        self.offset_polygon_prob = self.offset_polygon_prob.clamp(0.0, 1.0);
        self.move_point_prob = self.move_point_prob.clamp(0.0, 1.0);
        self.remove_point_prob = self.remove_point_prob.clamp(0.0, 1.0);
        self.micro_adjust_prob = self.micro_adjust_prob.clamp(0.0, 1.0);
        self.change_color_prob = self.change_color_prob.clamp(0.0, 1.0);
        self.adjust_brightness_prob = self.adjust_brightness_prob.clamp(0.0, 1.0);
        self.adjust_saturation_prob = self.adjust_saturation_prob.clamp(0.0, 1.0);
        self.scale_polygon_prob = self.scale_polygon_prob.clamp(0.0, 1.0);
        self.rotate_polygon_prob = self.rotate_polygon_prob.clamp(0.0, 1.0);
        self.adjacent_swap_prob = self.adjacent_swap_prob.clamp(0.0, 1.0);
        self.merge_polygon_prob = self.merge_polygon_prob.clamp(0.0, 1.0);
        self.clone_polygon_prob = self.clone_polygon_prob.clamp(0.0, 1.0);
        self.medium_move_prob = self.medium_move_prob.clamp(0.0, 1.0);
        self.swap_colors_prob = self.swap_colors_prob.clamp(0.0, 1.0);

        // Deltas: non-negative
        self.move_point_max_delta = self.move_point_max_delta.max(0.0);
        self.micro_adjust_delta = self.micro_adjust_delta.max(0.0);
        self.new_point_max_distance = self.new_point_max_distance.max(0.0);
        self.offset_polygon_magnitude = self.offset_polygon_magnitude.max(0.0);
        self.medium_move_delta = self.medium_move_delta.max(0.0);

        // Merge thresholds: non-negative
        self.merge_centroid_threshold = self.merge_centroid_threshold.max(0.0);
        self.merge_color_threshold = self.merge_color_threshold.max(0.0);

        // Chain count: clamp to [1, GPU_MAX_CHAIN_COUNT]
        self.chain_count = self.chain_count.clamp(1, settings::GPU_MAX_CHAIN_COUNT);

        // Lambda: clamp to [1, GPU_MAX_LAMBDA], enforce power-of-2 (round down)
        self.lambda = self.lambda.clamp(1, settings::GPU_MAX_LAMBDA);
        // Round down to nearest power of 2
        self.lambda = 1u32 << self.lambda.ilog2();

        // Crossover parameters
        self.crossover_prob = self.crossover_prob.clamp(0.0, 1.0);
        self.spatial_crossover_weight = self.spatial_crossover_weight.clamp(0.0, 1.0);
        self.tournament_size = self.tournament_size.clamp(1, 16);

        // Rasterize workgroup size: must be one of the supported configurations
        let valid_wg_sizes: &[[u32; 2]] = &[[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];
        if !valid_wg_sizes.contains(&self.rasterize_wg) {
            self.rasterize_wg = [settings::RASTERIZE_WG_X_DEFAULT, settings::RASTERIZE_WG_Y_DEFAULT];
        }

        // GPU batch iterations: clamp to [1, GPU_MAX_BATCH_ITERS], enforce power-of-2
        self.gpu_batch_iters = self.gpu_batch_iters.clamp(1, settings::GPU_MAX_BATCH_ITERS);
        self.gpu_batch_iters = 1u32 << self.gpu_batch_iters.ilog2();
    }
}

impl Default for MutationParams {
    fn default() -> Self {
        Self {
            add_polygon_prob: settings::ADD_POLYGON_PROB,
            remove_polygon_prob: settings::REMOVE_POLYGON_PROB,
            reorder_polygon_prob: settings::REORDER_POLYGON_PROB,
            offset_polygon_prob: settings::OFFSET_POLYGON_PROBABILITY,
            move_point_prob: settings::MOVE_POINT_PROBABILITY,
            remove_point_prob: settings::REMOVE_POINT_PROBABILITY,
            micro_adjust_prob: settings::MICRO_ADJUSTMENT_PROBABILITY,
            change_color_prob: settings::CHANGE_COLOR_PROB,
            adjust_brightness_prob: settings::ADJUST_BRIGHTNESS_PROB,
            adjust_saturation_prob: settings::ADJUST_SATURATION_PROB,
            scale_polygon_prob: settings::SCALE_POLYGON_PROB,
            rotate_polygon_prob: settings::ROTATE_POLYGON_PROB,
            adjacent_swap_prob: settings::ADJACENT_SWAP_PROB,
            merge_polygon_prob: settings::MERGE_POLYGON_PROB,
            clone_polygon_prob: settings::CLONE_POLYGON_PROB,
            medium_move_prob: settings::MEDIUM_MOVE_PROBABILITY,
            swap_colors_prob: settings::SWAP_COLORS_PROB,
            move_point_max_delta: settings::MOVE_POINT_MAX_DELTA,
            micro_adjust_delta: settings::MICRO_ADJUSTMENT_DELTA,
            new_point_max_distance: settings::NEW_POINT_MAX_DISTANCE,
            offset_polygon_magnitude: settings::OFFSET_POLYGON_MAGNITUDE,
            medium_move_delta: settings::MEDIUM_MOVE_DELTA,
            merge_centroid_threshold: settings::MERGE_CENTROID_THRESHOLD,
            merge_color_threshold: settings::MERGE_COLOR_THRESHOLD,
            min_alpha: settings::MIN_ALPHA,
            max_alpha: settings::MAX_ALPHA,
            min_polygons: settings::MIN_POLYGONS_PER_IMAGE as u32,
            max_polygons: settings::MAX_POLYGONS_PER_IMAGE as u32,
            crossover_prob: settings::CROSSOVER_PROB,
            spatial_crossover_weight: settings::SPATIAL_CROSSOVER_WEIGHT,
            tournament_size: settings::TOURNAMENT_SIZE,
            chain_count: settings::GPU_DEFAULT_CHAIN_COUNT,
            lambda: settings::GPU_DEFAULT_LAMBDA,
            single_mutation_mode: settings::SINGLE_MUTATION_MODE,
            adaptive_mutation: settings::ADAPTIVE_MUTATION,
            rasterize_wg: [settings::RASTERIZE_WG_X_DEFAULT, settings::RASTERIZE_WG_Y_DEFAULT],
            gpu_batch_iters: settings::GPU_DEFAULT_BATCH_ITERS,
            tile_culling: true,
            incremental_eval: false,
            integer_aabb: settings::INTEGER_AABB,
        }
    }
}
