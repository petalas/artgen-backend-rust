use serde::{Deserialize, Serialize};

use crate::settings;

/// Runtime-configurable mutation parameters.
/// Sent over WebSocket as JSON (camelCase) and used to build `GpuParams`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MutationParams {
    // Probabilities (expressed as fraction, e.g. 0.02 = 1 in 50)
    pub add_polygon_prob: f32,
    pub remove_polygon_prob: f32,
    pub reorder_polygon_prob: f32,
    pub offset_polygon_prob: f32,
    pub move_point_prob: f32,
    pub remove_point_prob: f32,
    pub micro_adjust_prob: f32,
    pub change_color_prob: f32,
    pub lighten_color_prob: f32,
    pub darken_color_prob: f32,

    // Deltas / magnitudes
    pub move_point_max_delta: f32,
    pub micro_adjust_delta: f32,
    pub new_point_max_distance: f32,
    pub offset_polygon_magnitude: f32,

    // Alpha range
    pub min_alpha: u8,
    pub max_alpha: u8,

    // Polygon count limits
    pub min_polygons: u32,
    pub max_polygons: u32,

    // Crossover & island parameters
    pub crossover_prob: f32,
    pub spatial_crossover_weight: f32,
    pub tournament_size: u32,
    pub island_count: u32,
    pub inter_island_interval: u32,
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
        self.lighten_color_prob = self.lighten_color_prob.clamp(0.0, 1.0);
        self.darken_color_prob = self.darken_color_prob.clamp(0.0, 1.0);

        // Deltas: non-negative
        self.move_point_max_delta = self.move_point_max_delta.max(0.0);
        self.micro_adjust_delta = self.micro_adjust_delta.max(0.0);
        self.new_point_max_distance = self.new_point_max_distance.max(0.0);
        self.offset_polygon_magnitude = self.offset_polygon_magnitude.max(0.0);

        // Crossover & island parameters
        self.crossover_prob = self.crossover_prob.clamp(0.0, 1.0);
        self.spatial_crossover_weight = self.spatial_crossover_weight.clamp(0.0, 1.0);
        self.tournament_size = self.tournament_size.clamp(1, 16);
        // Snap island_count to nearest divisor of GPU_CHAIN_COUNT
        let chain_count = settings::GPU_CHAIN_COUNT;
        self.island_count = self.island_count.clamp(1, chain_count);
        if self.island_count > 1 {
            // Find nearest divisor of chain_count
            let target = self.island_count;
            let mut best = 1u32;
            let mut best_dist = target.abs_diff(1);
            let mut d = 2u32;
            while d * d <= chain_count {
                if chain_count % d == 0 {
                    let dist_d = target.abs_diff(d);
                    if dist_d < best_dist {
                        best = d;
                        best_dist = dist_d;
                    }
                    let complement = chain_count / d;
                    let dist_c = target.abs_diff(complement);
                    if dist_c < best_dist {
                        best = complement;
                        best_dist = dist_c;
                    }
                }
                d += 1;
            }
            // Also check chain_count itself as a divisor
            let dist_cc = target.abs_diff(chain_count);
            if dist_cc < best_dist {
                best = chain_count;
            }
            self.island_count = best;
        }
        self.inter_island_interval = self.inter_island_interval.clamp(1, 10000);
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
            lighten_color_prob: settings::LIGHTEN_COLOR_PROB,
            darken_color_prob: settings::DARKEN_COLOR_PROB,
            move_point_max_delta: settings::MOVE_POINT_MAX_DELTA,
            micro_adjust_delta: settings::MICRO_ADJUSTMENT_DELTA,
            new_point_max_distance: settings::NEW_POINT_MAX_DISTANCE,
            offset_polygon_magnitude: settings::OFFSET_POLYGON_MAGNITUDE,
            min_alpha: settings::MIN_ALPHA,
            max_alpha: settings::MAX_ALPHA,
            min_polygons: settings::MIN_POLYGONS_PER_IMAGE as u32,
            max_polygons: settings::MAX_POLYGONS_PER_IMAGE as u32,
            crossover_prob: settings::CROSSOVER_PROB,
            spatial_crossover_weight: settings::SPATIAL_CROSSOVER_WEIGHT,
            tournament_size: settings::TOURNAMENT_SIZE,
            island_count: settings::ISLAND_COUNT,
            inter_island_interval: settings::INTER_ISLAND_INTERVAL,
        }
    }
}
