use serde::{Deserialize, Serialize};

/// Mirror of the backend MutationParams struct.
/// Deserialized from WebSocket JSON (camelCase keys).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MutationParams {
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

    pub move_point_max_delta: f32,
    pub micro_adjust_delta: f32,
    pub new_point_max_distance: f32,
    pub offset_polygon_magnitude: f32,

    pub min_alpha: u8,
    pub max_alpha: u8,

    pub min_polygons: u32,
    pub max_polygons: u32,

    // Crossover & island parameters
    pub crossover_prob: f32,
    pub spatial_crossover_weight: f32,
    pub tournament_size: u32,
    pub island_count: u32,
    pub inter_island_interval: u32,

    // Chain count
    pub chain_count: u32,
}

impl Default for MutationParams {
    fn default() -> Self {
        Self {
            add_polygon_prob: 1.0 / 50.0,
            remove_polygon_prob: 1.0 / 1500.0,
            reorder_polygon_prob: 1.0 / 500.0,
            offset_polygon_prob: 1.0 / 500.0,
            move_point_prob: 1.0 / 500.0,
            remove_point_prob: 1.0 / 500.0,
            micro_adjust_prob: 1.0 / 100.0,
            change_color_prob: 1.0 / 750.0,
            lighten_color_prob: 1.0 / 750.0,
            darken_color_prob: 1.0 / 750.0,
            move_point_max_delta: 0.1,
            micro_adjust_delta: 0.01,
            new_point_max_distance: 0.03,
            offset_polygon_magnitude: 0.1,
            min_alpha: 10,
            max_alpha: 65,
            min_polygons: 1,
            max_polygons: 1000,
            crossover_prob: 0.1,
            spatial_crossover_weight: 0.7,
            tournament_size: 3,
            island_count: 8,
            inter_island_interval: 500,
            chain_count: 128,
        }
    }
}
