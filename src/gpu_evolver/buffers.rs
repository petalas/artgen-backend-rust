use bytemuck::{Pod, Zeroable};

use crate::models::color::Color;
use crate::models::drawing::Drawing;
use crate::models::point::Point;
use crate::models::polygon::Polygon;
use crate::mutation_params::MutationParams;
use crate::settings::MAX_POLYGONS_PER_IMAGE;

/// GPU polygon: a single triangle with color.
/// 48 bytes, matching WGSL struct alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPolygon {
    /// RGBA color, normalized 0.0–1.0
    pub color: [f32; 4],   // 16 bytes, offset 0
    /// Triangle vertex 0, normalized 0.0–1.0
    pub v0: [f32; 2],     // 8 bytes, offset 16
    /// Triangle vertex 1
    pub v1: [f32; 2],     // 8 bytes, offset 24
    /// Triangle vertex 2
    pub v2: [f32; 2],     // 8 bytes, offset 32
    /// Padding to 48-byte stride
    pub _pad: [f32; 2],   // 8 bytes, offset 40
}

/// Per-chain drawing state on GPU.
/// polygon_count + fitness + pad + rng_state = 32 bytes header
/// polygons: MAX_POLYGONS_PER_IMAGE * 48 = 48000 bytes
/// Total: 48032 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuDrawingState {
    pub polygon_count: u32,    // offset 0
    pub fitness_bits: u32,     // offset 4 — bitcast f32 for atomicMax compatibility
    pub _pad: [u32; 2],       // offset 8 — align rng_state to 16
    pub rng_state: [u32; 4],  // offset 16 — PCG RNG state
    pub polygons: [GpuPolygon; MAX_POLYGONS_PER_IMAGE], // offset 32
}

// Manual Pod/Zeroable impl because bytemuck derive doesn't support arrays > 256
// SAFETY: GpuDrawingState is #[repr(C)], all fields are Pod types,
// and all bit patterns are valid (no padding gaps, no uninit bytes).
unsafe impl Zeroable for GpuDrawingState {}
unsafe impl Pod for GpuDrawingState {}

/// Uniform parameters passed to all shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParams {
    pub image_width: u32,
    pub image_height: u32,
    pub max_polygons: u32,
    pub min_polygons: u32,

    pub max_error_per_pixel: f32,
    pub per_point_multiplier: f32,
    pub iteration_number: u32,
    pub migration_interval: u32,

    // Mutation probabilities
    pub add_polygon_prob: f32,
    pub remove_polygon_prob: f32,
    pub reorder_polygon_prob: f32,
    pub offset_polygon_prob: f32,

    pub move_point_prob: f32,
    pub micro_adjust_prob: f32,
    pub change_color_prob: f32,
    pub lighten_color_prob: f32,

    pub darken_color_prob: f32,
    pub move_point_max_delta: f32,
    pub micro_adjust_delta: f32,
    pub new_point_max_distance: f32,

    pub offset_polygon_magnitude: f32,
    pub min_alpha_norm: f32,
    pub max_alpha_norm: f32,
    pub crossover_prob: f32,

    // vec4[6] — crossover & island params
    pub spatial_crossover_weight: f32,
    pub tournament_size: u32,
    pub island_count: u32,
    pub inter_island_interval: u32,

    // vec4[7] — chain count + padding
    pub chain_count_param: u32,
    pub _pad6: u32,
    pub _pad7: u32,
    pub _pad8: u32,
}

/// Control flags for CPU ↔ GPU communication (atomic u32s).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ControlFlags {
    pub new_best_found: u32,
    pub best_chain_id: u32,
    pub best_fitness_bits: u32,
    pub _pad: u32,
}

pub const GPU_DRAWING_STATE_SIZE: usize = std::mem::size_of::<GpuDrawingState>();
pub const GPU_POLYGON_SIZE: usize = std::mem::size_of::<GpuPolygon>();

/// Build GpuParams from runtime MutationParams + image dimensions.
pub fn gpu_params_from(mp: &MutationParams, w: u32, h: u32, migration_interval: u32, chain_count: u32) -> GpuParams {
    use crate::settings::*;
    GpuParams {
        image_width: w,
        image_height: h,
        max_polygons: mp.max_polygons.min(MAX_POLYGONS_PER_IMAGE as u32),
        min_polygons: mp.min_polygons,
        max_error_per_pixel: GPU_MAX_ERROR_PER_PIXEL,
        per_point_multiplier: PER_POINT_MULTIPLIER,
        iteration_number: 0,
        migration_interval,
        add_polygon_prob: mp.add_polygon_prob,
        remove_polygon_prob: mp.remove_polygon_prob,
        reorder_polygon_prob: mp.reorder_polygon_prob,
        offset_polygon_prob: mp.offset_polygon_prob,
        move_point_prob: mp.move_point_prob,
        micro_adjust_prob: mp.micro_adjust_prob,
        change_color_prob: mp.change_color_prob,
        lighten_color_prob: mp.lighten_color_prob,
        darken_color_prob: mp.darken_color_prob,
        move_point_max_delta: mp.move_point_max_delta,
        micro_adjust_delta: mp.micro_adjust_delta,
        new_point_max_distance: mp.new_point_max_distance,
        offset_polygon_magnitude: mp.offset_polygon_magnitude,
        min_alpha_norm: mp.min_alpha as f32 / 255.0,
        max_alpha_norm: mp.max_alpha as f32 / 255.0,
        crossover_prob: mp.crossover_prob,
        spatial_crossover_weight: mp.spatial_crossover_weight,
        tournament_size: mp.tournament_size,
        island_count: mp.island_count,
        inter_island_interval: mp.inter_island_interval,
        chain_count_param: chain_count,
        _pad6: 0,
        _pad7: 0,
        _pad8: 0,
    }
}

/// Build the default GpuParams from settings constants.
pub fn default_gpu_params(w: u32, h: u32, migration_interval: u32, chain_count: u32) -> GpuParams {
    gpu_params_from(&MutationParams::default(), w, h, migration_interval, chain_count)
}

/// Convert a CPU Drawing to GPU bytes for a single chain's DrawingState.
/// Multi-vertex polygons are fan-triangulated.
/// `seed` is used to initialize the PCG RNG state for this chain.
pub fn drawing_to_gpu(drawing: &Drawing, seed: u64) -> GpuDrawingState {
    let mut state = GpuDrawingState::zeroed();

    // Initialize RNG state from seed (PCG-style: state and increment)
    state.rng_state[0] = seed as u32;
    state.rng_state[1] = (seed >> 32) as u32;
    // Use different bits for increment (must be odd for full PCG period)
    // The | 1 must be on the LOW word (bit 0 of the full 64-bit increment)
    let inc = seed.wrapping_mul(6364136223846793005);
    state.rng_state[2] = inc as u32 | 1;
    state.rng_state[3] = (inc >> 32) as u32;

    state.fitness_bits = 0; // will be computed on GPU

    let mut gpu_idx = 0;
    for polygon in &drawing.polygons {
        let color = [
            polygon.color.r as f32 / 255.0,
            polygon.color.g as f32 / 255.0,
            polygon.color.b as f32 / 255.0,
            polygon.color.a as f32 / 255.0,
        ];

        let pts = &polygon.points;
        if pts.len() < 3 {
            continue;
        }

        // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
        for i in 1..pts.len() - 1 {
            if gpu_idx >= MAX_POLYGONS_PER_IMAGE {
                break;
            }
            state.polygons[gpu_idx] = GpuPolygon {
                color,
                v0: [pts[0].x, pts[0].y],
                v1: [pts[i].x, pts[i].y],
                v2: [pts[i + 1].x, pts[i + 1].y],
                _pad: [0.0; 2],
            };
            gpu_idx += 1;
        }

        if gpu_idx >= MAX_POLYGONS_PER_IMAGE {
            break;
        }
    }

    state.polygon_count = gpu_idx as u32;
    state
}

/// Convert GPU DrawingState back to a CPU Drawing.
/// Each GPU triangle becomes a 3-point Polygon.
pub fn gpu_to_drawing(state: &GpuDrawingState) -> Drawing {
    let count = state.polygon_count as usize;
    let mut polygons = Vec::with_capacity(count);

    for i in 0..count {
        let gp = &state.polygons[i];
        // GPU is the source of truth for alpha range — no hardcoded clamping
        let color = Color {
            r: (gp.color[0] * 255.0).round().clamp(0.0, 255.0) as u8,
            g: (gp.color[1] * 255.0).round().clamp(0.0, 255.0) as u8,
            b: (gp.color[2] * 255.0).round().clamp(0.0, 255.0) as u8,
            a: (gp.color[3] * 255.0).round().clamp(0.0, 255.0) as u8,
        };
        let points = vec![
            Point { x: gp.v0[0], y: gp.v0[1] },
            Point { x: gp.v1[0], y: gp.v1[1] },
            Point { x: gp.v2[0], y: gp.v2[1] },
        ];
        polygons.push(Polygon { points, color });
    }

    Drawing {
        polygons,
        is_dirty: false,
        fitness: f32::from_bits(state.fitness_bits),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_polygon_size() {
        assert_eq!(GPU_POLYGON_SIZE, 48);
    }

    #[test]
    fn test_gpu_drawing_state_size() {
        assert_eq!(GPU_DRAWING_STATE_SIZE, 48032);
    }

    #[test]
    fn test_roundtrip_simple_triangle_drawing() {
        let drawing = Drawing {
            polygons: vec![
                Polygon {
                    points: vec![
                        Point { x: 0.1, y: 0.2 },
                        Point { x: 0.3, y: 0.4 },
                        Point { x: 0.5, y: 0.6 },
                    ],
                    color: Color { r: 100, g: 150, b: 200, a: 30 },
                },
                Polygon {
                    points: vec![
                        Point { x: 0.7, y: 0.8 },
                        Point { x: 0.9, y: 0.1 },
                        Point { x: 0.2, y: 0.3 },
                    ],
                    color: Color { r: 50, g: 60, b: 70, a: 40 },
                },
            ],
            is_dirty: false,
            fitness: 0.0,
        };

        let gpu_state = drawing_to_gpu(&drawing, 42);
        assert_eq!(gpu_state.polygon_count, 2);

        let reconstructed = gpu_to_drawing(&gpu_state);
        assert_eq!(reconstructed.polygons.len(), 2);

        // Check first polygon
        let p0 = &reconstructed.polygons[0];
        assert_eq!(p0.points.len(), 3);
        assert!((p0.points[0].x - 0.1).abs() < 0.01);
        assert!((p0.points[0].y - 0.2).abs() < 0.01);
        assert_eq!(p0.color.r, 100);
        assert_eq!(p0.color.g, 150);
        assert_eq!(p0.color.b, 200);
        assert_eq!(p0.color.a, 30);
    }

    #[test]
    fn test_fan_triangulation() {
        // A 4-point polygon should become 2 GPU triangles
        let drawing = Drawing {
            polygons: vec![Polygon {
                points: vec![
                    Point { x: 0.0, y: 0.0 },
                    Point { x: 1.0, y: 0.0 },
                    Point { x: 1.0, y: 1.0 },
                    Point { x: 0.0, y: 1.0 },
                ],
                color: Color { r: 128, g: 128, b: 128, a: 30 },
            }],
            is_dirty: false,
            fitness: 0.0,
        };

        let gpu_state = drawing_to_gpu(&drawing, 42);
        assert_eq!(gpu_state.polygon_count, 2);

        // First triangle: (0,0) (1,0) (1,1)
        assert_eq!(gpu_state.polygons[0].v0, [0.0, 0.0]);
        assert_eq!(gpu_state.polygons[0].v1, [1.0, 0.0]);
        assert_eq!(gpu_state.polygons[0].v2, [1.0, 1.0]);

        // Second triangle: (0,0) (1,1) (0,1)
        assert_eq!(gpu_state.polygons[1].v0, [0.0, 0.0]);
        assert_eq!(gpu_state.polygons[1].v1, [1.0, 1.0]);
        assert_eq!(gpu_state.polygons[1].v2, [0.0, 1.0]);
    }

    #[test]
    fn test_params_size() {
        // Must be 128 bytes (8 vec4 = 32 u32s * 4 = 128)
        assert_eq!(std::mem::size_of::<GpuParams>(), 128);
    }

    #[test]
    fn test_control_flags_size() {
        assert_eq!(std::mem::size_of::<ControlFlags>(), 16);
    }
}
