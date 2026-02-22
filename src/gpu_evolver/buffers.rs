use bytemuck::{Pod, Zeroable};

use crate::models::color::Color;
use crate::models::drawing::Drawing;
use crate::models::point::Point;
use crate::models::polygon::Polygon;
use crate::mutation_params::MutationParams;
use crate::settings::MAX_POLYGONS_PER_IMAGE;

/// GPU polygon: a single triangle with color, quantized to 16 bytes.
/// Word 0: RGBA color packed as 4x u8 via pack4x8unorm convention
/// Word 1: v0 packed as two u16 (x | y<<16)
/// Word 2: v1 packed the same way
/// Word 3: v2 packed the same way
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPolygon {
    pub data: [u32; 4],   // 16 bytes total
}

/// Per-chain drawing state on GPU.
/// polygon_count + fitness + mutation_scale + stagnation_counter + rng_state = 32 bytes header
/// polygons: MAX_POLYGONS_PER_IMAGE * 16 = 16000 bytes
/// Total: 16032 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuDrawingState {
    pub polygon_count: u32,           // offset 0
    pub fitness_bits: u32,            // offset 4 — bitcast f32 for atomicMax compatibility
    pub mutation_scale_bits: u32,     // offset 8 — bitcast f32, adaptive mutation scale
    pub stagnation_counter: u32,      // offset 12 — iterations since last improvement
    pub rng_state: [u32; 4],          // offset 16 — only [0] is active RNG state; [1..3] unused padding
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
    pub _pad0: u32,

    // Mutation probabilities
    pub add_polygon_prob: f32,
    pub remove_polygon_prob: f32,
    pub reorder_polygon_prob: f32,
    pub offset_polygon_prob: f32,

    pub move_point_prob: f32,
    pub micro_adjust_prob: f32,
    pub change_color_prob: f32,
    pub adjust_brightness_prob: f32,

    pub adjust_saturation_prob: f32,
    pub move_point_max_delta: f32,
    pub micro_adjust_delta: f32,
    pub new_point_max_distance: f32,

    pub offset_polygon_magnitude: f32,
    pub min_alpha_norm: f32,
    pub max_alpha_norm: f32,
    pub crossover_prob: f32,

    // vec4[6] — crossover params
    pub spatial_crossover_weight: f32,
    pub tournament_size: u32,
    pub incremental_eval: u32,
    pub tile_culling: u32,

    // vec4[7] — chain count + lambda + adaptive mutation
    pub chain_count_param: u32,
    pub single_mutation_mode: u32,
    pub lambda: u32,
    pub adaptive_mutation: u32,

    // vec4[8] — new mutation probabilities
    pub scale_polygon_prob: f32,
    pub rotate_polygon_prob: f32,
    pub adjacent_swap_prob: f32,
    pub merge_polygon_prob: f32,

    // vec4[9] — more new mutation probabilities + deltas
    pub clone_polygon_prob: f32,
    pub medium_move_prob: f32,
    pub medium_move_delta: f32,
    pub swap_colors_prob: f32,

    // vec4[10] — merge thresholds + padding
    pub merge_centroid_threshold: f32,
    pub merge_color_threshold: f32,
    pub _pad2: u32,
    pub _pad3: u32,

    // vec4[11-15] — reserved padding to fill 256 bytes
    pub _reserved: [u32; 20],
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
pub fn gpu_params_from(mp: &MutationParams, w: u32, h: u32, chain_count: u32) -> GpuParams {
    use crate::settings::*;
    GpuParams {
        image_width: w,
        image_height: h,
        max_polygons: mp.max_polygons.min(MAX_POLYGONS_PER_IMAGE as u32),
        min_polygons: mp.min_polygons,
        max_error_per_pixel: MAX_ERROR_PER_PIXEL,
        per_point_multiplier: PER_POINT_MULTIPLIER,
        iteration_number: 0,
        _pad0: 0,
        add_polygon_prob: mp.add_polygon_prob,
        remove_polygon_prob: mp.remove_polygon_prob,
        reorder_polygon_prob: mp.reorder_polygon_prob,
        offset_polygon_prob: mp.offset_polygon_prob,
        move_point_prob: mp.move_point_prob,
        micro_adjust_prob: mp.micro_adjust_prob,
        change_color_prob: mp.change_color_prob,
        adjust_brightness_prob: mp.adjust_brightness_prob,
        adjust_saturation_prob: mp.adjust_saturation_prob,
        move_point_max_delta: mp.move_point_max_delta,
        micro_adjust_delta: mp.micro_adjust_delta,
        new_point_max_distance: mp.new_point_max_distance,
        offset_polygon_magnitude: mp.offset_polygon_magnitude,
        min_alpha_norm: mp.min_alpha as f32 / 255.0,
        max_alpha_norm: mp.max_alpha as f32 / 255.0,
        crossover_prob: mp.crossover_prob,
        spatial_crossover_weight: mp.spatial_crossover_weight,
        tournament_size: mp.tournament_size,
        incremental_eval: 1,
        tile_culling: 1,
        chain_count_param: chain_count,
        single_mutation_mode: if mp.single_mutation_mode { 1 } else { 0 },
        lambda: mp.lambda,
        adaptive_mutation: if mp.adaptive_mutation { 1 } else { 0 },
        scale_polygon_prob: mp.scale_polygon_prob,
        rotate_polygon_prob: mp.rotate_polygon_prob,
        adjacent_swap_prob: mp.adjacent_swap_prob,
        merge_polygon_prob: mp.merge_polygon_prob,
        clone_polygon_prob: mp.clone_polygon_prob,
        medium_move_prob: mp.medium_move_prob,
        medium_move_delta: mp.medium_move_delta,
        swap_colors_prob: mp.swap_colors_prob,
        merge_centroid_threshold: mp.merge_centroid_threshold,
        merge_color_threshold: mp.merge_color_threshold,
        _pad2: 0,
        _pad3: 0,
        _reserved: [0u32; 20],
    }
}

/// Build the default GpuParams from settings constants.
pub fn default_gpu_params(w: u32, h: u32, chain_count: u32) -> GpuParams {
    gpu_params_from(&MutationParams::default(), w, h, chain_count)
}

/// Pack RGBA color (0–255 u8 range) into a single u32, matching WGSL pack4x8unorm layout.
/// pack4x8unorm packs as: byte0=R, byte1=G, byte2=B, byte3=A (little-endian u32).
fn pack_color_u32(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

/// Pack a 2D vertex (0.0–1.0 floats) into a single u32 as two u16 values.
fn pack_vertex_u32(x: f32, y: f32) -> u32 {
    let xi = (x.clamp(0.0, 1.0) * 65535.0).round() as u32;
    let yi = (y.clamp(0.0, 1.0) * 65535.0).round() as u32;
    xi | (yi << 16)
}

/// Convert a CPU Drawing to GPU bytes for a single chain's DrawingState.
/// Multi-vertex polygons are fan-triangulated.
/// `seed` is used to initialize the PCG RNG state for this chain.
pub fn drawing_to_gpu(drawing: &Drawing, seed: u64) -> GpuDrawingState {
    let mut state = GpuDrawingState::zeroed();

    // Initialize RNG state — only [0] is used by the 32-bit PCG hash; [1..3] are unused padding
    state.rng_state[0] = seed as u32;

    state.fitness_bits = 0; // will be computed on GPU
    state.mutation_scale_bits = 1.0f32.to_bits(); // start at scale 1.0
    state.stagnation_counter = 0;

    let mut gpu_idx = 0;
    for polygon in &drawing.polygons {
        let color_packed = pack_color_u32(
            polygon.color.r,
            polygon.color.g,
            polygon.color.b,
            polygon.color.a,
        );

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
                data: [
                    color_packed,
                    pack_vertex_u32(pts[0].x, pts[0].y),
                    pack_vertex_u32(pts[i].x, pts[i].y),
                    pack_vertex_u32(pts[i + 1].x, pts[i + 1].y),
                ],
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

/// Unpack RGBA color from a packed u32 (matching WGSL unpack4x8unorm layout).
fn unpack_color_u32(packed: u32) -> Color {
    Color {
        r: (packed & 0xFF) as u8,
        g: ((packed >> 8) & 0xFF) as u8,
        b: ((packed >> 16) & 0xFF) as u8,
        a: ((packed >> 24) & 0xFF) as u8,
    }
}

/// Unpack a 2D vertex from a packed u32 (two u16 values).
fn unpack_vertex_u32(packed: u32) -> Point {
    let x = (packed & 0xFFFF) as f32 / 65535.0;
    let y = ((packed >> 16) & 0xFFFF) as f32 / 65535.0;
    Point { x, y }
}

/// Convert GPU DrawingState back to a CPU Drawing.
/// Each GPU triangle becomes a 3-point Polygon.
pub fn gpu_to_drawing(state: &GpuDrawingState) -> Drawing {
    let count = state.polygon_count as usize;
    let mut polygons = Vec::with_capacity(count);

    for i in 0..count {
        let gp = &state.polygons[i];
        let color = unpack_color_u32(gp.data[0]);
        let points = vec![
            unpack_vertex_u32(gp.data[1]),
            unpack_vertex_u32(gp.data[2]),
            unpack_vertex_u32(gp.data[3]),
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
        assert_eq!(GPU_POLYGON_SIZE, 16);
    }

    #[test]
    fn test_gpu_drawing_state_size() {
        assert_eq!(GPU_DRAWING_STATE_SIZE, 16032);
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

        // Check first polygon — vertex precision is ~1/65535, color is exact (u8 roundtrip)
        let p0 = &reconstructed.polygons[0];
        assert_eq!(p0.points.len(), 3);
        assert!((p0.points[0].x - 0.1).abs() < 0.001);
        assert!((p0.points[0].y - 0.2).abs() < 0.001);
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

        // First triangle: (0,0) (1,0) (1,1) — packed as u32
        assert_eq!(gpu_state.polygons[0].data[1], pack_vertex_u32(0.0, 0.0));
        assert_eq!(gpu_state.polygons[0].data[2], pack_vertex_u32(1.0, 0.0));
        assert_eq!(gpu_state.polygons[0].data[3], pack_vertex_u32(1.0, 1.0));

        // Second triangle: (0,0) (1,1) (0,1)
        assert_eq!(gpu_state.polygons[1].data[1], pack_vertex_u32(0.0, 0.0));
        assert_eq!(gpu_state.polygons[1].data[2], pack_vertex_u32(1.0, 1.0));
        assert_eq!(gpu_state.polygons[1].data[3], pack_vertex_u32(0.0, 1.0));
    }

    #[test]
    fn test_params_size() {
        // Must be 256 bytes (16 vec4 = 64 u32s * 4 = 256)
        assert_eq!(std::mem::size_of::<GpuParams>(), 256);
    }

    #[test]
    fn test_control_flags_size() {
        assert_eq!(std::mem::size_of::<ControlFlags>(), 16);
    }
}
