use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Arc;

use wgpu::*;
use wgpu::util::DeviceExt;

use super::buffers::{
    ControlFlags, GpuDrawingState, GpuParams, GPU_DRAWING_STATE_SIZE,
};

// Capacity for 17 lab sources, five supported raster workgroups, and two modes.
const MAX_BENCHMARK_BIN_PIPELINES: usize = 17 * 5 * 2;
const MAX_BENCHMARK_SOURCE_BYTES: usize = 256 * 1024;
const MAX_BENCHMARK_RASTER_PIPELINES: usize = 17 * 5;

#[derive(Debug, PartialEq, Eq)]
struct BenchmarkRasterKey {
    source: Arc<str>,
    wg: [u32; 2],
}

struct BenchmarkRasterPipeline {
    key: BenchmarkRasterKey,
    pipeline: ComputePipeline,
    shader: ShaderModule,
}

struct BenchmarkBinPipeline {
    source: Arc<str>,
    wg: [u32; 2],
    validation: bool,
    pipeline: ComputePipeline,
}

pub struct GpuPipeline {
    pub device: Device,
    pub queue: Queue,

    // Buffers
    pub chain_states_buf: Buffer,
    pub working_states_buf: Buffer,
    pub reference_texture: Texture,
    pub reference_view: TextureView,
    pub error_accumulators_buf: Buffer,
    pub control_flags_buf: Buffer,
    pub readback_staging_buf: Buffer,
    pub multi_readback_staging_buf: Buffer,  // persistent staging for batch chain readback
    pub eval_fitness_staging_buf: Buffer,    // persistent staging for evaluate_chain_fitness readback
    pub control_staging_bufs: [Buffer; 2],
    pub fitness_packed_buf: Buffer,
    pub fitness_staging_bufs: [Buffer; 2],

    // Tile culling buffers
    pub tile_data_buf: Buffer,
    pub tile_counts_buf: Buffer,

    // Incremental evaluation buffers
    pub chain_framebuffers_buf: Buffer,
    pub chain_total_errors_buf: Buffer,
    pub chain_update_flags_buf: Buffer,

    // Compute pipelines
    pub mutate_pipeline: ComputePipeline,
    pub rasterize_error_pipeline: ComputePipeline,
    pub select_pipeline: ComputePipeline,
    pub bin_polygons_pipeline: ComputePipeline,
    pub init_framebuffers_pipeline: ComputePipeline,
    pub update_framebuffers_pipeline: ComputePipeline,

    // Rasterize pipeline recreation support — stored for creating new pipeline variants
    rasterize_error_shader: ShaderModule,
    rasterize_error_pipeline_layout: PipelineLayout,
    benchmark_raster_source: Option<Arc<str>>,
    benchmark_raster_pipelines: Vec<BenchmarkRasterPipeline>,

    // Bin polygons pipeline recreation support (tile size must match rasterize WG)
    bin_polygons_pipeline_layout: PipelineLayout,
    // Empty in production. Source and compiled handles survive benchmark prepares.
    benchmark_bin_source: Option<Arc<str>>,
    benchmark_bin_pipelines: Vec<BenchmarkBinPipeline>,

    // Init framebuffers pipeline recreation support
    init_framebuffers_pipeline_layout: PipelineLayout,

    // Update framebuffers pipeline recreation support
    update_framebuffers_pipeline_layout: PipelineLayout,

    // Pipeline cache — accelerates pipeline creation on subsequent runs (Vulkan only)
    pipeline_cache: PipelineCache,
    pipeline_cache_path: Option<PathBuf>,

    // Params uniform buffer + bind group (shared across all pipelines at group(1))
    pub params_buf: Buffer,
    pub params_bind_group: BindGroup,

    // Bind groups
    pub mutate_bind_group: BindGroup,
    pub rasterize_error_bind_group: BindGroup,
    pub select_bind_group: BindGroup,
    pub bin_polygons_bind_group: BindGroup,
    pub init_framebuffers_bind_group: BindGroup,
    pub update_framebuffers_bind_group: BindGroup,

    // Timestamp profiling
    pub timestamp_query_set: QuerySet,
    pub timestamp_resolve_buf: Buffer,
    pub timestamp_staging_bufs: [Buffer; 2],
    pub timestamp_period: f32,

    // Config
    pub chain_count: u32,
    pub offspring_capacity: u32, // max offspring slots = min(chain_count * GPU_MAX_LAMBDA, ssbo limit)
    pub image_width: u32,
    pub image_height: u32,
    pub rasterize_wg: [u32; 2], // current workgroup size [wg_x, wg_y]
    pub(super) benchmark_validation: bool,
    pub num_tiles: u32,          // num_tiles_x * num_tiles_y for current rasterize_wg
}

impl GpuPipeline {
    pub async fn new(
        chain_count: u32,
        image_width: u32,
        image_height: u32,
        reference_rgba: &[u8],
        initial_states: &[GpuDrawingState],
    ) -> Self {
        assert!(initial_states.len() >= chain_count as usize);
        assert_eq!(reference_rgba.len(), (image_width * image_height * 4) as usize);

        // --- Device + Queue ---
        // Exclude GL/GLES — EGL conflicts with SDL2's display context.
        // Allow noncompliant adapters for WSL2 dozen (Vulkan-on-D3D12) driver.
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN | Backends::DX12,
            flags: wgpu::InstanceFlags::default()
                | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..InstanceDescriptor::new_without_display_handle()
        });

        // Enumerate all adapters and prefer discrete GPU
        let adapters: Vec<_> = instance.enumerate_adapters(Backends::VULKAN | Backends::DX12).await;
        println!("Found {} GPU adapter(s):", adapters.len());
        for (i, a) in adapters.iter().enumerate() {
            let info = a.get_info();
            println!("  [{}] {} ({:?}, {:?})", i, info.name, info.device_type, info.backend);
        }

        let adapter = adapters
            .into_iter()
            .max_by_key(|a| match a.get_info().device_type {
                DeviceType::DiscreteGpu => 4,
                DeviceType::IntegratedGpu => 3,
                DeviceType::VirtualGpu => 2,
                DeviceType::Cpu => 1,
                DeviceType::Other => 0,
            })
            .expect("Failed to find a suitable GPU adapter");

        let adapter_info = adapter.get_info();
        println!("Selected GPU adapter: {:?}", adapter_info.name);

        // Cap chain_count to fit within the adapter's max storage buffer binding size.
        // Largest buffer is chain_states or working_states (chain_count * GPU_DRAWING_STATE_SIZE).
        let adapter_limits = adapter.limits();
        let max_ssbo = adapter_limits.max_storage_buffer_binding_size;
        let max_chains_by_buffer = max_ssbo / (GPU_DRAWING_STATE_SIZE as u64);
        let chain_count = chain_count.min(max_chains_by_buffer as u32);

        // Calculate the largest buffer we actually need (working_states with offspring_capacity)
        let max_lambda = crate::settings::GPU_MAX_LAMBDA as u64;
        let max_buffer_needed = (chain_count as u64) * max_lambda * (GPU_DRAWING_STATE_SIZE as u64);
        // Clamp to adapter limit (don't request more than hardware supports)
        let max_buffer_size = max_buffer_needed.min(max_ssbo);

        // Tile culling buffers can be large: offspring_capacity * num_tiles * TILE_MAX_POLYS * 4
        // Compute the max tile buffer size we might need (for max resolution / min WG size)
        let tile_max_polys = crate::settings::TILE_MAX_POLYS as u64;
        let max_offspring_capacity = (max_ssbo / (GPU_DRAWING_STATE_SIZE as u64))
            .min((chain_count as u64) * max_lambda);
        // Worst case tile count: max resolution with smallest WG (8x8)
        let max_tiles = ((image_width as u64 + 7) / 8) * ((image_height as u64 + 7) / 8);
        let tile_data_size = max_offspring_capacity * max_tiles * tile_max_polys * 4;
        let tile_counts_size = max_offspring_capacity * max_tiles * 4;
        // The tile_data buffer can exceed max_ssbo on large configs — clamp it
        let tile_data_size = tile_data_size.min(max_ssbo);
        let tile_counts_size = tile_counts_size.min(max_ssbo);
        let max_buffer_size = max_buffer_size.max(tile_data_size).max(tile_counts_size);

        let required_limits = Limits {
            max_storage_buffer_binding_size: max_buffer_size,
            max_buffer_size,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroup_size_x: 512, // 1D workgroup layout needs up to 512 in x (for 32x16 tile)
            max_storage_buffers_per_shader_stage: 7, // select uses 7 bindings (chain_states, working_states, error_accum, control, fitness, chain_update_flags, chain_total_errors)
            ..Limits::downlevel_defaults()
        };

        let adapter_features = adapter.features();
        let subgroup_supported = adapter_features.contains(Features::SUBGROUP);
        let mut required_features = Features::TIMESTAMP_QUERY | Features::PIPELINE_CACHE;
        if subgroup_supported {
            required_features |= Features::SUBGROUP;
            println!("Subgroup feature supported — enabling wave intrinsics for error reduction");
        } else {
            panic!("GPU adapter does not support subgroups — required for rasterize_error shader");
        }

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("gpu_evolver_device"),
                required_features,
                required_limits,
                memory_hints: MemoryHints::Performance,
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .expect("Failed to create GPU device");

        // --- Pipeline cache ---
        // Load cached pipeline data from disk if available (Vulkan only).
        // This avoids recompiling shaders from scratch on subsequent runs.
        let cache_path = pipeline_cache_path(&adapter_info);
        let cache_data = cache_path.as_ref().and_then(|p| std::fs::read(p).ok());
        // SAFETY: cache data was either loaded from a previous run on the same adapter
        // (keyed by pipeline_cache_key) or is None. The fallback flag ensures a fresh
        // cache is created if the data is invalid.
        let pipeline_cache = unsafe { device.create_pipeline_cache(&PipelineCacheDescriptor {
            label: Some("gpu_evolver_cache"),
            data: cache_data.as_deref(),
            fallback: true,
        }) };
        if cache_data.is_some() {
            println!("Loaded pipeline cache from disk");
        }

        // --- Buffer sizes ---
        let _chain_states_size = (chain_count as usize) * GPU_DRAWING_STATE_SIZE;
        // Offspring capacity: chain_count * max_lambda, capped by SSBO limit
        let max_lambda = crate::settings::GPU_MAX_LAMBDA as usize;
        let offspring_capacity = ((max_ssbo as usize) / GPU_DRAWING_STATE_SIZE)
            .min(chain_count as usize * max_lambda);
        let working_states_size = offspring_capacity * GPU_DRAWING_STATE_SIZE;
        let error_accumulators_size = offspring_capacity * 8; // 2x u32 per offspring (stride-2: new error + old error for incremental eval)

        // --- Create buffers ---

        // Chain states (current best per chain) — only use first chain_count entries
        let chain_states_bytes: Vec<u8> = initial_states[..chain_count as usize]
            .iter()
            .flat_map(|s| bytemuck::bytes_of(s).to_vec())
            .collect();
        let chain_states_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("chain_states"),
            contents: &chain_states_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        // Working states (mutated candidates) — sized for offspring_capacity (chain_count * max_lambda)
        let working_states_buf = device.create_buffer(&BufferDescriptor {
            label: Some("working_states"),
            size: working_states_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Reference image — stored as a texture to leverage GPU texture cache hardware
        let reference_texture = device.create_texture(&TextureDescriptor {
            label: Some("reference_image"),
            size: Extent3d {
                width: image_width,
                height: image_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &reference_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            reference_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image_width * 4),
                rows_per_image: Some(image_height),
            },
            Extent3d {
                width: image_width,
                height: image_height,
                depth_or_array_layers: 1,
            },
        );
        let reference_view = reference_texture.create_view(&TextureViewDescriptor::default());

        // Error accumulators (atomic u32 per chain) — zero-initialized once here;
        // the select shader resets them via atomicExchange after each iteration.
        let error_accumulators_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("error_accumulators"),
            contents: &vec![0u8; error_accumulators_size],
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Control flags
        let control_init = ControlFlags {
            new_best_found: 0,
            best_chain_id: 0,
            best_fitness_bits: 0,
            _pad: 0,
        };
        let control_flags_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("control_flags"),
            contents: bytemuck::bytes_of(&control_init),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        // Readback staging (one DrawingState)
        let readback_staging_buf = device.create_buffer(&BufferDescriptor {
            label: Some("readback_staging"),
            size: GPU_DRAWING_STATE_SIZE as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Multi-chain readback staging (all chains — for readback_chains batch readback)
        let multi_readback_staging_buf = device.create_buffer(&BufferDescriptor {
            label: Some("multi_readback_staging"),
            size: (chain_count as u64) * (GPU_DRAWING_STATE_SIZE as u64),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Evaluate-fitness staging (errors + polygon_counts for all chains)
        // Layout: [chain_count * 8 bytes (error accumulators, stride-2)] + [chain_count * 4 bytes (polygon counts)]
        let eval_fitness_staging_buf = device.create_buffer(&BufferDescriptor {
            label: Some("eval_fitness_staging"),
            size: (chain_count as u64) * 12, // 8 bytes errors + 4 bytes polygon_count per chain
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Control staging (for polling new_best_found) — double-buffered
        let control_staging_bufs = std::array::from_fn(|i| {
            device.create_buffer(&BufferDescriptor {
                label: Some(if i == 0 { "control_staging_0" } else { "control_staging_1" }),
                size: std::mem::size_of::<ControlFlags>() as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // Fitness packed buffer (u32 per chain — for CPU readback of all chain fitness values)
        let fitness_packed_size = (chain_count as usize) * 4;
        let fitness_packed_buf = device.create_buffer(&BufferDescriptor {
            label: Some("fitness_packed"),
            size: fitness_packed_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fitness_staging_bufs = std::array::from_fn(|i| {
            device.create_buffer(&BufferDescriptor {
                label: Some(if i == 0 { "fitness_staging_0" } else { "fitness_staging_1" }),
                size: fitness_packed_size as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // --- Tile culling buffers ---
        // Sized for offspring_capacity * num_tiles (worst-case WG) * TILE_MAX_POLYS
        let rasterize_wg = [
            crate::settings::RASTERIZE_WG_X_DEFAULT,
            crate::settings::RASTERIZE_WG_Y_DEFAULT,
        ];
        let num_tiles_x = image_width.div_ceil(rasterize_wg[0]);
        let num_tiles_y = image_height.div_ceil(rasterize_wg[1]);
        let num_tiles = num_tiles_x * num_tiles_y;

        // Use worst-case (smallest WG = most tiles) for buffer allocation
        let max_num_tiles_x = image_width.div_ceil(8); // smallest WG_X = 8
        let max_num_tiles_y = image_height.div_ceil(8); // smallest WG_Y = 8
        let max_num_tiles = max_num_tiles_x * max_num_tiles_y;

        let tile_max_polys = crate::settings::TILE_MAX_POLYS;
        let tile_data_buf_size = (offspring_capacity as u64) * (max_num_tiles as u64) * (tile_max_polys as u64) * 4;
        let tile_counts_buf_size = (offspring_capacity as u64) * (max_num_tiles as u64) * 4;

        // Clamp to device max buffer size
        let device_max = device.limits().max_buffer_size;
        let tile_data_buf_size = tile_data_buf_size.min(device_max);
        let tile_counts_buf_size = tile_counts_buf_size.min(device_max);

        let tile_data_buf = device.create_buffer(&BufferDescriptor {
            label: Some("tile_data"),
            size: tile_data_buf_size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let tile_counts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("tile_counts"),
            size: tile_counts_buf_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        println!(
            "Tile culling buffers allocated: tile_data {:.1} MB, tile_counts {:.1} MB (max {} tiles, {} offspring)",
            tile_data_buf_size as f64 / (1024.0 * 1024.0),
            tile_counts_buf_size as f64 / (1024.0 * 1024.0),
            max_num_tiles,
            offspring_capacity,
        );

        // --- Incremental evaluation buffers ---
        // Chain framebuffers: cached RGBA per chain (chain_count * W * H * 4 bytes)
        // Clamp to device limits — fewer chains may be usable for incremental eval
        let pixels_per_chain = (image_width as u64) * (image_height as u64);
        let chain_framebuffers_ideal = (chain_count as u64) * pixels_per_chain * 4;
        let ssbo_limit = device.limits().max_storage_buffer_binding_size;
        let chain_framebuffers_size = chain_framebuffers_ideal.min(device_max).min(ssbo_limit);
        let chain_framebuffers_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_framebuffers"),
            size: chain_framebuffers_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Chain total errors: raw u32 error sum per chain
        let chain_total_errors_size = (chain_count as u64) * 4;
        let chain_total_errors_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_total_errors"),
            size: chain_total_errors_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Chain update flags: per-chain acceptance flag + dirty bbox for update_framebuffers
        // Layout: [accepted: u32, dirty_bbox_lo: u32, dirty_bbox_hi: u32, padding: u32] × chain_count
        let chain_update_flags_size = (chain_count as u64) * 16;
        let chain_update_flags_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_update_flags"),
            size: chain_update_flags_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_incr_chains = chain_framebuffers_size / (pixels_per_chain * 4);
        println!(
            "Incremental eval buffers allocated: framebuffers {:.1} MB ({} chains max), total_errors {} bytes",
            chain_framebuffers_size as f64 / (1024.0 * 1024.0),
            max_incr_chains,
            chain_total_errors_size,
        );

        // --- Timestamp query profiling ---
        let timestamp_query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("timestamp_queries"),
            ty: QueryType::Timestamp,
            count: 8, // 2 per pass x 4 passes
        });

        let timestamp_resolve_buf = device.create_buffer(&BufferDescriptor {
            label: Some("timestamp_resolve"),
            size: 8 * 8, // 8 x u64
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let timestamp_staging_bufs = std::array::from_fn(|i| {
            device.create_buffer(&BufferDescriptor {
                label: Some(if i == 0 { "timestamp_staging_0" } else { "timestamp_staging_1" }),
                size: 8 * 8,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let timestamp_period = queue.get_timestamp_period();

        // --- Shader modules ---
        let mutate_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mutate_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/mutate.wgsl").into()),
        });

        let select_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("select_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/select.wgsl").into()),
        });

        // --- Bind group layouts ---

        // Mutate: chain_states(read), working_states(rw); params via push constants
        let mutate_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mutate_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
            ],
        });

        // Rasterize+Error: working_states(read), reference_image(texture), error_accumulators(rw),
        //                   tile_data(read), tile_counts(read), chain_states(read), chain_framebuffers(read); params via push constants
        let rasterize_error_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rasterize_error_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
            ],
        });

        // Bin polygons: working_states(read), tile_data(rw), tile_counts(rw); params via push constants
        let bin_polygons_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bin_polygons_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
            ],
        });

        // Select: chain_states(rw), working_states(read), error_accumulators(rw), control(rw), fitness_packed(rw),
        //         chain_update_flags(rw), chain_total_errors(rw); params via push constants
        let select_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("select_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<ControlFlags>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
            ],
        });

        // --- Params uniform buffer + bind group (shared across all pipelines at group(1)) ---
        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("params_uniform"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("params_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuParams>() as u64),
                },
                count: None,
            }],
        });

        let params_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("params_bg"),
            layout: &params_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });

        // --- Compute pipelines ---
        let mutate_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mutate_layout"),
            bind_group_layouts: &[Some(&mutate_bgl), Some(&params_bgl)],
            immediate_size: 0,
        });
        let mutate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("mutate_pipeline"),
            layout: Some(&mutate_pipeline_layout),
            module: &mutate_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Some(&pipeline_cache),
        });

        let rasterize_error_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rasterize_error_layout"),
            bind_group_layouts: &[Some(&rasterize_error_bgl), Some(&params_bgl)],
            immediate_size: 0,
        });
        let (rasterize_error_pipeline, rasterize_error_shader) = create_rasterize_pipeline(
            &device,
            &rasterize_error_pipeline_layout,
            rasterize_wg,
            &pipeline_cache,
        );

        let bin_polygons_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("bin_polygons_layout"),
            bind_group_layouts: &[Some(&bin_polygons_bgl), Some(&params_bgl)],
            immediate_size: 0,
        });
        let bin_polygons_pipeline = create_bin_polygons_pipeline(
            &device,
            &bin_polygons_pipeline_layout,
            rasterize_wg,
            &pipeline_cache,
        );

        let select_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("select_layout"),
            bind_group_layouts: &[Some(&select_bgl), Some(&params_bgl)],
            immediate_size: 0,
        });
        let select_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("select_pipeline"),
            layout: Some(&select_pipeline_layout),
            module: &select_shader,
            entry_point: Some("select_main"),
            compilation_options: Default::default(),
            cache: Some(&pipeline_cache),
        });

        // Init framebuffers: chain_states(read), reference(texture), chain_framebuffers(rw), chain_total_errors(rw)
        let init_framebuffers_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("init_framebuffers_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let init_framebuffers_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("init_framebuffers_layout"),
            bind_group_layouts: &[Some(&init_framebuffers_bgl), Some(&params_bgl)],
            immediate_size: 0,
        });
        let init_framebuffers_pipeline = create_init_framebuffers_pipeline(
            &device,
            &init_framebuffers_pipeline_layout,
            rasterize_wg,
            &pipeline_cache,
        );

        // Update framebuffers: chain_states(read), chain_framebuffers(rw), chain_update_flags(read)
        let update_framebuffers_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("update_framebuffers_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GPU_DRAWING_STATE_SIZE as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let update_framebuffers_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("update_framebuffers_layout"),
            bind_group_layouts: &[Some(&update_framebuffers_bgl), Some(&params_bgl)],
            immediate_size: 0,
        });
        let update_framebuffers_pipeline = create_update_framebuffers_pipeline(
            &device,
            &update_framebuffers_pipeline_layout,
            rasterize_wg,
            &pipeline_cache,
        );

        // --- Bind groups ---
        let mutate_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mutate_bg"),
            layout: &mutate_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: working_states_buf.as_entire_binding() },
            ],
        });

        let rasterize_error_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("rasterize_error_bg"),
            layout: &rasterize_error_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&reference_view) },
                BindGroupEntry { binding: 2, resource: error_accumulators_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: tile_data_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: tile_counts_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: chain_framebuffers_buf.as_entire_binding() },
            ],
        });

        let bin_polygons_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bin_polygons_bg"),
            layout: &bin_polygons_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: tile_data_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: tile_counts_buf.as_entire_binding() },
            ],
        });

        let select_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("select_bg"),
            layout: &select_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: error_accumulators_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: control_flags_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: fitness_packed_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: chain_update_flags_buf.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: chain_total_errors_buf.as_entire_binding() },
            ],
        });

        let update_framebuffers_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("update_framebuffers_bg"),
            layout: &update_framebuffers_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: chain_framebuffers_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: chain_update_flags_buf.as_entire_binding() },
            ],
        });

        let init_framebuffers_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("init_framebuffers_bg"),
            layout: &init_framebuffers_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&reference_view) },
                BindGroupEntry { binding: 2, resource: chain_framebuffers_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: chain_total_errors_buf.as_entire_binding() },
            ],
        });

        Self {
            device,
            queue,
            chain_states_buf,
            working_states_buf,
            reference_texture,
            reference_view,
            error_accumulators_buf,
            control_flags_buf,
            readback_staging_buf,
            multi_readback_staging_buf,
            eval_fitness_staging_buf,
            control_staging_bufs,
            fitness_packed_buf,
            fitness_staging_bufs,
            tile_data_buf,
            tile_counts_buf,
            chain_framebuffers_buf,
            chain_total_errors_buf,
            chain_update_flags_buf,
            mutate_pipeline,
            rasterize_error_pipeline,
            select_pipeline,
            bin_polygons_pipeline,
            init_framebuffers_pipeline,
            update_framebuffers_pipeline,
            rasterize_error_shader,
            rasterize_error_pipeline_layout,
            benchmark_raster_source: None,
            benchmark_raster_pipelines: Vec::new(),
            bin_polygons_pipeline_layout,
            benchmark_bin_source: None,
            benchmark_bin_pipelines: Vec::new(),
            init_framebuffers_pipeline_layout,
            update_framebuffers_pipeline_layout,
            pipeline_cache,
            pipeline_cache_path: cache_path,
            timestamp_query_set,
            timestamp_resolve_buf,
            timestamp_staging_bufs,
            timestamp_period,
            params_buf,
            params_bind_group,
            mutate_bind_group,
            rasterize_error_bind_group,
            select_bind_group,
            bin_polygons_bind_group,
            init_framebuffers_bind_group,
            update_framebuffers_bind_group,
            chain_count,
            offspring_capacity: offspring_capacity as u32,
            image_width,
            image_height,
            rasterize_wg,
            num_tiles,
            benchmark_validation: false,
        }
    }

    /// Recreate the `rasterize_error` pipeline with a new workgroup size.
    /// Also recreates the `bin_polygons` pipeline since tile size must match.
    /// This is called between batches when the user changes the workgroup size.
    ///
    /// # Panics
    /// Panics if a benchmark override cannot be recreated. Checked benchmark
    /// preparation propagates this error through `try_set_rasterize_wg` instead.
    pub fn set_rasterize_wg(&mut self, wg: [u32; 2]) {
        self.try_set_rasterize_wg(wg)
            .expect("workgroup pipeline recreation failed");
    }

    pub(super) fn try_set_rasterize_wg(&mut self, wg: [u32; 2]) -> Result<(), String> {
        if wg == self.rasterize_wg {
            return Ok(());
        }
        // Compile the installed candidate before changing any active pipeline.
        let bin = if let Some(source) = self.benchmark_bin_source.clone() {
            self.cached_benchmark_bin(source, wg, self.benchmark_validation)?
        } else if self.benchmark_validation {
            self.cached_benchmark_bin(
                Arc::from(include_str!("../shaders/bin_polygons.wgsl")),
                wg,
                true,
            )?
        } else {
            create_bin_polygons_pipeline(
                &self.device,
                &self.bin_polygons_pipeline_layout,
                wg,
                &self.pipeline_cache,
            )
        };
        println!(
            "Recreating rasterize_error + bin_polygons pipelines: {}x{} -> {}x{}",
            self.rasterize_wg[0], self.rasterize_wg[1], wg[0], wg[1]
        );
        let (pipeline, shader) = if let Some(source) = self.benchmark_raster_source.clone() {
            self.cached_benchmark_raster(source, wg)?
        } else {
            create_rasterize_pipeline(
                &self.device,
                &self.rasterize_error_pipeline_layout,
                wg,
                &self.pipeline_cache,
            )
        };
        self.rasterize_error_pipeline = pipeline;
        self.rasterize_error_shader = shader;

        self.bin_polygons_pipeline = bin;

        self.init_framebuffers_pipeline = create_init_framebuffers_pipeline(
            &self.device,
            &self.init_framebuffers_pipeline_layout,
            wg,
            &self.pipeline_cache,
        );

        self.update_framebuffers_pipeline = create_update_framebuffers_pipeline(
            &self.device,
            &self.update_framebuffers_pipeline_layout,
            wg,
            &self.pipeline_cache,
        );

        self.rasterize_wg = wg;
        self.num_tiles = self.image_width.div_ceil(wg[0]) * self.image_height.div_ceil(wg[1]);
        Ok(())
    }

    pub(super) fn set_benchmark_raster_shader(&mut self, source: &str) -> Result<(), String> {
        validate_benchmark_raster_source(source)?;
        let source = self
            .benchmark_raster_pipelines
            .iter()
            .find(|entry| entry.key.source.as_ref() == source)
            .map_or_else(|| Arc::from(source), |entry| Arc::clone(&entry.key.source));
        let (pipeline, shader) =
            self.cached_benchmark_raster(Arc::clone(&source), self.rasterize_wg)?;
        self.rasterize_error_pipeline = pipeline;
        self.rasterize_error_shader = shader;
        self.benchmark_raster_source = Some(source);
        Ok(())
    }

    fn cached_benchmark_raster(
        &mut self,
        source: Arc<str>,
        wg: [u32; 2],
    ) -> Result<(ComputePipeline, ShaderModule), String> {
        let key = BenchmarkRasterKey { source, wg };
        if let Some(entry) = self
            .benchmark_raster_pipelines
            .iter()
            .find(|entry| entry.key == key)
        {
            return Ok((entry.pipeline.clone(), entry.shader.clone()));
        }
        if self.benchmark_raster_pipelines.len() == MAX_BENCHMARK_RASTER_PIPELINES {
            return Err("benchmark raster pipeline cache is full".into());
        }
        let (pipeline, shader) = if self.benchmark_raster_pipelines.is_empty()
            && self.benchmark_raster_source.is_none()
            && wg == self.rasterize_wg
            && key.source.as_ref() == include_str!("../shaders/rasterize_error.wgsl")
        {
            (
                self.rasterize_error_pipeline.clone(),
                self.rasterize_error_shader.clone(),
            )
        } else {
            self.compile_benchmark_raster(&key.source, wg)?
        };
        self.benchmark_raster_pipelines
            .push(BenchmarkRasterPipeline {
                key,
                pipeline: pipeline.clone(),
                shader: shader.clone(),
            });
        Ok((pipeline, shader))
    }

    fn compile_benchmark_raster(
        &self,
        source: &str,
        wg: [u32; 2],
    ) -> Result<(ComputePipeline, ShaderModule), String> {
        let specialized = specialize_raster_source(source, wg);
        let scopes = [
            self.device.push_error_scope(ErrorFilter::Validation),
            self.device.push_error_scope(ErrorFilter::OutOfMemory),
            self.device.push_error_scope(ErrorFilter::Internal),
        ];
        let compiled = create_rasterize_pipeline_from_source(
            &self.device,
            &self.rasterize_error_pipeline_layout,
            &specialized,
            &self.pipeline_cache,
        );
        let mut errors = Vec::with_capacity(3);
        for scope in scopes.into_iter().rev() {
            if let Some(error) = futures_lite::future::block_on(scope.pop()) {
                errors.push(error.to_string());
            }
        }
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        Ok(compiled)
    }

    pub(super) fn set_benchmark_bin_shader(&mut self, source: &str) -> Result<(), String> {
        validate_benchmark_bin_source(source)?;
        let source = self
            .benchmark_bin_pipelines
            .iter()
            .find(|entry| entry.source.as_ref() == source)
            .map_or_else(|| Arc::from(source), |entry| Arc::clone(&entry.source));
        let pipeline = self.cached_benchmark_bin(
            Arc::clone(&source),
            self.rasterize_wg,
            self.benchmark_validation,
        )?;
        self.bin_polygons_pipeline = pipeline;
        self.benchmark_bin_source = Some(source);
        Ok(())
    }

    /// Instrument the actual installed source, including after workgroup changes.
    pub(super) fn set_benchmark_validation(&mut self, enabled: bool) -> Result<(), String> {
        if enabled == self.benchmark_validation {
            return Ok(());
        }
        let source = self
            .benchmark_bin_source
            .clone()
            .unwrap_or_else(|| Arc::from(include_str!("../shaders/bin_polygons.wgsl")));
        self.bin_polygons_pipeline =
            self.cached_benchmark_bin(source, self.rasterize_wg, enabled)?;
        self.benchmark_validation = enabled;
        Ok(())
    }

    fn cached_benchmark_bin(
        &mut self,
        source: Arc<str>,
        wg: [u32; 2],
        validation: bool,
    ) -> Result<ComputePipeline, String> {
        if let Some(entry) = self.benchmark_bin_pipelines.iter().find(|entry| {
            entry.source == source && entry.wg == wg && entry.validation == validation
        }) {
            return Ok(entry.pipeline.clone());
        }
        if self.benchmark_bin_pipelines.len() == MAX_BENCHMARK_BIN_PIPELINES {
            return Err("benchmark bin pipeline cache is full".into());
        }
        let pipeline = if self.benchmark_bin_pipelines.is_empty()
            && self.benchmark_bin_source.is_none()
            && !self.benchmark_validation
            && !validation
            && wg == self.rasterize_wg
            && source.as_ref() == include_str!("../shaders/bin_polygons.wgsl")
        {
            self.bin_polygons_pipeline.clone()
        } else {
            self.compile_benchmark_bin(&source, wg, validation)?
        };
        self.benchmark_bin_pipelines.push(BenchmarkBinPipeline {
            source,
            wg,
            validation,
            pipeline: pipeline.clone(),
        });
        Ok(pipeline)
    }

    // Pipeline compilation and scope completion use wgpu's blocking driver path.
    // The lab's parent process timeout bounds a stalled driver, as for GPU drain.
    fn compile_benchmark_bin(
        &self,
        source: &str,
        wg: [u32; 2],
        validation: bool,
    ) -> Result<ComputePipeline, String> {
        let specialized = specialize_bin_source(source, wg);
        let specialized = if validation {
            benchmark_bin_source(&specialized)?
        } else {
            specialized
        };
        let scopes = [
            self.device.push_error_scope(ErrorFilter::Validation),
            self.device.push_error_scope(ErrorFilter::OutOfMemory),
            self.device.push_error_scope(ErrorFilter::Internal),
        ];
        let pipeline = create_bin_pipeline_from_source(
            &self.device,
            &self.bin_polygons_pipeline_layout,
            &specialized,
            &self.pipeline_cache,
        );
        let mut errors = Vec::with_capacity(3);
        for scope in scopes.into_iter().rev() {
            if let Some(error) = futures_lite::future::block_on(scope.pop()) {
                errors.push(error.to_string());
            }
        }
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        Ok(pipeline)
    }

    /// Save pipeline cache data to disk for faster startup next time.
    pub fn save_pipeline_cache(&self) {
        let Some(ref path) = self.pipeline_cache_path else {
            return;
        };
        let Some(data) = self.pipeline_cache.get_data() else {
            return;
        };
        // Atomic write: write to temp file, then rename
        let tmp = path.with_extension("bin.tmp");
        if let Err(e) = std::fs::write(&tmp, &data) {
            eprintln!("Failed to write pipeline cache: {}", e);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            eprintln!("Failed to rename pipeline cache: {}", e);
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        println!("Saved pipeline cache ({} bytes)", data.len());
    }
}

/// Create a rasterize_error compute pipeline with the given workgroup size.
/// Uses string replacement on the shader source since naga does not support
/// override constants in @workgroup_size or const expressions.
fn create_rasterize_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> (ComputePipeline, ShaderModule) {
    let source = specialize_raster_source(include_str!("../shaders/rasterize_error.wgsl"), wg);
    create_rasterize_pipeline_from_source(device, layout, &source, cache)
}

fn create_rasterize_pipeline_from_source(
    device: &Device,
    layout: &PipelineLayout,
    source: &str,
    cache: &PipelineCache,
) -> (ComputePipeline, ShaderModule) {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("rasterize_error_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("rasterize_error_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    });

    (pipeline, shader)
}

fn specialize_raster_source(source: &str, wg: [u32; 2]) -> String {
    source
        .replace(
            "const WG_X: u32 = 16;",
            &format!("const WG_X: u32 = {};", wg[0]),
        )
        .replace(
            "const WG_Y: u32 = 16;",
            &format!("const WG_Y: u32 = {};", wg[1]),
        )
}

fn validate_benchmark_raster_source(source: &str) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err("raster shader must contain 1..262144 bytes".into());
    }
    for marker in ["const WG_X: u32 = 16;", "const WG_Y: u32 = 16;"] {
        if source.matches(marker).count() != 1 {
            return Err(format!(
                "raster specialization must match {marker:?} exactly once"
            ));
        }
    }
    Ok(())
}

// Only the rasterize/error, init-framebuffers, and update-framebuffers shaders
// use these options. Every shared value they read is explicitly written before
// a workgroup barrier; unused polygon/reduction slots are never read. Avoid the
// redundant zeroing code, which is expensive to compile on the WSL Dozen driver.
// Keep default initialization for mutation: empty-parent crossover reads slot 0.
// Vulkan provides native subgroup IDs. Naga's DX12 translation infers them from
// local indices, so retain default initialization on that backend.
fn rasterize_compilation_options(device: &Device) -> PipelineCompilationOptions<'static> {
    PipelineCompilationOptions {
        zero_initialize_workgroup_memory: device.adapter_info().backend != Backend::Vulkan,
        ..Default::default()
    }
}

/// Create a `bin_polygons` compute pipeline with tile size matching the rasterize workgroup.
fn create_bin_polygons_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> ComputePipeline {
    let source = bin_polygons_source(wg);
    create_bin_pipeline_from_source(device, layout, &source, cache)
}

fn create_bin_pipeline_from_source(
    device: &Device,
    layout: &PipelineLayout,
    source: &str,
    cache: &PipelineCache,
) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bin_polygons_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bin_polygons_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: Some(cache),
    })
}

fn bin_polygons_source(wg: [u32; 2]) -> String {
    specialize_bin_source(include_str!("../shaders/bin_polygons.wgsl"), wg)
}

fn specialize_bin_source(source: &str, wg: [u32; 2]) -> String {
    source
        .replace(
            "const TILE_W: u32 = 16;",
            &format!("const TILE_W: u32 = {};", wg[0]),
        )
        .replace(
            "const TILE_H: u32 = 16;",
            &format!("const TILE_H: u32 = {};", wg[1]),
        )
}

fn validate_benchmark_bin_source(source: &str) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err("bin shader must contain 1..262144 bytes".into());
    }
    for marker in ["const TILE_W: u32 = 16;", "const TILE_H: u32 = 16;"] {
        if source.matches(marker).count() != 1 {
            return Err(format!(
                "bin specialization must match {marker:?} exactly once"
            ));
        }
    }
    Ok(())
}

fn benchmark_bin_source(source: &str) -> Result<String, String> {
    let add = "let slot = atomicAdd(&tile_counts[tile_global], 1u);";
    let store = "atomicStore(&tile_counts[tile_global], raw_count);";
    // The reserved last word records uncapped occupancy over ALL iterations.
    // Recognize exactly one known counting mechanism in the installed source.
    match (source.matches(add).count(), source.matches(store).count()) {
        (1, 0) => Ok(source.replace(add, &format!("{add}\n                atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], slot + 1u);"))),
        (0, 1) => Ok(source.replace(store, &format!("atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);\n        {store}"))),
        _ => Err("benchmark bin instrumentation requires exactly one recognized count marker".into()),
    }
}

/// Create an init_framebuffers compute pipeline with the given workgroup size.
fn create_init_framebuffers_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> ComputePipeline {
    let source = include_str!("../shaders/init_framebuffers.wgsl")
        .replace("const WG_X: u32 = 16;", &format!("const WG_X: u32 = {};", wg[0]))
        .replace("const WG_Y: u32 = 16;", &format!("const WG_Y: u32 = {};", wg[1]));

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("init_framebuffers_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("init_framebuffers_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    })
}

/// Create an update_framebuffers compute pipeline with the given workgroup size.
fn create_update_framebuffers_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> ComputePipeline {
    let source = include_str!("../shaders/update_framebuffers.wgsl")
        .replace("const WG_X: u32 = 16;", &format!("const WG_X: u32 = {};", wg[0]))
        .replace("const WG_Y: u32 = 16;", &format!("const WG_Y: u32 = {};", wg[1]));

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("update_framebuffers_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("update_framebuffers_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    })
}

/// Get the disk path for storing this adapter's pipeline cache.
/// Returns `None` if the backend doesn't support pipeline caching.
fn pipeline_cache_path(adapter_info: &AdapterInfo) -> Option<PathBuf> {
    let key = wgpu::util::pipeline_cache_key(adapter_info)?;
    Some(PathBuf::from(format!("pipeline_cache_{}.bin", key)))
}

#[cfg(test)]
mod benchmark_tests {
    use super::{
        benchmark_bin_source, bin_polygons_source, specialize_bin_source,
        specialize_raster_source, validate_benchmark_bin_source,
        validate_benchmark_raster_source, BenchmarkRasterKey, MAX_BENCHMARK_SOURCE_BYTES,
    };

    #[test]
    fn raster_source_specialization_preserves_candidate_and_cache_identity() {
        let original = include_str!("../shaders/rasterize_error.wgsl");
        let candidate = format!("{original}\n// actual candidate");
        validate_benchmark_raster_source(&candidate).unwrap();
        for wg in [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]] {
            let source = specialize_raster_source(&candidate, wg);
            assert!(source.ends_with("// actual candidate"));
            assert!(source.contains(&format!("const WG_X: u32 = {};", wg[0])));
            assert!(source.contains(&format!("const WG_Y: u32 = {};", wg[1])));
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let key = BenchmarkRasterKey {
                source: candidate.as_str().into(),
                wg,
            };
            assert_eq!(
                key,
                BenchmarkRasterKey {
                    source: candidate.as_str().into(),
                    wg
                }
            );
            assert_ne!(
                key,
                BenchmarkRasterKey {
                    source: original.into(),
                    wg
                }
            );
            assert_ne!(
                key,
                BenchmarkRasterKey {
                    source: candidate.as_str().into(),
                    wg: [1, 1]
                }
            );
        }
    }

    #[test]
    fn raster_source_boundaries_and_marker_cardinality_fail_closed() {
        let original = include_str!("../shaders/rasterize_error.wgsl");
        assert!(validate_benchmark_raster_source("").is_err());
        assert!(
            validate_benchmark_raster_source(&" ".repeat(MAX_BENCHMARK_SOURCE_BYTES + 1)).is_err()
        );
        for marker in ["const WG_X: u32 = 16;", "const WG_Y: u32 = 16;"] {
            assert!(validate_benchmark_raster_source(&original.replace(marker, "")).is_err());
            assert!(validate_benchmark_raster_source(&format!("{original}\n{marker}")).is_err());
        }
    }

    #[test]
    fn benchmark_bin_variants_validate_without_a_gpu() {
        for wg in [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]] {
            let normal = bin_polygons_source(wg);
            assert!(!normal.contains("atomicMax"));
            let source = benchmark_bin_source(&normal).unwrap();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }

    #[test]
    fn candidate_sources_keep_workgroup_and_validation_specialization() {
        // Keep the legacy atomic-add fixture independent of the production winner.
        // The variants test validates the currently installed source separately.
        let original = include_str!(
            "../../benchmarks/evolution/overnight-20261024/fixtures/bin-tile32-reference.wgsl"
        );
        let annotation = original
            .lines()
            .find(|line| line.starts_with("@compute @workgroup_size("))
            .unwrap();
        for internal_wg in [32, 64, 128, 256] {
            let candidate = original
                .replace(
                    annotation,
                    &format!("@compute @workgroup_size({internal_wg}, 1, 1)"),
                )
                .replace("+= 32u", &format!("+= {internal_wg}u"));
            validate_benchmark_bin_source(&candidate).unwrap();
            for wg in [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]] {
                let specialized = specialize_bin_source(&candidate, wg);
                assert!(specialized.contains(&format!("@workgroup_size({internal_wg}, 1, 1)")));
                assert!(specialized.contains(&format!("const TILE_W: u32 = {};", wg[0])));
                assert!(specialized.contains(&format!("const TILE_H: u32 = {};", wg[1])));
                let validation = benchmark_bin_source(&specialized).unwrap();
                assert!(validation.contains(
                    "atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], slot + 1u)"
                ));
                let module = wgpu::naga::front::wgsl::parse_str(&validation).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
    }

    #[test]
    fn candidate_source_boundaries_and_instrumentation_fail_closed() {
        let original = include_str!("../shaders/bin_polygons.wgsl");
        assert!(validate_benchmark_bin_source("").is_err());
        assert!(
            validate_benchmark_bin_source(&" ".repeat(MAX_BENCHMARK_SOURCE_BYTES + 1)).is_err()
        );
        assert!(
            validate_benchmark_bin_source(&original.replace("const TILE_W: u32 = 16;", ""))
                .is_err()
        );
        assert!(
            validate_benchmark_bin_source(&format!("{original}\nconst TILE_H: u32 = 16;")).is_err()
        );
        assert!(benchmark_bin_source("no atomicAdd marker").is_err());
        assert!(benchmark_bin_source(&format!("{original}\n{original}")).is_err());
    }

    #[test]
    fn local_count_validation_uses_uncapped_actual_store() {
        let source = r"
const TILE_W: u32 = 16;
const TILE_H: u32 = 16;
@group(0) @binding(0) var<storage, read_write> tile_counts: array<atomic<u32>>;
@compute @workgroup_size(32, 1, 1)
fn main() {
    let tile_global = 0u;
    let raw_count = 257u;
    atomicStore(&tile_counts[tile_global], raw_count);
}";
        validate_benchmark_bin_source(source).unwrap();
        let instrumented = benchmark_bin_source(source).unwrap();
        assert!(instrumented
            .contains("atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);"));
        assert!(instrumented.contains("atomicStore(&tile_counts[tile_global], raw_count);"));
        let module = wgpu::naga::front::wgsl::parse_str(&instrumented).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let mixed = format!("{source}\nlet slot = atomicAdd(&tile_counts[tile_global], 1u);");
        assert!(benchmark_bin_source(&mixed).is_err());
        assert!(benchmark_bin_source(&format!("{source}\n{source}")).is_err());
    }
}
