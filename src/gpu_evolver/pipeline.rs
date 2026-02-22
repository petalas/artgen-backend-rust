use std::num::NonZeroU64;
use std::path::PathBuf;

use wgpu::*;
use wgpu::util::DeviceExt;

use super::buffers::{
    ControlFlags, GpuDrawingState, GpuParams, GPU_DRAWING_STATE_SIZE,
};

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

    // Compute pipelines
    pub mutate_pipeline: ComputePipeline,
    pub rasterize_error_pipeline: ComputePipeline,
    pub select_pipeline: ComputePipeline,
    pub bin_polygons_pipeline: ComputePipeline,
    pub init_framebuffers_pipeline: ComputePipeline,

    // Rasterize pipeline recreation support — stored for creating new pipeline variants
    rasterize_error_shader: ShaderModule,
    rasterize_error_pipeline_layout: PipelineLayout,

    // Bin polygons pipeline recreation support (tile size must match rasterize WG)
    bin_polygons_pipeline_layout: PipelineLayout,

    // Init framebuffers pipeline recreation support
    init_framebuffers_pipeline_layout: PipelineLayout,

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
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::VULKAN | Backends::DX12,
            flags: wgpu::InstanceFlags::default()
                | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..Default::default()
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
        let max_ssbo = adapter_limits.max_storage_buffer_binding_size as u64;
        let max_chains_by_buffer = max_ssbo / (GPU_DRAWING_STATE_SIZE as u64);
        let chain_count = chain_count.min(max_chains_by_buffer as u32);

        // Calculate the largest buffer we actually need (working_states with offspring_capacity)
        let max_lambda = crate::settings::GPU_MAX_LAMBDA as u64;
        let max_buffer_needed = (chain_count as u64) * max_lambda * (GPU_DRAWING_STATE_SIZE as u64);
        // Clamp to adapter limit (don't request more than hardware supports)
        let max_buffer_size = max_buffer_needed.min(max_ssbo as u64);

        // Tile culling buffers can be large: offspring_capacity * num_tiles * TILE_MAX_POLYS * 4
        // Compute the max tile buffer size we might need (for max resolution / min WG size)
        let tile_max_polys = crate::settings::TILE_MAX_POLYS as u64;
        let max_offspring_capacity = ((max_ssbo as u64) / (GPU_DRAWING_STATE_SIZE as u64))
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
            max_storage_buffer_binding_size: max_buffer_size as u32,
            max_buffer_size,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroup_size_x: 512, // 1D workgroup layout needs up to 512 in x (for 32x16 tile)
            max_storage_buffers_per_shader_stage: 7, // select uses 7 bindings (chain_states, working_states, error_accum, control, fitness, chain_framebuffers, chain_total_errors)
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
            usage: BufferUsages::STORAGE,
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
        let ssbo_limit = device.limits().max_storage_buffer_binding_size as u64;
        let chain_framebuffers_size = chain_framebuffers_ideal.min(device_max).min(ssbo_limit);
        let chain_framebuffers_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_framebuffers"),
            size: chain_framebuffers_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Chain total errors: raw u32 error sum per chain
        let chain_total_errors_size = (chain_count as u64) * 4;
        let chain_total_errors_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_total_errors"),
            size: chain_total_errors_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
        //         chain_framebuffers(rw), chain_total_errors(rw); params via push constants
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
            bind_group_layouts: &[&mutate_bgl, &params_bgl],
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
            bind_group_layouts: &[&rasterize_error_bgl, &params_bgl],
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
            bind_group_layouts: &[&bin_polygons_bgl, &params_bgl],
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
            bind_group_layouts: &[&select_bgl, &params_bgl],
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
            bind_group_layouts: &[&init_framebuffers_bgl, &params_bgl],
            immediate_size: 0,
        });
        let init_framebuffers_pipeline = create_init_framebuffers_pipeline(
            &device,
            &init_framebuffers_pipeline_layout,
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
                BindGroupEntry { binding: 5, resource: chain_framebuffers_buf.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: chain_total_errors_buf.as_entire_binding() },
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
            mutate_pipeline,
            rasterize_error_pipeline,
            select_pipeline,
            bin_polygons_pipeline,
            init_framebuffers_pipeline,
            rasterize_error_shader,
            rasterize_error_pipeline_layout,
            bin_polygons_pipeline_layout,
            init_framebuffers_pipeline_layout,
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
            chain_count,
            offspring_capacity: offspring_capacity as u32,
            image_width,
            image_height,
            rasterize_wg,
            num_tiles,
        }
    }

    /// Recreate the rasterize_error pipeline with a new workgroup size.
    /// Also recreates the bin_polygons pipeline since tile size must match.
    /// This is called between batches when the user changes the workgroup size.
    pub fn set_rasterize_wg(&mut self, wg: [u32; 2]) {
        if wg == self.rasterize_wg {
            return;
        }
        println!(
            "Recreating rasterize_error + bin_polygons pipelines: {}x{} -> {}x{}",
            self.rasterize_wg[0], self.rasterize_wg[1], wg[0], wg[1]
        );
        let (pipeline, shader) = create_rasterize_pipeline(
            &self.device,
            &self.rasterize_error_pipeline_layout,
            wg,
            &self.pipeline_cache,
        );
        self.rasterize_error_pipeline = pipeline;
        self.rasterize_error_shader = shader;

        self.bin_polygons_pipeline = create_bin_polygons_pipeline(
            &self.device,
            &self.bin_polygons_pipeline_layout,
            wg,
            &self.pipeline_cache,
        );

        self.init_framebuffers_pipeline = create_init_framebuffers_pipeline(
            &self.device,
            &self.init_framebuffers_pipeline_layout,
            wg,
            &self.pipeline_cache,
        );

        self.rasterize_wg = wg;
        self.num_tiles = self.image_width.div_ceil(wg[0]) * self.image_height.div_ceil(wg[1]);
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
    let source = include_str!("../shaders/rasterize_error.wgsl")
        .replace("const WG_X: u32 = 16;", &format!("const WG_X: u32 = {};", wg[0]))
        .replace("const WG_Y: u32 = 16;", &format!("const WG_Y: u32 = {};", wg[1]));

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("rasterize_error_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("rasterize_error_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: Some(cache),
    });

    (pipeline, shader)
}

/// Create a bin_polygons compute pipeline with tile size matching the rasterize workgroup.
fn create_bin_polygons_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> ComputePipeline {
    let source = include_str!("../shaders/bin_polygons.wgsl")
        .replace("const TILE_W: u32 = 16;", &format!("const TILE_W: u32 = {};", wg[0]))
        .replace("const TILE_H: u32 = 16;", &format!("const TILE_H: u32 = {};", wg[1]));

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bin_polygons_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bin_polygons_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: Some(cache),
    })
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
        compilation_options: Default::default(),
        cache: Some(cache),
    })
}

/// Get the disk path for storing this adapter's pipeline cache.
/// Returns `None` if the backend doesn't support pipeline caching.
fn pipeline_cache_path(adapter_info: &AdapterInfo) -> Option<PathBuf> {
    let key = wgpu::util::pipeline_cache_key(adapter_info)?;
    Some(PathBuf::from(format!("pipeline_cache_{}.bin", key)))
}
