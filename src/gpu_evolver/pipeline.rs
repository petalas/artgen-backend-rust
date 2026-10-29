use std::num::NonZeroU64;

use wgpu::*;
use wgpu::util::DeviceExt;

use super::execution::{AllocationPlan, RasterGrid, ResourceSizes, Workload};

use super::buffers::{
    ControlFlags, GpuDrawingState, GpuParams, GPU_DRAWING_STATE_SIZE,
};
use super::changes::{
    pack_dirty_box, ChangedIndices, CHANGED_WORDS_OFFSET, DIRTY_MAX_OFFSET, DIRTY_MIN_OFFSET,
    GPU_CHAIN_CHANGE_SIZE,
};

use super::programs::{GpuPrograms, ProgramBindings, RasterSchedule, WorklistStorage};

pub use super::programs::{
    benchmark_mutate_compilation_options, benchmark_mutate_group_size_from_source,
    BenchmarkMutateOptions, ShaderVariantInstaller,
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
    pub readback_staging_bufs: [Buffer; 2], // best-drawing copy, one per staging slot
    pub multi_readback_staging_buf: Buffer,  // persistent staging for batch chain readback
    pub eval_fitness_staging_buf: Buffer,    // persistent staging for evaluate_chain_fitness readback
    pub control_staging_bufs: [Buffer; 2],
    pub fitness_packed_buf: Buffer,
    pub fitness_staging_bufs: [Buffer; 2],

    // Tile culling buffers
    pub tile_data_buf: Buffer,
    pub tile_counts_buf: Buffer,
    // The (offspring, tile) pairs that need raster, after an entry count.
    pub raster_work_buf: Buffer,

    // Incremental evaluation buffers
    pub chain_framebuffers_buf: Buffer,
    pub chain_total_errors_buf: Buffer,
    pub chain_update_flags_buf: Buffer,
    // One change record per chain (changes::GpuChainChange): select writes it,
    // update_framebuffers reads it. Only write_chain_states and
    // forget_chain_changes write it from the host.
    pub(super) chain_changes_buf: Buffer,

    // Selected programs, specialization and compilation caches share one owner.
    pub(super) programs: GpuPrograms,

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

        let allocation = AllocationPlan::new(
            chain_count, [image_width, image_height],
            adapter.limits().max_storage_buffer_binding_size,
        );
        let chain_count = allocation.chain_count;
        let max_buffer_size = allocation.max_buffer_size;

        let required_limits = Limits {
            max_storage_buffer_binding_size: max_buffer_size,
            max_buffer_size,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroup_size_x: 512, // 1D workgroup layout needs up to 512 in x (for 32x16 tile)
            max_storage_buffers_per_shader_stage: 8, // select uses 8 bindings (chain_states, working_states, error_accum, control, fitness, chain_update_flags, chain_total_errors, chain_changes)
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

        let resources = allocation.resources;
        let offspring_capacity = allocation.offspring_capacity as usize;
        let working_states_size = resources.working;
        let error_accumulators_size = usize::try_from(resources.errors)
            .expect("error accumulator allocation fits host address space");

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

        // Readback staging (one DrawingState per staging slot)
        let readback_staging_bufs = std::array::from_fn(|i| {
            device.create_buffer(&BufferDescriptor {
                label: Some(if i == 0 { "readback_staging_0" } else { "readback_staging_1" }),
                size: GPU_DRAWING_STATE_SIZE as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
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

        let max_num_tiles = RasterGrid::new([image_width, image_height], [8, 8]).tiles();
        let tile_data_buf_size = resources.tile_data;
        let tile_counts_buf_size = resources.tile_counts;

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

        // Zero-initialized: an empty list.
        let raster_work_buf = device.create_buffer(&BufferDescriptor {
            label: Some("raster_work"),
            size: resources.raster_work,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
        // Allocation preserves the existing independent framebuffer clamp.
        let pixels_per_chain = u64::from(image_width) * u64::from(image_height);
        let chain_framebuffers_size = resources.framebuffers;
        let chain_framebuffers_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_framebuffers"),
            size: chain_framebuffers_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Chain total errors: raw u32 error sum per chain
        let chain_total_errors_size = resources.parent_errors;
        let chain_total_errors_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_total_errors"),
            size: chain_total_errors_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Chain update flags: per-chain acceptance flag + dirty bbox for update_framebuffers
        // Layout: [accepted: u32, dirty_bbox_lo: u32, dirty_bbox_hi: u32, padding: u32] × chain_count
        let chain_update_flags_size = resources.updates;
        let chain_update_flags_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_update_flags"),
            size: chain_update_flags_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero-initialized: every chain's change record is unknown.
        let chain_changes_buf = device.create_buffer(&BufferDescriptor {
            label: Some("chain_changes"),
            size: resources.chain_changes,
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
                storage_entry(7, true, 4),
            ],
        });

        // Bin polygons: working_states(read), tile_data(rw), tile_counts(rw),
        //               chain_states(read), raster_work(rw); params via push constants
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
                storage_entry(3, true, GPU_DRAWING_STATE_SIZE as u64),
                storage_entry(4, false, 4),
            ],
        });

        // Select: chain_states(rw), working_states(read), error_accumulators(rw), control(rw), fitness_packed(rw),
        //         chain_update_flags(rw), chain_total_errors(rw), chain_changes(rw); params via push constants
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
                storage_entry(7, false, GPU_CHAIN_CHANGE_SIZE as u64),
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
        // Update framebuffers: chain_states(read), chain_framebuffers(rw), chain_update_flags(read),
        // tile_data(read), tile_counts(read) — the winning offspring's tile polygon lists —,
        // raster_work(rw), chain_changes(read)
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
                storage_entry(5, false, 4),
                storage_entry(6, true, GPU_CHAIN_CHANGE_SIZE as u64),
            ],
        });
        let programs = GpuPrograms::new(
            &device,
            &adapter_info,
            &ProgramBindings {
                params: &params_bgl,
                mutate: &mutate_bgl,
                raster: &rasterize_error_bgl,
                bin: &bin_polygons_bgl,
                select: &select_bgl,
                init: &init_framebuffers_bgl,
                update: &update_framebuffers_bgl,
            },
            rasterize_wg,
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
                BindGroupEntry { binding: 7, resource: raster_work_buf.as_entire_binding() },
            ],
        });

        let bin_polygons_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bin_polygons_bg"),
            layout: &bin_polygons_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: tile_data_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: tile_counts_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: raster_work_buf.as_entire_binding() },
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
                BindGroupEntry { binding: 7, resource: chain_changes_buf.as_entire_binding() },
            ],
        });

        let update_framebuffers_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("update_framebuffers_bg"),
            layout: &update_framebuffers_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: chain_framebuffers_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: chain_update_flags_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: tile_data_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: tile_counts_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: raster_work_buf.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: chain_changes_buf.as_entire_binding() },
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
            readback_staging_bufs,
            multi_readback_staging_buf,
            eval_fitness_staging_buf,
            control_staging_bufs,
            fitness_packed_buf,
            fitness_staging_bufs,
            tile_data_buf,
            tile_counts_buf,
            raster_work_buf,
            chain_framebuffers_buf,
            chain_total_errors_buf,
            chain_update_flags_buf,
            chain_changes_buf,
            programs,
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
        }
    }

    pub(super) fn raster_grid(&self, tile: [u32; 2]) -> RasterGrid {
        RasterGrid::new([self.image_width, self.image_height], tile)
    }

    pub(super) fn workload(&self, params: &crate::mutation_params::MutationParams) -> Workload {
        Workload::for_batch(params, [self.image_width, self.image_height],
            self.chain_count, self.offspring_capacity, self.programs.mutation().group_size())
    }

    pub(super) fn allocated_resources(&self) -> ResourceSizes {
        ResourceSizes {
            parents: self.chain_states_buf.size(), working: self.working_states_buf.size(),
            errors: self.error_accumulators_buf.size(), tile_data: self.tile_data_buf.size(),
            tile_counts: self.tile_counts_buf.size(), raster_work: self.raster_work_buf.size(),
            framebuffers: self.chain_framebuffers_buf.size(),
            parent_errors: self.chain_total_errors_buf.size(), updates: self.chain_update_flags_buf.size(),
            chain_changes: self.chain_changes_buf.size(), fitness: self.fitness_packed_buf.size(),
        }
    }

    /// Installed mutation handle. Its logical width, source and policy are retained together.
    #[must_use]
    pub fn mutate_pipeline(&self) -> &ComputePipeline {
        &self.programs.mutation().pipeline
    }

    /// Installed raster/error handle for the current tile geometry.
    #[must_use]
    pub fn rasterize_error_pipeline(&self) -> &ComputePipeline {
        &self.programs.tiles().raster
    }

    /// Record raster/error for `offspring` offspring over `grid` tiles.
    ///
    /// The bundled raster program runs a fixed number of raster workgroups
    /// that share the (offspring, tile) pairs listed by the bundled binner, or
    /// by a separate worklist pass after a shader-variant binner that lists none.
    /// An indirect dispatch sized to the list cut throughput about 6x on Dozen
    /// (the host fell behind; see LEARNINGS). Raster variants that do not read
    /// the list keep the full grid. The pass must already have the params bind
    /// group at 1.
    pub(super) fn record_raster(&self, pass: &mut ComputePass<'_>, grid: [u32; 2], offspring: u32) {
        let tiles = self.programs.tiles();
        if let RasterSchedule::ListedByPass { pipeline, bind_group } = &tiles.schedule {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(offspring, 1, 1);
        }
        pass.set_pipeline(&tiles.raster);
        pass.set_bind_group(0, &self.rasterize_error_bind_group, &[]);
        if let RasterSchedule::FullGrid = tiles.schedule {
            pass.dispatch_workgroups(grid[0], grid[1], offspring);
        } else {
            let pairs = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(offspring);
            let groups = u32::try_from(pairs.min(u64::from(RASTER_WORKGROUPS)))
                .expect("raster workgroups fit u32");
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    /// Empty the raster worklist outside the batch loop (update_framebuffers
    /// empties it inside). Takes effect at the next submission.
    pub fn reset_raster_work(&self) {
        self.queue.write_buffer(&self.raster_work_buf, 0, &[0; 4]);
    }

    /// Replace the first `states.len()` parent states. Takes effect at the next
    /// submission, and forgets every chain change record with them.
    ///
    /// # Panics
    /// Panics if there are more states than chains.
    pub fn write_chain_states(&self, states: &[GpuDrawingState]) {
        assert!(
            states.len() <= self.chain_count as usize,
            "{} states for {} chains",
            states.len(),
            self.chain_count
        );
        self.queue.write_buffer(&self.chain_states_buf, 0, bytemuck::cast_slice(states));
        self.forget_chain_changes();
    }

    /// Mark every chain change record unknown, so no shortcut can read one that
    /// describes an earlier drawing. The sole host writer of `chain_changes`.
    /// select rewrites a chain's record whenever it accepts a winner, and
    /// `update_framebuffers` reads it only in that iteration, so the batch loop
    /// never needs this. Takes effect at the next submission.
    pub(super) fn forget_chain_changes(&self) {
        let bytes = usize::try_from(self.chain_changes_buf.size())
            .expect("chain change records fit host memory");
        self.queue.write_buffer(&self.chain_changes_buf, 0, &vec![0; bytes]);
    }

    /// Prepare working states `0..count`, copies of parents, for a full
    /// evaluation: the whole image is dirty and the changed indices are
    /// unknown, so binning and raster take no shortcut. Takes effect at the
    /// next submission.
    pub(super) fn mark_full_evaluation(&self, count: u32) {
        assert!(count <= self.offspring_capacity, "{count} exceeds offspring capacity");
        let [lo, hi] = pack_dirty_box([0, 0, self.image_width, self.image_height]);
        let [lo, hi] = [lo.to_le_bytes(), hi.to_le_bytes()];
        let changed: [u8; 8] = bytemuck::cast(ChangedIndices::UNKNOWN.0);
        let words: [(u64, &[u8]); 3] =
            [(DIRTY_MIN_OFFSET, &lo), (DIRTY_MAX_OFFSET, &hi), (CHANGED_WORDS_OFFSET, &changed)];
        for i in 0..u64::from(count) {
            let base = i * GPU_DRAWING_STATE_SIZE as u64;
            for (offset, bytes) in words {
                self.queue.write_buffer(&self.working_states_buf, base + offset, bytes);
            }
        }
    }

    /// Installed selection handle.
    #[must_use]
    pub fn select_pipeline(&self) -> &ComputePipeline {
        self.programs.select()
    }

    /// Installed binning handle for the current tile geometry and diagnostic mode.
    #[must_use]
    pub fn bin_polygons_pipeline(&self) -> &ComputePipeline {
        &self.programs.tiles().bin
    }

    /// Installed framebuffer initialization handle for the current tile geometry.
    #[must_use]
    pub fn init_framebuffers_pipeline(&self) -> &ComputePipeline {
        &self.programs.tiles().init_framebuffers_pipeline
    }

    /// Installed framebuffer update handle for the current tile geometry.
    #[must_use]
    pub fn update_framebuffers_pipeline(&self) -> &ComputePipeline {
        &self.programs.tiles().update_framebuffers_pipeline
    }

    /// Current logical raster tile size.
    #[must_use]
    pub fn rasterize_wg(&self) -> [u32; 2] {
        self.programs.tiles().wg
    }

    /// Number of logical tiles in the current image.
    #[must_use]
    pub fn num_tiles(&self) -> u32 {
        let grid = self.raster_grid(self.rasterize_wg());
        grid.x * grid.y
    }

    /// Recreate the compatible raster, binning and framebuffer programs between batches.
    ///
    /// # Panics
    /// Panics if a benchmark override or tile program cannot be recreated.
    pub fn set_rasterize_wg(&mut self, wg: [u32; 2]) {
        self.try_set_rasterize_wg(wg)
            .expect("workgroup pipeline recreation failed");
    }

    /// Install compatible raster, binning and framebuffer programs for a tile size.
    ///
    /// The caller must own this pipeline exclusively and drain submitted GPU work
    /// before calling. Source overrides and diagnostic mode survive tile changes.
    ///
    /// # Errors
    /// Rejects exhausted benchmark caches, failed source specialization or GPU
    /// compilation errors. Every installed handle and its metadata remain
    /// unchanged on error, including when a framebuffer stage fails last.
    pub fn try_set_rasterize_wg(&mut self, wg: [u32; 2]) -> Result<(), String> {
        self.programs.try_set_rasterize_wg(wg)
    }

    /// Install benchmark shader variants in place of the bundled programs.
    /// Supplies the storage of the worklist pass that older binners need.
    pub fn shader_variants(&mut self) -> ShaderVariantInstaller<'_> {
        self.programs.shader_variants(&WorklistStorage {
            working_states: &self.working_states_buf,
            chain_states: &self.chain_states_buf,
            tile_counts: &self.tile_counts_buf,
            raster_work: &self.raster_work_buf,
        })
    }

    /// Save pipeline cache data to disk for faster startup next time.
    pub fn save_pipeline_cache(&self) {
        self.programs.save_pipeline_cache();
    }
}

/// Raster workgroups sharing the worklist. About 6k pairs need raster per
/// iteration on the v1 fixtures; extra workgroups exit at the loop test.
const RASTER_WORKGROUPS: u32 = 8192;

fn storage_entry(binding: u32, read_only: bool, min_size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_size),
        },
        count: None,
    }
}
