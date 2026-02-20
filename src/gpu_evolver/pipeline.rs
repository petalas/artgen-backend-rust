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
    pub params_buf: Buffer,
    pub readback_staging_buf: Buffer,
    pub control_staging_bufs: [Buffer; 2],
    pub fitness_packed_buf: Buffer,
    pub fitness_staging_bufs: [Buffer; 2],

    // Compute pipelines
    pub mutate_pipeline: ComputePipeline,
    pub rasterize_error_pipeline: ComputePipeline,
    pub select_pipeline: ComputePipeline,
    pub migrate_intra_pipeline: ComputePipeline,
    pub migrate_inter_pipeline: ComputePipeline,

    // Bind groups
    pub mutate_bind_group: BindGroup,
    pub rasterize_error_bind_group: BindGroup,
    pub select_bind_group: BindGroup,
    pub migrate_bind_group: BindGroup, // same layout as select, reused

    // Timestamp profiling
    pub timestamp_query_set: QuerySet,
    pub timestamp_resolve_buf: Buffer,
    pub timestamp_staging_bufs: [Buffer; 2],
    pub timestamp_period: f32,

    // Config
    pub chain_count: u32,
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
        gpu_params: &GpuParams,
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
            ..Default::default()
        });

        // Enumerate all adapters and prefer discrete GPU
        let adapters: Vec<_> = instance.enumerate_adapters(Backends::VULKAN | Backends::DX12);
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

        println!("Selected GPU adapter: {:?}", adapter.get_info().name);

        // Cap chain_count to fit within the adapter's max storage buffer binding size.
        // Largest buffer is chain_states or working_states (chain_count × GPU_DRAWING_STATE_SIZE).
        let adapter_limits = adapter.limits();
        let max_ssbo = adapter_limits.max_storage_buffer_binding_size as u64;
        let max_chains_by_buffer = max_ssbo / (GPU_DRAWING_STATE_SIZE as u64);
        let chain_count = chain_count.min(max_chains_by_buffer as u32);

        // Calculate the largest buffer we actually need (chain_states / working_states)
        let max_buffer_needed = (chain_count as u64) * (GPU_DRAWING_STATE_SIZE as u64);
        // Clamp to adapter limit (don't request more than hardware supports)
        let max_buffer_size = max_buffer_needed.min(max_ssbo as u64);

        let required_limits = Limits {
            max_storage_buffer_binding_size: max_buffer_size as u32,
            max_buffer_size,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 6, // select shader uses 5 storage bindings
            ..Limits::downlevel_defaults()
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("gpu_evolver_device"),
                required_features: Features::TIMESTAMP_QUERY,
                required_limits,
                memory_hints: MemoryHints::Performance,
            }, None)
            .await
            .expect("Failed to create GPU device");

        // --- Buffer sizes ---
        let chain_states_size = (chain_count as usize) * GPU_DRAWING_STATE_SIZE;
        let error_accumulators_size = (chain_count as usize) * 4; // u32 per chain

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

        // Working states (mutated candidates)
        let working_states_buf = device.create_buffer(&BufferDescriptor {
            label: Some("working_states"),
            size: chain_states_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
            ImageCopyTexture {
                texture: &reference_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            reference_rgba,
            ImageDataLayout {
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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

        // Params uniform
        let params_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Readback staging (one DrawingState)
        let readback_staging_buf = device.create_buffer(&BufferDescriptor {
            label: Some("readback_staging"),
            size: GPU_DRAWING_STATE_SIZE as u64,
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

        // --- Timestamp query profiling ---
        let timestamp_query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("timestamp_queries"),
            ty: QueryType::Timestamp,
            count: 8, // 2 per pass × 4 passes
        });

        let timestamp_resolve_buf = device.create_buffer(&BufferDescriptor {
            label: Some("timestamp_resolve"),
            size: 8 * 8, // 8 × u64
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

        let rasterize_error_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rasterize_error_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rasterize_error.wgsl").into()),
        });

        let select_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("select_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/select.wgsl").into()),
        });

        // --- Bind group layouts ---

        // Mutate: chain_states(read), working_states(rw), params(uniform)
        let mutate_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mutate_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Rasterize+Error: working_states(read), reference_image(texture), error_accumulators(rw), params(uniform)
        let rasterize_error_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rasterize_error_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Select: chain_states(rw), working_states(read), error_accumulators(rw), control(rw), params(uniform)
        let select_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("select_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // --- Compute pipelines ---
        let mutate_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mutate_layout"),
            bind_group_layouts: &[&mutate_bgl],
            push_constant_ranges: &[],
        });
        let mutate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("mutate_pipeline"),
            layout: Some(&mutate_pipeline_layout),
            module: &mutate_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let rasterize_error_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rasterize_error_layout"),
            bind_group_layouts: &[&rasterize_error_bgl],
            push_constant_ranges: &[],
        });
        let rasterize_error_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("rasterize_error_pipeline"),
            layout: Some(&rasterize_error_pipeline_layout),
            module: &rasterize_error_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let select_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("select_layout"),
            bind_group_layouts: &[&select_bgl],
            push_constant_ranges: &[],
        });
        let select_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("select_pipeline"),
            layout: Some(&select_pipeline_layout),
            module: &select_shader,
            entry_point: "select_main",
            compilation_options: Default::default(),
            cache: None,
        });

        let migrate_intra_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("migrate_intra_pipeline"),
            layout: Some(&select_pipeline_layout),
            module: &select_shader,
            entry_point: "migrate_intra_main",
            compilation_options: Default::default(),
            cache: None,
        });

        let migrate_inter_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("migrate_inter_pipeline"),
            layout: Some(&select_pipeline_layout),
            module: &select_shader,
            entry_point: "migrate_inter_main",
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Bind groups ---
        let mutate_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mutate_bg"),
            layout: &mutate_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let rasterize_error_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("rasterize_error_bg"),
            layout: &rasterize_error_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&reference_view) },
                BindGroupEntry { binding: 2, resource: error_accumulators_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
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
                BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: fitness_packed_buf.as_entire_binding() },
            ],
        });

        // Migrate uses the same bind group layout and bindings as select
        let migrate_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("migrate_bg"),
            layout: &select_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: chain_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: error_accumulators_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: control_flags_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: fitness_packed_buf.as_entire_binding() },
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
            params_buf,
            readback_staging_buf,
            control_staging_bufs,
            fitness_packed_buf,
            fitness_staging_bufs,
            mutate_pipeline,
            rasterize_error_pipeline,
            select_pipeline,
            migrate_intra_pipeline,
            migrate_inter_pipeline,
            timestamp_query_set,
            timestamp_resolve_buf,
            timestamp_staging_bufs,
            timestamp_period,
            mutate_bind_group,
            rasterize_error_bind_group,
            select_bind_group,
            migrate_bind_group,
            chain_count,
            image_width,
            image_height,
        }
    }
}
