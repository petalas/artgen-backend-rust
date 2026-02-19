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
    pub reference_image_buf: Buffer,
    pub render_targets_buf: Buffer,
    pub error_accumulators_buf: Buffer,
    pub control_flags_buf: Buffer,
    pub params_buf: Buffer,
    pub readback_staging_buf: Buffer,
    pub control_staging_buf: Buffer,

    // Compute pipelines
    pub mutate_pipeline: ComputePipeline,
    pub rasterize_pipeline: ComputePipeline,
    pub error_reduce_pipeline: ComputePipeline,
    pub select_pipeline: ComputePipeline,
    pub migrate_pipeline: ComputePipeline,

    // Bind groups
    pub mutate_bind_group: BindGroup,
    pub rasterize_bind_group: BindGroup,
    pub error_reduce_bind_group: BindGroup,
    pub select_bind_group: BindGroup,
    pub migrate_bind_group: BindGroup, // same layout as select, reused

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
        assert_eq!(initial_states.len(), chain_count as usize);
        assert_eq!(reference_rgba.len(), (image_width * image_height * 4) as usize);

        // --- Device + Queue ---
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        println!("GPU adapter: {:?}", adapter.get_info().name);

        let required_limits = Limits {
            max_storage_buffer_binding_size: 256 * 1024 * 1024, // 256 MB
            max_buffer_size: 256 * 1024 * 1024,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_invocations_per_workgroup: 256,
            ..Limits::downlevel_defaults()
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("gpu_evolver_device"),
                required_features: Features::empty(),
                required_limits,
                memory_hints: MemoryHints::Performance,
            }, None)
            .await
            .expect("Failed to create GPU device");

        // --- Buffer sizes ---
        let pixels_per_chain = (image_width * image_height) as usize;
        let chain_states_size = (chain_count as usize) * GPU_DRAWING_STATE_SIZE;
        let render_targets_size = (chain_count as usize) * pixels_per_chain * 4; // u32 per pixel
        let error_accumulators_size = (chain_count as usize) * 4; // u32 per chain

        // --- Create buffers ---

        // Chain states (current best per chain)
        let chain_states_bytes: Vec<u8> = initial_states
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

        // Reference image — pack RGBA bytes as u32 array
        let ref_packed: Vec<u32> = reference_rgba
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let reference_image_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("reference_image"),
            contents: bytemuck::cast_slice(&ref_packed),
            usage: BufferUsages::STORAGE,
        });

        // Render targets (per-chain rendered images)
        let render_targets_buf = device.create_buffer(&BufferDescriptor {
            label: Some("render_targets"),
            size: render_targets_size as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Error accumulators (atomic u32 per chain)
        let error_accumulators_buf = device.create_buffer(&BufferDescriptor {
            label: Some("error_accumulators"),
            size: error_accumulators_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        // Control staging (for polling new_best_found)
        let control_staging_buf = device.create_buffer(&BufferDescriptor {
            label: Some("control_staging"),
            size: std::mem::size_of::<ControlFlags>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Shader modules ---
        let mutate_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mutate_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/mutate.wgsl").into()),
        });

        let rasterize_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rasterize_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rasterize.wgsl").into()),
        });

        let error_reduce_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("error_reduce_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/error_reduce.wgsl").into()),
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

        // Rasterize: working_states(read), render_targets(rw), params(uniform)
        let rasterize_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rasterize_bgl"),
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

        // Error reduce: render_targets(read), reference_image(read), error_accumulators(rw), params(uniform)
        let error_reduce_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("error_reduce_bgl"),
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

        let rasterize_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rasterize_layout"),
            bind_group_layouts: &[&rasterize_bgl],
            push_constant_ranges: &[],
        });
        let rasterize_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("rasterize_pipeline"),
            layout: Some(&rasterize_pipeline_layout),
            module: &rasterize_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let error_reduce_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("error_reduce_layout"),
            bind_group_layouts: &[&error_reduce_bgl],
            push_constant_ranges: &[],
        });
        let error_reduce_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("error_reduce_pipeline"),
            layout: Some(&error_reduce_pipeline_layout),
            module: &error_reduce_shader,
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

        let migrate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("migrate_pipeline"),
            layout: Some(&select_pipeline_layout),
            module: &select_shader,
            entry_point: "migrate_main",
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

        let rasterize_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("rasterize_bg"),
            layout: &rasterize_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: working_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: render_targets_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let error_reduce_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("error_reduce_bg"),
            layout: &error_reduce_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: render_targets_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: reference_image_buf.as_entire_binding() },
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
            ],
        });

        Self {
            device,
            queue,
            chain_states_buf,
            working_states_buf,
            reference_image_buf,
            render_targets_buf,
            error_accumulators_buf,
            control_flags_buf,
            params_buf,
            readback_staging_buf,
            control_staging_buf,
            mutate_pipeline,
            rasterize_pipeline,
            error_reduce_pipeline,
            select_pipeline,
            migrate_pipeline,
            mutate_bind_group,
            rasterize_bind_group,
            error_reduce_bind_group,
            select_bind_group,
            migrate_bind_group,
            chain_count,
            image_width,
            image_height,
        }
    }
}
