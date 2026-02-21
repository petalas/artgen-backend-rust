use std::num::NonZeroU64;
use std::path::PathBuf;

use wgpu::*;
use wgpu::util::DeviceExt;

use super::buffers::{
    ControlFlags, GpuDrawingState, GpuParams, GPU_DRAWING_STATE_SIZE,
};

/// Directory where pipeline cache files are stored.
fn cache_dir() -> PathBuf {
    // Use a `.cache` subdirectory next to the executable, falling back to current dir
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    base.join(".cache")
}

/// Load pipeline cache data from disk for the given adapter, if available.
fn load_pipeline_cache_data(adapter_info: &AdapterInfo) -> Option<Vec<u8>> {
    let key = wgpu::util::pipeline_cache_key(adapter_info)?;
    let path = cache_dir().join(&key);
    match std::fs::read(&path) {
        Ok(data) => {
            println!("Loaded pipeline cache from {} ({} bytes)", path.display(), data.len());
            Some(data)
        }
        Err(_) => None,
    }
}

/// Save pipeline cache data to disk for the given adapter.
fn save_pipeline_cache_data(adapter_info: &AdapterInfo, cache: &PipelineCache) {
    let key = match wgpu::util::pipeline_cache_key(adapter_info) {
        Some(k) => k,
        None => return,
    };
    let dir = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("Failed to create pipeline cache directory {}: {}", dir.display(), e);
        return;
    }
    if let Some(data) = cache.get_data() {
        let path = dir.join(&key);
        let temp_path = path.with_extension("tmp");
        // Atomic write: write to temp file, then rename
        match std::fs::write(&temp_path, &data) {
            Ok(()) => {
                if let Err(e) = std::fs::rename(&temp_path, &path) {
                    eprintln!("Failed to rename pipeline cache file: {}", e);
                    // Clean up temp file on rename failure
                    let _ = std::fs::remove_file(&temp_path);
                } else {
                    println!("Saved pipeline cache to {} ({} bytes)", path.display(), data.len());
                }
            }
            Err(e) => {
                eprintln!("Failed to write pipeline cache: {}", e);
            }
        }
    }
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
    pub control_staging_bufs: [Buffer; 2],
    pub fitness_packed_buf: Buffer,
    pub fitness_staging_bufs: [Buffer; 2],

    // Compute pipelines
    pub mutate_pipeline: ComputePipeline,
    pub rasterize_error_pipeline: ComputePipeline,
    pub select_pipeline: ComputePipeline,

    // Rasterize pipeline recreation support — stored for creating new pipeline variants
    rasterize_error_shader: ShaderModule,
    rasterize_error_pipeline_layout: PipelineLayout,
    pipeline_cache: Option<PipelineCache>,

    // Bind groups
    pub mutate_bind_group: BindGroup,
    pub rasterize_error_bind_group: BindGroup,
    pub select_bind_group: BindGroup,

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

        let adapter_info = adapter.get_info();
        println!("Selected GPU adapter: {:?}", adapter_info.name);

        // Cap chain_count to fit within the adapter's max storage buffer binding size.
        // Largest buffer is chain_states or working_states (chain_count × GPU_DRAWING_STATE_SIZE).
        let adapter_limits = adapter.limits();
        let max_ssbo = adapter_limits.max_storage_buffer_binding_size as u64;
        let max_chains_by_buffer = max_ssbo / (GPU_DRAWING_STATE_SIZE as u64);
        let chain_count = chain_count.min(max_chains_by_buffer as u32);

        // Calculate the largest buffer we actually need (working_states with offspring_capacity)
        let max_lambda = crate::settings::GPU_MAX_LAMBDA as u64;
        let max_buffer_needed = (chain_count as u64) * max_lambda * (GPU_DRAWING_STATE_SIZE as u64);
        // Clamp to adapter limit (don't request more than hardware supports)
        let max_buffer_size = max_buffer_needed.min(max_ssbo as u64);

        let required_limits = Limits {
            max_storage_buffer_binding_size: max_buffer_size as u32,
            max_buffer_size,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 5, // select shader uses 5 storage bindings (params moved to push constants)
            max_push_constant_size: std::mem::size_of::<GpuParams>() as u32, // 128 bytes — Vulkan minimum guarantee
            ..Limits::downlevel_defaults()
        };

        // Request PIPELINE_CACHE feature if the adapter supports it (Vulkan only)
        let adapter_features = adapter.features();
        let pipeline_cache_supported = adapter_features.contains(Features::PIPELINE_CACHE);
        let mut required_features = Features::TIMESTAMP_QUERY | Features::PUSH_CONSTANTS;
        if pipeline_cache_supported {
            required_features |= Features::PIPELINE_CACHE;
            println!("Pipeline cache feature supported — enabling shader cache");
        } else {
            println!("Pipeline cache feature not supported — shaders will recompile each launch");
        }

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("gpu_evolver_device"),
                required_features,
                required_limits,
                memory_hints: MemoryHints::Performance,
            }, None)
            .await
            .expect("Failed to create GPU device");

        // --- Pipeline cache ---
        // Load cached shader binaries from disk to speed up pipeline creation.
        // If the cache file doesn't exist, is corrupt, or the adapter doesn't support
        // caching, we gracefully fall back to an empty cache (fallback: true).
        let pipeline_cache = if pipeline_cache_supported {
            let cache_data = load_pipeline_cache_data(&adapter_info);
            // SAFETY: cache data (if Some) was previously returned from PipelineCache::get_data()
            // and saved to disk. fallback=true ensures corrupt/incompatible data creates an empty cache.
            Some(unsafe {
                device.create_pipeline_cache(&PipelineCacheDescriptor {
                    label: Some("artgen_pipeline_cache"),
                    data: cache_data.as_deref(),
                    fallback: true,
                })
            })
        } else {
            None
        };

        // --- Buffer sizes ---
        let chain_states_size = (chain_count as usize) * GPU_DRAWING_STATE_SIZE;
        // Offspring capacity: chain_count * max_lambda, capped by SSBO limit
        let max_lambda = crate::settings::GPU_MAX_LAMBDA as usize;
        let offspring_capacity = ((max_ssbo as usize) / GPU_DRAWING_STATE_SIZE)
            .min(chain_count as usize * max_lambda);
        let working_states_size = offspring_capacity * GPU_DRAWING_STATE_SIZE;
        let error_accumulators_size = offspring_capacity * 4; // u32 per offspring

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

        // Rasterize+Error: working_states(read), reference_image(texture), error_accumulators(rw); params via push constants
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
            ],
        });

        // Select: chain_states(rw), working_states(read), error_accumulators(rw), control(rw), fitness_packed(rw); params via push constants
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
            ],
        });

        // --- Compute pipelines ---
        let push_constant_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<GpuParams>() as u32,
        };

        let mutate_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mutate_layout"),
            bind_group_layouts: &[&mutate_bgl],
            push_constant_ranges: &[push_constant_range.clone()],
        });
        let mutate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("mutate_pipeline"),
            layout: Some(&mutate_pipeline_layout),
            module: &mutate_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: pipeline_cache.as_ref(),
        });

        let rasterize_error_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rasterize_error_layout"),
            bind_group_layouts: &[&rasterize_error_bgl],
            push_constant_ranges: &[push_constant_range.clone()],
        });
        let rasterize_wg = [
            crate::settings::RASTERIZE_WG_X_DEFAULT,
            crate::settings::RASTERIZE_WG_Y_DEFAULT,
        ];
        let (rasterize_error_pipeline, rasterize_error_shader) = create_rasterize_pipeline(
            &device,
            &rasterize_error_pipeline_layout,
            pipeline_cache.as_ref(),
            rasterize_wg,
        );

        let select_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("select_layout"),
            bind_group_layouts: &[&select_bgl],
            push_constant_ranges: &[push_constant_range],
        });
        let select_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("select_pipeline"),
            layout: Some(&select_pipeline_layout),
            module: &select_shader,
            entry_point: "select_main",
            compilation_options: Default::default(),
            cache: pipeline_cache.as_ref(),
        });

        // Save compiled pipeline cache to disk after all pipelines are created
        if let Some(ref cache) = pipeline_cache {
            save_pipeline_cache_data(&adapter_info, cache);
        }

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
            control_staging_bufs,
            fitness_packed_buf,
            fitness_staging_bufs,
            mutate_pipeline,
            rasterize_error_pipeline,
            select_pipeline,
            rasterize_error_shader,
            rasterize_error_pipeline_layout,
            pipeline_cache,
            timestamp_query_set,
            timestamp_resolve_buf,
            timestamp_staging_bufs,
            timestamp_period,
            mutate_bind_group,
            rasterize_error_bind_group,
            select_bind_group,
            chain_count,
            offspring_capacity: offspring_capacity as u32,
            image_width,
            image_height,
            rasterize_wg,
        }
    }

    /// Recreate the rasterize_error pipeline with a new workgroup size.
    /// This is called between batches when the user changes the workgroup size.
    pub fn set_rasterize_wg(&mut self, wg: [u32; 2]) {
        if wg == self.rasterize_wg {
            return;
        }
        println!(
            "Recreating rasterize_error pipeline: {}x{} -> {}x{}",
            self.rasterize_wg[0], self.rasterize_wg[1], wg[0], wg[1]
        );
        let (pipeline, shader) = create_rasterize_pipeline(
            &self.device,
            &self.rasterize_error_pipeline_layout,
            self.pipeline_cache.as_ref(),
            wg,
        );
        self.rasterize_error_pipeline = pipeline;
        self.rasterize_error_shader = shader;
        self.rasterize_wg = wg;
    }
}

/// Create a rasterize_error compute pipeline with the given workgroup size.
/// Uses string replacement on the shader source since naga 22.x does not support
/// override constants in @workgroup_size or const expressions.
fn create_rasterize_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    cache: Option<&PipelineCache>,
    wg: [u32; 2],
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
        entry_point: "main",
        compilation_options: Default::default(),
        cache,
    });

    (pipeline, shader)
}
