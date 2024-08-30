use std::{borrow::Cow, mem};

use wgpu::{
    vertex_attr_array, BlendState, ComputePipeline, Device, Gles3MinorVersion, InstanceFlags,
    SubmissionIndex,
};

use crate::{
    buffer_dimensions::BufferDimensions,
    models::drawing::Drawing,
    settings::{MAX_ERROR_PER_PIXEL, PER_POINT_MULTIPLIER},
    texture::Texture,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
}

#[derive(Debug)]
pub struct GpuPipeline {
    pub best_drawing_bytes: Vec<u8>,
    pub best_drawing: Drawing,
    pub buffer_dimensions: BufferDimensions,
    pub compute_bind_group: wgpu::BindGroup,
    pub compute_pipeline: ComputePipeline,
    pub device: Device,
    pub drawing_output_buffer: wgpu::Buffer,
    pub drawing_texture: wgpu::Texture,
    pub error_output_buffer: wgpu::Buffer,
    pub error_source_buffer: wgpu::Buffer,
    pub h: usize,
    pub queue: wgpu::Queue,
    pub render_pipeline: wgpu::RenderPipeline,
    pub texture_extent: wgpu::Extent3d,
    pub w: usize,
}

impl GpuPipeline {
    pub async fn new(source_bytes: Vec<u8>, w: usize, h: usize) -> GpuPipeline {
        let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
        let flags: InstanceFlags = InstanceFlags::DEBUG;
        let gles_minor_version: Gles3MinorVersion = Gles3MinorVersion::Automatic;
        let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags,
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version,
        });

        let adapter_options = &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        };
        let adapter = instance
            .request_adapter(adapter_options)
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // It is a WebGPU requirement that ImageCopyBuffer.layout.bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0
        // So we calculate padded_bytes_per_row by rounding unpadded_bytes_per_row
        // up to the next multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
        // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
        let buffer_dimensions = BufferDimensions::new(w, h); // FIXME do this inside init with new calculated w, h
        assert!(w * h > 0);

        // The output buffer lets us retrieve the data as an array
        let drawing_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // will copy this to error_output_buffer after the compute pass
        let error_source_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (buffer_dimensions.width * buffer_dimensions.height * 4) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // final error output per pixel
        let error_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (buffer_dimensions.width * buffer_dimensions.height * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_extent = wgpu::Extent3d {
            width: buffer_dimensions.width as u32,
            height: buffer_dimensions.height as u32,
            depth_or_array_layers: 1,
        };

        let texture_format = wgpu::TextureFormat::Rgba8Unorm;

        // The render pipeline renders data into this texture
        let drawing_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        let view = drawing_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0, // 'source' bytes loaded from target image
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1, // 'current' render target for drawing
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2, // 'error' output <-- error=sqrt((re*re)+(ge*ge)+(be*be)) in RGBA32Float
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (buffer_dimensions.width * buffer_dimensions.height * 4) as u64,
                            ),
                        },
                        count: None,
                    },
                ],
                label: Some("compute_bind_group_layout"),
            });

        // doubt this makes much sense, attempting to pass in the dimensions as a 1x2 R32Uint texture
        let dimensions = (w as u32, h as u32);
        // let dimensions_texture = Texture::from_dimensions(&device, &queue, dimensions).unwrap();

        let source_texture = Texture::from_bytes(
            &device,
            &queue,
            &source_bytes.as_slice(),
            dimensions,
            &"source",
        )
        .expect("Failed to create source_texture");

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    // source image texture WxH Rgba8Unorm
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_texture.view),
                },
                wgpu::BindGroupEntry {
                    // render target texture WxH Rgba8Unorm
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    // error calc output texture WxH Rgba8Unorm
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        error_source_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
            label: Some("compute_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: (mem::size_of::<f32>() * 8) as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x4, 1 => Float32x4],
        };

        let mut primitive = wgpu::PrimitiveState::default();
        primitive.cull_mode = None;

        let blend_state: BlendState = BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(), // TODO: investigate if there's a better option
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(), // TODO: investigate if there's a better option
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: primitive,
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("error.compute2.wgsl"))),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &compute_module,
            entry_point: "main",
            compilation_options: Default::default(), // TODO: investigate if there's a better option
            cache: None,
        });

        let best_drawing: Drawing = Drawing::new_random();
        let best_drawing_bytes: Vec<u8> = vec![]; // can only set after drawing in post_init

        GpuPipeline {
            best_drawing_bytes,
            best_drawing,
            buffer_dimensions,
            compute_bind_group,
            compute_pipeline,
            device,
            drawing_output_buffer,
            drawing_texture,
            error_output_buffer,
            error_source_buffer,
            h,
            queue,
            render_pipeline,
            texture_extent,
            w,
        }
    }

    pub async fn draw(&self, drawing: &Drawing) -> SubmissionIndex {
        let vertices: Vec<Vertex> = drawing.to_vertices();

        // create buffer, write buffer (bytemuck?)
        let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );

        let command_buffer: wgpu::CommandBuffer = {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let view = &self
                .drawing_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // Set the background to be white
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), // WHY DOES DRAWING WHITE TRIANGLES ON TOP OF THIS DO ANYTHING?
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            // rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..vertices.len() as u32, 0..vertices.len() as u32);

            // encoder methods like begin_render_pass and copy_texture_to_buffer take a &'pass mut self
            // drop rpass before copy_texture_to_buffer to avoid: cannot borrow `encoder` as mutable more than once at a time
            drop(rpass);

            // Copy the data from the texture to the buffer
            encoder.copy_texture_to_buffer(
                self.drawing_texture.as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: &self.drawing_output_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(self.buffer_dimensions.padded_bytes_per_row as u32),
                        rows_per_image: None,
                    },
                },
                self.texture_extent,
            );

            encoder.finish()
        };

        self.queue.submit(Some(command_buffer))
    }

    pub async fn calculate_error(&self, width: u32, height: u32) -> SubmissionIndex {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("calculate_error_command_encoder"),
            });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.compute_bind_group, &[]);
        cpass.dispatch_workgroups(width / 8, height / 8, 1); // compute shader workgroup_size is (8, 8, 1)
        drop(cpass);

        encoder.copy_buffer_to_buffer(
            &self.error_source_buffer,
            0,
            &self.error_output_buffer,
            0,
            (width * height * 4) as u64,
        );

        self.queue.submit(Some(encoder.finish()))
    }

    pub async fn draw_and_evaluate(&self, drawing: &mut Drawing) -> (f32, f32, Vec<u8>, Vec<u8>) {
        // step 1 - render pipeline --> draw our triangles to a texture
        let draw_si = self.draw(&drawing).await;
        let best_drawing_bytes = self.get_bytes(&self.drawing_output_buffer, draw_si).await;

        // Step 2 - compute pipeline --> diff drawing texture vs source texture
        let ce_si = self.calculate_error(self.w as u32, self.h as u32).await;

        // Step 3 - calculate error and error heatmap (sum output of compute pipeline)
        // TODO: parallel reduction on GPU, something like https://eximia.co/implementing-parallel-reduction-in-cuda/
        let error_buffer = self.get_bytes(&self.error_output_buffer, ce_si).await;

        let (error, error_heatmap) = calculate_error_from_gpu(&error_buffer);
        let max_total_error: f32 = MAX_ERROR_PER_PIXEL * self.w as f32 * self.h as f32;
        let mut fitness: f32 = 100.0 * (1.0 - error / max_total_error);
        let penalty = fitness * PER_POINT_MULTIPLIER * drawing.num_points() as f32;
        fitness -= penalty;
        drawing.fitness = fitness;

        (error, fitness, best_drawing_bytes, error_heatmap)
    }

    async fn get_bytes(&self, output_buffer: &wgpu::Buffer, si: SubmissionIndex) -> Vec<u8> {
        let buffer_slice = output_buffer.slice(..);

        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device
            .poll(wgpu::MaintainBase::WaitForSubmissionIndex(si));

        if let Some(Ok(())) = receiver.receive().await {
            let padded_buffer = buffer_slice.get_mapped_range();
            let vec = padded_buffer.to_vec();
            drop(padded_buffer); // avoid --> "You cannot unmap a buffer that still has accessible mapped views."
            output_buffer.unmap(); // avoid --> Buffer ObjectId { id: Some(1) } is already mapped' (breaks looping logic)
            return vec;
        } else {
            // should we ever end up here?
            output_buffer.unmap(); // probably makes no difference but just to be safe
            return vec![];
        }
    }
}

// we are now calculating sqrt(((re * re) + (ge * ge) + (be * be))) on the gpu
// error_buffer is raw bytes straight out of the gpu so need to convert chunks of 4 back into f32
fn calculate_error_from_gpu(error_buffer: &Vec<u8>) -> (f32, Vec<u8>) {
    let error_buffer_f32: Vec<f32> = error_buffer
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    // FIXME: all errors coming back as 0
    let zeros = error_buffer_f32.iter().filter(|&v| *v == 0.0).count();
    println!(
        "{} out of {} errors came back 0",
        zeros,
        error_buffer_f32.len()
    );

    let mut error_heatmap: Vec<u8> = Vec::with_capacity(error_buffer.len() * 4);
    let mut error: f32 = 0.0;
    error_buffer_f32.into_iter().for_each(|sqrt| {
        error += sqrt; // FIXME: always 0 ?
        let err_color = f32::floor(255.0 * (1.0 - sqrt / MAX_ERROR_PER_PIXEL)) as u8;
        error_heatmap.extend_from_slice(&[255, err_color, err_color, 255]);
    });

    (error, error_heatmap)
}
