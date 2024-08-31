use std::{borrow::Cow, mem, ops::Deref, sync::Arc, time::Instant};

use image::{imageops::FilterType::Lanczos3, EncodableLayout, ImageReader};
use tracing::info;
use wgpu::{
    vertex_attr_array, BindGroup, BlendState, Buffer, ComputePipeline, Device, Extent3d,
    Gles3MinorVersion, InstanceFlags, Queue, RenderPipeline, SubmissionIndex, Surface,
    SurfaceConfiguration, Texture,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
    window::Window,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
}

use crate::{
    buffer_dimensions::BufferDimensions,
    gpu_pipeline::GpuPipeline,
    models::{
        color::{Color, BLACK, RED, WHITE},
        drawing::Drawing,
        point::Point,
        polygon::Polygon,
    },
    settings::{MAX_ERROR_PER_PIXEL, PER_POINT_MULTIPLIER, TARGET_FRAMETIME},
    texture_wrapper::TextureWrapper,
    utils::{to_color_vec, to_u8_vec},
};

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub enum Rasterizer {
    // Scanline fill, slower but works for any polygons
    Scanline,

    // Advanced Rasterization based on half space functions
    // It is faster than scanline but only implemented for triangles
    // https://web.archive.org/web/20050907040950/http://sw-shader.sourceforge.net:80/rasterizer.html
    HalfSpace,

    // will default to Half-Space unless the polygon has more than 3 points
    #[default]
    Optimal,

    // Offload rendering and array diff to wgpu
    // Still doing the final fitness calculation and mutation generation on the CPU until we can figure out how to do it all as a compute shader.
    GPU,
}

#[derive(Default, Debug)]
pub struct Stats {
    pub generated: usize,
    pub improvements: usize,
    pub cycle_time: usize,
    pub ticks: usize,
}

#[derive(Default)]
pub struct Engine<'a> {
    pub best_drawing_bytes: Vec<u8>,
    pub best_drawing: Drawing,
    pub buffer_dimensions: BufferDimensions,
    pub compute_bind_group: Option<BindGroup>,
    pub compute_pipeline: Option<ComputePipeline>,
    pub config: Option<SurfaceConfiguration>,
    pub surface: Option<Surface<'a>>,
    pub current_best: Drawing,
    pub device: Option<Device>,
    pub drawing_output_buffer: Option<Buffer>,
    pub drawing_texture_wrapper: Option<TextureWrapper>,
    pub drawing_texture: Option<Texture>,
    pub error_data: Vec<Color>,
    pub error_output_buffer: Option<Buffer>,
    pub error_source_buffer: Option<Buffer>,
    pub h: usize,
    pub initialized: bool,
    pub queue: Option<Queue>,
    pub raster_mode: Rasterizer,
    pub ref_image_bytes: Vec<u8>,
    pub ref_image_data: Vec<Color>,
    pub render_pipeline: Option<RenderPipeline>,
    pub stats: Stats,
    pub texture_extent: Option<Extent3d>,
    pub w: usize,
    pub window: Option<Arc<Window>>,
    pub working_data: Vec<Color>,
}

impl<'a> Engine<'a> {
    pub fn init(&mut self, filepath: &str, max_w: usize, max_h: usize) {
        let mut img = ImageReader::open(filepath)
            .expect("Failed to load image")
            .decode()
            .expect("Failed to decode image.");

        self.w = img.width() as usize;
        self.h = img.height() as usize;

        // downscale if source image is too large
        if self.w > max_w || self.h > max_h {
            img = img.resize(max_w as u32, max_h as u32, Lanczos3);
            self.w = img.width() as usize;
            self.h = img.height() as usize;
        }

        self.ref_image_bytes = img.into_rgba8().as_bytes().to_vec();
        self.ref_image_data = to_color_vec(&self.ref_image_bytes);
        self.post_init();
    }

    pub fn set_best(&mut self, drawing: Drawing) {
        assert!(self.initialized);
        self.current_best = drawing;
        self.current_best.fitness = 0.0; // do not trust
        if self.window.is_some() {
            self.redraw();
        }
    }

    fn post_init(&mut self) {
        let size: usize = (self.w * self.h) as usize;
        assert!(size > 0);
        self.working_data = vec![BLACK; size];
        self.error_data = vec![BLACK; size];
        assert!(self.ref_image_data.len() == size);
        assert!(self.working_data.len() == size);
        assert!(self.error_data.len() == size);
        self.initialized = true;
    }

    pub async fn tick(&mut self, max_time_ms: usize) {
        self.stats.ticks = 0;
        let mut elapsed: usize = 0;
        while elapsed < max_time_ms {
            self.stats.ticks += 1;
            let t0 = Instant::now();

            let mut clone = self.current_best.clone();
            clone.mutate();
            self.stats.generated += 1;
            self.calculate_fitness(&mut clone, false).await;
            if clone.fitness > self.current_best.fitness {
                // calculate again this time including error data (for display purposes)
                self.calculate_fitness(&mut clone, true).await; // TODO: benchmark if this is actually worth it
                self.current_best = clone;
                self.stats.improvements += 1;

                // FIXME: cleanup

                self.window.as_mut().expect("no window?").request_redraw();

                // if self.raster_mode != Rasterizer::GPU {
                //     self.current_best.draw(
                //         &mut self.working_data,
                //         self.w,
                //         self.h,
                //         self.raster_mode,
                //     );
                // }

                // if self.window.is_some() {
                // let pixels = to_u8_vec(&self.working_data); // FIXME if GPU mode don't convert to Colors and Back to u8s

                // let image =
                //     ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
                // self.window
                //     .as_mut()
                //     .expect("no window?")
                //     .set_image("image-001", image)
                //     .expect("Failed to set image.");
                // }
            }
            elapsed += t0.elapsed().as_millis() as usize;
        }

        self.stats.cycle_time = elapsed; // can't get f32 ms directly
    }

    pub async fn calculate_fitness(&mut self, drawing: &mut Drawing, draw_error: bool) {
        if self.raster_mode == Rasterizer::GPU {
            let (_, _, pixels, error_heatmap) = self.draw_and_evaluate(drawing).await;
            self.working_data = to_color_vec(&pixels);
            self.error_data = to_color_vec(&error_heatmap);
        } else {
            drawing.draw(&mut self.working_data, self.w, self.h, self.raster_mode);

            let num_pixels = self.w * self.h;
            assert!(num_pixels == self.working_data.len());
            assert!(self.ref_image_data.len() == self.working_data.len());

            let mut error = 0.0;
            for i in 0..num_pixels {
                // can't subtract u8 from u8 -> potential underflow
                let re = self.working_data[i].r as i32 - self.ref_image_data[i].r as i32;
                let ge = self.working_data[i].g as i32 - self.ref_image_data[i].g as i32;
                let be = self.working_data[i].b as i32 - self.ref_image_data[i].b as i32;

                let sqrt = f32::sqrt(((re * re) + (ge * ge) + (be * be)) as f32);
                error += sqrt;

                if draw_error {
                    // scale it to 0 - 255, full red = max error
                    let err_color = f32::floor(255.0 * (1.0 - sqrt / MAX_ERROR_PER_PIXEL)) as u8;
                    self.error_data[i].r = 255;
                    self.error_data[i].g = err_color;
                    self.error_data[i].b = err_color;
                    self.error_data[i].a = 255;
                }
            }

            if draw_error {
                // TODO
            }

            let max_total_error = MAX_ERROR_PER_PIXEL * self.w as f32 * self.h as f32;
            drawing.fitness = 100.0 * (1.0 - error / max_total_error);
            let penalty = drawing.fitness * PER_POINT_MULTIPLIER * drawing.num_points() as f32;
            drawing.fitness -= penalty;
        }
    }

    // testing our rasterization logic
    // should be able to perfectly fill a rectangle with 2 triangles
    pub async fn test(&mut self) {
        let p1 = Point { x: 0.0, y: 0.0 };
        let p2 = Point { x: 0.0, y: 1.0 };
        let p3 = Point { x: 1.0, y: 1.0 };
        let p4 = Point { x: 1.0, y: 0.0 };
        let c = crate::models::color::Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let poly1 = Polygon {
            points: [p1, p2, p3].to_vec(),
            color: c,
        };
        let poly2 = Polygon {
            points: [p1, p4, p3].to_vec(),
            color: c,
        };
        let mut d = Drawing {
            polygons: [poly1, poly2].to_vec(),
            is_dirty: false,
            fitness: 0.0,
        };

        // confirm we are starting with all 0s
        // let all_black = self
        //     .working_data
        //     .iter()
        //     .all(|c| c.r == 0 && c.g == 0 && c.b == 0 && c.a == 0);
        // assert!(all_black);

        self.calculate_fitness(&mut d, false).await;
        // self.current_best = d;

        // confirm every pixel is red after drawing 2 red triangles that cover 100% of the area
        let all_red = self
            .working_data
            .iter()
            .all(|c| c.r == 255 && c.g == 0 && c.b == 0 && c.a == 255);
        dbg!(all_red);
        assert!(all_red);
        // dbg!(&self.error_data);

        // self.window.as_mut().expect("no window?").request_redraw();

        // let pixels = to_u8_vec(&self.working_data);
        // let image = ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
        // self.window
        //     .as_mut()
        //     .expect("no window?")
        //     .set_image("image-001", image)
        //     .expect("Failed to set image.");
    }

    pub fn redraw(&mut self) {
        // todo!()
        self.current_best
            .draw(&mut self.working_data, self.w, self.h, self.raster_mode);
        // FIXME : window.request_redraw?
        // let pixels = to_u8_vec(&self.working_data);
        // let image = ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
        // self.window
        //     .as_mut()
        //     .expect("no window?")
        //     .set_image("image-001", image)
        //     .expect("Failed to set image.");
    }

    pub async fn draw(&self, drawing: &Drawing) -> SubmissionIndex {
        let vertices: Vec<Vertex> = drawing.to_vertices();

        // create buffer, write buffer (bytemuck?)
        let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
            self.device.as_ref().expect("no device?"),
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );

        let command_buffer: wgpu::CommandBuffer = {
            let mut encoder = self
                .device
                .as_ref()
                .expect("no device?")
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let view = &self
                .drawing_texture
                .as_ref()
                .expect("no drawing_texture?")
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

            rpass.set_pipeline(&self.render_pipeline.as_ref().expect("no render_pipeline?"));
            // rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..vertices.len() as u32, 0..vertices.len() as u32);

            // encoder methods like begin_render_pass and copy_texture_to_buffer take a &'pass mut self
            // drop rpass before copy_texture_to_buffer to avoid: cannot borrow `encoder` as mutable more than once at a time
            drop(rpass);

            // Copy the data from the texture to the buffer
            encoder.copy_texture_to_buffer(
                self.drawing_texture
                    .as_ref()
                    .expect("no drawing_texture?")
                    .as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: &self
                        .drawing_output_buffer
                        .as_ref()
                        .expect("no drawing_output_buffer?"),
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(self.buffer_dimensions.padded_bytes_per_row as u32),
                        rows_per_image: None,
                    },
                },
                self.texture_extent.expect("no texture_extent?"),
            );

            encoder.finish()
        };

        self.queue
            .as_ref()
            .expect("no queue?")
            .submit(Some(command_buffer))
    }

    pub async fn calculate_error(&self, width: u32, height: u32) -> SubmissionIndex {
        let mut encoder = self
            .device
            .as_ref()
            .expect("no device?")
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("calculate_error_command_encoder"),
            });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(
            &self
                .compute_pipeline
                .as_ref()
                .expect("no compute_pipeline?"),
        );
        cpass.set_bind_group(
            0,
            &self
                .compute_bind_group
                .as_ref()
                .expect("no compute_bind_group?"),
            &[],
        );
        cpass.dispatch_workgroups(width / 8, height / 8, 1); // compute shader workgroup_size is (8, 8, 1)
        drop(cpass);

        encoder.copy_buffer_to_buffer(
            &self
                .error_source_buffer
                .as_ref()
                .expect("no error_source_buffer?"),
            0,
            &self
                .error_output_buffer
                .as_ref()
                .expect("no error_output_buffer?"),
            0,
            (width * height * 4) as u64,
        );

        self.queue
            .as_ref()
            .expect("no queue?")
            .submit(Some(encoder.finish()))
    }

    pub async fn draw_and_evaluate(&self, drawing: &mut Drawing) -> (f32, f32, Vec<u8>, Vec<u8>) {
        // step 1 - render pipeline --> draw our triangles to a texture
        let draw_si = self.draw(&drawing).await;
        let dob = self
            .drawing_output_buffer
            .as_ref()
            .expect("no drawing_output_buffer?");
        let best_drawing_bytes = self.get_bytes(&dob, draw_si).await;

        // Step 2 - compute pipeline --> diff drawing texture vs source texture
        let ce_si = self.calculate_error(self.w as u32, self.h as u32).await;

        // Step 3 - calculate error and error heatmap (sum output of compute pipeline)
        // TODO: parallel reduction on GPU, something like https://eximia.co/implementing-parallel-reduction-in-cuda/
        let eob = self
            .error_output_buffer
            .as_ref()
            .expect("no drawing_output_buffer?");
        let error_buffer = self.get_bytes(&eob, ce_si).await;

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
            .as_ref()
            .expect("no device?")
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

impl<'a> ApplicationHandler for Engine<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        assert!(self.w > 0);
        assert!(self.h > 0);

        let window_attributes = Window::default_attributes().with_title("polygon renderer");
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.window = Some(window.clone());

        let (instance, surface, adapter, device, queue) =
            futures_lite::future::block_on(async move {
                let instance = wgpu::Instance::default();
                let surface = instance.create_surface(window).unwrap();

                let adapter_options = &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
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
                            required_limits: wgpu::Limits::downlevel_defaults()
                                .using_resolution(adapter.limits()),
                            memory_hints: wgpu::MemoryHints::Performance,
                        },
                        None,
                    )
                    .await
                    .expect("Failed to create device");

                (instance, surface, adapter, device, queue)
            });

        let config = surface
            .get_default_config(&adapter, self.w as u32, self.h as u32)
            .unwrap();
        surface.configure(&device, &config);

        //TODO see if we need to keep references to the rest
        self.config = Some(config);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);

        // It is a WebGPU requirement that ImageCopyBuffer.layout.bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0
        // So we calculate padded_bytes_per_row by rounding unpadded_bytes_per_row
        // up to the next multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
        // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
        let buffer_dimensions = BufferDimensions::new(self.w, self.h); // FIXME do this inside init with new calculated w, h
        assert!(self.w * self.h > 0);
        self.buffer_dimensions = buffer_dimensions;

        // The output buffer lets us retrieve the data as an array
        let drawing_output_buffer =
            self.device
                .as_mut()
                .expect("no device?")
                .create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (self.buffer_dimensions.padded_bytes_per_row
                        * self.buffer_dimensions.height) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
        self.drawing_output_buffer = Some(drawing_output_buffer);

        // will copy this to error_output_buffer after the compute pass
        let error_source_buffer =
            self.device
                .as_mut()
                .expect("no device?")
                .create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (self.buffer_dimensions.width * self.buffer_dimensions.height * 4) as u64,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
        self.error_source_buffer = Some(error_source_buffer);

        // final error output per pixel
        let error_output_buffer =
            self.device
                .as_mut()
                .expect("no device?")
                .create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (self.buffer_dimensions.width * self.buffer_dimensions.height * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
        self.error_output_buffer = Some(error_output_buffer);

        let texture_extent = wgpu::Extent3d {
            width: self.buffer_dimensions.width as u32,
            height: self.buffer_dimensions.height as u32,
            depth_or_array_layers: 1,
        };

        let texture_format = wgpu::TextureFormat::Rgba8Unorm;

        // The render pipeline renders data into this texture
        let drawing_texture =
            self.device
                .as_mut()
                .expect("no device?")
                .create_texture(&wgpu::TextureDescriptor {
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
        self.drawing_texture = Some(drawing_texture);

        let compute_bind_group_layout = self
            .device
            .as_mut()
            .expect("no device?")
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                (self.buffer_dimensions.width * self.buffer_dimensions.height * 4)
                                    as u64,
                            ),
                        },
                        count: None,
                    },
                ],
                label: Some("compute_bind_group_layout"),
            });

        let dimensions = (self.w as u32, self.h as u32);
        assert!(self.ref_image_bytes.len() == self.w * self.h * 4);
        let source_texture_wrapper = TextureWrapper::from_bytes(
            &self.device.as_mut().expect("no device?"),
            &self.queue.as_mut().expect("no queue?"),
            &self.ref_image_bytes.as_slice(),
            dimensions,
            &"source",
        )
        .expect("Failed to create source_texture");

        let compute_bind_group = self.device.as_mut().expect("no device?").create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        // source image texture WxH Rgba8Unorm
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_texture_wrapper.view),
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
                            self.error_source_buffer
                                .as_ref()
                                .expect("no error_source_buffer?")
                                .as_entire_buffer_binding(),
                        ),
                    },
                ],
                label: Some("compute_bind_group"),
            },
        );

        let render_pipeline_layout = self
            .device
            .as_mut()
            .expect("no device?")
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        // Load the shaders from disk
        let shader = self
            .device
            .as_mut()
            .expect("no device?")
            .create_shader_module(wgpu::ShaderModuleDescriptor {
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

        let render_pipeline = self
            .device
            .as_mut()
            .expect("no device?")
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        let compute_pipeline_layout = self
            .device
            .as_mut()
            .expect("no device?")
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_module = self
            .device
            .as_mut()
            .expect("no device?")
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "error.compute2.wgsl"
                ))),
            });

        let compute_pipeline = self
            .device
            .as_mut()
            .expect("no device?")
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&compute_pipeline_layout),
                module: &compute_module,
                entry_point: "main",
                compilation_options: Default::default(), // TODO: investigate if there's a better option
                cache: None,
            });

        // TODO only set things we need
        self.texture_extent = Some(texture_extent);
        self.render_pipeline = Some(render_pipeline);
        // self.drawing_texture = Some(drawing_texture);
        self.compute_pipeline = Some(compute_pipeline);
        self.compute_bind_group = Some(compute_bind_group);
        // self.drawing_output_buffer = Some(drawing_output_buffer);
        // self.error_output_buffer = Some(error_output_buffer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        // info!("{event:?}");

        match event {
            WindowEvent::CloseRequested => {
                // There is a bug on macos which panics when
                // the window closes. This
                // isn't a huge deal since the application
                // is already closing, but theoretically
                // would prevent cleanup (since it is a
                // panic) and is ugly from a DX/UX
                // perspective.
                //
                // ```
                // a delegate was not configured on the application
                // ```
                //
                // that can be worked around by taking the
                // window and dropping it here ourselves.
                // the fix has already been merged, but is
                // not in a winit release yet. https://github.com/rust-windowing/winit/pull/3684
                //
                // we use `.take()` to replace the options
                // with `None` in our `App`, then we own the
                // data and it will drop.
                //
                // `surface` keeps a reference to the
                // window, so we need to drop that first
                let _ = self.surface.take();
                // then we can drop the window
                let _ = self.window.take();

                event_loop.exit();
            }
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                let Self {
                    config: Some(config),
                    surface: Some(surface),
                    device: Some(device),
                    window: Some(window),
                    ..
                } = self
                else {
                    return;
                };

                // Reconfigure the surface with the new size
                config.width = width.max(1);
                config.height = height.max(1);

                surface.configure(device, config);
                // On macos the window needs to be redrawn
                // manually after resizing
                window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // allow a single_match here so that people
                // who use this example can easily match on
                // new keys
                #[allow(clippy::single_match)]
                match key.as_ref() {
                    // WARNING: Consider using
                    // `key_without_modifiers()` if
                    // available on your platform.
                    Key::Named(NamedKey::Escape) => {
                        // TODO: This is the same handling
                        // as `WindowEvent::CloseRequested`,
                        // which we'll be removing in a
                        // future version
                        let _ = self.surface.take();
                        // then we can drop the window
                        let _ = self.window.take();

                        event_loop.exit();
                    }
                    _ => (),
                }
            }
            WindowEvent::RedrawRequested => {
                // FIXME: what was this about?
                // let Self {
                //     surface: Some(surface),
                //     device: Some(device),
                //     queue: Some(queue),
                //     render_pipeline: Some(render_pipeline),
                //     ..
                // } = self
                // else {
                //     return;
                // };

                // FIXME: how to always just draw whatever is in self.working_data
                let frame = self
                    .surface
                    .as_mut()
                    .expect("no surface?")
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                frame.present();
            }
            _ => (),
        }
    }

    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        let _ = (event_loop, cause);
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: ()) {
        let _ = (event_loop, event);
        // info!("{event:?}");
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let _ = (event_loop, device_id, event);
        // info!("{event:?}");
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        futures_lite::future::block_on(async move {
            // self.test().await;
            self.tick(TARGET_FRAMETIME).await;
            // self.window.as_mut().expect("no window?").request_redraw(); // TODO: not sure if should call this? maybe only call on new best
        });
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = event_loop;
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = event_loop;
    }

    fn memory_warning(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = event_loop;
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
