use std::time::Instant;

use image::{imageops::FilterType::Lanczos3, EncodableLayout, ImageReader};
use show_image::{create_window, ImageInfo, ImageView, WindowProxy};
use tokio::runtime::{Handle, Runtime};
use wgpu::{core::pipeline, Gles3MinorVersion, InstanceFlags};

use crate::{
    buffer_dimensions::BufferDimensions,
    gpu_pipeline::{self, GpuPipeline},
    models::{
        color::{Color, WHITE},
        drawing::Drawing,
        point::Point,
        polygon::Polygon,
    },
    settings::{MAX_ERROR_PER_PIXEL, MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, PER_POINT_MULTIPLIER},
    utils::{to_color_vec, to_u8_vec},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rasterizer {
    // Scanline fill, slower but works for any polygons
    Scanline,

    // Advanced Rasterization based on half space functions
    // It is faster than scanline but only implemented for triangles
    // https://web.archive.org/web/20050907040950/http://sw-shader.sourceforge.net:80/rasterizer.html
    HalfSpace,

    // will default to Half-Space unless the polygon has more than 3 points
    Optimal,

    GPU,
}

#[derive(Debug)]
pub struct Stats {
    pub generated: usize,
    pub improvements: usize,
    pub cycle_time: usize,
    pub ticks: usize,
}

pub struct Engine {
    pub ref_image_data: Vec<Color>,
    pub working_data: Vec<Color>,
    pub error_data: Vec<Color>,
    pub w: usize,
    pub h: usize,
    pub current_best: Drawing,
    pub stats: Stats,
    pub window: Option<WindowProxy>,
    pub raster_mode: Rasterizer,
    pub initialized: bool,
    pub gpu_pipeline: Option<GpuPipeline>,
    pub rt: Runtime,
    pub handle: Option<Handle>,
}

impl Engine {
    pub fn new() -> Engine {
        let ref_image_data: Vec<Color> = vec![];
        let error_data: Vec<Color> = vec![];
        let working_data: Vec<Color> = vec![];

        Engine {
            ref_image_data,
            error_data,
            working_data,
            w: 0,
            h: 0,
            current_best: Drawing::new_random(),
            stats: Stats {
                generated: 0,
                improvements: 0,
                cycle_time: 0,
                ticks: 0,
            },
            window: None,
            raster_mode: Rasterizer::Optimal,
            initialized: false,
            gpu_pipeline: None,
            rt: Runtime::new().expect("failed to create runtime"),
            handle: None,
        }
    }

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

        self.ref_image_data = to_color_vec(&img.into_rgba8().as_bytes().to_vec());
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
        self.working_data = vec![WHITE; size];
        self.error_data = vec![WHITE; size];
        assert!(self.ref_image_data.len() == size);
        assert!(self.working_data.len() == size);
        assert!(self.error_data.len() == size);
        self.initialized = true;
    }

    pub fn init_window(&mut self) {
        self.window =
            Some(create_window("image", Default::default()).expect("Failed to create window."));
    }

    pub async fn init_gpu(&mut self, handle: Handle) {
        self.handle = Some(handle);
        self.gpu_pipeline =
            Some(GpuPipeline::new(to_u8_vec(&self.ref_image_data.clone()), self.w, self.h).await);
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
                self.calculate_fitness(&mut clone, true).await;
                self.current_best = clone;
                self.stats.improvements += 1;

                if self.raster_mode != Rasterizer::GPU {
                    self.current_best.draw(
                        &mut self.working_data,
                        self.w,
                        self.h,
                        self.raster_mode,
                    );
                }

                if self.window.is_some() {
                    let pixels = to_u8_vec(&self.working_data); // FIXME if GPU mode don't convert to Colors and Back to u8s
                    let image =
                        ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
                    self.window
                        .as_mut()
                        .expect("no window?")
                        .set_image("image-001", image)
                        .expect("Failed to set image.");
                }
            }
            elapsed += t0.elapsed().as_millis() as usize;
        }

        self.stats.cycle_time = elapsed; // can't get f32 ms directly
    }

    pub async fn calculate_fitness(&mut self, drawing: &mut Drawing, draw_error: bool) {
        if self.raster_mode == Rasterizer::GPU {
            let gpu_pipeline = self
                .gpu_pipeline
                .as_mut()
                .expect("gpu pipeline not initialized.");

            let (_, _, pixels, _) = gpu_pipeline.draw_and_evaluate(drawing).await;
            self.working_data = to_color_vec(&pixels);
            // FIXME: Cannot start a runtime from within a runtime
            // let future = async {
            //     let gpu_pipeline = self
            //         .gpu_pipeline
            //         .as_mut()
            //         .expect("gpu pipeline not initialized.");
            //     gpu_pipeline.draw_and_evaluate(drawing).await;
            // };
            // self.handle.as_mut().expect("welp").block_on(future);
        } else {
            drawing.draw(&mut self.working_data, self.w, self.h, self.raster_mode);

            let num_pixels = self.w * self.h;
            assert!(num_pixels == self.working_data.len());
            assert!(self.ref_image_data.len() == self.working_data.len());

            let mut error = 0.0;
            for i in 0..num_pixels {
                // let r = (i * 4) as usize;
                // let g = r + 1;
                // let b = g + 1;
                // let a = b + 1;

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
    pub fn test(&mut self) {
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
        let d = Drawing {
            polygons: [poly1, poly2].to_vec(),
            is_dirty: false,
            fitness: 0.0,
        };
        d.draw(&mut self.working_data, self.w, self.h, self.raster_mode);
        self.current_best = d;

        let all_red = self
            .working_data
            .iter()
            .all(|c| c.r == 255 && c.g == 0 && c.b == 0 && c.a == 255);
        // assert!(all_red);

        let pixels = to_u8_vec(&self.working_data);
        let image = ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
        self.window
            .as_mut()
            .expect("no window?")
            .set_image("image-001", image)
            .expect("Failed to set image.");
    }

    pub fn test2(&mut self) {
        let p1 = Point { x: 0.35, y: 0.25 };
        let p2 = Point { x: 0.025, y: 0.75 };
        let p3 = Point { x: 0.7, y: 0.6 };
        // let p4 = Point { x: 1.0, y: 0.0 };
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
        let d = Drawing {
            polygons: [poly1].to_vec(),
            is_dirty: false,
            fitness: 0.0,
        };
        d.draw(&mut self.working_data, self.w, self.h, self.raster_mode);
        self.current_best = d;

        // let all_red = self
        //     .working_data
        //     .chunks_exact(4)
        //     .all(|c| c == [255, 0, 0, 255]);
        // assert!(all_red);

        let pixels = to_u8_vec(&self.working_data);
        let image = ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
        self.window
            .as_mut()
            .expect("no window?")
            .set_image("image-001", image)
            .expect("Failed to set image.");
    }

    pub fn test3(&mut self) {
        let d = Drawing::new_random();
        d.draw(&mut self.working_data, self.w, self.h, self.raster_mode);
        self.current_best = d;

        let pixels = to_u8_vec(&self.working_data);
        let image = ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
        self.window
            .as_mut()
            .expect("no window?")
            .set_image("image-001", image)
            .expect("Failed to set image.");
    }

    pub fn redraw(&mut self) {
        self.current_best
            .draw(&mut self.working_data, self.w, self.h, self.raster_mode);
        let pixels = to_u8_vec(&self.working_data);
        let image = ImageView::new(ImageInfo::rgba8(self.w as u32, self.h as u32), &pixels);
        self.window
            .as_mut()
            .expect("no window?")
            .set_image("image-001", image)
            .expect("Failed to set image.");
    }
}
