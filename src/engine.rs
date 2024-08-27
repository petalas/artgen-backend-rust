use std::time::Instant;

use image::{imageops::FilterType::Lanczos3, EncodableLayout, ImageReader};
use show_image::{create_window, ImageInfo, ImageView, WindowProxy};

use crate::{
    models::{drawing::Drawing, point::Point, polygon::Polygon},
    settings::{MAX_ERROR_PER_PIXEL, PER_POINT_MULTIPLIER},
};

#[derive(Debug)]
pub struct Stats {
    pub generated: usize,
    pub improvements: usize,
    pub cycle_time: usize,
    pub ticks: usize,
}

pub struct Engine {
    pub ref_image_data: Vec<u8>,
    pub working_data: Vec<u8>,
    pub error_data: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub current_best: Drawing,
    pub stats: Stats,
    pub window: WindowProxy,
    pub raster_mode: usize,
}

impl Engine {
    pub fn new() -> Engine {
        let ref_image_data: Vec<u8> = vec![];
        let error_data: Vec<u8> = vec![];
        let working_data: Vec<u8> = vec![];
        let window = create_window("image", Default::default()).expect("Failed to create window.");
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
            window,
            raster_mode: 2,
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

        self.ref_image_data = img.into_rgba8().as_bytes().to_vec();
        println!("Loaded {}x{} image.", self.w, self.h);
        assert!(self.ref_image_data.len() == self.w * self.h * 4);
        self.post_init();
    }

    fn post_init(&mut self) {
        let size: usize = (self.w * self.h * 4) as usize;
        self.working_data = vec![0u8; size];
        self.error_data = vec![0u8; size];
        // self.calculate_fitness(self.current_best.clone(), true);
        println!("Engine ready.");
    }

    pub fn tick(&mut self, max_time_ms: usize) {
        self.stats.ticks = 0;
        let mut elapsed: usize = 0;
        while elapsed < max_time_ms {
            self.stats.ticks += 1;
            let t0 = Instant::now();

            let mut clone = self.current_best.clone();
            clone.mutate();
            self.stats.generated += 1;
            clone = self.calculate_fitness(clone, false);
            if clone.fitness > self.current_best.fitness {
                // calculate again this time including error data (for display purposes)
                clone = self.calculate_fitness(clone, true);
                self.current_best = clone;
                self.stats.improvements += 1;
                self.current_best
                    .draw(&mut self.working_data, self.w, self.h, self.raster_mode);

                let image = ImageView::new(
                    ImageInfo::rgba8(self.w as u32, self.h as u32),
                    &self.working_data,
                );
                self.window
                    .set_image("image-001", image)
                    .expect("Failed to set image.");
            }
            elapsed += t0.elapsed().as_millis() as usize;
        }

        self.stats.cycle_time = elapsed; // can't get f64 ms directly
    }

    fn calculate_fitness(&mut self, mut drawing: Drawing, draw_error: bool) -> Drawing {
        drawing.draw(&mut self.working_data, self.w, self.h, self.raster_mode);

        let num_pixels = self.w * self.h;

        let mut error = 0.0;
        for i in 0..num_pixels {
            let r = (i * 4) as usize;
            let g = r + 1;
            let b = g + 1;
            let a = b + 1;

            // can't subtract u8 from u8 -> potential underflow
            let re = self.working_data[r] as isize - self.ref_image_data[r] as isize;
            let ge = self.working_data[g] as isize - self.ref_image_data[g] as isize;
            let be = self.working_data[b] as isize - self.ref_image_data[b] as isize;

            let sqrt = f64::sqrt(((re * re) + (ge * ge) + (be * be)) as f64);
            error += sqrt;

            if draw_error {
                // scale it to 0 - 255, full red = max error
                let err_color = f64::floor(255.0 * (1.0 - sqrt / MAX_ERROR_PER_PIXEL)) as u8;
                self.error_data[r] = 255;
                self.error_data[g] = err_color;
                self.error_data[b] = err_color;
                self.error_data[a] = 255;
            }
        }

        if draw_error {
            // TODO
        }

        let max_total_error = MAX_ERROR_PER_PIXEL * self.w as f64 * self.h as f64;
        drawing.fitness = 100.0 * (1.0 - error / max_total_error);
        let penalty = drawing.fitness * PER_POINT_MULTIPLIER * drawing.num_points() as f64;
        drawing.fitness -= penalty;

        drawing
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
            .chunks_exact(4)
            .all(|c| c == [255, 0, 0, 255]);
        // assert!(all_red);

        let image = ImageView::new(
            ImageInfo::rgba8(self.w as u32, self.h as u32),
            &self.working_data,
        );
        self.window
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

        let image = ImageView::new(
            ImageInfo::rgba8(self.w as u32, self.h as u32),
            &self.working_data,
        );
        self.window
            .set_image("image-001", image)
            .expect("Failed to set image.");
    }

    pub fn redraw(&mut self) {
        self.current_best
            .draw(&mut self.working_data, self.w, self.h, self.raster_mode);
        let image = ImageView::new(
            ImageInfo::rgba8(self.w as u32, self.h as u32),
            &self.working_data,
        );
        self.window
            .set_image("image-001", image)
            .expect("Failed to set image.");
    }
}
