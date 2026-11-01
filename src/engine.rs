use image::{imageops::FilterType::Lanczos3, EncodableLayout, ImageReader};

use crate::{
    models::drawing::Drawing,
    settings::{MAX_ERROR_PER_PIXEL, PER_POINT_PENALTY},
};

/// The CPU triangle fill `Drawing::draw` uses.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub enum Rasterizer {
    /// Scanline fill.
    Scanline,

    /// Half-space functions, faster than scanline.
    /// https://web.archive.org/web/20050907040950/http://sw-shader.sourceforge.net:80/rasterizer.html
    #[default]
    HalfSpace,
}

/// A reference image and the CPU scratch buffers that score drawings against it.
#[derive(Default)]
pub struct Engine {
    pub error_data: Vec<u8>,
    pub h: usize,
    pub raster_mode: Rasterizer,
    pub ref_image_data: Vec<u8>,
    pub w: usize,
    pub working_data: Vec<u8>,
}

impl Engine {
    /// Load the reference image, resized when a side falls outside the bounds.
    pub fn init(&mut self, filepath: &str, min_w: usize, min_h: usize, max_w: usize, max_h: usize) {
        let mut img = ImageReader::open(filepath)
            .expect("Failed to load image")
            .decode()
            .expect("Failed to decode image.");

        self.w = img.width() as usize;
        self.h = img.height() as usize;

        // Resize if the source image is too small or too large
        if self.w < min_w || self.h < min_h || self.w > max_w || self.h > max_h {
            let target_w = self.w.clamp(min_w, max_w) as u32;
            let target_h = self.h.clamp(min_h, max_h) as u32;
            img = img.resize(target_w, target_h, Lanczos3);
            self.w = img.width() as usize;
            self.h = img.height() as usize;
        }

        let size = self.w * self.h * 4;
        assert!(size > 0);

        self.ref_image_data = img.into_rgba8().as_bytes().to_vec();
        self.working_data = vec![0u8; size];
        self.error_data = vec![0u8; size];
        assert!(self.ref_image_data.len() == size);
    }

    pub fn calculate_fitness(&mut self, drawing: &mut Drawing, draw_error: bool) {
        drawing.draw(&mut self.working_data, self.w, self.h, self.raster_mode);

        let num_pixels = self.w * self.h;
        assert!(num_pixels == self.working_data.len() / 4);
        assert!(self.ref_image_data.len() == self.working_data.len());

        let mut error = 0.0;
        for i in 0..num_pixels {
            let r = i * 4;
            let g = r + 1;
            let b = g + 1;
            let a = b + 1; // don't need to involve alpha in error calc

            // can't subtract u8 from u8 -> potential underflow
            let re = self.working_data[r] as i32 - self.ref_image_data[r] as i32;
            let ge = self.working_data[g] as i32 - self.ref_image_data[g] as i32;
            let be = self.working_data[b] as i32 - self.ref_image_data[b] as i32;

            let pixel_error = (re.abs() + ge.abs() + be.abs()) as f32;
            error += pixel_error;

            if draw_error {
                // this is for the error heatmap
                // scale it to 0 - 255, full red = max error
                let err_color = f32::floor(255.0 * (1.0 - pixel_error / MAX_ERROR_PER_PIXEL)) as u8;
                self.error_data[r] = 255;
                self.error_data[g] = err_color;
                self.error_data[b] = err_color;
                self.error_data[a] = 255;
            }
        }

        let max_total_error = MAX_ERROR_PER_PIXEL * self.w as f32 * self.h as f32;
        drawing.fitness = 100.0 * (1.0 - error / max_total_error);
        let penalty = drawing.fitness * PER_POINT_PENALTY * drawing.num_points() as f32;
        drawing.fitness -= penalty;
    }
}
