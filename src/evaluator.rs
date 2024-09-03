use std::{
    sync::{Arc, RwLock},
    time::Instant,
};

use tokio::sync::broadcast::Receiver;

use crate::{
    engine::Rasterizer,
    models::drawing::Drawing,
    settings::{MAX_ERROR_PER_PIXEL, PER_POINT_MULTIPLIER},
};

#[derive(Default, Debug, Clone)]
pub struct EvaluatorPayload {
    pub evaluations: usize,
    pub mutations: usize,
    pub elapsed: usize,
    pub best: Drawing,
    // pub global_best: Option<Arc<RwLock<Drawing>>>, // does not need to be returned back from workers
    pub working_data: Vec<u8>,
}

pub struct Evaluator {
    pub w: usize,
    pub h: usize,
    pub raster_mode: Rasterizer,
    pub working_data: Vec<u8>,
    pub ref_image_data: Vec<u8>,
    pub error_data: Vec<u8>,
    pub current_best: Drawing,
    pub br: Receiver<Drawing>, // receive new best on broadcast channel from main
}

impl Evaluator {
    pub fn new(
        ref_image_data: Vec<u8>,
        w: usize,
        h: usize,
        current_best: Drawing,
        br: Receiver<Drawing>,
    ) -> Evaluator {
        let size = w * h * 4;
        assert!(size > 0);
        assert!(ref_image_data.len() == size);
        let working_data = vec![0u8; size];
        let error_data = vec![0u8; size];

        let mut clone = current_best.clone();

        let mut _self = Evaluator {
            ref_image_data,
            w,
            h,
            raster_mode: Default::default(),
            working_data,
            error_data,
            current_best,
            br,
        };

        // just in case a random one that was never evaluated was used to init
        _self.current_best.fitness = _self.evaluate(&mut clone, false);
        _self
    }

    pub fn reset(&mut self, current_best: Drawing) {
        self.working_data.fill(0u8);
        self.error_data.fill(0u8);
        self.current_best = current_best;
    }

    pub fn evaluate(&mut self, drawing: &mut Drawing, draw_error: bool) -> f32 {
        if self.raster_mode == Rasterizer::GPU {
            todo!();
        } else {
            drawing.draw(&mut self.working_data, self.w, self.h, self.raster_mode);

            let num_pixels = self.w * self.h;
            assert!(num_pixels == self.working_data.len() / 4);
            assert!(self.ref_image_data.len() == self.working_data.len());

            let mut error = 0.0;
            for i in 0..num_pixels {
                let r = (i * 4) as usize;
                let g = r + 1;
                let b = g + 1;
                let a = b + 1; // don't need to involve alpha in error calc

                // can't subtract u8 from u8 -> potential underflow
                let re = self.working_data[r] as i32 - self.ref_image_data[r] as i32;
                let ge = self.working_data[g] as i32 - self.ref_image_data[g] as i32;
                let be = self.working_data[b] as i32 - self.ref_image_data[b] as i32;

                let sqrt = f32::sqrt(((re * re) + (ge * ge) + (be * be)) as f32);
                error += sqrt;

                if draw_error {
                    // this is for the error heatmap
                    // scale it to 0 - 255, full red = max error
                    let err_color = f32::floor(255.0 * (1.0 - sqrt / MAX_ERROR_PER_PIXEL)) as u8;
                    self.error_data[r] = 255;
                    self.error_data[g] = err_color;
                    self.error_data[b] = err_color;
                    self.error_data[a] = 255;
                }
            }

            let max_total_error = MAX_ERROR_PER_PIXEL * self.w as f32 * self.h as f32;
            drawing.fitness = 100.0 * (1.0 - error / max_total_error);
            let penalty = drawing.fitness * PER_POINT_MULTIPLIER * drawing.num_points() as f32;
            drawing.fitness -= penalty;

            drawing.fitness
        }
    }

    pub fn produce_new_best(&mut self) -> EvaluatorPayload {
        let mut evaluations: usize = 0;
        let mut mutations: usize = 0;
        let mut elapsed: usize = 0;
        let mut working_copy = self.current_best.clone();
        let mut new_best_fitness = 0f32;
        let old_best_fitness = self.current_best.fitness;

        while new_best_fitness <= old_best_fitness {
            evaluations += 1;
            let t0 = Instant::now();

            // attempt to receive new global best on broadcast channel from main
            working_copy = self.current_best.clone();
            if let Ok(received) = self.br.try_recv() {
                if received.fitness > working_copy.fitness {
                    working_copy = received;
                }
            }

            // ensure we have an actual mutation (probabilistic)
            working_copy.is_dirty = false;
            while !working_copy.is_dirty {
                working_copy.mutate();
                mutations += 1; // count how many mutations we have generated
            }

            new_best_fitness = self.evaluate(&mut working_copy, false);
            elapsed += t0.elapsed().as_millis() as usize;
        }

        EvaluatorPayload {
            best: working_copy,
            // global_best: None,
            evaluations,
            mutations,
            elapsed,
            working_data: self.working_data.clone(),
        }
    }
}
