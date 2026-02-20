pub mod buffers;
pub mod pipeline;

use std::time::Instant;

use wgpu::*;

use crate::models::drawing::Drawing;
use crate::mutation_params::MutationParams;
use crate::settings::{GPU_CHAIN_COUNT, GPU_ITERATIONS_PER_BATCH, GPU_MIGRATION_INTERVAL};

use buffers::{
    default_gpu_params, gpu_params_from, drawing_to_gpu, gpu_to_drawing, ControlFlags, GpuDrawingState,
    GpuParams, GPU_DRAWING_STATE_SIZE,
};
use pipeline::GpuPipeline;

pub struct GpuEvolver {
    pipeline: GpuPipeline,
    iteration: u32,
    total_evaluations: u64,
    start_time: Instant,
    best_fitness_bits: u32, // track best fitness across batches for control flag reset
}

impl GpuEvolver {
    pub async fn new(
        reference_rgba: &[u8],
        image_width: u32,
        image_height: u32,
        initial_drawing: &Drawing,
    ) -> Self {
        let chain_count = GPU_CHAIN_COUNT;
        let params = default_gpu_params(image_width, image_height, GPU_MIGRATION_INTERVAL);

        // Create initial chain states — all start from the same drawing but with different RNG seeds
        let initial_states: Vec<GpuDrawingState> = (0..chain_count)
            .map(|i| {
                let seed = 0xDEAD_BEEF_u64.wrapping_add(i as u64 * 0x9E3779B97F4A7C15);
                drawing_to_gpu(initial_drawing, seed)
            })
            .collect();

        let pipeline = GpuPipeline::new(
            chain_count,
            image_width,
            image_height,
            reference_rgba,
            &initial_states,
            &params,
        )
        .await;

        let actual_chains = pipeline.chain_count;
        if actual_chains < chain_count {
            println!(
                "GPU chain count capped: {} → {} (adapter max_storage_buffer_binding_size limit)",
                chain_count, actual_chains,
            );
        }
        println!(
            "GPU evolver initialized: {} chains, {}x{} image, {:.1} MB GPU memory",
            actual_chains,
            image_width,
            image_height,
            estimate_gpu_memory(actual_chains, image_width, image_height) as f64 / (1024.0 * 1024.0),
        );

        Self {
            pipeline,
            iteration: 0,
            total_evaluations: 0,
            start_time: Instant::now(),
            best_fitness_bits: 0,
        }
    }

    /// Run a batch of N iterations on the GPU.
    /// Returns `Some(Drawing)` if a new global best was found, `None` otherwise.
    pub fn run_batch(&mut self, mutation_params: &MutationParams) -> Option<Drawing> {
        let p = &self.pipeline;
        let iterations = GPU_ITERATIONS_PER_BATCH;

        // Update iteration number in params
        let mut params = gpu_params_from(mutation_params, p.image_width, p.image_height, GPU_MIGRATION_INTERVAL);
        params.iteration_number = self.iteration;
        p.queue.write_buffer(&p.params_buf, 0, bytemuck::bytes_of(&params));

        // Reset control flags before batch — preserve best_fitness_bits so atomicMax
        // only triggers new_best_found when fitness actually improves over last known best
        let control_reset = ControlFlags {
            new_best_found: 0,
            best_chain_id: 0,
            best_fitness_bits: self.best_fitness_bits,
            _pad: 0,
        };
        p.queue.write_buffer(&p.control_flags_buf, 0, bytemuck::bytes_of(&control_reset));

        // Reset error accumulators
        let zeros = vec![0u8; p.chain_count as usize * 4];
        p.queue.write_buffer(&p.error_accumulators_buf, 0, &zeros);

        // Encode N iterations × 5 passes into one command buffer
        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_evolver_batch"),
        });

        let wg_x = (p.image_width + 7) / 8;
        let wg_y = (p.image_height + 7) / 8;

        for i in 0..iterations {
            // Pass 1: Mutate (K workgroups of size 1)
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("mutate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&p.mutate_pipeline);
                pass.set_bind_group(0, &p.mutate_bind_group, &[]);
                pass.dispatch_workgroups(p.chain_count, 1, 1);
            }

            // Pass 2: Rasterize (W/8 × H/8 × K workgroups)
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("rasterize"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&p.rasterize_pipeline);
                pass.set_bind_group(0, &p.rasterize_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, p.chain_count);
            }

            // Pass 3: Error + Reduce (W/8 × H/8 × K workgroups)
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("error_reduce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&p.error_reduce_pipeline);
                pass.set_bind_group(0, &p.error_reduce_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, p.chain_count);
            }

            // Pass 4: Select (K workgroups of size 1)
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("select"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&p.select_pipeline);
                pass.set_bind_group(0, &p.select_bind_group, &[]);
                pass.dispatch_workgroups(p.chain_count, 1, 1);
            }

            // Pass 5: Migrate (every MIGRATION_INTERVAL iterations)
            let global_iter = self.iteration + i;
            if global_iter > 0 && global_iter % GPU_MIGRATION_INTERVAL == 0 {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("migrate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&p.migrate_pipeline);
                pass.set_bind_group(0, &p.migrate_bind_group, &[]);
                pass.dispatch_workgroups(p.chain_count, 1, 1);
            }

            // Reset error accumulators between iterations (via buffer copy of zeros)
            // We write zeros at start and after each select reads them via atomicExchange,
            // so they're already reset. No extra work needed.
        }

        // Copy control flags to staging for readback
        encoder.copy_buffer_to_buffer(
            &p.control_flags_buf,
            0,
            &p.control_staging_buf,
            0,
            std::mem::size_of::<ControlFlags>() as u64,
        );

        // Submit
        p.queue.submit(std::iter::once(encoder.finish()));

        self.iteration += iterations;
        self.total_evaluations += iterations as u64 * p.chain_count as u64;

        // Poll for completion and read control flags
        let control = self.read_control_flags();

        if control.new_best_found != 0 {
            self.best_fitness_bits = control.best_fitness_bits;
            let drawing = self.readback_best(control.best_chain_id);
            Some(drawing)
        } else {
            None
        }
    }

    fn read_control_flags(&self) -> ControlFlags {
        let p = &self.pipeline;
        let slice = p.control_staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        p.device.poll(Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map control staging buffer");

        let data = slice.get_mapped_range();
        let flags: ControlFlags = *bytemuck::from_bytes(&data);
        drop(data);
        p.control_staging_buf.unmap();
        flags
    }

    fn readback_best(&self, chain_id: u32) -> Drawing {
        let p = &self.pipeline;

        // Copy the winning chain's state to readback staging
        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("readback"),
        });

        let src_offset = chain_id as u64 * GPU_DRAWING_STATE_SIZE as u64;
        encoder.copy_buffer_to_buffer(
            &p.chain_states_buf,
            src_offset,
            &p.readback_staging_buf,
            0,
            GPU_DRAWING_STATE_SIZE as u64,
        );

        p.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = p.readback_staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        p.device.poll(Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map readback staging buffer");

        let data = slice.get_mapped_range();
        let state: &GpuDrawingState = bytemuck::from_bytes(&data);
        let drawing = gpu_to_drawing(state);
        drop(data);
        p.readback_staging_buf.unmap();

        drawing
    }

    pub fn total_evaluations(&self) -> u64 {
        self.total_evaluations
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    pub fn evals_per_sec(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 {
            self.total_evaluations as f64 / secs
        } else {
            0.0
        }
    }
}

fn estimate_gpu_memory(chain_count: u32, w: u32, h: u32) -> usize {
    let k = chain_count as usize;
    let pixels = (w * h) as usize;
    let chain_states = k * GPU_DRAWING_STATE_SIZE;
    let working_states = k * GPU_DRAWING_STATE_SIZE;
    let reference = pixels * 4;
    let render_targets = k * pixels * 4;
    let error_accumulators = k * 4;
    let control = 16;
    let params = std::mem::size_of::<GpuParams>();
    let staging = GPU_DRAWING_STATE_SIZE + 16;
    chain_states + working_states + reference + render_targets + error_accumulators + control + params + staging
}
