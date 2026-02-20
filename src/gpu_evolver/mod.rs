pub mod buffers;
pub mod pipeline;

use std::time::Instant;

use wgpu::*;

use crate::models::drawing::Drawing;
use crate::mutation_params::MutationParams;
use crate::settings::{GPU_MAX_CHAIN_COUNT, GPU_DEFAULT_CHAIN_COUNT, GPU_ITERATIONS_PER_BATCH, GPU_MIGRATION_INTERVAL};

use buffers::{
    default_gpu_params, gpu_params_from, drawing_to_gpu, gpu_to_drawing, ControlFlags, GpuDrawingState,
    GpuParams, GPU_DRAWING_STATE_SIZE,
};
use pipeline::GpuPipeline;

#[derive(Default)]
pub struct PassTimings {
    pub mutate_ns: f64,
    pub rasterize_error_ns: f64,
    pub select_ns: f64,
    pub migrate_ns: f64,
    pub sample_count: u64,
    pub migrate_sample_count: u64,
}

impl PassTimings {
    pub fn total_ns(&self) -> f64 {
        self.mutate_ns + self.rasterize_error_ns + self.select_ns + self.migrate_ns
    }

    pub fn print_averages(&self) {
        if self.sample_count == 0 {
            return;
        }
        let n = self.sample_count as f64;
        let mutate = self.mutate_ns / n / 1_000_000.0;
        let rasterize_error = self.rasterize_error_ns / n / 1_000_000.0;
        let select = self.select_ns / n / 1_000_000.0;
        let migrate = if self.migrate_sample_count > 0 {
            self.migrate_ns / self.migrate_sample_count as f64 / 1_000_000.0
        } else {
            0.0
        };
        let total = mutate + rasterize_error + select + migrate;

        if total <= 0.0 {
            return;
        }

        println!("GPU pass timings (avg over {} batches):", self.sample_count);
        println!("  mutate:          {:6.2}ms ({:5.1}%)", mutate, mutate / total * 100.0);
        println!("  rasterize+error: {:6.2}ms ({:5.1}%)", rasterize_error, rasterize_error / total * 100.0);
        println!("  select:          {:6.2}ms ({:5.1}%)", select, select / total * 100.0);
        if self.migrate_sample_count > 0 {
            println!("  migrate:         {:6.2}ms ({:5.1}%)", migrate, migrate / total * 100.0);
        }
        println!("  total:           {:6.2}ms", total);
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

pub struct GpuEvolver {
    pipeline: GpuPipeline,
    iteration: u32,
    total_evaluations: u64,
    start_time: Instant,
    best_fitness_bits: u32, // track best fitness across batches for control flag reset
    pass_timings: PassTimings,
    chain_fitness: Vec<f32>,
    active_chain_count: u32, // runtime chain count (≤ pipeline.chain_count)
}

impl GpuEvolver {
    pub async fn new(
        reference_rgba: &[u8],
        image_width: u32,
        image_height: u32,
        initial_drawing: &Drawing,
    ) -> Self {
        let max_chains = GPU_MAX_CHAIN_COUNT;
        let active_chains = GPU_DEFAULT_CHAIN_COUNT;
        let params = default_gpu_params(image_width, image_height, GPU_MIGRATION_INTERVAL, active_chains);

        // Create initial chain states for max capacity — all start from the same drawing but with different RNG seeds
        let initial_states: Vec<GpuDrawingState> = (0..max_chains)
            .map(|i| {
                let seed = 0xDEAD_BEEF_u64.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                drawing_to_gpu(initial_drawing, seed)
            })
            .collect();

        let pipeline = GpuPipeline::new(
            max_chains,
            image_width,
            image_height,
            reference_rgba,
            &initial_states,
            &params,
        )
        .await;

        let actual_max = pipeline.chain_count;
        if actual_max < max_chains {
            println!(
                "GPU max chain count capped: {} → {} (adapter max_storage_buffer_binding_size limit)",
                max_chains, actual_max,
            );
        }
        let active_chains = active_chains.min(actual_max);
        let wg_x = (image_width + 15) / 16;
        let wg_y = (image_height + 15) / 16;
        println!(
            "GPU evolver initialized: {} active chains (max {}), {}x{} image, {:.1} MB GPU memory, rasterize dispatch {}x{}x{}",
            active_chains,
            actual_max,
            image_width,
            image_height,
            estimate_gpu_memory(actual_max, image_width, image_height) as f64 / (1024.0 * 1024.0),
            wg_x,
            wg_y,
            active_chains,
        );

        Self {
            pipeline,
            iteration: 0,
            total_evaluations: 0,
            start_time: Instant::now(),
            best_fitness_bits: 0,
            pass_timings: PassTimings::default(),
            chain_fitness: vec![0.0; actual_max as usize],
            active_chain_count: active_chains,
        }
    }

    /// Run a batch of N iterations on the GPU.
    /// Returns `Some(Drawing)` if a new global best was found, `None` otherwise.
    /// When `collect_timestamps` is false, skips GPU timestamp resolve/readback for lower overhead.
    pub fn run_batch(&mut self, mutation_params: &MutationParams, collect_timestamps: bool) -> Option<Drawing> {
        let p = &self.pipeline;
        let iterations = GPU_ITERATIONS_PER_BATCH;
        let active = mutation_params.chain_count.min(p.chain_count);
        self.active_chain_count = active;

        // Update iteration number in params
        let mut params = gpu_params_from(mutation_params, p.image_width, p.image_height, GPU_MIGRATION_INTERVAL, active);
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

        // Note: error accumulators are NOT reset here — the select shader
        // resets them via atomicExchange after each iteration. They're
        // zero-initialized once at buffer creation and in reinit_chains().

        // Encode N iterations × 5 passes into one command buffer
        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_evolver_batch"),
        });

        let wg_x = (p.image_width + 15) / 16;
        let wg_y = (p.image_height + 15) / 16;

        let is_last_iter = |i: u32| i == iterations - 1;
        let mut migrate_ran = false;

        for i in 0..iterations {
            let ts = if collect_timestamps && is_last_iter(i) { Some(&p.timestamp_query_set) } else { None };

            // Pass 1: Mutate (K workgroups of size 1)
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("mutate"),
                    timestamp_writes: ts.map(|qs| ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(&p.mutate_pipeline);
                pass.set_bind_group(0, &p.mutate_bind_group, &[]);
                pass.dispatch_workgroups(active, 1, 1);
            }

            // Pass 2: Rasterize + Error (fused, W/16 × H/16 × K workgroups, tiled polygon prefetch)
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("rasterize_error"),
                    timestamp_writes: ts.map(|qs| ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }),
                });
                pass.set_pipeline(&p.rasterize_error_pipeline);
                pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, active);
            }

            // Pass 3: Select (K workgroups of size 1)
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("select"),
                    timestamp_writes: ts.map(|qs| ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    }),
                });
                pass.set_pipeline(&p.select_pipeline);
                pass.set_bind_group(0, &p.select_bind_group, &[]);
                pass.dispatch_workgroups(active, 1, 1);
            }

            // Pass 4: Migration — inter-island (rare, global ring) or intra-island (frequent, island ring)
            let global_iter = self.iteration + i;
            let inter_interval = mutation_params.inter_island_interval;
            if global_iter > 0 && inter_interval > 0 && global_iter % inter_interval == 0 {
                // Inter-island: global ring (takes priority when both intervals align)
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("migrate_inter"),
                    timestamp_writes: ts.map(|qs| ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(6),
                        end_of_pass_write_index: Some(7),
                    }),
                });
                pass.set_pipeline(&p.migrate_inter_pipeline);
                pass.set_bind_group(0, &p.migrate_bind_group, &[]);
                pass.dispatch_workgroups(active, 1, 1);
                if is_last_iter(i) {
                    migrate_ran = true;
                }
            } else if global_iter > 0 && global_iter % GPU_MIGRATION_INTERVAL == 0 {
                // Intra-island: island ring (frequent)
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("migrate_intra"),
                    timestamp_writes: ts.map(|qs| ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(6),
                        end_of_pass_write_index: Some(7),
                    }),
                });
                pass.set_pipeline(&p.migrate_intra_pipeline);
                pass.set_bind_group(0, &p.migrate_bind_group, &[]);
                pass.dispatch_workgroups(active, 1, 1);
                if is_last_iter(i) {
                    migrate_ran = true;
                }
            }

            // Reset error accumulators between iterations (via buffer copy of zeros)
            // We write zeros at start and after each select reads them via atomicExchange,
            // so they're already reset. No extra work needed.
        }

        // Resolve timestamp queries into resolve buffer, then copy to staging
        if collect_timestamps {
            encoder.resolve_query_set(&p.timestamp_query_set, 0..8, &p.timestamp_resolve_buf, 0);
            encoder.copy_buffer_to_buffer(
                &p.timestamp_resolve_buf,
                0,
                &p.timestamp_staging_buf,
                0,
                8 * 8,
            );
        }

        // Copy control flags to staging for readback
        encoder.copy_buffer_to_buffer(
            &p.control_flags_buf,
            0,
            &p.control_staging_buf,
            0,
            std::mem::size_of::<ControlFlags>() as u64,
        );

        // Copy fitness_packed to staging for readback (only active chains)
        let fitness_size = (active as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &p.fitness_packed_buf,
            0,
            &p.fitness_staging_buf,
            0,
            fitness_size,
        );

        // Submit
        p.queue.submit(std::iter::once(encoder.finish()));

        self.iteration += iterations;
        self.total_evaluations += iterations as u64 * active as u64;

        // Map staging buffers, poll once, read all results
        let control = self.read_batch_results(migrate_ran, collect_timestamps);

        if control.new_best_found != 0 {
            self.best_fitness_bits = control.best_fitness_bits;
            let drawing = self.readback_chain(control.best_chain_id);
            Some(drawing)
        } else {
            None
        }
    }

    /// Map staging buffers, poll once for GPU completion, then read all results.
    /// When `collect_timestamps` is false, skips the timestamp staging buffer entirely.
    fn read_batch_results(&mut self, migrate_ran: bool, collect_timestamps: bool) -> ControlFlags {
        let p = &self.pipeline;
        let active = self.active_chain_count as usize;

        // Map staging buffers before polling
        let control_slice = p.control_staging_buf.slice(..);
        let fitness_slice = p.fitness_staging_buf.slice(..((active * 4) as u64));

        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx3, rx3) = std::sync::mpsc::channel();

        control_slice.map_async(MapMode::Read, move |r| { tx1.send(r).unwrap(); });

        let ts_rx = if collect_timestamps {
            let timestamp_slice = p.timestamp_staging_buf.slice(..);
            let (tx2, rx2) = std::sync::mpsc::channel();
            timestamp_slice.map_async(MapMode::Read, move |r| { tx2.send(r).unwrap(); });
            Some(rx2)
        } else {
            None
        };

        fitness_slice.map_async(MapMode::Read, move |r| { tx3.send(r).unwrap(); });

        // Single poll waits for GPU completion — all maps resolve together
        p.device.poll(Maintain::Wait);

        rx1.recv().unwrap().expect("Failed to map control staging buffer");
        if let Some(rx2) = &ts_rx {
            rx2.recv().unwrap().expect("Failed to map timestamp staging buffer");
        }
        rx3.recv().unwrap().expect("Failed to map fitness staging buffer");

        // Read control flags
        let control_data = control_slice.get_mapped_range();
        let flags: ControlFlags = *bytemuck::from_bytes(&control_data);
        drop(control_data);

        // Read timestamps (only if collected)
        if ts_rx.is_some() {
            let ts_data = p.timestamp_staging_buf.slice(..).get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&ts_data);
            let period = p.timestamp_period as f64;
            let duration = |begin_idx: usize, end_idx: usize| -> f64 {
                timestamps[end_idx].wrapping_sub(timestamps[begin_idx]) as f64 * period
            };
            self.pass_timings.mutate_ns += duration(0, 1);
            self.pass_timings.rasterize_error_ns += duration(2, 3);
            self.pass_timings.select_ns += duration(4, 5);
            self.pass_timings.sample_count += 1;
            if migrate_ran {
                self.pass_timings.migrate_ns += duration(6, 7);
                self.pass_timings.migrate_sample_count += 1;
            }
            drop(ts_data);
            p.timestamp_staging_buf.unmap();
        }

        // Read chain fitness
        let fitness_data = fitness_slice.get_mapped_range();
        let packed: &[u32] = bytemuck::cast_slice(&fitness_data);
        for (i, &bits) in packed.iter().enumerate() {
            self.chain_fitness[i] = f32::from_bits(bits);
        }
        drop(fitness_data);

        // Unmap
        p.control_staging_buf.unmap();
        p.fitness_staging_buf.unmap();

        flags
    }

    pub fn readback_chain(&self, chain_id: u32) -> Drawing {
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


    /// Batch-readback multiple chains in a single GPU submission.
    /// Used for island thumbnail display (~every 2s), avoids N separate round-trips.
    pub fn readback_chains(&self, chain_ids: &[u32]) -> Vec<Drawing> {
        if chain_ids.is_empty() {
            return Vec::new();
        }

        let p = &self.pipeline;
        let count = chain_ids.len();
        let state_size = GPU_DRAWING_STATE_SIZE as u64;

        // Temporary staging buffer sized for all requested chains
        let staging = p.device.create_buffer(&BufferDescriptor {
            label: Some("multi_readback_staging"),
            size: count as u64 * state_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("multi_readback"),
        });

        for (i, &chain_id) in chain_ids.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                &p.chain_states_buf,
                chain_id as u64 * state_size,
                &staging,
                i as u64 * state_size,
                state_size,
            );
        }

        p.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| { tx.send(r).unwrap(); });
        p.device.poll(Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map multi-readback staging buffer");

        let data = slice.get_mapped_range();
        let drawings: Vec<Drawing> = (0..count)
            .map(|i| {
                let offset = i * GPU_DRAWING_STATE_SIZE;
                let state: &GpuDrawingState = bytemuck::from_bytes(&data[offset..offset + GPU_DRAWING_STATE_SIZE]);
                gpu_to_drawing(state)
            })
            .collect();
        drop(data);
        staging.unmap();

        drawings
    }

    /// Reinitialize all chains from a given drawing (for benchmarks).
    /// Resets iteration counters, timings, and fitness tracking.
    pub fn reinit_chains(&mut self, drawing: &Drawing) {
        let chain_count = self.pipeline.chain_count;
        let states: Vec<GpuDrawingState> = (0..chain_count)
            .map(|i| {
                let seed = 0xDEAD_BEEF_u64
                    .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .wrapping_add((self.iteration as u64).wrapping_mul(0x517CC1B727220A95));
                drawing_to_gpu(drawing, seed)
            })
            .collect();
        let bytes: Vec<u8> = states
            .iter()
            .flat_map(|s| bytemuck::bytes_of(s).to_vec())
            .collect();
        self.pipeline
            .queue
            .write_buffer(&self.pipeline.chain_states_buf, 0, &bytes);

        // Reset error accumulators so the first batch after reinit starts clean
        let zeros = vec![0u8; chain_count as usize * 4];
        self.pipeline
            .queue
            .write_buffer(&self.pipeline.error_accumulators_buf, 0, &zeros);

        self.iteration = 0;
        self.total_evaluations = 0;
        self.best_fitness_bits = 0;
        self.start_time = Instant::now();
        self.pass_timings = PassTimings::default();
        self.chain_fitness.fill(0.0);
    }

    pub fn chain_fitness(&self) -> &[f32] {
        &self.chain_fitness[..self.active_chain_count as usize]
    }

    pub fn chain_count(&self) -> u32 {
        self.active_chain_count
    }

    pub fn max_chain_count(&self) -> u32 {
        self.pipeline.chain_count
    }

    pub fn estimated_memory_bytes(&self) -> u64 {
        let p = &self.pipeline;
        estimate_gpu_memory(p.chain_count, p.image_width, p.image_height) as u64
    }

    pub fn pass_timings(&self) -> &PassTimings {
        &self.pass_timings
    }

    pub fn reset_pass_timings(&mut self) {
        self.pass_timings.reset();
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
    let error_accumulators = k * 4;
    let control = 16;
    let params = std::mem::size_of::<GpuParams>();
    let staging = GPU_DRAWING_STATE_SIZE + 16;
    chain_states + working_states + reference + error_accumulators + control + params + staging
}
