pub mod buffers;
pub mod pipeline;

use std::time::Instant;

use bytemuck::Zeroable;
use wgpu::*;

use crate::models::drawing::Drawing;
use crate::mutation_params::MutationParams;
use crate::settings::{GPU_MAX_CHAIN_COUNT, GPU_DEFAULT_CHAIN_COUNT, GPU_ITERATIONS_PER_BATCH, GPU_MIGRATION_INTERVAL};

use buffers::{
    gpu_params_from, drawing_to_gpu, gpu_to_drawing, ControlFlags, GpuDrawingState,
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

/// Which migration variant to run this iteration (if any).
enum MigrationType {
    Intra,
    Inter,
}

impl MigrationType {
    fn label(&self) -> &'static str {
        match self {
            Self::Intra => "migrate_intra",
            Self::Inter => "migrate_inter",
        }
    }

    fn pipeline<'a>(&self, p: &'a GpuPipeline) -> &'a ComputePipeline {
        match self {
            Self::Intra => &p.migrate_intra_pipeline,
            Self::Inter => &p.migrate_inter_pipeline,
        }
    }
}

/// Tracks state for a previously submitted batch whose staging buffers
/// have map_async issued but haven't been polled/read yet.
struct PendingBatch {
    staging_idx: usize,
    migrate_ran: bool,
    collect_timestamps: bool,
    active_chain_count: u32,
}

/// Receivers for the map_async callbacks on staging buffers.
struct PendingMapReceivers {
    control_rx: std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>,
    timestamp_rx: Option<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>,
    fitness_rx: std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>,
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
    staging_idx: usize,      // alternates 0/1 for double-buffered staging
    pending_batch: Option<PendingBatch>,
    pending_map_receivers: Option<PendingMapReceivers>,
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

        // Initialize offspring RNG states in working_states buffer
        init_offspring_rng(&pipeline, 0);

        let rwg = pipeline.rasterize_wg;
        let wg_x = (image_width + rwg[0] - 1) / rwg[0];
        let wg_y = (image_height + rwg[1] - 1) / rwg[1];
        println!(
            "GPU evolver initialized: {} active chains (max {}), offspring capacity {}, {}x{} image, {:.1} MB GPU memory, rasterize dispatch {}x{}x{} (wg {}x{})",
            active_chains,
            actual_max,
            pipeline.offspring_capacity,
            image_width,
            image_height,
            estimate_gpu_memory(actual_max, pipeline.offspring_capacity, image_width, image_height) as f64 / (1024.0 * 1024.0),
            wg_x,
            wg_y,
            active_chains,
            rwg[0],
            rwg[1],
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
            staging_idx: 0,
            pending_batch: None,
            pending_map_receivers: None,
        }
    }

    /// Run a batch of N iterations on the GPU using double-buffered staging.
    ///
    /// Returns the PREVIOUS batch's result (one batch behind). The first call
    /// always returns `None`. This overlap lets the GPU work on batch N+1
    /// while the CPU reads back batch N's results.
    ///
    /// Returns `Some(Drawing)` if the previous batch found a new global best.
    pub fn run_batch(&mut self, mutation_params: &MutationParams, collect_timestamps: bool) -> Option<Drawing> {
        // 1. If there's a pending batch from the last call, finish reading its results
        let prev_result = if let Some(pending) = self.pending_batch.take() {
            Some(self.finish_pending_readback(&pending))
        } else {
            None
        };

        // Check if rasterize workgroup size needs to change (between batches)
        self.pipeline.set_rasterize_wg(mutation_params.rasterize_wg);

        let p = &self.pipeline;
        let iterations = GPU_ITERATIONS_PER_BATCH;
        let active = mutation_params.chain_count.min(p.chain_count);
        self.active_chain_count = active;

        // 2. Pick which staging set to use for THIS batch's copies
        let write_idx = self.staging_idx;
        self.staging_idx = 1 - self.staging_idx;

        // 3. Build params for push constants
        let mut params = gpu_params_from(mutation_params, p.image_width, p.image_height, GPU_MIGRATION_INTERVAL, active);
        params.iteration_number = self.iteration;
        let params_bytes: &[u8] = bytemuck::bytes_of(&params);

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

        // 4. Encode N iterations x 5 passes into one command buffer
        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_evolver_batch"),
        });

        let rwg = p.rasterize_wg;
        let wg_x = (p.image_width + rwg[0] - 1) / rwg[0];
        let wg_y = (p.image_height + rwg[1] - 1) / rwg[1];
        let lambda = mutation_params.lambda;

        // Runtime check: active * lambda must fit in offspring_capacity
        assert!(active * lambda <= p.offspring_capacity,
            "active({}) * lambda({}) = {} exceeds offspring_capacity({})",
            active, lambda, active * lambda, p.offspring_capacity);

        let is_last_iter = |i: u32| i == iterations - 1;
        let mut migrate_ran = false;

        // Determine which iterations need separate passes (for timestamp profiling).
        // The last iteration uses separate passes when collect_timestamps is true,
        // so we can get per-stage timing via ComputePassDescriptor::timestamp_writes.
        //
        // All other iterations are consolidated into single compute passes (one per
        // iteration) that contain mutate + rasterize_error + select + optional migration
        // as dispatches within the same pass. This eliminates 2-3 Vulkan pipeline barriers
        // per iteration that were previously inserted at begin/end_compute_pass boundaries.
        //
        // SAFETY: wgpu's resource tracker inserts vkCmdPipelineBarrier between dispatches
        // within a single compute pass when buffer usage changes (e.g. STORAGE_READ_WRITE
        // -> STORAGE_READ). See flush_states() in wgpu-core/src/command/compute.rs which
        // calls drain_barriers() before every dispatch. STORAGE_READ_WRITE is in BufferUses::
        // EXCLUSIVE (not ORDERED), so transitions are never skipped.

        for i in 0..iterations {
            let use_separate_passes = collect_timestamps && is_last_iter(i);

            // Check if migration runs this iteration
            let global_iter = self.iteration + i;
            let inter_interval = mutation_params.inter_island_interval;
            let migration_type = if global_iter > 0 && inter_interval > 0 && global_iter % inter_interval == 0 {
                Some(MigrationType::Inter)
            } else if global_iter > 0 && global_iter % GPU_MIGRATION_INTERVAL == 0 {
                Some(MigrationType::Intra)
            } else {
                None
            };

            if use_separate_passes {
                // Separate passes for per-stage timestamp profiling
                let qs = &p.timestamp_query_set;

                // Pass 1: Mutate
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mutate"),
                        timestamp_writes: Some(ComputePassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }),
                    });
                    pass.set_pipeline(&p.mutate_pipeline);
                    pass.set_bind_group(0, &p.mutate_bind_group, &[]);
                    pass.set_push_constants(0, params_bytes);
                    pass.dispatch_workgroups(active, 1, 1);
                }

                // Pass 2: Rasterize + Error
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("rasterize_error"),
                        timestamp_writes: Some(ComputePassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some(2),
                            end_of_pass_write_index: Some(3),
                        }),
                    });
                    pass.set_pipeline(&p.rasterize_error_pipeline);
                    pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
                    pass.set_push_constants(0, params_bytes);
                    pass.dispatch_workgroups(wg_x, wg_y, active * lambda);
                }

                // Pass 3: Select
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("select"),
                        timestamp_writes: Some(ComputePassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some(4),
                            end_of_pass_write_index: Some(5),
                        }),
                    });
                    pass.set_pipeline(&p.select_pipeline);
                    pass.set_bind_group(0, &p.select_bind_group, &[]);
                    pass.set_push_constants(0, params_bytes);
                    pass.dispatch_workgroups(active, 1, 1);
                }

                // Pass 4: Migration (separate pass for timestamp)
                if let Some(mt) = migration_type {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some(mt.label()),
                        timestamp_writes: Some(ComputePassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some(6),
                            end_of_pass_write_index: Some(7),
                        }),
                    });
                    pass.set_pipeline(mt.pipeline(p));
                    pass.set_bind_group(0, &p.migrate_bind_group, &[]);
                    pass.set_push_constants(0, params_bytes);
                    pass.dispatch_workgroups(active, 1, 1);
                    migrate_ran = true;
                }
            } else {
                // Consolidated: mutate + rasterize + select (+ optional migration)
                // all within a single compute pass. wgpu inserts Vulkan barriers between
                // dispatches based on storage buffer usage tracking.
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("iteration"),
                    timestamp_writes: None,
                });

                // Dispatch 1: Mutate
                pass.set_pipeline(&p.mutate_pipeline);
                pass.set_bind_group(0, &p.mutate_bind_group, &[]);
                pass.set_push_constants(0, params_bytes);
                pass.dispatch_workgroups(active, 1, 1);

                // Dispatch 2: Rasterize + Error
                pass.set_pipeline(&p.rasterize_error_pipeline);
                pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
                pass.set_push_constants(0, params_bytes);
                pass.dispatch_workgroups(wg_x, wg_y, active * lambda);

                // Dispatch 3: Select
                pass.set_pipeline(&p.select_pipeline);
                pass.set_bind_group(0, &p.select_bind_group, &[]);
                pass.set_push_constants(0, params_bytes);
                pass.dispatch_workgroups(active, 1, 1);

                // Dispatch 4: Migration (if needed, same pass)
                if let Some(mt) = migration_type {
                    pass.set_pipeline(mt.pipeline(p));
                    pass.set_bind_group(0, &p.migrate_bind_group, &[]);
                    pass.set_push_constants(0, params_bytes);
                    pass.dispatch_workgroups(active, 1, 1);
                    if is_last_iter(i) {
                        migrate_ran = true;
                    }
                }
            }
        }

        // 5. Resolve timestamp queries into resolve buffer, then copy to staging[write_idx]
        if collect_timestamps {
            encoder.resolve_query_set(&p.timestamp_query_set, 0..8, &p.timestamp_resolve_buf, 0);
            encoder.copy_buffer_to_buffer(
                &p.timestamp_resolve_buf,
                0,
                &p.timestamp_staging_bufs[write_idx],
                0,
                8 * 8,
            );
        }

        // Copy control flags to staging[write_idx] for readback
        encoder.copy_buffer_to_buffer(
            &p.control_flags_buf,
            0,
            &p.control_staging_bufs[write_idx],
            0,
            std::mem::size_of::<ControlFlags>() as u64,
        );

        // Copy fitness_packed to staging[write_idx] for readback (only active chains)
        let fitness_size = (active as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &p.fitness_packed_buf,
            0,
            &p.fitness_staging_bufs[write_idx],
            0,
            fitness_size,
        );

        // 6. Submit — GPU starts working on this batch
        p.queue.submit(std::iter::once(encoder.finish()));

        self.iteration += iterations;
        self.total_evaluations += iterations as u64 * active as u64 * lambda as u64;

        // 7. Issue map_async on staging set [write_idx] — starts the async map
        //    but don't poll yet (that happens at the start of the NEXT run_batch call)
        self.start_async_map(write_idx, collect_timestamps, active);

        // 8. Store pending batch info so next call can read results
        self.pending_batch = Some(PendingBatch {
            staging_idx: write_idx,
            migrate_ran,
            collect_timestamps,
            active_chain_count: active,
        });

        // 9. Return the PREVIOUS batch's result (one batch behind)
        prev_result.flatten()
    }

    /// Issue map_async on the staging buffers at the given index.
    /// Stores the receiver channels so finish_pending_readback can poll them.
    fn start_async_map(&mut self, idx: usize, collect_timestamps: bool, active: u32) {
        let p = &self.pipeline;

        let control_slice = p.control_staging_bufs[idx].slice(..);
        let fitness_slice = p.fitness_staging_bufs[idx].slice(..((active as u64) * 4));

        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx3, rx3) = std::sync::mpsc::channel();

        control_slice.map_async(MapMode::Read, move |r| { tx1.send(r).unwrap(); });

        let ts_rx = if collect_timestamps {
            let timestamp_slice = p.timestamp_staging_bufs[idx].slice(..);
            let (tx2, rx2) = std::sync::mpsc::channel();
            timestamp_slice.map_async(MapMode::Read, move |r| { tx2.send(r).unwrap(); });
            Some(rx2)
        } else {
            None
        };

        fitness_slice.map_async(MapMode::Read, move |r| { tx3.send(r).unwrap(); });

        self.pending_map_receivers = Some(PendingMapReceivers {
            control_rx: rx1,
            timestamp_rx: ts_rx,
            fitness_rx: rx3,
        });
    }

    /// Poll the device, wait for all pending GPU work to complete, then read
    /// the mapped staging buffers from a previously submitted batch.
    /// Returns `Some(Drawing)` if that batch found a new global best.
    fn finish_pending_readback(&mut self, pending: &PendingBatch) -> Option<Drawing> {
        let receivers = self.pending_map_receivers.take()
            .expect("finish_pending_readback called without pending map receivers");

        let p = &self.pipeline;
        let idx = pending.staging_idx;
        let active = pending.active_chain_count as usize;

        // Single poll waits for ALL pending GPU work — both the previous batch's
        // map_async and any newly submitted command buffer
        p.device.poll(Maintain::Wait);

        receivers.control_rx.recv().unwrap().expect("Failed to map control staging buffer");
        if let Some(ref ts_rx) = receivers.timestamp_rx {
            ts_rx.recv().unwrap().expect("Failed to map timestamp staging buffer");
        }
        receivers.fitness_rx.recv().unwrap().expect("Failed to map fitness staging buffer");

        // Read control flags
        let control_slice = p.control_staging_bufs[idx].slice(..);
        let control_data = control_slice.get_mapped_range();
        let flags: ControlFlags = *bytemuck::from_bytes(&control_data);
        drop(control_data);

        // Read timestamps (only if collected)
        if pending.collect_timestamps {
            let ts_data = p.timestamp_staging_bufs[idx].slice(..).get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&ts_data);
            let period = p.timestamp_period as f64;
            let duration = |begin_idx: usize, end_idx: usize| -> f64 {
                timestamps[end_idx].wrapping_sub(timestamps[begin_idx]) as f64 * period
            };
            self.pass_timings.mutate_ns += duration(0, 1);
            self.pass_timings.rasterize_error_ns += duration(2, 3);
            self.pass_timings.select_ns += duration(4, 5);
            self.pass_timings.sample_count += 1;
            if pending.migrate_ran {
                self.pass_timings.migrate_ns += duration(6, 7);
                self.pass_timings.migrate_sample_count += 1;
            }
            drop(ts_data);
            p.timestamp_staging_bufs[idx].unmap();
        }

        // Read chain fitness
        let fitness_slice = p.fitness_staging_bufs[idx].slice(..((active as u64) * 4));
        let fitness_data = fitness_slice.get_mapped_range();
        let packed: &[u32] = bytemuck::cast_slice(&fitness_data);
        for (i, &bits) in packed.iter().enumerate() {
            self.chain_fitness[i] = f32::from_bits(bits);
        }
        drop(fitness_data);

        // Unmap
        p.control_staging_bufs[idx].unmap();
        p.fitness_staging_bufs[idx].unmap();

        if flags.new_best_found != 0 {
            self.best_fitness_bits = flags.best_fitness_bits;
            let drawing = self.readback_chain(flags.best_chain_id);
            Some(drawing)
        } else {
            None
        }
    }

    /// Flush any pending batch results immediately. Call this before operations
    /// that need the GPU state to be fully resolved (e.g., reinit_chains).
    /// Returns `Some(Drawing)` if the pending batch found a new global best.
    pub fn flush_pending(&mut self) -> Option<Drawing> {
        if let Some(pending) = self.pending_batch.take() {
            self.finish_pending_readback(&pending)
        } else {
            None
        }
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
    /// Flushes any pending batch results first to ensure clean GPU state.
    pub fn reinit_chains(&mut self, drawing: &Drawing) {
        // Flush any in-flight batch before reinitializing
        self.flush_pending();

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
        let zeros = vec![0u8; self.pipeline.offspring_capacity as usize * 4];
        self.pipeline
            .queue
            .write_buffer(&self.pipeline.error_accumulators_buf, 0, &zeros);

        // Reinitialize offspring RNG states
        init_offspring_rng(&self.pipeline, self.iteration);

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
        estimate_gpu_memory(p.chain_count, p.offspring_capacity, p.image_width, p.image_height) as u64
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

    pub fn rasterize_wg(&self) -> [u32; 2] {
        self.pipeline.rasterize_wg
    }

    /// Prepare for a benchmark: reinitialize chains, trigger pipeline recreation
    /// if needed, run 2 warmup batches to fill the double-buffer and populate
    /// chain fitness values, then reset all counters. Returns the initial fitness
    /// from GPU evaluation so the benchmark starts with accurate data.
    pub fn prepare_for_benchmark(&mut self, drawing: &Drawing, params: &MutationParams) {
        // 1. Reinit chains from snapshot (uploads drawing, zeros counters)
        self.reinit_chains(drawing);

        // 2. Trigger pipeline recreation before timing starts
        self.pipeline.set_rasterize_wg(params.rasterize_wg);

        // 3. Run 2 warmup batches to fill the double-buffer pipeline
        //    and get initial fitness values read back from GPU
        self.run_batch(params, false); // batch 1: submits, returns None
        self.run_batch(params, false); // batch 2: reads back batch 1 results

        // 4. Reset all counters so the benchmark starts clean
        self.iteration = 0;
        self.total_evaluations = 0;
        self.start_time = Instant::now();
        self.pass_timings = PassTimings::default();
        // Note: chain_fitness and best_fitness_bits are now populated
        // from the warmup readback — don't reset them
    }
}

/// Initialize offspring RNG states in the working_states buffer.
/// Each offspring slot gets a unique persistent RNG seed.
fn init_offspring_rng(pipeline: &GpuPipeline, iteration: u32) {
    let capacity = pipeline.offspring_capacity as usize;
    // Create zeroed states with unique RNG seeds — only the rng_state and mutation_scale matter
    let states: Vec<GpuDrawingState> = (0..capacity)
        .map(|i| {
            let mut s = GpuDrawingState::zeroed();
            let seed = 0xCAFE_BABE_u64
                .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15))
                .wrapping_add((iteration as u64).wrapping_mul(0x517CC1B727220A95));
            s.rng_state[0] = seed as u32;
            s.rng_state[1] = (seed >> 32) as u32;
            let inc = seed.wrapping_mul(6364136223846793005);
            s.rng_state[2] = inc as u32 | 1;
            s.rng_state[3] = (inc >> 32) as u32;
            s.mutation_scale_bits = 1.0f32.to_bits();
            s
        })
        .collect();
    let bytes: Vec<u8> = states
        .iter()
        .flat_map(|s| bytemuck::bytes_of(s).to_vec())
        .collect();
    pipeline.queue.write_buffer(&pipeline.working_states_buf, 0, &bytes);
}

fn estimate_gpu_memory(chain_count: u32, offspring_capacity: u32, w: u32, h: u32) -> usize {
    let k = chain_count as usize;
    let oc = offspring_capacity as usize;
    let pixels = (w * h) as usize;
    let chain_states = k * GPU_DRAWING_STATE_SIZE;
    let working_states = oc * GPU_DRAWING_STATE_SIZE;
    let reference = pixels * 4;
    let error_accumulators = oc * 4;
    let control = 16;
    let params = std::mem::size_of::<GpuParams>();
    // Double-buffered staging: 2x control (16B each) + 2x fitness (k*4 each) + 2x timestamp (64B each) + readback
    let staging = GPU_DRAWING_STATE_SIZE + 2 * 16 + 2 * (k * 4) + 2 * 64;
    chain_states + working_states + reference + error_accumulators + control + params + staging
}
