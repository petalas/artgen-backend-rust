pub mod benchmark;
pub mod buffers;
pub mod pipeline;
mod execution;
mod programs;
mod readback;

use std::time::Instant;

use bytemuck::Zeroable;
use wgpu::*;

use crate::models::drawing::Drawing;
use crate::mutation_params::MutationParams;
use crate::settings::{GPU_MAX_CHAIN_COUNT, GPU_DEFAULT_CHAIN_COUNT};

use buffers::{
    gpu_params_from, default_gpu_params, drawing_to_gpu, gpu_to_drawing, ControlFlags,
    GpuDrawingState, GPU_DRAWING_STATE_SIZE,
};
use pipeline::GpuPipeline;
use readback::{Readback, SubmittedBatch};

#[derive(Default)]
pub struct PassTimings {
    pub mutate_ns: f64,
    pub bin_polygons_ns: f64,
    pub rasterize_error_ns: f64,
    pub select_ns: f64,
    pub sample_count: u64,
}

impl PassTimings {
    pub fn total_ns(&self) -> f64 {
        self.mutate_ns + self.bin_polygons_ns + self.rasterize_error_ns + self.select_ns
    }

    pub fn print_averages(&self) {
        if self.sample_count == 0 {
            return;
        }
        let n = self.sample_count as f64;
        let mutate = self.mutate_ns / n / 1_000_000.0;
        let bin = self.bin_polygons_ns / n / 1_000_000.0;
        let rasterize_error = self.rasterize_error_ns / n / 1_000_000.0;
        let select = self.select_ns / n / 1_000_000.0;
        let total = mutate + bin + rasterize_error + select;

        if total <= 0.0 {
            return;
        }

        println!("GPU pass timings (avg over {} batches):", self.sample_count);
        println!("  mutate:          {:6.2}ms ({:5.1}%)", mutate, mutate / total * 100.0);
        if bin > 0.001 {
            println!("  bin_polygons:    {:6.2}ms ({:5.1}%)", bin, bin / total * 100.0);
        }
        println!("  rasterize+error: {:6.2}ms ({:5.1}%)", rasterize_error, rasterize_error / total * 100.0);
        println!("  select:          {:6.2}ms ({:5.1}%)", select, select / total * 100.0);
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
    readback: Readback,
    benchmark_batch_envelope: bool,
    active_chain_count: u32, // runtime chain count (≤ pipeline.chain_count)
    framebuffers_initialized: bool, // has init_framebuffers been dispatched?
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

        let rwg = pipeline.rasterize_wg();
        let grid = pipeline.raster_grid(rwg);
        let (wg_x, wg_y) = (grid.x, grid.y);
        println!(
            "GPU evolver initialized: {} active chains (max {}), offspring capacity {}, {}x{} image, {:.1} MB GPU memory, rasterize dispatch {}x{}x{} (wg {}x{})",
            active_chains,
            actual_max,
            pipeline.offspring_capacity,
            image_width,
            image_height,
            gpu_memory_bytes(&pipeline) as f64 / (1024.0 * 1024.0),
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
            readback: Readback::new(actual_max),
            benchmark_batch_envelope: false,
            active_chain_count: active_chains,
            framebuffers_initialized: false,
        }
    }

    /// Dispatch the init_framebuffers shader: full rasterize from chain_states → framebuffers + total errors.
    /// Blocks until the GPU work completes.
    fn dispatch_init_framebuffers(&mut self, active_chains: u32) {
        let p = &self.pipeline;
        let rwg = p.rasterize_wg();
        let grid = p.raster_grid(rwg);
        let (wg_x, wg_y) = (grid.x, grid.y);

        // Zero chain_total_errors before init
        let zeros = vec![0u8; p.chain_count as usize * 4];
        p.queue.write_buffer(&p.chain_total_errors_buf, 0, &zeros);

        let params = default_gpu_params(p.image_width, p.image_height, active_chains);
        let params_bytes: &[u8] = bytemuck::bytes_of(&params);
        p.queue.write_buffer(&p.params_buf, 0, params_bytes);

        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("init_framebuffers"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("init_framebuffers"),
                timestamp_writes: None,
            });
            pass.set_pipeline(p.init_framebuffers_pipeline());
            pass.set_bind_group(0, &p.init_framebuffers_bind_group, &[]);
            pass.set_bind_group(1, &p.params_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, active_chains);
        }
        let si = p.queue.submit(std::iter::once(encoder.finish()));
        p.device.poll(PollType::Wait { submission_index: Some(si), timeout: None }).unwrap();

        self.framebuffers_initialized = true;
        println!("Initialized chain framebuffers ({} chains)", active_chains);
    }

    /// Run a batch of N iterations on the GPU using double-buffered staging.
    ///
    /// Returns the PREVIOUS batch's result (one batch behind). The first call
    /// always returns `None`. The new batch is submitted to the GPU BEFORE
    /// blocking on the previous batch's readback, so the GPU is already
    /// computing batch N+1 while the CPU processes batch N's results.
    ///
    /// Returns `Some(Drawing)` if the previous batch found a new global best.
    pub fn run_batch(&mut self, mutation_params: &MutationParams, collect_timestamps: bool) -> Option<Drawing> {
        assert!(!self.benchmark_batch_envelope || mutation_params.gpu_batch_iters >= 2,
            "batch compute envelope requires at least two iterations");
        // Check if rasterize workgroup size needs to change (between batches)
        self.pipeline.set_rasterize_wg(mutation_params.rasterize_wg);

        // Initialize framebuffers if needed (incremental eval is always on).
        // This must happen before submission and requires a synchronous GPU round-trip,
        // so flush any pending batch first to avoid conflicting submissions.
        let active_preview = mutation_params.chain_count.min(self.pipeline.chain_count);
        if !self.framebuffers_initialized {
            let flush_result = self.flush_pending();
            self.dispatch_init_framebuffers(active_preview);
            return self.run_batch_inner(mutation_params, collect_timestamps).or(flush_result);
        }

        self.run_batch_inner(mutation_params, collect_timestamps)
    }

    /// Submit before collecting the previous batch, then register the new maps.
    /// A conflict on the single drawing staging buffer requires early collection.
    fn run_batch_inner(
        &mut self,
        mutation_params: &MutationParams,
        collect_timestamps: bool,
    ) -> Option<Drawing> {
        let early_result = self.readback.collect_if_conflicting(&self.pipeline);

        let p = &self.pipeline;
        let work = p.workload(mutation_params);
        let iterations = work.iterations;
        let batch_envelope = self.benchmark_batch_envelope && collect_timestamps;
        let active = work.active;
        self.active_chain_count = active;

        // 1. Pick which staging set to use for THIS batch's copies.
        //    The previous batch (if any) uses the OTHER staging set,
        //    so there's no conflict between the new submission's copy
        //    destinations and the previous batch's mapped-for-read buffers.
        let write_idx = self.readback.next_staging_index();

        // 2. Build params and upload to uniform buffer
        let mut params = gpu_params_from(mutation_params, p.image_width, p.image_height, active);
        params.iteration_number = self.iteration;
        let params_bytes: &[u8] = bytemuck::bytes_of(&params);
        p.queue.write_buffer(&p.params_buf, 0, params_bytes);

        // Reset control flags before batch — preserve best_fitness_bits so atomicMax
        // only triggers new_best_found when fitness actually improves over last known best.
        // Note: best_fitness_bits may be one batch stale (from batch N-2 instead of N-1)
        // since we haven't collected the previous batch yet. This is harmless — worst case
        // the shader re-detects an improvement already found, causing a redundant readback.
        let control_reset = ControlFlags {
            new_best_found: 0,
            best_chain_id: 0,
            best_fitness_bits: self.readback.best_fitness_bits,
            _pad: 0,
        };
        p.queue.write_buffer(&p.control_flags_buf, 0, bytemuck::bytes_of(&control_reset));

        // Note: error accumulators are NOT reset here — the select shader
        // resets them via atomicExchange after each iteration. They're
        // zero-initialized once at buffer creation and in reinit_chains().

        // 3. Encode N iterations into one command buffer
        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_evolver_batch"),
        });

        let (wg_x, wg_y) = (work.raster.x, work.raster.y);
        let offspring = work.offspring;
        let mutate_dispatch_y = work.mutate_y;

        // Bulk iterations use two passes. The sampled final iteration uses four
        // separate passes for the original per-stage timestamp pairs.
        //
        // SAFETY: wgpu's resource tracker inserts vkCmdPipelineBarrier between dispatches
        // within a single compute pass when buffer usage changes (e.g. STORAGE_READ_WRITE
        // -> STORAGE_READ). See flush_states() in wgpu-core/src/command/compute.rs which
        // calls drain_barriers() before every dispatch.

        // Bulk iterations: split into two passes per iteration.
        // Pass A: mutate → bin → rasterize_error (internal barriers OK via usage changes)
        // Pass B: select → update_framebuffers (internal barriers OK via usage changes)
        //
        // The pass boundary between A and B provides the critical memory barrier for
        // error_accumulators: rasterize_error (atomicAdd, RW) → select (atomicExchange, RW).
        // Without this barrier, select can read incomplete sums — wgpu's resource tracker
        // only inserts barriers on usage *changes*, and RW→RW is not a change.
        // The pass boundary between B(N) and A(N+1) similarly ensures select's
        // atomicExchange (zeroing) is visible to the next rasterize_error's atomicAdd.
        let bulk_count = if collect_timestamps { iterations.saturating_sub(1) } else { iterations };
        // Consume this marker at the first bulk pass without a per-iteration branch.
        // Query 7 still marks the end of the final select/update pass.
        let mut bulk_timestamp_writes = batch_envelope.then_some(ComputePassTimestampWrites {
            query_set: &p.timestamp_query_set,
            beginning_of_pass_write_index: Some(6),
            end_of_pass_write_index: None,
        });
        for _ in 0..bulk_count {
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("mutate_bin_rasterize"),
                    timestamp_writes: bulk_timestamp_writes.take(),
                });
                pass.set_pipeline(p.mutate_pipeline());
                pass.set_bind_group(0, &p.mutate_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(active, mutate_dispatch_y, 1);

                pass.set_pipeline(p.bin_polygons_pipeline());
                pass.set_bind_group(0, &p.bin_polygons_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(offspring, 1, 1);

                pass.set_pipeline(p.rasterize_error_pipeline());
                pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, offspring);
            }
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("select_update"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(p.select_pipeline());
                pass.set_bind_group(0, &p.select_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(active, 1, 1);

                pass.set_pipeline(p.update_framebuffers_pipeline());
                pass.set_bind_group(0, &p.update_framebuffers_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, active);
            }
        }

        // Last iteration with timestamps: separate passes for per-stage profiling.
        if collect_timestamps {
            let qs = &p.timestamp_query_set;

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("mutate"),
                    timestamp_writes: Some(ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(p.mutate_pipeline());
                pass.set_bind_group(0, &p.mutate_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(active, mutate_dispatch_y, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("bin_polygons"),
                    timestamp_writes: Some(ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }),
                });
                pass.set_pipeline(p.bin_polygons_pipeline());
                pass.set_bind_group(0, &p.bin_polygons_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(offspring, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("rasterize_error"),
                    timestamp_writes: Some(ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    }),
                });
                pass.set_pipeline(p.rasterize_error_pipeline());
                pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, offspring);
            }

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("select"),
                    timestamp_writes: Some(ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: if batch_envelope { None } else { Some(6) },
                        end_of_pass_write_index: Some(7),
                    }),
                });
                pass.set_pipeline(p.select_pipeline());
                pass.set_bind_group(0, &p.select_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(active, 1, 1);

                // Update framebuffers in same pass (timestamps cover both)
                pass.set_pipeline(p.update_framebuffers_pipeline());
                pass.set_bind_group(0, &p.update_framebuffers_bind_group, &[]);
                pass.set_bind_group(1, &p.params_bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, active);
            }
        }

        // 4. Resolve timestamp queries into resolve buffer, then copy to staging[write_idx]
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

        // Piggyback deferred best-chain readback: if a previous batch detected a
        // new global best, copy that chain's state to readback_staging_buf now
        // instead of doing a separate synchronous GPU round-trip.
        let copied_drawing = self.readback.encode_best_copy(p, &mut encoder);

        // 5. Submit — GPU starts working on this batch IMMEDIATELY
        let submission_index = p.queue.submit(std::iter::once(encoder.finish()));

        self.iteration += iterations;
        let evaluations = work.evaluations;
        self.total_evaluations += evaluations;

        // Collect the previous batch only after this submission. The readback
        // owner then registers this batch's maps and keeps their metadata together.
        self.readback
            .after_submit(
                p,
                SubmittedBatch {
                    staging_idx: write_idx,
                    collect_timestamps,
                    batch_envelope,
                    active_chain_count: active,
                    submission_index,
                    iterations,
                    evaluations,
                },
                copied_drawing,
            )
            .or(early_result)
    }

    /// Flush any pending batch results immediately. Call this before operations
    /// that need the GPU state to be fully resolved (e.g., reinit_chains).
    /// Returns `Some(Drawing)` if any pending readback (deferred or in-flight)
    /// produced a new global best.
    pub fn flush_pending(&mut self) -> Option<Drawing> {
        self.readback.flush(&self.pipeline)
    }

    pub fn readback_chain(&self, chain_id: u32) -> Drawing {
        readback::readback_chain(&self.pipeline, chain_id)
    }


    /// Batch-readback multiple chains in a single GPU submission.
    /// Avoids N separate round-trips. Uses a persistent staging buffer
    /// pre-allocated for chain_count chains.
    pub fn readback_chains(&self, chain_ids: &[u32]) -> Vec<Drawing> {
        if chain_ids.is_empty() {
            return Vec::new();
        }

        let p = &self.pipeline;
        let count = chain_ids.len();
        let state_size = GPU_DRAWING_STATE_SIZE as u64;

        assert!(
            count as u32 <= p.chain_count,
            "readback_chains: requested {} chains but staging buffer holds max {}",
            count, p.chain_count,
        );

        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("multi_readback"),
        });

        for (i, &chain_id) in chain_ids.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                &p.chain_states_buf,
                chain_id as u64 * state_size,
                &p.multi_readback_staging_buf,
                i as u64 * state_size,
                state_size,
            );
        }

        let si = p.queue.submit(std::iter::once(encoder.finish()));

        let copy_size = count as u64 * state_size;
        let slice = p.multi_readback_staging_buf.slice(..copy_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| { tx.send(r).unwrap(); });
        p.device.poll(PollType::Wait { submission_index: Some(si), timeout: None }).unwrap();
        rx.recv().unwrap().expect("Failed to map multi-readback staging buffer");

        let data = slice.get_mapped_range()
            .expect("Failed to access mapped drawings readback buffer");
        let drawings: Vec<Drawing> = (0..count)
            .map(|i| {
                let offset = i * GPU_DRAWING_STATE_SIZE;
                let state: &GpuDrawingState = bytemuck::from_bytes(&data[offset..offset + GPU_DRAWING_STATE_SIZE]);
                gpu_to_drawing(state)
            })
            .collect();
        drop(data);
        p.multi_readback_staging_buf.unmap();

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

        // Reset error accumulators so the first batch after reinit starts clean (stride-2)
        let zeros = vec![0u8; self.pipeline.offspring_capacity as usize * 8];
        self.pipeline
            .queue
            .write_buffer(&self.pipeline.error_accumulators_buf, 0, &zeros);

        // Reinitialize offspring RNG states
        init_offspring_rng(&self.pipeline, self.iteration);

        // Reset framebuffers so they get re-initialized on next incremental_eval batch
        self.framebuffers_initialized = false;

        self.iteration = 0;
        self.total_evaluations = 0;
        self.readback.benchmark_counters = benchmark::BenchmarkCounters::default();
        self.readback.best_fitness_bits = 0;
        self.start_time = Instant::now();
        self.reset_pass_timings();
        self.readback.chain_fitness.fill(0.0);
    }

    /// Evaluate fitness of the current chain_states on the GPU without running
    /// any mutations. Copies chain_states → working_states, dispatches
    /// rasterize_error, reads back error accumulators, computes fitness,
    /// and updates both chain_fitness[] and chain_states[].fitness_bits.
    ///
    /// This is synchronous — blocks until the GPU readback completes.
    /// Returns the best fitness across all active chains.
    pub fn evaluate_chain_fitness(&mut self, chain_count: u32) -> f32 {
        self.flush_pending();

        let p = &self.pipeline;
        let active = chain_count.min(p.chain_count);
        self.active_chain_count = active;

        let state_size = GPU_DRAWING_STATE_SIZE as u64;
        let copy_bytes = active as u64 * state_size;

        // Zero error accumulators for the active chains (stride-2)
        let zeros = vec![0u8; active as usize * 8];
        p.queue.write_buffer(&p.error_accumulators_buf, 0, &zeros);

        // Submit 1: copy chain_states → working_states
        // (must be a separate submit so the write_buffer calls below land AFTER the copy)
        {
            let mut copy_enc = p.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("evaluate_fitness_copy"),
            });
            copy_enc.copy_buffer_to_buffer(
                &p.chain_states_buf, 0,
                &p.working_states_buf, 0,
                copy_bytes,
            );
            p.queue.submit(std::iter::once(copy_enc.finish()));
        }

        // Write full-image dirty bbox into each working state so the always-on
        // tile skip in rasterize_error doesn't cull any tiles.
        // Packing: fitness_bits = (min_x | min_y<<16), stagnation_counter = (max_x | max_y<<16)
        let bbox_lo = 0u32; // min_x=0, min_y=0
        let bbox_hi = p.image_width | (p.image_height << 16);
        for i in 0..active as u64 {
            let base = i * state_size;
            p.queue.write_buffer(&p.working_states_buf, base + 4, &bbox_lo.to_le_bytes());
            p.queue.write_buffer(&p.working_states_buf, base + 12, &bbox_hi.to_le_bytes());
        }

        // Submit 2: bin polygons + rasterize_error
        let rwg = p.rasterize_wg();
        let grid = p.raster_grid(rwg);
        let (wg_x, wg_y) = (grid.x, grid.y);

        let params = default_gpu_params(p.image_width, p.image_height, active);
        let params_bytes: &[u8] = bytemuck::bytes_of(&params);
        p.queue.write_buffer(&p.params_buf, 0, params_bytes);

        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("evaluate_fitness"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("evaluate_fitness"),
                timestamp_writes: None,
            });

            // Bin polygons first (tile culling is always on)
            pass.set_pipeline(p.bin_polygons_pipeline());
            pass.set_bind_group(0, &p.bin_polygons_bind_group, &[]);
            pass.set_bind_group(1, &p.params_bind_group, &[]);
            pass.dispatch_workgroups(active, 1, 1);

            pass.set_pipeline(p.rasterize_error_pipeline());
            pass.set_bind_group(0, &p.rasterize_error_bind_group, &[]);
            pass.set_bind_group(1, &p.params_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, active);
        }

        // Use persistent staging buffer: errors (active * 8 bytes, stride-2) + polygon_counts (active * 4 bytes)
        let errors_size = active as u64 * 8; // stride-2: 2 u32s per offspring
        let staging_copy_size = errors_size + active as u64 * 4;
        encoder.copy_buffer_to_buffer(
            &p.error_accumulators_buf, 0,
            &p.eval_fitness_staging_buf, 0,
            errors_size,
        );
        // Copy polygon_count (first u32) from each chain's working_states
        for i in 0..active {
            encoder.copy_buffer_to_buffer(
                &p.working_states_buf,
                i as u64 * state_size,
                &p.eval_fitness_staging_buf,
                errors_size + i as u64 * 4,
                4,
            );
        }

        let si = p.queue.submit(std::iter::once(encoder.finish()));

        // Map and read error values + polygon counts
        let slice = p.eval_fitness_staging_buf.slice(..staging_copy_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| { tx.send(r).unwrap(); });
        p.device.poll(PollType::Wait { submission_index: Some(si), timeout: None }).unwrap();
        rx.recv().unwrap().expect("Failed to map eval fitness staging buffer");

        let data = slice.get_mapped_range()
            .expect("Failed to access mapped evaluation buffer");
        let all_u32s: &[u32] = bytemuck::cast_slice(&data);
        // Errors are at stride-2 (slot 0 = new error, slot 1 = old error); pick slot 0 for each
        let error_u32s = &all_u32s[..active as usize * 2];
        let errors: Vec<u32> = (0..active as usize).map(|i| error_u32s[i * 2]).collect();
        let polygon_counts = &all_u32s[active as usize * 2..];

        // Compute fitness from errors with point penalty (matching select.wgsl's compute_fitness)
        let max_error = crate::settings::MAX_ERROR_PER_PIXEL
            * (p.image_width * p.image_height) as f32;
        let per_point_mul = crate::settings::PER_POINT_MULTIPLIER;

        let mut best_fitness = 0.0f32;
        for i in 0..active as usize {
            let total_error = errors[i]; // already extracted from stride-2
            let polygon_count = polygon_counts[i];
            let mut fitness = 100.0 * (1.0 - total_error as f32 / max_error);
            let num_points = polygon_count * 3;
            fitness -= fitness * per_point_mul * num_points as f32;
            self.readback.chain_fitness[i] = fitness;
            if fitness > best_fitness {
                best_fitness = fitness;
            }
        }

        drop(data);
        p.eval_fitness_staging_buf.unmap();

        // Write fitness_bits back into chain_states and initialize chain_total_errors
        // so the first run_batch's select shader has correct baselines for incremental eval.
        for i in 0..active as usize {
            let fitness_bits = self.readback.chain_fitness[i].to_bits();
            let offset = i as u64 * state_size + 4; // fitness_bits is at offset 4 in GpuDrawingState
            p.queue.write_buffer(
                &p.chain_states_buf,
                offset,
                &fitness_bits.to_le_bytes(),
            );
            // chain_total_errors: select shader reads this for incremental error delta
            p.queue.write_buffer(
                &p.chain_total_errors_buf,
                i as u64 * 4,
                &errors[i].to_le_bytes(),
            );
        }
        self.readback.best_fitness_bits = best_fitness.to_bits();

        best_fitness
    }

    pub fn chain_fitness(&self) -> &[f32] {
        &self.readback.chain_fitness[..self.active_chain_count as usize]
    }

    pub fn chain_count(&self) -> u32 {
        self.active_chain_count
    }

    pub fn max_chain_count(&self) -> u32 {
        self.pipeline.chain_count
    }

    pub fn estimated_memory_bytes(&self) -> u64 {
        gpu_memory_bytes(&self.pipeline)
    }

    pub fn pass_timings(&self) -> &PassTimings {
        &self.readback.pass_timings
    }

    pub fn reset_pass_timings(&mut self) {
        self.readback.pass_timings.reset();
        self.readback.benchmark_envelope_timings = benchmark::BenchmarkEnvelopeTimings::default();
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
        self.pipeline.rasterize_wg()
    }

    /// Save the pipeline cache to disk for faster startup next time.
    pub fn save_pipeline_cache(&self) {
        self.pipeline.save_pipeline_cache();
    }

    /// Prepare for a benchmark: reinitialize chains, trigger pipeline recreation
    /// if needed, run warmup batches, evaluate initial fitness on GPU, then
    /// reset all counters. Returns the GPU-evaluated initial fitness.
    ///
    /// Warmup ensures GPU shader caches, memory pages, and clock boost are
    /// at steady state before timing begins. The chains are re-initialized
    /// after warmup so the benchmark starts from a clean, deterministic state.
    pub fn prepare_for_benchmark(&mut self, drawing: &Drawing, params: &MutationParams) -> f32 {
        // 1. Reset iteration before reinit so RNG seeds are deterministic
        //    across sequential benchmark runs (reinit_chains uses self.iteration
        //    to seed chain and offspring RNG states).
        self.iteration = 0;
        self.reinit_chains(drawing);

        // 2. Trigger pipeline recreation before timing starts
        self.pipeline.set_rasterize_wg(params.rasterize_wg);

        let chain_count = params.chain_count.min(self.pipeline.chain_count);

        // 3. Evaluate fitness so chains have valid fitness_bits for warmup
        self.evaluate_chain_fitness(chain_count);

        // 4. Pre-initialize framebuffers so warmup exercises the full
        //    incremental pipeline (including update_framebuffers)
        self.dispatch_init_framebuffers(chain_count);

        // 5. Run 2 warmup batches to warm GPU shader caches and memory pages,
        //    then flush results. This ensures the first real batch runs at
        //    steady-state speed.
        self.run_batch(params, false);
        self.run_batch(params, false);
        self.flush_pending();

        // 6. Reinit to original drawing for a clean, deterministic start
        self.reinit_chains(drawing);
        let start_fitness = self.evaluate_chain_fitness(chain_count);

        // 7. Pre-initialize framebuffers so this cost is excluded from benchmark timing
        self.dispatch_init_framebuffers(chain_count);

        // 8. Reset all counters so the benchmark starts clean
        self.iteration = 0;
        self.total_evaluations = 0;
        self.start_time = Instant::now();
        self.reset_pass_timings();

        start_fitness
    }
}

impl Drop for GpuEvolver {
    fn drop(&mut self) {
        self.pipeline.save_pipeline_cache();
    }
}

/// Sum actual allocated GPU buffer sizes.
fn gpu_memory_bytes(p: &GpuPipeline) -> u64 {
    p.chain_states_buf.size()
        + p.working_states_buf.size()
        + p.error_accumulators_buf.size()
        + p.control_flags_buf.size()
        + p.readback_staging_buf.size()
        + p.multi_readback_staging_buf.size()
        + p.eval_fitness_staging_buf.size()
        + p.control_staging_bufs[0].size()
        + p.control_staging_bufs[1].size()
        + p.fitness_packed_buf.size()
        + p.fitness_staging_bufs[0].size()
        + p.fitness_staging_bufs[1].size()
        + p.tile_data_buf.size()
        + p.tile_counts_buf.size()
        + p.chain_framebuffers_buf.size()
        + p.chain_total_errors_buf.size()
        + p.chain_update_flags_buf.size()
        + p.params_buf.size()
        + p.timestamp_resolve_buf.size()
        + p.timestamp_staging_bufs[0].size()
        + p.timestamp_staging_bufs[1].size()
        + (p.image_width as u64 * p.image_height as u64 * 4) // reference texture
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
            // Only [0] is used by the 32-bit PCG hash; [1..3] are unused padding
            s.rng_state[0] = seed as u32;
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

