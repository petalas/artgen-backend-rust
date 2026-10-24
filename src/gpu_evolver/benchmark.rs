//! Isolated support for mature-drawing restart benchmarks, not checkpoint restore.
//!
//! Preparation and snapshot allocations are outside measurement. The normal batch
//! path only accounts for work once per submitted/collected batch. Diagnostics
//! replace binning for validation or reuse selection queries for a batch compute
//! envelope. Diagnostic timings must not receive a production throughput score.
use std::time::{Duration, Instant};

use bytemuck::Zeroable;
use serde::Serialize;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType};

use super::buffers::{drawing_to_gpu, ControlFlags, GpuDrawingState, GPU_DRAWING_STATE_SIZE};
use super::{GpuEvolver, PassTimings};
use super::execution::ResourceSizes;
use crate::models::drawing::Drawing;
use crate::mutation_params::MutationParams;
use crate::settings::{
    GPU_MAX_BATCH_ITERS, GPU_MAX_LAMBDA, MAX_POLYGONS_PER_IMAGE,
};

pub use super::pipeline::{
    benchmark_mutate_compilation_options, benchmark_mutate_group_size_from_source,
    BenchmarkMutateOptions,
};

const MAX_WARMUP_BATCHES: u32 = 1024;
const READBACK_TIMEOUT: Duration = Duration::from_secs(120);

/// Sum of full-batch GPU compute spans, excluding copies and host work.
/// Available only in the opt-in diagnostic mode; selection timing is unavailable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BenchmarkEnvelopeTimings {
    pub compute_ns: f64,
    pub sample_count: u64,
}

/// The meaning of queries 6/7 belongs to the submitted batch, not current mode.
pub(super) fn decode_batch_timestamps(
    timestamps: &[u64; 8],
    period_ns: f64,
    batch_envelope: bool,
) -> (PassTimings, BenchmarkEnvelopeTimings) {
    let duration = |begin: usize, end: usize| {
        timestamps[end].wrapping_sub(timestamps[begin]) as f64 * period_ns
    };
    let final_pair_ns = duration(6, 7);
    let pass = PassTimings {
        mutate_ns: duration(0, 1),
        bin_polygons_ns: duration(2, 3),
        rasterize_error_ns: duration(4, 5),
        select_ns: if batch_envelope { 0.0 } else { final_pair_ns },
        sample_count: 1,
    };
    let envelope = if batch_envelope {
        BenchmarkEnvelopeTimings {
            compute_ns: final_pair_ns,
            sample_count: 1,
        }
    } else {
        BenchmarkEnvelopeTimings::default()
    };
    (pass, envelope)
}

fn validate_batch_envelope(params: &MutationParams, enabled: bool) -> Result<(), String> {
    if enabled && params.gpu_batch_iters < 2 {
        return Err("batch compute envelope requires at least two iterations".into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct BenchmarkCounters {
    pub submitted_iterations: u64,
    pub submitted_evaluations: u64,
    pub submitted_batches: u64,
    pub completed_iterations: u64,
    pub completed_evaluations: u64,
    pub completed_batches: u64,
}

impl BenchmarkCounters {
    pub(super) fn record_submission(&mut self, iterations: u32, evaluations: u64) {
        self.submitted_iterations += u64::from(iterations);
        self.submitted_evaluations += evaluations;
        self.submitted_batches += 1;
    }

    pub(super) fn record_completion(&mut self, iterations: u32, evaluations: u64) {
        self.completed_iterations += u64::from(iterations);
        self.completed_evaluations += evaluations;
        self.completed_batches += 1;
        debug_assert!(self.completed_batches <= self.submitted_batches);
    }
}

/// Packed words that can affect evolution. Unused polygons and RNG .yzw are omitted.
/// For offspring, `fitness_bits`/`stagnation_counter` encode the dirty bounding box.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BenchmarkPackedState {
    pub polygon_count: u32,
    pub fitness_bits: u32,
    pub mutation_scale_bits: u32,
    pub stagnation_counter: u32,
    pub rng_state: u32,
    pub polygons: Vec<[u32; 4]>,
}

impl BenchmarkPackedState {
    fn from_gpu(state: &GpuDrawingState) -> Result<Self, String> {
        if state.polygon_count as usize > MAX_POLYGONS_PER_IMAGE {
            return Err(format!("invalid GPU polygon count {}", state.polygon_count));
        }
        Ok(Self {
            polygon_count: state.polygon_count,
            fitness_bits: state.fitness_bits,
            mutation_scale_bits: state.mutation_scale_bits,
            stagnation_counter: state.stagnation_counter,
            rng_state: state.rng_state[0],
            polygons: state.polygons[..state.polygon_count as usize]
                .iter()
                .map(|polygon| polygon.data)
                .collect(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BenchmarkControl {
    pub new_best_found: u32,
    pub best_chain_id: u32,
    pub best_fitness_bits: u32,
}

/// Read only after the measured drain. This snapshot is not a resumable checkpoint.
/// Control `best_chain_id` is racy in production; verify it separately from replay
/// hashes. Timestamps, counters and validation occupancy are also not replay state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BenchmarkSnapshot {
    pub parents: Vec<BenchmarkPackedState>,
    pub offspring: Vec<BenchmarkPackedState>,
    pub parent_total_errors: Vec<u32>,
    pub parent_framebuffers: Vec<Vec<u32>>,
    pub offspring_errors: Vec<[u32; 2]>,
    pub chain_update_flags: Vec<[u32; 3]>,
    pub fitness_packed: Vec<u32>,
    pub control: BenchmarkControl,
    pub counters: BenchmarkCounters,
    pub validation_max_tile_occupancy: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkProvenance {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub timestamp_period_ns: f32,
    pub allocated_chains: u32,
    pub offspring_capacity: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
}

/// Keep this on the main thread around prepare/run/drain/snapshot. Scopes are
/// explicitly consumed so validation, allocation and internal errors are checked.
pub struct BenchmarkErrorScopes {
    scopes: [wgpu::ErrorScopeGuard; 3],
}

impl BenchmarkErrorScopes {
    pub async fn finish(self) -> Vec<String> {
        let mut errors = Vec::with_capacity(3);
        for scope in self.scopes.into_iter().rev() {
            if let Some(error) = scope.pop().await {
                errors.push(error.to_string());
            }
        }
        errors
    }
}

impl GpuEvolver {
    #[must_use]
    pub fn benchmark_error_scopes(&self) -> BenchmarkErrorScopes {
        let device = &self.pipeline.device;
        BenchmarkErrorScopes {
            scopes: [
                device.push_error_scope(wgpu::ErrorFilter::Validation),
                device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
                device.push_error_scope(wgpu::ErrorFilter::Internal),
            ],
        }
    }

    /// Warm the same profiled/unprofiled batch path, then restart from exactly the
    /// supplied seed. Warmup count and any previous run cannot affect the restart.
    ///
    /// # Errors
    /// Rejects invalid configuration/drawing, excessive warmup or failed GPU reset/drain.
    pub fn prepare_seeded_benchmark(
        &mut self,
        drawing: &Drawing,
        params: &MutationParams,
        seed: u64,
        warmup_batches: u32,
        profile: bool,
    ) -> Result<f32, String> {
        self.benchmark_validate_capacity(params)?;
        validate_drawing(drawing, params.max_polygons)?;
        if warmup_batches > MAX_WARMUP_BATCHES {
            return Err(format!("warmup_batches exceeds {MAX_WARMUP_BATCHES}"));
        }
        self.flush_pending();
        self.pipeline.try_set_rasterize_wg(params.rasterize_wg)?;
        self.reset_seeded_benchmark(drawing, params.chain_count, seed)?;
        for _ in 0..warmup_batches {
            self.run_batch(params, profile);
        }
        self.benchmark_drain()?;
        self.reset_seeded_benchmark(drawing, params.chain_count, seed)
    }

    /// Reject configurations that production would silently clamp or address past
    /// a clamped allocation. The supported workgroups match `MutationParams`.
    ///
    /// # Errors
    /// Rejects invalid shape, insufficient buffers, overflowing errors or device limits.
    pub fn benchmark_validate_capacity(&self, params: &MutationParams) -> Result<(), String> {
        validate_shape(params)?;
        validate_batch_envelope(params, self.benchmark_batch_envelope)?;
        let p = &self.pipeline;
        let active = u64::from(params.chain_count);
        let offspring = active * u64::from(params.lambda);
        if active > u64::from(p.chain_count) || offspring > u64::from(p.offspring_capacity) {
            return Err("active chains/lambda exceed allocated drawing slots".into());
        }
        validate_dispatch(p, params)?;
        validate_resources(p, params)
    }

    /// Install and cache a lab mutation source using the original layout and
    /// production options, including workgroup-memory zeroing. Retains the handle
    /// across prepares, raster workgroup changes and diagnostic toggles. Logical
    /// group width comes from the source; physical lanes do not change lambda.
    ///
    /// # Errors
    /// Rejects a failed drain, invalid source, exhausted cache or GPU errors.
    pub fn benchmark_set_mutate_shader(&mut self, source: &str) -> Result<(), String> {
        self.benchmark_set_mutate_shader_with_options(source, BenchmarkMutateOptions::default())
    }

    /// Install a mutation source and its explicit compiler policy while drained.
    ///
    /// # Errors
    /// Rejects a failed drain, invalid source, exhausted cache or GPU errors.
    pub fn benchmark_set_mutate_shader_with_options(
        &mut self,
        source: &str,
        options: BenchmarkMutateOptions,
    ) -> Result<(), String> {
        self.benchmark_drain()?;
        self.pipeline.set_benchmark_mutate_shader(source, options)
    }

    /// Actual zero-initialization policy retained with the mutation pipeline.
    #[must_use]
    pub fn benchmark_mutate_zero_initialize_workgroup_memory(&self) -> bool {
        self.pipeline
            .mutate_options
            .zero_initialize_workgroup_memory
    }

    /// Whether the installed mutation pipeline has the exact requested source and policy.
    #[must_use]
    pub fn benchmark_mutate_shader_matches(
        &self,
        source: &str,
        options: BenchmarkMutateOptions,
    ) -> bool {
        self.pipeline.mutate_source.as_ref() == source && self.pipeline.mutate_options == options
    }

    /// Actual logical offspring width retained with the selected mutation pipeline.
    #[must_use]
    pub fn benchmark_mutate_group_size(&self) -> u32 {
        self.pipeline.mutate_group_size
    }

    /// Install a lab bin shader while drained, retaining device, layouts and buffers.
    /// Repeated sources reuse compiled handles. Preparation and validation retain
    /// the override; compilation uses the production bin options and pipeline cache.
    ///
    /// # Errors
    /// Rejects a failed drain, oversized/malformed source, full cache or GPU errors.
    pub fn benchmark_set_bin_shader(&mut self, source: &str) -> Result<(), String> {
        self.benchmark_drain()?;
        self.pipeline.set_benchmark_bin_shader(source)
    }

    /// Install and cache the actual raster source, retaining it across prepares
    /// and workgroup changes. Uses the production layout and compilation options.
    ///
    /// # Errors
    /// Rejects a failed drain, invalid source, exhausted cache or GPU errors.
    pub fn benchmark_set_raster_shader(&mut self, source: &str) -> Result<(), String> {
        self.benchmark_drain()?;
        self.pipeline.set_benchmark_raster_shader(source)
    }

    /// Replace binning only for an untimed replay. Prepare resets the persistent
    /// diagnostic word; snapshot reports the maximum over every binned offspring.
    ///
    /// # Errors
    /// Returns an error if the preceding GPU work cannot be fully drained.
    pub fn benchmark_set_validation(&mut self, enabled: bool) -> Result<(), String> {
        self.benchmark_drain()?;
        self.pipeline.set_benchmark_validation(enabled)?;
        let p = &self.pipeline;
        p.queue
            .write_buffer(&p.tile_counts_buf, p.tile_counts_buf.size() - 4, &[0; 4]);
        Ok(())
    }

    /// Reuse queries 6/7 for the full-batch compute span when collecting timestamps.
    /// The last iteration's mutate/bin/rasterize samples remain available, while
    /// `PassTimings::select_ns` stays zero. This diagnostic changes timestamp
    /// placement and must not be scored as production throughput.
    ///
    /// # Errors
    /// Rejects a failed drain or fewer than two iterations when enabling. After
    /// draining, failed validation retains the mode and collected timings.
    pub fn benchmark_set_batch_envelope(
        &mut self,
        params: &MutationParams,
        enabled: bool,
    ) -> Result<(), String> {
        self.benchmark_drain()?;
        validate_batch_envelope(params, enabled)?;
        self.benchmark_batch_envelope = enabled;
        self.reset_pass_timings();
        Ok(())
    }

    #[must_use]
    pub fn benchmark_envelope_timings(&self) -> BenchmarkEnvelopeTimings {
        self.benchmark_envelope_timings
    }

    #[must_use]
    pub fn benchmark_counters(&self) -> BenchmarkCounters {
        self.benchmark_counters
    }

    /// Includes the existing required production readbacks and final best flush.
    /// Measure this separately from snapshot allocation/copy/readback.
    ///
    /// # Errors
    /// Returns an error if any submitted batch has not completed after flushing.
    pub fn benchmark_drain(&mut self) -> Result<BenchmarkCounters, String> {
        self.flush_pending();
        let counters = self.benchmark_counters;
        if counters.submitted_batches != counters.completed_batches
            || counters.submitted_iterations != counters.completed_iterations
            || counters.submitted_evaluations != counters.completed_evaluations
        {
            return Err("GPU drain left submitted work uncollected".into());
        }
        Ok(counters)
    }

    #[must_use]
    pub fn benchmark_provenance(&self) -> BenchmarkProvenance {
        let p = &self.pipeline;
        let info = p.device.adapter_info();
        let limits = p.device.limits();
        BenchmarkProvenance {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
            timestamp_period_ns: p.timestamp_period,
            allocated_chains: p.chain_count,
            offspring_capacity: p.offspring_capacity,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
        }
    }

    /// Copies active states and cached pixels in one submission. No padding words
    /// are exposed, including the fourth update flag and three unused RNG words.
    ///
    /// # Errors
    /// Rejects invalid counts, exhausted capacity, failed mapping or malformed GPU state.
    pub fn benchmark_snapshot(
        &mut self,
        active: u32,
        lambda: u32,
    ) -> Result<BenchmarkSnapshot, String> {
        let params = MutationParams {
            chain_count: active,
            lambda,
            rasterize_wg: self.pipeline.rasterize_wg,
            ..MutationParams::default()
        };
        self.benchmark_validate_capacity(&params)?;
        if active != self.active_chain_count {
            return Err("snapshot active count differs from prepared/run count".into());
        }
        let counters = self.benchmark_drain()?;
        let p = &self.pipeline;
        let offspring = u64::from(active) * u64::from(lambda);
        let state_size = GPU_DRAWING_STATE_SIZE as u64;
        let pixels = u64::from(p.image_width) * u64::from(p.image_height);
        let sources = [
            (&p.chain_states_buf, 0, u64::from(active) * state_size),
            (&p.working_states_buf, 0, offspring * state_size),
            (&p.chain_total_errors_buf, 0, u64::from(active) * 4),
            (&p.chain_framebuffers_buf, 0, u64::from(active) * pixels * 4),
            (&p.error_accumulators_buf, 0, offspring * 8),
            (&p.chain_update_flags_buf, 0, u64::from(active) * 16),
            (&p.fitness_packed_buf, 0, u64::from(active) * 4),
            (
                &p.control_flags_buf,
                0,
                std::mem::size_of::<ControlFlags>() as u64,
            ),
            (&p.tile_counts_buf, p.tile_counts_buf.size() - 4, 4),
        ];
        let bytes = readback_buffers(p, &sources)?;
        decode_snapshot(
            &bytes,
            active as usize,
            usize::try_from(offspring).map_err(|error| error.to_string())?,
            usize::try_from(pixels).map_err(|error| error.to_string())?,
            counters,
            p.benchmark_validation,
        )
    }

    fn reset_seeded_benchmark(
        &mut self,
        drawing: &Drawing,
        active: u32,
        seed: u64,
    ) -> Result<f32, String> {
        self.benchmark_drain()?;
        let states: Vec<_> = (0..self.pipeline.chain_count)
            .map(|index| drawing_to_gpu(drawing, u64::from(benchmark_seed(seed, index, 0))))
            .collect();
        self.pipeline.queue.write_buffer(
            &self.pipeline.chain_states_buf,
            0,
            bytemuck::cast_slice(&states),
        );
        self.framebuffers_initialized = false;
        self.iteration = 0;
        self.best_fitness_bits = 0;
        self.chain_fitness.fill(0.0);
        let p = &self.pipeline;
        p.queue
            .write_buffer(&p.tile_counts_buf, p.tile_counts_buf.size() - 4, &[0; 4]);
        let fitness = self.evaluate_chain_fitness(active);
        self.dispatch_init_framebuffers(active);
        self.reset_benchmark_gpu_buffers(active, seed, fitness)?;
        self.staging_idx = 0;
        self.pending_best_chain = None;
        self.deferred_best_readback = false;
        self.total_evaluations = 0;
        self.benchmark_counters = BenchmarkCounters::default();
        self.reset_pass_timings();
        self.start_time = Instant::now();
        Ok(fitness)
    }

    fn reset_benchmark_gpu_buffers(
        &self,
        active: u32,
        seed: u64,
        fitness: f32,
    ) -> Result<(), String> {
        // Evaluation overwrites working states and leaves error sums behind.
        // Restore independent offspring RNGs and clear sums AFTER evaluating.
        self.write_benchmark_offspring(seed);
        let p = &self.pipeline;
        let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("seeded_benchmark_reset"),
        });
        encoder.clear_buffer(&p.error_accumulators_buf, 0, None);
        encoder.clear_buffer(&p.chain_update_flags_buf, 0, None);
        // Retain diagnostic initial-drawing occupancy, but discard warmup history.
        encoder.clear_buffer(&p.tile_counts_buf, 0, Some(p.tile_counts_buf.size() - 4));
        let packed_fitness: Vec<_> = self.chain_fitness[..active as usize]
            .iter()
            .map(|fitness| fitness.to_bits())
            .collect();
        p.queue.write_buffer(
            &p.fitness_packed_buf,
            0,
            bytemuck::cast_slice(&packed_fitness),
        );
        let flags = ControlFlags {
            new_best_found: 0,
            best_chain_id: 0,
            best_fitness_bits: fitness.to_bits(),
            _pad: 0,
        };
        p.queue
            .write_buffer(&p.control_flags_buf, 0, bytemuck::bytes_of(&flags));
        let submission = p.queue.submit([encoder.finish()]);
        p.device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(READBACK_TIMEOUT),
            })
            .map_err(|error| format!("benchmark reset poll failed: {error}"))?;
        Ok(())
    }

    fn write_benchmark_offspring(&self, seed: u64) {
        let offspring: Vec<_> = (0..self.pipeline.offspring_capacity)
            .map(|index| {
                let mut state = GpuDrawingState::zeroed();
                state.rng_state[0] = benchmark_seed(seed, index, 1);
                state.mutation_scale_bits = 1.0f32.to_bits();
                state
            })
            .collect();
        let p = &self.pipeline;
        p.queue
            .write_buffer(&p.working_states_buf, 0, bytemuck::cast_slice(&offspring));
    }
}

fn validate_dispatch(
    p: &super::pipeline::GpuPipeline,
    params: &MutationParams,
) -> Result<(), String> {
    let limits = p.device.limits();
    let [wg_x, wg_y] = params.rasterize_wg;
    let pixels = u64::from(p.image_width) * u64::from(p.image_height);
    if pixels == 0 || pixels > u64::from(u32::MAX) / 765 {
        return Err("image is empty or u32 error accumulator can overflow".into());
    }
    let work = p.workload(params);
    let dispatch = [work.raster.x, work.raster.y, work.active, work.offspring];
    if wg_x * wg_y > limits.max_compute_invocations_per_workgroup
        || wg_x * wg_y > limits.max_compute_workgroup_size_x
        || dispatch
            .iter()
            .any(|&count| count > limits.max_compute_workgroups_per_dimension)
    {
        return Err("workgroup or dispatch exceeds device limits".into());
    }
    Ok(())
}

fn validate_resources(
    p: &super::pipeline::GpuPipeline,
    params: &MutationParams,
) -> Result<(), String> {
    let work = p.workload(params);
    let required = ResourceSizes::new(u64::from(work.active), u64::from(work.offspring),
        [p.image_width, p.image_height], work.raster);
    let limits = p.device.limits();
    required.validate(p.allocated_resources(),
        limits.max_buffer_size.min(limits.max_storage_buffer_binding_size))
}

fn validate_shape(params: &MutationParams) -> Result<(), String> {
    if params.chain_count == 0
        || params.lambda == 0
        || params.lambda > GPU_MAX_LAMBDA
        || !params.lambda.is_power_of_two()
        || params.gpu_batch_iters == 0
        || params.gpu_batch_iters > GPU_MAX_BATCH_ITERS
    {
        return Err("chain/lambda/batch counts are invalid".into());
    }
    if ![[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]].contains(&params.rasterize_wg) {
        return Err("unsupported rasterize workgroup".into());
    }
    if params.min_polygons == 0
        || params.min_polygons > params.max_polygons
        || params.max_polygons as usize > MAX_POLYGONS_PER_IMAGE
        || !(1..=16).contains(&params.tournament_size)
        || params.min_alpha > params.max_alpha
    {
        return Err("polygon/tournament/alpha bounds are invalid".into());
    }
    Ok(())
}

fn validate_drawing(drawing: &Drawing, max_polygons: u32) -> Result<(), String> {
    if drawing.polygons.len() > MAX_POLYGONS_PER_IMAGE {
        return Err("drawing exceeds source polygon capacity".into());
    }
    let mut triangles = 0;
    for polygon in &drawing.polygons {
        let points = polygon.points.len();
        if !(3..=MAX_POLYGONS_PER_IMAGE + 2).contains(&points) {
            return Err("drawing contains an invalid or oversized polygon".into());
        }
        triangles += points - 2;
        if triangles > max_polygons as usize {
            return Err("fan-triangulated drawing would be truncated".into());
        }
        if polygon.points.iter().any(|point| {
            !point.x.is_finite()
                || !point.y.is_finite()
                || !(0.0..=1.0).contains(&point.x)
                || !(0.0..=1.0).contains(&point.y)
        }) {
            return Err("drawing coordinates must be finite and normalized".into());
        }
    }
    Ok(())
}

/// `SplitMix64` finalization uses all 64 seed bits before narrowing to PCG32 state.
fn benchmark_seed(seed: u64, index: u32, domain: u64) -> u32 {
    let mut value = seed
        .wrapping_add(u64::from(index).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(domain.wrapping_mul(0x517C_C1B7_2722_0A95));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let bytes = (value ^ (value >> 31)).to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn readback_buffers(
    p: &super::pipeline::GpuPipeline,
    sources: &[(&Buffer, u64, u64)],
) -> Result<Vec<u8>, String> {
    assert!(
        sources.len() <= 9,
        "snapshot has a bounded set of resources"
    );
    let total: u64 = sources.iter().map(|(_, _, size)| size).sum();
    if total == 0 || total > p.device.limits().max_buffer_size {
        return Err("snapshot staging exceeds device buffer capacity".into());
    }
    let staging = p.device.create_buffer(&BufferDescriptor {
        label: Some("benchmark_snapshot"),
        size: total,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("benchmark_snapshot"),
    });
    let mut offset = 0;
    for &(source, source_offset, size) in sources {
        encoder.copy_buffer_to_buffer(source, source_offset, &staging, offset, size);
        offset += size;
    }
    let submission = p.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    staging.slice(..).map_async(MapMode::Read, move |result| {
        // A timeout may have dropped the receiver; the staging buffer is then dropped.
        let _ = tx.send(result);
    });
    p.device
        .poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(READBACK_TIMEOUT),
        })
        .map_err(|error| format!("benchmark snapshot poll failed: {error}"))?;
    rx.recv_timeout(READBACK_TIMEOUT)
        .map_err(|error| format!("snapshot map callback: {error}"))?
        .map_err(|error| format!("snapshot mapping failed: {error}"))?;
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .map_err(|error| format!("snapshot mapped range failed: {error}"))?;
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(bytes)
}

fn decode_snapshot(
    bytes: &[u8],
    active: usize,
    offspring: usize,
    pixels: usize,
    counters: BenchmarkCounters,
    validation: bool,
) -> Result<BenchmarkSnapshot, String> {
    let mut data = SnapshotReader { bytes };
    let parents = data.states(active)?;
    let offspring_states = data.states(offspring)?;
    let parent_total_errors = data.words(active)?;
    let framebuffers = data.words(active * pixels)?;
    let parent_framebuffers = framebuffers
        .chunks_exact(pixels)
        .map(<[u32]>::to_vec)
        .collect();
    let offspring_errors = data.words(offspring * 2)?.as_chunks::<2>().0.to_vec();
    let chain_update_flags = data
        .words(active * 4)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let fitness_packed = data.words(active)?;
    let control_words = data.words(4)?;
    let maximum = data.words(1)?[0];
    assert!(
        data.bytes.is_empty(),
        "snapshot decoder must consume exactly its resources"
    );
    Ok(BenchmarkSnapshot {
        parents,
        offspring: offspring_states,
        parent_total_errors,
        parent_framebuffers,
        offspring_errors,
        chain_update_flags,
        fitness_packed,
        control: BenchmarkControl {
            new_best_found: control_words[0],
            best_chain_id: control_words[1],
            best_fitness_bits: control_words[2],
        },
        counters,
        validation_max_tile_occupancy: validation.then_some(maximum),
    })
}

struct SnapshotReader<'a> {
    bytes: &'a [u8],
}

impl SnapshotReader<'_> {
    fn take(&mut self, size: usize) -> Result<&[u8], String> {
        let (value, remaining) = self
            .bytes
            .split_at_checked(size)
            .ok_or_else(|| "truncated benchmark readback".to_string())?;
        self.bytes = remaining;
        Ok(value)
    }

    fn states(&mut self, count: usize) -> Result<Vec<BenchmarkPackedState>, String> {
        let bytes = self.take(count * GPU_DRAWING_STATE_SIZE)?;
        bytes
            .as_chunks::<GPU_DRAWING_STATE_SIZE>()
            .0
            .iter()
            .map(|bytes| {
                let state = bytemuck::pod_read_unaligned::<GpuDrawingState>(bytes);
                BenchmarkPackedState::from_gpu(&state)
            })
            .collect()
    }

    fn words(&mut self, count: usize) -> Result<Vec<u32>, String> {
        Ok(self
            .take(count * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{color::Color, point::Point, polygon::Polygon};

    #[test]
    fn normal_timestamp_pairs_retain_last_iteration_attribution() {
        let (pass, envelope) =
            decode_batch_timestamps(&[100, 110, 200, 230, 250, 290, 300, 375], 2.5, false);
        assert_eq!(pass.mutate_ns, 25.0);
        assert_eq!(pass.bin_polygons_ns, 75.0);
        assert_eq!(pass.rasterize_error_ns, 100.0);
        assert_eq!(pass.select_ns, 187.5);
        assert_eq!(pass.total_ns(), 387.5);
        assert_eq!(pass.sample_count, 1);
        assert_eq!(envelope, BenchmarkEnvelopeTimings::default());
    }

    #[test]
    fn envelope_timestamp_pair_is_never_attributed_to_selection() {
        // Query 6 predates the last iteration, so 6/7 is the whole batch.
        let (pass, envelope) =
            decode_batch_timestamps(&[100, 110, 200, 230, 250, 290, 10, 375], 2.5, true);
        assert_eq!(pass.mutate_ns, 25.0);
        assert_eq!(pass.bin_polygons_ns, 75.0);
        assert_eq!(pass.rasterize_error_ns, 100.0);
        assert_eq!(pass.select_ns, 0.0);
        assert_eq!(pass.total_ns(), 200.0);
        assert_eq!(pass.sample_count, 1);
        assert_eq!(envelope.compute_ns, 912.5);
        assert_eq!(envelope.sample_count, 1);
        assert_eq!(
            serde_json::to_value(envelope).unwrap(),
            serde_json::json!({ "compute_ns": 912.5, "sample_count": 1 }),
        );
    }

    #[test]
    fn timestamp_decoding_preserves_wraparound_and_zero_duration_samples() {
        let timestamps = [u64::MAX - 2, 3, 10, 20, 30, 50, u64::MAX - 20, 19];
        let (normal, _) = decode_batch_timestamps(&timestamps, 0.5, false);
        let (diagnostic, envelope) = decode_batch_timestamps(&timestamps, 0.5, true);
        assert_eq!(normal.mutate_ns, 3.0);
        assert_eq!(normal.bin_polygons_ns, 5.0);
        assert_eq!(normal.rasterize_error_ns, 10.0);
        assert_eq!(normal.select_ns, 20.0);
        assert_eq!(diagnostic.select_ns, 0.0);
        assert_eq!(envelope.compute_ns, 20.0);
        let (zero, zero_envelope) = decode_batch_timestamps(&[0; 8], 1.0, true);
        assert_eq!(zero.total_ns(), 0.0);
        assert_eq!(zero.sample_count, 1);
        assert_eq!(zero_envelope.compute_ns, 0.0);
        assert_eq!(zero_envelope.sample_count, 1);
    }

    #[test]
    fn batch_envelope_requires_a_bulk_iteration_only_when_enabled() {
        let mut params = MutationParams::default();
        for iterations in [0, 1] {
            params.gpu_batch_iters = iterations;
            assert!(validate_batch_envelope(&params, true).is_err());
            assert!(validate_batch_envelope(&params, false).is_ok());
        }
        for iterations in [2, 64, GPU_MAX_BATCH_ITERS] {
            params.gpu_batch_iters = iterations;
            assert!(validate_batch_envelope(&params, true).is_ok());
        }
    }

    #[test]
    fn seed_uses_high_bits_and_separates_parent_and_offspring() {
        let a: Vec<_> = (0..256).map(|i| benchmark_seed(7, i, 1)).collect();
        let b: Vec<_> = (0..256)
            .map(|i| benchmark_seed(7 + (1u64 << 32), i, 1))
            .collect();
        assert_ne!(a, b);
        let unique: std::collections::HashSet<_> = a.iter().copied().collect();
        assert_eq!(unique.len(), 256);
        assert_ne!(benchmark_seed(7, 0, 0), benchmark_seed(7, 0, 1));
        assert_eq!(a[17], benchmark_seed(7, 17, 1));
    }

    #[test]
    fn snapshot_omits_padding_but_retains_all_active_words() {
        let mut state = GpuDrawingState::zeroed();
        state.polygon_count = 1;
        state.polygons[0].data = [11, 12, 13, 14];
        state.rng_state[0] = 99;
        let before = BenchmarkPackedState::from_gpu(&state).unwrap();
        state.rng_state[3] = 71;
        state.polygons[1].data = [u32::MAX; 4];
        assert_eq!(before, BenchmarkPackedState::from_gpu(&state).unwrap());
        state.polygons[0].data[3] += 1;
        assert_ne!(before, BenchmarkPackedState::from_gpu(&state).unwrap());
        state.polygon_count = u32::try_from(MAX_POLYGONS_PER_IMAGE).unwrap() + 1;
        assert!(BenchmarkPackedState::from_gpu(&state).is_err());
    }

    #[test]
    fn drawing_validation_rejects_truncation_and_nonfinite_coordinates() {
        let mut drawing = Drawing {
            polygons: vec![Polygon {
                points: vec![Point { x: 0.5, y: 0.5 }; 4],
                color: Color {
                    r: 1,
                    g: 2,
                    b: 3,
                    a: 255,
                },
            }],
            is_dirty: false,
            fitness: 0.0,
        };
        assert!(validate_drawing(&drawing, 2).is_ok());
        assert!(validate_drawing(&drawing, 1).is_err());
        drawing.polygons[0].points[0].x = f32::NAN;
        assert!(validate_drawing(&drawing, 2).is_err());
    }

    #[test]
    fn capacity_checks_fail_closed_at_allocation_and_device_limits() {
        let mut params = MutationParams::default();
        assert!(validate_shape(&params).is_ok());
        params.lambda = 3;
        assert!(validate_shape(&params).is_err());
        params.lambda = GPU_MAX_LAMBDA;
        params.rasterize_wg = [0, 16];
        assert!(validate_shape(&params).is_err());
    }

    #[test]
    fn counters_distinguish_submission_from_completion_and_drain() {
        let mut counters = BenchmarkCounters::default();
        counters.record_submission(64, 16384);
        counters.record_submission(1, 256);
        counters.record_completion(64, 16384);
        assert_eq!(counters.submitted_iterations, 65);
        assert_eq!(counters.completed_iterations, 64);
        assert_eq!(counters.submitted_batches, 2);
        assert_eq!(counters.completed_batches, 1);
        counters.record_completion(1, 256);
        assert_eq!(
            counters.completed_evaluations,
            counters.submitted_evaluations
        );
        assert_eq!(counters.completed_iterations, counters.submitted_iterations);
    }

    #[test]
    fn snapshot_decoder_preserves_pixels_raw_errors_and_controls() {
        let mut bytes = vec![0; GPU_DRAWING_STATE_SIZE * 2];
        let words: [u32; 15] = [
            42,
            0xff03_0201,
            0xff06_0504,
            0,
            0,
            1,
            2,
            3,
            777,
            88,
            1,
            0,
            88,
            999,
            257,
        ];
        // One parent, one offspring, two pixels, stride-2 errors, stride-4 flags,
        // packed fitness, four control words and the persistent occupancy word.
        for word in &words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let snapshot =
            decode_snapshot(&bytes, 1, 1, 2, BenchmarkCounters::default(), true).unwrap();
        assert_eq!(snapshot.parent_total_errors, [42]);
        assert_eq!(
            snapshot.parent_framebuffers,
            [vec![0xff03_0201, 0xff06_0504]]
        );
        assert_eq!(snapshot.chain_update_flags, [[1, 2, 3]]);
        assert_eq!(snapshot.control.best_fitness_bits, 88);
        assert_eq!(snapshot.validation_max_tile_occupancy, Some(257));
        assert!(decode_snapshot(
            &bytes[..bytes.len() - 4],
            1,
            1,
            2,
            BenchmarkCounters::default(),
            true
        )
        .is_err());
    }
}
