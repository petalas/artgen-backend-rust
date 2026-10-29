//! How the evolution stages are recorded into command encoders.
//!
//! A search iteration runs mutate → bin → raster → select →
//! `update_framebuffers`. A full evaluation runs bin → raster only, over parent
//! copies the host has prepared. This module records commands only: it never
//! writes the queue or submits. Callers upload params, prepare buffers, append
//! readback copies and submit.
//!
//! Every stage leaves the params bind group at slot 1. wgpu's resource tracker
//! inserts barriers between dispatches inside one compute pass only when a
//! buffer's usage changes (`flush_states` in wgpu-core `command/compute.rs`).
//! RW→RW is no change, so pass boundaries carry the error-sum barriers:
//! raster's atomicAdd before select's atomicExchange, and select's zeroing
//! before the next iteration's raster. A search iteration therefore records two
//! passes, mutate+bin+raster and select+update. The timestamped final iteration
//! records four, one per sampled stage.
//!
//! A batch is split here into up to `FINISH_THREADS` command encoders,
//! finished in parallel by [`BatchCommands::finish`]. The chunk boundaries fall
//! between iterations, so the split belongs with iteration recording; the
//! caller still makes the one submit, so the submit policy stays visible there.

use wgpu::{
    CommandBuffer, CommandEncoder, CommandEncoderDescriptor, ComputePass, ComputePassDescriptor,
    ComputePassTimestampWrites,
};

use crate::settings::BIN_ROW_BANDS;

use super::execution::{RasterGrid, Workload};
use super::pipeline::GpuPipeline;

/// Command buffers one batch is split into; each but the last is finished on
/// its own scoped thread.
const FINISH_THREADS: u32 = 4;

/// Timestamp queries a timed batch writes, resolved into `timestamp_resolve_buf`.
/// Pairs (begin, end) of the final iteration: 0/1 mutate, 2/3 bin, 4/5 raster,
/// 6/7 select+update. With the batch envelope, 6 instead marks the start of the
/// batch's first pass and 7 still marks the end of the last.
const TIMESTAMP_QUERIES: u32 = 8;

/// Which timestamp queries a batch writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BatchTimestamps {
    Off,
    FinalIteration,
    /// The final iteration's stages, except select, plus the whole batch.
    FinalIterationAndEnvelope,
}

impl BatchTimestamps {
    /// The envelope applies only to a timestamped batch.
    pub fn new(collect: bool, envelope: bool) -> Self {
        match (collect, envelope) {
            (false, _) => Self::Off,
            (true, false) => Self::FinalIteration,
            (true, true) => Self::FinalIterationAndEnvelope,
        }
    }

    pub fn collected(self) -> bool {
        self != Self::Off
    }

    pub fn envelope(self) -> bool {
        self == Self::FinalIterationAndEnvelope
    }
}

/// A recorded batch: its command encoders in iteration order.
pub(super) struct BatchCommands {
    earlier: Vec<CommandEncoder>,
    last: CommandEncoder,
}

impl BatchCommands {
    /// The encoder that ends the batch; copies recorded here follow every iteration.
    pub fn last(&mut self) -> &mut CommandEncoder {
        &mut self.last
    }

    /// Finish every encoder, the earlier ones on scoped threads, preserving
    /// order. Submit the result in one call.
    ///
    /// `finish` does the driver-side encoding (~11 ms per batch on Dozen), so
    /// one thread made the host the limiter. Submitting the parts separately
    /// would let the GPU run one while the host still encodes the next; on
    /// Dozen/NVIDIA that overlap makes `finish` about 3x slower and the host
    /// never gets ahead again (LEARNINGS, "Chunked submits make `finish` 3x
    /// slower").
    pub fn finish(self) -> Vec<CommandBuffer> {
        std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .earlier
                .into_iter()
                .map(|encoder| scope.spawn(move || encoder.finish()))
                .collect();
            let last = self.last.finish();
            let mut buffers: Vec<CommandBuffer> = handles
                .into_iter()
                .map(|handle| handle.join().expect("command encoder finish panicked"))
                .collect();
            buffers.push(last);
            buffers
        })
    }
}

/// Record `work.iterations` search iterations. A timed batch records its final
/// iteration with per-stage timestamps and resolves the queries.
pub(super) fn record_batch(
    p: &GpuPipeline,
    work: &Workload,
    timestamps: BatchTimestamps,
) -> BatchCommands {
    assert!(!timestamps.envelope() || work.iterations >= 2,
        "batch compute envelope requires at least two iterations");
    let untimed = if timestamps.collected() { work.iterations.saturating_sub(1) } else { work.iterations };
    let mut commands = BatchCommands {
        last: batch_encoder(p),
        earlier: Vec::with_capacity(FINISH_THREADS as usize),
    };
    // Consumed by the first pass without a per-iteration branch.
    let mut batch_start = timestamps.envelope().then(|| pass_timestamps(p, Some(6), None));
    let chunk_iterations = untimed.div_ceil(FINISH_THREADS).max(1);
    for index in 0..untimed {
        if index > 0 && index % chunk_iterations == 0 {
            let next = batch_encoder(p);
            commands.earlier.push(std::mem::replace(&mut commands.last, next));
        }
        record_search_iteration(p, &mut commands.last, work, batch_start.take());
    }
    if timestamps.collected() {
        record_timed_iteration(p, &mut commands.last, work, timestamps.envelope());
        commands.last.resolve_query_set(
            &p.timestamp_query_set, 0..TIMESTAMP_QUERIES, &p.timestamp_resolve_buf, 0);
    }
    commands
}

/// Record a full evaluation of working states `0..chains`: bin and raster only,
/// in one pass. mutate and `update_framebuffers` do not run, so the host must
/// first have done their part: copied the parents into those states, marked
/// them with `GpuPipeline::mark_full_evaluation`, and emptied the raster
/// worklist (and empty it again after submitting, for the next batch).
pub(super) fn record_full_evaluation(p: &GpuPipeline, encoder: &mut CommandEncoder, chains: u32) {
    let grid = p.raster_grid(p.rasterize_wg());
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("evaluate_fitness"),
        timestamp_writes: None,
    });
    record_bin(&mut pass, p, chains);
    record_raster(&mut pass, p, grid, chains);
}

/// Record the full rasterization of `chains` parents into their framebuffers
/// and total errors.
pub(super) fn record_framebuffer_init(p: &GpuPipeline, encoder: &mut CommandEncoder, chains: u32) {
    let grid = p.raster_grid(p.rasterize_wg());
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("init_framebuffers"),
        timestamp_writes: None,
    });
    pass.set_pipeline(p.init_framebuffers_pipeline());
    pass.set_bind_group(0, &p.init_framebuffers_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(grid.x, grid.y, chains);
}

fn batch_encoder(p: &GpuPipeline) -> CommandEncoder {
    p.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("gpu_evolver_batch") })
}

fn pass_timestamps(
    p: &GpuPipeline,
    begin: Option<u32>,
    end: Option<u32>,
) -> ComputePassTimestampWrites<'_> {
    ComputePassTimestampWrites {
        query_set: &p.timestamp_query_set,
        beginning_of_pass_write_index: begin,
        end_of_pass_write_index: end,
    }
}

fn begin_pass<'e>(
    encoder: &'e mut CommandEncoder,
    label: &str,
    timestamp_writes: Option<ComputePassTimestampWrites<'_>>,
) -> ComputePass<'e> {
    encoder.begin_compute_pass(&ComputePassDescriptor { label: Some(label), timestamp_writes })
}

/// Two passes; `batch_start` timestamps the beginning of the first.
fn record_search_iteration(
    p: &GpuPipeline,
    encoder: &mut CommandEncoder,
    work: &Workload,
    batch_start: Option<ComputePassTimestampWrites<'_>>,
) {
    {
        let mut pass = begin_pass(encoder, "mutate_bin_rasterize", batch_start);
        record_mutate(&mut pass, p, work);
        record_bin(&mut pass, p, work.offspring);
        record_raster(&mut pass, p, work.raster, work.offspring);
    }
    let mut pass = begin_pass(encoder, "select_update", None);
    record_select_update(&mut pass, p, work);
}

/// Four passes, each with its query pair; under the envelope, select's
/// beginning query stays the batch start.
fn record_timed_iteration(p: &GpuPipeline, encoder: &mut CommandEncoder, work: &Workload, envelope: bool) {
    {
        let mut pass = begin_pass(encoder, "mutate", Some(pass_timestamps(p, Some(0), Some(1))));
        record_mutate(&mut pass, p, work);
    }
    {
        let mut pass = begin_pass(encoder, "bin_polygons", Some(pass_timestamps(p, Some(2), Some(3))));
        record_bin(&mut pass, p, work.offspring);
    }
    {
        let mut pass = begin_pass(encoder, "rasterize_error", Some(pass_timestamps(p, Some(4), Some(5))));
        record_raster(&mut pass, p, work.raster, work.offspring);
    }
    let select_begin = if envelope { None } else { Some(6) };
    let mut pass = begin_pass(encoder, "select", Some(pass_timestamps(p, select_begin, Some(7))));
    record_select_update(&mut pass, p, work);
}

fn record_mutate(pass: &mut ComputePass<'_>, p: &GpuPipeline, work: &Workload) {
    pass.set_pipeline(p.mutate_pipeline());
    pass.set_bind_group(0, &p.mutate_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(work.active, work.mutate_y, 1);
}

/// One workgroup per offspring and band of tile rows.
fn record_bin(pass: &mut ComputePass<'_>, p: &GpuPipeline, offspring: u32) {
    pass.set_pipeline(p.bin_polygons_pipeline());
    pass.set_bind_group(0, &p.bin_polygons_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(offspring, BIN_ROW_BANDS, 1);
}

fn record_raster(pass: &mut ComputePass<'_>, p: &GpuPipeline, grid: RasterGrid, offspring: u32) {
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    p.record_raster(pass, [grid.x, grid.y], offspring);
}

/// select, then `update_framebuffers` in the same pass.
fn record_select_update(pass: &mut ComputePass<'_>, p: &GpuPipeline, work: &Workload) {
    pass.set_pipeline(p.select_pipeline());
    pass.set_bind_group(0, &p.select_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(work.active, 1, 1);

    pass.set_pipeline(p.update_framebuffers_pipeline());
    pass.set_bind_group(0, &p.update_framebuffers_bind_group, &[]);
    pass.set_bind_group(1, &p.params_bind_group, &[]);
    pass.dispatch_workgroups(work.raster.x, work.raster.y, work.active);
}
