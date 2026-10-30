//! Owns batch readback transport, deferred winning drawings and observations.
use std::sync::mpsc::{self, Receiver};

use wgpu::{
    BufferAsyncError, BufferSlice, CommandEncoder, CommandEncoderDescriptor, MapMode, PollType,
    SubmissionIndex,
};

use super::benchmark::{self, BenchmarkCounters, BenchmarkEnvelopeTimings};
use super::buffers::{gpu_to_drawing, ControlFlags, GpuDrawingState, GPU_DRAWING_STATE_SIZE};
use super::operators::{OperatorTally, OPERATOR_STATS_WORDS};
use super::pipeline::OPERATOR_STATS_STAGING_OFFSET;
use super::{GpuPipeline, PassTimings};
use crate::models::drawing::Drawing;

type MapReceiver = Receiver<Result<(), BufferAsyncError>>;

/// Describes the copies already submitted for one staging slot.
pub(super) struct SubmittedBatch {
    pub staging_idx: usize,
    pub collect_timestamps: bool,
    pub batch_envelope: bool,
    pub active_chain_count: u32,
    pub submission_index: SubmissionIndex,
    pub iterations: u32,
    pub evaluations: u64,
}

/// Metadata and every map completion belong to the same pending operation.
struct PendingBatch {
    batch: SubmittedBatch,
    control_rx: MapReceiver,
    timestamp_rx: Option<MapReceiver>,
    fitness_rx: MapReceiver,
    drawing_rx: Option<MapReceiver>,
}

impl PendingBatch {
    fn map(p: &GpuPipeline, batch: SubmittedBatch, copied_drawing: bool) -> Self {
        let idx = batch.staging_idx;
        let control_rx = map_read(p.control_staging_bufs[idx].slice(..));
        let timestamp_rx = batch
            .collect_timestamps
            .then(|| map_read(p.timestamp_staging_bufs[idx].slice(..)));
        let fitness_rx = map_read(
            p.fitness_staging_bufs[idx].slice(..(u64::from(batch.active_chain_count) * 4)),
        );
        let drawing_rx = copied_drawing.then(|| map_read(p.readback_staging_bufs[idx].slice(..)));
        Self {
            batch,
            control_rx,
            timestamp_rx,
            fitness_rx,
            drawing_rx,
        }
    }

    fn wait(&self, p: &GpuPipeline) {
        // Wait for this submission only, even when the next batch is running.
        p.device
            .poll(PollType::Wait {
                submission_index: Some(self.batch.submission_index.clone()),
                timeout: None,
            })
            .unwrap();
        self.control_rx
            .recv()
            .unwrap()
            .expect("Failed to map control staging buffer");
        if let Some(rx) = &self.timestamp_rx {
            rx.recv()
                .unwrap()
                .expect("Failed to map timestamp staging buffer");
        }
        self.fitness_rx
            .recv()
            .unwrap()
            .expect("Failed to map fitness staging buffer");
    }
}

pub(super) struct Readback {
    pending: Option<PendingBatch>,
    staging_idx: usize,
    pending_best_chain: Option<u32>,
    pub best_fitness_bits: u32,
    pub pass_timings: PassTimings,
    pub benchmark_envelope_timings: BenchmarkEnvelopeTimings,
    pub benchmark_counters: BenchmarkCounters,
    pub chain_fitness: Vec<f32>,
    /// Folded from the control staging of every collected batch.
    pub operator_tally: OperatorTally,
}

impl Readback {
    pub fn new(chain_count: u32) -> Self {
        Self {
            pending: None,
            staging_idx: 0,
            pending_best_chain: None,
            best_fitness_bits: 0,
            pass_timings: PassTimings::default(),
            benchmark_envelope_timings: BenchmarkEnvelopeTimings::default(),
            benchmark_counters: BenchmarkCounters::default(),
            chain_fitness: vec![0.0; chain_count as usize],
            operator_tally: OperatorTally::default(),
        }
    }

    pub fn next_staging_index(&mut self) -> usize {
        let idx = self.staging_idx;
        debug_assert!(self
            .pending
            .as_ref()
            .is_none_or(|p| p.batch.staging_idx != idx));
        self.staging_idx = 1 - idx;
        idx
    }

    /// Select the drawing source at the original copy position, before collecting
    /// the previous batch can replace the queued winning chain. The copy targets
    /// this batch's staging slot, so a drawing map pending in the other slot
    /// never forces an early collection.
    pub fn encode_best_copy(
        &mut self,
        p: &GpuPipeline,
        encoder: &mut CommandEncoder,
        staging_idx: usize,
    ) -> bool {
        let Some(chain_id) = self.pending_best_chain.take() else {
            return false;
        };
        encoder.copy_buffer_to_buffer(
            &p.chain_states_buf,
            u64::from(chain_id) * GPU_DRAWING_STATE_SIZE as u64,
            &p.readback_staging_bufs[staging_idx],
            0,
            GPU_DRAWING_STATE_SIZE as u64,
        );
        true
    }

    /// Submission precedes collection; new maps register only after the previous
    /// operation has completed and unmapped its buffers.
    pub fn after_submit(
        &mut self,
        p: &GpuPipeline,
        batch: SubmittedBatch,
        copied_drawing: bool,
    ) -> Option<Drawing> {
        self.benchmark_counters
            .record_submission(batch.iterations, batch.evaluations);
        let result = self.collect(p);
        self.pending = Some(PendingBatch::map(p, batch, copied_drawing));
        result
    }

    fn collect(&mut self, p: &GpuPipeline) -> Option<Drawing> {
        let pending = self.pending.take()?;
        let batch = &pending.batch;
        let idx = batch.staging_idx;
        pending.wait(p);
        self.benchmark_counters
            .record_completion(batch.iterations, batch.evaluations);

        let drawing = pending
            .drawing_rx
            .as_ref()
            .map(|rx| mapped_drawing(p, idx, rx, "Failed to access mapped best drawing buffer"));

        let control_data = p.control_staging_bufs[idx]
            .slice(..)
            .get_mapped_range()
            .expect("Failed to access mapped control flags buffer");
        let (flags, stats) = control_data.split_at(OPERATOR_STATS_STAGING_OFFSET);
        let flags: ControlFlags = *bytemuck::from_bytes(flags);
        let stats: &[u32; OPERATOR_STATS_WORDS] = bytemuck::cast_slice(stats)
            .try_into()
            .expect("control staging holds the operator stats");
        self.operator_tally.fold(stats);
        drop(control_data);

        if pending.timestamp_rx.is_some() {
            self.collect_timestamps(p, idx, batch.batch_envelope);
        }

        let fitness_data = p.fitness_staging_bufs[idx]
            .slice(..(u64::from(batch.active_chain_count) * 4))
            .get_mapped_range()
            .expect("Failed to access mapped chain fitness buffer");
        let packed: &[u32] = bytemuck::cast_slice(&fitness_data);
        for (i, &bits) in packed.iter().enumerate() {
            self.chain_fitness[i] = f32::from_bits(bits);
        }
        let best_chain = best_chain(&self.chain_fitness[..packed.len()]);
        drop(fitness_data);
        p.control_staging_bufs[idx].unmap();
        p.fitness_staging_bufs[idx].unmap();

        if flags.new_best_found != 0 {
            self.best_fitness_bits = flags.best_fitness_bits;
            // `flags.best_chain_id` is the last of racing stores from select's
            // atomicMax, so it varies between runs. The per-chain fitness gives
            // the same winner every run. Chain fitness never decreases, so the
            // copy made one batch later is at least this good.
            // A newer winning chain supersedes a queued copy not yet encoded.
            self.pending_best_chain = best_chain;
        }
        drawing
    }

    fn collect_timestamps(&mut self, p: &GpuPipeline, idx: usize, batch_envelope: bool) {
        let data = p.timestamp_staging_bufs[idx]
            .slice(..)
            .get_mapped_range()
            .expect("Failed to access mapped timestamp buffer");
        let timestamps: &[u64; 8] = bytemuck::cast_slice(&data)
            .try_into()
            .expect("timestamp staging buffer must contain eight queries");
        let period = f64::from(p.timestamp_period);
        let (sample, envelope) =
            benchmark::decode_batch_timestamps(timestamps, period, batch_envelope);
        self.pass_timings.mutate_ns += sample.mutate_ns;
        self.pass_timings.bin_polygons_ns += sample.bin_polygons_ns;
        self.pass_timings.rasterize_error_ns += sample.rasterize_error_ns;
        self.pass_timings.select_ns += sample.select_ns;
        self.pass_timings.sample_count += 1;
        self.benchmark_envelope_timings.compute_ns += envelope.compute_ns;
        self.benchmark_envelope_timings.sample_count += envelope.sample_count;
        if self.pass_timings.sample_count <= 3 {
            eprintln!(
                "[timestamps] sample #{}: mutate={:.2}ns bin={:.2}ns rast={:.2}ns sel={:.2}ns period={:.4}",
                self.pass_timings.sample_count,
                sample.mutate_ns, sample.bin_polygons_ns,
                sample.rasterize_error_ns, sample.select_ns, period
            );
            if batch_envelope {
                eprintln!(
                    "[timestamps] diagnostic batch compute={:.2}ns; select unavailable",
                    envelope.compute_ns
                );
            }
        }
        drop(data);
        p.timestamp_staging_bufs[idx].unmap();
    }

    pub fn flush(&mut self, p: &GpuPipeline) -> Option<Drawing> {
        let batch_result = self.collect(p);
        // With no next submission, resolve the newest queued chain synchronously.
        let sync_result = self
            .pending_best_chain
            .take()
            .map(|chain_id| readback_chain(p, chain_id));
        sync_result.or(batch_result)
    }

    /// Forget the totals along with the GPU counters they mirror.
    pub fn reset_operator_tally(&mut self) {
        assert!(self.pending.is_none(), "reset requires drained batch readback");
        self.operator_tally = OperatorTally::default();
    }

    /// Seeded preparation resets slot order; ordinary reinitialization preserves it.
    pub fn reset_staging(&mut self) {
        assert!(
            self.pending.is_none(),
            "reset requires drained batch readback"
        );
        self.staging_idx = 0;
        self.pending_best_chain = None;
    }
}

/// Highest fitness wins; ties go to the lowest chain id.
fn best_chain(chain_fitness: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &fitness) in chain_fitness.iter().enumerate() {
        if best.is_none_or(|(_, b)| fitness.total_cmp(&b).is_gt()) {
            best = Some((i, fitness));
        }
    }
    best.map(|(i, _)| u32::try_from(i).expect("chain count fits u32"))
}

fn map_read(slice: BufferSlice<'_>) -> MapReceiver {
    let (tx, rx) = mpsc::channel();
    slice.map_async(MapMode::Read, move |result| tx.send(result).unwrap());
    rx
}

fn mapped_drawing(
    p: &GpuPipeline,
    idx: usize,
    rx: &MapReceiver,
    access_error: &'static str,
) -> Drawing {
    rx.recv()
        .unwrap()
        .expect("Failed to map readback staging buffer");
    let data = p.readback_staging_bufs[idx]
        .slice(..)
        .get_mapped_range()
        .expect(access_error);
    let state: &GpuDrawingState = bytemuck::from_bytes(&data);
    let drawing = gpu_to_drawing(state);
    drop(data);
    p.readback_staging_bufs[idx].unmap();
    drawing
}

/// Uses slot 0; callers resolve pending batch readback first (as `flush` does).
pub(super) fn readback_chain(p: &GpuPipeline, chain_id: u32) -> Drawing {
    let mut encoder = p.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(
        &p.chain_states_buf,
        u64::from(chain_id) * GPU_DRAWING_STATE_SIZE as u64,
        &p.readback_staging_bufs[0],
        0,
        GPU_DRAWING_STATE_SIZE as u64,
    );
    let submission = p.queue.submit(std::iter::once(encoder.finish()));
    let rx = map_read(p.readback_staging_bufs[0].slice(..));
    p.device
        .poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    mapped_drawing(p, 0, &rx, "Failed to access mapped drawing readback buffer")
}

#[cfg(test)]
mod tests {
    use super::best_chain;

    #[test]
    fn best_chain_prefers_highest_fitness_then_lowest_id() {
        assert_eq!(best_chain(&[]), None);
        assert_eq!(best_chain(&[1.0, 3.0, 2.0]), Some(1));
        assert_eq!(best_chain(&[2.0, 3.0, 3.0, 1.0]), Some(1));
        assert_eq!(best_chain(&[-1.0, -0.5]), Some(1));
    }
}
