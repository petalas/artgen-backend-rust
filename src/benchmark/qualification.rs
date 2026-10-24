//! Shared trial accounting and replay identity, independent of numerical oracles.
//!
//! Callers own preparation, device error scopes and numerical verification. Only
//! ordinary batches, their production readbacks and the final drain are timed.
use std::time::Instant;

use anyhow::{ensure, Result};
use serde::Serialize;

use super::fixtures;
use crate::{
    gpu_evolver::{
        benchmark::{BenchmarkCounters, BenchmarkPackedState, BenchmarkSnapshot},
        GpuEvolver,
    },
    mutation_params::MutationParams,
};

/// Record meaning is caller context, not an additional persisted protocol field.
/// A qualification can have `diagnostic: false` and still have no timing score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordPurpose {
    Timing,
    Qualification,
    Diagnostic,
    Profile,
}

impl RecordPurpose {
    /// # Errors
    /// Scored timing requires a finite, positive elapsed clock.
    pub fn rate(self, completed_evaluations: u64, elapsed_seconds: f64) -> Result<Option<f64>> {
        if self != Self::Timing {
            return Ok(None);
        }
        ensure!(
            elapsed_seconds.is_finite() && elapsed_seconds > 0.0,
            "invalid elapsed time"
        );
        Ok(Some(completed_evaluations as f64 / elapsed_seconds))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub elapsed_seconds: f64,
    pub drain_seconds: f64,
    pub counters: BenchmarkCounters,
}

/// Measure the ordinary engine path. Snapshots and qualification stay outside.
///
/// # Errors
/// Propagates drain failures and rejects an invalid elapsed clock. Callers must
/// prepare/reset the evolver first and enforce their own batch/parameter limits.
pub fn measure(
    evolver: &mut GpuEvolver,
    params: &MutationParams,
    batches: u32,
) -> Result<Measurement> {
    let start = Instant::now();
    for _ in 0..batches {
        // Returned drawings and required engine allocations belong to the clock.
        let _best = evolver.run_batch(params, true);
    }
    let drain = Instant::now();
    let counters = evolver.benchmark_drain().map_err(anyhow::Error::msg)?;
    let drain_seconds = drain.elapsed().as_secs_f64();
    let elapsed_seconds = start.elapsed().as_secs_f64();
    ensure!(
        elapsed_seconds.is_finite() && elapsed_seconds > 0.0,
        "invalid elapsed time"
    );
    Ok(Measurement {
        elapsed_seconds,
        drain_seconds,
        counters,
    })
}

/// Both submitted and drained work must match the requested logical work.
/// Parameters come from the caller's validated fixture and batch budget.
#[must_use]
pub fn work_is_complete(c: BenchmarkCounters, params: &MutationParams, batches: u32) -> bool {
    let iterations = u64::from(batches) * u64::from(params.gpu_batch_iters);
    let evaluations = iterations * u64::from(params.chain_count) * u64::from(params.lambda);
    c.submitted_batches == u64::from(batches)
        && c.completed_batches == u64::from(batches)
        && c.submitted_iterations == iterations
        && c.completed_iterations == iterations
        && c.submitted_evaluations == evaluations
        && c.completed_evaluations == evaluations
}

// Tuple serialization preserves the seven-field protocol byte order. Decoding
// strips padding; reporting/control races, counters and occupancy are excluded.
#[derive(Serialize)]
struct CanonicalState<'a>(
    &'a [BenchmarkPackedState],
    &'a [BenchmarkPackedState],
    &'a [u32],
    &'a [Vec<u32>],
    &'a [[u32; 2]],
    &'a [[u32; 3]],
    &'a [u32],
);

impl<'a> From<&'a BenchmarkSnapshot> for CanonicalState<'a> {
    fn from(state: &'a BenchmarkSnapshot) -> Self {
        Self(
            &state.parents,
            &state.offspring,
            &state.parent_total_errors,
            &state.parent_framebuffers,
            &state.offspring_errors,
            &state.chain_update_flags,
            &state.fitness_packed,
        )
    }
}

/// # Errors
/// Returns canonical state serialization errors.
pub fn state_hash(state: &BenchmarkSnapshot) -> Result<String> {
    Ok(fixtures::sha256(&serde_json::to_vec(
        &CanonicalState::from(state),
    )?))
}

/// Report each differing evolution component without comparing telemetry.
pub fn compare_state(
    actual: &BenchmarkSnapshot,
    expected: &BenchmarkSnapshot,
    phase: &str,
    errors: &mut Vec<String>,
) {
    let actual = CanonicalState::from(actual);
    let expected = CanonicalState::from(expected);
    let fields = [
        ("parents", actual.0 == expected.0),
        ("offspring", actual.1 == expected.1),
        ("parent_total_errors", actual.2 == expected.2),
        ("parent_framebuffers", actual.3 == expected.3),
        ("offspring_errors", actual.4 == expected.4),
        ("chain_update_flags", actual.5 == expected.5),
        ("fitness_packed", actual.6 == expected.6),
    ];
    for (name, equal) in fields {
        if !equal {
            errors.push(format!("{phase} {name} differs from qualified reference"));
        }
    }
}
