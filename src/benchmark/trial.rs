//! One owner for preparation, completed work, readback and numerical checks.
//!
//! This slice is a `mature_drawing_restart`, not a continuation checkpoint.
//! Snapshot, oracle and JSON allocations stay outside measurement; allocations
//! and best-drawing readbacks required by the production batch path stay inside.
use std::{fs, path::Path, time::Instant};

use anyhow::{ensure, Result};
use bytemuck::Zeroable;
use serde::Serialize;

use super::{
    fixtures::{self, LoadedFixture, PROTOCOL_VERSION},
    reference::{self, Complexity},
    uncertainty::{self, PixelAudit},
};
use crate::{
    gpu_evolver::{
        benchmark::{
            BenchmarkCounters, BenchmarkPackedState, BenchmarkProvenance, BenchmarkSnapshot,
        },
        buffers::{gpu_to_drawing, GpuDrawingState, GpuPolygon},
        GpuEvolver,
    },
    models::drawing::Drawing,
    mutation_params::MutationParams,
    settings::{PER_POINT_MULTIPLIER, TILE_MAX_POLYS},
};

#[derive(Clone, Debug)]
pub struct TrialSpec {
    pub case_id: String,
    pub seed: u64,
    pub batches: u32,
    pub warmup_batches: u32,
    pub diagnostic: bool,
}

#[derive(Serialize)]
pub struct ChainQuality {
    pub chain: usize,
    pub triangles: u32,
    pub stored_error: u32,
    pub framebuffer_error: u64,
    pub scalar_error: u64,
    pub stored_fitness: f32,
    pub expected_stored_fitness: f32,
    pub scalar_fitness: f32,
    pub differing_pixels: u32,
    pub max_channel_delta: u32,
    pub total_channel_delta: u64,
    /// Larger nominal differences need a separate arithmetic/coverage envelope.
    pub pixel_audit: PixelAudit,
}

impl ChainQuality {
    fn passes_numerical_policy(&self) -> bool {
        self.framebuffer_error == u64::from(self.stored_error)
            && self.pixel_audit.unexplained_pixels == 0
            && u64::from(self.stored_error).abs_diff(self.scalar_error) <= self.total_channel_delta
            && self.stored_fitness.is_finite()
            && (self.stored_fitness - self.expected_stored_fitness).abs() <= 0.00002
    }
}

#[derive(Serialize)]
pub struct TrialRecord {
    pub protocol_version: u32,
    pub status: String,
    pub error_count: usize,
    pub errors: Vec<String>,
    pub case_id: String,
    pub seed: u64,
    pub batches: u32,
    pub diagnostic: bool,
    pub elapsed_seconds: f64,
    pub drain_seconds: f64,
    pub setup_seconds: f64,
    pub evals_per_second: Option<f64>,
    pub counters: BenchmarkCounters,
    pub initial_state_sha256: String,
    pub final_state_sha256: String,
    pub initial_similarity_percent: f32,
    pub final_fitness_percent: Vec<f32>,
    pub fixture_manifest_sha256: String,
    pub initial_complexity: Complexity,
    pub final_complexity: Vec<Complexity>,
    pub validation_max_tile_occupancy: Option<u32>,
    pub correctness: bool,
    pub initial_quality: Vec<ChainQuality>,
    pub final_quality: Vec<ChainQuality>,
    pub gpu: BenchmarkProvenance,
    pub params: MutationParams,
    pub timestamp_samples: u64,
    pub sampled_gpu_stage_totals_ns: [f64; 4],
    pub start_mode: String,
}

impl TrialRecord {
    fn prepared(
        input: &LoadedFixture,
        spec: &TrialSpec,
        initial: &BenchmarkSnapshot,
        initial_quality: Vec<ChainQuality>,
        initial_fitness: f32,
        gpu: BenchmarkProvenance,
    ) -> Result<Self> {
        Ok(Self {
            protocol_version: PROTOCOL_VERSION,
            status: "pending".into(),
            error_count: 0,
            errors: Vec::new(),
            case_id: spec.case_id.clone(),
            seed: spec.seed,
            batches: spec.batches,
            diagnostic: spec.diagnostic,
            elapsed_seconds: 0.0,
            drain_seconds: 0.0,
            setup_seconds: 0.0,
            evals_per_second: None,
            counters: BenchmarkCounters::default(),
            initial_state_sha256: state_hash(initial)?,
            final_state_sha256: String::new(),
            initial_similarity_percent: initial_fitness,
            final_fitness_percent: Vec::new(),
            fixture_manifest_sha256: input.manifest_sha256.clone(),
            initial_complexity: input.fixture.complexity.clone(),
            final_complexity: Vec::new(),
            validation_max_tile_occupancy: None,
            correctness: false,
            initial_quality,
            final_quality: Vec::new(),
            gpu,
            params: input.manifest.params.clone(),
            timestamp_samples: 0,
            sampled_gpu_stage_totals_ns: [0.0; 4],
            start_mode: input.manifest.start_mode.clone(),
        })
    }
}

/// # Errors
/// Rejects invalid inputs, failed GPU preparation/readback and unwritable outputs.
pub fn run(manifest: &Path, spec: &TrialSpec, output: &Path) -> Result<TrialRecord> {
    ensure!(
        (1..=4096).contains(&spec.batches),
        "batches must be 1..4096"
    );
    ensure!(spec.warmup_batches <= 16, "warmup must be 0..16");
    ensure!(
        !output.join("result.json").exists(),
        "result already exists"
    );
    fs::create_dir_all(output)?;
    let input = fixtures::load(manifest, &spec.case_id)?;
    ensure!(
        input.fixture.complexity.overflowed_tiles == 0,
        "initial tile-bin overflow"
    );
    let setup = Instant::now();
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &input.rgba,
        input.fixture.width,
        input.fixture.height,
        &input.drawing,
    ));
    let scopes = evolver.benchmark_error_scopes();
    let result = execute(&mut evolver, &input, spec, output, setup);
    let device_errors = futures_lite::future::block_on(scopes.finish());
    let mut record = result?;
    record.errors.extend(device_errors);
    finish_record(&mut record);
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(record)
}

fn execute(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    spec: &TrialSpec,
    output: &Path,
    setup: Instant,
) -> Result<TrialRecord> {
    let mut record = prepare(evolver, input, spec, output)?;
    record.setup_seconds = setup.elapsed().as_secs_f64();
    let params = &input.manifest.params;
    (
        record.elapsed_seconds,
        record.drain_seconds,
        record.counters,
    ) = measure(evolver, params, spec.batches)?;
    record.evals_per_second =
        Some(record.counters.completed_evaluations as f64 / record.elapsed_seconds);
    let final_state = evolver
        .benchmark_snapshot(params.chain_count, params.lambda)
        .map_err(anyhow::Error::msg)?;
    complete_record(&mut record, &final_state, input)?;
    save_state(output, &final_state)?;
    let timings = evolver.pass_timings();
    record.timestamp_samples = timings.sample_count;
    record.sampled_gpu_stage_totals_ns = [
        timings.mutate_ns,
        timings.bin_polygons_ns,
        timings.rasterize_error_ns,
        timings.select_ns,
    ];
    Ok(record)
}

fn prepare(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    spec: &TrialSpec,
    output: &Path,
) -> Result<TrialRecord> {
    let params = &input.manifest.params;
    evolver
        .benchmark_set_validation(spec.diagnostic)
        .map_err(anyhow::Error::msg)?;
    let initial_fitness = evolver
        .prepare_seeded_benchmark(&input.drawing, params, spec.seed, spec.warmup_batches, true)
        .map_err(anyhow::Error::msg)?;
    let initial = evolver
        .benchmark_snapshot(params.chain_count, params.lambda)
        .map_err(anyhow::Error::msg)?;
    // Keep boundary pixels and control words even when qualification aborts.
    fs::write(
        output.join("initial-state.json"),
        serde_json::to_vec(&initial)?,
    )?;
    let mut errors = Vec::new();
    let initial_quality = verify_quality(&initial, input, &mut errors)?;
    ensure!(
        errors.is_empty(),
        "initial numerical qualification failed: {errors:?}"
    );
    ensure!(
        initial
            .parents
            .iter()
            .all(|p| f32::from_bits(p.fitness_bits) > 97.0),
        "every initial parent must exceed 97% at the actual benchmark resolution"
    );
    TrialRecord::prepared(
        input,
        spec,
        &initial,
        initial_quality,
        initial_fitness,
        evolver.benchmark_provenance(),
    )
}

fn complete_record(
    record: &mut TrialRecord,
    final_state: &BenchmarkSnapshot,
    input: &LoadedFixture,
) -> Result<()> {
    let params = &input.manifest.params;
    record.final_quality = verify_quality(final_state, input, &mut record.errors)?;
    check_work(record.counters, params, record.batches, &mut record.errors);
    if let Some(maximum) = final_state.validation_max_tile_occupancy {
        if maximum > TILE_MAX_POLYS {
            record
                .errors
                .push(format!("tile-bin overflow: {maximum} > {TILE_MAX_POLYS}"));
        }
    }
    record.final_complexity = final_state
        .parents
        .iter()
        .map(|p| {
            reference::complexity(
                &p.polygons,
                input.fixture.width,
                input.fixture.height,
                params.rasterize_wg,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        record.final_quality.len() == record.initial_quality.len(),
        "active parent count changed"
    );
    for (quality, initial) in record.final_quality.iter().zip(&record.initial_quality) {
        if quality.stored_fitness + 0.00002 < initial.stored_fitness {
            record
                .errors
                .push(format!("parent {} fitness regressed", quality.chain));
        }
    }
    record.final_state_sha256 = state_hash(final_state)?;
    record.final_fitness_percent = final_state
        .parents
        .iter()
        .map(|p| f32::from_bits(p.fitness_bits))
        .collect();
    record.validation_max_tile_occupancy = final_state.validation_max_tile_occupancy;
    Ok(())
}

fn measure(
    evolver: &mut GpuEvolver,
    params: &MutationParams,
    batches: u32,
) -> Result<(f64, f64, BenchmarkCounters)> {
    let start = Instant::now();
    for _ in 0..batches {
        // Returned drawing collection is part of the real engine batch path.
        let _best = evolver.run_batch(params, true);
    }
    let drain = Instant::now();
    let counters = evolver.benchmark_drain().map_err(anyhow::Error::msg)?;
    let drain_seconds = drain.elapsed().as_secs_f64();
    let elapsed = start.elapsed().as_secs_f64();
    ensure!(elapsed.is_finite() && elapsed > 0.0, "invalid elapsed time");
    Ok((elapsed, drain_seconds, counters))
}

fn check_work(
    c: BenchmarkCounters,
    params: &MutationParams,
    batches: u32,
    errors: &mut Vec<String>,
) {
    let iterations = u64::from(batches) * u64::from(params.gpu_batch_iters);
    let evaluations = iterations * u64::from(params.chain_count) * u64::from(params.lambda);
    if c.completed_batches != u64::from(batches)
        || c.submitted_batches != c.completed_batches
        || c.completed_iterations != iterations
        || c.submitted_iterations != iterations
        || c.completed_evaluations != evaluations
        || c.submitted_evaluations != evaluations
    {
        errors.push(format!("submitted/completed work mismatch: {c:?}"));
    }
}

fn verify_quality(
    snapshot: &BenchmarkSnapshot,
    input: &LoadedFixture,
    errors: &mut Vec<String>,
) -> Result<Vec<ChainQuality>> {
    ensure!(
        !snapshot.parents.is_empty()
            && snapshot.parents.len() == snapshot.parent_framebuffers.len()
            && snapshot.parents.len() == snapshot.parent_total_errors.len(),
        "incomplete parent snapshot"
    );
    let mut quality = Vec::with_capacity(snapshot.parents.len());
    for (index, parent) in snapshot.parents.iter().enumerate() {
        let comparison = compare_chain(
            index,
            parent,
            &snapshot.parent_framebuffers[index],
            snapshot.parent_total_errors[index],
            input,
        )?;
        if !comparison.passes_numerical_policy() {
            errors.push(format!(
                "chain {index}: stored error={}, framebuffer={}, scalar={}, max channel delta={}, total channel delta={}, unexplained pixels={}, fitness={}, expected={}",
                comparison.stored_error,
                comparison.framebuffer_error,
                comparison.scalar_error,
                comparison.max_channel_delta,
                comparison.total_channel_delta,
                comparison.pixel_audit.unexplained_pixels,
                comparison.stored_fitness,
                comparison.expected_stored_fitness,
            ));
        }
        quality.push(comparison);
    }
    verify_control(snapshot, errors);
    Ok(quality)
}

fn compare_chain(
    chain: usize,
    parent: &BenchmarkPackedState,
    framebuffer: &[u32],
    stored_error: u32,
    input: &LoadedFixture,
) -> Result<ChainQuality> {
    let reference = reference::evaluate(
        &parent.polygons,
        &input.rgba,
        input.fixture.width,
        input.fixture.height,
    )?;
    ensure!(
        framebuffer.len() == reference.pixels.len(),
        "framebuffer size mismatch"
    );
    let (differing_pixels, max_channel_delta, total_channel_delta) =
        pixel_difference(framebuffer, &reference.pixels);
    let pixel_audit = uncertainty::audit(
        &parent.polygons,
        framebuffer,
        &reference.pixels,
        input.fixture.width,
        input.fixture.height,
    )?;
    Ok(ChainQuality {
        chain,
        triangles: parent.polygon_count,
        stored_error,
        framebuffer_error: image_error(framebuffer, &input.rgba),
        scalar_error: reference.raw_error,
        stored_fitness: f32::from_bits(parent.fitness_bits),
        expected_stored_fitness: fitness(stored_error, parent.polygon_count, framebuffer.len()),
        scalar_fitness: reference.fitness,
        differing_pixels,
        max_channel_delta,
        total_channel_delta,
        pixel_audit,
    })
}

fn verify_control(snapshot: &BenchmarkSnapshot, errors: &mut Vec<String>) {
    let control = &snapshot.control;
    let selected = snapshot.parents.get(control.best_chain_id as usize);
    let maximum = snapshot
        .parents
        .iter()
        .max_by(|a, b| f32::from_bits(a.fitness_bits).total_cmp(&f32::from_bits(b.fitness_bits)));
    // Every selection visits every active parent with atomicMax. The score must
    // therefore equal their final maximum, regardless of the separate ID store.
    if selected.is_none()
        || maximum.map(|parent| parent.fitness_bits) != Some(control.best_fitness_bits)
    {
        errors.push(format!(
            "control: best chain={} (active={}), fitness={}, active maximum={:?}",
            control.best_chain_id,
            snapshot.parents.len(),
            f32::from_bits(control.best_fitness_bits),
            maximum.map(|parent| f32::from_bits(parent.fitness_bits)),
        ));
    }
    // select.wgsl updates score and ID separately; they can describe different
    // parents. A zero flag leaves the reset ID without claiming an improvement.
    // The ID only chooses production drawing readback, never evolution input.
    // Retain observed disagreement in stderr and raw state artifacts. Final
    // drawings below always come from every active parent, independently of ID.
    if control.new_best_found != 0 {
        if let Some(parent) = selected {
            if parent.fitness_bits != control.best_fitness_bits {
                eprintln!(
                    "[benchmark control telemetry] chain {} fitness={} differs from reported best={}",
                    control.best_chain_id,
                    f32::from_bits(parent.fitness_bits),
                    f32::from_bits(control.best_fitness_bits),
                );
            }
        }
    }
}

fn image_error(pixels: &[u32], rgba: &[u8]) -> u64 {
    pixels
        .iter()
        .zip(rgba.as_chunks::<4>().0)
        .map(|(pixel, target)| {
            (0..3)
                .map(|channel| {
                    u64::from(((pixel >> (channel * 8)) & 255).abs_diff(u32::from(target[channel])))
                })
                .sum::<u64>()
        })
        .sum()
}

fn pixel_difference(actual: &[u32], expected: &[u32]) -> (u32, u32, u64) {
    let (mut count, mut maximum, mut total) = (0, 0, 0);
    for (actual, expected) in actual.iter().zip(expected) {
        count += u32::from(actual != expected);
        for channel in 0..3 {
            let delta =
                ((actual >> (channel * 8)) & 255).abs_diff((expected >> (channel * 8)) & 255);
            maximum = maximum.max(delta);
            total += u64::from(delta);
        }
    }
    (count, maximum, total)
}

fn fitness(error: u32, triangles: u32, pixels: usize) -> f32 {
    let mut result = 100.0 * (1.0 - error as f32 / (765.0 * pixels as f32));
    result -= result * PER_POINT_MULTIPLIER * (triangles * 3) as f32;
    result
}

fn state_hash(snapshot: &BenchmarkSnapshot) -> Result<String> {
    // verify_control checks the active ID range and maximum score, and reports
    // racy ID/score disagreement. These reporting words do not drive evolution.
    // Decoding already strips padding; counters and diagnostic occupancy are omitted.
    let canonical = (
        &snapshot.parents,
        &snapshot.offspring,
        &snapshot.parent_total_errors,
        &snapshot.parent_framebuffers,
        &snapshot.offspring_errors,
        &snapshot.chain_update_flags,
        &snapshot.fitness_packed,
    );
    Ok(fixtures::sha256(&serde_json::to_vec(&canonical)?))
}

fn save_state(output: &Path, state: &BenchmarkSnapshot) -> Result<()> {
    fs::write(output.join("state.json"), serde_json::to_vec(state)?)?;
    let drawings: Vec<_> = state.parents.iter().map(to_drawing).collect();
    fs::write(
        output.join("drawings.json"),
        serde_json::to_vec_pretty(&drawings)?,
    )?;
    Ok(())
}

fn to_drawing(parent: &BenchmarkPackedState) -> Drawing {
    let mut state = GpuDrawingState::zeroed();
    state.polygon_count = parent.polygon_count;
    state.fitness_bits = parent.fitness_bits;
    for (destination, data) in state.polygons.iter_mut().zip(&parent.polygons) {
        *destination = GpuPolygon { data: *data };
    }
    gpu_to_drawing(&state)
}

fn finish_record(record: &mut TrialRecord) {
    record.error_count = record.errors.len();
    record.correctness = record.errors.is_empty();
    record.status = if record.correctness {
        "completed"
    } else {
        "invalid"
    }
    .into();
    if !record.correctness || record.diagnostic {
        record.evals_per_second = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_evolver::benchmark::BenchmarkControl;

    fn snapshot() -> BenchmarkSnapshot {
        let parent = BenchmarkPackedState {
            polygon_count: 0,
            fitness_bits: 100.0f32.to_bits(),
            mutation_scale_bits: 1.0f32.to_bits(),
            stagnation_counter: 0,
            rng_state: 7,
            polygons: Vec::new(),
        };
        BenchmarkSnapshot {
            parents: vec![parent.clone()],
            offspring: vec![parent],
            parent_total_errors: vec![0],
            parent_framebuffers: vec![vec![0xffff_ffff]],
            offspring_errors: vec![[0, 0]],
            chain_update_flags: vec![[0, 0, 0]],
            fitness_packed: vec![100.0f32.to_bits()],
            control: BenchmarkControl {
                new_best_found: 0,
                best_chain_id: 0,
                best_fitness_bits: 100.0f32.to_bits(),
            },
            counters: BenchmarkCounters::default(),
            validation_max_tile_occupancy: None,
        }
    }

    fn input() -> LoadedFixture {
        let fixture = fixtures::Fixture {
            id: "ff".into(),
            width: 1,
            height: 1,
            drawing_file: "ff/drawing.json".into(),
            reference_file: "ff/reference.rgba".into(),
            drawing_sha256: String::new(),
            reference_sha256: String::new(),
            source_image_sha256: String::new(),
            stored_fitness_percent: 100.0,
            scalar_fitness_percent: 100.0,
            complexity: reference::complexity(&[], 1, 1, [8, 8]).unwrap(),
        };
        let mut params = MutationParams::default();
        params.chain_count = 1;
        params.lambda = 1;
        params.gpu_batch_iters = 1;
        LoadedFixture {
            manifest: fixtures::Manifest {
                protocol_version: PROTOCOL_VERSION,
                start_mode: "mature_drawing_restart".into(),
                resolution: 384,
                minimum_similarity_percent: 97.0,
                params,
                cases: vec![fixture.clone()],
            },
            fixture,
            drawing: Drawing {
                polygons: Vec::new(),
                is_dirty: false,
                fitness: 100.0,
            },
            rgba: vec![255; 4],
            manifest_sha256: "analytical-fixture".into(),
        }
    }

    fn prepared_record(input: &LoadedFixture, initial: &BenchmarkSnapshot) -> TrialRecord {
        let mut errors = Vec::new();
        let initial_quality = verify_quality(initial, input, &mut errors).unwrap();
        assert!(errors.is_empty());
        let spec = TrialSpec {
            case_id: "ff".into(),
            seed: 7,
            batches: 1,
            warmup_batches: 0,
            diagnostic: false,
        };
        let provenance = BenchmarkProvenance {
            name: "CPU-only numerical test".into(),
            vendor: 0,
            device: 0,
            device_type: String::new(),
            backend: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            timestamp_period_ns: 0.0,
            allocated_chains: 1,
            offspring_capacity: 1,
            max_storage_buffer_binding_size: 0,
            max_buffer_size: 0,
        };
        let mut record =
            TrialRecord::prepared(input, &spec, initial, initial_quality, 100.0, provenance)
                .unwrap();
        record.counters = BenchmarkCounters {
            submitted_batches: 1,
            submitted_iterations: 1,
            submitted_evaluations: 1,
            completed_batches: 1,
            completed_iterations: 1,
            completed_evaluations: 1,
        };
        record.evals_per_second = Some(42.0);
        record
    }

    #[test]
    fn pending_work_cannot_be_scored_as_completed() {
        let p = MutationParams::default();
        let c = BenchmarkCounters {
            submitted_batches: 1,
            submitted_evaluations: 16384,
            submitted_iterations: 64,
            ..Default::default()
        };
        let mut errors = Vec::new();
        check_work(c, &p, 1, &mut errors);
        assert_eq!(errors.len(), 1);
    }
    #[test]
    fn pixel_gate_detects_wrong_channels_and_error_totals() {
        let (count, max, total) = pixel_difference(&[0xff00_0010], &[0xff00_0012]);
        assert_eq!((count, max, total), (1, 2, 2));
        assert_eq!(image_error(&[0xff00_0010], &[0, 0, 0, 255]), 16);
    }

    #[test]
    fn replay_hash_excludes_telemetry_and_keeps_every_evolution_component() {
        let initial = snapshot();
        let expected = state_hash(&initial).unwrap();
        let mut telemetry = initial.clone();
        telemetry.control.best_chain_id = u32::MAX;
        telemetry.control.new_best_found = 1;
        telemetry.control.best_fitness_bits = 99.0f32.to_bits();
        telemetry.counters.completed_batches = 10;
        telemetry.validation_max_tile_occupancy = Some(257);
        assert_eq!(state_hash(&telemetry).unwrap(), expected);
        // GPU padding cannot enter BenchmarkSnapshot; its decoder has a separate
        // padding test. These mutations exercise the canonical hash itself.
        for component in 0..12 {
            let mut changed = initial.clone();
            match component {
                0 => changed.parents[0].rng_state += 1,
                1 => changed.offspring[0].rng_state += 1,
                2 => changed.parents[0].fitness_bits = 99.0f32.to_bits(),
                3 => changed.parents[0].mutation_scale_bits = 1.5f32.to_bits(),
                4 => changed.parents[0].stagnation_counter = 1,
                5 => {
                    changed.parents[0].polygon_count = 1;
                    changed.parents[0].polygons.push([1, 2, 3, 4]);
                }
                6 => changed.offspring[0].fitness_bits = 1,
                7 => changed.parent_total_errors[0] = 1,
                8 => changed.parent_framebuffers[0][0] = 0xfffe_ffff,
                9 => changed.offspring_errors[0][0] = 1,
                10 => changed.chain_update_flags[0][0] = 1,
                11 => changed.fitness_packed[0] = 99.0f32.to_bits(),
                _ => unreachable!("bounded component list"),
            }
            assert_ne!(
                state_hash(&changed).unwrap(),
                expected,
                "component {component}"
            );
        }
    }

    #[test]
    fn control_checks_range_and_maximum_without_requiring_a_race_free_pair() {
        let mut state = snapshot();
        state.parents.push(state.parents[0].clone());
        state.parents[0].fitness_bits = 99.0f32.to_bits();
        let mut errors = Vec::new();
        verify_control(&state, &mut errors);
        assert!(errors.is_empty(), "zero flag leaves the reset ID unclaimed");
        state.control.new_best_found = 1;
        verify_control(&state, &mut errors);
        assert!(
            errors.is_empty(),
            "a racy ID/score pair is reported as telemetry"
        );
        state.control.best_chain_id = 2;
        verify_control(&state, &mut errors);
        assert_eq!(errors.len(), 1, "out-of-range readback ID is invalid");
        errors.clear();
        state.control.best_chain_id = 1;
        state.control.best_fitness_bits = 98.0f32.to_bits();
        verify_control(&state, &mut errors);
        assert_eq!(errors.len(), 1, "maximum score itself must agree exactly");
    }

    #[test]
    fn one_quantum_still_requires_exact_cache_error_and_fitness() {
        let input = input();
        let state = snapshot();
        let mut parent = state.parents[0].clone();
        // One RGB byte below white has L1=1 and fitness 100*(1-1/765).
        parent.fitness_bits = 99.86928f32.to_bits();
        let mut quality = compare_chain(0, &parent, &[0xffff_fffe], 1, &input).unwrap();
        assert!(quality.passes_numerical_policy());
        assert_eq!(quality.total_channel_delta, 1);
        assert_eq!(quality.pixel_audit.unexplained_pixels, 0);
        quality.stored_error = 0;
        assert!(!quality.passes_numerical_policy());
        quality.stored_error = 1;
        quality.scalar_error = 3;
        assert!(!quality.passes_numerical_policy(), "L1 triangle inequality");
        quality.scalar_error = 0;
        quality.stored_fitness = 99.86918;
        assert!(!quality.passes_numerical_policy());
        quality.stored_fitness = f32::NAN;
        assert!(!quality.passes_numerical_policy());
    }

    #[test]
    fn boundary_audit_retains_large_nominal_differences_and_exact_raw_errors() {
        let input = input();
        let mut parent = snapshot().parents.remove(0);
        // Pixel center is exactly on the black triangle's diagonal. The nominal
        // inclusive renderer covers it; the arithmetic envelope also permits white.
        parent.polygon_count = 1;
        parent.polygons = vec![[0xff00_0000, 0, 65535, 65535 << 16]];
        parent.fitness_bits = 99.99994f32.to_bits();
        let mut quality = compare_chain(0, &parent, &[0xffff_ffff], 0, &input).unwrap();
        assert!(quality.passes_numerical_policy());
        assert_eq!(quality.differing_pixels, 1);
        assert_eq!(quality.max_channel_delta, 255);
        assert_eq!(quality.total_channel_delta, 765);
        assert_eq!(quality.scalar_error, 765);
        assert_eq!(quality.pixel_audit.outside_nominal_quantum, 1);
        assert_eq!(quality.pixel_audit.unexplained_pixels, 0);
        quality.stored_error = 1;
        assert!(!quality.passes_numerical_policy());
    }

    #[test]
    fn final_numerical_failures_remove_the_comparison_score() {
        let input = input();
        let initial = snapshot();
        for failure in 0..4 {
            let mut record = prepared_record(&input, &initial);
            let mut final_state = initial.clone();
            match failure {
                0 => {}
                1 => final_state.parent_total_errors[0] = 1,
                2 => {
                    final_state.parent_framebuffers[0][0] = 0xffff_fffd;
                    final_state.parent_total_errors[0] = 2;
                    final_state.parents[0].fitness_bits = 99.73856f32.to_bits();
                    final_state.control.best_fitness_bits = 99.73856f32.to_bits();
                }
                3 => {
                    final_state.parents[0].fitness_bits = 99.0f32.to_bits();
                    final_state.control.best_fitness_bits = 99.0f32.to_bits();
                }
                _ => unreachable!("bounded failure list"),
            }
            complete_record(&mut record, &final_state, &input).unwrap();
            finish_record(&mut record);
            assert_eq!(record.correctness, failure == 0, "failure {failure}");
            assert_eq!(record.evals_per_second.is_some(), failure == 0);
            if failure != 0 {
                assert!(record
                    .errors
                    .iter()
                    .any(|error| error.starts_with("chain 0:")));
                assert_eq!(record.status, "invalid");
            }
            if failure == 2 {
                assert_eq!(record.final_quality[0].pixel_audit.unexplained_pixels, 1);
            }
        }
    }

    #[test]
    fn diagnostic_records_retain_work_without_a_throughput_score() {
        let input = input();
        let state = snapshot();
        let mut record = prepared_record(&input, &state);
        record.diagnostic = true;
        complete_record(&mut record, &state, &input).unwrap();
        finish_record(&mut record);
        assert!(record.correctness);
        assert_eq!(record.status, "completed");
        assert_eq!(record.counters.completed_evaluations, 1);
        assert!(record.evals_per_second.is_none());
    }

    #[test]
    fn invalid_budgets_fail_before_fixture_io_and_gpu_initialization() {
        let mut spec = TrialSpec {
            case_id: "ff".into(),
            seed: 7,
            batches: 1,
            warmup_batches: 0,
            diagnostic: false,
        };
        for (batches, warmup, message) in
            [(0, 0, "batches"), (4097, 0, "batches"), (1, 17, "warmup")]
        {
            spec.batches = batches;
            spec.warmup_batches = warmup;
            // This cannot create artifacts or load a fixture if a guard regresses.
            let error = run(
                Path::new("/dev/null/manifest.json"),
                &spec,
                Path::new("/dev/null/benchmark-output"),
            )
            .err()
            .expect("invalid budget must fail");
            assert!(error.to_string().contains(message));
        }
    }
}
