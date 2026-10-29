//! Characterize the ordinary GPU API before and after structural changes.
//!
//! Run under the parent's process timeout. No GPU work occurs for --help.
//! Each timestamp mode runs from a seeded start, after reset with work pending,
//! and after seeded preparation with work pending. Only the final flush and
//! snapshot drain the ordinary sequence. Shared trial helpers enforce exact
//! cached/framebuffer byte L1 and the existing numerical policy.
//!
//! Compare comparison.json byte for byte. It includes delivery, the returned
//! drawing's own fitness and geometry digest, chain fitness, work counters and
//! canonical state digests. Timings, device provenance and racy control
//! reporting words remain in summary.json and the snapshots, outside comparison.
//! Ordinary deliveries can expose the existing best-ID race. Retain that evidence
//! for the parent's A/A qualification rather than normalizing returned drawings.
//!
//! Bounds: one device, six runs, <=64 ordinary events and four pending-exercise
//! events per run, four observed chains, <=1000 triangles per drawing. There are
//! at most 14 output files, each <=64 MiB. Fixture loading has its existing caps.
//! Power-of-ten deviations: production Drawing/readback allocations, serde derives
//! and wgpu callbacks belong to the existing interfaces. Observation storage is
//! preallocated; serialization, numerical qualification and hashing happen outside
//! the ordinary sequence. Existing production waits require the parent timeout;
//! snapshot maps retain their existing 120-second timeout. Follow-up changes to
//! those interfaces belong to their owners, not this characterization executable.
#![deny(warnings, clippy::all, clippy::pedantic)]

use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{ensure, Context, Result};
use artgen_backend_rust::{
    benchmark::{
        fixtures::{self, LoadedFixture},
        trial::{self, TimestampMode, TrialRecord, TrialSpec},
    },
    gpu_evolver::{
        benchmark::{BenchmarkCounters, BenchmarkPackedState, BenchmarkSnapshot},
        buffers::{gpu_to_drawing, GpuDrawingState, GpuPolygon},
        EngineCapacity, GpuEvolver,
    },
    models::drawing::Drawing,
    mutation_params::MutationParams,
    settings::POLYGON_CAPACITY,
};
use bytemuck::Zeroable;
use serde::Serialize;

const MAX_ARGS: usize = 6;
const MAX_ARG_BYTES: usize = 4096;
const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
const CHAINS: usize = 4;
const PENDING_BATCHES: u32 = 3;
const SEED: u64 = 7;
const USAGE: &str = "check-gpu-lifecycle --manifest PATH --output NEWDIR [--batches 8|64]\nFixed mature mona, seed 7, unchanged manifest parameters; timestamps false and true.\nUse a parent process timeout. Compare comparison.json exactly; inspect summary.json and per-run snapshots for evidence. --help performs no GPU work.";

#[derive(Debug, Serialize)]
struct Config {
    manifest: PathBuf,
    output: PathBuf,
    batches: u32,
}

#[derive(Debug)]
enum Command {
    Help,
    Run(Config),
}

fn parse_cli(args: &[String]) -> Result<Command> {
    if args == ["--help"] {
        return Ok(Command::Help);
    }
    ensure!(
        args.len() <= MAX_ARGS && args.len().is_multiple_of(2),
        "expected at most three --flag value pairs"
    );
    let (mut manifest, mut output, mut batches) = (None, None, None);
    for pair in args.as_chunks::<2>().0 {
        ensure!(
            pair.iter()
                .all(|arg| !arg.is_empty() && arg.len() <= MAX_ARG_BYTES),
            "arguments must contain 1..4096 bytes"
        );
        let slot = match pair[0].as_str() {
            "--manifest" => &mut manifest,
            "--output" => &mut output,
            "--batches" => &mut batches,
            _ => anyhow::bail!("unknown flag {}", pair[0]),
        };
        ensure!(
            slot.replace(pair[1].as_str()).is_none(),
            "duplicate flag {}",
            pair[0]
        );
    }
    let batches = batches.unwrap_or("8").parse()?;
    ensure!([8, 64].contains(&batches), "batches must be 8 or 64");
    Ok(Command::Run(Config {
        manifest: manifest.context("missing --manifest")?.into(),
        output: output.context("missing --output")?.into(),
        batches,
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Start {
    Ordinary,
    ResetPending,
    RepreparePending,
}

impl Start {
    fn name(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::ResetPending => "reset-pending",
            Self::RepreparePending => "reprepare-pending",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct DrawingObservation {
    fitness_bits: u32,
    polygon_count: usize,
    geometry_sha256: String,
}

impl DrawingObservation {
    fn new(drawing: &Drawing) -> Result<Self> {
        ensure!(
            drawing.polygons.len() <= POLYGON_CAPACITY,
            "oversized returned drawing"
        );
        ensure!(
            drawing.polygons.iter().all(|p| p.points.len() == 3),
            "non-triangle returned drawing"
        );
        Ok(Self {
            // A deferred drawing may score below the current acquisition maximum.
            // Preserve its mapped header, rather than replacing its fitness.
            fitness_bits: drawing.fitness.to_bits(),
            polygon_count: drawing.polygons.len(),
            geometry_sha256: fixtures::sha256(&serde_json::to_vec(&drawing.polygons)?),
        })
    }
}

struct RawEvent {
    call: u32,
    drawing: Option<Drawing>,
    counters: BenchmarkCounters,
    chain_fitness_bits: [u32; CHAINS],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Event {
    call: u32,
    drawing: Option<DrawingObservation>,
    counters: BenchmarkCounters,
    chain_fitness_bits: [u32; CHAINS],
}

fn fitness_bits(evolver: &GpuEvolver) -> [u32; CHAINS] {
    // Capacity is checked before any run. Inactive allocated chains are excluded.
    std::array::from_fn(|index| evolver.chain_fitness()[index].to_bits())
}

impl RawEvent {
    fn capture(evolver: &GpuEvolver, call: u32, drawing: Option<Drawing>) -> Self {
        Self {
            call,
            drawing,
            counters: evolver.benchmark_counters(),
            chain_fitness_bits: fitness_bits(evolver),
        }
    }

    fn observe(self) -> Result<Event> {
        Ok(Event {
            call: self.call,
            drawing: self
                .drawing
                .as_ref()
                .map(DrawingObservation::new)
                .transpose()?,
            counters: self.counters,
            chain_fitness_bits: self.chain_fitness_bits,
        })
    }
}

#[derive(Serialize)]
struct PendingExercise {
    events: Vec<Event>,
    counters_after_reset: BenchmarkCounters,
    chain_fitness_after_reset: [u32; CHAINS],
    timestamp_samples_after_reset: u64,
    first_call_after_reset: Event,
}

#[derive(Serialize)]
struct Run {
    start: Start,
    collect_timestamps: bool,
    pending_exercise: Option<PendingExercise>,
    events: Vec<Event>,
    flush: Event,
    terminal_drawing: Option<DrawingObservation>,
    maximum_parents: Vec<DrawingObservation>,
    qualification: TrialRecord,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    config: Config,
    correctness: bool,
    error_count: usize,
    errors: Vec<String>,
    runs: Vec<Run>,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).take(MAX_ARGS + 1).collect();
    let config = match parse_cli(&args)? {
        Command::Help => {
            println!("{USAGE}");
            return Ok(());
        }
        Command::Run(config) => config,
    };
    let input = fixtures::load(&config.manifest, "mona")?;
    validate_fixture(&input)?;
    fs::create_dir(&config.output).context("output must be a new directory")?;
    let mut report = Report {
        schema: "gpu-lifecycle-v1",
        config,
        correctness: false,
        error_count: 0,
        errors: Vec::with_capacity(6),
        runs: Vec::with_capacity(6),
    };
    if let Err(error) = run_checks(&input, &mut report) {
        report.errors.push(format!("{error:#}"));
    }
    report.error_count = report.errors.len()
        + report
            .runs
            .iter()
            .map(|run| run.qualification.error_count)
            .sum::<usize>();
    report.correctness = report.runs.len() == 6 && report.error_count == 0;
    write_json(&report.config.output.join("summary.json"), &report)?;
    write_json(
        &report.config.output.join("comparison.json"),
        &comparison(&report, &input.manifest_sha256),
    )?;
    ensure!(
        report.correctness,
        "GPU lifecycle failed with {} errors; see summary.json",
        report.error_count
    );
    Ok(())
}

fn validate_fixture(input: &LoadedFixture) -> Result<()> {
    let p = &input.manifest.params;
    ensure!(
        input.manifest.resolution == 384
            && input.fixture.width <= 384
            && input.fixture.height <= 384
            && p.chain_count == u32::try_from(CHAINS)?
            && p.lambda == 64
            && p.gpu_batch_iters == 64
            && p.rasterize_wg == [32, 16],
        "expected fixed 384px, four-chain, lambda64, batch64 mona protocol"
    );
    ensure!(
        input.fixture.complexity.overflowed_tiles == 0,
        "initial tile overflow"
    );
    Ok(())
}

fn run_checks(input: &LoadedFixture, report: &mut Report) -> Result<()> {
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &input.rgba,
        input.fixture.width,
        input.fixture.height,
        &input.drawing,
        EngineCapacity::for_params(&input.manifest.params),
    ));
    evolver
        .benchmark_validate_capacity(&input.manifest.params)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        evolver.chain_fitness().len() >= CHAINS,
        "fewer than four allocated chains"
    );
    for timestamps in [false, true] {
        let first = report.runs.len();
        for start in [
            Start::Ordinary,
            Start::ResetPending,
            Start::RepreparePending,
        ] {
            let run = run_once(&mut evolver, input, &report.config, timestamps, start)?;
            if start != Start::Ordinary {
                check_replay(&report.runs[first], &run, &mut report.errors);
            }
            report.runs.push(run);
        }
    }
    Ok(())
}

fn run_once(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    config: &Config,
    timestamps: bool,
    start: Start,
) -> Result<Run> {
    let output = config
        .output
        .join(format!("{}-timestamps-{timestamps}", start.name()));
    fs::create_dir(&output)?;
    let scopes = evolver.benchmark_error_scopes();
    let result = execute_run(evolver, input, config, timestamps, start, &output);
    let errors = futures_lite::future::block_on(scopes.finish());
    let mut run = result.with_context(|| {
        format!(
            "{} timestamps={timestamps}; device errors={errors:?}",
            start.name()
        )
    })?;
    run.qualification.errors.extend(errors);
    trial::finish_record(&mut run.qualification);
    Ok(run)
}

fn exercise_pending(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    timestamps: bool,
    start: Start,
) -> Result<Option<PendingExercise>> {
    if start == Start::Ordinary {
        return Ok(None);
    }
    let p = &input.manifest.params;
    evolver
        .prepare_seeded_benchmark(&input.drawing, p, SEED, 0, timestamps)
        .map_err(anyhow::Error::msg)?;
    let raw = collect_batches(evolver, input, PENDING_BATCHES, timestamps);
    let before = evolver.benchmark_counters();
    ensure!(
        before.submitted_batches == u64::from(PENDING_BATCHES)
            && before.completed_batches + 1 == before.submitted_batches,
        "exercise did not leave pending work"
    );
    reset_pending(evolver, input, timestamps, start)?;
    let counters_after_reset = evolver.benchmark_counters();
    let timestamp_samples_after_reset = evolver.pass_timings().sample_count;
    ensure!(
        counters_after_reset == BenchmarkCounters::default()
            && timestamp_samples_after_reset == 0
            && evolver.total_evaluations() == 0,
        "pending reset left work or timestamp accounting"
    );
    let chain_fitness_after_reset = fitness_bits(evolver);
    if start == Start::ResetPending {
        ensure!(
            chain_fitness_after_reset == [0; CHAINS],
            "reinitialization retained chain fitness"
        );
    }
    // Submit immediately through the reset interface before trial preparation can
    // clear anything. Leave this probe pending for the shared preparation path.
    let drawing = evolver.run_batch(p, timestamps);
    ensure!(
        drawing.is_none(),
        "first call after reset returned a stale drawing"
    );
    let first_call_after_reset = RawEvent::capture(evolver, 1, drawing).observe()?;
    Ok(Some(PendingExercise {
        events: raw
            .into_iter()
            .map(RawEvent::observe)
            .collect::<Result<_>>()?,
        counters_after_reset,
        chain_fitness_after_reset,
        timestamp_samples_after_reset,
        first_call_after_reset,
    }))
}

fn reset_pending(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    timestamps: bool,
    start: Start,
) -> Result<()> {
    match start {
        Start::ResetPending => evolver.reinit_chains(&input.drawing),
        Start::RepreparePending => {
            let fitness = evolver
                .prepare_seeded_benchmark(
                    &input.drawing,
                    &input.manifest.params,
                    SEED,
                    2,
                    timestamps,
                )
                .map_err(anyhow::Error::msg)?;
            ensure!(
                fitness.is_finite() && fitness > 97.0,
                "repreparation lost mature fitness"
            );
        }
        Start::Ordinary => anyhow::bail!("pending reset requires a reset scenario"),
    }
    Ok(())
}

fn prepare_run(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    batches: u32,
    output: &Path,
) -> Result<TrialRecord> {
    let spec = TrialSpec {
        case_id: "mona".into(),
        seed: SEED,
        batches,
        warmup_batches: 2,
        diagnostic: false,
        timestamp_mode: TimestampMode::Stages,
    };
    let record = trial::prepare(evolver, input, &spec, output)?;
    ensure!(
        fs::metadata(output.join("initial-state.json"))?.len() <= MAX_JSON_BYTES as u64,
        "initial artifact exceeds cap"
    );
    Ok(record)
}

fn qualify_endpoint(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    record: &mut TrialRecord,
    timestamps: bool,
    output: &Path,
) -> Result<BenchmarkSnapshot> {
    record.counters = evolver.benchmark_drain().map_err(anyhow::Error::msg)?;
    record.timestamp_samples = evolver.pass_timings().sample_count;
    let endpoint = evolver
        .benchmark_snapshot(
            input.manifest.params.chain_count,
            input.manifest.params.lambda,
        )
        .map_err(anyhow::Error::msg)?;
    trial::complete_record(record, &endpoint, input)?;
    write_json(&output.join("state.json"), &endpoint)?;
    let expected_samples = if timestamps {
        u64::from(record.batches)
    } else {
        0
    };
    if record.timestamp_samples != expected_samples {
        record.errors.push("timestamp sample count mismatch".into());
    }
    Ok(endpoint)
}

fn execute_run(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    config: &Config,
    timestamps: bool,
    start: Start,
    output: &Path,
) -> Result<Run> {
    let pending_exercise = exercise_pending(evolver, input, timestamps, start)?;
    let mut qualification = prepare_run(evolver, input, config.batches, output)?;
    let raw = collect_batches(evolver, input, config.batches, timestamps);
    // Capture the terminal Drawing before benchmark_snapshot's implicit drain.
    let drawing = evolver.flush_pending();
    let flush = RawEvent::capture(evolver, config.batches + 1, drawing).observe()?;
    let endpoint = qualify_endpoint(evolver, input, &mut qualification, timestamps, output)?;
    let events = raw
        .into_iter()
        .map(RawEvent::observe)
        .collect::<Result<Vec<_>>>()?;
    let terminal_drawing = flush
        .drawing
        .clone()
        .or_else(|| events.iter().rev().find_map(|event| event.drawing.clone()));
    let maximum_parents = maximum_drawings(&endpoint)?;
    check_events(
        &events,
        &flush,
        &input.manifest.params,
        &mut qualification.errors,
    );
    check_terminal(
        terminal_drawing.as_ref(),
        &maximum_parents,
        &mut qualification.errors,
    );
    Ok(Run {
        start,
        collect_timestamps: timestamps,
        pending_exercise,
        events,
        flush,
        terminal_drawing,
        maximum_parents,
        qualification,
    })
}

fn collect_batches(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    batches: u32,
    timestamps: bool,
) -> Vec<RawEvent> {
    debug_assert!(batches <= 64);
    let mut events = Vec::with_capacity(batches as usize);
    for call in 1..=batches {
        let drawing = evolver.run_batch(&input.manifest.params, timestamps);
        events.push(RawEvent::capture(evolver, call, drawing));
    }
    events
}

fn maximum_drawings(snapshot: &BenchmarkSnapshot) -> Result<Vec<DrawingObservation>> {
    let maximum = snapshot
        .parents
        .iter()
        .max_by(|a, b| f32::from_bits(a.fitness_bits).total_cmp(&f32::from_bits(b.fitness_bits)))
        .context("no final parents")?
        .fitness_bits;
    snapshot
        .parents
        .iter()
        .filter(|p| p.fitness_bits == maximum)
        .map(observe_parent)
        .collect()
}

fn observe_parent(parent: &BenchmarkPackedState) -> Result<DrawingObservation> {
    ensure!(
        parent.polygons.len() <= POLYGON_CAPACITY
            && parent.polygon_count as usize == parent.polygons.len(),
        "invalid packed parent count"
    );
    let mut state = GpuDrawingState::zeroed();
    state.polygon_count = parent.polygon_count;
    state.fitness_bits = parent.fitness_bits;
    for (polygon, data) in state.polygons.iter_mut().zip(&parent.polygons) {
        *polygon = GpuPolygon { data: *data };
    }
    DrawingObservation::new(&gpu_to_drawing(&state))
}

fn check_events(
    events: &[Event],
    flush: &Event,
    params: &MutationParams,
    errors: &mut Vec<String>,
) {
    if events.first().is_none_or(|event| event.drawing.is_some())
        || !events.iter().any(|event| event.drawing.is_some())
    {
        errors.push("ordinary first-None and Some delivery coverage missing".into());
    }
    for (index, event) in events.iter().chain(std::iter::once(flush)).enumerate() {
        let submitted = ((index + 1) as u64).min(events.len() as u64);
        let completed = if index == events.len() {
            submitted
        } else {
            submitted - 1
        };
        check_counters(event.counters, submitted, completed, params, errors);
        if event
            .chain_fitness_bits
            .iter()
            .any(|bits| !f32::from_bits(*bits).is_finite())
        {
            errors.push(format!("call {} has nonfinite chain fitness", event.call));
        }
        if let Some(drawing) = &event.drawing {
            if !f32::from_bits(drawing.fitness_bits).is_finite()
                || !event.chain_fitness_bits.contains(&drawing.fitness_bits)
            {
                errors.push(format!(
                    "call {} returned own header absent from chain fitness",
                    event.call
                ));
            }
        }
    }
}

fn check_counters(
    c: BenchmarkCounters,
    submitted: u64,
    completed: u64,
    p: &MutationParams,
    errors: &mut Vec<String>,
) {
    let iterations = u64::from(p.gpu_batch_iters);
    let evaluations = iterations * u64::from(p.chain_count) * u64::from(p.lambda);
    if c.submitted_batches != submitted
        || c.completed_batches != completed
        || c.submitted_iterations != submitted * iterations
        || c.completed_iterations != completed * iterations
        || c.submitted_evaluations != submitted * evaluations
        || c.completed_evaluations != completed * evaluations
    {
        errors.push(format!(
            "ordinary submitted/completed counters mismatch: {c:?}"
        ));
    }
}

fn check_terminal(
    terminal: Option<&DrawingObservation>,
    maxima: &[DrawingObservation],
    errors: &mut Vec<String>,
) {
    if terminal.is_some_and(|drawing| !maxima.contains(drawing)) {
        errors.push(
            "terminal Drawing own header/geometry differs from every final maximum parent".into(),
        );
    }
}

fn check_replay(expected: &Run, actual: &Run, errors: &mut Vec<String>) {
    // Control's separately stored ID can race. Seeded evolution state must still
    // replay exactly after reuse. Ordinary deliveries remain in comparison.json.
    if expected.qualification.initial_state_sha256 != actual.qualification.initial_state_sha256
        || expected.qualification.final_state_sha256 != actual.qualification.final_state_sha256
        || expected.qualification.counters != actual.qualification.counters
    {
        errors.push(format!(
            "{} timestamps={}: seeded state/work differs after pending reuse",
            actual.start.name(),
            actual.collect_timestamps
        ));
    }
}

#[derive(Serialize)]
struct ExactRun<'a> {
    start: Start,
    collect_timestamps: bool,
    pending_exercise: &'a Option<PendingExercise>,
    events: &'a [Event],
    flush: &'a Event,
    terminal_drawing: &'a Option<DrawingObservation>,
    maximum_parents: &'a [DrawingObservation],
    initial_state_sha256: &'a str,
    final_state_sha256: &'a str,
    counters: BenchmarkCounters,
    timestamp_samples: u64,
    correctness: bool,
    errors: &'a [String],
}

#[derive(Serialize)]
struct ExactReport<'a> {
    schema: &'static str,
    fixture_manifest_sha256: &'a str,
    seed: u64,
    batches: u32,
    correctness: bool,
    error_count: usize,
    errors: &'a [String],
    runs: Vec<ExactRun<'a>>,
}

fn comparison<'a>(report: &'a Report, manifest_sha256: &'a str) -> ExactReport<'a> {
    ExactReport {
        schema: report.schema,
        fixture_manifest_sha256: manifest_sha256,
        seed: SEED,
        batches: report.config.batches,
        correctness: report.correctness,
        error_count: report.error_count,
        errors: &report.errors,
        runs: report
            .runs
            .iter()
            .map(|run| ExactRun {
                start: run.start,
                collect_timestamps: run.collect_timestamps,
                pending_exercise: &run.pending_exercise,
                events: &run.events,
                flush: &run.flush,
                terminal_drawing: &run.terminal_drawing,
                maximum_parents: &run.maximum_parents,
                initial_state_sha256: &run.qualification.initial_state_sha256,
                final_state_sha256: &run.qualification.final_state_sha256,
                counters: run.qualification.counters,
                timestamp_samples: run.qualification.timestamp_samples,
                correctness: run.qualification.correctness,
                errors: &run.qualification.errors,
            })
            .collect(),
    }
}

struct LimitedWriter<W> {
    inner: W,
    remaining: usize,
}

impl<W: Write> Write for LimitedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "JSON exceeds artifact cap",
            ));
        }
        let written = self.inner.write(bytes)?;
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = LimitedWriter {
        inner: BufWriter::new(File::create_new(path)?),
        remaining: MAX_JSON_BYTES,
    };
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use artgen_backend_rust::{
        gpu_evolver::benchmark::BenchmarkControl,
        models::{color::Color, point::Point, polygon::Polygon},
    };

    #[test]
    fn help_needs_no_fixture_or_output() -> Result<()> {
        assert!(matches!(parse_cli(&["--help".into()])?, Command::Help));
        Ok(())
    }

    #[test]
    fn cli_requires_paths_and_bounds_work() -> Result<()> {
        let base = ["--manifest", "fixture.json", "--output", "new-dir"].map(str::to_owned);
        let Command::Run(config) = parse_cli(&base)? else {
            panic!("expected run")
        };
        assert_eq!(config.batches, 8);
        let mut long = base.to_vec();
        long.extend(["--batches".into(), "64".into()]);
        let Command::Run(config) = parse_cli(&long)? else {
            panic!("expected run")
        };
        assert_eq!(config.batches, 64);
        for value in ["0", "7", "9", "65", "garbage"] {
            let mut args = base.to_vec();
            args.extend(["--batches".into(), value.into()]);
            assert!(parse_cli(&args).is_err());
        }
        for args in [
            vec![],
            vec!["--output".into(), "dir".into()],
            vec!["--manifest".into(), "fixture".into()],
            vec!["--help".into(), "x".into()],
            vec!["--output".into(), "x".into(), "--output".into(), "y".into()],
        ] {
            assert!(parse_cli(&args).is_err());
        }
        let mut oversized = base.to_vec();
        oversized[1] = "x".repeat(MAX_ARG_BYTES + 1);
        assert!(parse_cli(&oversized).is_err());
        long.push("extra".into());
        assert!(parse_cli(&long).is_err());
        Ok(())
    }

    fn drawing() -> Drawing {
        Drawing {
            polygons: vec![Polygon {
                points: vec![
                    Point { x: 0.1, y: 0.2 },
                    Point { x: 0.3, y: 0.4 },
                    Point { x: 0.5, y: 0.6 },
                ],
                color: Color {
                    r: 12,
                    g: 34,
                    b: 56,
                    a: 78,
                },
            }],
            fitness: 98.0,
            is_dirty: false,
        }
    }

    #[test]
    fn observations_preserve_the_own_header_and_notice_geometry() -> Result<()> {
        let mut drawing = drawing();
        let original = DrawingObservation::new(&drawing)?;
        assert_eq!(original.fitness_bits, 98.0f32.to_bits());
        assert_eq!(original.polygon_count, 1);
        drawing.fitness = 99.0;
        let changed_header = DrawingObservation::new(&drawing)?;
        assert_ne!(original.fitness_bits, changed_header.fitness_bits);
        assert_eq!(original.geometry_sha256, changed_header.geometry_sha256);
        drawing.polygons[0].points[0].x = 0.25;
        let changed_geometry = DrawingObservation::new(&drawing)?;
        assert_ne!(
            changed_header.geometry_sha256,
            changed_geometry.geometry_sha256
        );
        Ok(())
    }

    fn delivery_events() -> Result<([Event; 2], Event)> {
        let chain_fitness_bits = [
            98.0f32.to_bits(),
            99.0f32.to_bits(),
            97.0f32.to_bits(),
            97.0f32.to_bits(),
        ];
        let first = Event {
            call: 1,
            drawing: None,
            chain_fitness_bits,
            counters: BenchmarkCounters {
                submitted_iterations: 64,
                submitted_evaluations: 16384,
                submitted_batches: 1,
                ..Default::default()
            },
        };
        let second = Event {
            call: 2,
            drawing: Some(DrawingObservation::new(&drawing())?),
            chain_fitness_bits,
            counters: BenchmarkCounters {
                submitted_iterations: 128,
                submitted_evaluations: 32768,
                submitted_batches: 2,
                completed_iterations: 64,
                completed_evaluations: 16384,
                completed_batches: 1,
            },
        };
        let flush = Event {
            call: 3,
            drawing: None,
            chain_fitness_bits,
            counters: BenchmarkCounters {
                submitted_iterations: 128,
                submitted_evaluations: 32768,
                submitted_batches: 2,
                completed_iterations: 128,
                completed_evaluations: 32768,
                completed_batches: 2,
            },
        };
        Ok(([first, second], flush))
    }

    #[test]
    fn delivery_accepts_a_valid_own_header_below_the_current_maximum() -> Result<()> {
        let (mut events, flush) = delivery_events()?;
        let params = MutationParams {
            chain_count: 4,
            lambda: 64,
            gpu_batch_iters: 64,
            ..Default::default()
        };
        let mut errors = Vec::new();
        check_events(&events, &flush, &params, &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
        events[1].drawing.as_mut().unwrap().fitness_bits = 96.0f32.to_bits();
        check_events(&events, &flush, &params, &mut errors);
        assert_eq!(errors.len(), 1);
        Ok(())
    }

    #[test]
    fn delivery_rejects_missing_some_and_a_nonempty_first_return() -> Result<()> {
        let (mut events, flush) = delivery_events()?;
        let mut errors = Vec::new();
        events[0].drawing = events[1].drawing.take();
        check_events(&events, &flush, &MutationParams::default(), &mut errors);
        assert_eq!(errors.len(), 1);
        errors.clear();
        events[0].drawing = None;
        check_events(&events, &flush, &MutationParams::default(), &mut errors);
        assert_eq!(errors.len(), 1);
        Ok(())
    }

    #[test]
    fn delivery_rejects_each_wrong_submitted_or_completed_counter() -> Result<()> {
        let (events, flush) = delivery_events()?;
        let params = MutationParams {
            chain_count: 4,
            lambda: 64,
            gpu_batch_iters: 64,
            ..Default::default()
        };
        for changed in 0..6 {
            let mut altered = events.clone();
            let counters = &mut altered[1].counters;
            match changed {
                0 => counters.submitted_batches += 1,
                1 => counters.completed_batches += 1,
                2 => counters.submitted_iterations += 1,
                3 => counters.completed_iterations += 1,
                4 => counters.submitted_evaluations += 1,
                _ => counters.completed_evaluations += 1,
            }
            let mut errors = Vec::new();
            check_events(&altered, &flush, &params, &mut errors);
            assert_eq!(errors.len(), 1);
        }
        Ok(())
    }

    #[test]
    fn comparison_omits_output_location_and_keeps_fixture_identity() -> Result<()> {
        let mut report = Report {
            schema: "gpu-lifecycle-v1",
            config: Config {
                manifest: "before/manifest.json".into(),
                output: "before".into(),
                batches: 8,
            },
            correctness: false,
            error_count: 0,
            errors: Vec::new(),
            runs: Vec::new(),
        };
        let before = serde_json::to_vec(&comparison(&report, "fixture hash"))?;
        report.config.manifest = "after/manifest.json".into();
        report.config.output = "after".into();
        assert_eq!(
            before,
            serde_json::to_vec(&comparison(&report, "fixture hash"))?
        );
        assert_ne!(
            before,
            serde_json::to_vec(&comparison(&report, "different fixture"))?
        );
        report.config.batches = 64;
        assert_ne!(
            before,
            serde_json::to_vec(&comparison(&report, "fixture hash"))?
        );
        Ok(())
    }

    #[test]
    fn terminal_must_match_geometry_and_own_fitness_of_a_maximum() -> Result<()> {
        let original = DrawingObservation::new(&drawing())?;
        let maxima = [original.clone()];
        let mut errors = Vec::new();
        check_terminal(None, &maxima, &mut errors);
        check_terminal(Some(&original), &maxima, &mut errors);
        assert!(errors.is_empty());
        let mut wrong = original.clone();
        wrong.fitness_bits = 99.0f32.to_bits();
        check_terminal(Some(&wrong), &maxima, &mut errors);
        wrong = original;
        wrong.geometry_sha256 = "different geometry".into();
        check_terminal(Some(&wrong), &maxima, &mut errors);
        assert_eq!(errors.len(), 2);
        Ok(())
    }

    #[test]
    fn maximum_parent_selection_keeps_ties_and_ignores_reporting_id() -> Result<()> {
        let parent = BenchmarkPackedState {
            polygon_count: 0,
            fitness_bits: 98.0f32.to_bits(),
            mutation_scale_bits: 0,
            stagnation_counter: 0,
            rng_state: 7,
            polygons: vec![],
        };
        let mut lower = parent.clone();
        lower.fitness_bits = 97.0f32.to_bits();
        let snapshot = BenchmarkSnapshot {
            parents: vec![lower, parent.clone(), parent],
            offspring: vec![],
            parent_total_errors: vec![],
            parent_framebuffers: vec![],
            offspring_errors: vec![],
            chain_update_flags: vec![],
            fitness_packed: vec![],
            control: BenchmarkControl {
                new_best_found: 1,
                best_chain_id: 0,
                best_fitness_bits: 98.0f32.to_bits(),
            },
            counters: BenchmarkCounters::default(),
            validation_max_tile_occupancy: None,
        };
        let maxima = maximum_drawings(&snapshot)?;
        assert_eq!(maxima.len(), 2);
        assert!(maxima
            .iter()
            .all(|parent| parent.fitness_bits == 98.0f32.to_bits()));
        Ok(())
    }

    #[test]
    fn json_cap_rejects_excess_bytes_without_exceeding_file_budget() -> Result<()> {
        let mut writer = LimitedWriter {
            inner: Vec::new(),
            remaining: 5,
        };
        serde_json::to_writer(&mut writer, &"abc")?;
        assert_eq!(writer.inner, b"\"abc\"");
        let error = writer.write(b"x").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::FileTooLarge);
        assert_eq!(writer.inner.len(), 5);
        Ok(())
    }
}
