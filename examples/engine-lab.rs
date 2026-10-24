//! Resident, fixed-parameter engine comparison. Run on the main thread under a
//! parent process timeout: production GPU waits can block and are not cancellable.
//! Setup/oracles/snapshots/JSON are untimed; ordinary engine readbacks are timed.
//! Naga/wgpu and the normal engine allocate internally. Live work is bounded by
//! one case/device, 17 sources, 64 batches and 256 active offspring. Evidence is
//! bounded by 960 timing rows and 51 diagnostic rows of 64 KiB, a 2 MiB summary,
//! 17 sources of 256 KiB, at most 102 diagnostic snapshots of 32 MiB, and
//! three qualifications at the fixed 384 resolution and 1000-triangle capacity.
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use artgen_backend_rust::{
    benchmark::{
        fixtures::{self, LoadedFixture, PROJECTS},
        trial::{self, TimestampMode, TrialSpec},
    },
    gpu_evolver::{
        benchmark::{BenchmarkCounters, BenchmarkProvenance, BenchmarkSnapshot},
        GpuEvolver,
    },
    mutation_params::MutationParams,
    settings::PER_POINT_MULTIPLIER,
};
use serde::Serialize;

const SCHEMA: &str = "engine-lab-v1";
const MAX_SOURCE_BYTES: u64 = 256 * 1024;
const MAX_ROW_BYTES: usize = 64 * 1024;
const MAX_TRIALS: usize = 3 * 16 * 10 * 2;
const FLAGS: [&str; 9] = [
    "manifest",
    "reference-shader",
    "pairs",
    "batches",
    "seed",
    "case",
    "output",
    "shader-stage",
    "diagnostics",
];
const USAGE: &str = "engine-lab --reference-shader PATH --variant NAME=PATH [--variant NAME=PATH ...]\n  --output NEWDIR [--manifest PATH] [--pairs 1..10] [--batches 1..64] [--seed U64] [--case ff|mona|darwin|all]\n  [--shader-stage bin|raster|mutate] [--diagnostics true|false]\nDefaults: fixtures .cache/evolution-bench/fixtures/v1/manifest.json, pairs 2, batches 8, shader-stage bin, diagnostics false.\nFixed manifest parameters: 4 chains, lambda 64, 64 iterations, raster WG 32x16, resolution 384.\nTwo Stages warmups, default seed 7 and all cases; use a parent process timeout. No GPU work for --help.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ShaderStage {
    Bin,
    Raster,
    Mutate,
}

impl ShaderStage {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "bin" => Ok(Self::Bin),
            "raster" => Ok(Self::Raster),
            "mutate" => Ok(Self::Mutate),
            _ => anyhow::bail!("shader-stage must be bin, raster or mutate"),
        }
    }

    fn install(self, evolver: &mut GpuEvolver, source: &str) -> Result<()> {
        match self {
            Self::Bin => evolver.benchmark_set_bin_shader(source),
            Self::Raster => evolver.benchmark_set_raster_shader(source),
            Self::Mutate => evolver.benchmark_set_mutate_shader(source),
        }
        .map_err(anyhow::Error::msg)
    }
}

#[derive(Debug, Serialize)]
struct Config {
    manifest: PathBuf,
    reference_shader: PathBuf,
    variants: Vec<(String, PathBuf)>,
    pairs: u32,
    batches: u32,
    seed: u64,
    case: String,
    output: PathBuf,
    shader_stage: ShaderStage,
    diagnostics: bool,
}

impl Config {
    fn parse(args: &[String]) -> Result<Self> {
        ensure!(
            args.len() <= 50 && args.len() % 2 == 0,
            "expected at most 25 --flag value pairs"
        );
        let mut values = BTreeMap::new();
        let mut variants = Vec::with_capacity(16);
        for pair in args.as_chunks::<2>().0 {
            ensure!(
                pair.iter().all(|arg| arg.len() <= 4096),
                "argument exceeds 4096 bytes"
            );
            let flag = pair[0]
                .strip_prefix("--")
                .context("expected --flag value")?;
            if flag == "variant" {
                parse_variant(&mut variants, &pair[1])?;
            } else {
                ensure!(FLAGS.contains(&flag), "unknown flag {flag}");
                ensure!(
                    values.insert(flag, pair[1].as_str()).is_none(),
                    "duplicate flag {flag}"
                );
            }
        }
        ensure!(!variants.is_empty(), "at least one variant is required");
        let pairs = values.get("pairs").unwrap_or(&"2").parse()?;
        let batches = values.get("batches").unwrap_or(&"8").parse()?;
        let seed = values.get("seed").unwrap_or(&"7").parse()?;
        ensure!((1..=10).contains(&pairs), "pairs must be 1..10");
        ensure!((1..=64).contains(&batches), "batches must be 1..64");
        let case = *values.get("case").unwrap_or(&"all");
        ensure!(
            case == "all" || PROJECTS.contains(&case),
            "case must be ff, mona, darwin or all"
        );
        Ok(Self {
            manifest: PathBuf::from(
                values
                    .get("manifest")
                    .unwrap_or(&".cache/evolution-bench/fixtures/v1/manifest.json"),
            ),
            reference_shader: PathBuf::from(
                *values
                    .get("reference-shader")
                    .context("missing --reference-shader")?,
            ),
            output: PathBuf::from(*values.get("output").context("missing --output")?),
            variants,
            pairs,
            batches,
            seed,
            case: case.into(),
            shader_stage: ShaderStage::parse(values.get("shader-stage").unwrap_or(&"bin"))?,
            diagnostics: values.get("diagnostics").unwrap_or(&"false").parse()?,
        })
    }

    fn cases(&self) -> impl Iterator<Item = &'static str> + '_ {
        PROJECTS
            .into_iter()
            .filter(|case| self.case == "all" || *case == self.case)
    }
}

fn parse_variant(variants: &mut Vec<(String, PathBuf)>, value: &str) -> Result<()> {
    ensure!(variants.len() < 16, "at most 16 variants");
    let (name, path) = value.split_once('=').context("variant must be NAME=PATH")?;
    ensure!(
        valid_name(name) && name != "reference",
        "invalid/reserved variant name"
    );
    ensure!(
        !path.is_empty() && !variants.iter().any(|(n, _)| n == name),
        "empty path or duplicate variant"
    );
    variants.push((name.to_owned(), PathBuf::from(path)));
    Ok(())
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 48
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

#[derive(Serialize)]
struct Source {
    shader_stage: ShaderStage,
    name: String,
    path: PathBuf,
    sha256: String,
    #[serde(skip)]
    wgsl: String,
}

fn read_sources(config: &Config) -> Result<Vec<Source>> {
    let mut sources = Vec::with_capacity(config.variants.len() + 1);
    for (name, path) in std::iter::once(("reference", &config.reference_shader)).chain(
        config
            .variants
            .iter()
            .map(|(name, path)| (name.as_str(), path)),
    ) {
        let bytes = fixtures::read_bounded(path, MAX_SOURCE_BYTES)?;
        let wgsl = String::from_utf8(bytes).context("shader must be UTF-8")?;
        let module = wgpu::naga::front::wgsl::parse_str(&wgsl)
            .map_err(|error| anyhow::anyhow!("{name}: {}", error.emit_to_string(&wgsl)))?;
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .with_context(|| format!("invalid shader {name}"))?;
        sources.push(Source {
            shader_stage: config.shader_stage,
            name: name.into(),
            path: path.clone(),
            sha256: fixtures::sha256(wgsl.as_bytes()),
            wgsl,
        });
    }
    Ok(sources)
}

#[derive(Serialize)]
struct TrialRow {
    schema: &'static str,
    case_id: String,
    variant: String,
    shader_stage: ShaderStage,
    timing_eligible: bool,
    initial_max_tile_occupancy: Option<u32>,
    validation_max_tile_occupancy: Option<u32>,
    candidate: String,
    pair: u32,
    order: u32,
    seed: u64,
    batches: u32,
    warmup_batches: u32,
    timestamp_mode: &'static str,
    elapsed_seconds: f64,
    drain_seconds: f64,
    counters: BenchmarkCounters,
    timestamp_samples: u64,
    sampled_gpu_stage_totals_ns: [f64; 4],
    initial_state_sha256: String,
    final_state_sha256: String,
    params_sha256: String,
    source_sha256: String,
    fixture_manifest_sha256: String,
    gpu: BenchmarkProvenance,
    control: artgen_backend_rust::gpu_evolver::benchmark::BenchmarkControl,
    correctness: bool,
    error_count: usize,
    errors: Vec<String>,
    evals_per_second: Option<f64>,
}

impl TrialRow {
    fn new(
        config: &Config,
        source: &Source,
        case: &str,
        manifest: &str,
        gpu: BenchmarkProvenance,
        timing_eligible: bool,
    ) -> Self {
        Self {
            schema: SCHEMA,
            case_id: case.into(),
            variant: source.name.clone(),
            shader_stage: config.shader_stage,
            timing_eligible,
            initial_max_tile_occupancy: None,
            validation_max_tile_occupancy: None,
            candidate: String::new(),
            pair: 0,
            order: 0,
            seed: config.seed,
            batches: config.batches,
            warmup_batches: 2,
            timestamp_mode: "stages",
            elapsed_seconds: 0.0,
            drain_seconds: 0.0,
            counters: BenchmarkCounters::default(),
            timestamp_samples: 0,
            sampled_gpu_stage_totals_ns: [0.0; 4],
            initial_state_sha256: String::new(),
            final_state_sha256: String::new(),
            params_sha256: String::new(),
            source_sha256: source.sha256.clone(),
            fixture_manifest_sha256: manifest.into(),
            gpu,
            control: artgen_backend_rust::gpu_evolver::benchmark::BenchmarkControl {
                new_best_found: 0,
                best_chain_id: 0,
                best_fitness_bits: 0,
            },
            correctness: false,
            error_count: 0,
            errors: Vec::new(),
            evals_per_second: None,
        }
    }
}

#[derive(Serialize)]
struct QualifiedCase {
    case_id: String,
    params: MutationParams,
    params_sha256: String,
    initial_state_sha256: String,
    final_state_sha256: String,
    gpu: BenchmarkProvenance,
}

#[derive(Serialize)]
struct Rates {
    case_id: String,
    candidate: String,
    side: String,
    samples: Vec<f64>,
}

#[derive(Serialize)]
struct Summary<'a> {
    schema: &'static str,
    status: &'static str,
    error_count: usize,
    errors: Vec<String>,
    config: &'a Config,
    sources: &'a [Source],
    fixture_manifest_sha256: String,
    warmup_batches: u32,
    timestamp_mode: &'static str,
    completed_trials: usize,
    expected_trials: usize,
    completed_diagnostics: usize,
    expected_diagnostics: usize,
    diagnostic_counters: BenchmarkCounters,
    cases: Vec<QualifiedCase>,
    rates: Vec<Rates>,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).take(51).collect();
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let config = Config::parse(&args)?;
    let sources = read_sources(&config)?;
    let manifest = fixtures::read_bounded(&config.manifest, 2 * 1024 * 1024)?;
    // Validate every fixture before allocating a device or touching the output.
    for case in PROJECTS {
        validate_fixed_input(&fixtures::load(&config.manifest, case)?)?;
    }
    fs::create_dir(&config.output).context("output must be a new directory")?;
    fs::write(config.output.join("manifest.json"), &manifest)?;
    for source in &sources {
        fs::write(
            config.output.join(format!("{}.wgsl", source.name)),
            &source.wgsl,
        )?;
    }
    let mut rows = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(config.output.join("trials.jsonl"))?;
    let mut summary = Summary {
        schema: SCHEMA,
        status: "running",
        error_count: 0,
        errors: Vec::new(),
        config: &config,
        sources: &sources,
        fixture_manifest_sha256: fixtures::sha256(&manifest),
        warmup_batches: 2,
        timestamp_mode: "stages",
        completed_trials: 0,
        expected_trials: config.cases().count()
            * config.variants.len()
            * usize::try_from(config.pairs)?
            * 2,
        completed_diagnostics: 0,
        expected_diagnostics: if config.diagnostics {
            config.cases().count() * sources.len()
        } else {
            0
        },
        diagnostic_counters: BenchmarkCounters::default(),
        cases: Vec::with_capacity(3),
        rates: Vec::with_capacity(3 * config.variants.len() * 2),
    };
    write_summary(&summary)?;
    let result = run_cases(&mut summary, &mut rows);
    if let Err(error) = &result {
        summary.errors.push(format!("{error:#}"));
        summary.error_count = summary.error_count.max(1);
    }
    summary.status = if result.is_ok() {
        "completed"
    } else {
        "failed"
    };
    write_summary(&summary)?;
    result
}

fn validate_fixed_input(input: &LoadedFixture) -> Result<()> {
    let p = &input.manifest.params;
    ensure!(
        input.manifest.resolution == 384,
        "fixed protocol requires resolution 384"
    );
    ensure!(
        p.chain_count == 4
            && p.lambda == 64
            && p.gpu_batch_iters == 64
            && p.rasterize_wg == [32, 16],
        "fixed protocol requires 4 chains, lambda 64, 64 iterations, raster WG 32x16"
    );
    ensure!(
        input.fixture.complexity.overflowed_tiles == 0,
        "initial tile overflow"
    );
    Ok(())
}

fn run_cases(summary: &mut Summary<'_>, rows: &mut File) -> Result<()> {
    let mut diagnostics = if summary.config.diagnostics {
        Some(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(summary.config.output.join("diagnostics.jsonl"))?,
        )
    } else {
        None
    };
    for case in summary.config.cases() {
        let input = fixtures::load(&summary.config.manifest, case)?;
        ensure!(
            input.manifest_sha256 == summary.fixture_manifest_sha256,
            "manifest changed between cases"
        );
        // One device per case, retained across all sources and pairs. Cases run
        // serially so only one full set of evolution buffers is resident at once.
        let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
            &input.rgba,
            input.fixture.width,
            input.fixture.height,
            &input.drawing,
        ));
        precompile_sources(&mut evolver, &input, summary)?;
        let (initial, endpoint) = qualify(&mut evolver, &input, summary)?;
        let params_sha256 = fixtures::sha256(&serde_json::to_vec(&input.manifest.params)?);
        summary.cases.push(QualifiedCase {
            case_id: case.into(),
            params: input.manifest.params.clone(),
            params_sha256,
            initial_state_sha256: state_hash(&initial)?,
            final_state_sha256: state_hash(&endpoint)?,
            gpu: evolver.benchmark_provenance(),
        });
        write_summary(summary)?;
        if let Some(file) = diagnostics.as_mut() {
            run_diagnostics(&mut evolver, &input, &initial, &endpoint, summary, file)?;
        }
        run_pairs(&mut evolver, &input, &initial, &endpoint, summary, rows)?;
    }
    ensure!(
        summary.completed_trials == summary.expected_trials,
        "trial count mismatch"
    );
    ensure!(
        summary.completed_diagnostics == summary.expected_diagnostics,
        "diagnostic count mismatch"
    );
    Ok(())
}

fn precompile_sources(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    summary: &Summary<'_>,
) -> Result<()> {
    let scopes = evolver.benchmark_error_scopes();
    // Establish the fixed workgroup before compiling every source. Qualification
    // still owns the single numerical oracle and the existing two warmups.
    let result = (|| -> Result<()> {
        evolver
            .prepare_seeded_benchmark(
                &input.drawing,
                &input.manifest.params,
                summary.config.seed,
                0,
                true,
            )
            .map_err(anyhow::Error::msg)?;
        for source in summary.sources {
            summary.config.shader_stage.install(evolver, &source.wgsl)?;
        }
        Ok(())
    })();
    let errors = futures_lite::future::block_on(scopes.finish());
    result?;
    ensure!(
        errors.is_empty(),
        "source precompile device errors: {errors:?}"
    );
    Ok(())
}

fn spec(config: &Config, case: &str) -> TrialSpec {
    TrialSpec {
        case_id: case.into(),
        seed: config.seed,
        batches: config.batches,
        warmup_batches: 2,
        diagnostic: false,
        timestamp_mode: TimestampMode::Stages,
    }
}

fn qualify(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    summary: &Summary<'_>,
) -> Result<(BenchmarkSnapshot, BenchmarkSnapshot)> {
    let output = summary
        .config
        .output
        .join(format!("qualification-{}", input.fixture.id));
    fs::create_dir(&output)?;
    let scopes = evolver.benchmark_error_scopes();
    let result = qualify_inner(evolver, input, summary, &output);
    let errors = futures_lite::future::block_on(scopes.finish());
    fs::write(
        output.join("device-errors.json"),
        serde_json::to_vec(&errors)?,
    )?;
    let (mut record, initial, endpoint) = result?;
    record.errors.extend(errors);
    trial::finish_record(&mut record);
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    ensure!(
        record.correctness,
        "reference qualification failed: {:?}",
        record.errors
    );
    Ok((initial, endpoint))
}

fn qualify_inner(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    summary: &Summary<'_>,
    output: &Path,
) -> Result<(trial::TrialRecord, BenchmarkSnapshot, BenchmarkSnapshot)> {
    summary
        .config
        .shader_stage
        .install(evolver, &summary.sources[0].wgsl)?;
    let spec = spec(summary.config, &input.fixture.id);
    let mut record = trial::prepare(evolver, input, &spec, output)?;
    let params = &input.manifest.params;
    let initial = evolver
        .benchmark_snapshot(params.chain_count, params.lambda)
        .map_err(anyhow::Error::msg)?;
    (
        record.elapsed_seconds,
        record.drain_seconds,
        record.counters,
    ) = measure(evolver, params, spec.batches)?;
    let endpoint = evolver
        .benchmark_snapshot(params.chain_count, params.lambda)
        .map_err(anyhow::Error::msg)?;
    trial::save_state(output, &endpoint)?;
    trial::complete_record(&mut record, &endpoint, input)?;
    record.timestamp_samples = evolver.pass_timings().sample_count;
    record.sampled_gpu_stage_totals_ns = stage_times(evolver);
    ensure!(
        record.timestamp_samples == u64::from(spec.batches),
        "qualification timestamp count mismatch"
    );
    ensure!(
        record.initial_state_sha256 == state_hash(&initial)?
            && record.final_state_sha256 == state_hash(&endpoint)?,
        "canonical hash protocol mismatch"
    );
    Ok((record, initial, endpoint))
}

fn run_diagnostics(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    initial: &BenchmarkSnapshot,
    endpoint: &BenchmarkSnapshot,
    summary: &mut Summary<'_>,
    rows: &mut File,
) -> Result<()> {
    for index in 0..summary.sources.len() {
        let source = &summary.sources[index];
        let output = summary
            .config
            .output
            .join(format!("diagnostic-{}-{}", input.fixture.id, source.name));
        fs::create_dir(&output)?;
        let scopes = evolver.benchmark_error_scopes();
        let enabled = evolver.benchmark_set_validation(true);
        let setup_errors = futures_lite::future::block_on(scopes.finish());
        enabled.map_err(anyhow::Error::msg)?;
        ensure!(
            setup_errors.is_empty(),
            "diagnostic setup: {setup_errors:?}"
        );
        let scopes = evolver.benchmark_error_scopes();
        let mut row = run_trial(
            evolver,
            input,
            initial,
            endpoint,
            summary.config,
            source,
            false,
        );
        row.candidate.clone_from(&source.name);
        if let Err(error) = evolver.benchmark_set_validation(false) {
            row.errors.push(error);
        }
        row.errors
            .extend(futures_lite::future::block_on(scopes.finish()));
        finish_row(&mut row)?;
        append_row(rows, &row)?;
        fs::write(output.join("result.json"), serde_json::to_vec_pretty(&row)?)?;
        summary.completed_diagnostics += 1;
        summary.error_count += row.error_count;
        add_counters(&mut summary.diagnostic_counters, row.counters);
        write_summary(summary)?;
        ensure!(row.correctness, "diagnostic failed: {:?}", row.errors);
    }
    Ok(())
}

fn finish_row(row: &mut TrialRow) -> Result<()> {
    row.error_count = row.errors.len();
    row.correctness = row.errors.is_empty();
    row.evals_per_second = None;
    if row.correctness && row.timing_eligible {
        ensure!(
            row.elapsed_seconds.is_finite() && row.elapsed_seconds > 0.0,
            "invalid elapsed time"
        );
        row.evals_per_second = Some(
            f64::from(u32::try_from(row.counters.completed_evaluations)?) / row.elapsed_seconds,
        );
    }
    Ok(())
}

fn add_counters(total: &mut BenchmarkCounters, work: BenchmarkCounters) {
    total.submitted_batches += work.submitted_batches;
    total.completed_batches += work.completed_batches;
    total.submitted_iterations += work.submitted_iterations;
    total.completed_iterations += work.completed_iterations;
    total.submitted_evaluations += work.submitted_evaluations;
    total.completed_evaluations += work.completed_evaluations;
}

fn record_diagnostic_state(
    config: &Config,
    row: &mut TrialRow,
    state: &BenchmarkSnapshot,
    phase: &str,
) -> Result<()> {
    if row.timing_eligible {
        return Ok(());
    }
    let maximum = state.validation_max_tile_occupancy;
    if phase == "initial" {
        row.initial_max_tile_occupancy = maximum;
    } else {
        row.validation_max_tile_occupancy = maximum;
    }
    if !matches!(maximum, Some(0..=256)) {
        row.errors.push(format!(
            "{phase} missing/overflowed tile occupancy: {maximum:?}"
        ));
    }
    let output = config
        .output
        .join(format!("diagnostic-{}-{}", row.case_id, row.variant));
    let bytes = serde_json::to_vec(state)?;
    ensure!(
        bytes.len() <= 32 * 1024 * 1024,
        "diagnostic snapshot exceeds 32 MiB"
    );
    fs::write(output.join(format!("{phase}-state.json")), bytes)?;
    Ok(())
}

fn run_pairs(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    initial: &BenchmarkSnapshot,
    endpoint: &BenchmarkSnapshot,
    summary: &mut Summary<'_>,
    rows: &mut File,
) -> Result<()> {
    for variant in 1..summary.sources.len() {
        for pair in 0..summary.config.pairs {
            let order = if pair % 2 == 0 {
                [0, variant]
            } else {
                [variant, 0]
            };
            for (position, source_index) in order.into_iter().enumerate() {
                let source = &summary.sources[source_index];
                let scopes = evolver.benchmark_error_scopes();
                let mut row = run_trial(
                    evolver,
                    input,
                    initial,
                    endpoint,
                    summary.config,
                    source,
                    true,
                );
                let device_errors = futures_lite::future::block_on(scopes.finish());
                row.candidate.clone_from(&summary.sources[variant].name);
                row.pair = pair;
                row.order = u32::try_from(position)?;
                row.errors.extend(device_errors);
                finish_row(&mut row)?;
                append_row(rows, &row)?;
                summary.completed_trials += 1;
                summary.error_count += row.error_count;
                if row.correctness {
                    add_rate(
                        summary,
                        &input.fixture.id,
                        variant,
                        source_index,
                        row.evals_per_second.context("missing valid rate")?,
                    );
                }
                write_summary(summary)?;
                println!("{}", serde_json::to_string(&row)?);
                ensure!(row.correctness, "trial failed: {:?}", row.errors);
            }
        }
    }
    Ok(())
}

fn run_trial(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    baseline_initial: &BenchmarkSnapshot,
    baseline_final: &BenchmarkSnapshot,
    config: &Config,
    source: &Source,
    timing_eligible: bool,
) -> TrialRow {
    let mut row = TrialRow::new(
        config,
        source,
        &input.fixture.id,
        &input.manifest_sha256,
        evolver.benchmark_provenance(),
        timing_eligible,
    );
    if let Err(error) = execute_trial(
        evolver,
        input,
        (baseline_initial, baseline_final),
        config,
        source,
        &mut row,
    ) {
        row.counters = evolver.benchmark_counters();
        row.timestamp_samples = evolver.pass_timings().sample_count;
        row.errors.push(format!("{error:#}"));
    }
    row
}

fn execute_trial(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    baseline: (&BenchmarkSnapshot, &BenchmarkSnapshot),
    config: &Config,
    source: &Source,
    row: &mut TrialRow,
) -> Result<()> {
    let params = &input.manifest.params;
    row.params_sha256 = fixtures::sha256(&serde_json::to_vec(params)?);
    config.shader_stage.install(evolver, &source.wgsl)?;
    evolver
        .prepare_seeded_benchmark(&input.drawing, params, config.seed, 2, true)
        .map_err(anyhow::Error::msg)?;
    let initial = evolver
        .benchmark_snapshot(params.chain_count, params.lambda)
        .map_err(anyhow::Error::msg)?;
    row.initial_state_sha256 = state_hash(&initial)?;
    record_diagnostic_state(config, row, &initial, "initial")?;
    compare_state(&initial, baseline.0, "initial", &mut row.errors);
    check_pixels(&initial, &input.rgba, true, &mut row.errors)?;
    ensure!(row.errors.is_empty(), "initial trial state failed");
    if row.timing_eligible {
        (row.elapsed_seconds, row.drain_seconds, row.counters) =
            measure(evolver, params, config.batches)?;
    } else {
        for _ in 0..config.batches {
            let _best = evolver.run_batch(params, true);
        }
        row.counters = evolver.benchmark_drain().map_err(anyhow::Error::msg)?;
    }
    row.timestamp_samples = evolver.pass_timings().sample_count;
    row.sampled_gpu_stage_totals_ns = stage_times(evolver);
    let endpoint = evolver
        .benchmark_snapshot(params.chain_count, params.lambda)
        .map_err(anyhow::Error::msg)?;
    row.final_state_sha256 = state_hash(&endpoint)?;
    record_diagnostic_state(config, row, &endpoint, "final")?;
    compare_state(&endpoint, baseline.1, "final", &mut row.errors);
    check_pixels(&endpoint, &input.rgba, false, &mut row.errors)?;
    check_work(
        row.counters,
        params,
        config.batches,
        row.timestamp_samples,
        &mut row.errors,
    );
    row.control = endpoint.control;
    Ok(())
}

fn measure(
    evolver: &mut GpuEvolver,
    params: &MutationParams,
    batches: u32,
) -> Result<(f64, f64, BenchmarkCounters)> {
    let start = Instant::now();
    for _ in 0..batches {
        let _best = evolver.run_batch(params, true);
    }
    let drain = Instant::now();
    let counters = evolver.benchmark_drain().map_err(anyhow::Error::msg)?;
    let drain_seconds = drain.elapsed().as_secs_f64();
    let elapsed = start.elapsed().as_secs_f64();
    ensure!(elapsed.is_finite() && elapsed > 0.0, "invalid elapsed time");
    Ok((elapsed, drain_seconds, counters))
}

fn stage_times(evolver: &GpuEvolver) -> [f64; 4] {
    let p = evolver.pass_timings();
    [
        p.mutate_ns,
        p.bin_polygons_ns,
        p.rasterize_error_ns,
        p.select_ns,
    ]
}

fn check_work(
    c: BenchmarkCounters,
    p: &MutationParams,
    batches: u32,
    samples: u64,
    errors: &mut Vec<String>,
) {
    let iterations = u64::from(batches) * u64::from(p.gpu_batch_iters);
    let evaluations = iterations * u64::from(p.chain_count) * u64::from(p.lambda);
    if c.submitted_batches != u64::from(batches)
        || c.completed_batches != u64::from(batches)
        || c.submitted_iterations != iterations
        || c.completed_iterations != iterations
        || c.submitted_evaluations != evaluations
        || c.completed_evaluations != evaluations
        || samples != u64::from(batches)
    {
        errors.push(format!(
            "work/timestamp count mismatch: {c:?}, samples={samples}"
        ));
    }
}

fn state_hash(state: &BenchmarkSnapshot) -> Result<String> {
    // Identical to the qualified trial protocol: seven decoded fields, no
    // padding, control race/reporting words, counters, or occupancy diagnostics.
    Ok(fixtures::sha256(&serde_json::to_vec(&(
        &state.parents,
        &state.offspring,
        &state.parent_total_errors,
        &state.parent_framebuffers,
        &state.offspring_errors,
        &state.chain_update_flags,
        &state.fitness_packed,
    ))?))
}

fn compare_state(
    actual: &BenchmarkSnapshot,
    expected: &BenchmarkSnapshot,
    phase: &str,
    errors: &mut Vec<String>,
) {
    let fields = [
        ("parents", actual.parents == expected.parents),
        ("offspring", actual.offspring == expected.offspring),
        (
            "parent_total_errors",
            actual.parent_total_errors == expected.parent_total_errors,
        ),
        (
            "parent_framebuffers",
            actual.parent_framebuffers == expected.parent_framebuffers,
        ),
        (
            "offspring_errors",
            actual.offspring_errors == expected.offspring_errors,
        ),
        (
            "chain_update_flags",
            actual.chain_update_flags == expected.chain_update_flags,
        ),
        (
            "fitness_packed",
            actual.fitness_packed == expected.fitness_packed,
        ),
    ];
    for (name, equal) in fields {
        if !equal {
            errors.push(format!("{phase} {name} differs from qualified reference"));
        }
    }
}

fn check_pixels(
    state: &BenchmarkSnapshot,
    rgba: &[u8],
    initial: bool,
    errors: &mut Vec<String>,
) -> Result<()> {
    ensure!(
        state.parents.len() == state.parent_framebuffers.len()
            && state.parents.len() == state.parent_total_errors.len()
            && state.parents.len() == state.fitness_packed.len(),
        "incomplete parent snapshot"
    );
    for (chain, parent) in state.parents.iter().enumerate() {
        let pixels = &state.parent_framebuffers[chain];
        ensure!(pixels.len() * 4 == rgba.len(), "framebuffer size mismatch");
        let error: u64 = pixels
            .iter()
            .zip(rgba.as_chunks::<4>().0)
            .map(|(pixel, target)| {
                (0..3)
                    .map(|channel| {
                        u64::from(
                            ((pixel >> (channel * 8)) & 255).abs_diff(u32::from(target[channel])),
                        )
                    })
                    .sum::<u64>()
            })
            .sum();
        let stored_error = state.parent_total_errors[chain];
        let mut expected = 100.0
            * (1.0
                - rounded_f32(stored_error) / (765.0 * rounded_f32(u32::try_from(pixels.len())?)));
        expected -= expected * PER_POINT_MULTIPLIER * rounded_f32(parent.polygon_count * 3);
        let fitness = f32::from_bits(parent.fitness_bits);
        if error != u64::from(stored_error)
            || !fitness.is_finite()
            || (fitness - expected).abs() > 0.00002
            || parent.fitness_bits != state.fitness_packed[chain]
            || (initial && fitness <= 97.0)
        {
            errors.push(format!("chain {chain}: L1={error}, cached={stored_error}, fitness={fitness}, expected={expected}, packed={}", state.fitness_packed[chain]));
        }
    }
    Ok(())
}

// Match the qualified oracle's u32-to-f32 rounding without unchecked casts.
// Both u16 terms and the power-of-two multiplication are exact; only the sum rounds.
fn rounded_f32(value: u32) -> f32 {
    let [a, b, c, d] = value.to_le_bytes();
    f32::from(u16::from_le_bytes([c, d])) * 65536.0 + f32::from(u16::from_le_bytes([a, b]))
}

fn add_rate(summary: &mut Summary<'_>, case: &str, candidate: usize, source: usize, rate: f64) {
    let candidate_name = &summary.sources[candidate].name;
    let side = &summary.sources[source].name;
    let index = summary
        .rates
        .iter()
        .position(|r| r.case_id == case && r.candidate == *candidate_name && r.side == *side);
    let index = index.unwrap_or_else(|| {
        summary.rates.push(Rates {
            case_id: case.into(),
            candidate: candidate_name.clone(),
            side: side.clone(),
            samples: Vec::with_capacity(10),
        });
        summary.rates.len() - 1
    });
    let r = &mut summary.rates[index];
    assert!(r.samples.len() < 10 && rate.is_finite() && rate > 0.0);
    r.samples.push(rate);
}

fn append_row(file: &mut File, row: &TrialRow) -> Result<()> {
    let bytes = serde_json::to_vec(row)?;
    ensure!(
        bytes.len() <= MAX_ROW_BYTES,
        "trial evidence exceeds 64 KiB"
    );
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.flush()?;
    // Persist completed evidence even if the parent's timeout kills the next trial.
    file.sync_data()?;
    Ok(())
}

fn write_summary(summary: &Summary<'_>) -> Result<()> {
    ensure!(
        summary.completed_trials <= MAX_TRIALS && summary.completed_diagnostics <= 3 * 17,
        "trial artifact bound exceeded"
    );
    let bytes = serde_json::to_vec_pretty(summary)?;
    ensure!(bytes.len() <= 2 * 1024 * 1024, "summary exceeds 2 MiB");
    let path = summary.config.output.join("summary.pending");
    fs::write(&path, bytes)?;
    fs::rename(path, summary.config.output.join("summary.json"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use artgen_backend_rust::gpu_evolver::benchmark::{BenchmarkControl, BenchmarkPackedState};

    fn args() -> Vec<String> {
        [
            "--reference-shader",
            "reference.wgsl",
            "--variant",
            "candidate=candidate.wgsl",
            "--output",
            "new-output",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    fn white() -> BenchmarkSnapshot {
        BenchmarkSnapshot {
            parents: vec![BenchmarkPackedState {
                polygon_count: 0,
                fitness_bits: 100.0f32.to_bits(),
                mutation_scale_bits: 1.0f32.to_bits(),
                stagnation_counter: 0,
                rng_state: 7,
                polygons: Vec::new(),
            }],
            offspring: Vec::new(),
            parent_total_errors: vec![0],
            parent_framebuffers: vec![vec![0xffff_ffff]],
            offspring_errors: Vec::new(),
            chain_update_flags: vec![[0; 3]],
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

    fn provenance() -> BenchmarkProvenance {
        BenchmarkProvenance {
            name: "cpu test".into(),
            vendor: 0,
            device: 0,
            device_type: "test".into(),
            backend: "test".into(),
            driver: String::new(),
            driver_info: String::new(),
            timestamp_period_ns: 1.0,
            allocated_chains: 4,
            offspring_capacity: 256,
            max_storage_buffer_binding_size: 0,
            max_buffer_size: 0,
        }
    }

    #[test]
    fn cli_stage_and_diagnostics_are_optional_and_strict() -> Result<()> {
        let config = Config::parse(&args())?;
        assert_eq!(config.shader_stage, ShaderStage::Bin);
        assert!(!config.diagnostics);
        for stage in ["bin", "raster", "mutate"] {
            for diagnostics in ["true", "false"] {
                let mut selected = args();
                selected.extend([
                    "--shader-stage".into(),
                    stage.into(),
                    "--diagnostics".into(),
                    diagnostics.into(),
                ]);
                let value = serde_json::to_value(Config::parse(&selected)?)?;
                assert_eq!(value["shader_stage"], stage);
                assert_eq!(value["diagnostics"], diagnostics == "true");
            }
        }
        for (flag, value) in [
            ("--shader-stage", "Raster"),
            ("--shader-stage", "Mutate"),
            ("--shader-stage", "select"),
            ("--diagnostics", "1"),
            ("--diagnostics", "TRUE"),
        ] {
            let mut invalid = args();
            invalid.extend([flag.into(), value.into()]);
            assert!(Config::parse(&invalid).is_err());
        }
        for flag in ["--shader-stage", "--diagnostics"] {
            let mut invalid = args();
            invalid.extend([flag.into(), "bin".into(), flag.into(), "true".into()]);
            assert!(Config::parse(&invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn diagnostic_evidence_requires_occupancy_and_excludes_rates() -> Result<()> {
        let mut config = Config::parse(&args())?;
        config.output =
            std::env::temp_dir().join(format!("engine-lab-diag-{}", uuid::Uuid::new_v4()));
        let output = config.output.join("diagnostic-ff-candidate");
        fs::create_dir_all(&output)?;
        let source = Source {
            shader_stage: ShaderStage::Bin,
            name: "candidate".into(),
            path: "candidate.wgsl".into(),
            sha256: "abc".into(),
            wgsl: String::new(),
        };
        let result = (|| -> Result<()> {
            for maximum in [None, Some(257), Some(256)] {
                let mut row =
                    TrialRow::new(&config, &source, "ff", "manifest", provenance(), false);
                let mut state = white();
                state.validation_max_tile_occupancy = maximum;
                record_diagnostic_state(&config, &mut row, &state, "initial")?;
                record_diagnostic_state(&config, &mut row, &state, "final")?;
                finish_row(&mut row)?;
                assert_eq!(row.correctness, maximum == Some(256));
                let value = serde_json::to_value(&row)?;
                assert_eq!(value["timing_eligible"], false);
                assert!(value["evals_per_second"].is_null());
                assert_eq!(row.initial_max_tile_occupancy, maximum);
                assert_eq!(row.validation_max_tile_occupancy, maximum);
                assert_eq!(
                    fs::read(output.join("final-state.json"))?,
                    serde_json::to_vec(&state)?
                );
                row.errors.push("device validation error".into());
                finish_row(&mut row)?;
                assert!(!row.correctness && row.evals_per_second.is_none());
            }
            Ok(())
        })();
        fs::remove_dir_all(&config.output)?;
        result
    }

    #[test]
    fn cli_enforces_fixed_protocol_and_work_bounds() -> Result<()> {
        let config = Config::parse(&args())?;
        assert_eq!((config.pairs, config.batches, config.seed), (2, 8, 7));
        assert_eq!(config.cases().collect::<Vec<_>>(), PROJECTS);
        for (case, seed) in [("ff", "11"), ("mona", "13"), ("darwin", "17")] {
            let mut selected = args();
            selected.extend(["--case".into(), case.into(), "--seed".into(), seed.into()]);
            let config = Config::parse(&selected)?;
            assert_eq!(config.cases().collect::<Vec<_>>(), [case]);
            assert_eq!(config.seed, seed.parse::<u64>()?);
        }
        for (flag, value) in [
            ("--pairs", "0"),
            ("--pairs", "11"),
            ("--batches", "0"),
            ("--batches", "65"),
            ("--seed", "invalid"),
            ("--case", "unknown"),
            ("--lambda", "32"),
        ] {
            let mut invalid = args();
            invalid.extend([flag.to_owned(), value.to_owned()]);
            assert!(Config::parse(&invalid).is_err());
        }
        let mut maximal = args();
        for index in 1..16 {
            maximal.extend(["--variant".into(), format!("v{index}=path")]);
        }
        maximal.extend([
            "--pairs".into(),
            "10".into(),
            "--batches".into(),
            "64".into(),
            "--seed".into(),
            "7".into(),
            "--manifest".into(),
            "manifest.json".into(),
            "--case".into(),
            "all".into(),
            "--shader-stage".into(),
            "raster".into(),
            "--diagnostics".into(),
            "true".into(),
        ]);
        assert_eq!(Config::parse(&maximal)?.variants.len(), 16);
        maximal.extend(["--variant".into(), "overflow=path".into()]);
        assert!(Config::parse(&maximal).is_err());
        for variant in ["reference=x", "candidate=x", "../escape=x", "empty="] {
            let mut invalid = args();
            invalid.extend(["--variant".into(), variant.into()]);
            assert!(Config::parse(&invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn canonical_comparison_covers_seven_fields_and_omits_reporting() -> Result<()> {
        let expected = white();
        let original_hash = state_hash(&expected)?;
        let mut reporting = expected.clone();
        reporting.control.best_chain_id = 999;
        reporting.counters.submitted_batches = 123;
        reporting.validation_max_tile_occupancy = Some(456);
        assert_eq!(state_hash(&reporting)?, original_hash);
        let mut errors = Vec::new();
        compare_state(&reporting, &expected, "final", &mut errors);
        assert!(errors.is_empty());
        for field in 0..7 {
            let mut changed = expected.clone();
            match field {
                0 => changed.parents[0].rng_state += 1,
                1 => changed.offspring.push(expected.parents[0].clone()),
                2 => changed.parent_total_errors[0] += 1,
                3 => changed.parent_framebuffers[0][0] -= 1,
                4 => changed.offspring_errors.push([1, 0]),
                5 => changed.chain_update_flags[0][1] += 1,
                _ => changed.fitness_packed[0] += 1,
            }
            errors.clear();
            compare_state(&changed, &expected, "final", &mut errors);
            assert_eq!(errors.len(), 1);
            assert_ne!(state_hash(&changed)?, original_hash);
        }
        Ok(())
    }

    #[test]
    fn raw_pixel_and_fitness_checks_reject_stale_associations() -> Result<()> {
        let state = white();
        let mut errors = Vec::new();
        check_pixels(&state, &[255; 4], true, &mut errors)?;
        assert!(errors.is_empty());
        for field in 0..4 {
            let mut changed = state.clone();
            match field {
                0 => changed.parent_framebuffers[0][0] = 0xff00_0000,
                1 => changed.parent_total_errors[0] = 1,
                2 => changed.parents[0].polygon_count = 1,
                _ => changed.fitness_packed[0] = 99.0f32.to_bits(),
            }
            errors.clear();
            check_pixels(&changed, &[255; 4], true, &mut errors)?;
            assert!(!errors.is_empty());
        }
        Ok(())
    }

    #[test]
    fn numeric_policy_preserves_f32_integer_rounding() {
        for (integer, expected) in [
            (0, 0.0f32),
            (65535, 65535.0),
            (65536, 65536.0),
            (16_777_217, 16_777_216.0),
            (u32::MAX, 4_294_967_296.0),
        ] {
            assert_eq!(rounded_f32(integer).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn incomplete_work_and_timestamp_samples_are_never_valid() {
        let params = MutationParams::default();
        let mut c = BenchmarkCounters {
            submitted_batches: 1,
            completed_batches: 1,
            submitted_iterations: u64::from(params.gpu_batch_iters),
            completed_iterations: u64::from(params.gpu_batch_iters),
            submitted_evaluations: u64::from(
                params.gpu_batch_iters * params.chain_count * params.lambda,
            ),
            completed_evaluations: u64::from(
                params.gpu_batch_iters * params.chain_count * params.lambda,
            ),
        };
        let mut errors = Vec::new();
        check_work(c, &params, 1, 1, &mut errors);
        assert!(errors.is_empty());
        c.completed_evaluations -= 1;
        check_work(c, &params, 1, 1, &mut errors);
        assert_eq!(errors.len(), 1);
        errors.clear();
        check_work(c, &params, 1, 2, &mut errors);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn source_preflight_rejects_bad_wgsl_utf8_and_oversized_files() -> Result<()> {
        let path = std::env::temp_dir().join(format!("engine-lab-source-{}", uuid::Uuid::new_v4()));
        let mut config = Config::parse(&args())?;
        config.reference_shader = path.clone();
        config.variants[0].1 = path.clone();
        let result = (|| -> Result<()> {
            for stage in [ShaderStage::Bin, ShaderStage::Raster, ShaderStage::Mutate] {
                config.shader_stage = stage;
                for bytes in [
                    b"invalid wgsl".to_vec(),
                    b"@compute @workgroup_size(1) fn main() { let x: u32 = true; }".to_vec(),
                    vec![0xff],
                    vec![b' '; usize::try_from(MAX_SOURCE_BYTES)? + 1],
                ] {
                    fs::write(&path, bytes)?;
                    assert!(read_sources(&config).is_err());
                }
            }
            Ok(())
        })();
        fs::remove_file(path)?;
        result
    }

    #[test]
    fn mutate_sources_and_rows_preserve_stage_and_exact_source_provenance() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("engine-lab-mutate-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory)?;
        let result = (|| -> Result<()> {
            let mut config = Config::parse(&args())?;
            config.shader_stage = ShaderStage::Mutate;
            config.reference_shader = directory.join("reference.wgsl");
            config.variants[0].1 = directory.join("candidate.wgsl");
            let original = include_str!("../src/shaders/mutate.wgsl");
            let candidate = format!("{original}\n// candidate λ\n");
            fs::write(&config.reference_shader, original)?;
            fs::write(&config.variants[0].1, &candidate)?;
            let sources = read_sources(&config)?;
            assert_eq!(sources.len(), 2);
            for (source, expected) in sources.iter().zip([original, candidate.as_str()]) {
                assert_eq!(source.shader_stage, ShaderStage::Mutate);
                assert_eq!(source.wgsl, expected);
                assert_eq!(source.sha256, fixtures::sha256(expected.as_bytes()));
                let serialized = serde_json::to_value(source)?;
                assert_eq!(serialized["shader_stage"], "mutate");
                assert!(serialized.get("wgsl").is_none());
                for timing_eligible in [true, false] {
                    let row = TrialRow::new(
                        &config,
                        source,
                        "ff",
                        "manifest",
                        provenance(),
                        timing_eligible,
                    );
                    let value = serde_json::to_value(row)?;
                    assert_eq!(value["shader_stage"], "mutate");
                    assert_eq!(value["source_sha256"], source.sha256);
                    assert_eq!(value["timing_eligible"], timing_eligible);
                }
            }
            assert_ne!(sources[0].sha256, sources[1].sha256);
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
    }

    #[test]
    fn summary_schema_preserves_raw_rates_and_source_identity() -> Result<()> {
        let config = Config::parse(&args())?;
        let sources = vec![
            Source {
                shader_stage: config.shader_stage,
                name: "reference".into(),
                path: "reference.wgsl".into(),
                sha256: "abc".into(),
                wgsl: "private source".into(),
            },
            Source {
                shader_stage: config.shader_stage,
                name: "candidate".into(),
                path: "candidate.wgsl".into(),
                sha256: "def".into(),
                wgsl: String::new(),
            },
        ];
        let mut summary = Summary {
            schema: SCHEMA,
            status: "running",
            error_count: 0,
            errors: Vec::new(),
            config: &config,
            sources: &sources,
            fixture_manifest_sha256: "manifest".into(),
            warmup_batches: 2,
            timestamp_mode: "stages",
            completed_trials: 0,
            expected_trials: 12,
            completed_diagnostics: 0,
            expected_diagnostics: 0,
            diagnostic_counters: BenchmarkCounters::default(),
            cases: Vec::new(),
            rates: Vec::new(),
        };
        for rate in [3.0, 1.0, 2.0, 4.0] {
            add_rate(&mut summary, "ff", 1, 0, rate);
        }
        summary.completed_diagnostics = 1;
        add_counters(
            &mut summary.diagnostic_counters,
            BenchmarkCounters {
                completed_batches: 8,
                completed_evaluations: 131_072,
                ..BenchmarkCounters::default()
            },
        );
        let value: serde_json::Value = serde_json::from_slice(&serde_json::to_vec(&summary)?)?;
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["sources"][0]["sha256"], "abc");
        assert_eq!(value["sources"][0]["shader_stage"], "bin");
        assert_eq!(value["completed_trials"], 0);
        assert_eq!(value["expected_trials"], 12);
        assert_eq!(value["completed_diagnostics"], 1);
        assert_eq!(
            value["diagnostic_counters"]["completed_evaluations"],
            131_072
        );
        assert!(value["sources"][0].get("wgsl").is_none());
        assert_eq!(
            value["rates"][0]["samples"],
            serde_json::json!([3.0, 1.0, 2.0, 4.0])
        );
        Ok(())
    }
}
