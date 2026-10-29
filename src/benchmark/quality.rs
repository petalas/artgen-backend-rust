//! Long-run quality recorder: evolve a blank drawing for a wall-clock budget.
//!
//! Batches run exactly as the app runs them (`run_batch` with timestamps, its
//! one-batch-behind delivery and its global-best rule). The trajectory is built
//! only from values the engine already delivered, so recording never adds a GPU
//! round-trip. Given a seed, the delivered sequence is a function of the batch
//! index; wall-clock time only maps batches to seconds.
use std::{fs, path::Path, time::Instant};

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{Map, Value};

use super::{
    fixtures::LoadedFixture,
    reference::{self, Complexity},
};
use crate::{
    gpu_evolver::{benchmark::BenchmarkProvenance, buffers::drawing_to_gpu, GpuEvolver},
    models::drawing::Drawing,
    mutation_params::MutationParams,
    settings::{PER_POINT_MULTIPLIER, TILE_MAX_POLYS},
};

pub const PROTOCOL_VERSION: u32 = 1;
/// Wall-clock checkpoints recorded at the first batch at or after each second.
pub const FIXED_SECONDS: [f64; 11] = [
    1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 300.0, 600.0,
];
pub const MAX_SECONDS: f64 = 3600.0;
const MAX_WARMUP_BATCHES: u32 = 16;
/// Far above 3600 s of batches at any measured rate (~30 batches/s).
const MAX_BATCHES: u64 = 1 << 24;
/// Every batch is recorded during the first second, up to this many.
const EARLY_SECONDS: f64 = 1.0;
const MAX_EARLY_POINTS: usize = 512;
/// After the first second a point is recorded each time elapsed grows 5%...
const TIME_GROWTH: f64 = 1.05;
/// ...and each time the best remaining error (100 - fitness) shrinks 2%, so
/// time-to-target can be interpolated between nearby points.
const ERROR_SHRINK: f64 = 0.98;
const MIN_ERROR_LEVEL: f64 = 1e-4;
/// Early 512 + powers of two 25 + time grid ~170 + fixed 11 + error grid ~690 + 2.
const MAX_POINTS: usize = 2048;
const MAX_GRID_STEPS: usize = 4096;
/// Largest accepted gap, in fitness percentage points, between the delivered
/// GPU fitness and the scalar renderer's score of the same final drawing. Edge
/// pixels may legitimately differ; a broken configuration differs by whole
/// percents (darwin with lambda 32 reported 99.97% for a drawing near 85%).
pub const MAX_SCALAR_GAP: f64 = 0.05;

#[derive(Clone, Debug)]
pub struct QualitySpec {
    pub case_id: String,
    pub seed: u64,
    pub seconds: f64,
    pub warmup_batches: u32,
    /// Serialized `MutationParams` fields (camelCase) merged onto the defaults.
    pub overrides: Map<String, Value>,
}

/// Best delivered drawing at one moment. `seconds` is measured after the batch
/// call returned; `batches` and `evaluations` count submitted work, which runs
/// one batch ahead of the delivered drawing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Point {
    pub seconds: f64,
    pub batches: u64,
    pub evaluations: u64,
    /// GPU fitness percent as delivered, including the point penalty.
    pub fitness: f64,
    /// `fitness / (1 - PER_POINT_MULTIPLIER * points)`: the error-only fitness.
    pub raw_fitness: f64,
    pub polygons: u32,
    pub points: u32,
}

#[derive(Serialize)]
pub struct QualityRecord {
    pub kind: &'static str,
    pub protocol_version: u32,
    pub case_id: String,
    pub seed: u64,
    pub seconds_budget: f64,
    pub warmup_batches: u32,
    pub width: u32,
    pub height: u32,
    pub overrides: Map<String, Value>,
    pub params: MutationParams,
    pub initial_fitness: f64,
    /// The scalar reference renderer's fitness of the final delivered drawing,
    /// computed after the clock stops.
    pub final_scalar_fitness: f64,
    /// Tile occupancy of the final delivered drawing (`overflowed_tiles` > 0
    /// means the GPU raster silently dropped polygons).
    pub final_complexity: Complexity,
    pub elapsed_seconds: f64,
    pub batches: u64,
    pub evaluations: u64,
    pub evals_per_second: f64,
    pub errors: Vec<String>,
    pub gpu: BenchmarkProvenance,
    pub trajectory: Vec<Point>,
}

/// Merge overrides onto `MutationParams::default()`.
///
/// # Errors
/// Rejects unknown keys, ill-typed values and values that production's
/// `sanitize` would silently change.
pub fn params_with(overrides: &Map<String, Value>) -> Result<MutationParams> {
    let mut value = serde_json::to_value(MutationParams::default())?;
    let fields = value
        .as_object_mut()
        .context("MutationParams is an object")?;
    for (key, override_value) in overrides {
        ensure!(
            fields.contains_key(key),
            "unknown MutationParams field {key} (fields are camelCase)"
        );
        fields.insert(key.clone(), override_value.clone());
    }
    let params: MutationParams = serde_json::from_value(value)?;
    let mut sanitized = params.clone();
    sanitized.sanitize();
    ensure!(
        sanitized == params,
        "params outside the accepted ranges; sanitize would change them"
    );
    Ok(params)
}

/// Decides which batches become trajectory points. Every grid walk is bounded.
struct Schedule {
    early_points: usize,
    next_time: f64,
    next_fixed: usize,
    next_error: f64,
}

impl Schedule {
    fn new(initial_remaining: f64) -> Self {
        let mut schedule = Self {
            early_points: 0,
            next_time: EARLY_SECONDS,
            next_fixed: 0,
            next_error: 100.0,
        };
        schedule.pass_error(initial_remaining);
        schedule
    }

    fn due(&mut self, batches: u64, seconds: f64, remaining: f64) -> bool {
        let mut due = batches.is_power_of_two();
        if seconds < EARLY_SECONDS && self.early_points < MAX_EARLY_POINTS {
            self.early_points += 1;
            due = true;
        }
        if seconds >= self.next_time {
            due = true;
            for _ in 0..MAX_GRID_STEPS {
                if self.next_time > seconds {
                    break;
                }
                self.next_time *= TIME_GROWTH;
            }
        }
        if self.next_fixed < FIXED_SECONDS.len() && seconds >= FIXED_SECONDS[self.next_fixed] {
            due = true;
            self.next_fixed = FIXED_SECONDS.partition_point(|&t| t <= seconds);
        }
        if remaining <= self.next_error {
            due = true;
            self.pass_error(remaining);
        }
        due
    }

    fn pass_error(&mut self, remaining: f64) {
        for _ in 0..MAX_GRID_STEPS {
            if self.next_error < remaining || self.next_error < MIN_ERROR_LEVEL {
                return;
            }
            self.next_error *= ERROR_SHRINK;
        }
    }
}

/// Global best as the app keeps it: replace only on strictly higher fitness.
struct Recorder {
    best: Drawing,
    schedule: Schedule,
    points: Vec<Point>,
}

impl Recorder {
    fn new(initial_fitness: f32) -> Self {
        let best = Drawing {
            polygons: Vec::new(),
            is_dirty: false,
            fitness: initial_fitness,
        };
        let mut points = Vec::with_capacity(MAX_POINTS);
        points.push(point_of(&best, 0.0, 0, 0));
        let schedule = Schedule::new(100.0 - f64::from(initial_fitness));
        Self {
            best,
            schedule,
            points,
        }
    }

    fn deliver(&mut self, delivered: Option<Drawing>) {
        if let Some(drawing) = delivered {
            if drawing.fitness > self.best.fitness {
                self.best = drawing;
            }
        }
    }

    fn observe(&mut self, batches: u64, evaluations: u64, seconds: f64) -> Result<()> {
        let remaining = 100.0 - f64::from(self.best.fitness);
        if self.schedule.due(batches, seconds, remaining) {
            self.push(point_of(&self.best, seconds, batches, evaluations))?;
        }
        Ok(())
    }

    fn push(&mut self, point: Point) -> Result<()> {
        ensure!(
            self.points.len() < MAX_POINTS,
            "trajectory capacity exhausted"
        );
        let last = self
            .points
            .last()
            .context("trajectory starts with a point")?;
        debug_assert!(point.seconds >= last.seconds && point.fitness >= last.fitness);
        ensure!(
            point.batches >= last.batches,
            "trajectory batches went backwards"
        );
        self.points.push(point);
        Ok(())
    }
}

fn point_of(best: &Drawing, seconds: f64, batches: u64, evaluations: u64) -> Point {
    let polygons = u32::try_from(best.polygons.len()).unwrap_or(u32::MAX);
    let points = best
        .polygons
        .iter()
        .map(|polygon| u32::try_from(polygon.points.len()).unwrap_or(u32::MAX))
        .fold(0u32, u32::saturating_add);
    let fitness = f64::from(best.fitness);
    let penalty = 1.0 - f64::from(PER_POINT_MULTIPLIER) * f64::from(points);
    Point {
        seconds,
        batches,
        evaluations,
        fitness,
        raw_fitness: fitness / penalty,
        polygons,
        points,
    }
}

fn validate(spec: &QualitySpec, output: &Path) -> Result<()> {
    ensure!(
        spec.seconds.is_finite() && spec.seconds > 0.0 && spec.seconds <= MAX_SECONDS,
        "seconds must be in (0, {MAX_SECONDS}]"
    );
    ensure!(
        spec.warmup_batches <= MAX_WARMUP_BATCHES,
        "warmup must be 0..{MAX_WARMUP_BATCHES}"
    );
    ensure!(
        !output.join("result.json").exists(),
        "result already exists"
    );
    Ok(())
}

/// Run one blank-start trajectory on a resident engine; writes `result.json`
/// and the final delivered `best.json` outside the clock.
///
/// # Errors
/// Rejects invalid specs and params, failed GPU preparation or drain, and
/// unwritable outputs. Device errors are returned in the record.
pub fn run(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    spec: &QualitySpec,
    output: &Path,
) -> Result<QualityRecord> {
    validate(spec, output)?;
    let params = params_with(&spec.overrides)?;
    fs::create_dir_all(output)?;
    let scopes = evolver.benchmark_error_scopes();
    let result = evolve(evolver, &params, spec);
    let mut errors = futures_lite::future::block_on(scopes.finish());
    let (recorder, initial_fitness, seconds) = result?;
    let last = *recorder
        .points
        .last()
        .context("trajectory has a final point")?;
    let (scalar, complexity) = final_check(&recorder.best, input, &params, &mut errors)?;
    let record = QualityRecord {
        kind: "quality",
        protocol_version: PROTOCOL_VERSION,
        case_id: spec.case_id.clone(),
        seed: spec.seed,
        seconds_budget: spec.seconds,
        warmup_batches: spec.warmup_batches,
        width: input.fixture.width,
        height: input.fixture.height,
        overrides: spec.overrides.clone(),
        params,
        initial_fitness: f64::from(initial_fitness),
        final_scalar_fitness: scalar,
        final_complexity: complexity,
        elapsed_seconds: seconds,
        batches: last.batches,
        evaluations: last.evaluations,
        evals_per_second: last.evaluations as f64 / seconds,
        errors,
        gpu: evolver.benchmark_provenance(),
        trajectory: recorder.points,
    };
    fs::write(
        output.join("best.json"),
        serde_json::to_vec(&recorder.best)?,
    )?;
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(record)
}

/// Score the final delivered drawing with the independent scalar renderer and
/// describe its tile occupancy; a disagreement with the GPU is a run error.
fn final_check(
    best: &Drawing,
    input: &LoadedFixture,
    params: &MutationParams,
    errors: &mut Vec<String>,
) -> Result<(f64, Complexity)> {
    let state = drawing_to_gpu(best, 0);
    let count = usize::try_from(state.polygon_count)?;
    let polygons: Vec<[u32; 4]> = state.polygons[..count].iter().map(|p| p.data).collect();
    let (width, height) = (input.fixture.width, input.fixture.height);
    let scalar = f64::from(reference::evaluate(&polygons, &input.rgba, width, height)?.fitness);
    let complexity = reference::complexity(&polygons, width, height, params.rasterize_wg)?;
    let delivered = f64::from(best.fitness);
    if (scalar - delivered).abs() > MAX_SCALAR_GAP {
        errors.push(format!(
            "delivered fitness {delivered:.5} differs from the scalar renderer's {scalar:.5}; \
             {} of {} tiles exceed the {TILE_MAX_POLYS}-polygon tile bin",
            complexity.overflowed_tiles, complexity.tile_count
        ));
    }
    Ok((scalar, complexity))
}

/// The clock covers ordinary batches and the final drain, nothing else.
fn evolve(
    evolver: &mut GpuEvolver,
    params: &MutationParams,
    spec: &QualitySpec,
) -> Result<(Recorder, f32, f64)> {
    evolver
        .benchmark_set_batch_envelope(params, false)
        .map_err(anyhow::Error::msg)?;
    evolver
        .benchmark_set_validation(false)
        .map_err(anyhow::Error::msg)?;
    let blank = Drawing {
        polygons: Vec::new(),
        is_dirty: false,
        fitness: 0.0,
    };
    let initial = evolver
        .prepare_seeded_benchmark(&blank, params, spec.seed, spec.warmup_batches, true)
        .map_err(anyhow::Error::msg)?;
    let mut recorder = Recorder::new(initial);
    let start = Instant::now();
    let mut batches = 0u64;
    while batches < MAX_BATCHES {
        let delivered = evolver.run_batch(params, true);
        batches += 1;
        let seconds = start.elapsed().as_secs_f64();
        recorder.deliver(delivered);
        recorder.observe(batches, evolver.total_evaluations(), seconds)?;
        if seconds >= spec.seconds {
            break;
        }
    }
    recorder.deliver(evolver.flush_pending());
    let counters = evolver.benchmark_drain().map_err(anyhow::Error::msg)?;
    let seconds = start.elapsed().as_secs_f64();
    ensure!(
        counters.completed_batches == batches,
        "completed batches differ from submitted batches"
    );
    let final_point = point_of(
        &recorder.best,
        seconds,
        batches,
        evolver.total_evaluations(),
    );
    recorder.push(final_point)?;
    Ok((recorder, initial, seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_survive_the_merge_and_unknown_or_clamped_fields_are_rejected() {
        assert_eq!(params_with(&Map::new()).unwrap(), MutationParams::default());
        let set = |json: &str| params_with(&serde_json::from_str(json).unwrap());
        assert_eq!(set(r#"{"crossoverProb":0.0}"#).unwrap().crossover_prob, 0.0);
        assert!(set(r#"{"crossover_prob":0.0}"#)
            .unwrap_err()
            .to_string()
            .contains("unknown"));
        assert!(set(r#"{"lambda":48}"#)
            .unwrap_err()
            .to_string()
            .contains("sanitize"));
        assert!(set(r#"{"lambda":"64"}"#).is_err());
    }

    #[test]
    fn schedule_records_early_batches_grids_and_fixed_seconds() {
        let mut schedule = Schedule::new(30.0);
        // Every batch in the first second; batch 3 is not a power of two.
        assert!(schedule.due(3, 0.5, 30.0));
        // 1.0 s passes the time grid and the fixed list.
        assert!(schedule.due(3, 1.0, 30.0));
        assert!(!schedule.due(3, 1.02, 30.0));
        assert!(schedule.due(3, 1.06, 30.0));
        // A 2% error drop is a point; a smaller one is not.
        assert!(!schedule.due(3, 1.07, 29.9));
        assert!(schedule.due(3, 1.08, 29.3));
        assert!(schedule.due(4, 1.08, 29.3));
        // A jump over several fixed seconds records once and moves past them.
        assert!(schedule.due(3, 25.0, 29.3));
        assert_eq!(schedule.next_fixed, 5);
    }

    #[test]
    fn raw_fitness_removes_the_point_penalty() {
        use crate::models::{color::Color, point::Point as Vertex, polygon::Polygon};
        let triangle = Polygon {
            points: vec![Vertex { x: 0.0, y: 0.0 }; 3],
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
        };
        let best = Drawing {
            polygons: vec![triangle; 1000],
            is_dirty: false,
            fitness: 99.0,
        };
        let point = point_of(&best, 1.0, 1, 1);
        assert_eq!((point.polygons, point.points), (1000, 3000));
        let expected = 99.0 / (1.0 - 3000.0 / 5_000_000.0);
        assert!((point.raw_fitness - expected).abs() < 1e-9);
    }
}
