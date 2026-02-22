use serde::{Deserialize, Serialize};

use crate::mutation_params::MutationParams;

/// How a parameter maps between its native range and normalized [0,1].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ParamKind {
    /// Probabilities mapped through log scale (e.g. 1e-4..1.0).
    LogScale { min: f32, max: f32 },
    /// Linear interpolation between min and max.
    Linear { min: f32, max: f32 },
    /// Power-of-2 integer: 2^min_exp .. 2^max_exp.
    Pow2 { min_exp: u32, max_exp: u32 },
    /// Boolean: <0.5 = false, >=0.5 = true.
    Boolean,
}

/// One tunable parameter in the search space.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParamSpec {
    pub name: String,
    pub kind: ParamKind,
    pub enabled: bool,
}

/// A single completed trial.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrialRecord {
    pub trial_number: u32,
    pub label: String,
    pub normalized_params: Vec<f32>,
    pub final_fitness: f32,
    pub improvements_per_sec: f64,
    pub result_id: String,
    pub phase: String,
}

/// Configuration for an auto-tune session.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoTuneConfig {
    pub snapshot_id: String,
    pub drawing_json: String,
    pub duration_secs: u32,
    pub resolution: u32,
    pub replicates: u32,
    pub initial_step_size: f32,
    pub step_decay: f32,
    pub min_step_size: f32,
    pub param_specs: Vec<ParamSpec>,
    pub base_params: MutationParams,
}

/// Phase of the coordinate descent optimizer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum CoordPhase {
    Baseline,
    ProbeHigh,
    ProbeLow,
    Done,
}

/// Result of probing a single parameter.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParamProbeResult {
    pub param_name: String,
    pub base_fitness: f32,
    pub high_fitness: Option<f32>,
    pub low_fitness: Option<f32>,
    pub chosen: String, // "base", "high", "low"
    pub improvement: f32,
}

/// Coordinate descent progress state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoordDescentProgress {
    pub pass: u32,
    pub param_index: usize,
    pub phase: CoordPhase,
    pub step_size: f32,
    pub base_normalized: Vec<f32>,
    pub base_fitness: f32,
    /// Accumulated fitness values for the current probe (for replicates > 1).
    pub replicate_fitnesses: Vec<f32>,
    /// Fitness from the high probe of the current param (after replicates averaged).
    pub high_fitness: Option<f32>,
    /// Fitness from the low probe of the current param (after replicates averaged).
    pub low_fitness: Option<f32>,
    /// Results from completed parameter probes in the current pass.
    pub param_results: Vec<ParamProbeResult>,
}

/// Persisted state for resume.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoTuneState {
    pub config: AutoTuneConfig,
    pub trials: Vec<TrialRecord>,
    pub next_trial_number: u32,
    pub progress: CoordDescentProgress,
}

/// Status broadcast to viewer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoTuneStatus {
    pub running: bool,
    pub trial_number: u32,
    pub total_trials: u32,
    pub phase: String,
    pub best_fitness: f32,
    pub best_trial: u32,
    pub snapshot_id: String,
    pub current_param: Option<String>,
    pub current_direction: Option<String>,
    pub pass: u32,
    pub step_size: f32,
    pub base_fitness: f32,
    pub param_results: Vec<ParamProbeResult>,
    pub config: AutoTuneConfig,
}
