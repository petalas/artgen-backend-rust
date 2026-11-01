use serde::{Deserialize, Serialize};

use crate::mutation_params::MutationParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkSnapshot {
    pub id: String,
    pub name: String,
    pub drawing_json: String,
    pub fitness: f32,
    pub polygon_count: u32,
    pub created_at: String, // ISO 8601
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkRequest {
    pub drawing_json: String,
    pub params: MutationParams,
    pub duration_secs: u32,
    pub label: String,
    pub resolution: u32,
    pub snapshot_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkSample {
    pub elapsed_secs: f32,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub worst_fitness: f32,
    pub improvements: u64,
    pub total_evals: u64,
    pub evals_per_sec: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub id: String,
    pub snapshot_id: String,
    pub label: String,
    pub start_fitness: f32,
    pub final_fitness: f32,
    pub total_improvements: u64,
    pub total_evals: u64,
    pub duration_secs: u32,
    pub actual_duration_secs: f32,
    pub improvements_per_sec: f64,
    pub samples: Vec<BenchmarkSample>,
    pub chain_count: u32,
    pub lambda: u32,
    pub gpu_batch_iters: u32,
    pub resolution: u32,
    pub params: MutationParams,
    pub pass_timings: Option<PassTimingsResult>,
}

/// GPU per-pass timing averages (milliseconds per iteration).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PassTimingsResult {
    pub mutate_ms: f32,
    pub bin_polygons_ms: f32,
    pub rasterize_error_ms: f32,
    pub select_ms: f32,
}

/// On-disk store (per-project)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkStore {
    pub snapshots: Vec<BenchmarkSnapshot>,
    pub results: Vec<BenchmarkResult>,
}

/// Export envelope — self-contained bundle
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkExport {
    pub project_name: String,
    pub snapshots: Vec<BenchmarkSnapshot>,
    pub results: Vec<BenchmarkResult>,
}
