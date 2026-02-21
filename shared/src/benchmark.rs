use serde::{Deserialize, Serialize};

use crate::mutation_params::MutationParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkRequest {
    pub drawing_json: String,
    pub params: MutationParams,
    pub duration_secs: u32,
    pub label: String,
    #[serde(default)]
    pub resolution: u32,
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

fn default_lambda() -> u32 {
    crate::settings::GPU_DEFAULT_LAMBDA
}

fn default_gpu_batch_iters() -> u32 {
    crate::settings::GPU_DEFAULT_BATCH_ITERS
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub label: String,
    pub start_fitness: f32,
    pub final_fitness: f32,
    pub total_improvements: u64,
    pub total_evals: u64,
    pub duration_secs: u32,
    #[serde(default)]
    pub actual_duration_secs: f32,
    pub improvements_per_sec: f64,
    pub samples: Vec<BenchmarkSample>,
    pub chain_count: u32,
    #[serde(default = "default_lambda")]
    pub lambda: u32,
    #[serde(default = "default_gpu_batch_iters")]
    pub gpu_batch_iters: u32,
    #[serde(default)]
    pub resolution: u32,
}
