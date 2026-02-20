use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkRequest {
    pub drawing_json: String,
    pub params: crate::mutation_params::MutationParams,
    pub duration_secs: u32,
    pub label: String,
}

#[derive(Clone, Serialize, Deserialize)]
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

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub label: String,
    pub start_fitness: f32,
    pub final_fitness: f32,
    pub total_improvements: u64,
    pub total_evals: u64,
    pub duration_secs: u32,
    pub improvements_per_sec: f64,
    pub samples: Vec<BenchmarkSample>,
    pub chain_count: u32,
    pub island_count: u32,
    pub lambda: u32,
}
