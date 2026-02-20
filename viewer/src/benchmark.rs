pub use artgen_shared::benchmark::{BenchmarkRequest, BenchmarkResult, BenchmarkSample};

/// UI-only snapshot state (not sent over WebSocket).
#[derive(Clone, Debug)]
pub struct BenchmarkSnapshot {
    pub name: String,
    pub drawing_json: String,
    pub fitness: f32,
    pub polygon_count: u32,
}

/// UI-only progress state (received from WebSocket, not sent back).
#[derive(Clone, Debug, Default)]
pub struct BenchmarkProgress {
    pub label: String,
    pub elapsed_secs: f32,
    pub duration_secs: u32,
    pub best_fitness: f32,
    pub improvements: u64,
}
