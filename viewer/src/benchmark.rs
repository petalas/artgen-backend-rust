pub use artgen_shared::benchmark::{
    BenchmarkExport, BenchmarkRequest, BenchmarkResult, BenchmarkSnapshot,
};

/// UI-only progress state (received from WebSocket, not sent back).
#[derive(Clone, Debug, Default)]
pub struct BenchmarkProgress {
    pub label: String,
    pub elapsed_secs: f32,
    pub duration_secs: u32,
    pub best_fitness: f32,
    pub improvements: u64,
}
