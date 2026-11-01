#![feature(portable_simd)]
// Proving wgpu::CommandBuffer: Send for BatchCommands::finish's scoped threads
// nests deeper than the default limit of 128.
#![recursion_limit = "256"]

pub mod engine;
pub mod models;
pub mod settings;
pub mod utils;
pub mod buffer_dimensions;
pub mod texture_wrapper;
pub mod evaluator;
pub mod gpu_evolver;
pub mod auto_tune;
pub mod benchmark;
pub mod mutation_params;
pub mod projects;