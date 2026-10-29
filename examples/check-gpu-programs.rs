//! Check failed program installation and recovery on the application's main thread.
//! Run `timeout 600s target/release/examples/check-gpu-programs` under the parent
//! benchmark lock. Native construction/compilation requires that outer timeout.
//! `--help` performs no GPU work. The existing rasterization, mutation and lifecycle
//! checkers retain their numerical rules; this checks exact handles and metadata.
//! Dozen teardown crashes in threaded GPU tests, as documented in LEARNINGS.md.

use anyhow::{bail, ensure, Result};
use artgen_backend_rust::gpu_evolver::{
    buffers::GpuDrawingState, pipeline::GpuPipeline, EngineCapacity,
};
use bytemuck::Zeroable;
use wgpu::ComputePipeline;

const USAGE: &str = "check-gpu-programs [--help]\nRun on the main thread under an outer timeout and the benchmark lock.";

fn main() -> Result<()> {
    // Two arguments suffice to reject every unsupported invocation.
    let args: Vec<_> = std::env::args_os().skip(1).take(2).collect();
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    ensure!(args.is_empty(), "{USAGE}");
    check_installation()
}

fn check_installation() -> Result<()> {
    let mut pipeline = futures_lite::future::block_on(GpuPipeline::new(
        EngineCapacity::new(1, 64),
        17,
        19,
        &[0; 17 * 19 * 4],
        &std::iter::once(GpuDrawingState::zeroed()).collect::<Vec<_>>(),
    ));
    ensure!(
        pipeline
            .device
            .limits()
            .max_compute_invocations_per_workgroup
            == 512
    );
    let tile = pipeline.rasterize_wg();
    let num_tiles = pipeline.num_tiles();
    let installed = installed_programs(&pipeline);

    // Raster uses 256 physical lanes for this logical tile. Framebuffer stages
    // require 1024 lanes, exceeding the device's requested 512-lane limit.
    // Rejection after raster/bin compilation must preserve every installed stage.
    let Err(error) = pipeline.try_set_rasterize_wg([32, 32]) else {
        bail!("oversized framebuffer workgroup unexpectedly installed");
    };
    ensure!(
        pipeline.rasterize_wg() == tile,
        "failed install changed tile size"
    );
    ensure!(
        pipeline.num_tiles() == num_tiles,
        "failed install changed tile count"
    );
    ensure!(
        installed_programs(&pipeline) == installed,
        "failed install changed program handles"
    );
    println!("Expected framebuffer-stage rejection: {error}");

    pipeline
        .try_set_rasterize_wg([8, 8])
        .map_err(anyhow::Error::msg)?;
    ensure!(pipeline.rasterize_wg() == [8, 8]);
    ensure!(pipeline.num_tiles() == 9);
    ensure!(pipeline.mutate_pipeline() == &installed[4]);
    ensure!(pipeline.select_pipeline() == &installed[5]);
    println!("GPU program installation checks passed: failed 32x32 preserved all handles and metadata; recovered 8x8");
    Ok(())
}

fn installed_programs(pipeline: &GpuPipeline) -> [ComputePipeline; 6] {
    [
        pipeline.rasterize_error_pipeline().clone(),
        pipeline.bin_polygons_pipeline().clone(),
        pipeline.init_framebuffers_pipeline().clone(),
        pipeline.update_framebuffers_pipeline().clone(),
        pipeline.mutate_pipeline().clone(),
        pipeline.select_pipeline().clone(),
    ]
}
