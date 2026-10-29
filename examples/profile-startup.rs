//! Run with `cargo run --release --example profile-startup -- IMAGE [DRAWING]`.
//! Reports initialization time and steady-state evolution after warmup. A
//! ten-second initialization budget makes slow startup an explicit failure.

use std::{path::PathBuf, time::Instant};

use artgen_backend_rust::{
    engine::Rasterizer,
    gpu_evolver::{EngineCapacity, GpuEvolver},
    models::drawing::Drawing,
    mutation_params::MutationParams,
    projects,
};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let image = PathBuf::from(args.next().expect("reference image path required"));
    let start = Instant::now();
    let bytes = std::fs::read(image)?;
    let (rgba, width, height) =
        projects::load_and_normalize_image(&bytes, Some(384)).map_err(anyhow::Error::msg)?;
    let drawing = args
        .next()
        .map(|path| Drawing::from_file(&path))
        .unwrap_or_else(Drawing::new_random);
    let decoded = start.elapsed();
    eprintln!("startup: image decoded");
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &rgba,
        width as u32,
        height as u32,
        &drawing,
        // As the application constructs it.
        EngineCapacity::ADJUSTABLE,
    ));
    let created = start.elapsed();
    eprintln!("startup: GPU created");
    let fitness = evolver.evaluate_chain_fitness(&MutationParams::default());
    let evaluated = start.elapsed();
    let mut preview = vec![0; width * height * 4];
    drawing.draw(&mut preview, width, height, Rasterizer::HalfSpace);
    let ready = start.elapsed();
    assert!(fitness.is_finite());
    let evolution = profile_evolution(&mut evolver, &drawing);
    println!(
        "{}",
        serde_json::json!({
            "decode_ms": decoded.as_secs_f64() * 1000.0,
            "gpu_create_ms": (created - decoded).as_secs_f64() * 1000.0,
            "initial_fitness_ms": (evaluated - created).as_secs_f64() * 1000.0,
            "preview_ms": (ready - evaluated).as_secs_f64() * 1000.0,
            "ready_ms": ready.as_secs_f64() * 1000.0,
            "fitness": fitness,
            "width": width,
            "height": height,
            "evolution": evolution,
        })
    );
    anyhow::ensure!(
        ready.as_secs() < 10,
        "startup exceeded the 10-second loading budget"
    );
    Ok(())
}

fn profile_evolution(evolver: &mut GpuEvolver, drawing: &Drawing) -> serde_json::Value {
    let params = MutationParams::default();
    let initial_fitness = evolver.prepare_for_benchmark(drawing, &params);
    let start = Instant::now();
    for _ in 0..8 {
        evolver.run_batch(&params, true);
    }
    evolver.flush_pending();
    let elapsed = start.elapsed().as_secs_f64();
    let evaluations = evolver.total_evaluations();
    let best = evolver
        .chain_fitness()
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(evaluations > 0 && best.is_finite() && best >= initial_fitness);
    serde_json::json!({
        "evals_per_sec": evaluations as f64 / elapsed,
        "evaluations": evaluations,
        "elapsed_secs": elapsed,
        "initial_fitness": initial_fitness,
        "best_fitness": best,
    })
}
