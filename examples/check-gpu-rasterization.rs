//! Run with `cargo run --release --example check-gpu-rasterization`.
//! Requires the same Vulkan/DX12 GPU features as the evolution engine.
//! Keep GPU checks on the main thread, as in the application.

use artgen_backend_rust::{
    gpu_evolver::GpuEvolver,
    models::{color::Color, drawing::Drawing, point::Point, polygon::Polygon},
    mutation_params::MutationParams,
    settings::PER_POINT_MULTIPLIER,
};

fn layered_drawing() -> Drawing {
    let triangles = [
        vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
        ],
        vec![
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ],
    ];
    Drawing {
        polygons: (0..194)
            .map(|i| Polygon {
                points: triangles[i % 2].clone(),
                color: Color {
                    r: if i >= 192 { 255 } else { 0 },
                    g: 0,
                    b: 0,
                    a: 255,
                },
            })
            .collect(),
        is_dirty: false,
        fitness: 0.0,
    }
}

fn main() {
    // Odd dimensions exercise threads outside the image. At 8x8, the drawing
    // exceeds the shared polygon chunk size of 192 but stays below the tile-bin
    // limit of 256, so all polygons participate in multiple load/barriers.
    let reference = [0, 0, 0, 255].repeat(17 * 19);
    let empty = Drawing {
        polygons: vec![],
        is_dirty: false,
        fitness: 0.0,
    };
    let drawing = layered_drawing();
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(&reference, 17, 19, &empty));
    assert_eq!(evolver.evaluate_chain_fitness(2), 0.0);
    // The final two red triangles must affect the result even at 8x8, where
    // they occupy the second shared polygon chunk after the black base.
    let expected =
        (100.0 * (1.0 - 1.0 / 3.0)) * (1.0 - PER_POINT_MULTIPLIER * drawing.num_points() as f32);
    for workgroup in [[8, 8], [16, 16], [32, 16]] {
        verify_workgroup(&mut evolver, &drawing, &empty, expected, workgroup);
    }
    assert!(evolver.readback_chain(0).polygons.is_empty());
    println!("GPU rasterization checks passed");
}

fn verify_workgroup(
    evolver: &mut GpuEvolver,
    drawing: &Drawing,
    empty: &Drawing,
    expected: f32,
    workgroup: [u32; 2],
) {
    for lambda in [1, 64] {
        let params = MutationParams {
            chain_count: 2,
            lambda,
            gpu_batch_iters: 1,
            rasterize_wg: workgroup,
            ..Default::default()
        };
        let fitness = evolver.prepare_for_benchmark(drawing, &params);
        assert!(
            (fitness - expected).abs() < 0.0001,
            "{workgroup:?}, lambda={lambda}: {fitness}"
        );
        for _ in 0..2 {
            evolver.run_batch(&params, true);
        }
        evolver.flush_pending();
        let incremental = evolver.chain_fitness()[..2].to_vec();
        // Recompute the evolved drawings from scratch before resetting. This
        // validates the cached framebuffers and incremental error totals.
        evolver.evaluate_chain_fitness(2);
        for (before, after) in incremental.iter().zip(evolver.chain_fitness()) {
            assert!(
                (before - after).abs() < 0.001,
                "incremental={before}, full={after}"
            );
        }
        evolver.reinit_chains(empty);
        assert_eq!(evolver.evaluate_chain_fitness(2), 0.0);
    }
}
