//! Run with `cargo run --release --example check-gpu-rasterization`.
//! Requires the same Vulkan/DX12 GPU features as the evolution engine.
//! Keep GPU checks on the main thread, as in the application.

use artgen_backend_rust::{
    gpu_evolver::{benchmark::BenchmarkSnapshot, GpuEvolver},
    models::{color::Color, drawing::Drawing, point::Point, polygon::Polygon},
    mutation_params::MutationParams,
    settings::{MAX_ERROR_PER_PIXEL, PER_POINT_MULTIPLIER},
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
    verify_visible_layer_acceptance();
    // Odd dimensions exercise threads outside the image. At 8x8, the drawing
    // crosses the old 192-entry chunk boundary but stays below the tile-bin
    // limit of 256, so late layers exercise the complete loaded prefix.
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
    // they follow the old chunk boundary after the black base.
    let expected =
        (100.0 * (1.0 - 1.0 / 3.0)) * (1.0 - PER_POINT_MULTIPLIER * drawing.num_points() as f32);
    for workgroup in [[8, 8], [16, 8], [16, 16], [32, 8], [32, 16]] {
        verify_workgroup(&mut evolver, &drawing, &empty, expected, workgroup);
    }
    assert!(evolver.readback_chain(0).polygons.is_empty());
    drop(evolver);
    verify_mixed_reference(&empty);
    verify_full_recompute_after_evolution();
    verify_shortcuts_near_polygon_capacity();
    println!("GPU rasterization checks passed");
}

fn no_op_params() -> MutationParams {
    MutationParams {
        add_polygon_prob: 0.0,
        remove_polygon_prob: 0.0,
        reorder_polygon_prob: 0.0,
        offset_polygon_prob: 0.0,
        move_point_prob: 0.0,
        remove_point_prob: 0.0,
        micro_adjust_prob: 0.0,
        change_color_prob: 0.0,
        adjust_brightness_prob: 0.0,
        adjust_saturation_prob: 0.0,
        scale_polygon_prob: 0.0,
        rotate_polygon_prob: 0.0,
        adjacent_swap_prob: 0.0,
        merge_polygon_prob: 0.0,
        clone_polygon_prob: 0.0,
        medium_move_prob: 0.0,
        swap_colors_prob: 0.0,
        crossover_prob: 0.0,
        micro_adjust_delta: 0.0,
        min_alpha: 0,
        max_alpha: 255,
        min_polygons: 1,
        max_polygons: 2,
        chain_count: 2,
        lambda: 1,
        gpu_batch_iters: 1,
        single_mutation_mode: false,
        adaptive_mutation: false,
        ..Default::default()
    }
}

fn visible_small_layer() -> Drawing {
    let mut drawing = colored_triangle(255);
    drawing.polygons[0].color = Color {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    drawing.polygons.push(Polygon {
        points: vec![
            Point { x: 0.499, y: 0.499 },
            Point { x: 0.501, y: 0.499 },
            Point { x: 0.5, y: 0.501 },
        ],
        color: Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        },
    });
    drawing
}

fn verify_visible_layer_acceptance() {
    // This layer has normalized twice-area 4e-6 but covers the center pixel.
    // A neutral acceptance must preserve the evaluated geometry, cached error
    // and point penalty. Darwin's first traced failure was iteration 2997.
    let mut reference = [255, 255, 255, 255].repeat(17 * 19);
    reference[(9 * 17 + 8) * 4..(9 * 17 + 8) * 4 + 3].fill(0);
    let drawing = visible_small_layer();
    let params = no_op_params();
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(&reference, 17, 19, &drawing));
    evolver
        .prepare_seeded_benchmark(&drawing, &params, 7, 0, true)
        .expect("visible-layer preparation failed");
    let initial = assert_cached_byte_l1(&mut evolver, &reference, "visible layer, initial");
    assert_eq!(initial.parent_total_errors, [0, 0]);
    assert!(initial.parents.iter().all(|p| p.polygon_count == 2));
    let mut accepted = false;
    for step in 1..=16 {
        let _best_drawing = evolver.run_batch(&params, true);
        let state = evolver
            .benchmark_snapshot(2, 1)
            .expect("visible-layer snapshot failed");
        accepted = state.chain_update_flags.iter().any(|flags| flags[0] == 1);
        println!(
            "visible-layer neutral step={step}, accepted={accepted}, counts={:?}",
            state
                .parents
                .iter()
                .map(|p| p.polygon_count)
                .collect::<Vec<_>>()
        );
        let checked = assert_cached_byte_l1(&mut evolver, &reference, "visible layer, after no-op");
        assert_eq!(checked.parent_total_errors, [0, 0]);
        assert_no_op_state(&initial, &checked);
        if accepted {
            break;
        }
    }
    assert!(
        accepted,
        "seeded check must exercise neutral acceptance within 16 steps"
    );
    let _best_fitness = evolver.evaluate_chain_fitness(2);
    let full = assert_cached_byte_l1(&mut evolver, &reference, "visible layer, full recompute");
    assert_eq!(full.parent_total_errors, [0, 0]);
    assert_no_op_state(&initial, &full);
}

fn assert_no_op_state(initial: &BenchmarkSnapshot, actual: &BenchmarkSnapshot) {
    assert_eq!(initial.parents.len(), actual.parents.len());
    for (before, after) in initial.parents.iter().zip(&actual.parents) {
        assert_eq!(
            after.polygon_count, 2,
            "visible layer must survive acceptance"
        );
        assert_eq!(after.polygons, before.polygons, "no-op geometry changed");
        assert_eq!(
            after.fitness_bits, before.fitness_bits,
            "no-op score changed"
        );
    }
}

fn colored_triangle(alpha: u8) -> Drawing {
    Drawing {
        polygons: vec![Polygon {
            points: vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 0.0, y: 1.0 },
            ],
            color: Color {
                r: 26,
                g: 102,
                b: 178,
                a: alpha,
            },
        }],
        is_dirty: false,
        fitness: 0.0,
    }
}

fn verify_mixed_reference(empty: &Drawing) {
    // Every RGB channel visits all 256 reference bytes. Black references cannot
    // expose subinteger error from reconstructing bytes through a UNORM sample.
    let reference: Vec<u8> = (0..17 * 19)
        .flat_map(|i| {
            let byte = (i % 256) as u8;
            [
                byte,
                byte.wrapping_add(85),
                byte.wrapping_add(170),
                byte.wrapping_add(1),
            ]
        })
        .collect();
    let opaque = colored_triangle(255);
    let translucent = colored_triangle(128);
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(&reference, 17, 19, empty));
    for workgroup in [[8, 8], [16, 8], [16, 16], [32, 8], [32, 16]] {
        let params = MutationParams {
            chain_count: 2,
            lambda: 1,
            gpu_batch_iters: 1,
            rasterize_wg: workgroup,
            ..Default::default()
        };
        for (name, drawing) in [
            ("empty", empty),
            ("opaque", &opaque),
            ("translucent", &translucent),
        ] {
            let context = format!("{workgroup:?}, {name}");
            evolver
                .prepare_seeded_benchmark(drawing, &params, 0x0123_4567_89ab_cdef, 0, false)
                .expect("mixed-reference preparation failed");
            let initialized = assert_cached_byte_l1(&mut evolver, &reference, &context);
            evolver.evaluate_chain_fitness(2);
            let evaluated = assert_cached_byte_l1(&mut evolver, &reference, &context);
            assert_eq!(
                initialized.parent_total_errors, evaluated.parent_total_errors,
                "{context}"
            );
            for (errors, total) in evaluated
                .offspring_errors
                .iter()
                .zip(&evaluated.parent_total_errors)
            {
                // Full evaluation visits every pixel: both new and cached old
                // error accumulators must equal the framebuffer's byte L1.
                assert_eq!(*errors, [*total, *total], "{context}: new/old errors");
            }
        }
    }
}

fn assert_cached_byte_l1(
    evolver: &mut GpuEvolver,
    reference: &[u8],
    context: &str,
) -> BenchmarkSnapshot {
    assert_eq!(reference.len(), 17 * 19 * 4);
    assert_chains_cached_byte_l1(evolver, reference, [2, 1], context)
}

/// Every cached parent framebuffer's byte L1 from `reference` equals its cached
/// total, and its fitness follows. `[chains, lambda]` is the run's workload.
fn assert_chains_cached_byte_l1(
    evolver: &mut GpuEvolver,
    reference: &[u8],
    [chains, lambda]: [u32; 2],
    context: &str,
) -> BenchmarkSnapshot {
    let snapshot = evolver.benchmark_snapshot(chains, lambda).expect("snapshot failed");
    assert_eq!(snapshot.parent_framebuffers.len(), chains as usize);
    for (chain, pixels) in snapshot.parent_framebuffers.iter().enumerate() {
        assert_eq!(pixels.len() * 4, reference.len());
        let raw_error: u32 = pixels
            .iter()
            .zip(reference.chunks_exact(4))
            .map(|(pixel, rgba)| {
                let rgb = pixel.to_le_bytes();
                (0..3)
                    .map(|channel| u32::from(rgb[channel].abs_diff(rgba[channel])))
                    .sum::<u32>()
            })
            .sum();
        assert_eq!(
            snapshot.parent_total_errors[chain], raw_error,
            "{context}, chain {chain}: cached total"
        );
        let mut expected =
            100.0 * (1.0 - raw_error as f32 / (MAX_ERROR_PER_PIXEL * pixels.len() as f32));
        expected -=
            expected * PER_POINT_MULTIPLIER * (snapshot.parents[chain].polygon_count * 3) as f32;
        let fitness = f32::from_bits(snapshot.parents[chain].fitness_bits);
        assert!(
            (fitness - expected).abs() < 0.0001,
            "{context}, chain {chain}: fitness={fitness}, expected={expected}"
        );
    }
    snapshot
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

fn scattered_triangles(count: usize) -> Drawing {
    // Small triangles on a grid, so a changed polygon meets few raster tiles.
    let side = (count as f32).sqrt().ceil() as usize;
    let cell = 1.0 / side as f32;
    Drawing {
        polygons: (0..count)
            .map(|i| {
                let (x, y) = ((i % side) as f32 * cell, (i / side) as f32 * cell);
                let shade = (i * 37 % 256) as u8;
                Polygon {
                    points: vec![
                        Point { x, y },
                        Point { x: x + cell, y },
                        Point { x, y: y + cell },
                    ],
                    color: Color {
                        r: shade,
                        g: 255 - shade,
                        b: 128,
                        a: 160,
                    },
                }
            })
            .collect(),
        is_dirty: false,
        fitness: 0.0,
    }
}

fn verify_full_recompute_after_evolution() {
    // A full recompute after accepted winners must match the incremental
    // fitness of every chain. Accepted chains keep the winner's change record,
    // which the batch shortcuts read; the full evaluation must not.
    const SIDE: u32 = 96;
    const CHAINS: u32 = 4;
    let reference: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            [(x * 255 / SIDE) as u8, (y * 255 / SIDE) as u8, 96, 255]
        })
        .collect();
    let drawing = scattered_triangles(64);
    let params = MutationParams {
        chain_count: CHAINS,
        gpu_batch_iters: 16,
        ..Default::default()
    };
    let mut evolver =
        futures_lite::future::block_on(GpuEvolver::new(&reference, SIDE, SIDE, &drawing));
    evolver
        .prepare_seeded_benchmark(&drawing, &params, 11, 0, true)
        .expect("full-recompute preparation failed");
    let start = evolver.chain_fitness()[..CHAINS as usize].to_vec();
    let mut incremental = start.clone();
    for _ in 0..64 {
        evolver.run_batch(&params, true);
        evolver.flush_pending();
        incremental = evolver.chain_fitness()[..CHAINS as usize].to_vec();
        if incremental.iter().zip(&start).all(|(now, before)| now > before) {
            break;
        }
    }
    assert!(
        incremental.iter().zip(&start).all(|(now, before)| now > before),
        "every chain must accept a winner: start={start:?}, now={incremental:?}"
    );
    evolver.evaluate_chain_fitness(CHAINS);
    for (chain, (before, after)) in incremental.iter().zip(evolver.chain_fitness()).enumerate() {
        assert!(
            (before - after).abs() < 0.001,
            "chain {chain}: incremental={before}, full={after}"
        );
    }
}

fn verify_shortcuts_near_polygon_capacity() {
    // Parents of up to 1000 polygons: select's chain change record no longer
    // lives in polygon slots 996..999, so the framebuffer update keeps its
    // unchanged-pixel shortcut at capacity. Accepted winners must leave cached
    // framebuffers, cached totals and a full recompute in exact agreement.
    const SIDE: u32 = 96;
    const CHAINS: u32 = 4;
    let reference: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            let shade = |v: u32| u8::try_from(v * 255 / SIDE).unwrap();
            [shade(x), 160, shade(y), 255]
        })
        .collect();
    let drawing = scattered_triangles(999);
    let params = MutationParams {
        chain_count: CHAINS,
        gpu_batch_iters: 16,
        ..Default::default()
    };
    assert_eq!(params.max_polygons, 1000);
    let workload = [CHAINS, params.lambda];
    let mut evolver =
        futures_lite::future::block_on(GpuEvolver::new(&reference, SIDE, SIDE, &drawing));
    evolver
        .prepare_seeded_benchmark(&drawing, &params, 5, 0, true)
        .expect("capacity preparation failed");
    let start = evolver.chain_fitness()[..CHAINS as usize].to_vec();
    let mut improved = false;
    for _ in 0..32 {
        evolver.run_batch(&params, true);
        evolver.flush_pending();
        let now = &evolver.chain_fitness()[..CHAINS as usize];
        improved = now.iter().zip(&start).all(|(now, before)| now > before);
        if improved {
            break;
        }
    }
    assert!(improved, "every chain must accept a winner near capacity");
    let cached = assert_chains_cached_byte_l1(&mut evolver, &reference, workload, "near capacity");
    let counts: Vec<_> = cached.parents.iter().map(|p| p.polygon_count).collect();
    assert!(counts.iter().all(|&count| count > 996), "counts {counts:?}");
    evolver.evaluate_chain_fitness(CHAINS);
    let full = assert_chains_cached_byte_l1(&mut evolver, &reference, workload, "near capacity, full");
    assert_eq!(
        cached.parent_total_errors, full.parent_total_errors,
        "incremental and full totals near capacity (counts {counts:?})"
    );
    println!("near-capacity shortcut check: counts {counts:?}, totals {:?}", full.parent_total_errors);
}
