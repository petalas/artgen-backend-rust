//! Run with `cargo run --release --example check-gpu-rasterization`.
//! Requires the same Vulkan/DX12 GPU features as the evolution engine.
//! Keep GPU checks on the main thread, as in the application.

use artgen_backend_rust::{
    benchmark::reference,
    gpu_evolver::{
        benchmark::BenchmarkSnapshot,
        buffers::drawing_to_gpu,
        operators::{Operator, OperatorStats, OPERATOR_COUNT},
        EngineCapacity, GpuEvolver,
    },
    models::{color::Color, drawing::Drawing, point::Point, polygon::Polygon},
    mutation_params::{MutationParams, PointPenalty},
    settings::{MAX_ERROR_PER_PIXEL, MAX_PER_POINT_PENALTY, PER_POINT_PENALTY, POLYGON_CAPACITY, TILE_MAX_POLYS},
};

fn layered_drawing() -> Drawing {
    let triangles = [
        [
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
        ],
        [
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ],
    ];
    Drawing {
        polygons: (0..194)
            .map(|i| Polygon {
                points: triangles[i % 2],
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
    verify_visible_layer_acceptance(PointPenalty { per_point: PER_POINT_PENALTY });
    // The penalty is a parameter: the host's full evaluation and select's
    // compute_fitness must both apply it.
    verify_visible_layer_acceptance(PointPenalty { per_point: 16.0 * PER_POINT_PENALTY });
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
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        17,
        19,
        &empty,
        EngineCapacity::new(2, 64),
    ));
    assert_eq!(evolver.evaluate_chain_fitness(&two_chains()), 0.0);
    // The final two red triangles must affect the result even at 8x8, where
    // they follow the old chunk boundary after the black base.
    let expected =
        (100.0 * (1.0 - 1.0 / 3.0)) * (1.0 - PER_POINT_PENALTY * drawing.num_points() as f32);
    for workgroup in [[8, 8], [16, 8], [16, 16], [32, 8], [32, 16]] {
        verify_workgroup(&mut evolver, &drawing, &empty, expected, workgroup);
    }
    assert!(evolver.readback_chain(0).polygons.is_empty());
    drop(evolver);
    verify_mixed_reference(&empty);
    verify_full_recompute_after_evolution();
    // The default search cap, then the storage capacity above it.
    verify_shortcuts_near_polygon_capacity(1000);
    verify_shortcuts_near_polygon_capacity(u32::try_from(POLYGON_CAPACITY).unwrap());
    verify_tile_list_overflow();
    verify_operator_stats();
    verify_tile_groups();
    println!("GPU rasterization checks passed");
}

/// Default parameters on two chains, for full evaluations.
fn two_chains() -> MutationParams {
    MutationParams { chain_count: 2, ..Default::default() }
}

fn no_op_params() -> MutationParams {
    MutationParams {
        add_polygon_prob: 0.0,
        reorder_polygon_prob: 0.0,
        offset_polygon_prob: 0.0,
        move_point_prob: 0.0,
        micro_adjust_prob: 0.0,
        micro_color_prob: 0.0,
        change_color_prob: 0.0,
        adjust_brightness_prob: 0.0,
        adjust_saturation_prob: 0.0,
        scale_polygon_prob: 0.0,
        rotate_polygon_prob: 0.0,
        adjacent_swap_prob: 0.0,
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
        points: [
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

fn verify_visible_layer_acceptance(penalty: PointPenalty) {
    // This layer has normalized twice-area 4e-6 but covers the center pixel.
    // A neutral acceptance must preserve the evaluated geometry, cached error
    // and point penalty. Darwin's first traced failure was iteration 2997.
    let mut reference = [255, 255, 255, 255].repeat(17 * 19);
    reference[(9 * 17 + 8) * 4..(9 * 17 + 8) * 4 + 3].fill(0);
    let drawing = visible_small_layer();
    let params = MutationParams {
        per_point_penalty: penalty.per_point,
        ..no_op_params()
    };
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        17,
        19,
        &drawing,
        EngineCapacity::for_params(&params),
    ));
    evolver
        .prepare_seeded_benchmark(&drawing, &params, 7, 0, true)
        .expect("visible-layer preparation failed");
    let initial =
        assert_cached_byte_l1(&mut evolver, &reference, penalty, "visible layer, initial");
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
        let checked = assert_cached_byte_l1(
            &mut evolver,
            &reference,
            penalty,
            "visible layer, after no-op",
        );
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
    let _best_fitness = evolver.evaluate_chain_fitness(&params);
    let full = assert_cached_byte_l1(
        &mut evolver,
        &reference,
        penalty,
        "visible layer, full recompute",
    );
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
            points: [
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
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        17,
        19,
        empty,
        EngineCapacity::new(2, 1),
    ));
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
            let penalty = params.point_penalty();
            let initialized = assert_cached_byte_l1(&mut evolver, &reference, penalty, &context);
            evolver.evaluate_chain_fitness(&params);
            let evaluated = assert_cached_byte_l1(&mut evolver, &reference, penalty, &context);
            assert_eq!(
                initialized.parent_total_errors, evaluated.parent_total_errors,
                "{context}"
            );
            // The evaluation's sums sit in the first offspring's slots; it
            // must clear them, or the next batch scores those offspring off
            // by the difference of its new and old sums.
            for errors in &evaluated.offspring_errors[..evaluated.parent_total_errors.len()] {
                assert_eq!(*errors, [0, 0], "{context}: evaluation left error sums");
            }
        }
    }
}

fn assert_cached_byte_l1(
    evolver: &mut GpuEvolver,
    reference: &[u8],
    penalty: PointPenalty,
    context: &str,
) -> BenchmarkSnapshot {
    assert_eq!(reference.len(), 17 * 19 * 4);
    assert_chains_cached_byte_l1(evolver, reference, [2, 1], penalty, context)
}

/// Every cached parent framebuffer's byte L1 from `reference` equals its cached
/// total, and its fitness follows with the run's penalty.
/// `[chains, lambda]` is the run's workload.
fn assert_chains_cached_byte_l1(
    evolver: &mut GpuEvolver,
    reference: &[u8],
    [chains, lambda]: [u32; 2],
    penalty: PointPenalty,
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
        let expected = penalty.apply(
            100.0 * (1.0 - raw_error as f32 / (MAX_ERROR_PER_PIXEL * pixels.len() as f32)),
            snapshot.parents[chain].polygon_count,
        );
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
        evolver.evaluate_chain_fitness(&params);
        for (before, after) in incremental.iter().zip(evolver.chain_fitness()) {
            assert!(
                (before - after).abs() < 0.001,
                "incremental={before}, full={after}"
            );
        }
        evolver.reinit_chains(empty);
        assert_eq!(evolver.evaluate_chain_fitness(&params), 0.0);
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
                    points: [
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
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        SIDE,
        SIDE,
        &drawing,
        EngineCapacity::for_params(&params),
    ));
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
    evolver.evaluate_chain_fitness(&params);
    for (chain, (before, after)) in incremental.iter().zip(evolver.chain_fitness()).enumerate() {
        assert!(
            (before - after).abs() < 0.001,
            "chain {chain}: incremental={before}, full={after}"
        );
    }
}

fn verify_shortcuts_near_polygon_capacity(max_polygons: u32) {
    // Parents one polygon below the cap: select's chain change record no longer
    // lives in polygon slots 996..999, so the framebuffer update keeps its
    // unchanged-pixel shortcut at capacity. Accepted winners must leave cached
    // framebuffers, cached totals and a full recompute in exact agreement.
    // At the storage capacity, crossover's keep bits span more than 32 words.
    const SIDE: u32 = 96;
    const CHAINS: u32 = 4;
    let reference: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            let shade = |v: u32| u8::try_from(v * 255 / SIDE).unwrap();
            [shade(x), 160, shade(y), 255]
        })
        .collect();
    let drawing = scattered_triangles(max_polygons as usize - 1);
    let params = MutationParams {
        chain_count: CHAINS,
        gpu_batch_iters: 16,
        max_polygons,
        ..Default::default()
    };
    let penalty = params.point_penalty();
    let workload = [CHAINS, params.lambda];
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        SIDE,
        SIDE,
        &drawing,
        EngineCapacity::for_params(&params),
    ));
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
    let cached =
        assert_chains_cached_byte_l1(&mut evolver, &reference, workload, penalty, "near capacity");
    let counts: Vec<_> = cached.parents.iter().map(|p| p.polygon_count).collect();
    assert!(counts.iter().all(|&count| count > max_polygons - 4), "counts {counts:?}");
    assert!(counts.iter().all(|&count| count <= max_polygons), "counts {counts:?}");
    evolver.evaluate_chain_fitness(&params);
    let full = assert_chains_cached_byte_l1(
        &mut evolver,
        &reference,
        workload,
        penalty,
        "near capacity, full",
    );
    assert_eq!(
        cached.parent_total_errors, full.parent_total_errors,
        "incremental and full totals near capacity (counts {counts:?})"
    );
    println!(
        "near-capacity shortcut check: counts {counts:?}, totals {:?}",
        full.parent_total_errors
    );
}

/// A triangle covering normalized x in [x0, x1] over the full image height.
fn column_triangle(x0: f32, x1: f32, color: Color) -> Polygon {
    Polygon {
        points: [
            Point { x: x0, y: 0.0 },
            Point { x: x1, y: 0.0 },
            Point { x: x0, y: 1.0 },
        ],
        color,
    }
}

/// The scalar renderer's fitness for packed polygons.
fn reference_fitness(polygons: &[[u32; 4]], reference: &[u8], [w, h]: [u32; 2]) -> f32 {
    reference::evaluate(polygons, reference, w, h, MutationParams::default().point_penalty())
        .expect("reference evaluation")
        .fitness
}

fn packed(drawing: &Drawing) -> Vec<[u32; 4]> {
    let state = drawing_to_gpu(drawing, 0);
    state.polygons[..state.polygon_count as usize].iter().map(|p| p.data).collect()
}

fn verify_tile_list_overflow() {
    // The first column of 16x16 tiles must stay white; the rest of the
    // reference is black. 300 triangles lie in that column: 256 transparent,
    // then 44 opaque black. Its tile lists hold 300 entries, so the raster
    // would drop the black ones and score white. The uploaded parents must
    // hold only the 256 triangles it renders.
    const IMAGE: [u32; 2] = [64, 32];
    const TILE: [u32; 2] = [16, 16];
    const CHAINS: u32 = 2;
    let reference: Vec<u8> = (0..IMAGE[0] * IMAGE[1])
        .flat_map(|i| if i % IMAGE[0] < TILE[0] { [255, 255, 255, 255] } else { [0, 0, 0, 255] })
        .collect();
    let clear = Color { r: 0, g: 0, b: 0, a: 0 };
    let black = Color { r: 0, g: 0, b: 0, a: 255 };
    let drawing = Drawing {
        polygons: (0..300)
            .map(|i| column_triangle(0.0, 0.2, if i < 256 { clear } else { black }))
            .collect(),
        is_dirty: false,
        fitness: 0.0,
    };
    let full = packed(&drawing);
    let overflow = reference::complexity(&full, IMAGE[0], IMAGE[1], TILE).unwrap();
    assert_eq!((overflow.overflowed_tiles, overflow.tile_overlaps.max), (2, 300));

    // Only additions of larger triangles, so offspring keep landing in the full
    // column. Each must be rejected; the others may darken the black area.
    let params = MutationParams {
        add_polygon_prob: 1.0,
        new_point_max_distance: 0.3,
        min_polygons: 1,
        max_polygons: 1000,
        chain_count: CHAINS,
        lambda: 64,
        gpu_batch_iters: 16,
        rasterize_wg: TILE,
        ..no_op_params()
    };
    let workload = [CHAINS, params.lambda];
    let penalty = params.point_penalty();
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        IMAGE[0],
        IMAGE[1],
        &drawing,
        EngineCapacity::for_params(&params),
    ));
    let start = evolver
        .prepare_seeded_benchmark(&drawing, &params, 3, 0, true)
        .expect("overflow preparation failed");
    let initial =
        assert_chains_cached_byte_l1(&mut evolver, &reference, workload, penalty, "fitted start");
    for parent in &initial.parents {
        assert_eq!(parent.polygons, full[..TILE_MAX_POLYS as usize], "fitted parent");
    }
    let fitted_score = reference_fitness(&full[..TILE_MAX_POLYS as usize], &reference, IMAGE);
    assert!((start - fitted_score).abs() < 1e-4, "start {start}, scalar {fitted_score}");
    assert!(reference_fitness(&full, &reference, IMAGE) < fitted_score - 10.0);

    let mut delivered = None;
    for _ in 0..32 {
        delivered = evolver.run_batch(&params, true).or(delivered);
    }
    delivered = evolver.flush_pending().or(delivered);
    let after = assert_chains_cached_byte_l1(&mut evolver, &reference, workload, penalty, "evolved");
    for (chain, parent) in after.parents.iter().enumerate() {
        assert!(parent.polygon_count > TILE_MAX_POLYS, "chain {chain} accepted no addition");
        let lists = reference::complexity(&parent.polygons, IMAGE[0], IMAGE[1], TILE).unwrap();
        assert_eq!(lists.overflowed_tiles, 0, "chain {chain} accepted an overflowing offspring");
        let scalar = reference_fitness(&parent.polygons, &reference, IMAGE);
        let gpu = f32::from_bits(parent.fitness_bits);
        assert!((gpu - scalar).abs() < 1e-4, "chain {chain}: GPU {gpu}, scalar {scalar}");
    }
    let delivered = delivered.expect("an improvement was delivered");
    let scalar = reference_fitness(&packed(&delivered), &reference, IMAGE);
    assert!(
        (delivered.fitness - scalar).abs() < 1e-4,
        "delivered {}, scalar {scalar}",
        delivered.fitness
    );
    let incremental = evolver.chain_fitness().to_vec();
    evolver.evaluate_chain_fitness(&params);
    for (before, after) in incremental.iter().zip(evolver.chain_fitness()) {
        assert!((before - after).abs() < 1e-4, "incremental={before}, full={after}");
    }
    println!(
        "tile-list overflow check: counts {:?}, delivered {:.4} ({} triangles)",
        after.parents.iter().map(|p| p.polygon_count).collect::<Vec<_>>(),
        delivered.fitness,
        delivered.polygons.len()
    );
}

/// Operator stats count every evaluated offspring once, under the operator that
/// produced it, and every accepted winner once, as improved when it raised its
/// chain's fitness and neutral when it kept it; the improvements' gains sum to
/// the fitness they added.
fn verify_operator_stats() {
    const SIDE: u32 = 96;
    const CHAINS: u32 = 4;
    const LAMBDA: u32 = 64;
    let reference: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            [(x * 255 / SIDE) as u8, 96, (y * 255 / SIDE) as u8, 255]
        })
        .collect();
    let drawing = scattered_triangles(64);
    let params = MutationParams {
        chain_count: CHAINS,
        lambda: LAMBDA,
        gpu_batch_iters: 1,
        ..Default::default()
    };
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        SIDE,
        SIDE,
        &drawing,
        EngineCapacity::for_params(&params),
    ));
    let (improved, neutral, gain, fitness_gain) =
        assert_gain_follows_fitness(&mut evolver, &drawing, &params, [SIDE, CHAINS, LAMBDA]);
    let stats = evolver.operator_stats();
    let counters = evolver.benchmark_counters();
    assert_eq!(stats.offered.iter().sum::<u64>(), counters.completed_evaluations);
    let offered_by = |op: Operator| stats.offered[op as usize];
    for op in [Operator::Crossover, Operator::Offset, Operator::MicroMove, Operator::ChangeColor] {
        assert!(offered_by(op) > 0, "{} produced no offspring", op.name());
    }
    assert!(improved > 0, "no chain improved");
    assert_rungs_cover_the_scaled_operators(stats, "no spread");
    let summary = format!("{} improved (gain {gain} units, fitness {fitness_gain:.1}), {neutral} neutral of {} offered",
        improved, stats.offered.iter().sum::<u64>());

    // The strongest penalty: the penalized error's penalty change follows P(n).
    let strong = MutationParams { per_point_penalty: MAX_PER_POINT_PENALTY, ..params.clone() };
    let (strong_improved, _, strong_gain, strong_fitness) =
        assert_gain_follows_fitness(&mut evolver, &drawing, &strong, [SIDE, CHAINS, LAMBDA]);
    assert!(strong_improved > 0, "no chain improved under the strongest penalty");
    let summary = format!("{summary}; strongest penalty {strong_improved} improved (gain {strong_gain} units, fitness {strong_fitness:.1})");

    // A step spread: the rungs still count exactly the step-scaled operators,
    // and λ 64 gives every rung its share of their offspring.
    let spread = MutationParams { step_spread_min: 0.25, step_spread_max: 4.0, ..params };
    evolver.prepare_seeded_benchmark(&drawing, &spread, 23, 0, true).expect("spread");
    for _ in 0..48 {
        evolver.run_batch(&spread, true);
    }
    evolver.benchmark_drain().expect("drain");
    let stats = evolver.operator_stats();
    assert_eq!(stats.offered.iter().sum::<u64>(), 48 * u64::from(CHAINS * LAMBDA));
    assert_rungs_cover_the_scaled_operators(stats, "spread");
    let rungs = stats.rung_offered;
    let fair = rungs.iter().sum::<u64>() / rungs.len() as u64;
    assert!(rungs.iter().all(|&n| n > fair / 2 && n < fair * 2), "uneven rungs {rungs:?}");
    let summary = format!("{summary}; spread rungs offered {rungs:?}, improved {:?}",
        stats.rung_improved);

    // Only additions: every mutated offspring is an addition.
    let adds = MutationParams {
        add_polygon_prob: 1.0,
        min_polygons: 1,
        max_polygons: 1000,
        chain_count: CHAINS,
        lambda: LAMBDA,
        gpu_batch_iters: 4,
        ..no_op_params()
    };
    evolver.prepare_seeded_benchmark(&drawing, &adds, 17, 0, true).expect("additions");
    for _ in 0..4 {
        evolver.run_batch(&adds, true);
    }
    evolver.benchmark_drain().expect("drain");
    let stats = evolver.operator_stats();
    assert_eq!(stats.offered[Operator::Add as usize], 4 * 4 * u64::from(CHAINS * LAMBDA));
    assert_eq!(stats.offered.iter().sum::<u64>(), stats.offered[Operator::Add as usize]);

    // An empty parent whose only operator (reorder) cannot run is evaluated
    // unchanged; select accepts such equal-fitness copies only as neutral.
    let empty = Drawing { polygons: vec![], is_dirty: false, fitness: 0.0 };
    let reorders = MutationParams { add_polygon_prob: 0.0, reorder_polygon_prob: 1.0, ..adds };
    evolver.prepare_seeded_benchmark(&empty, &reorders, 19, 0, true).expect("reorders");
    for _ in 0..4 {
        evolver.run_batch(&reorders, true);
    }
    evolver.benchmark_drain().expect("drain");
    let stats = evolver.operator_stats();
    let unchanged = Operator::Unchanged as usize;
    assert_eq!(stats.offered[unchanged], 4 * 4 * u64::from(CHAINS * LAMBDA));
    assert_eq!(stats.improved.iter().sum::<u64>(), 0);
    assert_eq!(stats.gain, [0; OPERATOR_COUNT]);
    assert!(stats.neutral[unchanged] > 0, "no neutral acceptance of an unchanged copy");
    println!("operator stats check: {summary}");
}

/// Run one batch, then 48 single-iteration batches from a seeded start, and
/// check the operator stats' gain against the chains' fitness rise.
/// Returns (improved, neutral, gain units, fitness rise in units).
fn assert_gain_follows_fitness(
    evolver: &mut GpuEvolver,
    drawing: &Drawing,
    params: &MutationParams,
    [side, chains, lambda]: [u32; 3],
) -> (u64, u64, i64, f64) {
    evolver
        .prepare_seeded_benchmark(drawing, params, 13, 2, true)
        .expect("operator stats preparation failed");
    assert_eq!(*evolver.operator_stats(), OperatorStats::default(), "warmup was not forgotten");

    // One iteration per batch, so each snapshot shows one select: its accept
    // flags, and the chains' fitness before and after. The baseline is a GPU
    // select's fitness, not the host's evaluation of the start.
    evolver.run_batch(params, true);
    let mut before = evolver.benchmark_snapshot(chains, lambda).expect("snapshot").fitness_packed;
    let base = evolver.operator_stats().clone();
    let (mut improved, mut neutral) = (0u64, 0u64);
    // Gain is in error units: fitness percentage points times M / 100.
    let units_per_point = f64::from(MAX_ERROR_PER_PIXEL) * f64::from(side * side) / 100.0;
    let mut fitness_gain = 0.0f64;
    for _ in 0..48 {
        evolver.run_batch(params, true);
        let after = evolver.benchmark_snapshot(chains, lambda).expect("snapshot");
        for (chain, flags) in after.chain_update_flags.iter().enumerate() {
            let (old, new) = (f32::from_bits(before[chain]), f32::from_bits(after.fitness_packed[chain]));
            if flags[0] == 1 {
                assert!(new >= old, "chain {chain} accepted a worse winner: {old} -> {new}");
                improved += u64::from(new > old);
                neutral += u64::from(new == old);
                fitness_gain += (f64::from(new) - f64::from(old)) * units_per_point;
            }
        }
        before = after.fitness_packed;
    }
    let stats = evolver.operator_stats();
    let delta = |now: &[u64], then: &[u64]| -> u64 { now.iter().sum::<u64>() - then.iter().sum::<u64>() };
    assert_eq!(delta(&stats.offered, &base.offered), 48 * u64::from(chains * lambda));
    assert_eq!(delta(&stats.improved, &base.improved), improved, "improved winners");
    assert_eq!(delta(&stats.neutral, &base.neutral), neutral, "neutral winners");
    let gain = stats.gain.iter().sum::<i64>() - base.gain.iter().sum::<i64>();
    // Each f32 fitness is within half an ulp (under one unit here) and each
    // penalty term within half a unit of rounding.
    assert!((gain as f64 - fitness_gain).abs() <= 3.0 * improved as f64,
        "gain {gain} units; the chains' fitness rose by {fitness_gain:.1} units over {improved} improvements");
    (improved, neutral, gain, fitness_gain)
}

/// Each rung block sums to the same block of the operators that scale with the step.
fn assert_rungs_cover_the_scaled_operators(stats: &OperatorStats, context: &str) {
    let scaled = |counts: &[u64]| -> u64 {
        Operator::ALL.into_iter().filter(|op| op.scales_with_step()).map(|op| counts[op as usize]).sum()
    };
    for (kind, by_operator, by_rung) in [
        ("offered", &stats.offered, &stats.rung_offered),
        ("improved", &stats.improved, &stats.rung_improved),
        ("neutral", &stats.neutral, &stats.rung_neutral),
    ] {
        assert_eq!(by_rung.iter().sum::<u64>(), scaled(by_operator), "{context}: rung {kind}");
    }
    let scaled_gain: i64 =
        Operator::ALL.into_iter().filter(|op| op.scales_with_step()).map(|op| stats.gain[op as usize]).sum();
    assert_eq!(stats.rung_gain.iter().sum::<i64>(), scaled_gain, "{context}: rung gain");
    assert!(scaled(&stats.offered) > 0, "{context}: no step-scaled offspring");
}

/// Overlapping triangles of varied size, colour and alpha from a fixed LCG, so
/// changing one changes the blend of the ones above and below it.
fn overlapping_triangles(count: usize) -> Drawing {
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    Drawing {
        polygons: (0..count)
            .map(|_| {
                let (cx, cy, size) = (next(), next(), 0.04 + 0.2 * next());
                let mut vertex = || Point {
                    x: (cx + size * (next() - 0.5)).clamp(0.0, 1.0),
                    y: (cy + size * (next() - 0.5)).clamp(0.0, 1.0),
                };
                let points = [vertex(), vertex(), vertex()];
                let mut channel = || (next() * 255.0) as u8;
                let color = Color { r: channel(), g: channel(), b: channel(), a: 40 + channel() / 2 };
                Polygon { points, color }
            })
            .collect(),
        is_dirty: false,
        fitness: 0.0,
    }
}

fn verify_tile_groups() {
    // Large images bin and raster their offspring in tile groups, since
    // tile_data holds the lists of a few offspring only (1024x682: 95 of 256).
    // Capping the group at 37 offspring (seven groups, the last of 34) must
    // search exactly as one group: same offspring, errors, winners and caches.
    // update_framebuffers then scans every polygon for winners of the first six
    // groups, whose lists the last group overwrote.
    const SIDE: u32 = 96;
    const CHAINS: u32 = 4;
    const BATCHES: usize = 12;
    let reference: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            let band = if (x / 16 + y / 8) % 2 == 0 { 60 } else { 190 };
            [(x * 255 / SIDE) as u8, band, (y * 255 / SIDE) as u8, 255]
        })
        .collect();
    // Not scattered_triangles: its grid puts hypotenuses on pixel centres,
    // where the GPU's contracted edge test and the scalar one can disagree.
    let drawing = overlapping_triangles(300);
    let params = MutationParams { chain_count: CHAINS, gpu_batch_iters: 8, ..Default::default() };
    let penalty = params.point_penalty();
    let workload = [CHAINS, params.lambda];
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &reference,
        SIDE,
        SIDE,
        &drawing,
        EngineCapacity::for_params(&params),
    ));
    let mut run = |limit: Option<u32>| {
        evolver.benchmark_set_tile_group_limit(limit);
        let start = evolver
            .prepare_seeded_benchmark(&drawing, &params, 13, 0, true)
            .expect("tile group preparation failed");
        // Timed batches record bin and raster as a pass per stage and group.
        for batch in 0..BATCHES {
            evolver.run_batch(&params, batch % 2 == 0);
        }
        let context = format!("tile group limit {limit:?}");
        let snapshot = assert_chains_cached_byte_l1(&mut evolver, &reference, workload, penalty, &context);
        assert_parents_match_scalar(&snapshot, &reference, SIDE, penalty, &context);
        let fitness: Vec<f32> = snapshot.parents.iter().map(|p| f32::from_bits(p.fitness_bits)).collect();
        assert!(fitness.iter().all(|&now| now > start), "{context}: every chain must improve on {start}: {fitness:?}");
        snapshot
    };
    let whole = run(None);
    let grouped = run(Some(37));
    assert!(whole.parents == grouped.parents, "grouped parents differ");
    assert!(whole.offspring == grouped.offspring, "grouped offspring differ");
    assert_eq!(whole.offspring_errors, grouped.offspring_errors, "grouped offspring errors");
    assert_eq!(whole.chain_update_flags, grouped.chain_update_flags, "grouped update flags");
    assert_eq!(whole.parent_total_errors, grouped.parent_total_errors, "grouped cached totals");
    assert!(whole.parent_framebuffers == grouped.parent_framebuffers, "grouped framebuffers differ");
    assert_eq!(whole.fitness_packed, grouped.fitness_packed, "grouped fitness");
    // A full evaluation in groups of one chain re-derives the cached totals.
    evolver.benchmark_set_tile_group_limit(Some(1));
    evolver.evaluate_chain_fitness(&params);
    let full = assert_chains_cached_byte_l1(&mut evolver, &reference, workload, penalty, "grouped full");
    assert_eq!(grouped.parent_total_errors, full.parent_total_errors, "incremental and grouped full totals");
    let counts: Vec<_> = grouped.parents.iter().map(|p| p.polygon_count).collect();
    println!("tile group check: 7 groups match 1 over {BATCHES} batches; counts {counts:?}, totals {:?}",
        full.parent_total_errors);
}

/// Every parent's cached framebuffer is the scalar renderer's for its polygons
/// to rounding: the GPU's blend may fuse multiply-adds, so a channel can floor
/// one lower or higher (a fresh upload shows the same differences). Its cached
/// total matches those pixels; its fitness follows the scalar fitness.
fn assert_parents_match_scalar(
    snapshot: &BenchmarkSnapshot,
    reference: &[u8],
    side: u32,
    penalty: PointPenalty,
    context: &str,
) {
    for (chain, parent) in snapshot.parents.iter().enumerate() {
        let scalar = reference::evaluate(&parent.polygons, reference, side, side, penalty)
            .expect("reference evaluation");
        let pixels = &snapshot.parent_framebuffers[chain];
        let (mut differing, mut largest) = (0usize, 0u8);
        for (gpu, cpu) in pixels.iter().zip(&scalar.pixels) {
            let diff = (0..3).map(|c| gpu.to_le_bytes()[c].abs_diff(cpu.to_le_bytes()[c])).max().unwrap();
            differing += usize::from(diff > 0);
            largest = largest.max(diff);
        }
        assert!(largest <= 1, "{context}, chain {chain}: a cached pixel differs by {largest} from the scalar render");
        let total = u64::from(snapshot.parent_total_errors[chain]);
        assert!(
            total.abs_diff(scalar.raw_error) <= 3 * differing as u64,
            "{context}, chain {chain}: cached total {total}, scalar {}",
            scalar.raw_error
        );
        let gpu = f32::from_bits(parent.fitness_bits);
        assert!((gpu - scalar.fitness).abs() < 0.01, "{context}, chain {chain}: GPU {gpu}, scalar {}", scalar.fitness);
    }
}
