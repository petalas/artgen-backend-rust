//! Untimed, main-thread cache forensics over an immutable benchmark fixture.
//! Run under a parent subprocess timeout; this never opens a user project.
//!
//! The iteration budget is at most 4096. Snapshot/Serde/wgpu allocations are
//! bounded by the validated fixture and device capacities, with at most five
//! live snapshots including the full-GPU diagnostic during refinement. Engine polls
//! require the parent's process timeout. Refinement replays at most one prefix
//! and 64 individual iterations, each trajectory still capped at 4096.
//! Serde/format macros and wgpu callbacks follow the benchmark framework bounds.
use std::{collections::BTreeMap, fs, path::PathBuf};

use anyhow::{ensure, Context, Result};
use artgen_backend_rust::{
    benchmark::{
        fixtures::{self, LoadedFixture},
        reference,
    },
    gpu_evolver::{
        benchmark::{BenchmarkPackedState, BenchmarkSnapshot},
        GpuEvolver,
    },
};
use serde::Serialize;
use serde_json::{json, Value};

const ITERATION_CAP: u32 = 4096;

#[derive(Serialize)]
struct Config {
    manifest: PathBuf,
    output: PathBuf,
    case: String,
    seed: u64,
    warmup: u32,
    batch_iters: u32,
    max_iterations: u32,
}

impl Config {
    fn parse() -> Result<Self> {
        let args: Vec<_> = std::env::args().skip(1).take(17).collect();
        ensure!(
            args.len() <= 16 && args.len() % 2 == 0,
            "expected at most eight --name value pairs"
        );
        let mut flags = BTreeMap::new();
        for pair in args.as_chunks::<2>().0 {
            let name = pair[0]
                .strip_prefix("--")
                .context("expected --name value")?;
            ensure!(
                [
                    "manifest",
                    "output",
                    "case",
                    "seed",
                    "warmup",
                    "batch-iters",
                    "max-iterations"
                ]
                .contains(&name),
                "unknown flag {name}"
            );
            ensure!(
                flags.insert(name, pair[1].as_str()).is_none(),
                "duplicate flag {name}"
            );
        }
        let config = Self {
            manifest: flags
                .get("manifest")
                .unwrap_or(&".cache/evolution-bench/fixtures/v1/manifest.json")
                .into(),
            output: flags
                .get("output")
                .unwrap_or(&".cache/evolution-bench/trace-gpu-errors")
                .into(),
            case: flags.get("case").unwrap_or(&"darwin").to_string(),
            seed: flags.get("seed").unwrap_or(&"7").parse()?,
            warmup: flags.get("warmup").unwrap_or(&"2").parse()?,
            batch_iters: flags.get("batch-iters").unwrap_or(&"64").parse()?,
            max_iterations: flags.get("max-iterations").unwrap_or(&"4096").parse()?,
        };
        ensure!(
            (1..=64).contains(&config.batch_iters),
            "batch-iters must be 1..64"
        );
        ensure!(
            (1..=ITERATION_CAP).contains(&config.max_iterations),
            "max-iterations must be 1..4096"
        );
        ensure!(
            config.max_iterations % config.batch_iters == 0,
            "budget must contain whole batches"
        );
        ensure!(config.warmup <= 16, "warmup must be 0..16");
        ensure!(!config.output.exists(), "output must be a new directory");
        Ok(config)
    }
}

#[derive(Serialize)]
struct Checkpoint {
    iteration: u64,
    cached: Vec<u32>,
    framebuffer: Vec<u64>,
    canonical_sha256: String,
}

fn main() -> Result<()> {
    let config = Config::parse()?;
    let input = fixtures::load(&config.manifest, &config.case)?;
    fs::create_dir_all(&config.output)?;
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &input.rgba,
        input.fixture.width,
        input.fixture.height,
        &input.drawing,
    ));
    let scopes = evolver.benchmark_error_scopes();
    let result = trace(&mut evolver, &input, &config);
    let errors = futures_lite::future::block_on(scopes.finish());
    let (checkpoints, failure) = result?;
    let mismatch_count = failure
        .as_ref()
        .map_or(0, |v| v["chains"].as_array().map_or(0, Vec::len));
    fs::write(
        config.output.join("trace.json"),
        serde_json::to_vec_pretty(&json!({
            "config": config, "fixture_manifest_sha256": input.manifest_sha256,
            "checkpoints": checkpoints, "failure": failure,
            "raw_error_mismatch_count": mismatch_count, "device_errors": errors,
            "performance_measurement": false, "production_timestamps_enabled": true,
        }))?,
    )?;
    println!(
        "trace={} raw_error_mismatches={mismatch_count} device_errors={}",
        config.output.join("trace.json").display(),
        errors.len()
    );
    ensure!(errors.is_empty(), "GPU device errors: {errors:?}");
    ensure!(
        mismatch_count == 0,
        "cached raw error differs from framebuffer byte L1"
    );
    Ok(())
}

fn prepare(evolver: &mut GpuEvolver, input: &LoadedFixture, config: &Config) -> Result<()> {
    // Warm the original partition even for a one-iteration diagnostic replay.
    evolver
        .prepare_seeded_benchmark(
            &input.drawing,
            &input.manifest.params,
            config.seed,
            config.warmup,
            true,
        )
        .map_err(anyhow::Error::msg)?;
    Ok(())
}

fn trace(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    config: &Config,
) -> Result<(Vec<Checkpoint>, Option<Value>)> {
    prepare(evolver, input, config)?;
    let mut params = input.manifest.params.clone();
    params.gpu_batch_iters = config.batch_iters;
    let mut before = snapshot(evolver, input)?;
    let mut checkpoints =
        Vec::with_capacity((config.max_iterations / config.batch_iters + 1) as usize);
    checkpoints.push(checkpoint(&before, &input.rgba)?);
    ensure!(
        before
            .parents
            .iter()
            .all(|p| f32::from_bits(p.fitness_bits) > 97.0),
        "initial parent is not mature"
    );
    if mismatches(&before, &input.rgba)? != 0 {
        let failure = forensic(evolver, input, &before, &before)?;
        return Ok((checkpoints, Some(failure)));
    }
    for _ in 0..config.max_iterations / config.batch_iters {
        let _best_drawing = evolver.run_batch(&params, true);
        let after = snapshot(evolver, input)?;
        let summary = checkpoint(&after, &input.rgba)?;
        let bad = summary
            .cached
            .iter()
            .zip(&summary.framebuffer)
            .any(|(&a, &b)| u64::from(a) != b);
        if bad || summary.iteration % 64 == 0 {
            println!(
                "iteration={} cached={:?} framebuffer={:?} canonical={}",
                summary.iteration, summary.cached, summary.framebuffer, summary.canonical_sha256
            );
        }
        checkpoints.push(summary);
        if bad {
            let mut failure = forensic(evolver, input, &before, &after)?;
            if config.batch_iters > 1 {
                failure["refinement"] = refine(evolver, input, config, &before, &after)?;
            }
            return Ok((checkpoints, Some(failure)));
        }
        before = after;
    }
    Ok((checkpoints, None))
}

fn refine(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    config: &Config,
    coarse_before: &BenchmarkSnapshot,
    coarse_after: &BenchmarkSnapshot,
) -> Result<Value> {
    prepare(evolver, input, config)?;
    let mut params = input.manifest.params.clone();
    params.gpu_batch_iters = config.batch_iters;
    let prefix = coarse_before.counters.completed_iterations;
    assert!(prefix <= u64::from(ITERATION_CAP));
    for _ in 0..prefix / u64::from(config.batch_iters) {
        let _best_drawing = evolver.run_batch(&params, true);
    }
    let mut before = snapshot(evolver, input)?;
    let expected = checkpoint(coarse_before, &input.rgba)?;
    let replayed = checkpoint(&before, &input.rgba)?;
    let prefix_matches = expected.canonical_sha256 == replayed.canonical_sha256;
    println!("refinement prefix_iteration={prefix} canonical_match={prefix_matches}");
    if !prefix_matches {
        return Ok(json!({"prefix_replay_matches": false,
            "expected": expected, "replayed": replayed}));
    }
    params.gpu_batch_iters = 1;
    for _ in 0..config.batch_iters {
        let _best_drawing = evolver.run_batch(&params, true);
        let after = snapshot(evolver, input)?;
        if mismatches(&after, &input.rgba)? != 0 {
            let mut evidence = forensic(evolver, input, &before, &after)?;
            evidence["prefix_replay_matches"] = json!(true);
            println!(
                "first_bad_iteration={}",
                after.counters.completed_iterations
            );
            return Ok(evidence);
        }
        before = after;
    }
    Ok(
        json!({"prefix_replay_matches": true, "single_iteration_replay_diverged": false,
        "end_matches_coarse": checkpoint(&before, &input.rgba)?.canonical_sha256
            == checkpoint(coarse_after, &input.rgba)?.canonical_sha256}),
    )
}

fn snapshot(evolver: &mut GpuEvolver, input: &LoadedFixture) -> Result<BenchmarkSnapshot> {
    let params = &input.manifest.params;
    evolver
        .benchmark_snapshot(params.chain_count, params.lambda)
        .map_err(anyhow::Error::msg)
}

fn checkpoint(state: &BenchmarkSnapshot, rgba: &[u8]) -> Result<Checkpoint> {
    // Identical to the trial canonical tuple: reporting control, occupancy and
    // batch counters are omitted, while every RNG and evolution word remains.
    let canonical = (
        &state.parents,
        &state.offspring,
        &state.parent_total_errors,
        &state.parent_framebuffers,
        &state.offspring_errors,
        &state.chain_update_flags,
        &state.fitness_packed,
    );
    Ok(Checkpoint {
        iteration: state.counters.completed_iterations,
        cached: state.parent_total_errors.clone(),
        framebuffer: state
            .parent_framebuffers
            .iter()
            .map(|fb| byte_error(fb, rgba))
            .collect::<Result<_>>()?,
        canonical_sha256: fixtures::sha256(&serde_json::to_vec(&canonical)?),
    })
}

fn mismatches(state: &BenchmarkSnapshot, rgba: &[u8]) -> Result<usize> {
    let summary = checkpoint(state, rgba)?;
    Ok(summary
        .cached
        .iter()
        .zip(&summary.framebuffer)
        .filter(|(a, b)| u64::from(**a) != **b)
        .count())
}

fn byte_error(pixels: &[u32], rgba: &[u8]) -> Result<u64> {
    ensure!(
        rgba.len() == pixels.len() * 4,
        "framebuffer/reference lengths differ"
    );
    let mut error = 0;
    for (&pixel, target) in pixels.iter().zip(rgba.as_chunks::<4>().0) {
        for channel in 0..3 {
            error +=
                u64::from(((pixel >> (channel * 8)) & 255).abs_diff(u32::from(target[channel])));
        }
    }
    Ok(error)
}

fn forensic(
    evolver: &mut GpuEvolver,
    input: &LoadedFixture,
    before: &BenchmarkSnapshot,
    after: &BenchmarkSnapshot,
) -> Result<Value> {
    let mut chains = Vec::with_capacity(after.parents.len());
    for chain in 0..after.parents.len() {
        let framebuffer_error = byte_error(&after.parent_framebuffers[chain], &input.rgba)?;
        if u64::from(after.parent_total_errors[chain]) == framebuffer_error {
            continue;
        }
        let parent = &after.parents[chain];
        let scalar = reference::evaluate(
            &parent.polygons,
            &input.rgba,
            input.fixture.width,
            input.fixture.height,
        )?;
        let before_scalar = reference::evaluate(
            &before.parents[chain].polygons,
            &input.rgba,
            input.fixture.width,
            input.fixture.height,
        )?;
        chains.push(json!({
            "chain": chain, "stored_error": after.parent_total_errors[chain],
            "framebuffer_error": framebuffer_error, "scalar_error": scalar.raw_error,
            "scalar_framebuffer_differing_pixels": pixel_differences(&scalar.pixels, &after.parent_framebuffers[chain]),
            "before_parent": before.parents[chain], "after_parent": parent,
            "before_scalar_error": before_scalar.raw_error,
            "before_scalar_framebuffer_differing_pixels": pixel_differences(&before_scalar.pixels, &before.parent_framebuffers[chain]),
            "last_iteration_update_flags": after.chain_update_flags[chain],
            "matching_last_iteration_offspring": matching_candidates(chain, input, before, after)?,
        }));
    }
    // Destructive diagnostic AFTER all pre-recompute state has been captured.
    // This overwrites working slots and cached totals, so never resume evolution.
    let _best_fitness = evolver.evaluate_chain_fitness(input.manifest.params.chain_count);
    let full = snapshot(evolver, input)?;
    Ok(json!({
        "before_iteration": before.counters.completed_iterations,
        "after_iteration": after.counters.completed_iterations,
        "before": checkpoint(before, &input.rgba)?, "after": checkpoint(after, &input.rgba)?,
        "chains": chains, "full_gpu_recomputed_errors": full.parent_total_errors,
        "full_gpu_recomputed_fitness_bits": full.parents.iter().map(|p| p.fitness_bits).collect::<Vec<_>>(),
        "parent_polygons_unchanged_by_full_recompute": full.parents.iter().zip(&after.parents).all(|(a,b)| a.polygons == b.polygons),
        "framebuffers_unchanged_by_full_recompute": full.parent_framebuffers == after.parent_framebuffers,
    }))
}

fn matching_candidates(
    chain: usize,
    input: &LoadedFixture,
    before: &BenchmarkSnapshot,
    state: &BenchmarkSnapshot,
) -> Result<Vec<Value>> {
    let lambda = input.manifest.params.lambda as usize;
    let parent = &state.parents[chain];
    let mut matches = Vec::with_capacity(lambda);
    for id in chain * lambda..(chain + 1) * lambda {
        let candidate = &state.offspring[id];
        let (culled, removed) = selection_cull(candidate, input.manifest.params.min_polygons);
        if candidate.polygons != parent.polygons && culled != parent.polygons {
            continue;
        }
        let scalar = reference::evaluate(
            &candidate.polygons,
            &input.rgba,
            input.fixture.width,
            input.fixture.height,
        )?;
        matches.push(json!({
            "offspring_id": id, "exact_polygon_match": candidate.polygons == parent.polygons,
            "selection_cull_polygon_match": culled == parent.polygons,
            "candidate": candidate, "removed": removed,
            "candidate_scalar_error": scalar.raw_error,
            "candidate_scalar_framebuffer_differing_pixels": pixel_differences(&scalar.pixels, &state.parent_framebuffers[chain]),
            "candidate_region_errors": region_errors(candidate, &before.parent_framebuffers[chain], &scalar.pixels, input)?,
        }));
    }
    Ok(matches)
}

fn region_errors(
    candidate: &BenchmarkPackedState,
    old: &[u32],
    new: &[u32],
    input: &LoadedFixture,
) -> Result<Value> {
    ensure!(
        old.len() == new.len() && old.len() * 4 == input.rgba.len(),
        "region dimensions differ"
    );
    let bbox = [
        candidate.fitness_bits & 65535,
        candidate.fitness_bits >> 16,
        candidate.stagnation_counter & 65535,
        candidate.stagnation_counter >> 16,
    ];
    let [wg_x, wg_y] = input.manifest.params.rasterize_wg;
    let tiles = [
        bbox[0] / wg_x * wg_x,
        bbox[1] / wg_y * wg_y,
        bbox[2].div_ceil(wg_x) * wg_x,
        bbox[3].div_ceil(wg_y) * wg_y,
    ];
    let mut pixel_errors = [0_u64; 2];
    let mut tile_errors = [0_u64; 2];
    for (index, (&old_pixel, &new_pixel)) in old.iter().zip(new).enumerate() {
        let x = index as u32 % input.fixture.width;
        let y = index as u32 / input.fixture.width;
        let in_bbox = x >= bbox[0] && y >= bbox[1] && x < bbox[2] && y < bbox[3];
        let in_tile = x >= tiles[0] && y >= tiles[1] && x < tiles[2] && y < tiles[3];
        for channel in 0..3 {
            let target = u32::from(input.rgba[index * 4 + channel]);
            let old_error = u64::from(((old_pixel >> (channel * 8)) & 255).abs_diff(target));
            let new_error = u64::from(((new_pixel >> (channel * 8)) & 255).abs_diff(target));
            if in_bbox {
                pixel_errors[0] += old_error;
                pixel_errors[1] += new_error;
            }
            if in_tile {
                tile_errors[0] += old_error;
                tile_errors[1] += new_error;
            }
        }
    }
    Ok(json!({"bbox": bbox, "intersecting_tiles_bbox": tiles,
        "per_pixel_bbox_old_new_error": pixel_errors, "whole_tiles_old_new_error": tile_errors}))
}

fn selection_cull(
    candidate: &BenchmarkPackedState,
    min_polygons: u32,
) -> (Vec<[u32; 4]>, Vec<Value>) {
    let mut polygons = candidate.polygons.clone();
    let mut removed = Vec::with_capacity(polygons.len());
    let mut write = 0;
    for (index, polygon) in candidate.polygons.iter().enumerate() {
        let vertices = [polygon[1], polygon[2], polygon[3]].map(|word| {
            [
                (word & 65535) as f32 / 65535.0,
                (word >> 16) as f32 / 65535.0,
            ]
        });
        let [a, b, c] = vertices;
        let cross = ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
        if cross > 0.00001 {
            polygons[write] = *polygon;
            write += 1;
        } else {
            removed
                .push(json!({"index": index, "packed": polygon, "twice_normalized_area": cross}));
        }
    }
    if write < polygons.len() && write >= min_polygons as usize {
        polygons.truncate(write);
    }
    (polygons, removed)
}

fn pixel_differences(a: &[u32], b: &[u32]) -> Value {
    assert_eq!(
        a.len(),
        b.len(),
        "pixel comparisons require matching dimensions"
    );
    let count = a.iter().zip(b).filter(|(a, b)| a != b).count();
    let first: Vec<_> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(32)
        .map(|(index, (&a, &b))| json!({"pixel_index": index, "scalar": a, "framebuffer": b}))
        .collect();
    json!({"count": count, "first": first})
}
