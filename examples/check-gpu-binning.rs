//! Main-thread checks of GPU binning through the real `GpuPipeline`.
//! With no arguments, checks the 01/04 dirty-binning experiments, whose skipped
//! bins must retain stale words. The restored reference intentionally fails
//! that strict contract.
//! Run with `timeout 600s cargo run --release --locked --example check-gpu-binning`.
//! `--scope consumed-only` instead compares consumed counts and ordered valid
//! prefixes with the frozen dd52d50 GPU binner. Unread words are ignored only in
//! that explicit mode. The original no-argument strict checks are unchanged.
//! `--scope full-buffers` also compares every count/index word with that frozen
//! N1 baseline, including unconsumed counts and unwritten poison/stale indices.
//! This diagnoses an N2 baseline probe that repeats the complete reset/scatter
//! body twice inside the existing one-lane workgroup: final buffers must match
//! N1. Both paired modes retain the consumed-prefix and analytical truth checks.
//!
//! Bounds: five supported workgroups, three original images plus four narrow
//! 512/513-axis images in either paired mode, three offspring, at most 1000
//! polygons, 195 tiles per offspring and 20 checks per buffer fixture. A consumed
//! or full-buffer check dispatches both GPU binners. Only one image's buffers
//! are live at once.
//! Each workgroup/image has 17 standard checks and six packed-edge checks in
//! either paired mode: 805 paired checks and 1610 GPU dispatches in total.
//! Strict mode runs only the standard checks, for 255 GPU dispatches in total.
//! Each mapping/poll has a 120-second limit. GPU work stays on this main thread.
//! Rule 3 deviation: wgpu and the scalar oracle allocate during this finite
//! checker, with one fixture live at a time and at most 195 tiles per offspring.
//! Rule 7 deviation: existing `GpuPipeline` construction has internal unbounded
//! adapter/compilation waits; the outer process timeout bounds the whole run.
//! Rules 8/9 deviation: formatting/assertion macros and wgpu's single mapping
//! callback use the existing framework interfaces, with no recursive callbacks.
//! Verification: the command above checks actual count/index readbacks. Keep the
//! bounds and outer timeout when extending cases; production APIs are unchanged.

use std::{sync::mpsc, time::Duration};

use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::{
    gpu_evolver::{
        buffers::{default_gpu_params, GpuDrawingState, GpuPolygon},
        pipeline::GpuPipeline,
    },
    settings::TILE_MAX_POLYS,
};
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

const WORKGROUPS: [[u32; 2]; 5] = [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];
const IMAGE_SIZES: [[u32; 2]; 3] = [[64, 48], [65, 51], [17, 19]];
const GUARD_SIZES: [[u32; 2]; 4] = [[512, 19], [513, 19], [19, 512], [19, 513]];
const OFFSPRING: usize = 3;
const BIN_CAPACITY: usize = 256;
const TIMEOUT: Duration = Duration::from_secs(120);
type DirtyBox = [u32; 4];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Strict,
    ConsumedOnly,
    FullBuffers,
}

struct BinBuffers {
    counts: wgpu::Buffer,
    indices: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct ReferenceBins {
    pipeline: wgpu::ComputePipeline,
    buffers: BinBuffers,
}

struct Bins {
    image_size: [u32; 2],
    scope: Scope,
    buffers: BinBuffers,
    reference: Option<ReferenceBins>,
    expected: Vec<u32>,
    dispatches: usize,
}

fn main() -> Result<()> {
    let scope = parse_scope()?;
    ensure!(usize::try_from(TILE_MAX_POLYS)? == BIN_CAPACITY);
    let reference = [0, 0, 0, 255].repeat(65 * 51);
    let initial = vec![GpuDrawingState::zeroed()];
    let mut pipeline =
        futures_lite::future::block_on(GpuPipeline::new(1, 65, 51, &reference, &initial));
    ensure!(usize::try_from(pipeline.offspring_capacity)? >= OFFSPRING);
    let mut dispatches = 0;
    for workgroup in WORKGROUPS {
        pipeline.set_rasterize_wg(workgroup);
        let reference = (scope != Scope::Strict).then(|| reference_pipeline(&pipeline));
        for image_size in IMAGE_SIZES {
            dispatches += check_fixture(&pipeline, image_size, reference.as_ref(), scope)?;
            if let Some(reference) = &reference {
                dispatches += check_packed_edges(&pipeline, image_size, reference, scope)?;
            }
        }
        if let Some(reference) = &reference {
            for image_size in GUARD_SIZES {
                dispatches += check_fixture(&pipeline, image_size, Some(reference), scope)?;
                dispatches += check_packed_edges(&pipeline, image_size, reference, scope)?;
            }
        }
    }
    match scope {
        Scope::Strict => println!("GPU dirty-binning candidate checks passed: 5 workgroups, 3 images, {dispatches} dispatches"),
        Scope::ConsumedOnly => println!("GPU consumed-only checks passed: 5 workgroups, 7 images, {dispatches} paired checks, {} GPU dispatches", dispatches * 2),
        Scope::FullBuffers => println!("GPU full-buffers checks passed: 5 workgroups, 7 images, {dispatches} paired checks, {} GPU dispatches", dispatches * 2),
    }
    Ok(())
}

fn parse_scope() -> Result<Scope> {
    let mut args = std::env::args_os().skip(1);
    match (args.next(), args.next(), args.next()) {
        (None, None, None) => Ok(Scope::Strict),
        (Some(flag), Some(scope), None) if flag == "--scope" && scope == "consumed-only" => {
            Ok(Scope::ConsumedOnly)
        }
        (Some(flag), Some(scope), None) if flag == "--scope" && scope == "full-buffers" => {
            Ok(Scope::FullBuffers)
        }
        _ => bail!("usage: check-gpu-binning [--scope consumed-only|full-buffers]"),
    }
}

fn reference_pipeline(p: &GpuPipeline) -> wgpu::ComputePipeline {
    let source = include_str!("../benchmarks/evolution/v1/experiments/reference-bin-dd52d50.wgsl")
        .replace(
            "const TILE_W: u32 = 16;",
            &format!("const TILE_W: u32 = {};", p.rasterize_wg[0]),
        )
        .replace(
            "const TILE_H: u32 = 16;",
            &format!("const TILE_H: u32 = {};", p.rasterize_wg[1]),
        );
    let shader = p.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("checker frozen full binner"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layouts = [
        p.bin_polygons_pipeline.get_bind_group_layout(0),
        p.bin_polygons_pipeline.get_bind_group_layout(1),
    ];
    let layout = p
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("checker compatible reference layout"),
            bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1])],
            immediate_size: 0,
        });
    p.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("checker original full binner"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

fn new_bins(
    p: &GpuPipeline,
    image_size: [u32; 2],
    reference: Option<&wgpu::ComputePipeline>,
    scope: Scope,
) -> Bins {
    assert!(IMAGE_SIZES.contains(&image_size) || GUARD_SIZES.contains(&image_size));
    assert!(WORKGROUPS.contains(&p.rasterize_wg));
    let tiles =
        image_size[0].div_ceil(p.rasterize_wg[0]) * image_size[1].div_ceil(p.rasterize_wg[1]);
    assert!(tiles <= 195);
    let slots = usize::try_from(tiles).expect("at most 195 tiles") * OFFSPRING;
    let expected: Vec<u32> = (0..slots * (BIN_CAPACITY + 1))
        .map(|i| 0xA000_0000 | u32::try_from(i).expect("bounded buffer word index"))
        .collect();
    Bins {
        image_size,
        scope,
        buffers: new_bin_buffers(p, &expected, slots),
        // Both binners start with identical poison and keep their own history.
        // Neither buffer set is re-poisoned between paired dispatches.
        reference: reference.map(|pipeline| ReferenceBins {
            pipeline: pipeline.clone(),
            buffers: new_bin_buffers(p, &expected, slots),
        }),
        expected,
        dispatches: 0,
    }
}

fn new_bin_buffers(p: &GpuPipeline, poison: &[u32], slots: usize) -> BinBuffers {
    assert!(slots <= 195 * OFFSPRING && poison.len() == slots * (BIN_CAPACITY + 1));
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let counts = p
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("checker poisoned counts"),
            contents: bytemuck::cast_slice(&poison[..slots]),
            usage,
        });
    let indices = p
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("checker poisoned indices"),
            contents: bytemuck::cast_slice(&poison[slots..]),
            usage,
        });
    let staging = p.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("checker bin readback"),
        size: counts.size() + indices.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // The engine's tile_data lacks COPY_SRC. Rebind checker-owned readable
    // buffers through the actual production pipeline's public layout instead.
    let entries =
        [(0, &p.working_states_buf), (1, &indices), (2, &counts)].map(|(binding, buffer)| {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        });
    let bind_group = p.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("checker bin bindings"),
        layout: &p.bin_polygons_pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    BinBuffers {
        counts,
        indices,
        staging,
        bind_group,
    }
}

fn dispatch(
    p: &GpuPipeline,
    image_size: [u32; 2],
    bins: &BinBuffers,
    pipeline: &wgpu::ComputePipeline,
    states: &[GpuDrawingState],
) -> Result<Vec<u32>> {
    ensure!(states.len() == OFFSPRING);
    let [w, h] = image_size;
    let params = default_gpu_params(w, h, 1);
    p.queue
        .write_buffer(&p.params_buf, 0, bytemuck::bytes_of(&params));
    p.queue
        .write_buffer(&p.working_states_buf, 0, bytemuck::cast_slice(states));
    let mut encoder = p
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("checker bin dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("checker production binning"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bins.bind_group, &[]);
        pass.set_bind_group(1, &p.params_bind_group, &[]);
        pass.dispatch_workgroups(u32::try_from(OFFSPRING)?, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&bins.counts, 0, &bins.staging, 0, bins.counts.size());
    encoder.copy_buffer_to_buffer(
        &bins.indices,
        0,
        &bins.staging,
        bins.counts.size(),
        bins.indices.size(),
    );
    readback(p, &bins.staging, encoder)
}

fn readback(
    p: &GpuPipeline,
    staging: &wgpu::Buffer,
    encoder: wgpu::CommandEncoder,
) -> Result<Vec<u32>> {
    let submission = p.queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            // The receiver may have been dropped after a bounded poll failed.
            let _ = tx.send(result);
        });
    p.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(TIMEOUT),
        })
        .context("binning readback poll")?;
    rx.recv_timeout(TIMEOUT)
        .context("binning map callback")?
        .context("binning mapping")?;
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .context("binning mapped range")?;
    let words = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(words)
}

fn check_dispatch(
    p: &GpuPipeline,
    bins: &mut Bins,
    states: &[GpuDrawingState],
    label: &str,
) -> Result<()> {
    ensure!(bins.dispatches < 20, "fixture dispatch budget exhausted");
    let actual = dispatch(
        p,
        bins.image_size,
        &bins.buffers,
        &p.bin_polygons_pipeline,
        states,
    )?;
    if let Some(reference) = &bins.reference {
        let expected = dispatch(
            p,
            bins.image_size,
            &reference.buffers,
            &reference.pipeline,
            states,
        )?;
        check_consumed_gpu(p, bins, states, &actual, &expected, label)?;
        if bins.scope == Scope::FullBuffers {
            check_full_buffers(p, bins, &actual, &expected, label)?;
        }
        bins.dispatches += 1;
        return Ok(());
    }
    let slots = usize::try_from(bins.buffers.counts.size() / 4)?;
    let tiles = slots / OFFSPRING;
    for (offspring, state) in states.iter().enumerate() {
        let full_bins = scalar_full_bins(state, bins.image_size, p.rasterize_wg);
        ensure!(full_bins.len() == tiles);
        for (tile, layers) in full_bins.iter().enumerate() {
            if !consumed(state, tile, bins.image_size, p.rasterize_wg) {
                continue;
            }
            let global = offspring * tiles + tile;
            bins.expected[global] = u32::try_from(layers.len())?;
            for (slot, &layer) in layers.iter().take(BIN_CAPACITY).enumerate() {
                bins.expected[slots + global * BIN_CAPACITY + slot] = layer;
            }
        }
    }
    ensure!(actual.len() == bins.expected.len());
    for (word, (&value, &expected)) in actual.iter().zip(&bins.expected).enumerate() {
        if value != expected {
            bail!(
                "size={:?}, wg={:?}, {label}, word={word}: got {value}, expected {expected}",
                bins.image_size,
                p.rasterize_wg
            );
        }
    }
    bins.dispatches += 1;
    Ok(())
}

fn check_full_buffers(
    p: &GpuPipeline,
    bins: &Bins,
    actual: &[u32],
    expected: &[u32],
    label: &str,
) -> Result<()> {
    ensure!(actual.len() == bins.expected.len() && expected.len() == actual.len());
    for (word, (&value, &reference)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            value == reference,
            "size={:?}, wg={:?}, {label}, word={word}: got {value}, reference {reference}",
            bins.image_size,
            p.rasterize_wg
        );
    }
    Ok(())
}

fn check_consumed_gpu(
    p: &GpuPipeline,
    bins: &Bins,
    states: &[GpuDrawingState],
    actual: &[u32],
    expected: &[u32],
    label: &str,
) -> Result<()> {
    ensure!(actual.len() == bins.expected.len() && expected.len() == actual.len());
    let slots = usize::try_from(bins.buffers.counts.size() / 4)?;
    let tiles = slots / OFFSPRING;
    for (offspring, state) in states.iter().enumerate() {
        for tile in 0..tiles {
            if !consumed(state, tile, bins.image_size, p.rasterize_wg) {
                continue;
            }
            let global = offspring * tiles + tile;
            ensure!(
                actual[global] == expected[global],
                "size={:?}, wg={:?}, {label}, offspring={offspring}, tile={tile}: count {}, reference {}",
                bins.image_size, p.rasterize_wg, actual[global], expected[global]
            );
            ensure!(
                expected[global] <= state.polygon_count,
                "invalid reference count"
            );
            let count = usize::try_from(expected[global])?.min(BIN_CAPACITY);
            let start = slots + global * BIN_CAPACITY;
            ensure!(
                actual[start..start + count] == expected[start..start + count],
                "size={:?}, wg={:?}, {label}, offspring={offspring}, tile={tile}: ordered prefix differs",
                bins.image_size, p.rasterize_wg
            );
            // The 500/999/1000-layer fixtures contain only full-image triangles.
            // Check their reference truth as well as agreement between binners.
            if [500, 999, 1000].contains(&state.polygon_count) {
                ensure!(
                    expected[global] == state.polygon_count,
                    "reference lost overlapping layers"
                );
                for (layer, &index) in expected[start..start + BIN_CAPACITY].iter().enumerate() {
                    ensure!(
                        index == u32::try_from(layer)?,
                        "reference layer order changed"
                    );
                }
            }
        }
    }
    Ok(())
}

fn consumed(state: &GpuDrawingState, tile: usize, size: [u32; 2], wg: [u32; 2]) -> bool {
    let tile = u32::try_from(tile).expect("at most 195 tiles");
    let columns = size[0].div_ceil(wg[0]);
    let x = tile % columns * wg[0];
    let y = tile / columns * wg[1];
    // Direct rasterizer predicate, independent of the candidate's tile rectangle.
    (x + wg[0]).min(size[0]) > (state.fitness_bits & 0xFFFF)
        && x < (state.stagnation_counter & 0xFFFF)
        && (y + wg[1]).min(size[1]) > (state.fitness_bits >> 16)
        && y < (state.stagnation_counter >> 16)
}

fn scalar_full_bins(state: &GpuDrawingState, size: [u32; 2], wg: [u32; 2]) -> Vec<Vec<u32>> {
    let columns = size[0].div_ceil(wg[0]);
    let rows = size[1].div_ceil(wg[1]);
    let count = usize::try_from(state.polygon_count).expect("bounded polygon count");
    assert!(count <= 257);
    let bounds: Vec<_> = state.polygons[..count]
        .iter()
        .map(|&poly| scalar_pixel_bounds(poly, size))
        .collect();
    // Construct ALL original bins by testing each polygon against each padded
    // tile. Dirty-box clipping and the shader's polygon-to-tile formulas are absent.
    let mut bins = Vec::with_capacity(usize::try_from(columns * rows).expect("bounded tiles"));
    for y in 0..rows {
        for x in 0..columns {
            let mut layers = Vec::with_capacity(count);
            for (layer, bounds) in bounds.iter().enumerate() {
                if bounds[0] < (x + 1) * wg[0]
                    && bounds[2] > x * wg[0]
                    && bounds[1] < (y + 1) * wg[1]
                    && bounds[3] > y * wg[1]
                {
                    layers.push(u32::try_from(layer).expect("at most 257 layers"));
                }
            }
            bins.push(layers);
        }
    }
    bins
}

fn scalar_pixel_bounds(poly: GpuPolygon, size: [u32; 2]) -> [u32; 4] {
    let mut lo = [u32::MAX; 2];
    let mut hi = [0; 2];
    for &word in &poly.data[1..] {
        for (axis, coordinate) in [word & 0xFFFF, word >> 16].into_iter().enumerate() {
            lo[axis] = lo[axis].min(coordinate);
            hi[axis] = hi[axis].max(coordinate);
        }
    }
    // Independent exact rational bounds. Fixture coordinates are separated
    // from integer pixel boundaries except the exactly representable 0 and 1.
    [
        lo[0] * size[0] / 65535,
        lo[1] * size[1] / 65535,
        (hi[0] * size[0]).div_ceil(65535),
        (hi[1] * size[1]).div_ceil(65535),
    ]
}

fn polygon(vertices: [[u16; 2]; 3]) -> GpuPolygon {
    let packed = vertices.map(|[x, y]| u32::from(x) | (u32::from(y) << 16));
    GpuPolygon {
        data: [0x8080_4020, packed[0], packed[1], packed[2]],
    }
}

fn layered_polygons() -> [GpuPolygon; 9] {
    [
        [[0, 0], [65535, 0], [0, 65535]],
        [[65535, 0], [65535, 65535], [0, 65535]],
        [[7000, 7000], [14000, 7000], [7000, 14000]],
        [[42000, 42000], [56000, 42000], [42000, 56000]],
        [[28000, 14000], [42000, 28000], [14000, 42000]],
        [[65535, 65535]; 3],
        [[0, 0]; 3],
        [[28000, 7000], [28000, 42000], [28000, 56000]],
        [[7000, 28000], [42000, 28000], [56000, 28000]],
    ]
    .map(polygon)
}

fn set_dirty(state: &mut GpuDrawingState, bounds: DirtyBox) {
    assert!(bounds.iter().all(|&v| u16::try_from(v).is_ok()));
    state.fitness_bits = bounds[0] | (bounds[1] << 16);
    state.stagnation_counter = bounds[2] | (bounds[3] << 16);
}

fn drawing(polygons: &[GpuPolygon], bounds: DirtyBox) -> GpuDrawingState {
    assert!(polygons.len() <= 1000);
    let mut state = GpuDrawingState::zeroed();
    state.polygon_count = u32::try_from(polygons.len()).expect("at most 1000 polygons");
    state.polygons[..polygons.len()].copy_from_slice(polygons);
    set_dirty(&mut state, bounds);
    state
}

fn check_fixture(
    p: &GpuPipeline,
    size: [u32; 2],
    reference: Option<&wgpu::ComputePipeline>,
    scope: Scope,
) -> Result<usize> {
    let mut bins = new_bins(p, size, reference, scope);
    let full = [0, 0, size[0], size[1]];
    let polygons = layered_polygons();
    let mut states: Vec<_> = (0..OFFSPRING).map(|_| drawing(&polygons, full)).collect();
    states[1].polygons[..polygons.len()].reverse();
    states[2] = drawing(&polygons[..1], full);
    check_dispatch(p, &mut bins, &states, "full image, poisoned counts/indices")?;
    check_regions(p, &mut bins, &mut states)?;
    check_overflow_and_empty(p, &mut bins)?;
    println!(
        "size={size:?}, wg={:?}: {} dispatches passed",
        p.rasterize_wg, bins.dispatches
    );
    Ok(bins.dispatches)
}

fn check_regions(p: &GpuPipeline, bins: &mut Bins, states: &mut [GpuDrawingState]) -> Result<()> {
    let [w, h] = bins.image_size;
    let top_left = [0, 0, 1, 1];
    let bottom_right = [w - 1, h - 1, w, h];
    // No buffer reinitialization between A, disjoint B, then A again.
    for (name, boxes) in [
        ("region A", [top_left, bottom_right, top_left]),
        ("region B", [bottom_right, top_left, bottom_right]),
        ("A again", [top_left, bottom_right, top_left]),
    ] {
        for (state, bounds) in states.iter_mut().zip(boxes) {
            set_dirty(state, bounds);
        }
        check_dispatch(p, bins, states, name)?;
    }
    let boxes = dirty_boxes(bins.image_size, p.rasterize_wg);
    let (groups, remainder) = boxes.as_chunks::<OFFSPRING>();
    assert_eq!(remainder, &[] as &[DirtyBox]);
    for (batch, boxes) in groups.iter().enumerate() {
        for (state, &bounds) in states.iter_mut().zip(boxes) {
            set_dirty(state, bounds);
        }
        check_dispatch(p, bins, states, &format!("boundary group {batch}"))?;
    }
    Ok(())
}

fn check_overflow_and_empty(p: &GpuPipeline, bins: &mut Bins) -> Result<()> {
    let [w, h] = bins.image_size;
    let full = [0, 0, w, h];
    let polygons = [polygon([[0, 0], [65535, 0], [0, 65535]]); 257];
    let mut states: Vec<_> = [255, 256, 257]
        .into_iter()
        .map(|count| drawing(&polygons[..count], full))
        .collect();
    check_dispatch(p, bins, &states, "255/256/257 occupancy, full image")?;
    for (state, bounds) in states
        .iter_mut()
        .zip([[0, 0, 1, 1], [w - 1, h - 1, w, h], [1, 1, 1, 1]])
    {
        set_dirty(state, bounds);
    }
    check_dispatch(
        p,
        bins,
        &states,
        "255/256/257 occupancy, partial dirty boxes",
    )?;
    let empty: Vec<_> = (0..OFFSPRING).map(|_| drawing(&[], full)).collect();
    check_dispatch(p, bins, &empty, "empty drawing resets old overflowing bins")?;
    Ok(())
}

fn tile_probe_starts(dim: u32, tile: u32) -> [u32; 3] {
    assert!(dim > 0 && dim <= 513 && [8, 16, 32].contains(&tile));
    let last = (dim - 1) / tile * tile;
    [tile.min(last), dim / 2 / tile * tile, last]
}

fn packed_axis_probes(dim: u32, tile: u32) -> Vec<u16> {
    let mut values = Vec::with_capacity(64);
    values.extend([0, 1, 65534, 65535]);
    for start in tile_probe_starts(dim, tile) {
        let end = (start + tile).min(dim);
        // Probe each tile edge and the adjacent guarded pixel boundaries.
        for pixel in [
            start.saturating_sub(1),
            start,
            (start + 1).min(dim),
            end,
            (end + 1).min(dim),
        ] {
            let q = pixel * 65535 / dim;
            for value in [
                q.saturating_sub(1),
                q,
                (q + 1).min(65535),
                (q + 2).min(65535),
            ] {
                values.push(u16::try_from(value).expect("packed coordinate"));
            }
        }
    }
    assert!(values.len() <= 64);
    values.sort_unstable();
    values.dedup();
    values
}

fn packed_edge_polygons(size: [u32; 2], wg: [u32; 2]) -> Vec<GpuPolygon> {
    let mut polygons = Vec::with_capacity(128);
    for (axis, (&dim, &tile)) in size.iter().zip(&wg).enumerate() {
        for q in packed_axis_probes(dim, tile) {
            let vertices = if axis == 0 {
                [[q, 0], [q, 65535], [q, 0]]
            } else {
                [[0, q], [65535, q], [0, q]]
            };
            polygons.push(polygon(vertices));
        }
    }
    assert!(polygons.len() <= 128);
    polygons
}

fn check_packed_edges(
    p: &GpuPipeline,
    size: [u32; 2],
    reference: &wgpu::ComputePipeline,
    scope: Scope,
) -> Result<usize> {
    let mut bins = new_bins(p, size, Some(reference), scope);
    let polygons = packed_edge_polygons(size, p.rasterize_wg);
    let full = [0, 0, size[0], size[1]];
    let mut states: Vec<_> = (0..OFFSPRING).map(|_| drawing(&polygons, full)).collect();
    states[1].polygons[..polygons.len()].reverse();
    check_dispatch(p, &mut bins, &states, "packed edges, full image bypass")?;
    let starts_x = tile_probe_starts(size[0], p.rasterize_wg[0]);
    let starts_y = tile_probe_starts(size[1], p.rasterize_wg[1]);
    for (state, (x, y)) in states.iter_mut().zip(starts_x.into_iter().zip(starts_y)) {
        let x = (x + 1).min(size[0] - 1);
        let y = (y + 1).min(size[1] - 1);
        set_dirty(state, [x, y, x + 1, y + 1]);
    }
    check_dispatch(
        p,
        &mut bins,
        &states,
        "packed edges, first/middle/final tiles",
    )?;
    for (state, (x, y)) in states.iter_mut().zip(starts_x.into_iter().zip(starts_y)) {
        let x = (x + 1).min(size[0] - 1);
        let y = (y + 1).min(size[1] - 1);
        set_dirty(state, [x, y, x, y]);
    }
    check_dispatch(p, &mut bins, &states, "packed edges, zero-width boxes")?;
    for (state, (x, y)) in states.iter_mut().zip(starts_x.into_iter().zip(starts_y)) {
        let x = (x + 1).min(size[0] - 1);
        let y = (y + 1).min(size[1] - 1);
        set_dirty(state, [x + 1, y + 1, x, y]);
    }
    check_dispatch(p, &mut bins, &states, "packed edges, inverted boxes")?;
    check_packed_overlap(p, &mut bins, &mut states)?;
    println!(
        "size={size:?}, wg={:?}: {} packed-edge paired checks passed",
        p.rasterize_wg, bins.dispatches
    );
    Ok(bins.dispatches)
}

fn check_packed_overlap(
    p: &GpuPipeline,
    bins: &mut Bins,
    states: &mut [GpuDrawingState],
) -> Result<()> {
    let size = bins.image_size;
    let full = [0, 0, size[0], size[1]];
    let polygons = [polygon([[0, 0], [65535, 0], [0, 65535]]); 500];
    for (state, bounds) in states.iter_mut().zip([
        full,
        [size[0] - 1, size[1] - 1, size[0], size[1]],
        [1, 1, 1, 1],
    ]) {
        *state = drawing(&polygons, bounds);
    }
    check_dispatch(
        p,
        bins,
        states,
        "500 raw count and first 256 ordered indices",
    )?;
    let polygons = [polygon([[0, 0], [65535, 0], [0, 65535]]); 1000];
    for (state, count) in states.iter_mut().zip([500, 999, 1000]) {
        *state = drawing(&polygons[..count], full);
    }
    check_dispatch(
        p,
        bins,
        states,
        "500/999/1000 raw counts and first 256 ordered indices, full image",
    )
}

fn dirty_boxes([w, h]: [u32; 2], [tw, th]: [u32; 2]) -> [DirtyBox; 30] {
    [
        [0, 0, w, h],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 1, h],
        [0, 1, w, 1],
        [tw, 0, tw, h],
        [0, th, w, th],
        [w, 0, w, h],
        [0, h, w, h],
        [2, 2, 1, 1],
        [tw + 1, 0, tw - 1, h],
        [0, th + 1, w, th - 1],
        [w - 1, h - 1, w - 1, h - 1],
        [w + 1, 0, w + 2, h],
        [0, h + 1, w, h + 2],
        [0, 0, 65535, 65535],
        [65535, 65535, 65535, 65535],
        [65535, 0, 1, h],
        [0, 65535, w, 1],
        [w, 0, 65535, h],
        [0, h, w, 65535],
        [w - 1, h - 1, w, h],
        [0, 0, 1, h],
        [0, 0, w, 1],
        [65535, 1, 65535, 2],
        [1, 65535, 2, 65535],
        [2, 0, 1, h],
        [0, 2, w, 1],
        [0, 0, 0, h],
        [0, 0, w, 0],
    ]
}
