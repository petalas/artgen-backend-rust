//! Main-thread checks of GPU binning through the real `GpuPipeline`.
//! Checks the dirty-binning contract against a scalar oracle: consumed tiles get
//! their count and ordered indices, skipped tiles retain stale words.
//! Run with `timeout 600s cargo run --release --locked --example check-gpu-binning`.
//! The paired consumed-only and full-buffers scopes compared against the frozen
//! dd52d50 GPU binner; they were removed with that source (see
//! benchmarks/evolution/v1/experiments/02-packed-rejection.md).
//!
//! Bounds: five supported workgroups, three images, three offspring, at most
//! 1000 polygons, 195 tiles per offspring, and 20 checks per buffer fixture:
//! 255 GPU dispatches in total. Only one image's buffers are live at once.
//! Buffer histories persist within each fixture.
//! Each mapping/poll has a 120-second limit. GPU work stays on this main thread.
//! Rule 3 deviation: wgpu and the scalar oracle allocate during this finite
//! checker, with one fixture live at a time and at most 195 tiles per offspring.
//! Rule 7 deviation: existing `GpuPipeline` construction has internal unbounded
//! adapter/compilation waits; the outer process timeout bounds the whole run.
//! Rules 8/9 deviation: formatting/assertion macros and wgpu's single mapping
//! callback use the existing framework interfaces, with no recursive callbacks.
//! Verification: the command above checks actual count/index readbacks. Keep the
//! bounds and outer timeout when extending cases; production APIs are unchanged.
//! `--shader PATH` installs a 1..262144-byte UTF-8 WGSL source through the
//! actual engine factory before fixture work. CLI and CPU WGSL validation run
//! before GPU initialization. The installed source survives all workgroup changes.
//! Shader runs require `--output NEWDIR` and retain exact source bytes, SHA-256
//! and completion evidence there. NEWDIR must not exist and its parent
//! must exist. A metadata record alone is not a pass; only `result.json` records
//! completed GPU checks. No output directory is created for runs without a shader.
//! Rule 10 deviation: this source preparation runs rustfmt and static inspection
//! only. The parent owns Cargo compilation and GPU proof before adoption.

use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    time::Duration,
};

use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::{
    gpu_evolver::{
        buffers::{default_gpu_params, GpuDrawingState, GpuPolygon},
        pipeline::GpuPipeline,
        EngineCapacity,
    },
    settings::{BIN_ROW_BANDS, TILE_MAX_POLYS},
};
use bytemuck::Zeroable;
use sha2::{Digest, Sha256};
use wgpu::util::DeviceExt;

const WORKGROUPS: [[u32; 2]; 5] = [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];
const IMAGE_SIZES: [[u32; 2]; 3] = [[64, 48], [65, 51], [17, 19]];
const OFFSPRING: usize = 3;
const BIN_CAPACITY: usize = 256;
const TIMEOUT: Duration = Duration::from_secs(120);
const MAX_SOURCE_BYTES: usize = 256 * 1024;
const USAGE: &str =
    "usage: check-gpu-binning [--shader PATH --output NEWDIR]\n\
     --shader requires a nonempty UTF-8 WGSL file at most 256 KiB and --output.\n\
     NEWDIR must not exist and its parent must exist.";
const CHECKS: usize = 255;
type DirtyBox = [u32; 4];

struct Config {
    shader: Option<OptionalSource>,
}

struct OptionalSource {
    path: PathBuf,
    output: PathBuf,
    text: String,
    sha256: String,
}

struct BinBuffers {
    counts: wgpu::Buffer,
    indices: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Bins {
    image_size: [u32; 2],
    buffers: BinBuffers,
    expected: Vec<u32>,
    dispatches: usize,
}

fn main() -> Result<()> {
    let Some(config) = parse_config(std::env::args_os().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    ensure!(usize::try_from(TILE_MAX_POLYS)? == BIN_CAPACITY);
    let reference = [0, 0, 0, 255].repeat(65 * 51);
    let initial = vec![GpuDrawingState::zeroed()];
    let mut pipeline =
        futures_lite::future::block_on(GpuPipeline::new(
            EngineCapacity::new(1, 64),
            65,
            51,
            &reference,
            &initial,
        ));
    ensure!(usize::try_from(pipeline.offspring_capacity)? >= OFFSPRING);
    let evidence = if let Some(source) = &config.shader {
        pipeline
            .shader_variants()
            .set_bin_shader(&source.text)
            .map_err(anyhow::Error::msg)
            .context("invalid bin shader for the actual GPU factory")?;
        Some((retain_source(source)?, source))
    } else {
        None
    };
    let mut dispatches = 0;
    for workgroup in WORKGROUPS {
        pipeline.set_rasterize_wg(workgroup);
        for image_size in IMAGE_SIZES {
            dispatches += check_fixture(&pipeline, image_size)?;
        }
    }
    ensure!(dispatches == CHECKS, "check count changed");
    if let Some((directory, source)) = evidence {
        record_result(&directory, source, dispatches)?;
    }
    println!("GPU dirty-binning checks passed: 5 workgroups, 3 images, {dispatches} dispatches");
    Ok(())
}

fn parse_config(args: impl Iterator<Item = OsString>) -> Result<Option<Config>> {
    let args: Vec<_> = args.take(5).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        return Ok(None);
    }
    ensure!(args.len() <= 4 && args.len().is_multiple_of(2), USAGE);
    let mut path = None;
    let mut output = None;
    for pair in args.as_chunks::<2>().0 {
        let flag = pair[0].to_str().context("CLI flags must be UTF-8")?;
        let value = pair[1].to_str().context("CLI values must be UTF-8")?;
        match flag {
            "--shader" => {
                ensure!(path.is_none(), "duplicate --shader");
                ensure!(
                    !value.is_empty() && !value.starts_with("--"),
                    "missing shader path"
                );
                path = Some(PathBuf::from(value));
            }
            "--output" => {
                ensure!(output.is_none(), "duplicate --output");
                ensure!(
                    !value.is_empty() && !value.starts_with("--"),
                    "missing output path"
                );
                output = Some(PathBuf::from(value));
            }
            _ => bail!("unknown flag {flag:?}; {USAGE}"),
        }
    }
    let shader = match (path, output) {
        (None, None) => None,
        (Some(path), Some(output)) => Some(read_source(path, output)?),
        _ => bail!("--shader and --output must be supplied together; {USAGE}"),
    };
    Ok(Some(Config { shader }))
}

fn read_source(path: PathBuf, output: PathBuf) -> Result<OptionalSource> {
    let file = File::open(&path).with_context(|| format!("open shader {}", path.display()))?;
    let metadata = file.metadata()?;
    ensure!(metadata.is_file(), "shader must be a regular file");
    ensure!(
        metadata.len() <= MAX_SOURCE_BYTES as u64,
        "shader exceeds 256 KiB"
    );
    // Bound the opened target itself, including symlinks and files growing after metadata.
    let mut bytes = Vec::with_capacity(MAX_SOURCE_BYTES + 1);
    file.take((MAX_SOURCE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_SOURCE_BYTES,
        "shader must contain 1..262144 bytes"
    );
    let text = String::from_utf8(bytes).context("shader must be UTF-8")?;
    for marker in ["const TILE_W: u32 = 16;", "const TILE_H: u32 = 16;"] {
        ensure!(
            text.matches(marker).count() == 1,
            "invalid bin specialization marker {marker:?}"
        );
    }
    let module = wgpu::naga::front::wgsl::parse_str(&text)
        .map_err(|error| anyhow::anyhow!("invalid WGSL: {}", error.emit_to_string(&text)))?;
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| anyhow::anyhow!("invalid WGSL: {error}"))?;
    match fs::symlink_metadata(&output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect output path"),
        Ok(_) => bail!("output must be a new directory"),
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    ensure!(
        fs::metadata(parent)?.is_dir(),
        "output parent must be an existing directory"
    );
    Ok(OptionalSource {
        path,
        output,
        sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        text,
    })
}

fn retain_source(source: &OptionalSource) -> Result<PathBuf> {
    let directory = &source.output;
    fs::create_dir(directory)?;
    write_new(&directory.join("source.wgsl"), source.text.as_bytes())?;
    let metadata = serde_json::json!({
        "status": "source-installed-checks-not-complete",
        "source_path": source.path, "source_sha256": source.sha256,
        "source_bytes": source.text.len(), "source_copy": "source.wgsl",
        "benchmark_validation": false, "compilation_options": "production-default",
        "expected_checks": CHECKS, "expected_gpu_dispatches": CHECKS,
    });
    write_new(
        &directory.join("metadata.json"),
        &serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(directory.clone())
}

fn record_result(directory: &Path, source: &OptionalSource, checks: usize) -> Result<()> {
    let result = serde_json::json!({
        "status": "passed", "source_sha256": source.sha256,
        "checks": checks, "gpu_dispatches": checks,
    });
    write_new(
        &directory.join("result.json"),
        &serde_json::to_vec_pretty(&result)?,
    )?;
    println!("Shader proof evidence: {}", directory.display());
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

fn new_bins(p: &GpuPipeline, image_size: [u32; 2]) -> Bins {
    assert!(IMAGE_SIZES.contains(&image_size));
    assert!(WORKGROUPS.contains(&p.rasterize_wg()));
    let tiles =
        image_size[0].div_ceil(p.rasterize_wg()[0]) * image_size[1].div_ceil(p.rasterize_wg()[1]);
    assert!(tiles <= 195);
    let slots = usize::try_from(tiles).expect("at most 195 tiles") * OFFSPRING;
    let expected: Vec<u32> = (0..slots * (BIN_CAPACITY + 1))
        .map(|i| 0xA000_0000 | u32::try_from(i).expect("bounded buffer word index"))
        .collect();
    Bins {
        image_size,
        buffers: new_bin_buffers(p, &expected, slots),
        expected,
        dispatches: 0,
    }
}

fn new_bin_buffers(p: &GpuPipeline, poison: &[u32], slots: usize) -> BinBuffers {
    assert!(slots <= 195 * OFFSPRING);
    assert_eq!(poison.len(), slots * (BIN_CAPACITY + 1));
    assert!(std::mem::size_of_val(poison) < 2 * 1024 * 1024);
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
    // The binner also appends raster work; a one-word buffer keeps the count
    // and drops entries, leaving the engine's worklist untouched.
    let raster_work = p.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("checker raster work"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let entries = [
        (0, &p.working_states_buf),
        (1, &indices),
        (2, &counts),
        (3, &p.chain_states_buf),
        (4, &raster_work),
    ]
    .map(|(binding, buffer)| {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        });
    // The engine's tile-group table, at tile group 0 (offset 0 in `dispatch`).
    let tile_group = wgpu::BindGroupEntry {
        binding: 5,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &p.tile_group_buf,
            offset: 0,
            size: std::num::NonZeroU64::new(4),
        }),
    };
    let entries: Vec<_> = entries.into_iter().chain([tile_group]).collect();
    let bind_group = p.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("checker bin bindings"),
        layout: &p.bin_polygons_pipeline().get_bind_group_layout(0),
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
        pass.set_bind_group(0, &bins.bind_group, &[p.tile_group_offset(0)]);
        pass.set_bind_group(1, &p.params_bind_group, &[]);
        pass.dispatch_workgroups(u32::try_from(OFFSPRING)?, BIN_ROW_BANDS, 1);
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
        p.bin_polygons_pipeline(),
        states,
    )?;
    let slots = usize::try_from(bins.buffers.counts.size() / 4)?;
    let tiles = slots / OFFSPRING;
    for (offspring, state) in states.iter().enumerate() {
        let full_bins = scalar_full_bins(state, bins.image_size, p.rasterize_wg());
        ensure!(full_bins.len() == tiles);
        for (tile, layers) in full_bins.iter().enumerate() {
            if !consumed(state, tile, bins.image_size, p.rasterize_wg()) {
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
                p.rasterize_wg()
            );
        }
    }
    bins.dispatches += 1;
    Ok(())
}

fn consumed(state: &GpuDrawingState, tile: usize, size: [u32; 2], wg: [u32; 2]) -> bool {
    let tile = u32::try_from(tile).expect("at most 528 tiles");
    let columns = size[0].div_ceil(wg[0]);
    let x = tile % columns * wg[0];
    let y = tile / columns * wg[1];
    // Direct rasterizer predicate, independent of the candidate's tile rectangle.
    let [min_x, min_y, max_x, max_y] = state.dirty_box();
    (x + wg[0]).min(size[0]) > min_x && x < max_x && (y + wg[1]).min(size[1]) > min_y && y < max_y
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
    state.set_dirty_box(bounds);
}

fn drawing(polygons: &[GpuPolygon], bounds: DirtyBox) -> GpuDrawingState {
    assert!(polygons.len() <= 1000);
    let mut state = GpuDrawingState::zeroed();
    state.polygon_count = u32::try_from(polygons.len()).expect("at most 1000 polygons");
    state.polygons[..polygons.len()].copy_from_slice(polygons);
    set_dirty(&mut state, bounds);
    state
}

fn check_fixture(p: &GpuPipeline, size: [u32; 2]) -> Result<usize> {
    let mut bins = new_bins(p, size);
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
        p.rasterize_wg(), bins.dispatches
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
    let boxes = dirty_boxes(bins.image_size, p.rasterize_wg());
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

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "const TILE_W: u32 = 16;\nconst TILE_H: u32 = 16;\n@compute @workgroup_size(1) fn main() {}\n";
    const VALID_SHA256: &str = "0cdfed85897b82314e8521faa07986c9f9c238245723be3559bfc64834d061da";

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Result<Self> {
            let path = Path::new(".cache/evolution-overnight-20261024/hot-bin-checker/cpu-tests")
                .join(uuid::Uuid::new_v4().to_string());
            fs::create_dir_all(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove this test's directory");
        }
    }

    fn shader_args(path: &Path, output: &Path) -> Vec<OsString> {
        vec![
            "--shader".into(),
            path.into(),
            "--output".into(),
            output.into(),
        ]
    }

    #[test]
    fn default_cli_and_help() -> Result<()> {
        let config = parse_config(std::iter::empty())?.unwrap();
        assert!(config.shader.is_none());
        for flag in ["--help", "-h"] {
            assert!(parse_config([OsString::from(flag)].into_iter())?.is_none());
        }
        Ok(())
    }

    #[test]
    fn invalid_cli_is_rejected() {
        let cases: &[&[&str]] = &[
            &["--scope"],
            &["--scope", "strict"],
            &["--scope", "full-buffers"],
            &["--unknown", "x"],
            &["--shader", "a", "--shader", "b"],
            &["--output", "a", "--output", "b"],
            &["--shader", "a"],
            &["--output", "a"],
            &["--shader", "", "--output", "a"],
            &["--shader", "--output", "--output", "a"],
            &["--shader", "a", "--output", ""],
            &["--help", "--shader", "a"],
            &["--shader", "a", "--output", "b", "extra"],
        ];
        for args in cases {
            assert!(
                parse_config(args.iter().map(OsString::from)).is_err(),
                "{args:?}"
            );
        }
    }

    #[test]
    fn selected_source_retains_exact_evidence() -> Result<()> {
        let directory = TestDirectory::new()?;
        let path = directory.0.join("input.wgsl");
        fs::write(&path, VALID)?;
        let output = directory.0.join("proof");
        let config = parse_config(shader_args(&path, &output).into_iter())?.unwrap();
        let source = config.shader.unwrap();
        assert_eq!(source.sha256, VALID_SHA256);
        assert_eq!(source.text.as_bytes(), VALID.as_bytes());
        assert!(!output.exists());
        fs::write(&path, "changed after validation")?;
        let retained = retain_source(&source)?;
        assert_eq!(fs::read(retained.join("source.wgsl"))?, VALID.as_bytes());
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(retained.join("metadata.json"))?)?;
        assert_eq!(metadata["source_sha256"], VALID_SHA256);
        assert_eq!(metadata["source_bytes"], VALID.len());
        assert_eq!(metadata["benchmark_validation"], false);
        assert_eq!(metadata["expected_checks"], CHECKS);
        assert!(!retained.join("result.json").exists());
        record_result(&retained, &source, CHECKS)?;
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(retained.join("result.json"))?)?;
        assert_eq!(result["status"], "passed");
        assert_eq!(result["source_sha256"], VALID_SHA256);
        assert_eq!(result["checks"], CHECKS);
        assert_eq!(result["gpu_dispatches"], CHECKS);
        Ok(())
    }

    #[test]
    fn shader_flags_accept_either_order() -> Result<()> {
        let directory = TestDirectory::new()?;
        let path = directory.0.join("input.wgsl");
        fs::write(&path, VALID)?;
        let args = vec![
            "--output".into(),
            directory.0.join("proof").into_os_string(),
            "--shader".into(),
            path.into_os_string(),
        ];
        let config = parse_config(args.into_iter())?.unwrap();
        assert_eq!(config.shader.unwrap().sha256, VALID_SHA256);
        Ok(())
    }

    #[test]
    fn invalid_sources_fail_before_output_creation() -> Result<()> {
        let directory = TestDirectory::new()?;
        let path = directory.0.join("input.wgsl");
        let output = directory.0.join("proof");
        assert!(parse_config(shader_args(&path, &output).into_iter()).is_err());
        for bytes in [
            vec![],
            vec![0xff],
            vec![b' '; MAX_SOURCE_BYTES + 1],
            format!("{VALID} this is malformed WGSL").into_bytes(),
            VALID.replace("const TILE_W: u32 = 16;", "").into_bytes(),
            format!("{VALID}const TILE_H: u32 = 16;").into_bytes(),
            VALID
                .replace("fn main() {}", "fn main() { let x: u32 = true; }")
                .into_bytes(),
        ] {
            fs::write(&path, bytes)?;
            assert!(parse_config(shader_args(&path, &output).into_iter()).is_err());
            assert!(!output.exists());
        }
        assert!(parse_config(shader_args(&directory.0, &output).into_iter()).is_err());
        fs::write(&path, VALID)?;
        assert!(
            parse_config(shader_args(&path, &directory.0.join("missing/proof")).into_iter())
                .is_err()
        );
        assert!(!directory.0.join("missing").exists());
        Ok(())
    }

    #[test]
    fn size_limit_is_inclusive() -> Result<()> {
        let directory = TestDirectory::new()?;
        let path = directory.0.join("input.wgsl");
        let output = directory.0.join("proof");
        let mut bytes = VALID.as_bytes().to_vec();
        bytes.resize(MAX_SOURCE_BYTES, b' ');
        fs::write(&path, &bytes)?;
        let config = parse_config(shader_args(&path, &output).into_iter())?.unwrap();
        assert_eq!(config.shader.unwrap().text.as_bytes(), bytes);
        assert!(!output.exists());
        Ok(())
    }

    #[test]
    fn existing_output_is_never_overwritten() -> Result<()> {
        let directory = TestDirectory::new()?;
        let path = directory.0.join("input.wgsl");
        let output = directory.0.join("proof");
        fs::write(&path, VALID)?;
        fs::write(&output, b"keep this file")?;
        assert!(parse_config(shader_args(&path, &output).into_iter()).is_err());
        assert_eq!(fs::read(&output)?, b"keep this file");
        fs::remove_file(&output)?;
        fs::create_dir(&output)?;
        assert!(parse_config(shader_args(&path, &output).into_iter()).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_targets_obey_source_bound_and_output_exclusion() -> Result<()> {
        use std::os::unix::fs::symlink;
        let directory = TestDirectory::new()?;
        let path = directory.0.join("input.wgsl");
        let link = directory.0.join("input-link.wgsl");
        let output = directory.0.join("proof");
        fs::write(&path, vec![b' '; MAX_SOURCE_BYTES + 1])?;
        symlink("input.wgsl", &link)?;
        assert!(parse_config(shader_args(&link, &output).into_iter()).is_err());
        fs::write(&path, VALID)?;
        symlink("missing-directory", &output)?;
        assert!(parse_config(shader_args(&link, &output).into_iter()).is_err());
        assert!(fs::symlink_metadata(&output)?.file_type().is_symlink());
        fs::remove_file(&output)?;
        let config = parse_config(shader_args(&link, &output).into_iter())?.unwrap();
        assert_eq!(config.shader.unwrap().sha256, VALID_SHA256);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_cli_is_rejected() {
        use std::os::unix::ffi::OsStringExt;
        let invalid = OsString::from_vec(vec![0xff]);
        assert!(parse_config([invalid.clone(), "x".into()].into_iter()).is_err());
        assert!(parse_config(
            [
                "--shader".into(),
                invalid.clone(),
                "--output".into(),
                "new".into()
            ]
            .into_iter()
        )
        .is_err());
        assert!(parse_config(["--output".into(), invalid].into_iter()).is_err());
    }
}
