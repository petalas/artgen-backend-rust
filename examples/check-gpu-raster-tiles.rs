//! Direct raster coverage and integer-error proof. Run the built example under
//! `timeout 600s ... --reference AE1 --candidate T2=PATH --candidate T4=PATH --output NEWDIR`.
//! Fault modes intentionally fail after three successful real GPU dispatches.
//! All GPU calls stay on main. No timing claims, selection, or CPU float renderer.
//! Fixed bounds: 180 fixtures, 256 offspring, 27 tiles, 256 indices/tile,
//! 287x33 pixels, four parents. Largest checker buffer: 7,077,888 bytes.
//! POT deviations: driver construction/compilation needs the outer 600s timeout;
//! maps/scopes have 120s bounds. wgpu, JSON, and source compilation allocate;
//! live resources are bounded to one fixture, the factory's finite 85-entry
//! pipeline cache, and one instrumented pipeline.
//! Source rewriting is limited to checked unique anchors. One map callback has
//! one bounded channel. Diagnostics/runtime proof require the parent's CPU grant.

use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::{
    gpu_evolver::{
        benchmark::benchmark_raster_tiles_per_group_from_source,
        buffers::{
            default_gpu_params, GpuDrawingState, GpuParams, GpuPolygon, GPU_DRAWING_STATE_SIZE,
        },
        pipeline::{BenchmarkRasterResources, GpuPipeline},
        GpuEvolver,
    },
    models::{color::Color, drawing::Drawing, point::Point, polygon::Polygon},
    mutation_params::MutationParams,
};
use bytemuck::Zeroable;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    future::Future,
    io::{Read, Write},
    num::NonZeroU64,
    path::{Path, PathBuf},
    sync::mpsc,
    task::{Context as TaskContext, Poll, Waker},
    time::{Duration, Instant},
};

const WGS: [[u32; 2]; 5] = [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];
const CAP: usize = 256;
const TILES: usize = 27;
const PAIRS: usize = CAP * TILES;
const POLYS: usize = 256;
const WIDTH: u32 = 287;
const HEIGHT: u32 = 33;
const PIXELS: usize = WIDTH as usize * HEIGHT as usize;
const ERRORS: usize = CAP * 2;
const VISITS: usize = PAIRS + 1;
const WORDS: usize = 3 * ERRORS + VISITS;
const MAX_SOURCE: usize = 256 * 1024;
const MAX_ARTIFACTS: usize = 64 * 1024 * 1024;
const WAIT: Duration = Duration::from_secs(120);
const OUTER: Duration = Duration::from_secs(600);
const AE1: &str = "ae1fd75fdc118debf2a50cbad699e513b5af0dcbfa3754b45966f7a8ee925af3";
const T2: &str = "c3b2d775f76fcd9f8ebadf77d957cad40663fb766641e662174e245f6de80da3";
const T4: &str = "64d4584f32865fed0d15f1d422d4afcbd482b175755bbc606eb9b1879b36feb4";
const DECL: &str =
    "@group(0) @binding(5) var<storage, read>       chain_states:       array<DrawingState>;";
const COORD: &str = "let tile_x = wid.x * RASTER_TILES_PER_GROUP + tile_offset;";
const TILE_ID: &str = "let tile_id = wid.y * num_tiles_x + tile_x;";
type Dirty = [u32; 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Fault {
    None,
    Missing,
    Duplicate,
}
struct Source {
    name: String,
    path: PathBuf,
    text: String,
    t: u32,
}
struct Options {
    sources: Vec<Source>,
    output: PathBuf,
    fault: Fault,
}
#[derive(Clone, Copy, Debug, Serialize)]
struct Fixture {
    image: [u32; 2],
    wg: [u32; 2],
    lambda: u32,
    parents: u32,
    rotation: usize,
    fault: bool,
}
impl Fixture {
    fn active(self) -> usize {
        (self.lambda * self.parents) as usize
    }
    fn grid(self) -> [u32; 2] {
        [
            self.image[0].div_ceil(self.wg[0]),
            self.image[1].div_ceil(self.wg[1]),
        ]
    }
    fn tiles(self) -> usize {
        let [x, y] = self.grid();
        (x * y) as usize
    }
}
struct Scratch {
    states: Vec<GpuDrawingState>,
    counts: Vec<u32>,
    indices: Vec<u32>,
    parents: Vec<u32>,
    reference: Vec<u8>,
    initial: [u32; ERRORS],
    expected: Vec<u32>,
    old: [u32; CAP],
    actual: Vec<u32>,
}
struct Buffers {
    states: wgpu::Buffer,
    counts: wgpu::Buffer,
    indices: wgpu::Buffer,
    parents: wgpu::Buffer,
    errors: [wgpu::Buffer; 3],
    visits: wgpu::Buffer,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}
struct Prepared {
    groups: [wgpu::BindGroup; 3],
    params: wgpu::BindGroup,
    coverage: wgpu::ComputePipeline,
}
struct Rasters<'a> {
    pipeline: &'a GpuPipeline,
    buffers: &'a Buffers,
    baseline: &'a BenchmarkRasterResources,
    candidate: &'a BenchmarkRasterResources,
    prepared: &'a Prepared,
}
struct Evidence {
    dir: PathBuf,
    events: File,
    bytes: usize,
    submitted: usize,
    passed: usize,
    negative: bool,
}

fn main() -> Result<()> {
    let Some(options) = options()? else {
        return Ok(());
    };
    let mut evidence = Evidence::new(&options.output)?;
    let result = run(&options, &mut evidence, Instant::now() + OUTER);
    evidence.event(&json!({"event":"result", "ok":result.is_ok(), "fixtures_passed":evidence.passed,
        "dispatches":evidence.submitted*3, "negative_detected":evidence.negative,
        "exit_status":i32::from(result.is_err()), "error":result.as_ref().err().map(|e|format!("{e:#}"))}))?;
    result
}

struct RawOptions {
    reference: Option<PathBuf>,
    candidates: Vec<(String, PathBuf)>,
    output: Option<PathBuf>,
    fault: Fault,
}
fn options() -> Result<Option<Options>> {
    let args: Vec<OsString> = env::args_os().skip(1).take(12).collect();
    let Some(RawOptions {
        reference,
        candidates,
        output,
        fault,
    }) = parse_args(&args)?
    else {
        return Ok(None);
    };
    let output = output.context("missing output")?;
    ensure!(
        !output.exists() && output.file_name().is_some(),
        "output must be a new directory"
    );
    ensure!(
        output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .is_dir(),
        "output parent must exist"
    );
    let mut sources = Vec::with_capacity(3);
    sources.push(read_source(
        "T1".into(),
        &reference.context("missing reference")?,
    )?);
    for (name, path) in candidates {
        sources.push(read_source(name, &path)?);
    }
    validate_sources(&sources, fault)?;
    Ok(Some(Options {
        sources,
        output,
        fault,
    }))
}

fn parse_args(args: &[OsString]) -> Result<Option<RawOptions>> {
    if args.len() == 1 && args[0] == "--help" {
        println!("--reference PATH --candidate T2=PATH --candidate T4=PATH --output NEWDIR [--fault missing|duplicate]");
        return Ok(None);
    }
    ensure!(
        args.len() <= 10 && args.len() % 2 == 0,
        "expected bounded flag/value pairs"
    );
    let mut reference = None;
    let mut candidates = Vec::with_capacity(2);
    let mut output = None;
    let mut fault = Fault::None;
    for pair in args.as_chunks::<2>().0 {
        let flag = pair[0].to_str().context("flag must be UTF-8")?;
        let value = pair[1].to_str().context("value must be UTF-8")?;
        match flag {
            "--reference" => {
                ensure!(reference.is_none(), "duplicate reference");
                reference = Some(PathBuf::from(value));
            }
            "--output" => {
                ensure!(output.is_none(), "duplicate output");
                output = Some(PathBuf::from(value));
            }
            "--candidate" => {
                ensure!(candidates.len() < 2, "at most two candidates");
                let (name, path) = value
                    .split_once('=')
                    .context("candidate must be name=PATH")?;
                ensure!(
                    !name.is_empty()
                        && name.len() <= 32
                        && name
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
                    "invalid candidate name"
                );
                candidates.push((name.to_owned(), PathBuf::from(path)));
            }
            "--fault" => {
                ensure!(fault == Fault::None, "duplicate fault");
                fault = match value {
                    "missing" => Fault::Missing,
                    "duplicate" => Fault::Duplicate,
                    _ => bail!("invalid fault"),
                };
            }
            _ => bail!("unknown flag {flag}"),
        }
    }
    Ok(Some(RawOptions {
        reference,
        candidates,
        output,
        fault,
    }))
}

fn validate_sources(sources: &[Source], fault: Fault) -> Result<()> {
    ensure!(
        sources[0].t == 1 && hash(sources[0].text.as_bytes()) == AE1,
        "reference must be exact AE1"
    );
    for i in 1..sources.len() {
        let s = &sources[i];
        ensure!([2, 4].contains(&s.t), "candidate T must be 2 or 4");
        ensure!(
            hash(s.text.as_bytes()) == if s.t == 2 { T2 } else { T4 },
            "candidate differs from frozen source"
        );
        ensure!(
            sources[..i]
                .iter()
                .all(|p| p.name != s.name && p.path != s.path && p.t != s.t),
            "duplicate source/name/T"
        );
        instrument(&fault_source(&s.text, fault)?, s.t)?;
    }
    ensure!(sources.iter().any(|s| s.t == 2), "T2 candidate required");
    ensure!(
        fault != Fault::None || sources.len() == 3,
        "positive matrix requires T2 and T4"
    );
    instrument(&sources[0].text, 1)?;
    Ok(())
}

fn read_source(name: String, path: &Path) -> Result<Source> {
    let path = path
        .canonicalize()
        .with_context(|| format!("source {}", path.display()))?;
    let metadata = fs::metadata(&path)?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_SOURCE as u64,
        "source must be a regular file within 256 KiB"
    );
    let mut bytes = Vec::with_capacity(MAX_SOURCE + 1);
    File::open(&path)?
        .take((MAX_SOURCE + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_SOURCE,
        "source exceeds 256 KiB or is empty"
    );
    let text = String::from_utf8(bytes).context("source must be UTF-8")?;
    let t = benchmark_raster_tiles_per_group_from_source(&text).map_err(anyhow::Error::msg)?;
    Ok(Source {
        name,
        path,
        text,
        t,
    })
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
impl Evidence {
    fn new(dir: &Path) -> Result<Self> {
        fs::create_dir(dir)?;
        let events = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("events.jsonl"))?;
        Ok(Self {
            dir: dir.to_owned(),
            events,
            bytes: 0,
            submitted: 0,
            passed: 0,
            negative: false,
        })
    }
    fn event(&mut self, value: &Value) -> Result<()> {
        let bytes = serde_json::to_vec(value)?;
        ensure!(
            self.bytes + bytes.len() < MAX_ARTIFACTS,
            "64 MiB artifact budget exhausted"
        );
        self.events.write_all(&bytes)?;
        self.events.write_all(b"\n")?;
        self.events.flush()?;
        self.bytes += bytes.len() + 1;
        Ok(())
    }
    fn source(&mut self, name: &str, text: &str) -> Result<()> {
        ensure!(
            self.bytes + text.len() <= MAX_ARTIFACTS,
            "source artifact budget exhausted"
        );
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.dir.join(format!("{name}.wgsl")))?;
        file.write_all(text.as_bytes())?;
        file.flush()?;
        self.bytes += text.len();
        Ok(())
    }
}

fn once(source: &str, from: &str, to: &str) -> Result<String> {
    ensure!(
        source.matches(from).count() == 1,
        "expected unique source anchor: {from}"
    );
    Ok(source.replacen(from, to, 1))
}
fn fault_source(source: &str, fault: Fault) -> Result<String> {
    match fault {
        Fault::None=>Ok(source.to_owned()),
        Fault::Missing=>once(source,"if (skip_mask & (1u << tile_offset)) != 0u {",
            "if (skip_mask & (1u << tile_offset)) != 0u || (wid.x * RASTER_TILES_PER_GROUP + tile_offset) == 1u {"),
        Fault::Duplicate=>once(source,COORD,
            "let original_tile_x = wid.x * RASTER_TILES_PER_GROUP + tile_offset;\n        let tile_x = select(original_tile_x, 0u, original_tile_x == 1u);"),
    }
}
fn instrument(source: &str, t: u32) -> Result<String> {
    ensure!(
        source.matches("chain_states").count() == 1,
        "binding5 must be unused"
    );
    let source = once(
        source,
        DECL,
        "@group(0) @binding(5) var<storage, read_write> checker_visits: array<atomic<u32>>;",
    )?;
    let (anchor, x) = if t == 1 {
        ("let tile_id = wid.y * num_tiles_x + wid.x;", "wid.x")
    } else {
        (TILE_ID, "tile_x")
    };
    let addition = format!(
        r"{anchor}
        if local_idx == 0u {{
            let sentinel = arrayLength(&checker_visits) - 1u;
            var visit = sentinel;
            if {x} < num_tiles_x && wid.y < num_tiles_y && chain_id < params.chain_count_param * lambda {{
                let mapped = chain_id * num_tiles + tile_id;
                if mapped < sentinel {{ visit = mapped; }}
            }}
            atomicAdd(&checker_visits[visit], 1u);
        }}"
    );
    once(&source, anchor, &addition)
}

fn word(index: usize) -> u32 {
    u32::try_from(index).expect("bounded checker index fits u32")
}
fn packed(x: u32, y: u32) -> u32 {
    assert!(x <= 65535 && y <= 65535);
    x | y << 16
}
fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
fn dirty(f: Fixture, slot: usize) -> Dirty {
    let [width, height] = f.image;
    let [tile_width, tile_height] = f.wg;
    if f.fault {
        return [0, 0, width, height];
    }
    match (slot + f.rotation) % 10 {
        0 => [0, 0, width, height],
        1 => [width, height, width, height],
        2 => [0, 0, tile_width.min(width), tile_height],
        3 => [(f.grid()[0] - 1) * tile_width, 0, width, height],
        4 => [tile_width, 0, width, height],
        5 => [0, 0, (width.saturating_sub(tile_width)).max(1), height],
        6 => [1, 1, 1, height],
        7 => [3, 1, 2, height],
        8 => [width + 1, 0, width + 4, height],
        _ => [0, tile_height, width, (2 * tile_height).min(height)],
    }
}
fn consumed(f: Fixture, b: Dirty, tile: usize) -> bool {
    let nx = f.grid()[0];
    let tile = word(tile);
    let x = tile % nx * f.wg[0];
    let y = tile / nx * f.wg[1];
    (x + f.wg[0]).min(f.image[0]) > b[0]
        && x < b[2]
        && (y + f.wg[1]).min(f.image[1]) > b[1]
        && y < b[3]
}
fn l1(pixel: u32, reference: &[u8]) -> u32 {
    pixel.to_le_bytes()[..3]
        .iter()
        .zip(reference)
        .map(|(a, b)| u32::from(a.abs_diff(*b)))
        .sum()
}
impl Scratch {
    fn new() -> Self {
        Self {
            states: vec![GpuDrawingState::zeroed(); CAP],
            counts: vec![0; PAIRS],
            indices: vec![0; PAIRS * POLYS],
            parents: vec![0; 4 * PIXELS],
            reference: vec![0; 4 * PIXELS],
            initial: [0; ERRORS],
            expected: vec![0; VISITS],
            old: [0; CAP],
            actual: vec![0; WORDS],
        }
    }
    fn prepare(&mut self, f: Fixture) {
        assert!(f.active() <= CAP && f.tiles() <= TILES);
        self.expected.fill(0);
        self.old.fill(0);
        self.counts.fill(0xabab_abab);
        self.initial = std::array::from_fn(|i| {
            if f.fault {
                0
            } else {
                0x0100_0000 + word(i) * 97
            }
        });
        for (i, rgba) in self.reference.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            rgba.copy_from_slice(
                &if f.fault {
                    u32::MAX
                } else {
                    mix(word(i) ^ 0x6578_6163) | 0xff00_0000
                }
                .to_le_bytes(),
            );
        }
        let pixels = (f.image[0] * f.image[1]) as usize;
        for parent in 0..4 {
            for i in 0..pixels {
                self.parents[parent * pixels + i] = if f.fault {
                    u32::MAX
                } else {
                    mix(word(i) ^ ((word(parent) + 1) * 0x0123_4567)) | 0xff00_0000
                };
            }
        }
        for slot in 0..f.active() {
            let bbox = dirty(f, slot);
            let state = &mut self.states[slot];
            state.polygon_count = if f.fault { 0 } else { 256 };
            state.fitness_bits = packed(bbox[0], bbox[1]);
            state.stagnation_counter = packed(bbox[2], bbox[3]);
            fill_polygons(state, slot);
            for tile in 0..f.tiles() {
                let pair = slot * f.tiles() + tile;
                self.counts[pair] = if f.fault {
                    0
                } else {
                    [0, 1, 255, 256][(slot + tile + f.rotation) % 4]
                };
                for i in 0..POLYS {
                    self.indices[pair * POLYS + i] = word((i * 17 + tile * 7 + slot * 3) % 256);
                }
                self.expected[pair] = u32::from(consumed(f, bbox, tile));
            }
            self.old[slot] = expected_old(f, slot, &self.parents, &self.reference);
        }
    }
}
fn fill_polygons(state: &mut GpuDrawingState, slot: usize) {
    for (i, p) in state.polygons[..256].iter_mut().enumerate() {
        let vertices = if i % 2 == 0 {
            [packed(0, 0), packed(65535, 0), packed(0, 65535)]
        } else {
            [packed(65535, 65535), packed(0, 65535), packed(65535, 0)]
        };
        *p = GpuPolygon {
            data: [
                mix(word(slot * 256 + i)) | 0x4100_0000,
                vertices[0],
                vertices[1],
                vertices[2],
            ],
        };
    }
}
fn expected_old(f: Fixture, slot: usize, parents: &[u32], reference: &[u8]) -> u32 {
    let pixels = (f.image[0] * f.image[1]) as usize;
    let parent = slot / f.lambda as usize;
    let mut total = 0;
    // Enumerate original (tile,pixel) coordinates. Never use grouped-X mapping.
    for tile in 0..f.tiles() {
        if !consumed(f, dirty(f, slot), tile) {
            continue;
        }
        let tx = word(tile) % f.grid()[0] * f.wg[0];
        let ty = word(tile) / f.grid()[0] * f.wg[1];
        for y in ty..(ty + f.wg[1]).min(f.image[1]) {
            for x in tx..(tx + f.wg[0]).min(f.image[0]) {
                let i = (y * f.image[0] + x) as usize;
                let ri = (y * WIDTH + x) as usize * 4;
                total += l1(parents[parent * pixels + i], &reference[ri..ri + 3]);
            }
        }
    }
    total
}

fn bytes(words: usize) -> u64 {
    (words * 4) as u64
}
fn storage(d: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}
impl Buffers {
    fn new(p: &GpuPipeline) -> Self {
        let d = &p.device;
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("checker params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("checker readback"),
            size: bytes(WORDS),
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let texture = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("checker reference"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            states: storage(d, "checker states", (CAP * GPU_DRAWING_STATE_SIZE) as u64),
            counts: storage(d, "checker counts", bytes(PAIRS)),
            indices: storage(d, "checker indices", bytes(PAIRS * POLYS)),
            parents: storage(d, "checker cached parents", bytes(4 * PIXELS)),
            errors: std::array::from_fn(|_| storage(d, "checker errors", bytes(ERRORS))),
            visits: storage(d, "checker visits", bytes(VISITS)),
            params,
            staging,
            texture,
            view,
        }
    }
    fn upload(&self, p: &GpuPipeline, f: Fixture, s: &Scratch) {
        let mut params = default_gpu_params(f.image[0], f.image[1], f.parents);
        params.lambda = f.lambda;
        p.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        p.queue
            .write_buffer(&self.states, 0, bytemuck::cast_slice(&s.states));
        p.queue
            .write_buffer(&self.counts, 0, bytemuck::cast_slice(&s.counts));
        p.queue
            .write_buffer(&self.indices, 0, bytemuck::cast_slice(&s.indices));
        p.queue
            .write_buffer(&self.parents, 0, bytemuck::cast_slice(&s.parents));
        for b in &self.errors {
            p.queue.write_buffer(b, 0, bytemuck::cast_slice(&s.initial));
        }
        p.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &s.reference,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 4),
                rows_per_image: Some(HEIGHT),
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    }
}
fn storage_entry(binding: u32, read_only: bool, minimum: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(minimum),
        },
        count: None,
    }
}
fn coverage_layout(d: &wgpu::Device) -> wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("coverage binding5 only"),
        entries: &[
            storage_entry(0, true, GPU_DRAWING_STATE_SIZE as u64),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage_entry(2, false, 4),
            storage_entry(3, true, 4),
            storage_entry(4, true, 4),
            storage_entry(5, false, 4),
            storage_entry(6, true, 4),
        ],
    })
}
fn binding(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: b.as_entire_binding(),
    }
}
fn group(
    p: &GpuPipeline,
    b: &Buffers,
    layout: &wgpu::BindGroupLayout,
    index: usize,
) -> wgpu::BindGroup {
    p.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("checker raster resources"),
        layout,
        entries: &[
            binding(0, &b.states),
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&b.view),
            },
            binding(2, &b.errors[index]),
            binding(3, &b.indices),
            binding(4, &b.counts),
            binding(5, if index == 2 { &b.visits } else { &b.states }),
            binding(6, &b.parents),
        ],
    })
}
fn prepare(
    p: &GpuPipeline,
    b: &Buffers,
    base: &BenchmarkRasterResources,
    c: &BenchmarkRasterResources,
    text: &str,
) -> Prepared {
    let layout = coverage_layout(&p.device);
    let params = c.pipeline.get_bind_group_layout(1);
    let pipeline_layout = p
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("coverage pipeline layout"),
            bind_group_layouts: &[Some(&layout), Some(&params)],
            immediate_size: 0,
        });
    let shader = p.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("actual source plus visits"),
        source: wgpu::ShaderSource::Wgsl(text.into()),
    });
    let coverage = p
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("checker coverage"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: c.compilation_options.clone(),
            cache: None,
        });
    let groups = [
        group(p, b, &base.pipeline.get_bind_group_layout(0), 0),
        group(p, b, &c.pipeline.get_bind_group_layout(0), 1),
        group(p, b, &layout, 2),
    ];
    let params = p.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("checker params"),
        layout: &params,
        entries: &[binding(0, &b.params)],
    });
    Prepared {
        groups,
        params,
        coverage,
    }
}
fn encode(
    p: &GpuPipeline,
    b: &Buffers,
    prepared: &Prepared,
    base: &BenchmarkRasterResources,
    candidate: &BenchmarkRasterResources,
    f: Fixture,
    base_x: u32,
) -> wgpu::CommandBuffer {
    let mut encoder = p
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("three real rasters"),
        });
    encoder.clear_buffer(&b.visits, 0, None);
    for (i, pipeline) in [&base.pipeline, &candidate.pipeline, &prepared.coverage]
        .into_iter()
        .enumerate()
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("checker raster"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &prepared.groups[i], &[]);
        // Baseline owns its actual layout too; independently created explicit
        // layouts are structurally compatible with the candidate's uniform group.
        pass.set_bind_group(1, &prepared.params, &[]);
        let dispatch_x = if i == 0 {
            base_x
        } else {
            p.raster_dispatch_x_for_width(f.image[0])
        };
        pass.dispatch_workgroups(dispatch_x, f.grid()[1], word(f.active()));
    }
    for (i, error) in b.errors.iter().enumerate() {
        encoder.copy_buffer_to_buffer(error, 0, &b.staging, bytes(i * ERRORS), bytes(ERRORS));
    }
    encoder.copy_buffer_to_buffer(&b.visits, 0, &b.staging, bytes(3 * ERRORS), bytes(VISITS));
    encoder.finish()
}
fn readback(
    p: &GpuPipeline,
    b: &Buffers,
    command: wgpu::CommandBuffer,
    s: &mut Scratch,
    e: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    ensure!(e.submitted < 180, "fixture submission budget exhausted");
    let submission = p.queue.submit([command]);
    e.submitted += 1;
    e.event(&json!({"event":"submitted","fixture":e.submitted,"dispatches":3}))?;
    let (tx, rx) = mpsc::sync_channel(1);
    b.staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _receiver_gone = tx.send(r);
        });
    let result = map_readback(p, b, submission, &rx, &mut s.actual, deadline);
    b.staging.unmap();
    result
}
fn map_readback(
    p: &GpuPipeline,
    b: &Buffers,
    submission: wgpu::SubmissionIndex,
    rx: &mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    actual: &mut [u32],
    deadline: Instant,
) -> Result<()> {
    let until = Instant::now() + remaining(deadline)?.min(WAIT);
    let status = p
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(until.saturating_duration_since(Instant::now())),
        })
        .context("readback poll")?;
    ensure!(status.wait_finished(), "submission did not finish");
    rx.recv_timeout(until.saturating_duration_since(Instant::now()))
        .context("map callback timeout")?
        .context("map failed")?;
    let mapped = b
        .staging
        .slice(..)
        .get_mapped_range()
        .context("mapped range")?;
    ensure!(
        mapped.len() == WORDS * 4 && actual.len() == WORDS,
        "readback size mismatch"
    );
    actual.copy_from_slice(
        bytemuck::try_cast_slice::<u8, u32>(&mapped)
            .map_err(|e| anyhow::anyhow!("readback alignment {e}"))?,
    );
    drop(mapped);
    Ok(())
}
fn remaining(deadline: Instant) -> Result<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    ensure!(!remaining.is_zero(), "600s deadline exhausted");
    Ok(remaining)
}
fn drain(p: &GpuPipeline, deadline: Instant) -> Result<()> {
    let submission = p.queue.submit([]);
    let status = p
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(remaining(deadline)?.min(WAIT)),
        })
        .context("drain")?;
    ensure!(status.wait_finished(), "drain unfinished");
    Ok(())
}
fn scopes(d: &wgpu::Device) -> [wgpu::ErrorScopeGuard; 3] {
    [
        d.push_error_scope(wgpu::ErrorFilter::Validation),
        d.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
        d.push_error_scope(wgpu::ErrorFilter::Internal),
    ]
}
fn finish_scopes(
    d: &wgpu::Device,
    guards: [wgpu::ErrorScopeGuard; 3],
    deadline: Instant,
) -> Result<()> {
    let [v, m, i] = guards;
    let mut errors = Vec::with_capacity(3);
    for future in [i.pop(), m.pop(), v.pop()] {
        let mut future = std::pin::pin!(future);
        let mut cx = TaskContext::from_waker(Waker::noop());
        let until = Instant::now() + remaining(deadline)?.min(WAIT);
        let mut done = false;
        for _ in 0..120_000 {
            if let Poll::Ready(error) = future.as_mut().poll(&mut cx) {
                if let Some(error) = error {
                    errors.push(error.to_string());
                }
                done = true;
                break;
            }
            ensure!(Instant::now() < until, "GPU scope timeout");
            d.poll(wgpu::PollType::Poll)?;
            std::thread::sleep(Duration::from_millis(1));
        }
        ensure!(done, "scope budget exhausted");
    }
    ensure!(errors.is_empty(), "GPU errors: {errors:?}");
    Ok(())
}

fn visit_diffs(visits: &[u32], expected: &[u32]) -> [usize; 3] {
    let mut missing = 0;
    let mut duplicate = 0;
    let mut unexpected = 0;
    for (&a, &x) in visits.iter().zip(expected) {
        if a < x {
            missing += (x - a) as usize;
        }
        if a > x {
            if x == 1 {
                duplicate += (a - x) as usize;
            } else {
                unexpected += (a - x) as usize;
            }
        }
    }
    [missing, duplicate, unexpected]
}
fn compare(f: Fixture, s: &Scratch, e: &mut Evidence) -> Result<()> {
    let arrays: [&[u32]; 3] = std::array::from_fn(|i| &s.actual[i * ERRORS..(i + 1) * ERRORS]);
    let visits = &s.actual[3 * ERRORS..];
    let errors_equal = arrays[0] == arrays[1] && arrays[0] == arrays[2];
    let mut old_bad = 0;
    let mut inactive_bad = 0;
    let mut new_sum = 0u64;
    let mut old_sum = 0u64;
    for slot in 0..CAP {
        for a in arrays {
            if slot < f.active() {
                old_bad += usize::from(a[slot * 2 + 1] != s.initial[slot * 2 + 1] + s.old[slot]);
            } else {
                inactive_bad +=
                    usize::from(a[slot * 2..slot * 2 + 2] != s.initial[slot * 2..slot * 2 + 2]);
            }
        }
        if slot < f.active() {
            new_sum += u64::from(arrays[0][slot * 2].saturating_sub(s.initial[slot * 2]));
            old_sum += u64::from(s.old[slot]);
        }
    }
    let [missing, duplicate, unexpected] = visit_diffs(visits, &s.expected);
    e.event(&json!({"event":"comparison","fixture":f,"errors_equal":errors_equal,
        "old_bad":old_bad,"inactive_bad":inactive_bad,"missing":missing,"duplicate":duplicate,
        "unexpected":unexpected,"initial_errors":s.initial.as_slice(),"baseline_errors":arrays[0],
        "candidate_errors":arrays[1],"coverage_errors":arrays[2],"expected_old_l1":s.old.as_slice(),
        "expected_visits":s.expected,"actual_visits":visits,"new_l1_sum":new_sum,"old_l1_sum":old_sum}))?;
    ensure!(
        errors_equal && old_bad == 0 && inactive_bad == 0,
        "invalid error proof or negative setup"
    );
    ensure!(
        visits[VISITS - 1] == 0 && unexpected == 0,
        "invalid/padding/inactive coordinates visited"
    );
    if f.fault {
        ensure!(
            arrays.iter().all(|a| a.iter().all(|&v| v == 0)),
            "negative fixture must have both zero errors"
        );
        e.event(
            &json!({"event":"negative_proof","errors_equal":true,"missing":missing,
            "duplicate":duplicate,"unexpected":unexpected,"gpu_setup_valid":true}),
        )?;
        ensure!(missing == 1, "negative must omit exactly one real tile");
    } else {
        ensure!(missing == 0 && duplicate == 0, "coverage mismatch");
        ensure!(
            new_sum > 0 && old_sum > 0 && new_sum != old_sum,
            "normal fixture must exercise distinct nonzero errors"
        );
    }
    Ok(())
}
fn snapshot_evidence(e: &mut Evidence, name: &str, r: &BenchmarkRasterResources) -> Result<()> {
    e.source(name, &r.source)?;
    e.event(&json!({"event":"factory_snapshot","name":name,"sha256":hash(r.source.as_bytes()),
        "wg":r.wg,"tiles_per_group":r.tiles_per_group,
        "compilation_options":{"constants":r.compilation_options.constants,
            "zero_initialize_workgroup_memory":r.compilation_options.zero_initialize_workgroup_memory}}))
}
fn validate_device(p: &GpuPipeline) -> Result<()> {
    let l = p.device.limits();
    ensure!(
        p.offspring_capacity as usize >= CAP,
        "engine offspring capacity too small"
    );
    ensure!(
        l.max_storage_buffers_per_shader_stage >= 6,
        "six storage bindings required"
    );
    ensure!(
        l.max_storage_buffer_binding_size >= bytes(PAIRS * POLYS)
            && l.max_buffer_size >= bytes(PAIRS * POLYS),
        "checker buffer exceeds device limit"
    );
    ensure!(
        l.max_compute_workgroups_per_dimension >= word(CAP),
        "dispatch dimension too small"
    );
    Ok(())
}
fn prepare_candidate(
    p: &mut GpuPipeline,
    b: &Buffers,
    base: &BenchmarkRasterResources,
    source: &Source,
    fault: Fault,
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<(BenchmarkRasterResources, Prepared)> {
    let wg = p.rasterize_wg;
    remaining(deadline)?;
    let guards = scopes(&p.device);
    let faulted = if source.t == 2 {
        fault_source(&source.text, fault)?
    } else {
        source.text.clone()
    };
    p.set_benchmark_raster_shader(&faulted)
        .map_err(anyhow::Error::msg)?;
    let candidate = p.benchmark_raster_resources();
    ensure!(
        candidate.wg == wg && candidate.tiles_per_group == source.t,
        "factory snapshot mismatch"
    );
    let name = format!("{}x{}-candidate-{}", wg[0], wg[1], source.name);
    snapshot_evidence(evidence, &name, &candidate)?;
    let coverage = instrument(&candidate.source, candidate.tiles_per_group)?;
    evidence.source(&format!("{name}-coverage"), &coverage)?;
    let prepared = prepare(p, b, base, &candidate, &coverage);
    finish_scopes(&p.device, guards, deadline)?;
    Ok((candidate, prepared))
}
fn new_pipeline(scratch: &Scratch, deadline: Instant) -> Result<GpuPipeline> {
    let p = futures_lite::future::block_on(GpuPipeline::new(
        4,
        WIDTH,
        HEIGHT,
        &scratch.reference,
        &scratch.states[..4],
    ));
    drain(&p, deadline)?;
    validate_device(&p)?;
    Ok(p)
}
fn run(options: &Options, evidence: &mut Evidence, deadline: Instant) -> Result<()> {
    for s in &options.sources {
        evidence.source(&format!("input-{}", s.name), &s.text)?;
        evidence.event(&json!({"event":"input","name":s.name,"path":s.path,"sha256":hash(s.text.as_bytes()),"t":s.t}))?;
    }
    let mut scratch = Scratch::new();
    let mut p = new_pipeline(&scratch, deadline)?;
    evidence.event(&json!({"event":"device","adapter":format!("{:?}",p.device.adapter_info()),"fault":options.fault,
        "expected_fixtures":if options.fault==Fault::None{180}else{1},"fixture_definition":"W=NX*WG_X-1,H=2*WG_Y+1; K=4; lambda=1/64; NX=1/2/3/4/5/9; dirty rotation=fixture_index mod10"}))?;
    let b = Buffers::new(&p);
    for wg in WGS {
        remaining(deadline)?;
        let guards = scopes(&p.device);
        p.set_rasterize_wg(wg);
        p.set_benchmark_raster_shader(&options.sources[0].text)
            .map_err(anyhow::Error::msg)?;
        let base = p.benchmark_raster_resources();
        snapshot_evidence(evidence, &format!("{}x{}-base", wg[0], wg[1]), &base)?;
        let base_x: [u32; 6] =
            [1, 2, 3, 4, 5, 9].map(|nx| p.raster_dispatch_x_for_width(nx * wg[0] - 1));
        let fault_base_x = p.raster_dispatch_x_for_width(2 * wg[0]);
        finish_scopes(&p.device, guards, deadline)?;
        for source in &options.sources {
            if options.fault != Fault::None && source.t != 2 {
                continue;
            }
            let (candidate, prepared) =
                prepare_candidate(&mut p, &b, &base, source, options.fault, evidence, deadline)?;
            let rasters = Rasters {
                pipeline: &p,
                buffers: &b,
                baseline: &base,
                candidate: &candidate,
                prepared: &prepared,
            };
            run_fixtures(
                &rasters,
                &mut scratch,
                evidence,
                deadline,
                options.fault,
                base_x,
                fault_base_x,
            )?;
            if options.fault != Fault::None {
                return reject_fault(options.fault, &scratch, evidence);
            }
        }
    }
    ensure!(
        evidence.passed == 180 && evidence.submitted == 180,
        "incomplete positive matrix"
    );
    // Synthetic proof ends before separate actual host-path integration.
    drop(b);
    drop(p);
    drop(scratch);
    integration(options, evidence, deadline)
}
fn fixture(wg: [u32; 2], lambda: u32, nx: u32, rotation: usize, fault: Fault) -> Fixture {
    if fault == Fault::None {
        Fixture {
            image: [nx * wg[0] - 1, 2 * wg[1] + 1],
            wg,
            lambda,
            parents: 4,
            rotation,
            fault: false,
        }
    } else {
        Fixture {
            image: [2 * wg[0], wg[1]],
            wg,
            lambda: 1,
            parents: 1,
            rotation: 0,
            fault: true,
        }
    }
}
fn run_fixtures(
    rasters: &Rasters<'_>,
    scratch: &mut Scratch,
    evidence: &mut Evidence,
    deadline: Instant,
    fault: Fault,
    base_x: [u32; 6],
    fault_base_x: u32,
) -> Result<()> {
    let Rasters {
        pipeline: p,
        buffers: b,
        baseline: base,
        candidate,
        prepared,
    } = rasters;
    for lambda in [1, 64] {
        for (i, nx) in [1, 2, 3, 4, 5, 9].into_iter().enumerate() {
            let f = fixture(candidate.wg, lambda, nx, evidence.submitted % 10, fault);
            scratch.prepare(f);
            remaining(deadline)?;
            let guards = scopes(&p.device);
            b.upload(p, f, scratch);
            evidence.event(
                &json!({"event":"fixture","definition":f,"actual_t":candidate.tiles_per_group,
            "candidate_dispatch":[p.raster_dispatch_x_for_width(f.image[0]),f.grid()[1],f.active()],
            "baseline_dispatch":[if f.fault{fault_base_x}else{base_x[i]},f.grid()[1],f.active()]}),
            )?;
            let command = encode(
                p,
                b,
                prepared,
                base,
                candidate,
                f,
                if f.fault { fault_base_x } else { base_x[i] },
            );
            let result = readback(p, b, command, scratch, evidence, deadline);
            let scope_result = finish_scopes(&p.device, guards, deadline);
            result?;
            scope_result?;
            compare(f, scratch, evidence)?;
            if f.fault {
                return Ok(());
            }
            evidence.passed += 1;
        }
    }
    Ok(())
}
fn reject_fault(fault: Fault, s: &Scratch, e: &mut Evidence) -> Result<()> {
    let visits = &s.actual[3 * ERRORS..];
    ensure!(
        visits[0] == if fault == Fault::Duplicate { 2 } else { 1 } && visits[1] == 0,
        "fault did not corrupt expected real coordinates"
    );
    e.negative = true;
    e.event(
        &json!({"event":"negative_rejected","fault":fault,"errors_equal":true,"exit_status":1}),
    )?;
    bail!("expected real {fault:?} coverage fault detected")
}

fn integration_params(iters: u32) -> MutationParams {
    MutationParams {
        add_polygon_prob: 0.0,
        remove_polygon_prob: 0.0,
        reorder_polygon_prob: 0.0,
        offset_polygon_prob: 0.0,
        move_point_prob: 0.0,
        remove_point_prob: 0.0,
        micro_adjust_prob: 0.0,
        change_color_prob: 1.0,
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
        min_alpha: 255,
        max_alpha: 255,
        min_polygons: 1,
        max_polygons: 1,
        chain_count: 1,
        lambda: 1,
        single_mutation_mode: false,
        adaptive_mutation: false,
        rasterize_wg: [8, 8],
        gpu_batch_iters: iters,
        ..Default::default()
    }
}
fn edge_drawing() -> Drawing {
    let x = 16.5 / 17.0;
    Drawing {
        polygons: vec![Polygon {
            points: vec![
                Point {
                    x: x - 0.009,
                    y: 0.491,
                },
                Point {
                    x: x + 0.009,
                    y: 0.491,
                },
                Point { x, y: 0.509 },
            ],
            color: Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
        }],
        is_dirty: false,
        fitness: 0.0,
    }
}
fn checked_edge(
    g: &mut GpuEvolver,
    reference: &[u8],
    e: &mut Evidence,
    label: &str,
) -> Result<u32> {
    g.benchmark_drain().map_err(anyhow::Error::msg)?;
    let s = g.benchmark_snapshot(1, 1).map_err(anyhow::Error::msg)?;
    ensure!(
        s.parent_framebuffers.len() == 1 && s.parent_framebuffers[0].len() == 17 * 19,
        "host snapshot shape"
    );
    let pixels = &s.parent_framebuffers[0];
    let mut raw = 0;
    for (i, &pixel) in pixels.iter().enumerate() {
        raw += l1(pixel, &reference[i * 4..i * 4 + 3]);
        ensure!(
            i == 9 * 17 + 16 || pixel & 0x00ff_ffff == 0x00ff_ffff,
            "host changed a non-target pixel"
        );
    }
    ensure!(
        s.parent_total_errors == [raw],
        "host framebuffer-byte L1 disagrees with cached total"
    );
    e.event(&json!({"event":"host_snapshot","label":label,"raw_l1":raw,
        "parents":s.parents,"cached_total":s.parent_total_errors,"pixels":pixels,
        "flags":s.chain_update_flags,"offspring_errors":s.offspring_errors}))?;
    Ok(raw)
}
fn integration(o: &Options, e: &mut Evidence, deadline: Instant) -> Result<()> {
    let mut reference = [255, 255, 255, 255].repeat(17 * 19);
    reference[(9 * 17 + 16) * 4..(9 * 17 + 16) * 4 + 3].fill(0);
    let drawing = edge_drawing();
    for source in o.sources.iter().filter(|s| s.t != 1) {
        for iters in [1, 2] {
            remaining(deadline)?;
            let mut g =
                futures_lite::future::block_on(GpuEvolver::new(&reference, 17, 19, &drawing));
            let params = integration_params(iters);
            g.benchmark_drain().map_err(anyhow::Error::msg)?;
            g.benchmark_set_raster_shader(&source.text)
                .map_err(anyhow::Error::msg)?;
            g.prepare_seeded_benchmark(&drawing, &params, 7, 0, true)
                .map_err(anyhow::Error::msg)?;
            ensure!(
                g.rasterize_wg() == [8, 8] && g.benchmark_raster_tiles_per_group() == source.t,
                "host specialization mismatch"
            );
            let label = format!("{}-iters{iters}", source.name);
            e.event(&json!({"event":"host_fixture","label":label,"source_sha256":hash(source.text.as_bytes()),
            "wg":[8,8],"t":source.t,"params":params,"seed":7,"last_tile_x":2,"target_pixel":[16,9]}))?;
            ensure!(
                checked_edge(&mut g, &reference, e, &format!("{label}-initial"))? == 765,
                "host initial L1 must be 765"
            );
            let mut accepted = false;
            for step in 1..=16 {
                remaining(deadline)?;
                let _drawing_report = g.run_batch(&params, true);
                let raw = checked_edge(&mut g, &reference, e, &format!("{label}-step{step}"))?;
                if raw < 765 {
                    // Strictly lower cached byte error and changed target pixel prove
                    // an actual accepted parent, independent of the racy best ID.
                    accepted = true;
                    let _fitness = g.evaluate_chain_fitness(1);
                    ensure!(
                        checked_edge(&mut g, &reference, e, &format!("{label}-full"))? == raw,
                        "full-fitness recompute changed accepted byte error"
                    );
                    break;
                }
            }
            ensure!(
                accepted,
                "missing right-edge acceptance integration proof for {label}"
            );
            e.event(
                &json!({"event":"host_integration_passed","label":label,"strict_acceptance":true}),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCES: [(&str, u32, &str); 3] = [
        (
            include_str!(
                "../benchmarks/evolution/overnight-20261024/fixtures/raster-ae1-reference.wgsl"
            ),
            1,
            AE1,
        ),
        (
            include_str!(
                "../benchmarks/evolution/overnight-20261024/fixtures/raster-direct-t2.wgsl"
            ),
            2,
            T2,
        ),
        (
            include_str!(
                "../benchmarks/evolution/overnight-20261024/fixtures/raster-direct-t4.wgsl"
            ),
            4,
            T4,
        ),
    ];
    fn float_helpers(source: &str) -> &str {
        source
            .split_once("fn rasterize_blend")
            .expect("raster helper")
            .1
            .split_once("@compute")
            .expect("compute entry")
            .0
    }
    #[test]
    fn source_and_fault_rewrites_validate() -> Result<()> {
        let mut validated = 0;
        let helpers = float_helpers(SOURCES[0].0);
        for (text, tiles, pin) in SOURCES {
            assert_eq!(hash(text.as_bytes()), pin);
            for [x, y] in WGS {
                let specialized = once(
                    &once(
                        text,
                        "const WG_X: u32 = 16;",
                        &format!("const WG_X: u32 = {x};"),
                    )?,
                    "const WG_Y: u32 = 16;",
                    &format!("const WG_Y: u32 = {y};"),
                )?;
                for fault in [Fault::None, Fault::Missing, Fault::Duplicate] {
                    if tiles != 2 && fault != Fault::None {
                        continue;
                    }
                    let actual = fault_source(&specialized, fault)?;
                    assert_eq!(
                        benchmark_raster_tiles_per_group_from_source(&actual)
                            .map_err(anyhow::Error::msg)?,
                        tiles
                    );
                    for variant in [&actual, &instrument(&actual, tiles)?] {
                        assert_eq!(float_helpers(variant), helpers);
                        let module =
                            wgpu::naga::front::wgsl::parse_str(variant).map_err(|error| {
                                anyhow::anyhow!("{}", error.emit_to_string(variant))
                            })?;
                        wgpu::naga::valid::Validator::new(
                            wgpu::naga::valid::ValidationFlags::all(),
                            wgpu::naga::valid::Capabilities::all(),
                        )
                        .validate(&module)?;
                        validated += 1;
                    }
                }
            }
        }
        assert_eq!(validated, 50);
        Ok(())
    }
    #[test]
    fn strict_predicate_and_visit_guards() {
        let shape = fixture([8, 8], 1, 3, 0, Fault::None);
        assert!(consumed(shape, [1, 1, 1, 7], 0));
        assert!(consumed(shape, [3, 1, 2, 7], 0));
        assert!(!consumed(shape, [8, 0, 8, 8], 0));
        assert_eq!(visit_diffs(&[1, 0, 0], &[1, 1, 0]), [1, 0, 0]);
        assert_eq!(visit_diffs(&[2, 0, 0], &[1, 1, 0]), [1, 1, 0]);
        assert_eq!(visit_diffs(&[1, 1, 1], &[1, 1, 0]), [0, 0, 1]);
        let mut count = 0;
        for wg in WGS {
            for _ in [1, 2, 4] {
                for lambda in [1, 64] {
                    for nx in [1, 2, 3, 4, 5, 9] {
                        let f = fixture(wg, lambda, nx, count % 10, Fault::None);
                        assert!(
                            f.active() <= CAP
                                && f.tiles() <= TILES
                                && f.image[0] <= WIDTH
                                && f.image[1] <= HEIGHT
                        );
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 180);
    }
    #[test]
    fn old_byte_l1_uses_parent_and_valid_image_bytes() {
        let f = fixture([8, 8], 1, 3, 0, Fault::None);
        let mut scratch = Scratch::new();
        scratch.prepare(f);
        let expected: u32 = (0..f.image[1])
            .flat_map(|y| (0..f.image[0]).map(move |x| (x, y)))
            .map(|(x, y)| {
                let pixel = scratch.parents[(y * f.image[0] + x) as usize].to_le_bytes();
                (0..3)
                    .map(|c| {
                        u32::from(
                            pixel[c].abs_diff(scratch.reference[(y * WIDTH + x) as usize * 4 + c]),
                        )
                    })
                    .sum::<u32>()
            })
            .sum();
        assert_eq!(scratch.old[0], expected);
        assert!(expected > 0);
        assert_eq!(scratch.old[1], 0);
        assert_ne!(scratch.old[0], scratch.old[3]);
    }
    #[test]
    fn cli_and_source_anchors_fail_before_gpu() -> Result<()> {
        for args in [
            vec!["--bad", "x"],
            vec!["--fault", "bad"],
            vec!["--output", "a", "--output", "b"],
            vec!["--candidate", "badname"],
            vec!["--reference", "a", "--reference", "b"],
            vec!["--candidate", "../bad=x"],
        ] {
            assert!(parse_args(&args.into_iter().map(OsString::from).collect::<Vec<_>>()).is_err());
        }
        assert!(once("x x", "x", "y").is_err());
        assert!(instrument("", 1).is_err());
        assert!(fault_source(SOURCES[0].0, Fault::Missing).is_err());
        let path = PathBuf::from(format!(
            ".cache/evolution-overnight-20261024/raster-direct-coarsen/checker/invalid-source-{}",
            std::process::id()
        ));
        for bytes in [vec![0xff], vec![b' '; MAX_SOURCE + 1]] {
            fs::write(&path, bytes)?;
            assert!(read_source("invalid".into(), &path).is_err());
        }
        fs::remove_file(path)?;
        Ok(())
    }
}
