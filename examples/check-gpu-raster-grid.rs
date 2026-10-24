//! Bounded correctness check of the actual compact-prefix/indirect raster path.
//!
//! Run the built example with an OUTER native-call timeout:
//! `timeout 600s target/release/examples/check-gpu-raster-grid --output NEW_DIR`.
//! `--help` and invalid CLI arguments initialize no GPU and create no files.
//! All device calls, compilation, dispatch, polling and mapping stay on main.
//! This is an unscored checker, with no timing or evolution campaign.
//!
//! There are 75 fixtures: five workgroups x four lambdas x three history phases,
//! then 15 large-grid/history cases. Each submits prefix, indirect raster and
//! actual ordinary raster, then reads both stride-two accumulators BEFORE any
//! selection/reset. Prefix boundaries come from exhaustive ORIGINAL tile tests,
//! not a second implementation of compact entry mapping. Every original tile
//! must have exactly one visit iff BOTH strict axis predicates hold. Inactive
//! slots and unused visit words must have zero visits and unchanged errors.
//!
//! `--fault missing|duplicate` runs one unscored fixture and must FAIL. It changes
//! the actual entry's padding guard or flattened ordinal. Missing skips q=0;
//! duplicate maps q=1 to q=0. Both tiles have zero new/old error. Thus both error
//! arrays still agree, but coverage rejects the missing or repeated work.
//! The same missing/duplicate rejection is checked by CPU unit guards.
//!
//! Engine setup/compilation blocks this serial workstream; source preparation
//! is independent of the engine owner. Only this file is owned. Checker-owned
//! buffers are reused, including prefix suffixes and inactive working states.
//! Maximum addressing: 256 offspring, 288 tiles each, 256 indices per tile,
//! four cached parents, image <=384x384. Largest buffer is 75,497,472 bytes.
//! Fixed CPU arrays and GPU resources are allocated before fixture execution.
//! Engine construction uses 384x384 to request the necessary storage limit.
//!
//! Power-of-Ten deviations: rules 2/7, existing engine construction and native
//! compilation require the outer 600s timeout; own waits/scopes/maps have 120s
//! limits and finite poll budgets, with an internal 600s deadline between calls.
//! Rule 3, wgpu/driver work, bounded JSON/error formatting and source compilation
//! allocate at runtime; at most one submission/map and one workgroup's variants
//! are live, source <=256 KiB, events <=75 fixtures plus bounded setup events.
//! Rules 8/9, checked source rewriting adds ONLY instrumentation, with one exact
//! declaration and insertion marker; framework macros/one map callback have no
//! recursive callback chain. Rule 10, compilation/GPU validation is parent-gated.
//! Follow-up: parent must run real readbacks and negative modes before adoption.
//! events.jsonl is flushed incrementally; result.json records normal errors.

use std::{
    env,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    future::Future,
    io::Write,
    num::NonZeroU64,
    path::{Path, PathBuf},
    sync::mpsc,
    task::{Context as TaskContext, Poll, Waker},
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::gpu_evolver::{
    buffers::{default_gpu_params, GpuDrawingState, GpuParams, GpuPolygon, GPU_DRAWING_STATE_SIZE},
    pipeline::GpuPipeline,
};
use bytemuck::Zeroable;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const WORKGROUPS: [[u32; 2]; 5] = [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];
const LAMBDAS: [u32; 4] = [1, 3, 32, 64];
const CAPACITY: usize = 256;
const MAX_TILES: usize = 288;
const TILE_POLYS: usize = 256;
const MAX_PAIRS: usize = CAPACITY * MAX_TILES;
const VISIT_WORDS: usize = MAX_PAIRS + 1; // final word detects invalid mapped coordinates
const MAX_PIXELS: usize = 384 * 384;
const ERROR_WORDS: usize = CAPACITY * 2;
const PREFIX_WORDS: usize = CAPACITY + 1;
const PREFIX_OFFSET: usize = ERROR_WORDS * 2;
const ARGS_OFFSET: usize = PREFIX_OFFSET + PREFIX_WORDS;
const VISITS_OFFSET: usize = ARGS_OFFSET + 3;
const SNAPSHOT_WORDS: usize = VISITS_OFFSET + VISIT_WORDS;
const FIXTURES: usize = 75;
const MAX_SOURCE: usize = 256 * 1024;
const WAIT: Duration = Duration::from_secs(120);
const OUTER: Duration = Duration::from_secs(600);
const SCOPE_POLLS: usize = 120_000;
const ACCEPTED_HASH: &str = "a292287ae6d9c212c7970dd3e712001a1b5a827975e6216355384fc0106eb3af";
const ORIGINAL: &str = include_str!("../src/shaders/rasterize_error.wgsl");
const DECLARATION: &str =
    "@group(0) @binding(5) var<storage, read>       chain_states:       array<DrawingState>;";
const MARKER: &str = "    var tile_poly_count = 0u;";
type Dirty = [u32; 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Fault {
    None,
    Missing,
    Duplicate,
}

struct Options {
    output: PathBuf,
    fault: Fault,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Fixture {
    image: [u32; 2],
    wg: [u32; 2],
    lambda: u32,
    parents: u32,
    phase: u32,
    target: Option<u32>,
}

impl Fixture {
    fn active(self) -> usize {
        (self.lambda * self.parents) as usize
    }

    fn tiles(self) -> [u32; 2] {
        [
            self.image[0].div_ceil(self.wg[0]),
            self.image[1].div_ceil(self.wg[1]),
        ]
    }

    fn num_tiles(self) -> usize {
        let [x, y] = self.tiles();
        (x * y) as usize
    }
}

struct Scratch {
    states: Vec<GpuDrawingState>,
    counts: Vec<u32>,
    indices: Vec<u32>,
    framebuffers: Vec<u32>,
    reference: Vec<u8>,
    initial_errors: [u32; ERROR_WORDS],
    expected_prefix: [u32; PREFIX_WORDS],
    expected_visits: Vec<u32>,
    actual: Vec<u32>,
}

struct Buffers {
    states: wgpu::Buffer,
    counts: wgpu::Buffer,
    indices: wgpu::Buffer,
    framebuffers: wgpu::Buffer,
    errors: [wgpu::Buffer; 2],
    prefix: wgpu::Buffer,
    args: wgpu::Buffer,
    visits: wgpu::Buffer,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct Prepared {
    prefix_group: wgpu::BindGroup,
    compact_group: wgpu::BindGroup,
    ordinary_group: wgpu::BindGroup,
    params_group: wgpu::BindGroup,
    instrumented: wgpu::ComputePipeline,
}

struct Evidence {
    output: PathBuf,
    events: File,
    submitted: usize,
    passed: usize,
}

#[derive(Default, Debug, Serialize)]
struct Comparison {
    prefix: usize,
    args: usize,
    missing: usize,
    duplicate: usize,
    unexpected: usize,
    errors: usize,
    inactive_errors: usize,
    first: Option<String>,
}

impl Comparison {
    fn passed(&self) -> bool {
        self.prefix
            + self.args
            + self.missing
            + self.duplicate
            + self.unexpected
            + self.errors
            + self.inactive_errors
            == 0
    }

    fn note(&mut self, message: String) {
        if self.first.is_none() {
            self.first = Some(message);
        }
    }
}

fn main() -> Result<()> {
    let Some(options) = parse_args(env::args_os().skip(1).take(6))? else {
        println!("usage: check-gpu-raster-grid --output NEW_DIR [--fault missing|duplicate]");
        println!("Run with an outer timeout 600s. Fault modes intentionally return failure.");
        return Ok(());
    };
    let mut evidence = Evidence::new(&options.output)?;
    let outcome = run(&options, &mut evidence, Instant::now() + OUTER);
    evidence.finish(&outcome)?;
    outcome
}

fn parse_args(args: impl Iterator<Item = OsString>) -> Result<Option<Options>> {
    let args: Vec<_> = args.take(6).collect();
    if args.len() == 1 && args[0] == "--help" {
        return Ok(None);
    }
    ensure!(
        args.len() == 2 || args.len() == 4,
        "usage: --output NEW_DIR [--fault missing|duplicate]"
    );
    ensure!(args[0] == "--output", "first argument must be --output");
    ensure!(
        !args[1].is_empty() && args[1].len() <= 4096,
        "output path must contain 1..4096 bytes"
    );
    let fault = if args.len() == 4 {
        ensure!(args[2] == "--fault", "expected --fault");
        match args[3].to_str() {
            Some("missing") => Fault::Missing,
            Some("duplicate") => Fault::Duplicate,
            _ => bail!("fault must be missing or duplicate"),
        }
    } else {
        Fault::None
    };
    Ok(Some(Options {
        output: PathBuf::from(&args[1]),
        fault,
    }))
}

impl Evidence {
    fn new(output: &Path) -> Result<Self> {
        fs::create_dir(output).context("output must be a new directory")?;
        let events = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("events.jsonl"))?;
        Ok(Self {
            output: output.to_path_buf(),
            events,
            submitted: 0,
            passed: 0,
        })
    }

    fn event(&mut self, value: &Value) -> Result<()> {
        serde_json::to_writer(&mut self.events, value)?;
        self.events.write_all(b"\n")?;
        self.events.flush()?;
        Ok(())
    }

    fn source(&mut self, name: &str, source: &str) -> Result<()> {
        ensure!(
            !source.is_empty() && source.len() <= MAX_SOURCE,
            "source size bound"
        );
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.output.join(format!("{name}.wgsl")))?;
        file.write_all(source.as_bytes())?;
        file.flush()?;
        self.event(&json!({"event":"source", "name":name, "bytes":source.len(),
            "sha256":hash(source.as_bytes())}))
    }

    fn finish(&mut self, outcome: &Result<()>) -> Result<()> {
        let result = json!({"passed":outcome.is_ok(), "submitted":self.submitted,
            "fixtures_passed":self.passed, "scored":false,
            "error":outcome.as_ref().err().map(|e| short_error(&format!("{e:#}")))});
        self.event(&json!({"event":"finished", "result":result}))?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.output.join("result.json"))?;
        serde_json::to_writer_pretty(&mut file, &result)?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn short_error(message: &str) -> String {
    message.chars().take(16 * 1024).collect()
}

fn fixtures(fault: Fault) -> Vec<Fixture> {
    let mut cases = Vec::with_capacity(FIXTURES);
    if fault != Fault::None {
        cases.push(Fixture {
            image: [65, 51],
            wg: WORKGROUPS[0],
            lambda: 3,
            parents: 4,
            phase: 3,
            target: None,
        });
        return cases;
    }
    for wg in WORKGROUPS {
        for lambda in LAMBDAS {
            for phase in 0..3 {
                cases.push(Fixture {
                    image: [65, 51],
                    wg,
                    lambda,
                    parents: [4, 1, 3][phase as usize],
                    phase,
                    target: None,
                });
            }
        }
    }
    for (target, lambda, parents) in [
        (73728, 64, 4),
        (65534, 64, 4),
        (65535, 64, 4),
        (65536, 64, 4),
        (0, 1, 1),
        (1, 3, 1),
        (288, 32, 1),
        (73728, 64, 4),
        (0, 64, 4),
        (65535, 64, 4),
        (65536, 64, 4),
        (65537, 64, 4),
        (1, 1, 1),
        (73728, 64, 4),
        (0, 3, 1),
    ] {
        cases.push(Fixture {
            image: [384, 384],
            wg: WORKGROUPS[0],
            lambda,
            parents,
            phase: 4,
            target: Some(target),
        });
    }
    assert_eq!(cases.len(), FIXTURES);
    cases
}

fn validate_fixture(f: Fixture) -> Result<()> {
    ensure!(
        WORKGROUPS.contains(&f.wg) && LAMBDAS.contains(&f.lambda),
        "unsupported fixture"
    );
    ensure!(
        (1..=4).contains(&f.parents) && f.active() <= CAPACITY,
        "active state bound"
    );
    ensure!(
        f.image.into_iter().all(|v| (1..=384).contains(&v)),
        "image bound"
    );
    ensure!(f.num_tiles() <= MAX_TILES, "tile addressing bound");
    if let Some(total) = f.target {
        ensure!(
            total as usize <= f.active() * f.num_tiles(),
            "target exceeds capacity"
        );
        ensure!(
            f.image == [384, 384] && f.wg == [32, 16],
            "large fixture shape"
        );
    }
    Ok(())
}

// Deliberately evaluate the ORIGINAL strict predicate directly. In particular,
// do not normalize zero-width, inverted, or out-of-image u16 rectangles.
fn visits_tile(f: Fixture, dirty: Dirty, tile: usize) -> bool {
    let nx = f.tiles()[0];
    let tile = u32_index(tile);
    let x = tile % nx * f.wg[0];
    let y = tile / nx * f.wg[1];
    (x + f.wg[0]).min(f.image[0]) > dirty[0]
        && x < dirty[2]
        && (y + f.wg[1]).min(f.image[1]) > dirty[1]
        && y < dirty[3]
}

fn dirty_box(f: Fixture, slot: usize) -> Dirty {
    let [width, height] = f.image;
    let [tile_width, tile_height] = f.wg;
    if f.phase == 3 {
        return [0, 0, width, height];
    }
    match (slot + f.phase as usize * 5) % 16 {
        0 => [0, 0, width, height],
        1 => [
            tile_width - 1,
            tile_height - 1,
            tile_width - 1,
            tile_height - 1,
        ],
        2 => [tile_width - 1, 1, tile_width - 2, tile_height - 1],
        3 => [tile_width, 0, tile_width, height],
        4 => [width - 1, height - 1, width, height],
        5 => [65535, 65535, 0, 0],
        6 => [width + 1, 0, 65535, height],
        7 => [0, 0, 65535, 65535],
        8 => [tile_width + 1, tile_height + 1, 1, 1],
        9 => [0, tile_height, width, tile_height],
        10 => [1, 1, 2, 2],
        11 => [width, height, 65535, 65535],
        12 => [
            tile_width - 1,
            tile_height - 1,
            tile_width + 1,
            tile_height + 1,
        ],
        13 => [0, tile_height - 1, width, tile_height - 2],
        14 => [width - 1, height - 1, 65535, 65535],
        _ => [0, 0, 0, 0],
    }
}

fn budget_box(f: Fixture, remaining: &mut u32) -> Dirty {
    assert_eq!(f.num_tiles(), MAX_TILES);
    let take = if *remaining >= 288 {
        288
    } else if *remaining >= 12 {
        *remaining / 12 * 12
    } else {
        *remaining
    };
    *remaining -= take;
    if take >= 12 {
        [0, 0, f.image[0], take / 12 * f.wg[1]]
    } else {
        [0, 0, take * f.wg[0], f.wg[1]]
    }
}

fn rectangle(state: &GpuDrawingState) -> Dirty {
    [
        state.fitness_bits & 65535,
        state.fitness_bits >> 16,
        state.stagnation_counter & 65535,
        state.stagnation_counter >> 16,
    ]
}

fn expected_coverage(f: Fixture, scratch: &mut Scratch) {
    scratch.expected_prefix.fill(0);
    scratch.expected_visits.fill(0);
    let tiles = f.num_tiles();
    for slot in 0..f.active() {
        let mut count = 0;
        for tile in 0..tiles {
            let expected = u32::from(visits_tile(f, rectangle(&scratch.states[slot]), tile));
            scratch.expected_visits[slot * tiles + tile] = expected;
            count += expected;
        }
        scratch.expected_prefix[slot + 1] = scratch.expected_prefix[slot] + count;
    }
}

fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn u32_index(index: usize) -> u32 {
    u32::try_from(index).expect("fixed checker index fits u32")
}

fn packed(x: u32, y: u32) -> u32 {
    assert!(x <= 65535 && y <= 65535);
    x | y << 16
}

fn polygons(slot: usize) -> [GpuPolygon; 3] {
    let colors = [0x91_23_73_e1, 0x57_d3_39_11, 0xd1_41_19_ab];
    let vertices = [
        [(0, 0), (65535, 0), (0, 65535)],
        [(65535, 65535), (0, 65535), (65535, 0)],
        [(317, 1321), (64513, 2719), (32767, 65213)],
    ];
    std::array::from_fn(|i| GpuPolygon {
        data: [
            colors[i] ^ ((u32_index(slot) * 17) & 255),
            packed(vertices[i][0].0, vertices[i][0].1),
            packed(vertices[i][1].0, vertices[i][1].1),
            packed(vertices[i][2].0, vertices[i][2].1),
        ],
    })
}

impl Scratch {
    fn new() -> Self {
        let mut indices = vec![0xf0f0_f0f0; MAX_PAIRS * TILE_POLYS];
        for tile in indices.as_chunks_mut::<TILE_POLYS>().0 {
            tile[..3].copy_from_slice(&[0, 1, 2]);
        }
        let mut reference = vec![0; MAX_PIXELS * 4];
        for (i, pixel) in reference.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let key = mix(u32_index(i) ^ 0x6578_6163);
            pixel.copy_from_slice(&key.to_le_bytes());
        }
        // The fault fixture uses WG32x16. Its first two tiles have no polygons,
        // so both new and old error are zero despite nonuniform other tiles.
        for y in 0..16 {
            reference[y * 384 * 4..(y * 384 + 64) * 4].fill(255);
        }
        Self {
            states: vec![GpuDrawingState::zeroed(); CAPACITY],
            counts: vec![0xabab_abab; MAX_PAIRS],
            indices,
            framebuffers: vec![0; 4 * MAX_PIXELS],
            reference,
            initial_errors: std::array::from_fn(|i| 0x0100_0000 + u32_index(i) * 97),
            expected_prefix: [0; PREFIX_WORDS],
            expected_visits: vec![0; VISIT_WORDS],
            actual: vec![0; SNAPSHOT_WORDS],
        }
    }

    fn prepare(&mut self, f: Fixture) {
        let mut budget = f.target.unwrap_or(0);
        for slot in 0..f.active() {
            let dirty = if f.target.is_some() {
                budget_box(f, &mut budget)
            } else {
                dirty_box(f, slot)
            };
            let state = &mut self.states[slot];
            state.polygon_count = 3;
            state.fitness_bits = packed(dirty[0], dirty[1]);
            state.stagnation_counter = packed(dirty[2], dirty[3]);
            state.polygons[..3].copy_from_slice(&polygons(slot));
            for tile in 0..f.num_tiles() {
                self.counts[slot * f.num_tiles() + tile] = if f.phase == 3 && slot == 0 && tile < 2
                {
                    0
                } else {
                    [0, 1, 3][(slot + tile) % 3]
                };
            }
        }
        assert_eq!(budget, 0);
        let pixels = (f.image[0] * f.image[1]) as usize;
        for parent in 0..4 {
            for i in 0..pixels {
                let white =
                    parent == 0 && i % (f.image[0] as usize) < 64 && i / (f.image[0] as usize) < 16;
                self.framebuffers[parent * pixels + i] = if white {
                    u32::MAX
                } else {
                    mix(u32_index(i) ^ ((u32_index(parent) + 1) * 0x0123_4567)) | 0xff00_0000
                };
            }
        }
        expected_coverage(f, self);
        if let Some(total) = f.target {
            assert_eq!(self.expected_prefix[f.active()], total);
        }
    }
}

fn bytes(words: usize) -> u64 {
    (words * 4) as u64
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | extra,
    })
}

impl Buffers {
    fn new(p: &GpuPipeline) -> Self {
        let storage = wgpu::BufferUsages::empty();
        let states = buffer(
            &p.device,
            "grid working states",
            (CAPACITY * GPU_DRAWING_STATE_SIZE) as u64,
            storage,
        );
        let counts = buffer(&p.device, "grid tile counts", bytes(MAX_PAIRS), storage);
        let indices = buffer(
            &p.device,
            "grid ordered tile indices",
            bytes(MAX_PAIRS * TILE_POLYS),
            storage,
        );
        let framebuffers = buffer(
            &p.device,
            "grid cached parents",
            bytes(4 * MAX_PIXELS),
            storage,
        );
        let errors =
            std::array::from_fn(|_| buffer(&p.device, "grid errors", bytes(ERROR_WORDS), storage));
        let prefix = buffer(&p.device, "grid prefix", bytes(PREFIX_WORDS), storage);
        let args = buffer(
            &p.device,
            "grid indirect args",
            12,
            wgpu::BufferUsages::INDIRECT,
        );
        let visits = buffer(
            &p.device,
            "grid exact tile visits",
            bytes(VISIT_WORDS),
            storage,
        );
        let (params, staging) = transfer_buffers(&p.device);
        let texture = reference_texture(&p.device);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            states,
            counts,
            indices,
            framebuffers,
            errors,
            prefix,
            args,
            visits,
            params,
            staging,
            texture,
            view,
        }
    }

    fn initialize(&self, p: &GpuPipeline, scratch: &Scratch) {
        p.queue
            .write_buffer(&self.states, 0, bytemuck::cast_slice(&scratch.states));
        p.queue
            .write_buffer(&self.indices, 0, bytemuck::cast_slice(&scratch.indices));
        p.queue
            .write_buffer(&self.counts, 0, bytemuck::cast_slice(&scratch.counts));
        let poison: [u32; PREFIX_WORDS] = std::array::from_fn(|i| 0xf0f0_0000 | u32_index(i));
        p.queue
            .write_buffer(&self.prefix, 0, bytemuck::cast_slice(&poison));
        p.queue
            .write_buffer(&self.args, 0, bytemuck::cast_slice(&[0xffff_ffffu32; 3]));
        p.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &scratch.reference,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(384 * 4),
                rows_per_image: Some(384),
            },
            wgpu::Extent3d {
                width: 384,
                height: 384,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload(&self, p: &GpuPipeline, f: Fixture, scratch: &Scratch, index: usize) {
        let mut params = default_gpu_params(f.image[0], f.image[1], f.parents);
        params.lambda = f.lambda;
        params.iteration_number = u32_index(index);
        p.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        // Upload ONLY the active state/count prefix. The GPU tails retain their
        // actual history, including headers formerly consumed at larger N.
        p.queue.write_buffer(
            &self.states,
            0,
            bytemuck::cast_slice(&scratch.states[..f.active()]),
        );
        p.queue.write_buffer(
            &self.counts,
            0,
            bytemuck::cast_slice(&scratch.counts[..f.active() * f.num_tiles()]),
        );
        p.queue.write_buffer(
            &self.framebuffers,
            0,
            bytemuck::cast_slice(&scratch.framebuffers),
        );
        for errors in &self.errors {
            p.queue
                .write_buffer(errors, 0, bytemuck::cast_slice(&scratch.initial_errors));
        }
    }
}

fn transfer_buffers(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid params"),
        size: std::mem::size_of::<GpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid readback"),
        size: bytes(SNAPSHOT_WORDS),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (params, staging)
}

fn reference_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("grid reference bytes"),
        size: wgpu::Extent3d {
            width: 384,
            height: 384,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
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

fn coverage_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // Identical to the actual compact layout except binding 5's storage access.
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grid coverage layout"),
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
            storage_entry(5, false, GPU_DRAWING_STATE_SIZE as u64),
            storage_entry(6, true, 4),
            storage_entry(7, true, 4),
        ],
    })
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn raster_group(
    p: &GpuPipeline,
    b: &Buffers,
    layout: &wgpu::BindGroupLayout,
    compact: bool,
) -> wgpu::BindGroup {
    let entries = [
        binding(0, &b.states),
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&b.view),
        },
        binding(2, &b.errors[usize::from(!compact)]),
        binding(3, &b.indices),
        binding(4, &b.counts),
        binding(5, if compact { &b.visits } else { &b.states }),
        binding(6, &b.framebuffers),
        binding(7, &b.prefix),
    ];
    p.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grid raster bindings"),
        layout,
        entries: &entries[..if compact { 8 } else { 7 }],
    })
}

fn instrument(source: &str, fault: Fault) -> Result<String> {
    ensure!(source.len() <= MAX_SOURCE, "compact source exceeds bound");
    ensure!(
        source.matches(DECLARATION).count() == 1 && source.matches(MARKER).count() == 1,
        "compact instrumentation markers changed"
    );
    let source = fault_source(source, fault)?;
    let injection = format!("    if local_idx == 0u {{\n        if chain_id >= params.chain_count_param * params.lambda || shared_compact_x >= num_tiles_x || shared_compact_y >= num_tiles_y {{\n            atomicAdd(&checker_visits[{MAX_PAIRS}u], 1u);\n        }} else {{\n            atomicAdd(&checker_visits[chain_id * num_tiles + tile_id], 1u);\n        }}\n    }}\n\n{MARKER}");
    let source = source.replacen(
        DECLARATION,
        "@group(0) @binding(5) var<storage, read_write> checker_visits: array<atomic<u32>>;",
        1,
    );
    let output = source.replacen(MARKER, &injection, 1);
    ensure!(
        output.len() <= MAX_SOURCE,
        "instrumented source exceeds bound"
    );
    Ok(output)
}

fn fault_source(source: &str, fault: Fault) -> Result<String> {
    let (from, to) = match fault {
        Fault::None => return Ok(source.to_owned()),
        Fault::Missing => ("    if q >= compact_prefix[count] {",
            "    if q == 0u || q >= compact_prefix[count] {"),
        Fault::Duplicate => ("    let q = wid.x + grid.x * wid.y;",
            "    let checker_original_q = wid.x + grid.x * wid.y;\n    let q = select(checker_original_q, 0u, checker_original_q == 1u);"),
    };
    ensure!(
        source.matches(from).count() == 1,
        "actual entry fault marker changed"
    );
    Ok(source.replacen(from, to, 1))
}

fn prepare(
    p: &GpuPipeline,
    b: &Buffers,
    f: Fixture,
    fault: Fault,
    evidence: &mut Evidence,
) -> Result<Prepared> {
    let r = p
        .benchmark_compact_raster_resources()
        .context("compact resources missing after enable")?;
    let name = format!("{}x{}-{}", f.wg[0], f.wg[1], evidence.submitted);
    evidence.source(&format!("prefix-{name}"), &r.prefix_source)?;
    evidence.source(&format!("compact-{name}"), &r.raster_source)?;
    let source = instrument(&r.raster_source, fault)?;
    evidence.source(&format!("coverage-{name}"), &source)?;
    let params_layout = r.prefix_pipeline.get_bind_group_layout(1);
    let layout = coverage_layout(&p.device);
    let instrumented = compile_coverage(&p.device, &layout, &params_layout, &source);
    let prefix_group = p.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grid actual prefix layout"),
        layout: &r.prefix_layout,
        entries: &[
            binding(0, &b.states),
            binding(1, &b.prefix),
            binding(2, &b.args),
        ],
    });
    let compact_group = raster_group(p, b, &layout, true);
    let ordinary_layout = p.rasterize_error_pipeline.get_bind_group_layout(0);
    let ordinary_group = raster_group(p, b, &ordinary_layout, false);
    let params_group = p.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grid original Params layout"),
        layout: &params_layout,
        entries: &[binding(0, &b.params)],
    });
    Ok(Prepared {
        prefix_group,
        compact_group,
        ordinary_group,
        params_group,
        instrumented,
    })
}

fn compile_coverage(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::BindGroupLayout,
    source: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("grid instrumented pipeline layout"),
        bind_group_layouts: &[Some(layout), Some(params)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("actual compact entry plus coverage"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grid instrumented raster"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn remaining(deadline: Instant) -> Result<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    ensure!(
        !remaining.is_zero(),
        "internal 600s deadline exhausted; retain events.jsonl"
    );
    Ok(remaining)
}

fn scopes(device: &wgpu::Device) -> [wgpu::ErrorScopeGuard; 3] {
    [
        device.push_error_scope(wgpu::ErrorFilter::Validation),
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
        device.push_error_scope(wgpu::ErrorFilter::Internal),
    ]
}

fn finish_scopes(
    device: &wgpu::Device,
    scopes: [wgpu::ErrorScopeGuard; 3],
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    let [validation, memory, internal] = scopes;
    let mut failed = false;
    // Pop ALL guards in LIFO order before a fallible operation can exit.
    for future in [internal.pop(), memory.pop(), validation.pop()] {
        let mut future = std::pin::pin!(future);
        let mut context = TaskContext::from_waker(Waker::noop());
        let until = Instant::now() + remaining(deadline)?.min(WAIT);
        let mut done = false;
        for _ in 0..SCOPE_POLLS {
            if let Poll::Ready(error) = future.as_mut().poll(&mut context) {
                if let Some(error) = error {
                    evidence.event(
                        &json!({"event":"gpu_error", "error":short_error(&error.to_string())}),
                    )?;
                    failed = true;
                }
                done = true;
                break;
            }
            ensure!(Instant::now() < until, "GPU scope timeout");
            device.poll(wgpu::PollType::Poll).context("scope poll")?;
            std::thread::sleep(Duration::from_millis(1));
        }
        ensure!(done, "GPU scope poll budget exhausted");
    }
    ensure!(!failed, "GPU error; see events.jsonl");
    Ok(())
}

fn drain(p: &GpuPipeline, deadline: Instant) -> Result<()> {
    let submission = p.queue.submit([]);
    let status = p
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(remaining(deadline)?.min(WAIT)),
        })
        .context("drain engine initialization")?;
    ensure!(
        status.wait_finished(),
        "initialization drain did not finish"
    );
    Ok(())
}

fn run(options: &Options, evidence: &mut Evidence, deadline: Instant) -> Result<()> {
    ensure!(
        hash(ORIGINAL.as_bytes()) == ACCEPTED_HASH,
        "ordinary C source changed"
    );
    ensure!(
        std::mem::size_of::<GpuParams>() == 256,
        "Params ABI changed"
    );
    ensure!(
        GPU_DRAWING_STATE_SIZE == 16032 && std::mem::offset_of!(GpuDrawingState, polygons) == 32,
        "accepted C DrawingState ABI changed"
    );
    let cases = fixtures(options.fault);
    for &case in &cases {
        validate_fixture(case)?;
    }
    evidence.event(
        &json!({"event":"begin", "fixtures":cases.len(), "fault":options.fault,
        "capacity":CAPACITY, "max_tiles":MAX_TILES, "max_index_bytes":bytes(MAX_PAIRS * TILE_POLYS),
        "staging_bytes":bytes(SNAPSHOT_WORDS), "prefix_words":PREFIX_WORDS, "args_bytes":12,
        "poll_map_timeout_seconds":120, "outer_timeout_seconds":600, "scored":false,
        "coverage":"direct strict original tile predicate, exactly once or zero",
        "errors":"both stride-two accumulators before selection", "retained_history":true,
        "instrumented_compilation_options":"wgpu::PipelineCompilationOptions::default()"}),
    )?;
    evidence.source("ordinary-accepted-C", ORIGINAL)?;
    remaining(deadline)?;
    evidence.event(&json!({"event":"device_initialization_started"}))?;
    let mut scratch = Scratch::new();
    let initial = vec![GpuDrawingState::zeroed(); 4];
    let mut p =
        futures_lite::future::block_on(GpuPipeline::new(4, 384, 384, &scratch.reference, &initial));
    validate_device(&p)?;
    drain(&p, deadline)?;
    let guards = scopes(&p.device);
    let buffers = Buffers::new(&p);
    finish_scopes(&p.device, guards, evidence, deadline)?;
    evidence
        .event(&json!({"event":"device_ready", "limits":format!("{:?}", p.device.limits())}))?;
    run_cases(
        &mut p,
        &buffers,
        &mut scratch,
        &cases,
        options.fault,
        evidence,
        deadline,
    )
}

fn validate_device(p: &GpuPipeline) -> Result<()> {
    ensure!(
        p.offspring_capacity as usize >= CAPACITY,
        "engine state capacity too small"
    );
    let limits = p.device.limits();
    ensure!(
        limits.max_storage_buffers_per_shader_stage >= 7,
        "seven storage bindings required"
    );
    ensure!(
        limits.max_storage_buffer_binding_size >= bytes(MAX_PAIRS * TILE_POLYS)
            && limits.max_buffer_size >= bytes(MAX_PAIRS * TILE_POLYS),
        "checker buffer exceeds device limits"
    );
    ensure!(
        limits.max_compute_workgroups_per_dimension >= 65535,
        "dispatch limit too small"
    );
    Ok(())
}

fn prepare_workgroup(
    p: &mut GpuPipeline,
    b: &Buffers,
    f: Fixture,
    fault: Fault,
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<Prepared> {
    remaining(deadline)?;
    evidence.event(&json!({"event":"prepare_workgroup", "wg":f.wg}))?;
    let guards = scopes(&p.device);
    let result = (|| {
        // Every earlier submission was mapped/drained on this same thread.
        p.benchmark_set_compact_raster(false)
            .map_err(anyhow::Error::msg)?;
        p.set_rasterize_wg(f.wg);
        p.benchmark_set_compact_raster(true)
            .map_err(anyhow::Error::msg)?;
        prepare(p, b, f, fault, evidence)
    })();
    let checked = finish_scopes(&p.device, guards, evidence, deadline);
    let prepared = result?;
    checked?;
    Ok(prepared)
}

fn run_cases(
    p: &mut GpuPipeline,
    b: &Buffers,
    scratch: &mut Scratch,
    cases: &[Fixture],
    fault: Fault,
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    assert!(cases.len() <= FIXTURES);
    let mut prepared = None;
    let mut current_wg = None;
    for (index, &case) in cases.iter().enumerate() {
        remaining(deadline)?;
        if current_wg != Some(case.wg) {
            prepared = Some(prepare_workgroup(p, b, case, fault, evidence, deadline)?);
            current_wg = Some(case.wg);
        }
        let prepared = prepared.as_ref().context("workgroup preparation missing")?;
        scratch.prepare(case);
        evidence.event(
            &json!({"event":"fixture_started", "index":index, "fixture":case,
            "active":case.active(), "total":scratch.expected_prefix[case.active()]}),
        )?;
        let guards = scopes(&p.device);
        if index == 0 {
            b.initialize(p, scratch);
        }
        b.upload(p, case, scratch, index);
        let result = dispatch(p, b, prepared, case, scratch, evidence, deadline);
        let checked = finish_scopes(&p.device, guards, evidence, deadline);
        result?;
        checked?;
        let comparison = compare(case, scratch);
        let args = &scratch.actual[ARGS_OFFSET..VISITS_OFFSET];
        let total = scratch.expected_prefix[case.active()];
        let product = dispatch_product(args);
        evidence.event(
            &json!({"event":"fixture_result", "index":index, "fixture":case,
            "comparison":comparison, "total":total, "args":args,
            "product":product, "padding":product.map(|p| p.saturating_sub(u64::from(total)))}),
        )?;
        if fault != Fault::None {
            return reject_fault(fault, &comparison, evidence);
        }
        ensure!(
            comparison.passed(),
            "fixture {index} failed: {:?}",
            comparison.first
        );
        evidence.passed += 1;
    }
    evidence.event(&json!({"event":"all_fixtures_passed", "count":evidence.passed}))?;
    Ok(())
}

fn reject_fault(fault: Fault, comparison: &Comparison, evidence: &mut Evidence) -> Result<()> {
    ensure!(
        comparison.errors == 0 && comparison.inactive_errors == 0,
        "negative fixture must retain equal error arrays"
    );
    ensure!(
        comparison.prefix == 0 && comparison.args == 0 && comparison.unexpected == 0,
        "negative fixture changed unrelated observations"
    );
    ensure!(
        match fault {
            Fault::Missing => comparison.missing == 1 && comparison.duplicate == 0,
            Fault::Duplicate => comparison.duplicate == 1 && comparison.missing == 1,
            Fault::None => false,
        },
        "negative fixture did not produce exactly the requested coverage fault"
    );
    evidence.event(
        &json!({"event":"negative_proof", "fault":fault, "rejected":true,
        "error_arrays_equal":true, "zero_error_tiles":true}),
    )?;
    bail!("intentional {fault:?} coverage fault rejected");
}

fn encode(
    p: &GpuPipeline,
    b: &Buffers,
    prepared: &Prepared,
    f: Fixture,
) -> Result<wgpu::CommandEncoder> {
    let r = p
        .benchmark_compact_raster_resources()
        .context("compact resources missing")?;
    let mut encoder = p
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grid fixture"),
        });
    encoder.clear_buffer(&b.visits, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("actual compact prefix"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.prefix_pipeline);
        pass.set_bind_group(0, &prepared.prefix_group, &[]);
        pass.set_bind_group(1, &prepared.params_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("actual compact entry with visits"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&prepared.instrumented);
        pass.set_bind_group(0, &prepared.compact_group, &[]);
        pass.set_bind_group(1, &prepared.params_group, &[]);
        pass.dispatch_workgroups_indirect(&b.args, 0);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("actual ordinary C before selection"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&p.rasterize_error_pipeline);
        pass.set_bind_group(0, &prepared.ordinary_group, &[]);
        pass.set_bind_group(1, &prepared.params_group, &[]);
        let [tiles_x, tiles_y] = f.tiles();
        pass.dispatch_workgroups(tiles_x, tiles_y, u32_index(f.active()));
    }
    for (buffer, offset, words) in [
        (&b.errors[0], 0, ERROR_WORDS),
        (&b.errors[1], ERROR_WORDS, ERROR_WORDS),
        (&b.prefix, PREFIX_OFFSET, PREFIX_WORDS),
        (&b.args, ARGS_OFFSET, 3),
        (&b.visits, VISITS_OFFSET, VISIT_WORDS),
    ] {
        encoder.copy_buffer_to_buffer(buffer, 0, &b.staging, bytes(offset), bytes(words));
    }
    Ok(encoder)
}

fn dispatch(
    p: &GpuPipeline,
    b: &Buffers,
    prepared: &Prepared,
    f: Fixture,
    scratch: &mut Scratch,
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    ensure!(evidence.submitted < FIXTURES, "submission budget exhausted");
    let encoder = encode(p, b, prepared, f)?;
    let submission = p.queue.submit([encoder.finish()]);
    evidence.submitted += 1;
    evidence.event(
        &json!({"event":"submitted", "submission":evidence.submitted,
        "dispatches":3, "raster":"indirect"}),
    )?;
    let (tx, rx) = mpsc::sync_channel(1);
    b.staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            // The receiver can disappear after timeout; sending then owns no map.
            let _cancelled = tx.send(result);
        });
    let result = readback(p, b, submission, &rx, &mut scratch.actual, deadline);
    b.staging.unmap();
    result?;
    evidence.event(&json!({"event":"readback_completed", "submission":evidence.submitted}))?;
    Ok(())
}

fn readback(
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
        .context("grid GPU readback poll")?;
    ensure!(status.wait_finished(), "GPU submission did not finish");
    rx.recv_timeout(until.saturating_duration_since(Instant::now()))
        .context("grid map callback timeout")?
        .context("grid staging map")?;
    let mapped = b
        .staging
        .slice(..)
        .get_mapped_range()
        .context("grid mapped range")?;
    ensure!(
        mapped.len() == SNAPSHOT_WORDS * 4 && actual.len() == SNAPSHOT_WORDS,
        "readback size mismatch"
    );
    actual.copy_from_slice(
        bytemuck::try_cast_slice::<u8, u32>(&mapped)
            .map_err(|error| anyhow::anyhow!("unaligned readback: {error}"))?,
    );
    drop(mapped);
    Ok(())
}

fn expected_args(total: u32) -> [u32; 3] {
    if total == 0 {
        return [0, 1, 1];
    }
    let y = total.div_ceil(65535);
    [total.div_ceil(y), y, 1]
}

fn dispatch_product(args: &[u32]) -> Option<u64> {
    assert_eq!(args.len(), 3);
    u64::from(args[0])
        .checked_mul(u64::from(args[1]))?
        .checked_mul(u64::from(args[2]))
}

fn check_visits(actual: &[u32], expected: &[u32], comparison: &mut Comparison) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.len() <= VISIT_WORDS);
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if got == want {
            continue;
        }
        if want == 0 {
            comparison.unexpected += 1;
        } else if got == 0 {
            comparison.missing += 1;
        } else {
            comparison.duplicate += 1;
        }
        comparison.note(format!("visit word {index}: expected {want}, got {got}"));
    }
}

fn compare(f: Fixture, scratch: &Scratch) -> Comparison {
    let mut comparison = Comparison::default();
    let actual = &scratch.actual;
    for i in 0..=f.active() {
        let got = actual[PREFIX_OFFSET + i];
        let want = scratch.expected_prefix[i];
        if got != want {
            comparison.prefix += 1;
            comparison.note(format!("prefix[{i}]: expected {want}, got {got}"));
        }
    }
    let total = scratch.expected_prefix[f.active()];
    let args = &actual[ARGS_OFFSET..VISITS_OFFSET];
    if args != expected_args(total) {
        comparison.args += 1;
        comparison.note(format!(
            "args: expected {:?}, got {args:?}",
            expected_args(total)
        ));
    }
    // These checks also describe the padding contract independent of exact args.
    let product = dispatch_product(args);
    if args.iter().any(|&v| v > 65535)
        || args[2] != 1
        || product.is_none_or(|product| {
            product < u64::from(total) || product - u64::from(total) >= u64::from(args[1])
        })
    {
        comparison.args += 1;
        comparison.note(format!(
            "invalid dimensions/product/padding: {args:?}, total {total}"
        ));
    }
    check_visits(
        &actual[VISITS_OFFSET..],
        &scratch.expected_visits,
        &mut comparison,
    );
    for i in 0..ERROR_WORDS {
        if actual[i] != actual[ERROR_WORDS + i] {
            comparison.errors += 1;
            comparison.note(format!(
                "accumulator[{i}]: compact {}, ordinary {}",
                actual[i],
                actual[ERROR_WORDS + i]
            ));
        }
        if i >= f.active() * 2
            && (actual[i] != scratch.initial_errors[i]
                || actual[ERROR_WORDS + i] != scratch.initial_errors[i])
        {
            comparison.inactive_errors += 1;
            comparison.note(format!("inactive accumulator[{i}] changed"));
        }
    }
    comparison
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Fixture {
        Fixture {
            image: [65, 51],
            wg: [32, 16],
            lambda: 3,
            parents: 4,
            phase: 0,
            target: None,
        }
    }

    #[test]
    fn strict_predicate_preserves_zero_and_inverted_consumers() {
        let f = small();
        assert!(visits_tile(f, [31, 15, 31, 15], 0));
        assert!(visits_tile(f, [31, 1, 30, 15], 0));
        assert!(!visits_tile(f, [32, 0, 32, 51], 0));
        assert!(!visits_tile(f, [33, 17, 1, 1], 0));
        assert!(visits_tile(f, [64, 50, 65535, 65535], f.num_tiles() - 1));
        assert!(!visits_tile(f, [65, 51, 65535, 65535], f.num_tiles() - 1));
    }

    #[test]
    fn coverage_rejects_missing_duplicate_and_unexpected_zero_error_tiles() {
        for (got, missing, duplicate, unexpected) in [
            ([0, 0, 1], 1, 0, 0),
            ([2, 0, 1], 0, 1, 0),
            ([1, 1, 1], 0, 0, 1),
        ] {
            let mut comparison = Comparison::default();
            check_visits(&got, &[1, 0, 1], &mut comparison);
            assert!(!comparison.passed());
            assert_eq!(
                (
                    comparison.missing,
                    comparison.duplicate,
                    comparison.unexpected
                ),
                (missing, duplicate, unexpected)
            );
            assert_eq!(comparison.errors, 0);
        }
        let mut comparison = Comparison::default();
        check_visits(&[1, 0, 1], &[1, 0, 1], &mut comparison);
        assert!(comparison.passed());
    }

    #[test]
    fn flattened_dimensions_have_bounded_padding() {
        for (total, args) in [
            (0, [0, 1, 1]),
            (65534, [65534, 1, 1]),
            (65535, [65535, 1, 1]),
            (65536, [32768, 2, 1]),
            (65537, [32769, 2, 1]),
            (73728, [36864, 2, 1]),
        ] {
            assert_eq!(expected_args(total), args);
            let product = args[0] * args[1];
            assert!(product >= total && product - total < args[1]);
        }
        assert_eq!(dispatch_product(&[u32::MAX; 3]), None);
    }

    #[test]
    fn fixture_plan_is_finite_and_has_history_and_padding() {
        let cases = fixtures(Fault::None);
        assert_eq!(cases.len(), FIXTURES);
        for case in &cases {
            validate_fixture(*case).expect("bounded built-in fixture");
        }
        assert!(cases.iter().any(|f| f.target == Some(65537)));
        assert!(cases
            .windows(2)
            .any(|pair| pair[0].active() > pair[1].active()));
        for wg in WORKGROUPS {
            for lambda in LAMBDAS {
                assert_eq!(
                    cases
                        .iter()
                        .filter(|f| f.wg == wg && f.lambda == lambda && f.target.is_none())
                        .count(),
                    3
                );
            }
        }
    }

    #[test]
    fn large_target_rectangles_match_direct_tile_predicate() {
        for case in fixtures(Fault::None) {
            if let Some(target) = case.target {
                let mut budget = target;
                let mut total = 0;
                for _ in 0..case.active() {
                    let dirty = budget_box(case, &mut budget);
                    for tile in 0..case.num_tiles() {
                        total += u32::from(visits_tile(case, dirty, tile));
                    }
                }
                assert_eq!(budget, 0);
                assert_eq!(total, target);
            }
        }
    }

    #[test]
    fn cli_rejects_bad_arguments_before_initialization() {
        assert!(parse_args([OsString::from("--help")].into_iter())
            .unwrap()
            .is_none());
        for args in [
            vec![],
            vec!["--output"],
            vec!["--other", "x"],
            vec!["--output", "x", "--fault", "unknown"],
            vec!["--output", "x", "--fault", "missing", "extra"],
        ] {
            assert!(parse_args(args.into_iter().map(OsString::from)).is_err());
        }
    }

    #[test]
    fn instrumentation_requires_unique_markers_and_preserves_suffix() {
        let source = format!("{DECLARATION}\n{MARKER}\nFLOAT_BODY_UNCHANGED");
        let instrumented = instrument(&source, Fault::None).unwrap();
        assert!(instrumented.ends_with(&format!("{MARKER}\nFLOAT_BODY_UNCHANGED")));
        assert!(instrument(&source.replace(MARKER, ""), Fault::None).is_err());
        assert!(instrument(&format!("{source}\n{MARKER}"), Fault::None).is_err());
    }
}
