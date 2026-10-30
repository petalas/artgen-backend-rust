//! Numerical mutation equivalence, entirely on the calling/main thread.
//!
//! After building, run under an OUTER process timeout (construction and native
//! shader compilation cannot be interrupted by this checker):
//! `timeout 600s target/release/examples/check-gpu-mutation --reference BASE.wgsl
//! --candidate copy2=COPY2.wgsl --candidate copy4=COPY4.wgsl
//! --candidate copy8=COPY8.wgsl --output NEW_DIRECTORY`.
//! `--help` parses no sources, creates no files, and initializes no GPU.
//!
//! Sources are compiled once through the public mutation pipeline's
//! actual bind-group layouts, with mutation-only compilation options. Reference zeroing stays true; candidate
//! zeroing defaults to true and can be disabled explicitly. The only rewrite is
//! the engine's `specialize_polygon_storage`, which gives a historical source's
//! `DrawingState` the engine's `POLYGON_CAPACITY` stride; every source must
//! support the fixtures' 1000-polygon cap. No other engine API, solver, source
//! rewriting, or scored campaign is involved. In particular,
//! marker-free physical workgroups of 128/256/512 lanes retain dispatch(4, 1, 1).
//! An exact `OFFSPRING_GROUP_SIZE` declaration changes only dispatch Y to
//! `ceil(fixture.lambda / source_group_size)`, using the engine's checked parser.
//! A candidate can clone in 2/4/8 lanes per child, `storageBarrier()`, then run the
//! original RNG/mutation body in the first 64 lanes. This tool does not assume
//! that implementation. Every header bit and every active polygon word must
//! agree. Inactive polygon padding may differ, and its differences are logged.
//! A later active tail is always compared, even if it used to be padding.
//!
//! There are 220 fixtures: 128 small-parent cases covering every integer lambda
//! 1..64 with adaptive scale off and on at crossover .2; 20 small cases at
//! lambdas 1/3/32/33/64 and crossover 0/1; 36 targeted dense cases plus 36
//! immediate small-parent history probes. Three candidates mean 880 GPU
//! dispatches; four mean 1100. All 1024 parents remain bound because tournament
//! selection uses arrayLength, although only four chains produce children.
//! The maximum is 256 offspring, each with `POLYGON_CAPACITY` (2048) physical
//! polygon slots; fixtures cap drawings at 1000 polygons.
//!
//! Resource/checkpoint decisions: only this example is owned. Device setup and
//! shader compilation block the main thread; buffer reuse and comparison form
//! one serial workstream. One 33,587,200-byte parent buffer, one 8,396,800-byte
//! output, and one 8,398,848-byte staging buffer are reused. Each variant keeps
//! its own full CPU history, including padding and RNG, so untouched slots are
//! checked against that variant's prior bytes. Histories are never normalized.
//! The normal pipeline is created for 16 chains at 17x19 solely to request the
//! binding limit needed by the checker-owned 1024-parent buffer. Its evolution
//! buffers are unused. No pipeline cache is saved by this program.
//!
//! Power-of-ten deviations, confined to this finite checker:
//! Rules 2/7: existing `GpuPipeline::new`, native compilation and the unchanged
//! source's RNG rejection loops need the required outer 600-second process
//! timeout. Own polls/maps/scopes have 120-second deadlines and bounded loops;
//! an internal 600-second deadline also limits work between native calls.
//! Rule 3: wgpu/driver staging, error formatting and bounded JSON serialization
//! allocate at runtime. Checker arrays are allocated before fixture execution:
//! <=5 histories plus one scratch snapshot, <=400 fixtures, <=5 sources of
//! 256 KiB, <=12 first-failure samples per result. One GPU submission/map is live
//! at a time. Framework allocations cannot be inspected here; input, submission
//! and fixture caps bound their use, and all resources drop on normal errors.
//! Rules 8/9: serde derives, formatting/assertion macros and wgpu's one bounded
//! map callback use framework interfaces. There is no recursive callback chain.
//! Rule 10: source preparation does not authorize Cargo or GPU execution. The
//! parent must grant a CPU slot before compilation and real readback validation.
//! Follow-up for these deviations is parent-owned build/GPU proof under the
//! outer timeout before adoption; preserve these bounds when extending cases.
//! events.jsonl is flushed after each event, including submissions, completed
//! readbacks, mismatches and errors. result.json summarizes normal termination.
//! Kills/native aborts leave the preceding events and copied source hashes.

use std::{
    env,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    future::Future,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    task::{Context as TaskContext, Poll, Waker},
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::{
    gpu_evolver::{
        benchmark::{
            benchmark_mutate_compilation_options, benchmark_mutate_group_size_from_source,
            BenchmarkMutateOptions,
        },
        buffers::{default_gpu_params, GpuDrawingState, GpuParams, GPU_DRAWING_STATE_SIZE},
        pipeline::{specialize_polygon_storage, GpuPipeline},
        EngineCapacity,
    },
    settings::POLYGON_CAPACITY,
};
use bytemuck::Zeroable;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const ACTIVE_CHAINS: usize = 4;
const PARENTS: usize = 1024;
const MAX_LAMBDA: usize = 64;
const OFFSPRING: usize = ACTIVE_CHAINS * MAX_LAMBDA;
// The fixtures' polygon cap and largest drawing, the historical reference's limit.
const POLYGONS: usize = 1000;
// Physical polygon slots per state: the engine's storage stride.
const SLOTS: usize = POLYGON_CAPACITY;
const HEADER_WORDS: usize = 8;
const POLYGON_WORDS: usize = 4;
const STATE_WORDS: usize = HEADER_WORDS + SLOTS * POLYGON_WORDS;
const OUTPUT_WORDS: usize = OFFSPRING * STATE_WORDS;
const ERROR_WORDS: usize = OFFSPRING * 2;
const SNAPSHOT_WORDS: usize = OUTPUT_WORDS + ERROR_WORDS;
const PARENT_BYTES: u64 = (PARENTS * STATE_WORDS * 4) as u64;
const OUTPUT_BYTES: u64 = (OUTPUT_WORDS * 4) as u64;
const ERROR_BYTES: u64 = (ERROR_WORDS * 4) as u64;
const SNAPSHOT_BYTES: u64 = (SNAPSHOT_WORDS * 4) as u64;
const MAX_CANDIDATES: usize = 4;
const MAX_VARIANTS: usize = MAX_CANDIDATES + 1;
const MAX_ARGS: usize = 2 * (MAX_CANDIDATES + 3);
const MAX_PATH_BYTES: usize = 4096;
const MAX_SOURCE_BYTES: usize = 256 * 1024;
const MAX_ERROR_CHARS: usize = 16 * 1024;
const MAX_SAMPLES: usize = 12;
const MAX_FIXTURES: usize = 400;
const FIXTURES: usize = 220;
const MAX_DISPATCHES: usize = 2000;
const WAIT: Duration = Duration::from_secs(120);
const OUTER: Duration = Duration::from_secs(600);
const SCOPE_POLLS: usize = 120_000;
const IMAGE: [u32; 2] = [17, 19];
const SMALL: [u32; 4] = [0, 1, 3, 17];
const DENSE: [[u32; 4]; 3] = [[63, 64, 65, 0], [511, 512, 513, 1], [999, 1000, 0, 3]];
const SEED: u32 = 0x6d75_7461;

struct SourceArg {
    name: String,
    path: PathBuf,
}

struct Options {
    sources: Vec<SourceArg>,
    output: PathBuf,
    candidate_mutation: BenchmarkMutateOptions,
}

struct Source {
    name: String,
    /// The source text with the engine's drawing-state stride; what is compiled.
    /// Evidence keeps the original text and its hash.
    compiled: String,
    sha256: String,
    mutate_group_size: u32,
    options: BenchmarkMutateOptions,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Fixture {
    counts: [u32; 4],
    lambda: u32,
    adaptive: bool,
    crossover: f32,
    history_probe: bool,
}

struct Variant {
    source: Source,
    pipeline: wgpu::ComputePipeline,
    prior: Vec<u32>,
}

struct Buffers {
    parents: wgpu::Buffer,
    output: wgpu::Buffer,
    errors: wgpu::Buffer,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
    states_group: wgpu::BindGroup,
    params_group: wgpu::BindGroup,
}

struct Inputs {
    parents: Vec<u32>,
    counts: Option<[u32; 4]>,
    sha256: String,
}

struct Evidence {
    dir: PathBuf,
    events: File,
    submitted: usize,
    completed: usize,
    fixtures: usize,
    comparisons: usize,
    failed: usize,
}

#[derive(Serialize)]
struct Difference {
    kind: &'static str,
    slot: usize,
    word: usize,
    expected: u32,
    actual: u32,
}

#[derive(Default, Serialize)]
struct Comparison {
    invalid_counts: usize,
    header_words: usize,
    active_polygon_words: usize,
    unproduced_slot_words_changed: usize,
    error_words_changed: usize,
    error_words_different: usize,
    inactive_padding_words_different: usize,
    first: Vec<Difference>,
    first_padding: Option<Difference>,
}

fn main() -> Result<()> {
    let Some(options) = parse_args(env::args_os().skip(1).take(MAX_ARGS + 1))? else {
        print_help();
        return Ok(());
    };
    let mut evidence = Evidence::new(&options.output)?;
    let started = Instant::now();
    // Rust's release panic=abort cannot be caught. All earlier events are already
    // flushed; ordinary GPU/scoped/map errors take the Result path below.
    let outcome = run(&options, &mut evidence, started + OUTER);
    evidence.finish(&outcome)?;
    outcome
}

fn parse_args(args: impl Iterator<Item = OsString>) -> Result<Option<Options>> {
    let args: Vec<_> = args.take(MAX_ARGS + 1).collect();
    ensure!(args.len() <= MAX_ARGS, "too many arguments");
    if args.len() == 1 && args[0] == "--help" {
        return Ok(None);
    }
    let mut reference = None;
    let mut candidates = Vec::with_capacity(MAX_CANDIDATES);
    let mut output = None;
    let mut mutate_zero_initialize = None;
    ensure!(args.len() % 2 == 0, "options require values; use --help");
    for pair in args.as_chunks::<2>().0 {
        let flag = bounded_arg(&pair[0])?;
        let value = bounded_arg(&pair[1])?;
        match flag {
            "--reference" => {
                ensure!(reference.is_none(), "duplicate --reference");
                reference = Some(SourceArg {
                    name: "reference".into(),
                    path: value.into(),
                });
            }
            "--candidate" => add_candidate(&mut candidates, value)?,
            "--output" => {
                ensure!(output.is_none(), "duplicate --output");
                output = Some(PathBuf::from(value));
            }
            "--mutate-zero-initialize" => {
                ensure!(
                    mutate_zero_initialize.is_none(),
                    "duplicate --mutate-zero-initialize"
                );
                mutate_zero_initialize = Some(value.parse::<bool>()?);
            }
            _ => bail!("unknown option {flag:?}; use --help"),
        }
    }
    ensure!(
        !candidates.is_empty(),
        "at least one --candidate is required"
    );
    let mut sources = Vec::with_capacity(MAX_VARIANTS);
    sources.push(reference.context("missing --reference")?);
    sources.extend(candidates);
    Ok(Some(Options {
        sources,
        output: output.context("missing --output")?,
        candidate_mutation: BenchmarkMutateOptions {
            zero_initialize_workgroup_memory: mutate_zero_initialize.unwrap_or(true),
        },
    }))
}

fn bounded_arg(arg: &OsString) -> Result<&str> {
    let value = arg.to_str().context("arguments must be UTF-8")?;
    ensure!(
        !value.is_empty() && value.len() <= MAX_PATH_BYTES,
        "argument must contain 1..4096 bytes"
    );
    ensure!(
        !value.chars().any(char::is_control),
        "control characters are not allowed in arguments"
    );
    Ok(value)
}

fn add_candidate(candidates: &mut Vec<SourceArg>, value: &str) -> Result<()> {
    ensure!(
        candidates.len() < MAX_CANDIDATES,
        "at most four candidates are allowed"
    );
    let (name, path) = value
        .split_once('=')
        .context("candidate must be NAME=PATH")?;
    ensure!(
        !name.is_empty()
            && name.len() <= 32
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
        "candidate name must contain 1..32 ASCII letters/digits/_/-"
    );
    ensure!(
        name != "reference" && candidates.iter().all(|c| c.name != name),
        "duplicate/reserved candidate name"
    );
    ensure!(!path.is_empty(), "candidate path is empty");
    candidates.push(SourceArg {
        name: name.into(),
        path: path.into(),
    });
    Ok(())
}

fn print_help() {
    println!("check-gpu-mutation --reference PATH --candidate NAME=PATH [--candidate NAME=PATH ...] --output NEWDIR\n\
        [--mutate-zero-initialize true|false] applies only to candidates, default true. Reference uses true.\n\
        Compare one reference with 1..4 mutation sources on the main thread.\n\
        NEWDIR must not exist; its parent must exist. UTF-8 sources: 1..256 KiB.\n\
        Run the built executable under `timeout 600s`. Poll/map limit: 120s.\n\
        220 fixtures, dispatch(4,1,1), 1024 parents, up to 256 children.\n\
        Compare all header bits, active polygons/counts, errors and untouched slots.\n\
        Log inactive polygon padding differences without treating them as semantic.\n\
        Save exact sources/SHA-256, events.jsonl and result.json. No throughput scores.\n\
        --help prints this text without filesystem writes or GPU initialization.");
}

impl Evidence {
    fn new(path: &Path) -> Result<Self> {
        // create_dir deliberately refuses existing destinations, including links.
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent = fs::canonicalize(parent).context("output parent must exist")?;
        let leaf = path
            .file_name()
            .context("output must name a new directory")?;
        let dir = parent.join(leaf);
        bounded_path(&dir)?;
        fs::create_dir(&dir).context("create NEWDIR; existing output is never overwritten")?;
        let events = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("events.jsonl"))?;
        Ok(Self {
            dir,
            events,
            submitted: 0,
            completed: 0,
            fixtures: 0,
            comparisons: 0,
            failed: 0,
        })
    }

    fn event(&mut self, value: &Value) -> Result<()> {
        serde_json::to_writer(&mut self.events, value)?;
        self.events.write_all(b"\n")?;
        self.events.flush().context("flush incremental evidence")
    }

    fn finish(&mut self, outcome: &Result<()>) -> Result<()> {
        let error = outcome
            .as_ref()
            .err()
            .map(|e| short_error(&format!("{e:#}")));
        if let Some(message) = &error {
            self.event(&json!({"event":"error", "message":message}))?;
        }
        let result = json!({"event":"result", "schema":1, "passed":outcome.is_ok(),
            "fixtures_completed":self.fixtures, "expected_fixtures":FIXTURES,
            "dispatches_submitted":self.submitted, "dispatches_completed":self.completed,
            "candidate_comparisons":self.comparisons, "failed_results":self.failed,
            "error":error, "scored_campaign":false});
        self.event(&result)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.dir.join("result.json"))?;
        serde_json::to_writer_pretty(file, &result).context("write final result")
    }
}

fn bounded_path(path: &Path) -> Result<&str> {
    let text = path.to_str().context("resolved path is not UTF-8")?;
    ensure!(
        text.len() <= MAX_PATH_BYTES,
        "resolved path exceeds 4096 bytes"
    );
    Ok(text)
}

fn load_sources(options: &Options, evidence: &mut Evidence) -> Result<Vec<Source>> {
    let mut sources = Vec::with_capacity(MAX_VARIANTS);
    for (index, arg) in options.sources.iter().enumerate() {
        let compilation_options = if index == 0 {
            BenchmarkMutateOptions::default()
        } else {
            options.candidate_mutation
        };
        let (path, text) =
            read_source(&arg.path).with_context(|| format!("load source {}", arg.name))?;
        let path_text = bounded_path(&path)?;
        let sha256 = hash(text.as_bytes());
        let mutate_group_size = benchmark_mutate_group_size_from_source(&text)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("mutation group metadata for {}", arg.name))?;
        let storage = specialize_polygon_storage(&text)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("polygon storage of {}", arg.name))?;
        ensure!(
            storage.polygon_limit >= POLYGONS,
            "{} supports {} polygons; the fixtures need {POLYGONS}",
            arg.name,
            storage.polygon_limit
        );
        let copy = format!("{}.wgsl", arg.name);
        let mut destination = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(evidence.dir.join(&copy))?;
        destination.write_all(text.as_bytes())?;
        destination.flush()?;
        evidence.event(&json!({"event":"source", "name":arg.name, "path":path_text,
            "copy":copy, "sha256":sha256, "bytes":text.len(),
            "mutate_group_size":mutate_group_size, "polygon_limit":storage.polygon_limit,
            "constants":benchmark_mutate_compilation_options(compilation_options).constants,
            "zero_initialize_workgroup_memory":compilation_options.zero_initialize_workgroup_memory}))?;
        sources.push(Source {
            name: arg.name.clone(),
            compiled: storage.source,
            sha256,
            mutate_group_size,
            options: compilation_options,
        });
    }
    Ok(sources)
}

fn read_source(path: &Path) -> Result<(PathBuf, String)> {
    let path = fs::canonicalize(path).context("resolve source")?;
    bounded_path(&path)?;
    // Reject nonfiles before open, so an ordinary FIFO/device path cannot block
    // source preparation. Recheck the opened file before reading it as well.
    let metadata = fs::metadata(&path)?;
    ensure!(metadata.is_file(), "source must be a regular file");
    ensure!(
        metadata.len() > 0 && metadata.len() <= MAX_SOURCE_BYTES as u64,
        "source must contain 1..262144 bytes"
    );
    let file = File::open(&path).context("open source")?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_SOURCE_BYTES as u64,
        "opened source changed type or size"
    );
    let mut bytes = Vec::with_capacity(MAX_SOURCE_BYTES + 1);
    file.take((MAX_SOURCE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_SOURCE_BYTES,
        "source changed beyond the size bound"
    );
    let text = String::from_utf8(bytes).context("WGSL source must be UTF-8")?;
    Ok((path, text))
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn short_error(message: &str) -> String {
    message.chars().take(MAX_ERROR_CHARS).collect()
}

fn run(options: &Options, evidence: &mut Evidence, deadline: Instant) -> Result<()> {
    check_abi()?;
    let fixtures = fixtures();
    evidence.event(
        &json!({"event":"begin", "schema":1, "fixtures":fixtures.len(),
        "variants":options.sources.len(), "dispatch_rule":"[4,ceil(lambda/mutate_group_size),1]",
        "parents":PARENTS,
        "offspring_capacity":OFFSPRING, "polygons_per_slot":SLOTS, "max_polygons":POLYGONS,
        "parent_bytes":PARENT_BYTES, "output_bytes":OUTPUT_BYTES, "staging_bytes":SNAPSHOT_BYTES,
        "seed":SEED, "outer_process_timeout_seconds":600, "poll_map_timeout_seconds":120,
        "semantics":"all header bits and active polygon prefix; padding differences logged",
        "unproduced_slots":"compare every word against variant's own prior full snapshot",
        "compilation_options":"per-source mutation policy; see source and compilation events",
        "reference_zero_initialize_workgroup_memory":true,
        "candidate_zero_initialize_workgroup_memory":options.candidate_mutation.zero_initialize_workgroup_memory,
        "scored_campaign":false}),
    )?;
    let sources = load_sources(options, evidence)?;
    remaining(deadline)?;
    evidence.event(&json!({"event":"device_initialization_started"}))?;
    let initial = vec![GpuDrawingState::zeroed(); 16];
    let image = [0u8, 0, 0, 255].repeat((IMAGE[0] * IMAGE[1]) as usize);
    let p = futures_lite::future::block_on(GpuPipeline::new(
        EngineCapacity::new(16, 64),
        IMAGE[0],
        IMAGE[1],
        &image,
        &initial,
    ));
    evidence
        .event(&json!({"event":"device_ready", "limits":format!("{:?}", p.device.limits())}))?;
    remaining(deadline)?;
    ensure!(
        p.device.limits().max_storage_buffer_binding_size >= PARENT_BYTES
            && p.device.limits().max_buffer_size >= PARENT_BYTES,
        "device cannot bind all 1024 parents; tournament ABI must not be shortened"
    );
    let layouts = [
        p.mutate_pipeline().get_bind_group_layout(0),
        p.mutate_pipeline().get_bind_group_layout(1),
    ];
    let scopes = error_scopes(&p.device);
    let buffers = new_buffers(&p.device, &layouts);
    finish_scopes(&p.device, scopes, evidence, "buffer_setup", deadline)?;
    let variants = compile_variants(&p.device, &layouts, sources, evidence, deadline)?;
    run_fixtures(&p, &buffers, variants, &fixtures, evidence, deadline)
}

fn check_abi() -> Result<()> {
    ensure!(
        POLYGONS <= SLOTS,
        "engine polygon capacity is below the fixtures' cap"
    );
    ensure!(
        GPU_DRAWING_STATE_SIZE == STATE_WORDS * 4,
        "engine DrawingState ABI changed"
    );
    ensure!(
        std::mem::size_of::<GpuParams>() == 256,
        "engine Params ABI changed"
    );
    ensure!(
        std::mem::offset_of!(GpuDrawingState, polygons) == HEADER_WORDS * 4,
        "polygon offset changed"
    );
    ensure!(
        std::mem::offset_of!(GpuDrawingState, rng_state) == 16,
        "RNG offset changed"
    );
    const {
        assert!(PARENT_BYTES <= 33 * 1024 * 1024);
        assert!(OUTPUT_BYTES < 9 * 1024 * 1024 && SNAPSHOT_BYTES < 9 * 1024 * 1024);
        assert!(MAX_VARIANTS * FIXTURES <= MAX_DISPATCHES);
    }
    Ok(())
}

fn fixtures() -> Vec<Fixture> {
    let mut cases = Vec::with_capacity(MAX_FIXTURES);
    for lambda in 1..=u32::try_from(MAX_LAMBDA).expect("lambda limit fits u32") {
        for adaptive in [false, true] {
            cases.push(Fixture {
                counts: SMALL,
                lambda,
                adaptive,
                crossover: 0.2,
                history_probe: false,
            });
        }
    }
    for lambda in [1, 3, 32, 33, 64] {
        for crossover in [0.0, 1.0] {
            for adaptive in [false, true] {
                cases.push(Fixture {
                    counts: SMALL,
                    lambda,
                    adaptive,
                    crossover,
                    history_probe: false,
                });
            }
        }
    }
    dense_fixtures(&mut cases);
    assert_eq!(cases.len(), FIXTURES);
    assert!(cases.len() <= MAX_FIXTURES);
    cases
}

fn dense_fixtures(cases: &mut Vec<Fixture>) {
    // Target clone-stripe boundaries rather than expanding a dense crossproduct.
    let modes = [(false, 0.0), (true, 0.2), (true, 1.0), (false, 0.2)];
    for counts in DENSE {
        for (lambda, next_lambda) in [(64, 1), (1, 33), (33, 64)] {
            for (adaptive, crossover) in modes {
                cases.push(Fixture {
                    counts,
                    lambda,
                    adaptive,
                    crossover,
                    history_probe: false,
                });
                // Preserve large/different tails, then change slot ownership and
                // shrink counts. A subsequent dense case makes tails active again.
                cases.push(Fixture {
                    counts: SMALL,
                    lambda: next_lambda,
                    adaptive,
                    crossover: 0.2,
                    history_probe: true,
                });
            }
        }
    }
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn new_buffers(device: &wgpu::Device, layouts: &[wgpu::BindGroupLayout; 2]) -> Buffers {
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let parents = buffer(
        device,
        "mutation checker parents 1024",
        PARENT_BYTES,
        storage,
    );
    let output = buffer(device, "mutation checker output 256", OUTPUT_BYTES, storage);
    let errors = buffer(
        device,
        "mutation checker error guards",
        ERROR_BYTES,
        storage,
    );
    let params = buffer(
        device,
        "mutation checker params",
        256,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let staging = buffer(
        device,
        "mutation checker staging",
        SNAPSHOT_BYTES,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let states_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mutation checker states"),
        layout: &layouts[0],
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: parents.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let params_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mutation checker params"),
        layout: &layouts[1],
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params.as_entire_binding(),
        }],
    });
    Buffers {
        parents,
        output,
        errors,
        params,
        staging,
        states_group,
        params_group,
    }
}

fn compile_variants(
    device: &wgpu::Device,
    layouts: &[wgpu::BindGroupLayout; 2],
    sources: Vec<Source>,
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<Vec<Variant>> {
    let mut variants = Vec::with_capacity(MAX_VARIANTS);
    let initial = initial_working_state();
    for source in sources {
        remaining(deadline)?;
        evidence.event(
            &json!({"event":"compilation_started", "variant":source.name,
            "sha256":source.sha256, "mutate_group_size":source.mutate_group_size,
            "zero_initialize_workgroup_memory":source.options.zero_initialize_workgroup_memory,
            "constants":benchmark_mutate_compilation_options(source.options).constants}),
        )?;
        let scopes = error_scopes(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mutation checker compatible layout"),
            bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1])],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&source.name),
            source: wgpu::ShaderSource::Wgsl(source.compiled.as_str().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&source.name),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: benchmark_mutate_compilation_options(source.options),
            cache: None,
        });
        finish_scopes(device, scopes, evidence, &source.name, deadline)?;
        evidence.event(
            &json!({"event":"compilation_completed", "variant":source.name,
            "sha256":source.sha256, "mutate_group_size":source.mutate_group_size,
            "zero_initialize_workgroup_memory":source.options.zero_initialize_workgroup_memory,
            "constants":benchmark_mutate_compilation_options(source.options).constants}),
        )?;
        variants.push(Variant {
            source,
            pipeline,
            prior: initial.clone(),
        });
    }
    assert!((2..=MAX_VARIANTS).contains(&variants.len()));
    Ok(variants)
}

fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn initial_working_state() -> Vec<u32> {
    let mut words = vec![0u32; SNAPSHOT_WORDS];
    for (slot, state) in words[..OUTPUT_WORDS]
        .as_chunks_mut::<STATE_WORDS>()
        .0
        .iter_mut()
        .enumerate()
    {
        let slot_seed = u32::try_from(slot).expect("offspring slot < 256");
        for (word, value) in state.iter_mut().enumerate() {
            let word = u32::try_from(word).expect("state word < STATE_WORDS");
            *value = mix(SEED ^ slot_seed.wrapping_mul(0x9e37_79b9) ^ word);
        }
        state[0] = 1000;
        state[1] = 12.5f32.to_bits();
        state[2] = 0.75f32.to_bits();
        state[4] = match slot % 64 {
            0 => 0,
            1 => 1,
            2 => u32::MAX,
            3 => 0x8000_0000,
            _ => mix(SEED ^ slot_seed),
        };
    }
    for (i, word) in words[OUTPUT_WORDS..].iter_mut().enumerate() {
        let i = u32::try_from(i).expect("error word < 512");
        *word = mix(SEED ^ 0xe770_0000 ^ i) | 1;
    }
    words
}

fn fill_parents(words: &mut [u32], counts: [u32; 4]) {
    assert_eq!(words.len(), PARENTS * STATE_WORDS);
    assert!(counts
        .iter()
        .all(|&c| c <= u32::try_from(POLYGONS).expect("polygon capacity is 1000")));
    for (slot, state) in words
        .as_chunks_mut::<STATE_WORDS>()
        .0
        .iter_mut()
        .enumerate()
    {
        // Parent indices and count fit u16, so conversion to f32 is exact.
        let parent_index = u16::try_from(slot).expect("parent slot < 1024");
        let parent_count = u16::try_from(PARENTS).expect("parent capacity is 1024");
        let key = SEED ^ u32::from(parent_index).wrapping_mul(0x9e37_79b9);
        state[0] = counts[slot % ACTIVE_CHAINS];
        state[1] = (1.0 + f32::from(parent_index) / f32::from(parent_count) * 95.0).to_bits();
        state[2] = [0.125f32, 0.75, 2.0, 4.0][slot % ACTIVE_CHAINS].to_bits();
        state[3] = mix(key);
        state[4..HEADER_WORDS].copy_from_slice(&[
            mix(key ^ 4),
            mix(key ^ 5),
            mix(key ^ 6),
            mix(key ^ 7),
        ]);
        // Fill inactive parent polygons too. All packed coordinates/colors are
        // valid, and no zero/identical tail can conceal a missed clone stripe.
        for (i, polygon) in state[HEADER_WORDS..]
            .as_chunks_mut::<POLYGON_WORDS>()
            .0
            .iter_mut()
            .enumerate()
        {
            let i = u32::try_from(i).expect("polygon slot < POLYGON_CAPACITY");
            let p = key ^ i.wrapping_mul(0x85eb_ca6b);
            polygon.copy_from_slice(&[mix(p), mix(p ^ 1), mix(p ^ 2), mix(p ^ 3)]);
        }
    }
}

fn fixture_params(case: Fixture, index: usize) -> GpuParams {
    let mut params = default_gpu_params(
        IMAGE[0],
        IMAGE[1],
        u32::try_from(ACTIVE_CHAINS).expect("active chain count is 4"),
    );
    // Deliberately bypass campaign normalization, which rounds lambda to a
    // power of two. These artificial values never enter an evolution campaign.
    params.lambda = case.lambda;
    params.adaptive_mutation = u32::from(case.adaptive);
    params.crossover_prob = case.crossover;
    params.min_polygons = 0;
    params.max_polygons = u32::try_from(POLYGONS).expect("polygon capacity is 1000");
    params.iteration_number = u32::try_from(index).expect("fixture index < 400");
    params.tournament_size = [1, 3, 7][index % 3];
    params
}

fn run_fixtures(
    p: &GpuPipeline,
    buffers: &Buffers,
    mut variants: Vec<Variant>,
    cases: &[Fixture],
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    let mut inputs = Inputs {
        parents: vec![0u32; PARENTS * STATE_WORDS],
        counts: None,
        sha256: String::new(),
    };
    let mut actual = vec![0u32; SNAPSHOT_WORDS];
    for (index, &case) in cases.iter().enumerate() {
        remaining(deadline)?;
        let scopes = error_scopes(&p.device);
        let prepared = prepare_inputs(p, buffers, &mut inputs, &variants, case, index, evidence);
        let scope_result = finish_scopes(&p.device, scopes, evidence, "fixture_inputs", deadline);
        prepared?;
        scope_result?;
        let produced = ACTIVE_CHAINS * case.lambda as usize;
        let failures_before = evidence.failed;
        for variant_index in 0..variants.len() {
            let variant = &variants[variant_index];
            dispatch(p, buffers, variant, &mut actual, (index, case), evidence, deadline)?;
            let reference = if variant_index == 0 {
                None
            } else {
                Some(variants[0].prior.as_slice())
            };
            let comparison = compare(reference, &actual, &variant.prior, produced);
            let passed = !comparison.failed();
            evidence.failed += usize::from(!passed);
            evidence.comparisons += usize::from(variant_index > 0);
            evidence.event(&json!({"event":"comparison", "fixture":index,
                "variant":variant.source.name, "sha256":variant.source.sha256,
                "mutate_group_size":variant.source.mutate_group_size,
                "zero_initialize_workgroup_memory":variant.source.options.zero_initialize_workgroup_memory,
                "reference":variant_index == 0, "passed":passed, "produced_slots":produced,
                "state_sha256":hash(bytemuck::cast_slice(&actual[..OUTPUT_WORDS])),
                "error_sha256":hash(bytemuck::cast_slice(&actual[OUTPUT_WORDS..])),
                "differences":comparison}))?;
            variants[variant_index].prior.copy_from_slice(&actual);
        }
        evidence.fixtures += 1;
        evidence.event(&json!({"event":"fixture_completed", "fixture":index,
            "passed":evidence.failed == failures_before}))?;
        ensure!(
            evidence.failed == failures_before,
            "semantic differences in fixture {index}; see events.jsonl"
        );
    }
    ensure!(
        evidence.fixtures == FIXTURES && evidence.completed == FIXTURES * variants.len(),
        "incomplete coverage"
    );
    Ok(())
}

fn prepare_inputs(
    p: &GpuPipeline,
    buffers: &Buffers,
    inputs: &mut Inputs,
    variants: &[Variant],
    case: Fixture,
    index: usize,
    evidence: &mut Evidence,
) -> Result<()> {
    ensure!(
        (1..=u32::try_from(MAX_LAMBDA).expect("lambda limit fits u32")).contains(&case.lambda),
        "invalid fixture lambda"
    );
    let rng_hash = matching_rng(variants)?;
    if inputs.counts != Some(case.counts) {
        fill_parents(&mut inputs.parents, case.counts);
        inputs.sha256 = hash(bytemuck::cast_slice(&inputs.parents));
        p.queue
            .write_buffer(&buffers.parents, 0, bytemuck::cast_slice(&inputs.parents));
        inputs.counts = Some(case.counts);
    }
    let params = fixture_params(case, index);
    p.queue
        .write_buffer(&buffers.params, 0, bytemuck::bytes_of(&params));
    evidence.event(
        &json!({"event":"fixture_started", "fixture":index, "case":case,
        "parent_sha256":inputs.sha256, "rng_input_sha256":rng_hash,
        "params_sha256":hash(bytemuck::bytes_of(&params)),
        "params_words":bytemuck::cast_slice::<GpuParams,u32>(std::slice::from_ref(&params))}),
    )
}

fn matching_rng(variants: &[Variant]) -> Result<String> {
    let mut digest = Sha256::new();
    for slot in 0..OFFSPRING {
        let start = slot * STATE_WORDS + 4;
        let reference = &variants[0].prior[start..start + 4];
        digest.update(bytemuck::cast_slice(reference));
        for variant in &variants[1..] {
            ensure!(
                &variant.prior[start..start + 4] == reference,
                "RNG inputs diverged before dispatch at slot {slot}, variant {}",
                variant.source.name
            );
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn error_scopes(device: &wgpu::Device) -> [wgpu::ErrorScopeGuard; 3] {
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
    phase: &str,
    deadline: Instant,
) -> Result<()> {
    let mut failed = false;
    let [validation, memory, internal] = scopes;
    // Pop every guard in LIFO order before a fallible wait. Early timeout/I/O
    // exits can then drop futures without dropping live guards out of order.
    for future in [internal.pop(), memory.pop(), validation.pop()] {
        let mut future = std::pin::pin!(future);
        let mut context = TaskContext::from_waker(Waker::noop());
        let until = Instant::now() + remaining(deadline)?.min(WAIT);
        let mut resolved = false;
        for _ in 0..SCOPE_POLLS {
            if let Poll::Ready(error) = future.as_mut().poll(&mut context) {
                if let Some(error) = error {
                    evidence.event(&json!({"event":"gpu_error", "phase":phase,
                        "message":short_error(&error.to_string())}))?;
                    failed = true;
                }
                resolved = true;
                break;
            }
            ensure!(
                Instant::now() < until,
                "GPU error scope timeout during {phase}"
            );
            device
                .poll(wgpu::PollType::Poll)
                .context("poll error scope")?;
            std::thread::sleep(Duration::from_millis(1));
        }
        ensure!(resolved, "GPU scope poll budget exhausted during {phase}");
    }
    ensure!(!failed, "GPU error during {phase}; see events.jsonl");
    Ok(())
}

fn dispatch(
    p: &GpuPipeline,
    buffers: &Buffers,
    variant: &Variant,
    actual: &mut [u32],
    fixture: (usize, Fixture),
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    ensure!(
        evidence.submitted < MAX_DISPATCHES,
        "dispatch budget exhausted"
    );
    remaining(deadline)?;
    let (fixture, case) = fixture;
    let dimensions = mutation_dispatch(case, variant.source.mutate_group_size);
    evidence.event(&json!({"event":"dispatch_started", "fixture":fixture,
        "variant":variant.source.name, "dispatch":dimensions,
        "mutate_group_size":variant.source.mutate_group_size,
        "zero_initialize_workgroup_memory":variant.source.options.zero_initialize_workgroup_memory}))?;
    let scopes = error_scopes(&p.device);
    p.queue.write_buffer(
        &buffers.output,
        0,
        bytemuck::cast_slice(&variant.prior[..OUTPUT_WORDS]),
    );
    // Mutation's actual ABI has no error binding. These separate poisoned
    // accumulators must remain untouched and equal across all source variants.
    p.queue.write_buffer(
        &buffers.errors,
        0,
        bytemuck::cast_slice(&variant.prior[OUTPUT_WORDS..]),
    );
    let encoder = encode_dispatch(p, buffers, &variant.pipeline, dimensions);
    let read = readback(p, buffers, encoder, actual, evidence, deadline);
    // Resolve scopes even when poll/map fails; mapped views never escape readback.
    let scope_result = finish_scopes(&p.device, scopes, evidence, &variant.source.name, deadline);
    read?;
    scope_result?;
    evidence.completed += 1;
    evidence.event(&json!({"event":"dispatch_completed", "fixture":fixture,
        "variant":variant.source.name, "completed_dispatches":evidence.completed,
        "mutate_group_size":variant.source.mutate_group_size,
        "zero_initialize_workgroup_memory":variant.source.options.zero_initialize_workgroup_memory,
        "readback_bytes":SNAPSHOT_BYTES, "gpu_scopes_clear":true}))
}

fn mutation_dispatch(fixture: Fixture, group_size: u32) -> [u32; 3] {
    [
        u32::try_from(ACTIVE_CHAINS).expect("active chain count is 4"),
        fixture.lambda.div_ceil(group_size),
        1,
    ]
}

fn encode_dispatch(
    p: &GpuPipeline,
    buffers: &Buffers,
    pipeline: &wgpu::ComputePipeline,
    dimensions: [u32; 3],
) -> wgpu::CommandEncoder {
    let mut encoder = p
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mutation checker one dispatch and snapshot"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mutation equivalence"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &buffers.states_group, &[]);
        pass.set_bind_group(1, &buffers.params_group, &[]);
        pass.dispatch_workgroups(dimensions[0], dimensions[1], dimensions[2]);
    }
    encoder.copy_buffer_to_buffer(&buffers.output, 0, &buffers.staging, 0, OUTPUT_BYTES);
    encoder.copy_buffer_to_buffer(
        &buffers.errors,
        0,
        &buffers.staging,
        OUTPUT_BYTES,
        ERROR_BYTES,
    );
    encoder
}

fn readback(
    p: &GpuPipeline,
    buffers: &Buffers,
    encoder: wgpu::CommandEncoder,
    actual: &mut [u32],
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    assert_eq!(actual.len(), SNAPSHOT_WORDS);
    let submission = p.queue.submit([encoder.finish()]);
    evidence.submitted += 1;
    evidence.event(&json!({"event":"dispatch_submitted", "submission":evidence.submitted}))?;
    let (tx, rx) = mpsc::sync_channel(1);
    buffers
        .staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            // Receiver can be gone after a timeout. Discarding this send failure is
            // safe: it communicates cancellation to nobody and owns no mapped view.
            let _cancelled = tx.send(result);
        });
    let result = map_snapshot(
        p,
        &buffers.staging,
        submission,
        &rx,
        actual,
        evidence,
        deadline,
    );
    // map_snapshot drops its view on every exit before unmapping/cancelling.
    buffers.staging.unmap();
    result
}

fn map_snapshot(
    p: &GpuPipeline,
    staging: &wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
    rx: &mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    actual: &mut [u32],
    evidence: &mut Evidence,
    deadline: Instant,
) -> Result<()> {
    let until = Instant::now() + remaining(deadline)?.min(WAIT);
    let status = p
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(until.saturating_duration_since(Instant::now())),
        })
        .context("mutation readback GPU poll")?;
    ensure!(status.wait_finished(), "GPU submission did not finish");
    evidence
        .event(&json!({"event":"submission_poll_finished", "submission":evidence.submitted}))?;
    rx.recv_timeout(until.saturating_duration_since(Instant::now()))
        .context("mutation map callback timeout")?
        .context("mutation staging map")?;
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .context("mutation mapped range")?;
    ensure!(
        mapped.len() == SNAPSHOT_WORDS * 4,
        "unexpected readback size"
    );
    actual.copy_from_slice(
        bytemuck::try_cast_slice::<u8, u32>(&mapped)
            .map_err(|error| anyhow::anyhow!("unaligned readback: {error}"))?,
    );
    drop(mapped);
    Ok(())
}

fn remaining(deadline: Instant) -> Result<Duration> {
    let duration = deadline.saturating_duration_since(Instant::now());
    ensure!(
        !duration.is_zero(),
        "internal 600-second deadline exhausted; retain partial events.jsonl"
    );
    Ok(duration)
}

impl Comparison {
    fn new() -> Self {
        Self {
            first: Vec::with_capacity(MAX_SAMPLES),
            ..Self::default()
        }
    }

    fn failed(&self) -> bool {
        self.invalid_counts
            + self.header_words
            + self.active_polygon_words
            + self.unproduced_slot_words_changed
            + self.error_words_changed
            + self.error_words_different
            != 0
    }

    fn sample(&mut self, kind: &'static str, slot: usize, word: usize, expected: u32, actual: u32) {
        if self.first.len() < MAX_SAMPLES {
            self.first.push(Difference {
                kind,
                slot,
                word,
                expected,
                actual,
            });
        }
    }
}

fn compare(
    reference: Option<&[u32]>,
    actual: &[u32],
    prior: &[u32],
    produced: usize,
) -> Comparison {
    assert_eq!(actual.len(), SNAPSHOT_WORDS);
    assert_eq!(prior.len(), SNAPSHOT_WORDS);
    assert!((ACTIVE_CHAINS..=OFFSPRING).contains(&produced));
    let mut result = Comparison::new();
    check_own_history(actual, prior, produced, &mut result);
    for slot in 0..produced {
        let start = slot * STATE_WORDS;
        if actual[start] > u32::try_from(POLYGONS).expect("polygon capacity is 1000") {
            result.invalid_counts += 1;
            result.sample(
                "invalid_count",
                slot,
                0,
                u32::try_from(POLYGONS).expect("polygon capacity is 1000"),
                actual[start],
            );
        }
        if let Some(reference) = reference {
            assert_eq!(reference.len(), SNAPSHOT_WORDS);
            compare_produced(
                &reference[start..start + STATE_WORDS],
                &actual[start..start + STATE_WORDS],
                slot,
                &mut result,
            );
        }
    }
    if let Some(reference) = reference {
        for word in 0..ERROR_WORDS {
            if actual[OUTPUT_WORDS + word] != reference[OUTPUT_WORDS + word] {
                result.error_words_different += 1;
                result.sample(
                    "error_difference",
                    word / 2,
                    word % 2,
                    reference[OUTPUT_WORDS + word],
                    actual[OUTPUT_WORDS + word],
                );
            }
        }
    }
    result
}

fn check_own_history(actual: &[u32], prior: &[u32], produced: usize, result: &mut Comparison) {
    for slot in produced..OFFSPRING {
        let start = slot * STATE_WORDS;
        for word in 0..STATE_WORDS {
            if actual[start + word] != prior[start + word] {
                result.unproduced_slot_words_changed += 1;
                result.sample(
                    "unproduced_slot_changed",
                    slot,
                    word,
                    prior[start + word],
                    actual[start + word],
                );
            }
        }
    }
    for word in 0..ERROR_WORDS {
        if actual[OUTPUT_WORDS + word] != prior[OUTPUT_WORDS + word] {
            result.error_words_changed += 1;
            result.sample(
                "error_accumulator_changed",
                word / 2,
                word % 2,
                prior[OUTPUT_WORDS + word],
                actual[OUTPUT_WORDS + word],
            );
        }
    }
}

fn compare_produced(reference: &[u32], actual: &[u32], slot: usize, result: &mut Comparison) {
    assert_eq!(reference.len(), STATE_WORDS);
    assert_eq!(actual.len(), STATE_WORDS);
    for word in 0..HEADER_WORDS {
        if actual[word] != reference[word] {
            result.header_words += 1;
            result.sample("header", slot, word, reference[word], actual[word]);
        }
    }
    // If counts differ the header already fails. Still check the union of both
    // active prefixes: no active tail can be reclassified as ignorable padding.
    let active = reference[0]
        .max(actual[0])
        .min(u32::try_from(POLYGONS).expect("polygon capacity is 1000"));
    let active = usize::try_from(active).expect("active polygon count <= 1000");
    let active_end = HEADER_WORDS + active * POLYGON_WORDS;
    for word in HEADER_WORDS..active_end {
        if actual[word] != reference[word] {
            result.active_polygon_words += 1;
            result.sample("active_polygon", slot, word, reference[word], actual[word]);
        }
    }
    for word in active_end..STATE_WORDS {
        if actual[word] != reference[word] {
            result.inactive_padding_words_different += 1;
            if result.first_padding.is_none() {
                result.first_padding = Some(Difference {
                    kind: "inactive_polygon_padding",
                    slot,
                    word,
                    expected: reference[word],
                    actual: actual[word],
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HISTORICAL_MUTATE: &str = include_str!(
        "../benchmarks/evolution/overnight-20261024/fixtures/mutate-e133-reference.wgsl"
    );
    const GROUPED_MUTATE_G1: &str = include_str!(
        "../benchmarks/evolution/overnight-20261024/fixtures/mutate-g1-wg64-copy64.wgsl"
    );
    // Source/evidence metadata only; this empty kernel does not test GPU equivalence.
    const GROUPED_MUTATE_G64: &str = "const OFFSPRING_GROUP_SIZE: u32 = 64u;\n\
        @compute @workgroup_size(64) fn main() {}";

    #[test]
    fn source_metadata_and_evidence_derive_group_for_each_exact_source() -> Result<()> {
        let directory = std::env::temp_dir().join(format!(
            "check-gpu-mutation-sources-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir(&directory)?;
        let result = (|| -> Result<()> {
            let bundled: &str = &artgen_backend_rust::gpu_evolver::bundled_shader_source(
                artgen_backend_rust::gpu_evolver::BundledShader::Mutate,
            );
            let bundled_group =
                benchmark_mutate_group_size_from_source(bundled).map_err(anyhow::Error::msg)?;
            let inputs = [
                (HISTORICAL_MUTATE, 64),
                (GROUPED_MUTATE_G1, 1),
                (GROUPED_MUTATE_G64, 64),
                (bundled, bundled_group),
            ];
            let mut options = Options {
                sources: Vec::new(),
                output: directory.join("evidence"),
                candidate_mutation: BenchmarkMutateOptions {
                    zero_initialize_workgroup_memory: false,
                },
            };
            for (index, (text, _)) in inputs.iter().enumerate() {
                let path = directory.join(format!("source-{index}.wgsl"));
                fs::write(&path, text)?;
                options.sources.push(SourceArg {
                    name: format!("source-{index}"),
                    path,
                });
            }
            let mut evidence = Evidence::new(&options.output)?;
            let sources = load_sources(&options, &mut evidence)?;
            drop(evidence);
            let events = fs::read_to_string(options.output.join("events.jsonl"))?;
            assert_eq!(sources.len(), inputs.len());
            assert_eq!(events.lines().count(), inputs.len());
            for ((source, (text, group)), line) in sources.iter().zip(inputs).zip(events.lines()) {
                assert_source_metadata_evidence(source, text, group, line, &options.output)?;
            }
            let case = Fixture {
                lambda: 33,
                ..fixtures()[0]
            };
            assert_eq!(
                mutation_dispatch(case, sources[1].mutate_group_size),
                [4, 33, 1]
            );
            assert_eq!(
                mutation_dispatch(case, sources[2].mutate_group_size),
                [4, 1, 1]
            );
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
    }

    fn assert_source_metadata_evidence(
        source: &Source,
        text: &str,
        group: u32,
        line: &str,
        output: &Path,
    ) -> Result<()> {
        assert_eq!(source.compiled, specialize_polygon_storage(text).unwrap().source);
        assert_eq!(source.sha256, hash(text.as_bytes()));
        assert_eq!(source.mutate_group_size, group);
        assert_eq!(
            source.options.zero_initialize_workgroup_memory,
            source.name == "source-0"
        );
        let event: Value = serde_json::from_str(line)?;
        assert_eq!(event["event"], "source");
        assert_eq!(event["name"], source.name);
        assert_eq!(event["sha256"], source.sha256);
        assert_eq!(event["mutate_group_size"], group);
        assert_eq!(
            event["zero_initialize_workgroup_memory"],
            source.options.zero_initialize_workgroup_memory
        );
        assert_eq!(event["constants"], json!([]));
        assert_eq!(event["bytes"], text.len());
        assert_eq!(
            fs::read_to_string(output.join(format!("{}.wgsl", source.name)))?,
            text
        );
        Ok(())
    }

    #[test]
    fn all_lambda_tails_cover_each_absolute_offspring_once() {
        let cases = fixtures();
        for group in [1, 2, 4, 8, 16, 32, 64] {
            let source = format!("const OFFSPRING_GROUP_SIZE: u32 = {group}u;");
            let checked = benchmark_mutate_group_size_from_source(&source).unwrap();
            for lambda in 1..=64 {
                let case = Fixture { lambda, ..cases[0] };
                let [active, groups, z] = mutation_dispatch(case, checked);
                assert_eq!(active, 4);
                assert_eq!(z, 1);
                assert!((1..=64).contains(&groups));
                let mut visits = [0u8; OFFSPRING];
                for chain in 0..active {
                    for y in 0..groups {
                        for leader in 0..checked {
                            let local = y.checked_mul(checked).unwrap()
                                .checked_add(leader).unwrap();
                            if local < lambda {
                                let id = chain.checked_mul(lambda).unwrap()
                                    .checked_add(local).unwrap();
                                visits[id as usize] += 1;
                            }
                        }
                    }
                }
                let produced = ACTIVE_CHAINS * lambda as usize;
                assert!(visits[..produced].iter().all(|&visits| visits == 1));
                assert!(visits[produced..].iter().all(|&visits| visits == 0));
            }
        }
    }

    #[test]
    fn fixture_matrix_keeps_all_lambda_modes_historical_dispatch_and_parent_bounds() {
        let cases = fixtures();
        assert_eq!(cases.len(), FIXTURES);
        let baseline = benchmark_mutate_group_size_from_source(HISTORICAL_MUTATE).unwrap();
        assert_eq!(baseline, 64);
        for lambda in 1..=64 {
            let modes: Vec<_> = cases
                .iter()
                .filter(|case| {
                    case.lambda == lambda
                        && case.counts == SMALL
                        && case.crossover == 0.2
                        && !case.history_probe
                })
                .map(|case| case.adaptive)
                .collect();
            assert_eq!(modes, [false, true]);
            assert_eq!(
                mutation_dispatch(Fixture { lambda, ..cases[0] }, baseline),
                [4, 1, 1]
            );
            let last = (u32::try_from(PARENTS).unwrap() - 1)
                .checked_mul(lambda)
                .unwrap()
                .checked_add(lambda - 1)
                .unwrap();
            assert!(last < 65_536);
        }
    }

    #[test]
    fn help_and_argument_bounds_need_no_gpu() {
        assert!(parse_args([OsString::from("--help")].into_iter())
            .unwrap()
            .is_none());
        let args = [
            "--reference",
            "baseline.wgsl",
            "--candidate",
            "copy2=candidate.wgsl",
            "--output",
            "new-results",
        ];
        let parsed = parse_args(args.map(OsString::from).into_iter())
            .unwrap()
            .unwrap();
        assert_eq!(parsed.sources.len(), 2);
        assert_eq!(parsed.sources[1].name, "copy2");
        assert_eq!(parsed.output, Path::new("new-results"));
        assert!(parse_args([OsString::from("--unknown")].into_iter()).is_err());
        let mut too_many = args.map(OsString::from).to_vec();
        too_many.resize(MAX_ARGS + 1, OsString::from("extra"));
        assert!(parse_args(too_many.into_iter()).is_err());
        let mut invalid = args.map(OsString::from);
        invalid[3] = OsString::from("../escape=candidate.wgsl");
        assert!(parse_args(invalid.into_iter()).is_err());
        let mut invalid = args.map(OsString::from);
        invalid[5] = OsString::from("x".repeat(MAX_PATH_BYTES + 1));
        assert!(parse_args(invalid.into_iter()).is_err());
    }

    #[test]
    fn candidate_zeroing_cli_defaults_true_and_rejects_invalid_or_duplicate() {
        let base = [
            "--reference",
            "reference.wgsl",
            "--candidate",
            "candidate=candidate.wgsl",
            "--output",
            "new-results",
        ]
        .map(OsString::from)
        .to_vec();
        assert!(
            parse_args(base.clone().into_iter())
                .unwrap()
                .unwrap()
                .candidate_mutation
                .zero_initialize_workgroup_memory
        );
        for value in ["true", "false"] {
            let mut input = base.clone();
            input.extend([
                OsString::from("--mutate-zero-initialize"),
                OsString::from(value),
            ]);
            let options = parse_args(input.clone().into_iter()).unwrap().unwrap();
            assert_eq!(
                options.candidate_mutation.zero_initialize_workgroup_memory,
                value == "true"
            );
            input.extend([
                OsString::from("--mutate-zero-initialize"),
                OsString::from(value),
            ]);
            assert!(parse_args(input.into_iter()).is_err());
        }
        for value in ["TRUE", "0", "invalid", ""] {
            let mut input = base.clone();
            input.extend([
                OsString::from("--mutate-zero-initialize"),
                OsString::from(value),
            ]);
            assert!(parse_args(input.into_iter()).is_err());
        }
    }

    #[test]
    fn inactive_padding_can_differ_but_becoming_active_must_fail() {
        let mut reference = initial_working_state();
        for slot in 0..ACTIVE_CHAINS {
            reference[slot * STATE_WORDS] = 1;
        }
        let mut candidate = reference.clone();
        let tail_word = HEADER_WORDS + 17 * POLYGON_WORDS;
        candidate[tail_word] ^= 1;
        let padding = compare(Some(&reference), &candidate, &candidate, ACTIVE_CHAINS);
        assert!(!padding.failed());
        assert_eq!(padding.inactive_padding_words_different, 1);
        reference[0] = 18;
        candidate[0] = 18;
        let active = compare(Some(&reference), &candidate, &candidate, ACTIVE_CHAINS);
        assert!(active.failed());
        assert_eq!(active.active_polygon_words, 1);
        candidate[0] = 1;
        let wrong_count = compare(Some(&reference), &candidate, &candidate, ACTIVE_CHAINS);
        assert!(wrong_count.failed());
        assert_eq!(wrong_count.active_polygon_words, 1);
    }

    #[test]
    fn unproduced_slots_use_their_own_full_history_and_all_errors_are_checked() {
        let reference = initial_working_state();
        let mut prior = reference.clone();
        let untouched = ACTIVE_CHAINS * STATE_WORDS + HEADER_WORDS + 999 * POLYGON_WORDS;
        prior[untouched] ^= 1;
        let clean = compare(Some(&reference), &prior, &prior, ACTIVE_CHAINS);
        assert!(!clean.failed());
        let mut actual = prior.clone();
        actual[untouched] ^= 2;
        actual[OUTPUT_WORDS + ERROR_WORDS - 1] ^= 1;
        let broken = compare(Some(&reference), &actual, &prior, ACTIVE_CHAINS);
        assert!(broken.failed());
        assert_eq!(broken.unproduced_slot_words_changed, 1);
        assert_eq!(broken.error_words_changed, 1);
        assert_eq!(broken.error_words_different, 1);
    }

    #[test]
    fn every_header_word_is_semantic_including_unused_rng_lanes() {
        let reference = initial_working_state();
        let mut actual = reference.clone();
        for word in 1..HEADER_WORDS {
            actual[word] ^= 1;
        }
        let result = compare(Some(&reference), &actual, &actual, ACTIVE_CHAINS);
        assert!(result.failed());
        assert_eq!(result.header_words, HEADER_WORDS - 1);
    }
}
