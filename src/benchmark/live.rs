//! Fixed-work acceptance through the backend's production drawing and WS paths.
//!
//! Runtime allocation is required by PNG, JSON and tungstenite. Counts, payload
//! sizes and aggregate retained bytes are bounded here; both sockets and threads
//! are owned by Connection. GPU waits remain subject to the supervisor's timeout.
use std::{
    collections::VecDeque,
    fs,
    io::{BufWriter, Write},
    net::{Shutdown, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::{
    benchmark::{
        fixtures::{self, LoadedFixture},
        trial::{self, TimestampMode, TrialRecord, TrialSpec},
    },
    engine::Rasterizer,
    gpu_evolver::{
        benchmark::{BenchmarkCounters, BenchmarkSnapshot},
        buffers::{drawing_to_gpu, gpu_to_drawing},
        EngineCapacity, GpuEvolver,
    },
    models::drawing::Drawing,
    mutation_params::MutationParams,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[path = "live_latency.rs"]
mod latency;
use latency::PublicationTrace;
pub(crate) use latency::SenderTurn;

const MAX_BATCHES: u32 = 1024;
const MAX_PUBLICATIONS: usize = 512;
const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;
const MAX_RETAINED_BYTES: usize = 128 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const MAX_TURNS: usize = 8192;
const RUN_TIMEOUT: Duration = Duration::from_secs(120);
const IO_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Debug, Serialize)]
pub struct Spec {
    pub case_id: String,
    pub seed: u64,
    pub batches: u32,
    pub warmup_batches: u32,
    pub publish_every: u32,
    pub deadline_ms: u32,
    #[serde(skip)]
    latency_trace: bool,
    #[serde(skip)]
    manifest: PathBuf,
    #[serde(skip)]
    output: PathBuf,
}

impl Spec {
    pub fn parse(args: &[String]) -> Result<Self> {
        ensure!(args.len() <= 20, "too many live acceptance arguments");
        ensure!(
            args.iter().all(|arg| arg.len() <= 4096),
            "live option exceeds length limit"
        );
        let mut spec = Self {
            case_id: "mona".into(),
            seed: 13,
            batches: 256,
            warmup_batches: 2,
            publish_every: 2,
            deadline_ms: 300,
            latency_trace: false,
            manifest: PathBuf::new(),
            output: PathBuf::new(),
        };
        let mut seen = Vec::with_capacity(8);
        for pair in args.chunks(2) {
            ensure!(pair.len() == 2, "live options require a value");
            ensure!(
                !seen.contains(&pair[0].as_str()),
                "duplicate option {}",
                pair[0]
            );
            seen.push(pair[0].as_str());
            match pair[0].as_str() {
                "--manifest" => spec.manifest = PathBuf::from(&pair[1]),
                "--output" => spec.output = PathBuf::from(&pair[1]),
                "--case" => spec.case_id.clone_from(&pair[1]),
                "--seed" => spec.seed = pair[1].parse()?,
                "--batches" => spec.batches = pair[1].parse()?,
                "--warmup" => spec.warmup_batches = pair[1].parse()?,
                "--publish-every" => spec.publish_every = pair[1].parse()?,
                "--deadline-ms" => spec.deadline_ms = pair[1].parse()?,
                "--latency-trace" => {
                    ensure!(pair[1] == "v1", "latency trace version must be v1");
                    spec.latency_trace = true;
                }
                flag => bail!("unknown live option {flag}"),
            }
        }
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.manifest.as_os_str().is_empty(),
            "--manifest is required"
        );
        ensure!(!self.output.as_os_str().is_empty(), "--output is required");
        ensure!(
            fixtures::PROJECTS.contains(&self.case_id.as_str()),
            "unknown case"
        );
        ensure!(
            (2..=MAX_BATCHES).contains(&self.batches),
            "batches must be 2..1024"
        );
        ensure!(self.warmup_batches <= 16, "warmup must be 0..16");
        ensure!(
            self.publish_every >= 2 && self.batches.is_multiple_of(self.publish_every),
            "publication cadence must be >=2 and divide batches"
        );
        ensure!(
            self.publication_count() <= MAX_PUBLICATIONS,
            "too many publications"
        );
        ensure!(
            (1..=10000).contains(&self.deadline_ms),
            "deadline must be 1..10000ms"
        );
        Ok(())
    }

    fn publication_count(&self) -> usize {
        (self.batches / self.publish_every) as usize
    }

    fn trial(&self) -> TrialSpec {
        TrialSpec {
            case_id: self.case_id.clone(),
            seed: self.seed,
            batches: self.batches,
            warmup_batches: self.warmup_batches,
            diagnostic: false,
            timestamp_mode: TimestampMode::Stages,
        }
    }
}

#[derive(Clone, Serialize)]
pub struct Publication {
    pub run_id: String,
    pub sequence: usize,
    pub completed_batches: u64,
    pub completed_evaluations: u64,
    pub due_ms: f64,
    pub send_ms: Option<f64>,
    pub receive_ms: Option<f64>,
    pub drawing_sha256: String,
    pub png_sha256: String,
    pub fitness_percent: f32,
    pub polygons: usize,
    pub payload_valid: bool,
    pub acquisition: Acquisition,
}

#[derive(Clone, Serialize)]
pub struct Acquisition {
    completed_batches: u64,
    fitness_bits: u32,
    maximum_fitness_bits: u32,
    is_acquisition_maximum: bool,
}

struct Latest {
    drawing: Drawing,
    acquisition: Acquisition,
}

impl Latest {
    fn acquired(drawing: Drawing, evolver: &GpuEvolver) -> Result<Self> {
        ensure!(
            drawing.fitness.is_finite(),
            "non-finite ordinary drawing score"
        );
        let scores = evolver.chain_fitness();
        ensure!(
            scores.len() >= 4 && scores[..4].iter().all(|score| score.is_finite()),
            "incomplete or non-finite acquired chain scores"
        );
        let maximum = scores[..4]
            .iter()
            .copied()
            .max_by(f32::total_cmp)
            .context("missing acquired chain scores")?;
        let fitness_bits = drawing.fitness.to_bits();
        ensure!(
            scores[..4]
                .iter()
                .any(|score| score.to_bits() == fitness_bits),
            "ordinary drawing score absent from its acquisition batch"
        );
        let acquisition = Acquisition {
            completed_batches: evolver.benchmark_counters().completed_batches,
            fitness_bits,
            maximum_fitness_bits: maximum.to_bits(),
            is_acquisition_maximum: fitness_bits == maximum.to_bits(),
        };
        Ok(Self {
            drawing,
            acquisition,
        })
    }

    fn observe(&mut self, drawing: Option<Drawing>, evolver: &GpuEvolver) -> Result<()> {
        if let Some(drawing) = drawing {
            let acquired = Self::acquired(drawing, evolver)?;
            // Preserve the ordinary application's improvement policy and its
            // possible best-ID race. The endpoint check must detect any failure.
            if acquired.drawing.fitness > self.drawing.fitness {
                *self = acquired;
            }
        }
        Ok(())
    }
}

struct Payload {
    drawing_json: String,
    png: Vec<u8>,
    fitness: f32,
    polygons: usize,
}

struct Ledger {
    trace: Option<latency::Trace>,
    start: Option<Instant>,
    records: Vec<Publication>,
    queue: VecDeque<(usize, Payload)>,
    queue_bytes: usize,
    received: Vec<Option<String>>,
    received_bytes: usize,
    errors: Vec<String>,
    received_count: usize,
}

impl Ledger {
    fn publication_slot(
        &self,
        spec: &Spec,
        state: &crate::WsState,
        counters: BenchmarkCounters,
        due_at: Instant,
    ) -> Result<(usize, usize, f64)> {
        let sequence = self.records.len() + 1;
        ensure!(
            sequence <= spec.publication_count(),
            "publication schedule exhausted"
        );
        let expected = sequence as u64 * u64::from(spec.publish_every);
        ensure!(
            counters.completed_batches == expected,
            "publication completed-work origin skipped"
        );
        let bytes = state.drawing_json.len() + state.best_png.len();
        ensure!(
            bytes <= MAX_PAYLOAD_BYTES,
            "publication payload exceeds byte limit"
        );
        ensure!(
            self.queue_bytes + bytes <= MAX_RETAINED_BYTES,
            "publication queue byte limit"
        );
        let due_ms = due_at
            .duration_since(self.start.context("live clock is not started")?)
            .as_secs_f64()
            * 1000.0;
        Ok((sequence, bytes, due_ms))
    }
}

pub struct Session {
    spec: Spec,
    run_id: String,
    wall_start: Instant,
    stopped: AtomicBool,
    ledger: Mutex<Ledger>,
    changed: Condvar,
}

fn is_transport_close(error: &anyhow::Error) -> bool {
    use std::io::ErrorKind;
    use tungstenite::{error::ProtocolError, Error};
    let io_error = match error.downcast_ref::<Error>() {
        Some(
            Error::ConnectionClosed | Error::Protocol(ProtocolError::ResetWithoutClosingHandshake),
        ) => {
            return true;
        }
        Some(Error::Io(error)) => Some(error),
        _ => error.downcast_ref::<std::io::Error>(),
    };
    io_error.is_some_and(|error| {
        matches!(
            error.kind(),
            ErrorKind::BrokenPipe
                | ErrorKind::ConnectionReset
                | ErrorKind::ConnectionAborted
                | ErrorKind::NotConnected
                | ErrorKind::UnexpectedEof
        )
    })
}

impl Session {
    fn new(spec: &Spec) -> Self {
        let count = spec.publication_count();
        Self {
            spec: spec.clone(),
            run_id: uuid::Uuid::new_v4().to_string(),
            wall_start: Instant::now(),
            stopped: AtomicBool::new(false),
            ledger: Mutex::new(Ledger {
                trace: spec.latency_trace.then(|| latency::Trace::new(count)),
                start: None,
                records: Vec::with_capacity(count),
                queue: VecDeque::with_capacity(count),
                queue_bytes: 0,
                received: (0..count).map(|_| None).collect(),
                received_bytes: 0,
                errors: Vec::with_capacity(count + 16),
                received_count: 0,
            }),
            changed: Condvar::new(),
        }
    }

    fn start(&self) -> Result<()> {
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        ensure!(log.start.is_none(), "live clock already started");
        log.start = Some(Instant::now());
        Ok(())
    }

    pub fn expired(&self) -> bool {
        self.stopped.load(Ordering::Acquire) || self.wall_start.elapsed() >= RUN_TIMEOUT
    }

    pub fn fail(&self, error: impl Into<String>) {
        if let Ok(mut log) = self.ledger.lock() {
            // One receiver, one sender and at most 512 validation failures.
            if log.errors.len() < MAX_PUBLICATIONS * 4 + 32 {
                log.errors.push(error.into());
            }
        }
        self.stopped.store(true, Ordering::Release);
        self.changed.notify_all();
    }

    pub fn fail_connection(&self, error: impl Into<anyhow::Error>) {
        let error = error.into();
        // Owned shutdown can interrupt a turn already preparing its stats send.
        if self.stopped.load(Ordering::Acquire) && is_transport_close(&error) {
            return;
        }
        self.fail(format!("{error:#}"));
    }

    fn publish(
        &self,
        state: &crate::WsState,
        counters: BenchmarkCounters,
        acquisition: &Acquisition,
        due_at: Instant,
        mut trace: Option<&mut PublicationTrace>,
    ) -> Result<()> {
        if let Some(row) = trace.as_deref_mut() {
            row.ledger_lock_attempt_ns = self.trace_clock().map(latency::Clock::now);
        }
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        if let Some(row) = trace.as_deref_mut() {
            row.ledger_lock_acquired_ns = self.trace_clock().map(latency::Clock::now);
        }
        let (sequence, bytes, due_ms) =
            log.publication_slot(&self.spec, state, counters, due_at)?;
        let record = Publication {
            run_id: self.run_id.clone(),
            sequence,
            completed_batches: counters.completed_batches,
            completed_evaluations: counters.completed_evaluations,
            due_ms,
            send_ms: None,
            receive_ms: None,
            drawing_sha256: fixtures::sha256(state.drawing_json.as_bytes()),
            png_sha256: fixtures::sha256(&state.best_png),
            fitness_percent: state.fitness,
            polygons: state.polygons,
            payload_valid: false,
            acquisition: acquisition.clone(),
        };
        if let Some(row) = trace.as_deref_mut() {
            row.hashes_done_ns = self.trace_clock().map(latency::Clock::now);
        }
        log.records.push(record);
        log.queue.push_back((
            sequence,
            Payload {
                drawing_json: state.drawing_json.clone(),
                png: state.best_png.clone(),
                fitness: state.fitness,
                polygons: state.polygons,
            },
        ));
        log.queue_bytes += bytes;
        if let Some(row) = trace {
            row.payload_bytes = Some(bytes);
            row.queue_depth = Some(log.queue.len());
            row.enqueue_commit_ns = self.trace_clock().map(latency::Clock::now);
            log.trace
                .as_mut()
                .context("missing enabled trace")?
                .retain(*row)?;
        }
        Ok(())
    }

    pub fn message(
        &self,
        state: &mut crate::WsState,
        last_image: &mut Instant,
        mut trace: Option<&mut SenderTurn>,
    ) -> Result<Option<(String, (usize, f64))>> {
        let image_interval = Duration::from_millis(200);
        let gated = if let Some(row) = trace.as_deref_mut() {
            row.image_is_throttled(*last_image, image_interval)
        } else {
            last_image.elapsed() < image_interval
        };
        if gated {
            return Ok(None);
        }
        if let Some(row) = trace.as_deref_mut() {
            row.ledger_lock_attempt_ns = Some(row.now());
        }
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        if let Some(row) = trace.as_deref_mut() {
            row.ledger_lock_acquired_ns = Some(row.now());
            row.queue_depth = Some(log.queue.len());
            row.queue_head_sequence = log.queue.front().map(|(sequence, _)| *sequence);
        }
        let Some((sequence, payload)) = log.queue.pop_front() else {
            return Ok(None);
        };
        if let Some(row) = trace.as_deref_mut() {
            row.dequeue_ns = Some(row.now());
            row.publication_sequence = Some(sequence);
        }
        log.queue_bytes -= payload.drawing_json.len() + payload.png.len();
        state.drawing_json = payload.drawing_json;
        state.best_png = payload.png;
        state.fitness = payload.fitness;
        state.polygons = payload.polygons;
        let start = log.start.context("live clock is not started")?;
        Ok(Some(self.publication_message(
            state,
            last_image,
            &log.records[sequence - 1],
            start,
            trace,
        )))
    }

    fn publication_message(
        &self,
        state: &mut crate::WsState,
        last_image: &mut Instant,
        record: &Publication,
        start: Instant,
        mut trace: Option<&mut SenderTurn>,
    ) -> (String, (usize, f64)) {
        let sequence = record.sequence;
        let send_at = Instant::now();
        let send_ms = send_at.duration_since(start).as_secs_f64() * 1000.0;
        if let Some(row) = trace.as_deref_mut() {
            row.legacy_sendmarker_ns = Some(row.at(send_at));
        }
        state.total_evals = record.completed_evaluations;
        state.elapsed_secs = send_ms / 1000.0;
        let mut message = crate::drawing_update_message(state);
        message["liveAcceptance"] = serde_json::json!({ "run_id": self.run_id,
            "sequence": sequence, "completed_batches": record.completed_batches,
            "completed_evaluations": record.completed_evaluations, "due_ms": record.due_ms,
            "send_ms": send_ms, "drawing_sha256": record.drawing_sha256,
            "png_sha256": record.png_sha256 });
        if let Some(row) = trace.as_deref_mut() {
            row.message_value_done_ns = Some(row.now());
        }
        *last_image = Instant::now();
        if let Some(row) = trace.as_deref_mut() {
            row.image_cadence_reset_ns = Some(row.at(*last_image));
        }
        let message = message.to_string();
        if let Some(row) = trace {
            row.wire_json_done_ns = Some(row.now());
        }
        (message, (sequence, send_ms))
    }

    pub fn sent(&self, (sequence, send_ms): (usize, f64)) -> Result<()> {
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        let record = log
            .records
            .get_mut(sequence - 1)
            .context("unknown sent publication")?;
        ensure!(record.send_ms.is_none(), "duplicate sent publication");
        record.send_ms = Some(send_ms);
        Ok(())
    }

    fn receive(&self, text: String, received_at: Instant) -> Result<()> {
        ensure!(
            text.len() <= MAX_PAYLOAD_BYTES * 2,
            "received payload exceeds limit"
        );
        let value: serde_json::Value = serde_json::from_str(&text)?;
        let Some(meta) = value.get("liveAcceptance") else {
            ensure!(
                value["type"] == "stats",
                "expected ordinary stats or a tagged publication update"
            );
            return Ok(());
        };
        ensure!(
            value["type"] == "update",
            "expected tagged publication update"
        );
        ensure!(
            meta["run_id"].as_str() == Some(self.run_id.as_str()),
            "wrong publication run"
        );
        let sequence = usize::try_from(
            meta["sequence"]
                .as_u64()
                .context("missing publication sequence")?,
        )
        .context("publication sequence exceeds addressable range")?;
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        ensure!(
            sequence == log.received_count + 1,
            "missing or duplicate publication identity"
        );
        ensure!(
            log.received_bytes + text.len() <= MAX_RETAINED_BYTES,
            "receipt byte limit"
        );
        let start = log.start.context("receipt before live clock")?;
        let record = log
            .records
            .get_mut(sequence - 1)
            .context("unscheduled publication")?;
        ensure!(record.receive_ms.is_none(), "duplicate publication receipt");
        ensure!(
            meta["completed_batches"].as_u64() == Some(record.completed_batches)
                && meta["completed_evaluations"].as_u64() == Some(record.completed_evaluations),
            "wrong completed-work origin"
        );
        record.receive_ms = Some(received_at.duration_since(start).as_secs_f64() * 1000.0);
        if let Some(trace) = &mut log.trace {
            let row = trace
                .publications
                .get_mut(sequence - 1)
                .context("missing trace receipt identity")?;
            row.receive_ns = self.trace_clock().map(|clock| clock.at(received_at));
        }
        log.received_bytes += text.len();
        log.received[sequence - 1] = Some(text);
        log.received_count += 1;
        self.changed.notify_all();
        Ok(())
    }

    fn wait_for_receipts(&self) -> Result<f64> {
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        for _ in 0..MAX_TURNS {
            if log.received_count == self.spec.publication_count() {
                return Ok(log
                    .start
                    .context("live clock is not started")?
                    .elapsed()
                    .as_secs_f64());
            }
            ensure!(
                !self.expired(),
                "live receipt timeout or connection failure"
            );
            let remaining = RUN_TIMEOUT.saturating_sub(self.wall_start.elapsed());
            let (next, _) = self
                .changed
                .wait_timeout(log, remaining)
                .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
            log = next;
        }
        bail!("live receipt wake limit exhausted")
    }

    fn elapsed_seconds(&self) -> Result<f64> {
        let log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        Ok(log
            .start
            .context("live clock is not started")?
            .elapsed()
            .as_secs_f64())
    }

    fn validate_payloads(&self, width: u32, height: u32) -> Result<()> {
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        for index in 0..log.records.len() {
            let result = match &log.received[index] {
                Some(text) => validate_payload(text, &log.records[index], width, height),
                None => Err(anyhow::anyhow!("missing publication payload")),
            };
            match result {
                Ok(()) => log.records[index].payload_valid = true,
                Err(error) => log
                    .errors
                    .push(format!("publication {}: {error:#}", index + 1)),
            }
        }
        Ok(())
    }

    fn save_receipts(&self, output: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
        let log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        let mut files = serde_json::Map::new();
        let mut writer: Option<BufWriter<fs::File>> = None;
        let mut name = String::new();
        let mut bytes = 0;
        let mut digest = Sha256::new();
        for text in log.received.iter().flatten() {
            if bytes + text.len() + 1 > MAX_ARTIFACT_BYTES {
                let file = writer.take().context("missing receipt shard")?;
                files.insert(name.clone(), close_receipt_shard(file, &digest, bytes)?);
                bytes = 0;
                digest = Sha256::new();
            }
            if writer.is_none() {
                ensure!(files.len() < 3, "receipt artifact count limit");
                name = format!("receipt-{:03}.jsonl", files.len());
                writer = Some(BufWriter::new(
                    fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(output.join(&name))?,
                ));
            }
            let file = writer.as_mut().context("missing receipt writer")?;
            file.write_all(text.as_bytes())?;
            file.write_all(b"\n")?;
            digest.update(text.as_bytes());
            digest.update(b"\n");
            bytes += text.len() + 1;
        }
        if let Some(file) = writer {
            files.insert(name, close_receipt_shard(file, &digest, bytes)?);
        }
        Ok(files)
    }
}

fn close_receipt_shard(
    mut writer: BufWriter<fs::File>,
    digest: &Sha256,
    bytes: usize,
) -> Result<serde_json::Value> {
    writer.flush()?;
    Ok(serde_json::json!({ "sha256": format!("{:x}", digest.clone().finalize()), "bytes": bytes }))
}

fn validate_payload(text: &str, expected: &Publication, width: u32, height: u32) -> Result<()> {
    let value: serde_json::Value = serde_json::from_str(text)?;
    let meta = &value["liveAcceptance"];
    ensure!(
        value["type"] == "update",
        "wrong production WS message type"
    );
    ensure!(
        meta["run_id"].as_str() == Some(expected.run_id.as_str())
            && meta["sequence"].as_u64() == Some(expected.sequence as u64),
        "wrong publication identity"
    );
    ensure!(
        meta["due_ms"].as_f64() == Some(expected.due_ms)
            && meta["send_ms"].as_f64() == expected.send_ms,
        "wrong publication timestamps"
    );
    let json = value["drawingJson"]
        .as_str()
        .context("missing production drawing JSON")?;
    ensure!(
        fixtures::sha256(json.as_bytes()) == expected.drawing_sha256
            && meta["drawing_sha256"].as_str() == Some(expected.drawing_sha256.as_str()),
        "drawing identity mismatch"
    );
    let drawing: Drawing = serde_json::from_str(json)?;
    ensure!(
        drawing.fitness.is_finite()
            && drawing.fitness.to_bits() == expected.acquisition.fitness_bits,
        "drawing score changed from original mapped header"
    );
    let wire_score: f32 = serde_json::from_value(value["fitness"].clone())?;
    ensure!(
        wire_score.to_bits() == drawing.fitness.to_bits()
            && wire_score.to_bits() == expected.fitness_percent.to_bits(),
        "wire score/drawing mismatch"
    );
    ensure!(
        value["polygons"].as_u64() == Some(drawing.polygons.len() as u64)
            && drawing.polygons.len() == expected.polygons,
        "wire geometry count mismatch"
    );
    let encoded = value["image"].as_str().context("missing production PNG")?;
    let png = BASE64.decode(encoded)?;
    ensure!(
        fixtures::sha256(&png) == expected.png_sha256
            && meta["png_sha256"].as_str() == Some(expected.png_sha256.as_str()),
        "PNG identity mismatch"
    );
    validate_png(&png, &drawing, width, height)
}

fn validate_png(png: &[u8], drawing: &Drawing, width: u32, height: u32) -> Result<()> {
    ensure!(
        drawing.polygons.len() <= 1000,
        "published geometry exceeds limit"
    );
    for polygon in &drawing.polygons {
        ensure!(
            polygon.points.len() == 3,
            "published geometry is not packed triangles"
        );
        ensure!(
            polygon.points.iter().all(|point| point.x.is_finite()
                && point.y.is_finite()
                && (0.0..=1.0).contains(&point.x)
                && (0.0..=1.0).contains(&point.y)),
            "invalid published coordinates"
        );
    }
    let reader = image::ImageReader::new(std::io::Cursor::new(png)).with_guessed_format()?;
    ensure!(
        reader.into_dimensions()? == (width, height),
        "PNG dimensions differ from fixture"
    );
    let image = image::load_from_memory(png)?.into_rgba8();
    let mut expected = vec![0; width as usize * height as usize * 4];
    drawing.draw(
        &mut expected,
        width as usize,
        height as usize,
        Rasterizer::HalfSpace,
    );
    ensure!(
        image.as_raw() == &expected,
        "PNG does not match production CPU drawing renderer"
    );
    Ok(())
}

struct Connection {
    session: Arc<Session>,
    sockets: Vec<TcpStream>,
    threads: Vec<JoinHandle<()>>,
}

impl Connection {
    fn start(state: crate::SharedWsState, session: Arc<Session>) -> Result<Self> {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
        let address = listener.local_addr()?;
        let client = TcpStream::connect_timeout(&address, IO_TIMEOUT)?;
        listener.set_nonblocking(true)?;
        let (server, _) = listener.accept()?;
        drop(listener);
        for stream in [&server, &client] {
            stream.set_read_timeout(Some(IO_TIMEOUT))?;
            stream.set_write_timeout(Some(IO_TIMEOUT))?;
        }
        let mut connection = Self {
            session: session.clone(),
            sockets: vec![server.try_clone()?, client.try_clone()?],
            threads: Vec::with_capacity(2),
        };
        let sender_session = session.clone();
        connection
            .threads
            .push(
                thread::Builder::new()
                    .name("live-ws-server".into())
                    .spawn(move || {
                        crate::ws_handle_client_mode(server, state, Some(sender_session));
                    })?,
            );
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        connection.threads.push(
            thread::Builder::new()
                .name("live-ws-receiver".into())
                .spawn(move || {
                    if let Err(error) = receive_client(client, address, &session, &ready_tx) {
                        session.fail_connection(error);
                    }
                })?,
        );
        ready_rx
            .recv_timeout(IO_TIMEOUT)
            .context("live receiver init timeout")??;
        Ok(connection)
    }

    fn close(&mut self) -> Result<()> {
        self.session.stopped.store(true, Ordering::Release);
        self.session.changed.notify_all();
        let mut errors = Vec::with_capacity(4);
        for socket in self.sockets.drain(..) {
            if let Err(error) = socket.shutdown(Shutdown::Both) {
                if error.kind() != std::io::ErrorKind::NotConnected {
                    errors.push(error.to_string());
                }
            }
        }
        for handle in self.threads.drain(..) {
            if handle.join().is_err() {
                errors.push("live connection thread panicked".into());
            }
        }
        ensure!(errors.is_empty(), "live cleanup: {errors:?}");
        Ok(())
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            self.session.fail(error.to_string());
        }
    }
}

fn receive_client(
    stream: TcpStream,
    address: std::net::SocketAddr,
    session: &Session,
    ready: &mpsc::SyncSender<Result<()>>,
) -> Result<()> {
    let mut config = tungstenite::protocol::WebSocketConfig::default();
    config.max_message_size = Some(MAX_PAYLOAD_BYTES * 2);
    config.max_frame_size = Some(MAX_PAYLOAD_BYTES * 2);
    let (mut socket, _) =
        tungstenite::client::client_with_config(format!("ws://{address}"), stream, Some(config))?;
    let initial = socket.read()?;
    let initial = initial.into_text()?;
    let value: serde_json::Value = serde_json::from_str(&initial)?;
    ensure!(value["type"] == "init", "expected production WS init");
    let projects = socket.read()?.into_text()?;
    let projects: serde_json::Value = serde_json::from_str(&projects)?;
    ensure!(
        projects["type"] == "project_list",
        "expected initial project list"
    );
    socket
        .get_ref()
        .set_read_timeout(Some(Duration::from_millis(100)))?;
    ready
        .send(Ok(()))
        .map_err(|_| anyhow::anyhow!("live readiness receiver closed"))?;
    for _ in 0..MAX_TURNS {
        if session.expired() {
            return Ok(());
        }
        match socket.read() {
            Ok(tungstenite::Message::Text(text)) => {
                let received_at = Instant::now();
                session.receive(text.to_string(), received_at)?;
            }
            Ok(tungstenite::Message::Close(_)) => {
                return Err(anyhow::Error::new(tungstenite::Error::ConnectionClosed)
                    .context("live server closed connection"));
            }
            Ok(tungstenite::Message::Ping(_) | tungstenite::Message::Pong(_)) => {}
            Ok(tungstenite::Message::Binary(_) | tungstenite::Message::Frame(_)) => {
                bail!("unexpected live server data message");
            }
            Err(tungstenite::Error::Io(error))
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(error) => return Err(error.into()),
        }
    }
    bail!("live receiver turn limit exhausted")
}

struct Run {
    spec: Spec,
    session: Arc<Session>,
    engine: Option<TrialRecord>,
    errors: Vec<String>,
    elapsed_seconds: f64,
    drain_seconds: f64,
    terminal_matches_maximum: bool,
    cache_seed: serde_json::Map<String, serde_json::Value>,
    receipt_artifacts: serde_json::Map<String, serde_json::Value>,
}

/// The binary dispatches here before starting its ordinary service or workers.
pub fn run(args: &[String]) -> Result<bool> {
    let mut spec = Spec::parse(args)?;
    ensure!(!spec.output.exists(), "live output must be a new directory");
    fs::create_dir(&spec.output).context("create isolated live output")?;
    spec.output = fs::canonicalize(&spec.output)?;
    let mut run = Run {
        session: Arc::new(Session::new(&spec)),
        spec,
        engine: None,
        errors: Vec::with_capacity(16),
        elapsed_seconds: 0.0,
        drain_seconds: 0.0,
        terminal_matches_maximum: false,
        cache_seed: serde_json::Map::new(),
        receipt_artifacts: serde_json::Map::new(),
    };
    // GpuEvolver persists its cache relative to cwd. Enter the owned output
    // before creating it or any threads, and restore cwd after every exit.
    let original = std::env::current_dir()?;
    let result = fs::canonicalize(&run.spec.manifest)
        .map_err(anyhow::Error::from)
        .and_then(|manifest| {
            run.spec.manifest = manifest;
            seed_pipeline_cache(&original, &run.spec.output)
        })
        .and_then(|files| {
            run.cache_seed = files;
            std::env::set_current_dir(&run.spec.output).map_err(anyhow::Error::from)
        })
        .and_then(|()| run.execute());
    if let Err(error) = result {
        run.errors.push(format!("{error:#}"));
    }
    if let Err(error) = std::env::set_current_dir(original) {
        run.errors
            .push(format!("restore working directory: {error}"));
    }
    let record = run.finish()?;
    let successful = record["correctness"] == true;
    fs::write(
        run.spec.output.join("result.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(successful)
}

impl Run {
    fn execute(&mut self) -> Result<()> {
        let setup = Instant::now();
        let input = fixtures::load(&self.spec.manifest, &self.spec.case_id)?;
        ensure!(
            input.manifest.params == MutationParams::default(),
            "live requires exact default parameters"
        );
        let params = &input.manifest.params;
        ensure!(
            (params.chain_count, params.lambda, params.gpu_batch_iters) == (4, 64, 64),
            "live requires 4 chains, lambda64 and 64 iterations"
        );
        ensure!(
            input.fixture.complexity.overflowed_tiles == 0,
            "initial tile-bin overflow"
        );
        let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
            &input.rgba,
            input.fixture.width,
            input.fixture.height,
            &input.drawing,
            EngineCapacity::for_params(params),
        ));
        let scopes = evolver.benchmark_error_scopes();
        let result = self.prepared_run(&mut evolver, &input, setup);
        self.errors
            .extend(futures_lite::future::block_on(scopes.finish()));
        result
    }

    fn prepared_run(
        &mut self,
        evolver: &mut GpuEvolver,
        input: &LoadedFixture,
        setup: Instant,
    ) -> Result<()> {
        self.engine = Some(trial::prepare(
            evolver,
            input,
            &self.spec.trial(),
            &self.spec.output,
        )?);
        let mut latest = self.seed_drawing(input)?;
        let state = initial_ws_state(input, &latest.drawing)?;
        let mut render_buf = vec![0; input.rgba.len()];
        let mut connection = Connection::start(state.clone(), self.session.clone())?;
        self.engine
            .as_mut()
            .context("missing prepared trial")?
            .setup_seconds = setup.elapsed().as_secs_f64();
        self.session.start()?;
        let workload = self.measure(evolver, input, &state, &mut latest, &mut render_buf);
        if let Err(error) = workload {
            self.errors.push(format!("{error:#}"));
        }
        if let Err(error) = connection.close() {
            self.errors.push(format!("{error:#}"));
        }
        self.receipt_artifacts = self.session.save_receipts(&self.spec.output)?;
        // Correctness readbacks and disk artifacts are outside the measured path.
        let params = &input.manifest.params;
        let snapshot = evolver
            .benchmark_snapshot(params.chain_count, params.lambda)
            .map_err(anyhow::Error::msg)?;
        self.terminal_matches_maximum = terminal_matches(&latest.drawing, &snapshot);
        if !self.terminal_matches_maximum {
            self.errors
                .push("terminal published drawing does not match a packed maximum parent".into());
        }
        let engine = self.engine.as_mut().context("missing prepared trial")?;
        trial::save_state(&self.spec.output, &snapshot)?;
        trial::complete_record(engine, &snapshot, input)?;
        let timings = evolver.pass_timings();
        engine.timestamp_samples = timings.sample_count;
        engine.sampled_gpu_stage_totals_ns = [
            timings.mutate_ns,
            timings.bin_polygons_ns,
            timings.rasterize_error_ns,
            timings.select_ns,
        ];
        self.session
            .validate_payloads(input.fixture.width, input.fixture.height)?;
        Ok(())
    }

    fn seed_drawing(&self, input: &LoadedFixture) -> Result<Latest> {
        let engine = self.engine.as_ref().context("missing prepared trial")?;
        let score = engine
            .initial_quality
            .first()
            .context("missing initial quality")?
            .stored_fitness;
        let mut drawing = gpu_to_drawing(&drawing_to_gpu(&input.drawing, self.spec.seed));
        drawing.fitness = score;
        Ok(Latest {
            drawing,
            acquisition: Acquisition {
                completed_batches: 0,
                fitness_bits: score.to_bits(),
                maximum_fitness_bits: score.to_bits(),
                is_acquisition_maximum: true,
            },
        })
    }

    fn measure(
        &mut self,
        evolver: &mut GpuEvolver,
        input: &LoadedFixture,
        state: &crate::SharedWsState,
        latest: &mut Latest,
        render_buf: &mut [u8],
    ) -> Result<()> {
        let work = self.submit(evolver, input, state, latest, render_buf);
        let drain = Instant::now();
        let final_drawing = evolver.flush_pending();
        let observation = latest.observe(final_drawing, evolver);
        let checked = evolver.benchmark_drain().map_err(anyhow::Error::msg);
        self.drain_seconds = drain.elapsed().as_secs_f64();
        let counters = evolver.benchmark_counters();
        if let Some(engine) = &mut self.engine {
            engine.counters = counters;
            engine.drain_seconds = self.drain_seconds;
        }
        self.elapsed_seconds = self.session.elapsed_seconds()?;
        work?;
        observation?;
        checked?;
        self.publish_if_due(counters, state, latest, render_buf, input)?;
        let receipt_result = self.session.wait_for_receipts();
        self.elapsed_seconds = self.session.elapsed_seconds()?;
        if let Some(engine) = &mut self.engine {
            engine.elapsed_seconds = self.elapsed_seconds;
            engine.evals_per_second = Some(evaluation_rate(
                counters.completed_evaluations,
                self.elapsed_seconds,
            )?);
        }
        receipt_result?;
        Ok(())
    }

    fn submit(
        &self,
        evolver: &mut GpuEvolver,
        input: &LoadedFixture,
        state: &crate::SharedWsState,
        latest: &mut Latest,
        render_buf: &mut [u8],
    ) -> Result<()> {
        for _ in 0..self.spec.batches {
            ensure!(
                !self.session.expired(),
                "live work timeout or connection failure"
            );
            let drawing = evolver.run_batch(&input.manifest.params, true);
            latest.observe(drawing, evolver)?;
            self.publish_if_due(
                evolver.benchmark_counters(),
                state,
                latest,
                render_buf,
                input,
            )?;
        }
        Ok(())
    }

    fn publish_if_due(
        &self,
        counters: BenchmarkCounters,
        state: &crate::SharedWsState,
        latest: &Latest,
        render_buf: &mut [u8],
        input: &LoadedFixture,
    ) -> Result<()> {
        if counters.completed_batches == 0
            || !counters
                .completed_batches
                .is_multiple_of(u64::from(self.spec.publish_every))
        {
            return Ok(());
        }
        let due_at = Instant::now();
        let mut trace = self.session.trace_clock().map(|clock| PublicationTrace {
            sequence: usize::try_from(
                counters.completed_batches / u64::from(self.spec.publish_every),
            )
            .expect("bounded live publication count"),
            completed_batches: counters.completed_batches,
            completed_evaluations: counters.completed_evaluations,
            due_ns: Some(clock.at(due_at)),
            ..PublicationTrace::default()
        });
        let result = self.prepare_publication(
            (counters, due_at),
            state,
            latest,
            render_buf,
            input,
            trace.as_mut(),
        );
        if result.is_err() {
            if let Some(row) = trace {
                if let Err(error) = self.session.retain_partial_trace(row) {
                    self.session
                        .fail(format!("retain partial trace: {error:#}"));
                }
            }
        }
        result
    }

    // This preparation stays on the production render/encode path. Optional
    // marks observe each operation once; the existing ledger owns committed rows.
    fn prepare_publication(
        &self,
        origin: (BenchmarkCounters, Instant),
        state: &crate::SharedWsState,
        latest: &Latest,
        render_buf: &mut [u8],
        input: &LoadedFixture,
        mut trace: Option<&mut PublicationTrace>,
    ) -> Result<()> {
        let (counters, due_at) = origin;
        let (w, h) = (input.fixture.width as usize, input.fixture.height as usize);
        latest.drawing.draw(render_buf, w, h, Rasterizer::HalfSpace);
        if let Some(row) = trace.as_deref_mut() {
            row.render_done_ns = self.session.trace_clock().map(latency::Clock::now);
        }
        let png = crate::encode_rgba_as_png(render_buf, w, h);
        if let Some(row) = trace.as_deref_mut() {
            row.png_done_ns = self.session.trace_clock().map(latency::Clock::now);
        }
        let drawing_json = serde_json::to_string(&latest.drawing)?;
        if let Some(row) = trace.as_deref_mut() {
            row.drawing_json_done_ns = self.session.trace_clock().map(latency::Clock::now);
            row.ws_lock_attempt_ns = self.session.trace_clock().map(latency::Clock::now);
        }
        let mut state = state
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("WS state poisoned"))?;
        if let Some(row) = trace.as_deref_mut() {
            row.ws_lock_acquired_ns = self.session.trace_clock().map(latency::Clock::now);
        }
        state.best_png = png;
        state.drawing_json = drawing_json;
        state.fitness = latest.drawing.fitness;
        state.polygons = latest.drawing.polygons.len();
        state.total_evals = counters.completed_evaluations;
        state.generation += 1;
        state.image_generation += 1;
        self.session
            .publish(&state, counters, &latest.acquisition, due_at, trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire_trial_quality() -> trial::ChainQuality {
        use artgen_backend_rust::benchmark::uncertainty::PixelAudit;

        trial::ChainQuality {
            chain: 0,
            triangles: 0,
            stored_error: 0,
            framebuffer_error: 0,
            scalar_error: 0,
            stored_fitness: 97.3,
            expected_stored_fitness: 97.3,
            scalar_fitness: 97.1,
            differing_pixels: 0,
            max_channel_delta: 0,
            total_channel_delta: 0,
            pixel_audit: PixelAudit::default(),
        }
    }

    fn wire_trial_record() -> TrialRecord {
        use artgen_backend_rust::{
            benchmark::reference, gpu_evolver::benchmark::BenchmarkProvenance,
        };

        TrialRecord {
            protocol_version: 1,
            status: "completed".into(),
            error_count: 0,
            errors: Vec::new(),
            case_id: "mona".into(),
            seed: 13,
            batches: 4,
            diagnostic: false,
            elapsed_seconds: 1.0,
            drain_seconds: 0.0,
            setup_seconds: 0.0,
            evals_per_second: Some(65536.0),
            counters: counters(),
            initial_state_sha256: "initial".into(),
            final_state_sha256: "final".into(),
            initial_similarity_percent: 98.2,
            final_fitness_percent: vec![97.3],
            fixture_manifest_sha256: "manifest".into(),
            initial_complexity: reference::complexity(&[], 1, 1, [8, 8]).unwrap(),
            final_complexity: Vec::new(),
            validation_max_tile_occupancy: None,
            correctness: true,
            initial_quality: vec![wire_trial_quality()],
            final_quality: vec![wire_trial_quality()],
            gpu: BenchmarkProvenance {
                name: "CPU serialization test".into(),
                vendor: 0,
                device: 0,
                device_type: String::new(),
                backend: String::new(),
                driver: String::new(),
                driver_info: String::new(),
                timestamp_period_ns: 0.2,
                allocated_chains: 4,
                offspring_capacity: 256,
                max_storage_buffer_binding_size: 0,
                max_buffer_size: 0,
            },
            params: MutationParams::default(),
            timestamp_samples: 0,
            sampled_gpu_stage_totals_ns: [0.0; 4],
            timestamp_mode: TimestampMode::Stages,
            batch_compute_envelope: None,
            start_mode: "mature_drawing_restart".into(),
        }
    }

    #[test]
    fn ordinary_trial_f32_fields_keep_their_typed_json_wire_values() {
        let record = wire_trial_record();
        let value = trial_record_value(&record).unwrap();
        let wire: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        let manifest_params: serde_json::Value = serde_json::from_str(
            r#"{
                "addPolygonProb": 0.5, "removePolygonProb": 0.000862,
                "reorderPolygonProb": 0.002, "offsetPolygonProb": 0.00565,
                "movePointProb": 0.000294, "removePointProb": 0.002,
                "microAdjustProb": 0.002786, "changeColorProb": 0.000473,
                "adjustBrightnessProb": 0.000473, "adjustSaturationProb": 0.001333,
                "scalePolygonProb": 0.00565, "rotatePolygonProb": 0.002,
                "adjacentSwapProb": 0.002, "mergePolygonProb": 0.000667,
                "clonePolygonProb": 0.071429, "mediumMoveProb": 0.004,
                "swapColorsProb": 0.00565, "movePointMaxDelta": 0.113,
                "microAdjustDelta": 0.001, "newPointMaxDistance": 0.005,
                "offsetPolygonMagnitude": 0.01, "mediumMoveDelta": 0.03,
                "mergeCentroidThreshold": 0.15, "mergeColorThreshold": 0.2,
                "minAlpha": 10, "maxAlpha": 65, "minPolygons": 1,
                "maxPolygons": 1000, "crossoverProb": 0.2,
                "spatialCrossoverWeight": 0.66, "tournamentSize": 3,
                "chainCount": 4, "lambda": 64, "singleMutationMode": true,
                "adaptiveMutation": false, "rasterizeWg": [32, 16], "gpuBatchIters": 64
            }"#,
        )
        .unwrap();
        let ordinary: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        assert_eq!(ordinary["params"], manifest_params);
        assert_eq!(wire["params"], manifest_params);
        assert_eq!(wire["initial_similarity_percent"], 98.2);
        assert_eq!(wire["final_fitness_percent"], serde_json::json!([97.3]));
        for field in ["initial_quality", "final_quality"] {
            assert_eq!(wire[field][0]["stored_fitness"], 97.3);
            assert_eq!(wire[field][0]["expected_stored_fitness"], 97.3);
            assert_eq!(wire[field][0]["scalar_fitness"], 97.1);
        }
        assert_eq!(wire["gpu"]["timestamp_period_ns"], 0.2);
        assert_eq!(wire, ordinary);
    }

    #[test]
    fn finish_preserves_ordinary_fields_and_marks_the_live_protocol_as_version_two() {
        let record = wire_trial_record();
        let ordinary: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        let session = Arc::new(Session::new(&spec()));
        session.ledger.lock().unwrap().records = vec![publication(1), publication(2)];
        let mut run = Run {
            spec: spec(),
            session,
            engine: Some(record),
            errors: Vec::new(),
            elapsed_seconds: 1.0,
            drain_seconds: 0.0,
            terminal_matches_maximum: true,
            cache_seed: serde_json::Map::new(),
            receipt_artifacts: serde_json::Map::new(),
        };
        let value = run.finish().unwrap();
        let wire: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert_eq!(wire["live_protocol_version"], 2);
        assert_eq!(wire["protocol_version"], 1);
        assert_eq!(wire["correctness"], true);
        for (field, expected) in ordinary.as_object().unwrap() {
            assert_eq!(&wire[field], expected, "ordinary field {field} changed");
        }
        run.engine = None;
        let invalid = run.finish().unwrap();
        assert_eq!(invalid["live_protocol_version"], 2);
        assert_eq!(invalid["protocol_version"], 1);
        assert_eq!(invalid["correctness"], false);
    }

    fn spec() -> Spec {
        Spec {
            case_id: "mona".into(),
            seed: 13,
            batches: 4,
            warmup_batches: 2,
            publish_every: 2,
            deadline_ms: 300,
            latency_trace: false,
            manifest: "manifest.json".into(),
            output: "new".into(),
        }
    }

    fn cpu_input() -> LoadedFixture {
        let drawing = Drawing {
            polygons: Vec::new(),
            is_dirty: false,
            fitness: 99.0,
        };
        let fixture = fixtures::Fixture {
            id: "mona".into(),
            width: 2,
            height: 2,
            drawing_file: String::new(),
            reference_file: String::new(),
            drawing_sha256: String::new(),
            reference_sha256: String::new(),
            source_image_sha256: String::new(),
            stored_fitness_percent: 99.0,
            scalar_fitness_percent: 99.0,
            complexity: artgen_backend_rust::benchmark::reference::complexity(&[], 2, 2, [8, 8])
                .unwrap(),
        };
        LoadedFixture {
            manifest: fixtures::Manifest {
                protocol_version: 1,
                start_mode: "mature_drawing_restart".into(),
                resolution: 2,
                minimum_similarity_percent: 97.0,
                params: MutationParams::default(),
                cases: vec![fixture.clone()],
            },
            fixture,
            drawing,
            rgba: vec![255; 16],
            manifest_sha256: String::new(),
        }
    }

    fn cpu_run(traced: bool) -> Run {
        let mut spec = spec();
        spec.latency_trace = traced;
        spec.output =
            std::env::temp_dir().join(format!("artgen-live-latency-{}", uuid::Uuid::new_v4()));
        Run {
            session: Arc::new(Session::new(&spec)),
            spec,
            engine: None,
            errors: Vec::new(),
            elapsed_seconds: 0.0,
            drain_seconds: 0.0,
            terminal_matches_maximum: false,
            cache_seed: serde_json::Map::new(),
            receipt_artifacts: serde_json::Map::new(),
        }
    }

    fn cpu_publish(
        run: &Run,
        input: &LoadedFixture,
        state: &crate::SharedWsState,
        sequence: u32,
    ) -> Result<()> {
        let latest = Latest {
            drawing: input.drawing.clone(),
            acquisition: publication(1).acquisition,
        };
        run.publish_if_due(
            BenchmarkCounters {
                completed_batches: u64::from(sequence) * 2,
                completed_evaluations: u64::from(sequence) * 32768,
                ..Default::default()
            },
            state,
            &latest,
            &mut [0; 16],
            input,
        )
    }

    #[test]
    fn latency_trace_parser_rejects_versions_types_and_duplicates_before_startup() {
        let flags = ["--manifest", "manifest.json", "--output", "new"]
            .map(String::from)
            .to_vec();
        for version in ["v0", "v2", "1", "true", "", "{}"] {
            let mut args = flags.clone();
            args.extend(["--latency-trace".into(), version.into()]);
            assert!(Spec::parse(&args).is_err());
        }
        let mut args = flags;
        args.extend(["--latency-trace".into(), "v1".into()]);
        let parsed = Spec::parse(&args).unwrap();
        assert!(parsed.latency_trace);
        assert!(!parsed.trial().diagnostic);
        assert_eq!(parsed.trial().timestamp_mode, TimestampMode::Stages);
        args.extend(["--latency-trace".into(), "v1".into()]);
        assert!(Spec::parse(&args).is_err());
        args.pop();
        assert!(Spec::parse(&args).is_err());
    }

    #[test]
    fn disabled_latency_trace_keeps_default_allocations_wire_keys_and_counters() {
        let mut run = cpu_run(false);
        let input = cpu_input();
        let state = initial_ws_state(&input, &input.drawing).unwrap();
        assert!(run.session.trace_clock().is_none());
        assert!(run.session.sender_trace().unwrap().is_none());
        assert!(run.session.ledger.lock().unwrap().trace.is_none());
        run.session.start().unwrap();
        cpu_publish(&run, &input, &state, 1).unwrap();
        let mut last_image = Instant::now()
            .checked_sub(Duration::from_millis(201))
            .unwrap();
        let (message, ticket) = run
            .session
            .message(&mut state.0.lock().unwrap(), &mut last_image, None)
            .unwrap()
            .unwrap();
        run.session.sent(ticket).unwrap();
        let value: serde_json::Value = serde_json::from_str(&message).unwrap();
        assert_eq!(value["liveAcceptance"]["completed_batches"], 2);
        assert_eq!(value["liveAcceptance"]["completed_evaluations"], 32768);
        assert_eq!(value["liveAcceptance"].as_object().unwrap().len(), 8);
        assert!(!message.contains("_ns"));
        assert!(!message.contains("latency_trace"));
        let result = run.finish().unwrap();
        assert_eq!(result["mode"], "live_acceptance");
        assert_eq!(result["live_protocol_version"], 2);
        assert!(result.get("score_eligible").is_none());
        assert!(result.get("latency_trace").is_none());
        assert!(!run.spec.output.exists());
    }

    #[test]
    fn latency_trace_real_loopback_covers_producer_handler_receipts_and_artifacts() {
        let mut run = cpu_run(true);
        fs::create_dir(&run.spec.output).unwrap();
        let result = (|| -> Result<()> {
            let input = cpu_input();
            let state = initial_ws_state(&input, &input.drawing)?;
            let mut connection = Connection::start(state.clone(), run.session.clone())?;
            run.session.start()?;
            for sequence in 1..=2 {
                cpu_publish(&run, &input, &state, sequence)?;
                wait_for_receipt(&run.session, sequence as usize);
            }
            connection.close()?;
            run.session.validate_payloads(2, 2)?;
            run.receipt_artifacts = run.session.save_receipts(&run.spec.output)?;
            run.engine = Some(wire_trial_record());
            run.elapsed_seconds = run.session.elapsed_seconds()?;
            run.terminal_matches_maximum = true;
            let record = run.finish()?;
            ensure!(record["correctness"] == true, "{record}");
            ensure!(
                record["mode"] == "live_latency_diagnostic" && record["score_eligible"] == false,
                "diagnostic was score eligible"
            );
            let bytes = fs::read(run.spec.output.join(latency::FILE))?;
            ensure!(
                record["latency_trace"]["sha256"] == fixtures::sha256(&bytes),
                "sidecar hash mismatch"
            );
            ensure!(
                record["latency_trace"]["bytes"] == bytes.len(),
                "sidecar byte count mismatch"
            );
            let trace: serde_json::Value = serde_json::from_slice(&bytes)?;
            ensure!(
                trace["complete"] == true && trace["errors"] == serde_json::json!([]),
                "{trace}"
            );
            ensure!(
                trace["measurement_start_ns"].is_u64(),
                "missing measurement origin"
            );
            check_latency_trace_publications(&trace)?;
            let receipts = fs::read_to_string(run.spec.output.join("receipt-000.jsonl"))?;
            ensure!(
                receipts.lines().count() == 2 && !receipts.contains("_ns"),
                "wire receipt changed"
            );
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&run.spec.output);
        assert!(result.is_ok(), "{result:?}");
        assert!(cleanup.is_ok(), "{cleanup:?}");
    }

    fn check_latency_trace_publications(trace: &serde_json::Value) -> Result<()> {
        ensure!(
            trace["publications"].as_array().context("rows")?.len() == 2,
            "wrong publication count"
        );
        ensure!(
            trace["sender_turns"]
                .as_array()
                .context("turns")?
                .iter()
                .any(|turn| turn["publication_sequence"].is_null()
                    && turn["write_return_ns"].is_u64()),
            "stats send missing"
        );
        for sequence in 1..=2 {
            let row = &trace["publications"][sequence - 1];
            ensure!(
                row["sequence"] == sequence && row["completed_batches"] == sequence * 2,
                "trace origin mismatch"
            );
            ensure!(
                row["completed_evaluations"] == sequence * 32768,
                "trace work mismatch"
            );
            ensure!(
                row["receive_ns"].is_u64() && row["render_done_ns"].is_u64(),
                "real hook missing"
            );
        }
        Ok(())
    }

    #[test]
    fn latency_trace_throttling_retains_queue_without_another_ledger_lock() {
        let run = cpu_run(true);
        let input = cpu_input();
        let state = initial_ws_state(&input, &input.drawing).unwrap();
        let mut sender = run.session.sender_trace().unwrap().unwrap();
        run.session.start().unwrap();
        cpu_publish(&run, &input, &state, 1).unwrap();
        let mut last_image = Instant::now() + RUN_TIMEOUT;
        // Holding the ledger proves the throttled branch does not lock it.
        let log = run.session.ledger.lock().unwrap();
        let row = sender.begin_turn().unwrap();
        assert!(run
            .session
            .message(&mut state.0.lock().unwrap(), &mut last_image, Some(row))
            .unwrap()
            .is_none());
        assert_eq!(row.image_gate_passed, Some(false));
        assert!(row.ledger_lock_attempt_ns.is_none());
        assert_eq!(log.queue.len(), 1);
        assert_eq!(log.queue.front().unwrap().0, 1);
        drop(log);
        drop(sender);
    }

    #[test]
    fn latency_trace_producer_failure_retains_the_completed_preparation_stages() {
        let run = cpu_run(true);
        fs::create_dir(&run.spec.output).unwrap();
        let result = (|| -> Result<()> {
            let input = cpu_input();
            let state = initial_ws_state(&input, &input.drawing)?;
            // No measurement clock: the actual producer must reject enqueue,
            // retaining render/PNG/JSON/lock evidence completed before failure.
            let error = cpu_publish(&run, &input, &state, 1).expect_err("unstarted clock");
            run.session.fail(error.to_string());
            let artifact = run
                .session
                .save_trace(&[])?
                .context("missing partial trace")?;
            ensure!(artifact["complete"] == false, "partial trace passed");
            let trace: serde_json::Value =
                serde_json::from_slice(&fs::read(run.spec.output.join(latency::FILE))?)?;
            let row = &trace["publications"][0];
            ensure!(
                row["sequence"] == 1
                    && row["render_done_ns"].is_u64()
                    && row["png_done_ns"].is_u64()
                    && row["drawing_json_done_ns"].is_u64()
                    && row["ledger_lock_acquired_ns"].is_u64(),
                "partial stages disappeared"
            );
            ensure!(
                row["enqueue_commit_ns"].is_null() && row["receive_ns"].is_null(),
                "failure fabricated completion"
            );
            ensure!(
                run.session.ledger.lock().unwrap().queue.is_empty(),
                "failed publication enqueued"
            );
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&run.spec.output);
        assert!(result.is_ok(), "{result:?}");
        assert!(cleanup.is_ok(), "{cleanup:?}");
    }

    #[test]
    fn latency_trace_early_handler_error_retains_partial_turns_and_producer_rows() {
        let run = cpu_run(true);
        fs::create_dir(&run.spec.output).unwrap();
        let result = (|| -> Result<()> {
            let input = cpu_input();
            let state = initial_ws_state(&input, &input.drawing)?;
            let mut connection = Connection::start(state.clone(), run.session.clone())?;
            run.session.start()?;
            cpu_publish(&run, &input, &state, 1)?;
            wait_for_receipt(&run.session, 1);
            connection.sockets[1].write_all(&[0x81, 0x81, 0, 0, 0, 0, b'{'])?;
            wait_for_server_exit(&connection);
            connection.close()?;
            let artifact = run.session.save_trace(&[])?.context("trace missing")?;
            ensure!(artifact["complete"] == false, "partial trace passed");
            let trace: serde_json::Value =
                serde_json::from_slice(&fs::read(run.spec.output.join(latency::FILE))?)?;
            ensure!(
                trace["publications"].as_array().context("rows")?.len() == 1,
                "partial publication disappeared"
            );
            ensure!(
                !trace["sender_turns"]
                    .as_array()
                    .context("turns")?
                    .is_empty(),
                "sender evidence disappeared on error"
            );
            ensure!(
                !trace["errors"].as_array().context("errors")?.is_empty(),
                "protocol error lost"
            );
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&run.spec.output);
        assert!(result.is_ok(), "{result:?}");
        assert!(cleanup.is_ok(), "{cleanup:?}");
    }

    fn counters() -> BenchmarkCounters {
        BenchmarkCounters {
            submitted_batches: 4,
            completed_batches: 4,
            submitted_iterations: 256,
            completed_iterations: 256,
            submitted_evaluations: 65536,
            completed_evaluations: 65536,
        }
    }

    fn publication(sequence: usize) -> Publication {
        let batches = sequence as u64 * 2;
        Publication {
            run_id: "test".into(),
            sequence,
            completed_batches: batches,
            completed_evaluations: batches * 16384,
            due_ms: sequence as f64 * 200.0,
            send_ms: Some(sequence as f64 * 200.0 + 10.0),
            receive_ms: Some(sequence as f64 * 200.0 + 20.0),
            drawing_sha256: "drawing".into(),
            png_sha256: "png".into(),
            fitness_percent: 99.0,
            polygons: 0,
            payload_valid: true,
            acquisition: Acquisition {
                completed_batches: 0,
                fitness_bits: 99.0f32.to_bits(),
                maximum_fitness_bits: 99.0f32.to_bits(),
                is_acquisition_maximum: true,
            },
        }
    }

    fn transport_close_errors() -> [anyhow::Error; 8] {
        use std::io::{Error, ErrorKind};
        use tungstenite::{error::ProtocolError, Error as WsError};
        [
            WsError::Io(ErrorKind::BrokenPipe.into()).into(),
            WsError::Io(ErrorKind::ConnectionReset.into()).into(),
            WsError::Io(ErrorKind::ConnectionAborted.into()).into(),
            WsError::Io(ErrorKind::NotConnected.into()).into(),
            WsError::Io(ErrorKind::UnexpectedEof.into()).into(),
            WsError::ConnectionClosed.into(),
            WsError::Protocol(ProtocolError::ResetWithoutClosingHandshake).into(),
            Error::from(ErrorKind::BrokenPipe).into(),
        ]
    }

    fn connection_after_receipts() -> Connection {
        let session = Arc::new(Session::new(&spec()));
        let drawing = Drawing {
            polygons: Vec::new(),
            is_dirty: false,
            fitness: 99.0,
        };
        let state = Arc::new((
            Mutex::new(crate::WsState {
                best_png: crate::encode_rgba_as_png(&[255; 16], 2, 2),
                drawing_json: serde_json::to_string(&drawing).unwrap(),
                fitness: drawing.fitness,
                image_width: 2,
                image_height: 2,
                ..Default::default()
            }),
            Condvar::new(),
        ));
        let connection = Connection::start(state.clone(), session.clone()).unwrap();
        session.start().unwrap();
        for sequence in 1..=session.spec.publication_count() {
            let record = publication(sequence);
            let work = BenchmarkCounters {
                completed_batches: record.completed_batches,
                completed_evaluations: record.completed_evaluations,
                ..Default::default()
            };
            session
                .publish(
                    &state.0.lock().unwrap(),
                    work,
                    &record.acquisition,
                    Instant::now(),
                    None,
                )
                .unwrap();
            wait_for_receipt(&session, sequence);
        }
        session.validate_payloads(2, 2).unwrap();
        let log = session.ledger.lock().unwrap();
        assert!(log.errors.is_empty(), "{:?}", log.errors);
        assert!(!session.expired());
        connection
    }

    fn wait_for_receipt(session: &Session, sequence: usize) {
        let deadline = Instant::now() + IO_TIMEOUT;
        let mut log = session.ledger.lock().unwrap();
        for _ in 0..MAX_TURNS {
            if log.received_count == sequence {
                return;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "real WS receipts timed out");
            (log, _) = session.changed.wait_timeout(log, remaining).unwrap();
        }
        panic!("real WS receipt wake limit exhausted");
    }

    fn wait_for_server_exit(connection: &Connection) {
        let deadline = Instant::now() + IO_TIMEOUT;
        for _ in 0..2000 {
            if connection.threads[0].is_finished() {
                return;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "real WS handler did not exit");
            thread::sleep(remaining.min(Duration::from_millis(1)));
        }
        panic!("real WS handler wait limit exhausted");
    }

    #[test]
    fn late_message_malformed_client_json_is_not_ignored() {
        let mut connection = connection_after_receipts();
        connection.sockets[1]
            .write_all(&[0x81, 0x81, 0, 0, 0, 0, b'{'])
            .unwrap();
        wait_for_server_exit(&connection);
        connection.close().unwrap();
        assert!(!connection.session.ledger.lock().unwrap().errors.is_empty());
    }

    #[test]
    fn late_message_binary_client_data_is_not_ignored() {
        let mut connection = connection_after_receipts();
        connection.sockets[1]
            .write_all(&[0x82, 0x80, 0, 0, 0, 0])
            .unwrap();
        wait_for_server_exit(&connection);
        connection.close().unwrap();
        assert!(!connection.session.ledger.lock().unwrap().errors.is_empty());
    }

    #[test]
    fn late_message_binary_server_data_is_not_ignored() {
        let mut connection = connection_after_receipts();
        connection.sockets[0].write_all(&[0x82, 0]).unwrap();
        let deadline = Instant::now() + IO_TIMEOUT;
        for _ in 0..2000 {
            if connection.threads[1].is_finished() {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "receiver ignored binary data");
            thread::sleep(remaining.min(Duration::from_millis(1)));
        }
        connection.close().unwrap();
        assert!(!connection.session.ledger.lock().unwrap().errors.is_empty());
    }

    #[test]
    fn reserved_opcode_after_receipts_invalidates_the_session() {
        let mut connection = connection_after_receipts();
        // Masked FIN + reserved data opcode 3, with an empty payload.
        connection.sockets[1]
            .write_all(&[0x83, 0x80, 0, 0, 0, 0])
            .unwrap();
        wait_for_server_exit(&connection);
        connection.close().unwrap();
        let log = connection.session.ledger.lock().unwrap();
        assert!(
            !log.errors.is_empty(),
            "reserved opcode vanished after all real WS receipts"
        );
    }

    #[test]
    fn client_close_frame_after_receipts_invalidates_the_session() {
        let mut connection = connection_after_receipts();
        connection.sockets[1]
            .write_all(&[0x88, 0x80, 0, 0, 0, 0])
            .unwrap();
        wait_for_server_exit(&connection);
        connection.close().unwrap();
        let log = connection.session.ledger.lock().unwrap();
        assert!(
            !log.errors.is_empty(),
            "client close frame vanished before owned shutdown"
        );
    }

    #[test]
    fn client_socket_close_before_owned_shutdown_invalidates_the_session() {
        let mut connection = connection_after_receipts();
        connection.sockets[1].shutdown(Shutdown::Both).unwrap();
        wait_for_server_exit(&connection);
        connection.close().unwrap();
        let log = connection.session.ledger.lock().unwrap();
        assert!(
            !log.errors.is_empty(),
            "client socket closure vanished before owned shutdown"
        );
    }

    #[test]
    fn owned_shutdown_after_real_receipts_keeps_the_session_clean() {
        let mut connection = connection_after_receipts();
        connection.close().unwrap();
        let log = connection.session.ledger.lock().unwrap();
        assert!(log.errors.is_empty(), "{:?}", log.errors);
    }

    #[test]
    fn transport_close_before_owned_shutdown_invalidates_the_session() {
        for error in transport_close_errors() {
            let session = Session::new(&spec());
            session.fail_connection(error.context("WS operation"));
            assert_eq!(session.ledger.lock().unwrap().errors.len(), 1);
            assert!(session.expired());
        }
    }

    #[test]
    fn expected_transport_close_after_owned_shutdown_keeps_the_session_clean() {
        for error in transport_close_errors() {
            let session = Arc::new(Session::new(&spec()));
            let mut connection = Connection {
                session: session.clone(),
                sockets: Vec::new(),
                threads: Vec::new(),
            };
            connection.close().unwrap();
            session.fail_connection(error.context("WS operation"));
            let log = session.ledger.lock().unwrap();
            assert!(log.errors.is_empty(), "{:?}", log.errors);
        }
    }

    #[test]
    fn other_errors_after_owned_shutdown_still_invalidate_the_session() {
        use std::io::ErrorKind;
        use tungstenite::{error::ProtocolError, Error as WsError};
        let errors: [anyhow::Error; 6] = [
            WsError::Io(ErrorKind::TimedOut.into()).into(),
            WsError::Io(ErrorKind::InvalidData.into()).into(),
            WsError::AlreadyClosed.into(),
            WsError::Protocol(ProtocolError::SendAfterClosing).into(),
            WsError::Protocol(ProtocolError::UnknownDataFrameType(3)).into(),
            serde_json::from_str::<serde_json::Value>("{")
                .unwrap_err()
                .into(),
        ];
        for error in errors {
            let session = Arc::new(Session::new(&spec()));
            let mut connection = Connection {
                session: session.clone(),
                sockets: Vec::new(),
                threads: Vec::new(),
            };
            connection.close().unwrap();
            session.fail_connection(error);
            assert_eq!(session.ledger.lock().unwrap().errors.len(), 1);
            session.fail("late payload validation failed");
            assert_eq!(session.ledger.lock().unwrap().errors.len(), 2);
        }
    }

    #[test]
    fn a_runtime_timeout_does_not_hide_a_transport_failure() {
        let mut session = Session::new(&spec());
        session.wall_start = Instant::now() - RUN_TIMEOUT;
        assert!(session.expired());
        session.fail_connection(tungstenite::Error::Io(
            std::io::ErrorKind::BrokenPipe.into(),
        ));
        assert_eq!(session.ledger.lock().unwrap().errors.len(), 1);
    }

    #[test]
    fn websocket_send_during_owned_shutdown_keeps_the_session_clean() {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect_timeout(&address, IO_TIMEOUT).unwrap();
        let (server, _) = listener.accept().unwrap();
        drop(listener);
        for stream in [&client, &server] {
            stream.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
            stream.set_write_timeout(Some(IO_TIMEOUT)).unwrap();
        }
        let session = Arc::new(Session::new(&spec()));
        let sender_session = session.clone();
        let mut connection = Connection {
            session: session.clone(),
            sockets: vec![server.try_clone().unwrap(), client.try_clone().unwrap()],
            threads: Vec::with_capacity(1),
        };
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        connection.threads.push(thread::spawn(move || {
            let mut socket = tungstenite::accept(server).unwrap();
            ready_tx.send(()).unwrap();
            // The owned shutdown wakes this blocked turn before the stats send.
            let mut byte = [0];
            assert_eq!(socket.get_ref().peek(&mut byte).unwrap(), 0);
            let error = crate::send_ws_update(&mut socket, "{\"type\":\"stats\"}".into(), None)
                .expect_err("a send on the shut-down socket must fail");
            sender_session.fail_connection(error);
        }));
        let (_client, _) = tungstenite::client(format!("ws://{address}"), client).unwrap();
        ready_rx.recv_timeout(IO_TIMEOUT).unwrap();
        connection.close().unwrap();
        let log = session.ledger.lock().unwrap();
        assert!(log.errors.is_empty(), "{:?}", log.errors);
    }

    #[test]
    fn a_connection_thread_panic_is_recorded_during_owned_shutdown() {
        let session = Arc::new(Session::new(&spec()));
        let connection = Connection {
            session: session.clone(),
            sockets: Vec::new(),
            threads: vec![thread::spawn(|| panic!("test connection panic"))],
        };
        drop(connection);
        let log = session.ledger.lock().unwrap();
        assert_eq!(log.errors.len(), 1);
        assert!(log.errors[0].contains("thread panicked"));
    }

    #[test]
    fn completed_work_and_receipts_pass_the_fixed_schedule() {
        let records = [publication(1), publication(2)];
        let (metrics, errors) = assess(&spec(), counters(), &records, 1.0);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            (
                metrics.previews_due,
                metrics.previews_sent,
                metrics.previews_received
            ),
            (2, 2, 2)
        );
        assert_eq!(metrics.p95_latency_ms, Some(20.0));
    }

    #[test]
    fn submitted_work_cannot_stand_in_for_unfinished_work() {
        let mut work = counters();
        work.completed_batches = 3;
        work.completed_iterations = 192;
        work.completed_evaluations = 49152;
        let (metrics, errors) = assess(&spec(), work, &[publication(1), publication(2)], 1.0);
        assert!(!errors.is_empty());
        assert_eq!(work.completed_evaluations, 49152);
        assert_eq!(metrics.previews_received, 2);
    }

    #[test]
    fn a_missing_payload_fails_even_when_receipt_counters_are_complete() {
        let mut records = [publication(1), publication(2)];
        records[1].payload_valid = false;
        let (metrics, errors) = assess(&spec(), counters(), &records, 1.0);
        assert!(!errors.is_empty());
        assert_eq!(metrics.previews_received, 2);
        assert!(validate_payload("{}", &records[1], 2, 2).is_err());
    }

    #[test]
    fn late_receipt_fails_and_preserves_its_latency() {
        let mut records = [publication(1), publication(2)];
        records[1].receive_ms = Some(701.0);
        let (metrics, errors) = assess(&spec(), counters(), &records, 1.0);
        assert!(!errors.is_empty());
        assert_eq!(metrics.missed_deadlines, 1);
        assert_eq!(metrics.p95_latency_ms, Some(286.95));
    }

    #[test]
    fn duplicate_or_missing_receipt_identity_is_rejected() {
        let session = Session::new(&spec());
        session.start().unwrap();
        let packet =
            serde_json::json!({ "liveAcceptance": { "run_id": session.run_id, "sequence": 2 } });
        assert!(session.receive(packet.to_string(), Instant::now()).is_err());
    }

    #[test]
    fn untagged_stats_are_ignored_but_other_messages_are_rejected() {
        let session = Session::new(&spec());
        session
            .receive("{\"type\":\"stats\"}".into(), Instant::now())
            .unwrap();
        for text in [
            "{\"type\":\"update\",\"image\":\"extra image\"}",
            "{\"type\":\"unknown\"}",
            "{\"type\":\"project_list\"}",
            "{}",
            "{\"type\":\"stats\",\"liveAcceptance\":{}}",
        ] {
            let error = session
                .receive(text.into(), Instant::now())
                .expect_err("only ordinary stats and tagged updates belong in the live trace");
            // The actual receiver reports returned validation errors after stop too.
            session.fail_connection(error);
        }
        let log = session.ledger.lock().unwrap();
        assert_eq!(log.received_count, 0);
        assert_eq!(log.errors.len(), 5);
    }

    #[test]
    fn png_is_checked_against_the_production_cpu_drawing() {
        let drawing = Drawing {
            polygons: Vec::new(),
            is_dirty: false,
            fitness: 99.0,
        };
        let white = crate::encode_rgba_as_png(&[255; 16], 2, 2);
        let black = crate::encode_rgba_as_png(&[0; 16], 2, 2);
        assert!(validate_png(&white, &drawing, 2, 2).is_ok());
        assert!(validate_png(&white, &drawing, 3, 2).is_err());
        assert!(validate_png(&black, &drawing, 2, 2).is_err());
    }

    #[test]
    fn parser_freezes_deadline_and_bounds_work_before_startup() {
        let flags = ["--manifest", "manifest.json", "--output", "new"]
            .map(String::from)
            .to_vec();
        let parsed = Spec::parse(&flags).unwrap();
        assert_eq!(
            (
                parsed.batches,
                parsed.deadline_ms,
                parsed.publication_count()
            ),
            (256, 300, 128)
        );
        let mut invalid = flags;
        invalid.extend(["--batches".into(), "1026".into()]);
        assert!(Spec::parse(&invalid).is_err());
    }

    #[test]
    fn acquisition_maximum_is_telemetry_when_a_previous_winner_is_overtaken() {
        let mut records = [publication(1), publication(2)];
        records[0].acquisition.maximum_fitness_bits = 99.5f32.to_bits();
        records[0].acquisition.is_acquisition_maximum = false;
        let (_, errors) = assess(&spec(), counters(), &records, 1.0);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn terminal_publication_requires_both_maximum_score_and_geometry() {
        use artgen_backend_rust::gpu_evolver::benchmark::{BenchmarkControl, BenchmarkPackedState};
        let parent = BenchmarkPackedState {
            polygon_count: 0,
            fitness_bits: 99.0f32.to_bits(),
            mutation_scale_bits: 1.0f32.to_bits(),
            stagnation_counter: 0,
            rng_state: 7,
            polygons: Vec::new(),
        };
        let mut snapshot = BenchmarkSnapshot {
            parents: vec![parent],
            offspring: Vec::new(),
            parent_total_errors: vec![0],
            parent_framebuffers: vec![Vec::new()],
            offspring_errors: Vec::new(),
            chain_update_flags: Vec::new(),
            fitness_packed: Vec::new(),
            control: BenchmarkControl {
                new_best_found: 0,
                best_chain_id: 0,
                best_fitness_bits: 99.0f32.to_bits(),
            },
            counters: counters(),
            validation_max_tile_occupancy: None,
        };
        let drawing = Drawing {
            polygons: Vec::new(),
            is_dirty: false,
            fitness: 99.0,
        };
        assert!(terminal_matches(&drawing, &snapshot));
        snapshot.parents[0].fitness_bits = 99.5f32.to_bits();
        assert!(!terminal_matches(&drawing, &snapshot));
        snapshot.parents[0].fitness_bits = 99.0f32.to_bits();
        snapshot.parents[0].polygon_count = 1;
        snapshot.parents[0].polygons.push([0; 4]);
        assert!(!terminal_matches(&drawing, &snapshot));
    }

    #[test]
    fn raw_receipt_artifact_preserves_actual_payload_and_content_hash() {
        let output = std::env::temp_dir().join(format!("artgen-live-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&output).unwrap();
        let result = (|| -> Result<()> {
            let session = Session::new(&spec());
            session.ledger.lock().unwrap().received[0] = Some("{\"actual\":1}".into());
            let inventory = session.save_receipts(&output)?;
            let bytes = fs::read(output.join("receipt-000.jsonl"))?;
            ensure!(bytes == b"{\"actual\":1}\n", "raw receipt changed");
            ensure!(
                inventory["receipt-000.jsonl"]["sha256"] == fixtures::sha256(&bytes),
                "wrong artifact hash"
            );
            ensure!(
                inventory["receipt-000.jsonl"]["bytes"] == bytes.len(),
                "wrong artifact size"
            );
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(output);
        assert!(result.is_ok(), "{result:?}");
        assert!(cleanup.is_ok(), "{cleanup:?}");
    }
}

fn initial_ws_state(input: &LoadedFixture, drawing: &Drawing) -> Result<crate::SharedWsState> {
    let (w, h) = (input.fixture.width as usize, input.fixture.height as usize);
    let mut pixels = vec![0; input.rgba.len()];
    drawing.draw(&mut pixels, w, h, Rasterizer::HalfSpace);
    Ok(Arc::new((
        Mutex::new(crate::WsState {
            ref_png: crate::encode_rgba_as_png(&input.rgba, w, h),
            best_png: crate::encode_rgba_as_png(&pixels, w, h),
            fitness: drawing.fitness,
            polygons: drawing.polygons.len(),
            drawing_json: serde_json::to_string(drawing)?,
            image_width: input.fixture.width,
            image_height: input.fixture.height,
            target_resolution: input.manifest.resolution,
            mutation_params: input.manifest.params.clone(),
            ..Default::default()
        }),
        Condvar::new(),
    )))
}

fn terminal_matches(drawing: &Drawing, snapshot: &BenchmarkSnapshot) -> bool {
    let Some(maximum) = snapshot
        .parents
        .iter()
        .map(|parent| f32::from_bits(parent.fitness_bits))
        .max_by(f32::total_cmp)
    else {
        return false;
    };
    let packed = drawing_to_gpu(drawing, 0);
    snapshot.parents.iter().any(|parent| {
        parent.fitness_bits == maximum.to_bits()
            && parent.fitness_bits == drawing.fitness.to_bits()
            && parent.polygon_count == packed.polygon_count
            && parent
                .polygons
                .iter()
                .zip(&packed.polygons)
                .all(|(source, actual)| *source == actual.data)
    })
}

fn seed_pipeline_cache(
    source: &Path,
    output: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut files = serde_json::Map::new();
    let mut total_bytes = 0;
    for (index, entry) in fs::read_dir(source)?.enumerate() {
        ensure!(index < 4096, "cache seed directory exceeds inventory limit");
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with("pipeline_cache_")
            || Path::new(name).extension().is_none_or(|ext| ext != "bin")
        {
            continue;
        }
        ensure!(
            entry.file_type()?.is_file(),
            "pipeline cache seed is not a regular file"
        );
        ensure!(files.len() < 16, "too many pipeline cache seeds");
        let bytes = fixtures::read_bounded(&entry.path(), 64 * 1024 * 1024)?;
        total_bytes += bytes.len();
        ensure!(
            total_bytes <= MAX_RETAINED_BYTES,
            "pipeline cache seed byte limit"
        );
        fs::write(output.join(name), &bytes)?;
        files.insert(
            name.to_owned(),
            serde_json::json!({ "bytes": bytes.len(), "sha256": fixtures::sha256(&bytes) }),
        );
    }
    Ok(files)
}

#[derive(Serialize)]
struct Metrics {
    previews_due: usize,
    previews_sent: usize,
    previews_received: usize,
    missed_deadlines: usize,
    p95_latency_ms: Option<f64>,
}

fn assess(
    spec: &Spec,
    counters: BenchmarkCounters,
    records: &[Publication],
    elapsed: f64,
) -> (Metrics, Vec<String>) {
    let mut errors = Vec::with_capacity(records.len() + 4);
    let iterations = u64::from(spec.batches) * 64;
    let evaluations = iterations * 4 * 64;
    if counters
        != (BenchmarkCounters {
            submitted_batches: u64::from(spec.batches),
            completed_batches: u64::from(spec.batches),
            submitted_iterations: iterations,
            completed_iterations: iterations,
            submitted_evaluations: evaluations,
            completed_evaluations: evaluations,
        })
    {
        errors.push("submitted/completed work mismatch".into());
    }
    let mut metrics = Metrics {
        previews_due: records.len(),
        previews_sent: records.iter().filter(|r| r.send_ms.is_some()).count(),
        previews_received: records.iter().filter(|r| r.receive_ms.is_some()).count(),
        missed_deadlines: 0,
        p95_latency_ms: None,
    };
    if [
        metrics.previews_due,
        metrics.previews_sent,
        metrics.previews_received,
    ]
    .iter()
    .any(|count| *count != spec.publication_count())
    {
        errors.push("missing due, sent or received publications".into());
    }
    if !elapsed.is_finite() || elapsed <= 0.0 {
        errors.push("invalid measured elapsed time".into());
    }
    let mut latencies = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        if let Err(error) = assess_publication(spec, record, index, elapsed) {
            errors.push(format!("publication {}: {error:#}", index + 1));
        }
        if let Some(received) = record.receive_ms {
            let latency = received - record.due_ms;
            if latency > f64::from(spec.deadline_ms) {
                metrics.missed_deadlines += 1;
            }
            if latency.is_finite() && latency >= 0.0 {
                latencies.push(latency);
            }
        }
    }
    if metrics.missed_deadlines > 0 {
        errors.push("publication deadline missed".into());
    }
    metrics.p95_latency_ms = percentile95(latencies);
    (metrics, errors)
}

fn percentile95(mut latencies: Vec<f64>) -> Option<f64> {
    let last = latencies.len().checked_sub(1)?;
    latencies.sort_by(f64::total_cmp);
    let position = f64::from(u32::try_from(last).ok()?) * 0.95;
    let scaled = last.checked_mul(95)?;
    let low = scaled / 100;
    let high = scaled.div_ceil(100);
    Some(latencies[low] + (latencies[high] - latencies[low]) * position.fract())
}

fn evaluation_rate(completed_evaluations: u64, elapsed_seconds: f64) -> Result<f64> {
    let evaluations = u32::try_from(completed_evaluations)
        .context("completed evaluations exceed bounded live workload")?;
    Ok(f64::from(evaluations) / elapsed_seconds)
}

fn assess_publication(spec: &Spec, record: &Publication, index: usize, elapsed: f64) -> Result<()> {
    let batches = (index + 1) as u64 * u64::from(spec.publish_every);
    ensure!(
        record.sequence == index + 1
            && record.completed_batches == batches
            && record.completed_evaluations == batches * 16384,
        "wrong completed-work schedule"
    );
    let send = record.send_ms.context("missing successful send")?;
    let receive = record.receive_ms.context("missing client receipt")?;
    ensure!(
        [record.due_ms, send, receive]
            .iter()
            .all(|time| time.is_finite())
            && record.due_ms >= 0.0
            && record.due_ms <= send
            && send <= receive
            && receive <= elapsed * 1000.0,
        "invalid publication timing"
    );
    ensure!(
        record.payload_valid,
        "missing or inconsistent publication payload"
    );
    ensure!(
        record.acquisition.completed_batches <= record.completed_batches
            && record.acquisition.fitness_bits == record.fitness_percent.to_bits(),
        "inconsistent original acquisition binding"
    );
    Ok(())
}

impl Run {
    fn finish(&mut self) -> Result<serde_json::Value> {
        let log = self
            .session
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        self.errors.extend(log.errors.iter().cloned());
        let counters = self
            .engine
            .as_ref()
            .map_or_else(BenchmarkCounters::default, |engine| engine.counters);
        let (metrics, errors) = assess(&self.spec, counters, &log.records, self.elapsed_seconds);
        self.errors.extend(errors);
        let value = if let Some(engine) = &mut self.engine {
            engine.errors.extend(self.errors.iter().cloned());
            trial::finish_record(engine);
            trial_record_value(engine)?
        } else {
            serde_json::json!({})
        };
        let mut errors = self
            .engine
            .as_ref()
            .map_or_else(|| self.errors.clone(), |engine| engine.errors.clone());
        if !self.terminal_matches_maximum && !errors.iter().any(|e| e.contains("terminal")) {
            errors.push("terminal maximum publication is unverified".into());
        }
        drop(log);
        let trace_artifact = self.finish_latency_trace(&mut errors);
        let log = self
            .session
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        let valid = errors.is_empty();
        let extras = serde_json::json!({ "protocol_version": 1, "live_protocol_version": 2, "mode": "live_acceptance",
            "status": if valid { "completed" } else { "invalid" }, "correctness": valid,
            "error_count": errors.len(), "errors": errors, "case_id": self.spec.case_id,
            "seed": self.spec.seed, "batches": self.spec.batches, "warmup_batches": self.spec.warmup_batches,
            "publish_every": self.spec.publish_every, "deadline_ms": self.spec.deadline_ms,
            "run_id": self.session.run_id, "elapsed_seconds": self.elapsed_seconds,
            "drain_seconds": self.drain_seconds, "counters": counters,
            "evals_per_second": if valid { Some(evaluation_rate(counters.completed_evaluations, self.elapsed_seconds)?) } else { None },
            "publications": log.records, "terminal_matches_maximum": self.terminal_matches_maximum,
            "cache_seed": self.cache_seed, "receipt_artifacts": self.receipt_artifacts,
            "historical_score_policy": "ordinary_mapped_header_and_acquisition_membership",
            "historical_physical_fitness_independently_verified": false });
        self.finish_record_metadata(value, &extras, metrics, trace_artifact)
    }

    fn finish_latency_trace(&self, errors: &mut Vec<String>) -> Option<serde_json::Value> {
        if self.spec.latency_trace {
            match self.session.save_trace(errors) {
                Ok(Some(artifact)) => {
                    if let Some(error) = artifact["error"].as_str() {
                        errors.push(format!("latency trace: {error}"));
                    }
                    Some(artifact)
                }
                Ok(None) => {
                    errors.push("enabled latency trace missing".into());
                    None
                }
                Err(error) => {
                    errors.push(format!("save latency trace: {error:#}"));
                    None
                }
            }
        } else {
            None
        }
    }

    fn finish_record_metadata(
        &self,
        mut value: serde_json::Value,
        extras: &serde_json::Value,
        metrics: Metrics,
        trace_artifact: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let object = value
            .as_object_mut()
            .context("trial record is not an object")?;
        object.extend(
            extras
                .as_object()
                .context("live record is not an object")?
                .clone(),
        );
        object.extend(
            serde_json::to_value(metrics)?
                .as_object()
                .context("metrics not an object")?
                .clone(),
        );
        if self.spec.latency_trace {
            object.insert("mode".into(), "live_latency_diagnostic".into());
            object.insert("score_eligible".into(), false.into());
            object.insert("latency_trace".into(), trace_artifact.into());
        }
        Ok(value)
    }
}

fn trial_record_value(record: &TrialRecord) -> Result<serde_json::Value> {
    // Format typed f32 fields before Value would widen them to f64.
    let bytes = serde_json::to_vec(record)?;
    Ok(serde_json::from_slice(&bytes)?)
}
