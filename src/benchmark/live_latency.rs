//! Private, bounded evidence for one unscored live diagnostic. No wire fields.
use std::{
    io::Write,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use super::{Session, MAX_ARTIFACT_BYTES, MAX_PUBLICATIONS, MAX_TURNS};

pub(super) const FILE: &str = "latency-trace-v1.json";
// Fixed keys and at most twenty decimal digits per numeric field fit 4096
// bytes per row. Reserve room for every row before accepting a diagnostic.
const MAX_ROW_JSON_BYTES: usize = 4096;

#[derive(Clone, Copy)]
pub(crate) struct Clock(pub(super) Instant);

impl Clock {
    pub(crate) fn at(self, instant: Instant) -> u64 {
        u64::try_from(instant.duration_since(self.0).as_nanos())
            .expect("bounded live trace exceeds u64 nanoseconds")
    }

    pub(crate) fn now(self) -> u64 {
        self.at(Instant::now())
    }
}

#[derive(Clone, Copy, Default, Serialize)]
pub(super) struct PublicationTrace {
    pub sequence: usize,
    pub completed_batches: u64,
    pub completed_evaluations: u64,
    pub payload_bytes: Option<usize>,
    pub queue_depth: Option<usize>,
    pub due_ns: Option<u64>,
    pub render_done_ns: Option<u64>,
    pub png_done_ns: Option<u64>,
    pub drawing_json_done_ns: Option<u64>,
    pub ws_lock_attempt_ns: Option<u64>,
    pub ws_lock_acquired_ns: Option<u64>,
    pub ledger_lock_attempt_ns: Option<u64>,
    pub ledger_lock_acquired_ns: Option<u64>,
    pub hashes_done_ns: Option<u64>,
    pub enqueue_commit_ns: Option<u64>,
    pub receive_ns: Option<u64>,
}

impl PublicationTrace {
    fn validate(&self, sender_turns: &[SenderTurn]) -> Result<()> {
        let stages = [
            self.due_ns,
            self.render_done_ns,
            self.png_done_ns,
            self.drawing_json_done_ns,
            self.ws_lock_attempt_ns,
            self.ws_lock_acquired_ns,
            self.ledger_lock_attempt_ns,
            self.ledger_lock_acquired_ns,
            self.hashes_done_ns,
            self.enqueue_commit_ns,
        ];
        ensure!(
            stages.iter().all(Option::is_some),
            "trace producer stages incomplete"
        );
        ordered(&stages)?;
        let mut sends = sender_turns
            .iter()
            .filter(|turn| turn.publication_sequence == Some(self.sequence));
        let turn = sends
            .next()
            .context("trace publication has no sender turn")?;
        ensure!(sends.next().is_none(), "trace duplicate publication send");
        ensure!(
            turn.queue_head_sequence == Some(self.sequence) && turn.image_gate_passed == Some(true),
            "trace dequeue identity mismatch"
        );
        let stages = [
            turn.dequeue_ns,
            turn.legacy_sendmarker_ns,
            turn.message_value_done_ns,
            turn.image_cadence_reset_ns,
            turn.wire_json_done_ns,
            turn.write_begin_ns,
            turn.write_return_ns,
        ];
        ensure!(
            stages.iter().all(Option::is_some),
            "trace sender stages incomplete"
        );
        ensure!(
            self.enqueue_commit_ns <= turn.dequeue_ns,
            "trace dequeued before enqueue"
        );
        let receive = self.receive_ns.context("trace receipt missing")?;
        ensure!(
            turn.write_begin_ns.is_some_and(|begin| begin <= receive),
            "trace receipt before write begin"
        );
        // A receiver can finish before the sender resumes from ws.send.
        Ok(())
    }
}

#[derive(Serialize)]
pub(crate) struct SenderTurn {
    #[serde(skip)]
    clock: Clock,
    pub turn: usize,
    pub publication_sequence: Option<usize>,
    pub turn_start_ns: u64,
    pub poll_target_ns: Option<u64>,
    pub sleep_begin_ns: Option<u64>,
    pub requested_sleep_ns: Option<u64>,
    pub wake_ns: Option<u64>,
    pub ws_lock_attempt_ns: Option<u64>,
    pub ws_lock_acquired_ns: Option<u64>,
    pub eligibility_check_ns: Option<u64>,
    pub image_eligible_at_ns: Option<u64>,
    pub image_gate_passed: Option<bool>,
    pub ledger_lock_attempt_ns: Option<u64>,
    pub ledger_lock_acquired_ns: Option<u64>,
    pub queue_head_sequence: Option<usize>,
    pub queue_depth: Option<usize>,
    pub dequeue_ns: Option<u64>,
    pub legacy_sendmarker_ns: Option<u64>,
    pub message_value_done_ns: Option<u64>,
    pub image_cadence_reset_ns: Option<u64>,
    pub wire_json_done_ns: Option<u64>,
    pub write_begin_ns: Option<u64>,
    pub write_return_ns: Option<u64>,
}

impl SenderTurn {
    pub(crate) fn now(&self) -> u64 {
        self.clock.now()
    }

    pub(crate) fn at(&self, instant: Instant) -> u64 {
        self.clock.at(instant)
    }

    pub(crate) fn image_is_throttled(&mut self, last_image: Instant, interval: Duration) -> bool {
        let check = Instant::now();
        self.eligibility_check_ns = Some(self.at(check));
        self.image_eligible_at_ns = Some(self.at(last_image + interval));
        let gated = check.duration_since(last_image) < interval;
        self.image_gate_passed = Some(!gated);
        gated
    }

    fn new(clock: Clock, turn: usize) -> Self {
        Self {
            clock,
            turn,
            turn_start_ns: clock.now(),
            publication_sequence: None,
            poll_target_ns: None,
            sleep_begin_ns: None,
            requested_sleep_ns: None,
            wake_ns: None,
            ws_lock_attempt_ns: None,
            ws_lock_acquired_ns: None,
            eligibility_check_ns: None,
            image_eligible_at_ns: None,
            image_gate_passed: None,
            ledger_lock_attempt_ns: None,
            ledger_lock_acquired_ns: None,
            queue_head_sequence: None,
            queue_depth: None,
            dequeue_ns: None,
            legacy_sendmarker_ns: None,
            message_value_done_ns: None,
            image_cadence_reset_ns: None,
            wire_json_done_ns: None,
            write_begin_ns: None,
            write_return_ns: None,
        }
    }

    fn validate(&self) -> Result<()> {
        ordered(&[
            Some(self.turn_start_ns),
            self.sleep_begin_ns,
            self.wake_ns,
            self.ws_lock_attempt_ns,
            self.ws_lock_acquired_ns,
            self.eligibility_check_ns,
            self.ledger_lock_attempt_ns,
            self.ledger_lock_acquired_ns,
            self.dequeue_ns,
            self.legacy_sendmarker_ns,
            self.message_value_done_ns,
            self.image_cadence_reset_ns,
            self.wire_json_done_ns,
            self.write_begin_ns,
            self.write_return_ns,
        ])?;
        if self.write_return_ns.is_some() {
            ensure!(
                [
                    self.poll_target_ns,
                    self.sleep_begin_ns,
                    self.requested_sleep_ns,
                    self.wake_ns,
                    self.ws_lock_attempt_ns,
                    self.ws_lock_acquired_ns,
                    self.eligibility_check_ns,
                    self.image_eligible_at_ns,
                    self.message_value_done_ns,
                    self.wire_json_done_ns,
                    self.write_begin_ns,
                ]
                .iter()
                .all(Option::is_some),
                "trace completed turn stages incomplete"
            );
        }
        if self.image_gate_passed == Some(false) {
            ensure!(
                self.ledger_lock_attempt_ns.is_none() && self.dequeue_ns.is_none(),
                "throttled turn touched publication ledger"
            );
        }
        if let (Some(check), Some(eligible_at), Some(passed)) = (
            self.eligibility_check_ns,
            self.image_eligible_at_ns,
            self.image_gate_passed,
        ) {
            ensure!(
                passed == (check >= eligible_at),
                "trace image gate result mismatch"
            );
        }
        if self.image_gate_passed == Some(true) {
            ensure!(
                self.ledger_lock_attempt_ns.is_some()
                    && self.ledger_lock_acquired_ns.is_some()
                    && self.queue_depth.is_some(),
                "trace eligible turn queue observation missing"
            );
        }
        Ok(())
    }
}

pub(super) struct Trace {
    pub publications: Vec<PublicationTrace>,
    pub sender_turns: Vec<SenderTurn>,
    sender_taken: bool,
    sender_returned: bool,
    publication_limit: usize,
}

impl Trace {
    pub fn new(publication_limit: usize) -> Self {
        assert!(publication_limit <= MAX_PUBLICATIONS);
        let bytes = publication_limit
            .checked_mul(size_of::<PublicationTrace>())
            .and_then(|n| n.checked_add(MAX_TURNS * size_of::<SenderTurn>()))
            .expect("trace capacity arithmetic overflow");
        assert!(bytes <= MAX_ARTIFACT_BYTES, "trace memory byte limit");
        assert!(
            (publication_limit + MAX_TURNS) * MAX_ROW_JSON_BYTES + 8192 <= MAX_ARTIFACT_BYTES,
            "trace rows exceed artifact byte budget"
        );
        Self {
            publications: Vec::with_capacity(publication_limit),
            sender_turns: Vec::with_capacity(MAX_TURNS),
            sender_taken: false,
            sender_returned: false,
            publication_limit,
        }
    }

    pub fn retain(&mut self, row: PublicationTrace) -> Result<()> {
        ensure!(
            self.publications.len() < self.publication_limit,
            "trace publication capacity exhausted"
        );
        ensure!(
            row.sequence == self.publications.len() + 1,
            "trace publication identity skipped"
        );
        self.publications.push(row);
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.sender_returned, "trace sender evidence not returned");
        ensure!(
            self.publications.len() == self.publication_limit,
            "trace publications incomplete"
        );
        for (index, row) in self.publications.iter().enumerate() {
            ensure!(
                row.sequence == index + 1,
                "trace publication identity mismatch"
            );
            row.validate(&self.sender_turns)?;
            if index > 0 {
                ensure!(
                    self.publications[index - 1].enqueue_commit_ns <= row.due_ns,
                    "trace producer order across publications violated"
                );
                ensure!(
                    self.publications[index - 1].receive_ns <= row.receive_ns,
                    "trace receiver order across publications violated"
                );
            }
        }
        for (index, turn) in self.sender_turns.iter().enumerate() {
            ensure!(turn.turn == index + 1, "trace turn identity mismatch");
            turn.validate()?;
            if let Some(sequence) = turn.publication_sequence {
                ensure!(
                    (1..=self.publication_limit).contains(&sequence),
                    "trace unknown sent publication"
                );
            }
            if index > 0 {
                ensure!(
                    self.sender_turns[index - 1]
                        .write_return_ns
                        .is_none_or(|before| before <= turn.turn_start_ns),
                    "trace sender order across turns violated"
                );
            }
        }
        Ok(())
    }
}

/// The sender owns these rows until every handler exit transfers them once.
pub(crate) struct SenderTrace {
    session: Arc<Session>,
    clock: Clock,
    rows: Vec<SenderTurn>,
}

impl SenderTrace {
    pub(crate) fn begin_turn(&mut self) -> Result<&mut SenderTurn> {
        ensure!(
            self.rows.len() < MAX_TURNS,
            "trace sender capacity exhausted"
        );
        self.rows
            .push(SenderTurn::new(self.clock, self.rows.len() + 1));
        self.rows.last_mut().context("missing trace sender turn")
    }
}

impl Drop for SenderTrace {
    fn drop(&mut self) {
        let result = (|| -> Result<()> {
            let mut log = self
                .session
                .ledger
                .lock()
                .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
            let trace = log
                .trace
                .as_mut()
                .context("trace disabled during sender transfer")?;
            ensure!(!trace.sender_returned, "duplicate trace sender transfer");
            trace.sender_turns = std::mem::take(&mut self.rows);
            trace.sender_returned = true;
            Ok(())
        })();
        if let Err(error) = result {
            self.session
                .fail(format!("trace sender transfer: {error:#}"));
        }
    }
}

impl Session {
    pub(crate) fn trace_clock(&self) -> Option<Clock> {
        self.spec.latency_trace.then_some(Clock(self.wall_start))
    }

    pub(crate) fn sender_trace(self: &Arc<Self>) -> Result<Option<SenderTrace>> {
        let Some(clock) = self.trace_clock() else {
            return Ok(None);
        };
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        let trace = log.trace.as_mut().context("missing enabled trace")?;
        ensure!(!trace.sender_taken, "trace sender already owned");
        trace.sender_taken = true;
        Ok(Some(SenderTrace {
            session: self.clone(),
            clock,
            rows: std::mem::take(&mut trace.sender_turns),
        }))
    }

    pub(super) fn retain_partial_trace(&self, row: PublicationTrace) -> Result<()> {
        let mut log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        log.trace
            .as_mut()
            .context("missing enabled trace")?
            .retain(row)
    }

    pub(super) fn save_trace(&self, errors: &[String]) -> Result<Option<serde_json::Value>> {
        let log = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("live ledger poisoned"))?;
        let Some(trace) = &log.trace else {
            return Ok(None);
        };
        let validation = trace.validate();
        let trace_error = validation.as_ref().err().map(|error| format!("{error:#}"));
        let all_errors: Vec<&str> = errors
            .iter()
            .chain(log.errors.iter())
            .map(String::as_str)
            .chain(trace_error.as_deref())
            .collect();
        let sidecar = Sidecar {
            trace_version: 1,
            mode: "live_latency_diagnostic",
            score_eligible: false,
            run_id: &self.run_id,
            clock: "session_wall_start",
            measurement_start_ns: log.start.map(|start| Clock(self.wall_start).at(start)),
            limits: Limits {
                publication_rows: self.spec.publication_count(),
                sender_turn_rows: MAX_TURNS,
                max_bytes: MAX_ARTIFACT_BYTES,
            },
            complete: all_errors.is_empty(),
            errors: all_errors,
            publications: &trace.publications,
            sender_turns: &trace.sender_turns,
        };
        sidecar
            .save(&self.spec.output, trace_error.as_deref())
            .map(Some)
    }
}

#[derive(Serialize)]
struct Limits {
    publication_rows: usize,
    sender_turn_rows: usize,
    max_bytes: usize,
}

#[derive(Serialize)]
struct Sidecar<'a> {
    trace_version: u32,
    mode: &'static str,
    score_eligible: bool,
    run_id: &'a str,
    clock: &'static str,
    measurement_start_ns: Option<u64>,
    limits: Limits,
    complete: bool,
    errors: Vec<&'a str>,
    publications: &'a [PublicationTrace],
    sender_turns: &'a [SenderTurn],
}

impl<'a> Sidecar<'a> {
    fn save(
        mut self,
        output: &std::path::Path,
        trace_error: Option<&'a str>,
    ) -> Result<serde_json::Value> {
        let mut buffer = BoundedBuffer(Vec::with_capacity(MAX_ARTIFACT_BYTES));
        let byte_error = if serde_json::to_writer(&mut buffer, &self).is_err() {
            // Keep every row. Full original errors remain in result.json; only
            // their sidecar copy can exhaust the remaining artifact budget.
            let error = "trace artifact byte limit; full errors retained in result.json";
            self.complete = false;
            self.errors = std::iter::once(error).chain(trace_error).collect();
            buffer.0.clear();
            serde_json::to_writer(&mut buffer, &self)?;
            Some(error)
        } else {
            None
        };
        let path = output.join(FILE);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&buffer.0)?;
        Ok(serde_json::json!({"file": FILE, "bytes": buffer.0.len(),
            "sha256": artgen_backend_rust::benchmark::fixtures::sha256(&buffer.0),
            "complete": self.complete, "error": byte_error.or(trace_error)}))
    }
}

struct BoundedBuffer(Vec<u8>);

impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_ARTIFACT_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("trace artifact byte limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn ordered(stages: &[Option<u64>]) -> Result<()> {
    ensure!(stages.len() <= 32, "trace stage list exceeds bound");
    let mut previous = None;
    for stage in stages.iter().flatten() {
        ensure!(
            previous.is_none_or(|before| before <= *stage),
            "trace own-thread stage order violated"
        );
        previous = Some(*stage);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_trace() -> Trace {
        let clock = Clock(Instant::now());
        let row = PublicationTrace {
            sequence: 1,
            completed_batches: 2,
            completed_evaluations: 32768,
            payload_bytes: Some(100),
            queue_depth: Some(1),
            due_ns: Some(1),
            render_done_ns: Some(2),
            png_done_ns: Some(3),
            drawing_json_done_ns: Some(4),
            ws_lock_attempt_ns: Some(5),
            ws_lock_acquired_ns: Some(6),
            ledger_lock_attempt_ns: Some(7),
            ledger_lock_acquired_ns: Some(8),
            hashes_done_ns: Some(9),
            enqueue_commit_ns: Some(10),
            receive_ns: Some(26),
        };
        let mut turn = SenderTurn::new(clock, 1);
        turn.turn_start_ns = 11;
        turn.poll_target_ns = Some(12);
        turn.sleep_begin_ns = Some(12);
        turn.requested_sleep_ns = Some(1);
        turn.wake_ns = Some(13);
        turn.ws_lock_attempt_ns = Some(14);
        turn.ws_lock_acquired_ns = Some(15);
        turn.eligibility_check_ns = Some(16);
        turn.image_eligible_at_ns = Some(16);
        turn.image_gate_passed = Some(true);
        turn.ledger_lock_attempt_ns = Some(17);
        turn.ledger_lock_acquired_ns = Some(18);
        turn.queue_head_sequence = Some(1);
        turn.queue_depth = Some(1);
        turn.dequeue_ns = Some(19);
        turn.publication_sequence = Some(1);
        turn.legacy_sendmarker_ns = Some(20);
        turn.message_value_done_ns = Some(21);
        turn.image_cadence_reset_ns = Some(22);
        turn.wire_json_done_ns = Some(23);
        turn.write_begin_ns = Some(24);
        turn.write_return_ns = Some(30);
        let mut trace = Trace::new(1);
        trace.retain(row).unwrap();
        trace.sender_turns.push(turn);
        trace.sender_returned = true;
        trace
    }

    #[test]
    fn trace_stage_names_identity_and_order_are_stable() {
        let mut trace = synthetic_trace();
        assert!(trace.validate().is_ok());
        let row = serde_json::to_value(trace.publications[0]).unwrap();
        for (index, name) in [
            "due_ns",
            "render_done_ns",
            "png_done_ns",
            "drawing_json_done_ns",
            "ws_lock_attempt_ns",
            "ws_lock_acquired_ns",
            "ledger_lock_attempt_ns",
            "ledger_lock_acquired_ns",
            "hashes_done_ns",
            "enqueue_commit_ns",
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(row[name], index + 1);
        }
        assert_eq!(row.as_object().unwrap().len(), 16);
        let turn = serde_json::to_value(&trace.sender_turns[0]).unwrap();
        assert_eq!(turn.as_object().unwrap().len(), 23);
        assert_eq!(turn["publication_sequence"], 1);
        assert_eq!(turn["queue_head_sequence"], 1);
        assert_eq!(turn["legacy_sendmarker_ns"], 20);
        assert_eq!(turn["image_cadence_reset_ns"], 22);
        trace.publications[0].png_done_ns = Some(1);
        assert!(trace.validate().is_err());
        trace.publications[0].png_done_ns = Some(3);
        trace.sender_turns[0].wire_json_done_ns = Some(21);
        assert!(trace.validate().is_err());
    }

    #[test]
    fn receipt_before_write_return_is_valid_but_before_write_begin_is_invalid() {
        let mut trace = synthetic_trace();
        assert!(trace.validate().is_ok());
        trace.publications[0].receive_ns = Some(23);
        assert!(trace.validate().is_err());
        trace.publications[0].receive_ns = Some(26);
        trace.sender_turns[0].publication_sequence = Some(2);
        assert!(trace.validate().is_err());
    }

    #[test]
    fn trace_capacity_exhaustion_keeps_all_prior_rows_without_growth() {
        let mut trace = synthetic_trace();
        let mut extra = trace.publications[0];
        extra.sequence = 2;
        let capacity = trace.publications.capacity();
        assert!(trace.retain(extra).is_err());
        assert_eq!(trace.publications.len(), 1);
        assert_eq!(trace.publications[0].sequence, 1);
        assert_eq!(trace.publications.capacity(), capacity);

        let args = [
            "--manifest",
            "manifest.json",
            "--output",
            "new",
            "--batches",
            "2",
            "--latency-trace",
            "v1",
        ]
        .map(String::from);
        let mut spec = super::super::Spec::parse(&args).unwrap();
        spec.output =
            std::env::temp_dir().join(format!("artgen-trace-cap-{}", uuid::Uuid::new_v4()));
        let session = Arc::new(Session::new(&spec));
        let mut sender = session.sender_trace().unwrap().unwrap();
        let capacity = sender.rows.capacity();
        for _ in 0..MAX_TURNS {
            sender.begin_turn().unwrap();
        }
        let error = sender.begin_turn().err().expect("sender cap must fail");
        session.fail(error.to_string());
        assert_eq!(sender.rows.len(), MAX_TURNS);
        assert_eq!(sender.rows.capacity(), capacity);
        drop(sender);
        let log = session.ledger.lock().unwrap();
        assert_eq!(log.trace.as_ref().unwrap().sender_turns.len(), MAX_TURNS);
        assert_eq!(log.errors.len(), 1);
        assert!(session.expired());
        drop(log);
        std::fs::create_dir(&spec.output).unwrap();
        let result = (|| -> Result<()> {
            let artifact = session.save_trace(&[])?.context("missing capacity trace")?;
            ensure!(artifact["complete"] == false, "exhausted trace passed");
            let value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(spec.output.join(FILE))?)?;
            ensure!(
                value["sender_turns"].as_array().context("turns")?.len() == MAX_TURNS,
                "exhaustion lost rows"
            );
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(&spec.output);
        assert!(result.is_ok(), "{result:?}");
        assert!(cleanup.is_ok(), "{cleanup:?}");
    }

    #[test]
    fn fixed_rows_fit_the_reserved_artifact_budget_at_maximum_integers() {
        let trace = synthetic_trace();
        let rows = [
            serde_json::to_value(trace.publications[0]).unwrap(),
            serde_json::to_value(&trace.sender_turns[0]).unwrap(),
        ];
        for mut row in rows {
            for value in row.as_object_mut().unwrap().values_mut() {
                if value.is_number() {
                    *value = u64::MAX.into();
                }
            }
            assert!(serde_json::to_vec(&row).unwrap().len() <= MAX_ROW_JSON_BYTES);
        }
    }

    #[test]
    fn artifact_byte_exhaustion_retains_every_row_and_leaves_original_errors_owned_by_caller() {
        let trace = synthetic_trace();
        let original_error = "x".repeat(MAX_ARTIFACT_BYTES);
        let sidecar = Sidecar {
            trace_version: 1,
            mode: "live_latency_diagnostic",
            score_eligible: false,
            run_id: "test",
            clock: "session_wall_start",
            measurement_start_ns: Some(0),
            limits: Limits {
                publication_rows: 1,
                sender_turn_rows: MAX_TURNS,
                max_bytes: MAX_ARTIFACT_BYTES,
            },
            complete: false,
            errors: vec![&original_error],
            publications: &trace.publications,
            sender_turns: &trace.sender_turns,
        };
        let output =
            std::env::temp_dir().join(format!("artgen-trace-byte-cap-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&output).unwrap();
        let result = (|| -> Result<()> {
            let artifact = sidecar.save(&output, None)?;
            ensure!(
                artifact["complete"] == false && artifact["error"].is_string(),
                "byte exhaustion passed"
            );
            let bytes = std::fs::read(output.join(FILE))?;
            ensure!(
                bytes.len() <= MAX_ARTIFACT_BYTES,
                "artifact byte cap failed"
            );
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            ensure!(
                value["publications"].as_array().context("rows")?.len() == 1
                    && value["sender_turns"].as_array().context("turns")?.len() == 1,
                "byte exhaustion lost rows"
            );
            ensure!(
                original_error.len() == MAX_ARTIFACT_BYTES,
                "original error changed"
            );
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(output);
        assert!(result.is_ok(), "{result:?}");
        assert!(cleanup.is_ok(), "{cleanup:?}");
    }

    #[test]
    fn artifact_writer_enforces_the_byte_limit_without_partial_row_writes() {
        let mut writer = BoundedBuffer(vec![0; MAX_ARTIFACT_BYTES - 1]);
        assert!(writer.write_all(&[1, 2]).is_err());
        assert_eq!(writer.0.len(), MAX_ARTIFACT_BYTES - 1);
        writer.write_all(&[1]).unwrap();
        assert_eq!(writer.0.len(), MAX_ARTIFACT_BYTES);
        assert!(writer.write_all(&[2]).is_err());
    }
}
