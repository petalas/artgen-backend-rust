# Implementation bounds

The first slice owns fixture import, the standalone trial lifecycle and local
subprocess supervision. Production batch ordering, population settings, precision
and seed defaults remain unchanged. The integer reference-byte correction fixes
a numerical inconsistency before a campaign baseline exists. A separate source
repair removes post-selection cleanup that changed already-evaluated geometry;
selected candidate geometry/count and cached error must describe the same image.

New input boundaries reject unsupported versions, duplicate projects, traversal,
hash mismatches, malformed drawings, silently sanitized parameters, truncation,
unqualified similarity and inadequate GPU buffer/dispatch capacity. Every timed
run has a finite completed-work budget and a parent timeout. New GPU readbacks
have a 120-second poll/mapping bound. Python file, process and trial counts are
capped in the supervisor; its tests exercise timeout and abnormal exits.

## Power of Ten deviations

Rule: 3, fix memory capacity before steady operation.
Location: `src/benchmark/`, `src/gpu_evolver/benchmark.rs`, `scripts/evolution-bench.py`.
Reason: wgpu command submission and mapping, Serde JSON and Python subprocess
supervision allocate through their runtimes. Replacing the production engine's
allocation behavior would change the path being measured.
Scope: one worker at a time, existing production batches inside measurement;
all additional snapshots, scalar buffers and JSON serialization outside it.
Compensating control: at most 1024 pixels per image axis and 1000 packed triangles;
validated device capacities; one pending production batch; bounded input files,
warmup, measured batches, trial counts and output sizes.
Verification: fixture corruption/limit tests, scalar boundary tests, supervisor
timeout/accounting tests and real main-thread GPU validation.
Follow-up: profile production allocations independently before proposing an
allocation optimization; preserve this measurement contract.

Rule: 4, functions fit a page.
Location: existing `GpuEvolver::run_batch`, pipeline construction and completion.
Reason: splitting existing production orchestration while instrumenting its
accounting would expand the change and undermine comparison with the same path.
Scope: only per-batch submitted/completed accounting is added to those functions.
Compensating control: the new benchmark lifecycle separates prepare, measure,
snapshot and comparison, with checked final drain counters.
Verification: completion unit tests and repeated same-seed GPU trials.
Follow-up: restructure the engine only as an isolated, measured later change.

Rule: 7, bounded and checked returns.
Location: existing production GPU methods called by the benchmark.
Reason: those methods panic on device failures and some existing polls have no
internal timeout. Replacing the application's error model is beyond this slice.
Scope: only the existing engine calls inside an isolated worker.
Compensating control: parent subprocess timeout and exit-status checks reject
abnormal exits and missing results; scoped GPU errors also invalidate trials.
Verification: synthetic worker crash, missing-result and timeout tests plus GPU
device-scope checks on the real path.
Follow-up: adopt fallible bounded GPU methods in a separate application change.

Rule: 8/9, framework macros and callbacks.
Location: Serde derives, Rust formatting/test macros, wgpu mapping callbacks,
Python argparse, JSON and unittest.
Reason: these platform interfaces require macro expansion or indirect callbacks.
Scope: compiler-generated serialization, formatting and bounded host callbacks;
no runtime code generation or new shader interfaces.
Compensating control: fixed typed records, explicit callback ownership and time
limits, no callback recursion, scalar analytical and public-interface tests.
Verification: compiler, Clippy, unit tests and supervised real GPU execution.
Follow-up: retain the typed boundaries when extending the protocol.

Rule: 10, maximum diagnostics.
Location: repository-wide Rust Clippy policy and Python runtime.
Reason: the repository already has Clippy warnings; making every existing build
pedantic-clean would require unrelated changes. Python has no Rust-style compile
or fixed-memory verification.
Scope: new modules and touched entry points are inspected with Clippy all/pedantic
and warning-enabled Python unittest; remaining bounded numeric conversions are
reviewed against the input limits above. No warnings are suppressed to hide a
failed behavior check.
Compensating control: fixed numeric ranges, explicit integer validation, retained
diagnostic logs and numerical/replay tests.
Verification: commands and output retained in the local decision trail.
Follow-up: a repository-wide diagnostics policy is a separate change.

The remaining numeric cast diagnostics in the scalar renderer and arithmetic
envelope are intentional GPU arithmetic or checked-range conversions. Decoded
vertices are at most 65535, dimensions at most 1024, and channels are floored and
clamped to 0..255 before conversion. Large integer error totals deliberately
round to f32 exactly where the shader does. The envelope's f64-to-f32 conversions
round outward with adjacent representable values. These diagnostics stay visible;
analytical tests and exact cached-error/replay gates cover the conversion rules.
