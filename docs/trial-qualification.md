# Trial qualification and result accounting

`benchmark::qualification` owns the measured batch clock, submitted/completed
work predicate, canonical replay identity and timing-score calculation.
`benchmark::trial` owns isolated preparation, numerical qualification and
`TrialRecord`. `engine-lab` retains its resident device lifecycle, timestamp
sample gate and independent raw-pixel/fitness checks. The scalar renderer and
uncertainty audit remain independent of the GPU implementation.

The clock starts before ordinary `run_batch` calls and ends after the final
drain. Required best-drawing readbacks and engine allocations stay inside it.
Preparation, snapshots, numerical verification and serialization stay outside.
Both callers require submitted and completed batches, iterations and evaluations
to equal the requested work. The lab additionally requires one timestamp sample
per requested batch. Each caller retains its existing error text and count.

`RecordPurpose` is internal context. It does not add a serialized field or change
either persisted record shape. Timing records receive completed evaluations
divided by elapsed seconds, then lose the score if final checks fail. Diagnostic
and profile records have no score. Resident reference qualifications explicitly
use `Qualification`, so their `TrialRecord` has a null rate even with
`diagnostic:false`. The lab retains its checked `u32` evaluation-count limit.

The Python `validate_result` interface still requires a correct finite score for
ordinary timings and a null score for diagnostics. `validate_qualification`
requires a null score while applying the same work, clock, identity, quality and
protocol checks. It does not relax `validate_result`. The retained campaign
controller and artifacts remain unchanged. Their inconclusive disposition is in
[the original report](../benchmarks/evolution/continuation-20261024/14-prepared-control-inconclusive.md).

Canonical identity remains the JSON array of seven decoded fields, in order:
parents, offspring, parent total errors, parent framebuffers, offspring errors,
chain update flags and packed fitness. Hashing and resident comparison use the
same borrowed canonical view. Padding, reporting/control words, counters and
occupancy diagnostics remain outside replay identity. Python saved-state
verification keeps its independent implementation of this persisted protocol.

The existing canonical-component, work, numerical-error and diagnostic tests
protect these boundaries. Added result-interface tests distinguish timing from
qualification and reject each incomplete work counter. The Python regression
uses a retained native qualification record, with its origin and checksum beside
the fixture. Derived timing shapes test score validation only and do not create
new benchmark evidence.

## Bounded Power of Ten deviations

These preexisting constraints remain local to the qualification callers. New
shared functions use structured control flow, no recursion, and fewer than 60
lines each. Runtime input checks and typed snapshots enforce the invariants;
assertions do not replace external-input errors.

- Rules 2 and 3, `qualification::measure` and its callers. Production GPU waits
  can block without a cancellation bound, and wgpu/ordinary batches allocate
  internally. Changing those mechanisms belongs to the engine lifecycle rather
  than this accounting change. The isolated runner bounds batches to 4096;
  resident lab bounds batches to 64 and uses one device/case at a time with 256
  active offspring. Parent supervisors supply process timeouts. Existing CLI
  budget tests verify admission limits; GPU timeout behavior is not exercised
  here. Follow-up belongs to a GPU cancellation/resource-budget change before
  removing the parent timeout requirement.
- Rules 3 and 9, canonical serialization and numerical verification. Serde JSON
  allocates output and renderer/audit iterators call fixed, inspectable functions
  over admitted fixture/snapshot sizes. Removing these allocations or callbacks
  would change independent numerical code or require a serialization redesign.
  They stay outside the clock. Component-mutation, fixed-hash and result-quality
  tests check the unchanged boundaries. A future serialization or numerical
  resource-budget change should preallocate storage and audit those callbacks.
- Rule 4, resident `Config::parse` and `main`, at 68 and 64 physical lines.
  These existing entry points cross the one-page guideline to keep CLI admission
  and process ownership visible. Splitting their lifecycle would broaden this
  refactor. CLI budget and source-preflight tests remain the compensating checks.
  Revisit this deviation when CLI admission or process orchestration changes.
- Rule 8, Serde derives, error-formatting and assertion macros. The existing
  Rust/Serde protocol uses generated implementations over fixed record fields.
  Replacing that machinery would broaden the serialization contract. Persisted
  shapes, the fixed canonical hash and the native JSON regression constrain its
  output. Any future record-schema change must revisit these checks.
- Rule 10, existing repository diagnostics and numeric conversions. Global
  Clippy pedantic and Ruff findings predate this change; repository-wide lint
  configuration and unrelated findings remain outside this candidate's file
  ownership. The shared rate preserves the isolated runner's `u64` to `f64`
  conversion, and the lab still checks `u32` bounds. Focused Rust checks and tests,
  Clippy pedantic, Ruff and baseline comparisons identify remaining findings
  without adding suppressions. Ruff retains the same five configured findings;
  rustfmt retains two existing test-formatting differences in `engine-lab` to
  avoid unrelated edits. The repository owner should handle global diagnostics
  in a separately scoped cleanup.
