# Mature evolution benchmark, protocol 1

This first slice measures completed GPU evolution work on frozen mature drawings
from ff, mona, and darwin. Each active parent must start above 97% GPU fitness at
the actual benchmark resolution. It is a `mature_drawing_restart`: all chains
restart from the saved drawing with explicit independent RNG streams. It does not
restore a previously running population or claim continuation-checkpoint support.

Run commands from the repository root. The supervisor uses Python 3.10 or newer
and its standard library. GPU runs need the same device features as the engine.
It builds the release example with locked dependencies unless `--binary` supplies
an existing executable. GPU work stays on each subprocess's main thread.

The first user-run reference capture is recorded in [baseline notes](baseline.md),
including repeatability, build/fixture identity and the remaining profiling gate.
The [first diagnostic profiles](profile.md) separate host CPU/wait observations
from full GPU batch spans and document the rerunnable profiling command.

## Import once

```sh
python3 scripts/evolution-bench.py freeze
```

This copies the existing project drawings and normalized reference bytes into
`.cache/evolution-bench/fixtures/v1/`. It records full mutation settings, hashes,
source-image identity, dimensions, stored scores, scalar scores and complexity.
The longest side is 384px with the production Lanczos/even-dimension conversion.
Import does not perform GPU qualification. Existing fixture destinations are
immutable and are refused; use `--fixtures` for a new corpus revision.

Keep this local corpus for subsequent comparisons. A changing project directory
is never a timed trial's input. The supervisor also copies the corpus into each
run bundle and checks all hashes before and after every worker process.

## Validate the workflow

```sh
python3 scripts/evolution-bench.py validate
```

This runs two short, identically seeded trials and one overflow-diagnostic replay
for each project. It compares canonical final states, work counters and quality,
and checks GPU results against an independent scalar renderer. The diagnostic
binning variant retains maximum occupancy over every iteration. Its replay must
match the production variant and remain within the 256-entry tile capacity.
Its timings have no throughput score. Validation always labels machine timings
as development data unsuitable for a clean baseline.

Preparation uses the same timestamp policy as measured production batches.
Warmup count does not alter the restored seed. The measured interval includes
command encoding, submission, ordinary drawing readback and the final checked
drain. Initialization, extra state copies, scalar checks and JSON writes are
outside it. Submitted and completed work must agree after draining.

## Capture the reference series when the machine is quiet

```sh
python3 scripts/evolution-bench.py baseline
```

Run this after stopping other evolution runs and GPU-intensive applications.
The supervisor leaves existing processes untouched. It records available GPU,
CPU and backend activity; absent telemetry does not establish an idle machine.
Baseline capture reports isolation as unverified, with any observed contention
retained. The operator's quiet session and the recorded noise must be reviewed
before accepting it as the clean campaign reference.

Defaults are seed 7, two warmup batches, 64 measured batches, and five alternating
A/A pairs per project, with a diagnostic replay for every pair. This creates 45
worker processes across the three projects. The active
population, offspring count, workgroup and iterations per batch come from the
frozen manifest. Repeating a seed estimates timing variation for that trajectory;
these repeats are not independent evolution seeds. Use `--seed` for additional
replay validation, retaining a separate bundle for each schedule.

The command prints progress for each trial. Every invocation creates a new
evidence directory and prints its `summary.json`
path. `--output` chooses a new directory explicitly. Failures, stderr, stdout,
timeouts and partial records remain in that directory, and the command exits
nonzero. `--timeout` bounds each subprocess. A POSIX lock prevents overlapping
supervised runs; each worker receives an isolated working directory and the same
copied pipeline-cache seed. User project files and the caller's caches remain intact.

The bundle includes the executable, source/build identity, fixture inventory,
per-trial state/drawings, numerical checks, completed work, raw timings and
per-project noise statistics. Each project has equal total influence regardless
of case count. A reference series is not an optimization result or an automatic
campaign freeze. Inspect per-case variation before freezing effect thresholds.
Detailed timestamp attribution covers only the final iteration of each batch;
the selection interval includes framebuffer updates. Those sums are not total
application time or a utilization measurement.

## Numerical checks

The scalar renderer consumes packed GPU triangles, rather than the different
CPU drawing renderer. It preserves inclusive edges, layer order, floating-point
blending and final byte flooring. Raw framebuffer RGB L1 must exactly equal the
GPU's cached total. Stored fitness must agree with that total and polygon count
within 0.00002 percentage points. Every nominal pixel difference is reported.

Ordinary channel rounding has a one-byte tolerance. Larger differences require
an independent outward arithmetic envelope for that pixel, including uncertain
edge coverage and subsequent blending. This follows the division/UNORM bounds
and contraction rules in the [WGSL floating-point specification](https://www.w3.org/TR/WGSL/#floating-point-accuracy).
It is a conservative diagnostic, not a portable proof of bitwise rasterization.
Unexplained pixels fail. Exact same-backend canonical evolution replay is still
mandatory; a numerical envelope cannot excuse changed evolution state.

The first diagnostic exposed inconsistent integer error totals when an approximate
UNORM reference sample was multiplied by 255 and truncated after subtraction.
Both error shaders now round recovered reference bytes before RGB L1 arithmetic.
The mixed-reference GPU regression checks compare cached, fully recomputed and
framebuffer byte errors. This is a correctness repair before reference capture,
not an optimization win. Geometry, blending and selection rules are unchanged.

The full-budget Darwin check also exposed cleanup that removed tiny visible
triangles after selection had cached their score. Selection now retains the
evaluated geometry and polygon count. This explicitly removes the unscored
cleanup; a neutral-acceptance GPU regression fails on the old implementation.
Any future geometry cleanup must happen before evaluation. This repair also
precedes reference capture and is not counted as a throughput optimization.

## Checks and current boundary

```sh
cargo test --locked -p artgen-backend-rust -p artgen-shared
python3 -W error -m unittest discover -s scripts -p test_evolution_bench.py
cargo run --release --locked --example check-gpu-rasterization
```

Numerical GPU tests use a standalone main-thread executable because this Dozen
driver can crash while tearing down a GPU created by Rust's threaded test runner.
The existing renderer checks remain alongside the mixed-reference regression.

This slice supplies reproducible engine reference trials and A/A evidence. The
broader campaign still needs clean-machine noise qualification, sustained live
preview acceptance, true continuation checkpoints, broader held-out states and
seeds, and a frozen candidate comparison policy. No engine optimization attempt
should be accepted from a short development run.

The implementation's bounded resource and framework exceptions are recorded in
[implementation notes](implementation-notes.md).
