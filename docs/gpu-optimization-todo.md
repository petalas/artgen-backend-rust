# GPU evolution performance plan

## First campaign agreement

The user chose completed evaluations per second as the first optimization score, with evolution behavior and quality preserved. Prepare and qualify the measurement environment before changing the engine for speed.

The scored workloads are mature states from **ff, mona, and darwin**, all strictly above **97% similarity**. Complexity and current similarity are part of each workload's identity. Blank starts, early evolution, and simple synthetic cases are measurement controls or correctness checks; they do not contribute to the optimization score.

This plan began with source inspection at `543b766`. The first mature-drawing restart slice now includes local frozen inputs, seeded completed-work trials, an independent scalar renderer, overflow replay and supervised A/A reference capture. See the [implemented commands and current boundary](../benchmarks/evolution/v1/README.md). The user has captured the first reference; [baseline notes](../benchmarks/evolution/v1/baseline.md) record its checks, variation and identity.

Machine isolation remains operator-declared rather than continuously verified. Six isolated experiments have run under the [provisional screening policy](../benchmarks/evolution/v1/comparison-policy.md) and the separate setup/non-regression objective for experiment 6. The first four failed promotion. The tile-parallel fifth candidate passed the full timing screen. The sixth reduced setup time but failed its throughput guard, so candidate 5 remains the best archived version. No runtime throughput change is installed.

Candidate 5 also passed the [new-state and disjoint-seed engine confirmation](../benchmarks/evolution/v1/acceptance-result.md), with 2.30456 times throughput across 45 valid workers and exact replay. The first live package stopped invalid because its result formatter widened reported float values. Its remaining 19 workers were not run. [Live protocol 2](../benchmarks/evolution/v1/live-protocol-2.md) repairs serialization while retaining exact parameter checks and every performance threshold. Fresh live A/A qualification remains required, along with true continuation and broader campaign gates.

The earlier startup comparison covers one fixture and short runs; retain it as evidence for that change, rather than using it as the campaign baseline.

Qualification exposed two source correctness defects before reference capture: approximate reference-byte reconstruction undercounted integer L1, and post-selection cleanup removed visible triangles after their error was cached. The corrected engine recovers exact reference bytes and preserves the evaluated candidate's geometry. These repairs establish the reference behavior and do not count toward the proposed throughput target.

Freeze the reference image, resolution, active population, offspring count, batch size, workgroup size, mutation parameters, adaptive settings, precision, fitness and selection rules, and seed schedule for every before/after comparison. Faster execution must preserve the same logical evaluations. Exact reuse of equivalent work is allowed; counting unfinished or skipped evolution steps is invalid.

Proposed first-campaign target: a 20% improvement in the fixed suite's aggregate throughput after at least six valid, isolated experiments. This is a planning target, not a measured opportunity. Review the attempt and GPU-time budget after environment qualification, before starting a sustained run. Stop on the agreed predicate or an evidence-backed plateau; keep the original predicate in the report.

## Exploratory attempts retained

| Attempt | Declared budget | Aggregate paired result | Disposition |
| --- | --- | --- | --- |
| [1. Dirty-region bins](../benchmarks/evolution/v1/experiments/01-dirty-binning.md) | Five pairs/project, 64 batches | 1.625% gain | REJECT, below strict 3% |
| [2. Packed rejection](../benchmarks/evolution/v1/experiments/02-packed-rejection.md) | Five pairs/project, 8 batches | 2.117% gain | REJECT preliminary, no full-budget inference |
| [3. Parallel bounds](../benchmarks/evolution/v1/experiments/03-parallel-bounds.md) | Five pairs/project, 8 batches | No measurable gain | REJECT preliminary, broad interval |
| [4. Combined work avoidance](../benchmarks/evolution/v1/experiments/04-combined-binning.md) | Five pairs/project, 8 batches | No measurable gain | REJECT preliminary |
| [5. Tile-parallel filling](../benchmarks/evolution/v1/experiments/05-tile-parallel-binning.md) | Five pairs/project, 64 batches | 2.3145 times throughput | Timing screen passed; adoption pending |
| [6. Remove bin-cache zeroing](../benchmarks/evolution/v1/experiments/06-bin-cache-initialization.md) | Five pairs/project, 64 batches versus candidate 5 | 39.83% less setup, throughput ratio 0.99803 | REJECT, mona throughput guard failed |

All six preserve the mature inputs and pass exact endpoint replay and cached
framebuffer byte-L1 checks. Background activity was uncontrolled. Short
screens do not replace the full-budget reference. Patches and evidence remain
available, and the production engine remains at the reference.

The [amplification diagnostic](../benchmarks/evolution/v1/binning-sensitivity.md)
then measured a 69% wall-time increase when the original bin body was repeated,
with identical evolved states. Tile-parallel filling addresses that measured code
path and passes the full timing screen with exact endpoint replay. Benchmark
setup increased by roughly 6.6 seconds. Removing automatic bin-cache zeroing
then reduced benchmark preparation by 39.83%, but mona missed the separately
frozen throughput guard. That change was rejected without excluding its slow
pair or adding samples. Compiler phases and direct UI readiness were not
measured. Candidate 5 retains its original bin initialization. New mature states
and disjoint seeds have passed engine confirmation. Live acceptance still needs
fresh protocol-2 A/A qualification before the runtime default changes.
The original 20%/six-attempt target remains a broader
campaign proposal, rather than an achieved adoption result.

## What the current code tells us

| Observed fact | Consequence for the environment | Evidence at the audited revision |
|---|---|---|
| The Divan GPU benchmark uses the older renderer. | Retain component benchmarks, and measure evolution through `GpuEvolver::run_batch`. | `benches/bench.rs:28` |
| The CLI uses a blank start at 256px and enables adaptive mutation; production defaults differ. | Replace implicit defaults with a complete, versioned trial specification. | `src/main.rs:1639`; `shared/src/settings.rs` |
| Evaluation totals advance after submission; the CLI does not drain its last batch before reporting. | Time completed work through a final checked drain, and distinguish submitted and completed work in records. | `src/gpu_evolver/mod.rs:512`; `src/main.rs:1736` |
| Warmup performs two batches, then reseeds from the accumulated iteration count. | Add explicit trial seeds and state restoration independent of warmup history and batch partitioning. | `src/gpu_evolver/mod.rs:815`; `src/gpu_evolver/mod.rs:1068` |
| CLI, warmup, viewer benchmark, and live evolution use different timestamp policies. | Warm and measure the same declared production path. Keep separate attribution runs. | `src/main.rs:1714`; `src/main.rs:2088`; `src/gpu_evolver/mod.rs:1087` |
| Pass timers sample the last iteration; selection timing includes framebuffer updates. | Record coverage and actual stage names. Sampled pass sums are not full GPU or wall time. | `src/gpu_evolver/mod.rs:393`; `src/gpu_evolver/mod.rs:456` |
| The viewer benchmark includes locks and conditional preview publication. | Use a controlled engine score plus a separate live integration acceptance case. | `src/main.rs:1359`; `src/main.rs:1483` |
| Tile bins store at most 256 polygon indices, while framebuffer initialization and updates visit all polygons. | Detect overflow throughout a trial. An affected run cannot establish correctness or a performance win. | `src/shaders/bin_polygons.wgsl:151`; `src/shaders/init_framebuffers.wgsl:165` |
| Existing GPU checks cover partial tiles and incremental/full agreement, but both paths can share renderer defects. | Add an independent packed-representation oracle and broader correctness cases. | `examples/check-gpu-rasterization.rs`; `src/gpu_evolver/mod.rs:902` |

Additional items require investigation, not an assumed fix: independent atomic updates of best fitness and chain ID; selection by raw error before complexity penalty; score consistency after degenerate culling; capacity calculations for tile and framebuffer buffers. Reproduce each suspected defect at a narrow seam. Resolve any failure that invalidates the benchmark in a separately recorded correctness change, then establish the baseline on the corrected revision. Do not count a correctness repair as a throughput optimization.

The CPU renderer is useful for diagnostics, but is not an exact GPU oracle. Its vertex rounding and per-layer integer blending differ from the packed GPU coordinates and floating-point blending. Document the current semantics from code; the older renderer and historical comments do not define the evolution contract.

## Delivery milestones and exit gates

### 1. Freeze contracts and workloads

Create a versioned manifest under `benchmarks/evolution/v1/`. It identifies normalized RGBA bytes, initial state or continuation checkpoint, complete parameters, seed schedules, fixed completed-work budgets, case weights, splits, and content hashes. Record source-image hashes and preprocessing details as well as normalized hashes. A live project directory is an import source, not a mutable benchmark dependency.

Start with at least one frozen mature checkpoint from each of `ff`, `mona`, and `darwin`, using the normal 384px longest-side resolution and preserving aspect ratio. Require every restored active parent chain to exceed 97.0% on the GPU fitness displayed as similarity in the viewer. Record raw L1 similarity and complexity-penalized fitness separately. Verify the threshold at the exact normalized reference bytes and resolution used by the trial; saved JSON fitness alone cannot qualify a checkpoint. Offspring may naturally score lower during mutation. Do not filter those proposals or change evolution to keep them above the threshold.

Add distinct mature checkpoints per project to cover different polygon complexity and similarity levels above 97%, aiming for six scored cases when suitable states are available. Additional 256px or 512px cases must qualify independently at their own resolution. Freeze the chosen matrix before measuring candidates. A checkpoint that fails qualification must be replaced by a genuinely mature state generated with the unchanged baseline in an isolated copy, rather than a blank start, reduced polygon count, or relabeled score. Local project assets can remain in an untracked, hashed corpus when their redistribution status is unclear.

For every checkpoint record packed triangle and vertex counts, normalized polygon area and bounding-box distributions, per-tile overlap counts and overflow status, and per-chain similarity. Also retain adaptive mutation scales and stagnation counters. During trials record acceptance rates and dirty-region coverage with a declared sampling policy. These describe why evaluations have different costs; do not divide throughput by polygon count or blend cases at different evolution stages. Compare each candidate against the exact same frozen starting population and completed-work budget.

All three named projects are scored development workloads. Reserve separate mature checkpoints and disjoint seeds for confirmation. These test unseen states of known projects, not unseen image families. Add held-out image families when available, keeping every state and resolution of an unseen family out of development; any held-out performance workload must also exceed 97%. Simple synthetic targets remain outside the throughput score.

Include true continuation checkpoints with diverse active chains, offspring RNG state, mutation scales, stagnation counters, and relevant control state. Single-drawing restarts remain useful as separately labeled cases. Define the checkpoint by a successful next-step replay, rather than guessing which fields are sufficient. Rebuild derivable caches outside the timed interval and verify them before use.

Keep correctness fixtures separate from scored workloads. Cover odd dimensions such as 17x19, shared polygon chunk boundaries, 255/256/257 overlapping tile entries, empty and near-capacity drawings, alpha extremes, winding, shared edges, degenerate triangles, reset/reload, and long incremental sequences. Exercise supported workgroups and control combinations without multiplying every dimension into the scored matrix.

**Exit gate:** every fixture is immutable, validated, available locally, and reproducible by hash. Every scored initial population has verified similarity above 97% at the measured resolution, recorded complexity, and a documented origin. All three projects are represented. Workload weights and validation splits are fixed.

### 2. Implement one trial owner

Add a thin standalone main-thread executable, proposed as `examples/evolution-bench.rs`, backed by a cohesive runner under `src/benchmark/`. The existing `src/benchmark.rs` can expose that runner alongside current shared UI types. Avoid putting the protocol in the viewer or duplicating it across the existing CLI, examples, and comparison script.

The runner owns validation, GPU initialization, warmup, seeded reset or checkpoint restoration, timing, submission, completion, final state collection, and verification. A proposed internal interface is `run_trial(spec) -> Result<TrialRecord, TrialFailure>`. UI conversion stays outside it.

A `TrialSpec` records protocol/case identity, project and checkpoint identity, fixture hashes, initial complexity and verified similarity, full requested parameters, explicit seed or checkpoint hash, fixed batch/work budget, instrumentation policy, cache policy, and time/memory limits. Validation rejects an unqualified scored checkpoint, silent parameter sanitization, population clamping, unsupported features, or incomplete buffer capacity. Check drawing, offspring, tile, framebuffer, and dispatch capacity together.

A `TrialRecord` records status; requested and effective settings; submitted and completed batches, iterations, and logical evaluations; error and overflow counts; measured wall and drain times; GPU timing coverage; initial and final quality and complexity; state/checkpoint hashes; artifact paths; and build, GPU, driver, backend, toolchain, dependency-lock and protocol identity. Preserve integer RGB L1 totals and polygon counts alongside percentage fitness. Store the actual final drawing and independently checked score. Include quality and complexity checkpoints so a faster run cannot conceal a simpler or lower-similarity workload.

Represent completed, invalid, failed, timed-out, and cancelled trials explicitly. A cancelled or partial run cannot produce a valid comparison score. The parent runner imposes a timeout and retains stderr/exit status, including abnormal GPU-process exits. Cap all run counts and artifacts before launching them.

**Exit gate:** one-batch and multi-batch accounting matches completed GPU work; zero/invalid budgets fail clearly; cancellation, timeout, device error and last-batch handling produce correct statuses. Final records print error counts and completed work, not just elapsed time.

### 3. Establish independent correctness and replay

Define the authoritative representation and arithmetic before writing the oracle. Build a small scalar reference over packed GPU vertices/colors with the documented pixel-center, winding, coverage, floating-point blending, final quantization, L1 and complexity rules. Use analytical tiny cases to qualify it. Keep full unbinned GPU rendering as another differential check where useful.

Compare per-pixel output and integer errors as well as fitness. Freeze numerical tolerances based on the unchanged engine's behavior and an understood reference, before evaluating candidates. A loose aggregate fitness tolerance is insufficient. The binned full evaluator alone cannot validate a defect shared with incremental evaluation.

Add per-trial overflow detection or a proven complete fallback. Checking only the initial or final drawing misses overflow in intermediate offspring. Diagnostic instrumentation must have a separate mode if it changes timed work; it must exercise the same seeded trajectory. Dense cases cannot silently disappear from the suite or become smaller to pass this gate.

Expose reproducible seeded reset/checkpoint restoration without altering the normal application's default seed behavior. Verify repeated same-seed runs, reset after unrelated trials, warmup independence, and execution partitioning where it is intended to be equivalent. Freeze batch partitioning when a proven behavioral dependency exists instead of assuming equivalence.

At fixed completed-work checkpoints compare canonical active population state, relevant RNG/control state, pixel/error results, stored versus recomputed fitness, selection behavior, and the drawing returned as best. Omit irrelevant padding from logical hashes. Same-driver deterministic cases require exact canonical replay. If the unchanged engine is nondeterministic, explain and qualify that behavior before choosing its gate; do not loosen replay requirements to accept a candidate.

Use the standalone main-thread GPU validation path on the current Dozen setup. Preserve `check-gpu-rasterization`'s visible late-chunk fixture, incremental comparisons before resets, and analytical checks. Extend it with the missing coverage and replay cases above.

**Exit gate:** correctness and replay checks pass on the frozen baseline. Every observed discrepancy is resolved or explicitly blocks the affected campaign. There is a failing test for each known measurement/correctness defect being repaired.

### 4. Qualify timing and machine isolation

Use optimized, locked builds and archive executable hashes for the reference and candidate. Record the resolved nightly compiler and flags, dependency lock hash, adapter/backend/features, device limits, Vulkan/Dozen and GPU driver versions, and operating environment. A toolchain, dependency, driver, protocol, or corpus change requires a new baseline series.

Run GPU jobs serially under a local benchmark lock. Use isolated working directories, copied cache seeds, and dedicated outputs. Check the live backend and other GPU activity before a campaign; do not silently stop the user's session. Unsupported telemetry is recorded as unavailable, not as zero utilization. Collect CPU load and supported GPU clock, temperature, power and utilization telemetry. Do not require clock locking or elevated privileges.

Defer timed work while the user's backend is evolving or other unaccounted GPU work is active. Leave those processes untouched. Contention detected during a sample invalidates its entire reference/candidate pair; retain both records and repeat the pair after the GPU is quiet. Freeze the detection and invalidation rules with the protocol. Missing telemetry cannot establish that the GPU is idle, and a slow timing alone cannot justify excluding a sample.

Declare cold/warm startup separately from steady evolution. Warm the exact production instrumentation path and verify stabilization experimentally. Restore the precise measured start after warmup, without shifting its RNG stream. Calibrate a fixed completed-work budget once per case so scored samples last roughly 15 to 30 seconds on the baseline. Store that budget; execute the identical logical work on the candidate. Setup, compilation and warmup are outside the score, final GPU completion is inside it.

For the primary engine mode include CPU command encoding, submission, required result mapping/readback, and completion. Keep disk writes, report generation, and added correctness diagnostics outside the interval. Production-required operations remain inside it. Use a separate live integration case with fixed preview cadence, client count and publication workload to catch practical regressions.

Freeze a live acceptance specification alongside the scored cases. Use one of the qualified above-97% mature project checkpoints with default settings, a fixed completed-evaluation budget, the actual backend PNG/serialization/WebSocket path, one local receiving client, and a fixed request trace and preview policy. Include the final GPU drain and due preview publications in its measured interval. Record completed evaluations per second, previews due/sent/received, missed publication deadlines, and each run's p95 latency from a scheduled preview to client receipt. Qualify the budget and sampling rules so the unchanged application can produce enough publications to estimate this latency.

Run at least five alternating paired live trials. Apply the same throughput non-regression rule as for each scored case. For per-run p95 publication latency, require a non-increasing paired median and an upper paired confidence bound within a frozen equivalence margin no wider than 1%. These are proposed defaults to qualify during A/A testing. Require no missing due publications, device errors, or missed-deadline rate increase. Freeze the deadline, statistical procedures, margins and sample cap before testing candidates. Failure to establish either live throughput or latency non-regression is `INCONCLUSIVE` and prevents `KEEP`.

Use existing production timestamps for the scored path. Collect detailed attribution in separate runs and report sampled-iteration coverage, selection-plus-update grouping, CPU encoding/wait/readback costs and unaccounted wall time. Add finer GPU boundaries only when needed for diagnosis, then quantify their overhead independently. Historical GPU percentages and CPU profiles do not establish the current limiter.

**Exit gate:** demonstrate repeated A/A comparisons, cheap versus realistic workloads, a known slowdown inside the measured interval, a delay outside it, a no-work/unfinished-work failure, bad fixture/checkpoint rejection, and complete error accounting. Establish the variance and detectable effect before freezing the protocol.

### 5. Record and freeze the baseline

Run at least five alternating paired repetitions per scored case, using identical paired state/seed schedules. Separate same-seed timing repetitions from independent search-seed validation so they are not mistaken for independent evidence. Begin with a small paired schedule, then expand when uncertainty warrants it. Record order, all raw samples and every exclusion. Re-run both sides when conditions invalidate a pair.

Choose fixed case weights before seeing candidate results. Give `ff`, `mona`, and `darwin` equal total weight, dividing each project's share across its scored checkpoints and resolutions. The primary score is the weighted geometric mean of matched per-case throughput ratios. Report each project's throughput and ratio separately, with checkpoint similarity, complexity and resolution. Additional snapshots from an easier project must not increase that project's influence. Keep per-case results and worst-case behavior visible so a broad regression cannot hide behind the mean.

Proposed acceptance defaults, to freeze after the A/A audit:

- The paired aggregate improvement must exceed both a 3% practical threshold and the calibrated noise band, with a positive paired confidence bound.
- Every scored case must have a non-regressing paired median. Its lower confidence bound must meet a predeclared equivalence bound no wider than 1%. This is measurement tolerance, not permission to trade away a known slowdown.
- Any result that cannot establish the required non-regression is `INCONCLUSIVE`, not `KEEP`. Add samples up to a fixed cap or reject the attempt; do not repeatedly retry until a favorable sample appears.
- All correctness, replay, capacity, overflow and live integration gates pass. Verify quality again at fixed completed-work checkpoints and with disjoint validation seeds. End-of-run fitness alone is insufficient.

Use a paired analysis that preserves the case/seed structure. Treat confidence procedures and sample caps as part of the versioned comparison code. Do not pool different images and seed repetitions as interchangeable observations.

**Exit gate:** a baseline bundle exists with the exact revision/build, case manifest, raw runs, correctness report, traces, telemetry, noise analysis, thresholds and comparison-tool identity. A second invocation reproduces the decision within the declared noise. The independent review has no unresolved measurement-validity finding.

### 6. Run the first measured experiment loop

Only start scored campaign attempts after milestones 1 through 5 pass. Separate exploratory screens can test isolated mechanisms before broader qualification, but cannot return KEEP or relax the remaining gates. Each attempt states one mechanism, predicted limiting cost, affected cases, expected correctness invariants and rejection condition. Keep source changes isolated and preserve unrelated work. Archive the reference/candidate binary hashes, source patch and trial specifications before measurement.

Use shorter fixed-work trials of the qualified mature projects for cheap performance screening. Label their findings provisional; they cannot accept a win. Simple or early-state controls remain useful for runner diagnostics and never substitute for the mature suite. Promising candidates receive the full alternating confirmation suite and quality gates. Check reserved mature checkpoints, disjoint seeds, and any qualified held-out image families before a campaign winner is adopted, without using confirmation results to tune implementation details.

Order hypotheses by avoiding work, reusing work, reducing frequency, deferral, scheduling, concurrency, then making the operation cheaper. Select concrete attempts from fresh traces. Initial candidate families include submission/readback scheduling, host serialization/allocation, redundant bindings/transfers, exact binning improvements, and conservatively smaller dirty regions. Treat the February suggestions below as hypotheses needing source and trace checks. Mutation probabilities, population/resolution reductions, lower precision, approximate scoring, candidate screening and changed selection rules belong to a separately scored algorithm campaign.

For every attempt append its hypothesis, patch/binary identities, before/after cases, error counts, work counts, correctness result, verdict and explanation. Use `KEEP`, `REJECT`, or `INCONCLUSIVE`. Keep a change only when all frozen gates pass. A harness defect increments the protocol version and requires rebaselining, rather than retrospectively changing the acceptance rule.

If progress stalls, change hypothesis family or collect a new trace. Combine near-misses only after measuring each separately. Finish by replaying the held-out validation, longer mature-state runs and the live integration case. Report kept and rejected attempts, remaining limiting cost and any unresolved constraint.

## Planned artifacts and controls

| Artifact | Purpose |
|---|---|
| `benchmarks/evolution/v1/manifest.json` and fixture/checkpoint files | Frozen, hashed workloads, splits, seeds, weights and work budgets |
| `src/benchmark/` modules | Trial contract, lifecycle, validation and result collection |
| `examples/evolution-bench.rs` | Main-thread entry point for preflight, validation and one trial |
| `scripts/compare-evolution.py`, or an equally small typed comparator | One versioned implementation of pairing, scoring and decisions |
| `.cache/evolution-bench/<campaign>/<attempt>/` | Immutable executable copies, raw JSON, stderr, quality checkpoints, traces, telemetry and reports |
| Local append-only decision trail | Hypotheses, changes, evidence, rejected attempts and verdicts |

With five pairs and 15-second samples, confirmation contains 150 seconds of timed work per scored case. Three cases require 7.5 minutes; six require 15 minutes, plus startup, warmup, live integration and verification. That is planning arithmetic, not a runtime estimate. Freeze the mature case matrix and budget the campaign after measuring setup overhead. Do not launch a multi-hour unattended campaign as part of planning.

The environment implementation is ready for review when one command can preflight, validate, run an A/A comparison and emit a complete decision bundle; corrupt inputs, unfinished work and known slowdowns are detected; the numerical/replay gates pass; and another invocation can reproduce the same verdict. Engine optimization starts after that checkpoint.

Foundational Thinking shapes the contracts, continuation checkpoints and isolation before experimentation. Build the Lever shapes the single runner and comparator to be implemented. Explain the Number and Benchmark checklist shape completion accounting, profiling, pairing and noise qualification. Prove It Works shapes independent numerical and live integration gates.

## Historical February 2026 idea backlog

The material below is retained as a historical backlog. Its completion claims, numerical speedup estimates, hardware assumptions and bottleneck rankings need fresh source inspection and measurement before use. The first campaign is governed by the measurement plan above.

Synthesized from 6 expert analyses (GPU compute, evolutionary algorithm, CPU-GPU hybrid, data structures, wgpu API, numerical methods). February 2026. Round 2 — focuses on NEW ideas beyond previous optimizations.

**Target hardware:** RTX 5090 (170 SMs, 32-wide warps, 48KB shared memory per SM)

---

## Already Completed

- [x] Fused rasterize + error computation in single pass
- [x] Double-buffered async staging (CPU reads batch N-1 while GPU computes batch N)
- [x] Persistent PCG-32 RNG per offspring slot
- [x] Shared memory polygon tiling (768 polys/tile cooperative prefetch)
- [x] Reference image stored as GPU texture (2D cache-friendly)
- [x] GpuParams via uniform buffer (shared bind group across pipelines)
- [x] Adaptive mutation scale (1.2x accept, pow(0.99,1/lambda) reject, burst on stagnation)
- [x] (1+lambda)-ES with configurable lambda (1-64, power-of-2)
- [x] Island model with intra/inter-island ring migration
- [x] 13+ mutation operators (structural, geometric, vertex, color, crossover)
- [x] Dirty bounding box tracking per offspring
- [x] Tile culling via bin_polygons shader (optional)
- [x] Incremental evaluation mode (optional)
- [x] Half-space edge function with AABB culling
- [x] Vertex quantization (2x u16) and color quantization (pack4x8unorm)
- [x] Vulkan pipeline cache for shader binary reuse
- [x] Double-buffered command encoding
- [x] Timestamp query profiling infrastructure
- [x] Atomic lock-free selection (atomicMax/atomicExchange)
- [x] Degenerate triangle culling on acceptance
- [x] Compute pass boundary consolidation (single pass for non-profiled iterations)

---

## Tier 1 — High Impact, Reasonable Effort

### 1.1 Reorder submit/collect in run_batch() — [7-30% throughput]
**Source:** CPU-GPU Hybrid expert
**Problem:** `finish_pending_readback()` blocks on `device.poll(Wait)` *before* submitting the next batch, creating a GPU idle bubble while CPU does mutex checks, stats, PNG encoding.
**Fix:** Submit batch N+1 first, then block on batch N results. GPU starts immediately while CPU processes.
**Effort:** Medium — restructure `run_batch()` control flow
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 1.2 Fix incremental eval parent re-rasterization — [5-20x on incremental path]
**Source:** GPU Compute + Data Structures experts
**Problem:** With incremental eval enabled, `rasterize_error.wgsl` re-rasterizes the FULL parent drawing at every pixel in the dirty bbox. But `chain_framebuffers` already stores the parent's rendered output — just never updated after acceptance.
**Fix:** (a) Update `chain_framebuffers` in the select shader after acceptance, (b) read cached pixel instead of re-rasterizing parent (O(1) vs O(polygon_count) per pixel).
**Effort:** Medium — shader changes in select + rasterize_error
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 1.3 Tighter dirty bboxes for z-order mutations — [2-7.5x rasterize speedup for affected mutations]
**Source:** Numerical Methods expert
**Problem:** `adjacent_swap`, `remove_polygon`, `swap_colors` mutations return `full_image_bbox()` in mutate shader, forcing full-image re-rasterization even though only 2 polygons' bounding boxes change.
**Fix:** Replace with `merge_bbox(polygon_a_bbox, polygon_b_bbox)`. Since z-order mutations are 20-30% of operations, weighted average rasterize speedup is 2-7.5x under incremental eval.
**Effort:** Low — ~20 lines in mutate.wgsl
**Details:** [numerical-methods-optimization.md](numerical-methods-optimization.md)

### 1.4 Stratified offspring mutation — [15-25% faster convergence]
**Source:** Evolutionary Algorithm expert
**Problem:** With lambda=64, all offspring sample from the same probability distribution. Rare-but-high-impact mutations (add/remove polygon) fire probabilistically once per ~150 iterations instead of every iteration.
**Fix:** Partition offspring into deterministic strata — reserved slots for add_polygon, remove_polygon, structural mutations, etc. Latin hypercube sampling over mutation type space.
**Effort:** Low-Medium — mutation selection logic in mutate.wgsl
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

### 1.5 Color-aware polygon initialization — [2-5x add_polygon acceptance]
**Source:** Evolutionary Algorithm expert
**Problem:** New polygons get random colors, requiring many subsequent color mutations to become useful.
**Fix:** Sample reference image texture at polygon centroid to initialize color. ~15 lines of code — add texture binding to mutate shader.
**Effort:** Low
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

---

## Tier 2 — Medium Impact, Low Effort (Quick Wins)

### 2.1 Skip unconditional polygon writeback in multi-mutation — [~500KB/iter saved]
**Source:** GPU Compute expert
**Problem:** Per-polygon loop in `mutate.wgsl` unpacks, potentially mutates, then repacks and writes back ALL polygons even when nothing changed (~96% are unmodified).
**Fix:** Add `modified` flag, skip writeback for unmodified polygons.
**Effort:** Very low — zero risk
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 2.2 Fix `shared_polys` over-allocation — [improved SM occupancy]
**Source:** Data Structures expert
**Problem:** `shared_polys` is hardcoded to 1536 entries but actual `TILE_CAP = THREAD_COUNT * 3` at default 16x16 uses only 768. Wastes 12KB shared memory per workgroup, reducing SM occupancy.
**Fix:** Change declaration to `array<Polygon, TILE_CAP>`. Trivial.
**Effort:** Trivial
**Details:** [data-structures-algorithms.md](data-structures-algorithms.md)

### 2.3 Submission index tracking for targeted polling — [eliminates spurious waits]
**Source:** wgpu API expert
**Problem:** `device.poll()` uses `submission_index: None`, waiting for ALL pending work including the just-submitted batch.
**Fix:** Capture `queue.submit()` return value, pass to `device.poll()`.
**Effort:** Trivial
**Details:** [wgpu-api-optimization.md](wgpu-api-optimization.md)

### 2.4 Persistent staging buffers — [eliminates per-call allocation]
**Source:** wgpu API expert
**Problem:** `readback_chains` and `evaluate_chain_fitness` create and destroy MAP_READ staging buffers on every call, triggering Vulkan memory management operations.
**Fix:** Pre-allocate persistent staging buffers at pipeline creation time.
**Effort:** Low
**Details:** [wgpu-api-optimization.md](wgpu-api-optimization.md)

### 2.5 Conditional timestamp collection — [eliminate 95% of extra pass transitions]
**Source:** CPU-GPU Hybrid expert
**Problem:** Every batch forces the last iteration into 3 separate compute passes for timestamp profiling instead of the single bulk pass.
**Fix:** Collect timestamps only every Nth batch (e.g., every 20th).
**Effort:** Trivial — 10 minute change
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 2.6 Push constants for GpuParams — [eliminate UBO write + bind group overhead]
**Source:** wgpu API expert
**Problem:** 256-byte GpuParams written via `queue.write_buffer()` + `set_bind_group(1, ...)` called 3-4 times per iteration (192-256 redundant bind group commands per batch).
**Fix:** Use `immediate_size: 256` in pipeline layout. Data lives in command processor registers for zero-latency reads.
**Effort:** Medium — pipeline layout changes across all shaders
**Details:** [wgpu-api-optimization.md](wgpu-api-optimization.md)

---

## Tier 3 — High Impact, High Effort (Architectural)

### 3.1 Coarse-to-fine two-phase screening — [up to 8x rasterize speedup]
**Source:** Evolutionary Algorithm expert
**Problem:** All 64 offspring rasterized at full 512x512 despite 90-95% rejection rate.
**Fix:** Coarse screen at 128x128 for all offspring, full-resolution only for top 4. Reduces total rasterization from 16.8M to ~2M pixels per iteration.
**Effort:** High — new shader pass, intermediate buffers, architectural rework
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

### 3.2 Progressive resolution evaluation — [up to 16x early speedup]
**Source:** Numerical Methods expert
**Problem:** Full-resolution evaluation wasteful in early evolution when fitness is low.
**Fix:** Three-tier schedule: 1/4 res below 60% fitness, 1/2 at 60-80%, full above 80%.
**Effort:** High — reference image mipmaps, dynamic dispatch sizing, fitness-dependent switching
**Details:** [numerical-methods-optimization.md](numerical-methods-optimization.md)

### 3.3 Copy-on-write offspring representation — [50-100x memory reduction]
**Source:** Data Structures expert
**Problem:** Each offspring duplicates full 16032-byte GpuDrawingState even though single-mutation modifies only 1-2 polygons.
**Fix:** Delta representation: (polygon_index, old_polygon, new_polygon) at ~48 bytes per mutation.
**Effort:** Very high — fundamental restructuring of mutate and rasterize shaders
**Details:** [data-structures-algorithms.md](data-structures-algorithms.md)

### 3.4 Error-guided polygon placement — [3-10x add_polygon acceptance at high fitness]
**Source:** Evolutionary Algorithm expert
**Problem:** New polygons land at random positions, <1% hit rate on high-error regions at high fitness.
**Fix:** Lightweight 4x4 per-chain error grid (64 bytes) biases add_polygon origins and mutation targeting toward high-error cells.
**Effort:** Medium-High — error grid computation, sampling logic in mutate shader
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

---

## Tier 4 — Medium Impact, Medium Effort

### 4.1 SoA shared memory layout — [33% more polys/tile]
**Source:** GPU Compute expert
**Problem:** `shared_polys` stores full 16-byte Polygon structs. Color only needed for ~5-10% that pass half-space test.
**Fix:** Store only 12-byte geometry in shared memory, defer 4-byte color read to L2 on hit. Fits 1024 polys/tile vs 768.
**Effort:** Medium
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 4.2 Integer-space AABB culling — [~700M saved FP ops per offspring]
**Source:** GPU Compute expert
**Problem:** AABB cull requires unpacking all 3 vertices to float before checking bounds.
**Fix:** Perform AABB test in u16 pixel coordinates (bit ops on raw packed u32 words) before float unpack. Saves 6 FP ops per rejected polygon (~90% reject rate).
**Effort:** Low-Medium
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

### 4.3 Parallelize bin_polygons shader — [32-64x binning utilization]
**Source:** GPU Compute + Data Structures experts
**Problem:** `@workgroup_size(1,1,1)` — one thread per offspring. 128 single-thread workgroups across 170 SMs.
**Fix:** Use 64-thread workgroup, parallelize tile count reset. Polygon scatter remains serial for ordering but reset phase is trivially parallelizable.
**Effort:** Medium
**Details:** [gpu-compute-optimization.md](gpu-compute-optimization.md)

The later [parallel-bounds experiment](../benchmarks/evolution/v1/experiments/03-parallel-bounds.md)
tested 32-lane bounds preprocessing with serial reset/scatter. It did not establish
a throughput gain at its declared preliminary budget. That is a different
mechanism from parallel count resets, and gives no evidence for the historical
32 to 64 times utilization claim in this heading.

### 4.4 Polygon splitting mutation — [adds representational capacity]
**Source:** Evolutionary Algorithm expert
**Problem:** Adding random new triangles is inefficient for fine detail.
**Fix:** Subdivide existing large polygon into 2-3 sub-triangles at centroid. Fitness-neutral (accepted immediately), then evolution differentiates colors.
**Effort:** Medium
**Details:** [evolutionary-algorithm.md](evolutionary-algorithm.md)

### 4.5 L2-squared error metric — [5-15% faster perceptual convergence]
**Source:** Numerical Methods expert
**Problem:** L1 error treats all channel deviations equally. L2-squared penalizes concentrated single-channel errors more.
**Fix:** Use `dr*dr+dg*dg+db*db` with right-shift-by-8 scaling to prevent u32 overflow.
**Effort:** Low-Medium
**Details:** [numerical-methods-optimization.md](numerical-methods-optimization.md)

### 4.6 Offload CPU rendering/encoding to worker thread — [15-30% combined with 1.1]
**Source:** CPU-GPU Hybrid expert
**Problem:** Main thread does inline CPU rasterization, PNG encoding, JSON serialization, file I/O between batches.
**Fix:** Bounded-channel worker thread for rendering/encoding tasks.
**Effort:** Medium
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 4.7 Adaptive batch size controller — [auto-tune gpu_batch_iters]
**Source:** CPU-GPU Hybrid expert
**Problem:** Static `gpu_batch_iters` (default 50) — optimal varies enormously with config.
**Fix:** EWMA-based controller targeting ~10ms per batch, auto-tunes dynamically.
**Effort:** Medium
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

### 4.8 Multi-mutation RNG batching — [30-50% less arithmetic in mutation loop]
**Source:** Data Structures expert
**Problem:** Multi-mutation path draws ~18 random numbers per polygon regardless of whether any mutation fires.
**Fix:** Batch 2-3 RNG calls, use bit extraction, skip inner loop when no mutation probability exceeds threshold.
**Effort:** Medium
**Details:** [data-structures-algorithms.md](data-structures-algorithms.md)

### 4.9 Async non-blocking best-chain readback — [eliminate 0.5-3ms stall]
**Source:** CPU-GPU Hybrid expert
**Problem:** New global best triggers synchronous GPU round-trip that defeats double-buffering.
**Fix:** Piggyback best-chain copy onto next batch's command encoder, read one batch later.
**Effort:** Medium
**Details:** [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md)

---

## Ruled Out

| Idea | Why |
|------|-----|
| f16 precision for blending | ~0.3-0.5 error/channel after 150 blends creates noise floor drowning micro-adjust improvements |
| Integer (u16) arithmetic | 2.5x more ALU ops — no native u16 in WGSL, division-by-255 approximation overhead |
| SSIM error metric | Requires local neighborhood statistics incompatible with per-pixel accumulation |
| RNG upgrade from PCG-32 | Adequate period and quality for noise-tolerant evolutionary process |

---

## Expert Analysis Documents

| Expert | Document |
|--------|----------|
| GPU Compute | [gpu-compute-optimization.md](gpu-compute-optimization.md) |
| Evolutionary Algorithm | [evolutionary-algorithm.md](evolutionary-algorithm.md) |
| CPU-GPU Hybrid | [cpu-gpu-hybrid.md](cpu-gpu-hybrid.md) |
| Data Structures | [data-structures-algorithms.md](data-structures-algorithms.md) |
| wgpu/Vulkan API | [wgpu-api-optimization.md](wgpu-api-optimization.md) |
| Numerical Methods | [numerical-methods-optimization.md](numerical-methods-optimization.md) |
