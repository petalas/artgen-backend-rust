# CPU-GPU Hybrid Architecture Analysis

## Current Architecture

**CPU path** (`cpu_main` / `Evaluator`): Spawns `num_cpus` worker threads. Each runs a tight loop in `produce_new_best()`: clone drawing, mutate until dirty, rasterize via `fill_triangle` (half-space with SIMD alpha blend), compute fitness (Euclidean error), compare against local best. When a worker finds an improvement, it sends the result to the main thread via `mpsc`, which broadcasts it back to all workers via `tokio::broadcast`.

**GPU path** (`gpu_main_loop_headless` / `GpuEvolver`): 512 independent (1+1) chains running entirely on the GPU. Each batch encodes 50 iterations of 5 compute passes (mutate, rasterize, error_reduce, select, migrate) into a single command buffer, submits, then blocks on `device.poll(Maintain::Wait)` to read back control flags. If a global best was found, it reads back one `GpuDrawingState` (48032 bytes). The main thread on the CPU is idle during GPU compute -- it only does control flag checking, periodic PNG rendering, WS state updates, and file saves.

**Key observation**: In the GPU path, the main thread calls `evolver.run_batch()` which calls `device.poll(Maintain::Wait)`, blocking until the GPU finishes. During that time, all CPU cores are idle. On a machine with 8+ cores, this wastes significant compute capacity.

## Opportunity 1: Run CPU Workers Concurrently with GPU Batches

### Concept

The simplest and highest-impact hybrid: spawn CPU worker threads alongside the GPU pipeline. Both populations evolve independently in parallel, with periodic migration between them.

### Architecture

```
Main Thread (orchestrator)
  |
  +---> GpuEvolver (512 chains, batch of 50 iters)
  |       - Submits command buffer
  |       - Polls asynchronously via callback
  |
  +---> CPU Worker 1 (Evaluator)
  +---> CPU Worker 2 (Evaluator)
  +---> ...
  +---> CPU Worker N (Evaluator)
  |
  +---> Shared global_best (Arc<RwLock<Drawing>>)
```

### Implementation Approach

1. **Async GPU polling**: Replace `device.poll(Maintain::Wait)` with `device.poll(Maintain::Poll)` in a non-blocking loop, or move GPU batch submission + readback to a dedicated thread. The key change is making `run_batch` non-blocking so the main thread can service CPU worker results while the GPU computes.

2. **Shared global best**: Both CPU workers and GPU use a single `Arc<RwLock<Drawing>>` for the global best. CPU workers update it via their existing broadcast pattern. The GPU evolver checks it between batches and injects it into chains (via `queue.write_buffer` to update a specific chain's state).

3. **Threading model for headless mode** (the primary target):
   ```
   Thread 0: Main loop - manages WsState, project switching, coordinates GPU + CPU
   Thread 1: GPU poller - submits batches, polls device, reads back results
   Threads 2..N: CPU evaluators (one per remaining core)
   ```

4. **Concrete code changes**:
   - In `gpu_main_loop_headless`, after `GpuEvolver::new()`, also spawn `(num_cpus - 2)` CPU evaluator threads with their own `Evaluator` instances (same `ref_image_data`, same initial `Drawing`).
   - CPU workers communicate via the existing `mpsc::Sender<EvaluatorPayload>` pattern.
   - Move GPU batch loop to a separate thread that communicates via its own `mpsc::Sender<Drawing>`.
   - Main thread selects on both channels: whichever produces a new best first wins.

### Expected Gains

- On an 8-core machine: GPU continues at full throughput (~25M evals/sec) while 6 CPU threads add ~6000-12000 evals/sec (rough estimate based on CPU evaluator throughput at 256x256).
- CPU eval/sec is tiny compared to GPU, but **CPU evaluations are higher quality**: CPU uses Euclidean (L2) error and variable-vertex polygons. A CPU improvement may produce solutions the GPU's fixed-triangle L1 metric would never find.
- Primary benefit is **search diversity**, not raw throughput.

### Complexity

Medium. The main challenge is synchronizing the global best between CPU workers and GPU chains without excessive locking. The existing broadcast pattern for CPU is fine; the GPU side just needs periodic injection.


## Opportunity 2: CPU-to-GPU Migration (Seeding GPU Chains)

### Concept

When a CPU worker finds an improvement, inject that drawing into one or more GPU chains. This gives the GPU population high-quality starting points that the CPU's more nuanced mutation operators discovered.

### Implementation

1. Convert the CPU `Drawing` to `GpuDrawingState` via the existing `drawing_to_gpu()` function.
2. Write it into a specific chain's slot via `queue.write_buffer(&chain_states_buf, chain_id * GPU_DRAWING_STATE_SIZE, bytes)`.
3. The target chain immediately inherits the CPU solution and begins mutating from there.

### Target Chain Selection

- **Worst-chain replacement**: Read back all chain fitnesses (add a fitness staging buffer) and replace the worst chain.
- **Periodic slot**: Reserve chain IDs 0..3 as "CPU injection slots." Simpler, no extra readback needed.
- **Random replacement**: Pick a random chain. Cheapest, works well with 512 chains where losing one is negligible.

### Timing

Inject between GPU batches (after `run_batch` returns, before the next submission). The `queue.write_buffer` call is fast and non-blocking.

### Fitness Recalibration

CPU uses L2 (Euclidean) error; GPU uses L1 (sum of abs diff). When a CPU drawing is injected, its `fitness_bits` must be set to 0 so the GPU recomputes fitness on the next iteration using its own metric. This is already handled by `drawing_to_gpu()` which sets `fitness_bits = 0`.

### Expected Gains

- Prevents GPU population stagnation by injecting diverse solutions.
- Most impactful when the GPU is stuck in a local optimum and the CPU discovers a structurally different solution (e.g., different polygon ordering, different polygon count).


## Opportunity 3: GPU-to-CPU Migration (Feeding CPU Workers)

### Concept

When the GPU finds a new global best, broadcast it to CPU workers so they can start mutating from the GPU's best rather than their own local optima.

### Implementation

This mostly exists already in the CPU path's broadcast pattern. The change is:
1. When `run_batch()` returns `Some(drawing)`, also send it on the `broadcast::Sender<Drawing>` that CPU workers listen to.
2. CPU workers' existing `try_recv` in `produce_new_best()` already handles receiving a better drawing and resetting.

### Fitness Recalibration (Reverse Direction)

GPU drawings come back as triangles-only (3 vertices per polygon). The CPU evaluator will re-evaluate fitness using its L2 metric automatically in `produce_new_best()`. No special handling needed -- the CPU just treats it as another candidate.

### Format Concerns

GPU drawings from `gpu_to_drawing()` produce `Polygon` with exactly 3 points each. The CPU mutation operators (`add_polygon`, `remove_polygon`, `reorder_polygons`, per-polygon `mutate()`) all work fine with 3-point polygons. The CPU may add more vertices to these polygons over time (via `add_point` mutations), which the GPU cannot represent. This is actually a feature: the CPU can explore higher-vertex-count polygons while the GPU sticks to triangles.

### Expected Gains

- CPU workers get warm-started from GPU discoveries rather than searching independently.
- Faster convergence when GPU finds a good basin that CPU can then refine.


## Opportunity 4: Double-Buffered GPU Submission

### Concept

Instead of submit-wait-submit-wait (serial), use two sets of staging buffers so the CPU can prepare the next batch's parameters while the GPU processes the current batch.

### Current Bottleneck

In `run_batch()`:
1. `queue.write_buffer` (params, control flags, error accumulators) -- fast, non-blocking
2. Encode 50 iterations of 5 passes into command buffer -- CPU-bound, ~microseconds
3. `queue.submit` -- fast, non-blocking
4. `read_control_flags()` -> `device.poll(Maintain::Wait)` -- **blocks until GPU finishes**
5. Optional `readback_best()` -> another `device.poll(Maintain::Wait)` -- **blocks again**

The blocking poll in step 4 is where the CPU is wasted.

### Implementation

1. **Two staging buffers**: Create `control_staging_buf_a` and `control_staging_buf_b`. Alternate which one receives the copy-back each batch.
2. **Pipelined submission**:
   - Submit batch N using staging buffer A.
   - While GPU processes batch N, encode batch N+1's command buffer and write its parameters.
   - Poll for batch N's completion, read staging buffer A.
   - Submit batch N+1 using staging buffer B.
   - Repeat.

3. **Practical simplification**: Since command encoding is fast (~microseconds) and GPU batch execution is the dominant cost (~milliseconds), the real win here is overlapping the `readback_best` with the next batch. Currently, if a new best is found, `readback_best()` does another submit + wait, adding ~0.5-1ms of pipeline stall.

### Expected Gains

- Modest: maybe 5-10% throughput improvement by hiding readback latency.
- More impactful if batch sizes are reduced (e.g., batch=10 instead of 50), where the submit/poll overhead is a larger fraction.
- Not worth doing unless Opportunity 1 is too complex; the CPU worker approach yields much larger gains.


## Opportunity 5: CPU Handles Variable-Polygon-Count Structural Mutations

### Concept

The GPU is limited to fixed-size `array<Polygon, 1000>` with `polygon_count` tracking how many are active. GPU mutations can add/remove triangles, but structural exploration is limited by the shader's simple PCG RNG and fixed probability table. The CPU has richer mutation operators and can work with variable-vertex polygons.

### Architecture: Specialist Roles

- **GPU**: Fine-grained exploitation. Small mutations (micro_adjust, color tweaks, move_point) on triangulated drawings. High throughput, low diversity.
- **CPU**: Coarse-grained exploration. Structural mutations: add/remove polygons aggressively, reorder z-order, merge adjacent triangles back into quads, split polygons, topology changes.

### Implementation

1. Configure CPU workers with different `MutationParams` than the GPU -- much higher `add_polygon_prob`, `remove_polygon_prob`, `reorder_polygon_prob`.
2. CPU workers' discoveries get injected into GPU chains (Opportunity 2).
3. GPU's best gets periodically pulled by CPU for further structural mutation.

### Specific CPU-Only Mutations to Add

- **Triangle merging**: Take two adjacent GPU triangles sharing an edge and the same color, merge back into a quad. Reduces polygon count, opens up different mutation paths.
- **Polygon splitting**: Take a large quad and split into multiple triangles at different z-depths with slightly different colors (simulated transparency layering).
- **Region-focused mutation**: Identify the highest-error region in the image (from GPU error map readback) and concentrate CPU mutations there.

### Expected Gains

- Better structural exploration that the GPU's simple shader mutations cannot achieve.
- This is where the CPU adds the most value in a hybrid -- doing things the GPU fundamentally cannot do well (complex control flow, variable-length data structures).


## Opportunity 6: Async GPU + CPU Work During GPU Compute

### Concept

Use wgpu's async callbacks to get notified when GPU work completes, rather than blocking. The main thread (or a dedicated GPU thread) can do useful CPU work during the wait.

### What the CPU Can Do While Waiting

1. **PNG rendering for WebSocket clients**: Currently done after `run_batch()` returns. Could be done during the next GPU batch.
2. **Drawing serialization**: `serde_json::to_string(&global_best)` for WS state updates.
3. **File I/O**: Saving `best.json` to disk.
4. **WS message dispatch**: Processing incoming WS commands (param updates, project switches).
5. **Statistics computation**: Evals/sec calculations.

### Implementation

Replace the current synchronous loop:
```rust
// Current (blocking)
loop {
    if let Some(new_best) = evolver.run_batch(&mp) { ... }
    // CPU idle during run_batch!
    render_png_if_needed();
    update_ws_state();
    save_to_disk();
}
```

With an event-driven approach:
```rust
// Proposed (async)
loop {
    evolver.submit_batch(&mp);  // non-blocking

    // Do CPU work while GPU computes
    render_png_if_needed();
    update_ws_state();
    save_to_disk();
    process_ws_commands();

    let result = evolver.await_batch();  // block only when ready
    if let Some(new_best) = result { ... }
}
```

### Expected Gains

- Hides PNG encoding latency (~2-5ms at 256x256) behind GPU compute.
- Hides disk I/O latency behind GPU compute.
- Most impactful in headless mode where WS updates are frequent.
- Modest overall: these CPU tasks are infrequent (5 fps for images, 10s for saves) and the GPU batch (~10-20ms) is long enough that they already fit in the gaps. But it eliminates the occasional frame drop when PNG encoding and GPU readback collide.


## Opportunity 7: Error-Map-Guided CPU Mutations

### Concept

Read back the GPU's error accumulator data (or a per-pixel error map) and use it to guide CPU mutations toward high-error regions of the image.

### Implementation

1. Add a staging buffer for the full per-chain error map (or the best chain's rendered image).
2. After each GPU batch, read back the rendered image of the best chain.
3. Diff against reference on CPU to get a per-pixel error heatmap.
4. Pass this heatmap to CPU workers as a "mutation guidance map."
5. CPU mutations preferentially place new polygons, move vertices, and adjust colors in high-error regions.

### Required Changes to CPU Mutation Logic

- `Drawing::mutate()` would take an optional `&ErrorMap` parameter.
- `add_polygon()` places new polygon centroid in a high-error region (weighted random sampling).
- `move_point()` biases movement toward nearby high-error pixels.
- `change_color()` samples the reference image color at the polygon's centroid for initial color.

### Expected Gains

- Significantly better convergence in the CPU path -- instead of random blind mutations, the CPU is "smart" about where to focus.
- This is a well-known technique in evolutionary art (fitness-proportional region targeting).
- Main cost is the readback of the rendered image (~150KB at 256x256), but this only needs to happen every few seconds.


## Priority Ranking

| Priority | Opportunity | Impact | Effort | Notes |
|----------|-----------|--------|--------|-------|
| 1 | **Run CPU workers concurrently** (Opp 1) | High | Medium | Unlocks all other hybrid opportunities. Main thread no longer blocks. |
| 2 | **Bidirectional migration** (Opp 2+3) | High | Low | Building on Opp 1. `drawing_to_gpu` / `gpu_to_drawing` already exist. |
| 3 | **Specialist roles** (Opp 5) | Medium-High | Medium | Give CPU and GPU different mutation profiles. CPU does structural, GPU does fine-tuning. |
| 4 | **Async GPU submission** (Opp 6) | Medium | Low | Overlap PNG/IO with GPU compute. Quick win. |
| 5 | **Error-guided CPU mutations** (Opp 7) | Medium | High | Requires changes to mutation logic. Large payoff but significant code changes. |
| 6 | **Double-buffered GPU** (Opp 4) | Low | Medium | Marginal throughput gain. Only matters at small batch sizes. |


## Implementation Roadmap

### Phase 1: Concurrent CPU + GPU (1-2 days)

1. Move GPU evolution to a dedicated thread.
2. Spawn `num_cpus - 2` CPU evaluator threads.
3. Both communicate results to the main thread via separate channels.
4. Main thread merges results, updates global best, manages WS/IO.
5. Global best broadcast: when either side finds improvement, share with both populations.

### Phase 2: Bidirectional Migration (0.5 day)

1. CPU-to-GPU: When CPU finds a new best, `drawing_to_gpu()` it and `queue.write_buffer` into a random GPU chain slot.
2. GPU-to-CPU: When GPU finds a new best, broadcast via the existing `broadcast::Sender<Drawing>`.
3. Both paths already exist in code; this is mostly wiring.

### Phase 3: Specialist Mutation Profiles (0.5-1 day)

1. Create `MutationParams::cpu_explorer()` with high structural mutation rates.
2. Create `MutationParams::gpu_exploiter()` (already default, but could tune micro_adjust higher).
3. CPU workers use explorer params; GPU uses exploiter params.

### Phase 4: Async GPU + CPU Background Work (0.5 day)

1. Split `run_batch` into `submit_batch` (non-blocking) and `poll_result` (check completion).
2. Main thread does PNG/IO during GPU compute.
3. Falls back to blocking poll if GPU takes too long.

### Estimated Total Throughput Improvement

- Phase 1 alone: ~15-30% improvement in convergence rate (from search diversity, not raw evals).
- Phase 1+2: ~25-40% improvement (migration prevents both populations from stagnating).
- Full hybrid (Phase 1-4): ~40-60% improvement in time-to-quality, with the CPU providing structural innovations that the GPU then rapidly optimizes.

These are rough estimates. The actual improvement depends heavily on the image and current fitness level. At low fitness, GPU dominance means CPU adds little. At high fitness (>85%), structural mutations become more valuable and the CPU's contribution grows.
