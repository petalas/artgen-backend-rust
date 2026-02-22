# CPU-GPU Hybrid Architecture Analysis

Updated 2026-02-22. Supersedes prior version. Analyzes the `--gpu --headless` path (`gpu_main_loop_headless` in `src/main.rs`) and the `GpuEvolver` subsystem (`src/gpu_evolver/`).

## Current Architecture Summary

### GPU Pipeline (3 passes per iteration, batched N times)

```
Per batch (gpu_batch_iters, default 50, configurable via UI/CLI):

  for i in 0..gpu_batch_iters:
    [Mutate] -> [BinPolygons?] -> [Rasterize+Error] -> [Select]

  Copy control_flags, fitness_packed, timestamps -> staging buffers[write_idx]
  Submit single command buffer
  Issue map_async on staging buffers[write_idx]
```

Key changes since prior analysis:
- Island model / migration pass **removed** -- chains are independent (1+lambda)-ES
- **Tile culling** added: optional bin_polygons pass bins triangles into screen tiles
- **Incremental eval** added: cached framebuffers per chain, delta error computation
- **Subgroup operations** added: `subgroupAdd()` for error reduction in rasterize_error
- **Bulk iteration consolidation**: all non-timestamped iterations go into ONE compute pass
- GpuParams is now a 256-byte uniform buffer (not 128-byte push constants)

### CPU-GPU Synchronization (Double-Buffered Staging)

```
run_batch() call N:
  1. finish_pending_readback() for batch N-1   <-- device.poll(Wait) stall
  2. pipeline.set_rasterize_wg() -- recreates pipeline if WG changed
  3. dispatch_init_framebuffers() -- if incremental_eval && !framebuffers_initialized
  4. Write params (256B) + control flags (16B) via queue.write_buffer()
  5. Encode gpu_batch_iters iterations into one command buffer
  6. Copy control_flags + fitness_packed + timestamps -> staging[write_idx]
  7. queue.submit()
  8. Issue map_async on staging[write_idx]
  9. Store pending_batch for next call
  10. Return batch N-1's result
```

Staging alternates between index 0 and 1. The CPU reads N-1's results while the GPU executes batch N.

### Main Loop Cadence (headless mode, `src/main.rs` line 1954)

```
inner loop {
  lock mutex -> check switch_request
  take benchmark_request
  check paused (sleep 50ms + continue if paused)
  lock mutex -> clone mutation_params
  evolver.run_batch(&mp, true)        <-- blocks on previous batch poll(Wait)
  if new_best { improvements++; image_dirty = true }
  every 200ms if image_dirty -> CPU rasterize + PNG encode + WsState update + disk save
  every 2s -> build_gpu_stats() + stats update
}
```

### Data Flow Sizes

| Buffer | Size | Direction | Frequency |
|--------|------|-----------|-----------|
| GpuParams uniform | 256 B | CPU->GPU | every batch |
| ControlFlags | 16 B | CPU->GPU (reset) + GPU->CPU (readback) | every batch |
| fitness_packed | active_chains * 4 B | GPU->CPU | every batch |
| timestamps | 48 B | GPU->CPU | every batch (when collected) |
| GpuDrawingState readback | 16,032 B | GPU->CPU | on new_best (rare) |
| chain_states reinit | chains * 16,032 B | CPU->GPU | on project switch / benchmark |

---

## Optimization Opportunities

### 1. Split `run_batch()` into Submit + Collect for True CPU-GPU Overlap

**Problem**: `run_batch()` calls `finish_pending_readback()` (blocking `device.poll(Wait)`) at the top, then does GPU work encoding and submission. The main loop does CPU work *after* `run_batch()` returns. This means:

```
Timeline (current):
  [CPU: poll+readback N-1] [CPU: encode+submit N] [CPU: main loop work] [GPU idle] [GPU: batch N]
                                                                          ^^^^^^^^
                                                                          GPU bubble
```

The GPU has a bubble between when it finishes batch N-1 and when batch N's submission reaches it. The CPU is doing main-loop work (mutex checks, PNG encoding, stats) during this bubble instead of having already submitted batch N.

**Proposal**: Split `run_batch()` into `submit_batch()` and `collect_results()`:

```rust
// NEW API:
impl GpuEvolver {
    /// Encode and submit a batch. Non-blocking (returns immediately after queue.submit).
    pub fn submit_batch(&mut self, mp: &MutationParams, collect_timestamps: bool);

    /// Poll for previous batch results. Blocks via device.poll(Wait).
    /// Returns Some(Drawing) if the previous batch found a new global best.
    pub fn collect_results(&mut self) -> Option<Drawing>;
}

// Main loop restructured:
loop {
    // 1. Harvest previous batch (blocks only briefly if GPU is done)
    let result = evolver.collect_results();

    // 2. Submit next batch IMMEDIATELY (GPU starts working)
    evolver.submit_batch(&mp, collect_timestamps);

    // 3. Do ALL CPU work while GPU runs batch N+1
    handle_result(result);
    do_stats_and_png_work();
}
```

The key insight: submit first, then do CPU work. The current code does CPU work *between* collect and submit, creating a GPU bubble.

**Code references**: `src/gpu_evolver/mod.rs` lines 203-420 (`run_batch`), lines 456-516 (`finish_pending_readback`). The internal state machine (`pending_batch`, `pending_map_receivers`, `staging_idx`) already supports this split.

**Impact**: High. Eliminates 1-5ms GPU idle time per batch. With batches running every 5-15ms, this is a 7-30% throughput improvement. Largest impact when CPU work per iteration is highest (PNG encoding at 200ms intervals, JSON serialization, file I/O).

### 2. CPU Worker Thread for Heavy Rendering / Serialization

**Problem**: The main evolution thread performs these CPU-heavy operations inline:
- `drawing.draw(&mut render_buf, w, h, Rasterizer::HalfSpace)` -- CPU rasterization for display (line 2076)
- `encode_rgba_as_png(&render_buf, w, h)` -- PNG encoding for WebSocket (line 2077)
- `serde_json::to_string(&global_best)` -- JSON serialization of full drawing (line 2093)
- `global_best.to_file(&json_filename)` -- JSON write to disk (line 2101)
- `std::fs::write(&png_path, &png)` -- PNG write to disk (line 2102)

At 384x384 resolution with 500+ polygons, CPU rasterization takes ~2-5ms and PNG encoding takes ~1-3ms. During this time, the GPU has already finished its batch and is idle waiting for the next submission.

**Proposal**: Spawn a single dedicated CPU worker thread. The main evolution thread sends work items via a bounded channel (capacity 1-2). The worker thread owns the WsState update for image data.

```rust
enum CpuWork {
    RenderAndUpdate {
        drawing: Drawing,
        w: usize,
        h: usize,
        save_to_disk: bool,
        json_path: String,
        png_path: String,
    },
}

// Main loop (simplified):
if image_dirty && last_image_render.elapsed() >= image_render_interval {
    image_dirty = false;
    // Non-blocking send; drop if worker is busy (bounded channel capacity 1)
    cpu_worker_tx.try_send(CpuWork::RenderAndUpdate {
        drawing: global_best.clone(),
        ...
    }).ok();
}
```

The worker thread does the rendering, PNG encoding, JSON serialization, WsState mutex updates, and disk I/O -- all on its own core, completely off the evolution hot path.

**Code references**: `src/main.rs` lines 2072-2108 (image render + save block), lines 2112-2136 (stats block).

**Impact**: High. Combined with proposal 1, this ensures the main thread does almost zero work between `collect_results()` and `submit_batch()`. The GPU bubble shrinks to just the mutex read for `mutation_params` (~1 microsecond). Most impactful at higher resolutions and more polygons.

### 3. Async Non-Blocking Best-Chain Readback

**Problem**: When `finish_pending_readback()` detects `new_best_found`, it calls `readback_chain()` which does a full synchronous GPU round-trip (`src/gpu_evolver/mod.rs` lines 529-564):

```rust
// SYNCHRONOUS: encode copy -> submit -> poll(Wait) -> map -> read -> unmap
fn readback_chain(&self, chain_id: u32) -> Drawing {
    // 1. Copy chain_states[chain_id] -> readback_staging_buf
    // 2. queue.submit()
    // 3. device.poll(Wait)   <-- BLOCKS
    // 4. map + read + unmap
}
```

This adds 0.5-3ms of blocking time on every improvement. Worse, the `device.poll(Wait)` call here can force the GPU to drain the *just-submitted* batch N's command buffer (from step 7 of `run_batch`), eliminating the overlap benefit.

**Proposal**: Piggyback the best-chain copy onto the *next* batch's command encoder:

```rust
struct GpuEvolver {
    pending_best_chain_id: Option<u32>,   // set when new_best detected
    best_readback_staging: [Buffer; 2],   // double-buffered like other staging
}

// In submit_batch(), if pending_best_chain_id is set:
encoder.copy_buffer_to_buffer(
    &p.chain_states_buf,
    chain_id as u64 * state_size,
    &p.best_readback_staging[write_idx],
    0,
    state_size,
);

// In collect_results(), if previous batch had a best readback:
//   map + read the staging buffer from the PREVIOUS batch (already polled)
```

This adds one batch of latency to improvement detection (~5-15ms), which is imperceptible to the user but eliminates the synchronous stall entirely.

**Code references**: `src/gpu_evolver/mod.rs` lines 509-513 (where `readback_chain` is called), lines 529-564 (`readback_chain` implementation).

**Impact**: Medium. Eliminates 0.5-3ms synchronous stall per improvement. Most impactful during early evolution when improvements are frequent (every few batches).

### 4. Adaptive Batch Size Controller

**Problem**: `gpu_batch_iters` is user-configurable (default 50) but static during evolution. The optimal batch size depends heavily on configuration:

| Config | Optimal batch iters | Reason |
|--------|-------------------|--------|
| 64x64, 4 chains, lambda=1 | 200-500 | Each iteration is <0.05ms; amortize readback overhead |
| 384x384, 4 chains, lambda=64 | 20-50 | Each iteration is ~2ms; total batch is 40-100ms |
| 512x512, 64 chains, lambda=64 | 5-10 | Each iteration is ~50ms; large batches block CPU too long |

With the wrong batch size, either (a) readback overhead dominates (too small) or (b) the CPU cannot respond to WS commands for hundreds of ms (too large).

**Proposal**: Auto-tune batch size to target ~8-12ms per batch:

```rust
struct BatchSizeController {
    current: u32,
    target_ms: f64,     // default 10.0
    ewma_ms: f64,       // exponential weighted moving average of actual batch time
    alpha: f64,         // EWMA smoothing factor (0.2)
}

impl BatchSizeController {
    fn update(&mut self, actual_ms: f64) {
        self.ewma_ms = self.alpha * actual_ms + (1.0 - self.alpha) * self.ewma_ms;
        let ratio = self.target_ms / self.ewma_ms;
        let new = (self.current as f64 * ratio).round() as u32;
        self.current = new.clamp(4, 4096);
        // Enforce power-of-2 by rounding to nearest
        self.current = 1 << (self.current as f64).log2().round() as u32;
    }
}
```

The EWMA avoids oscillation. The target of 10ms balances throughput (amortized overhead) with responsiveness (UI updates every 200ms, so 20 batches between image updates).

**Code references**: `src/main.rs` line 2053 (`evolver.run_batch(&mp, true)`), `src/gpu_evolver/mod.rs` line 219 (`let iterations = mutation_params.gpu_batch_iters.max(1)`).

**Impact**: Medium-High. Up to 2x throughput improvement for small-image configs where overhead dominates. Up to 5x better responsiveness for large-image configs.

### 5. Conditional Timestamp Collection

**Problem**: Every batch passes `collect_timestamps: true` (`src/main.rs` line 2053). Timestamps are only consumed every 2 seconds for stats display. With 50+ batches/second, 98%+ of timestamp data is collected and discarded.

Each timestamped batch adds:
- 3 separate compute passes for the last iteration (instead of folding into the bulk pass)
- `resolve_query_set` + `copy_buffer_to_buffer` for timestamp staging
- `map_async` + readback for timestamp staging buffer
- Potential driver overhead from timestamp query set writes

**Proposal**: Collect timestamps only every Nth batch:

```rust
let collect_timestamps = batches % 20 == 0;
evolver.run_batch(&mp, collect_timestamps);
```

This gives ~2-3 timestamp samples per stats interval (2 seconds), which is sufficient for meaningful averages.

**Code references**: `src/main.rs` line 2053, `src/gpu_evolver/mod.rs` lines 278-308 (bulk vs timestamped passes), lines 371-380 (timestamp resolve + copy).

**Impact**: Low-Medium. Eliminates 3 extra compute pass transitions per batch for 95% of batches. The biggest win is avoiding the forced separation of the last iteration into 3 separate passes, which means ALL iterations go into the single bulk compute pass. On NVIDIA drivers, fewer VkCmdBeginComputePass/End transitions means less driver overhead.

### 6. Consolidate Mutex Acquisitions Per Loop Iteration

**Problem**: The inner evolution loop acquires `ws_state.0.lock()` at least 3 times per iteration:
- Check `switch_request` (line 1957)
- Check `benchmark_request` (line 1994)
- Check `paused` (line 2011)
- Clone `mutation_params` (line 2052)

WS server threads (`ws_handle_client`) also acquire this mutex to serialize JSON for clients. Large `gpu_stats` payloads with up to 1024 chain fitness values can hold the lock for several milliseconds during serialization.

**Proposal**: One read lock per iteration:

```rust
let (should_break, bench_req, is_paused, mp) = {
    let mut s = ws_state.0.lock().unwrap();
    let brk = s.switch_request.is_some();
    let bench = s.benchmark_request.take();
    let paused = s.paused;
    let mp = s.mutation_params.clone();
    (brk, bench, paused, mp)
};
if should_break { /* handle switch */ }
if let Some(req) = bench_req { /* handle benchmark */ }
if is_paused { /* sleep + continue */ }
// Evolution proceeds with `mp`, no further locks needed until results are ready
```

**Impact**: Low-Medium. Reduces lock contention from ~4 acquisitions to 1 per iteration. Most noticeable with multiple active WS clients.

### 7. GPU-Side Fitness Summary Buffer (Avoid Chain Readback for Stats)

**Problem**: `build_gpu_stats()` (called every 2 seconds) reads back chain fitness values that are already available in the `fitness_packed` staging buffer from the regular batch readback. However, it also calls `evolver.readback_chains()` to read back full 16KB `GpuDrawingState` for each chain's drawing -- this is used only for the viewer UI's chain thumbnail display.

Actually, looking at the code more carefully, the stats path in the headless loop (`src/main.rs` lines 2112-2136) does NOT call `readback_chains()`. It only reads `chain_fitness` (already available from the batch readback), computes `evals_per_sec`, and builds a `GpuStatsWs` struct. So this is already efficient.

The expensive `readback_chains()` exists for benchmarks and project switches, where it is necessary. No optimization needed here.

**Status**: Already well-optimized in the current code. The `fitness_packed_buf` readback in `finish_pending_readback` provides per-chain fitness without extra round-trips.

### 8. Deduplicate `queue.write_buffer` for Unchanged Params

**Problem**: Every batch writes the full 256-byte `GpuParams` via `queue.write_buffer()` (`src/gpu_evolver/mod.rs` lines 228-231), even when nothing has changed since the last batch. This is a DMA transfer that may contend with compute dispatch on some drivers.

The `iteration_number` field changes every batch, so a naive "did params change?" check won't work. However, `iteration_number` is only used by the mutate shader for RNG seeding -- it could be moved to a separate small buffer or push constant.

**Proposal**: Split `iteration_number` out of `GpuParams` and into a 4-byte push constant or a separate tiny uniform. Then only re-upload the full 256-byte params when the user changes a setting (tracked by a generation counter on `MutationParams`). The 4-byte iteration counter is written every batch but costs almost nothing.

**Impact**: Low. The 256-byte write is already cheap. More of an architectural cleanliness improvement.

### 9. Hybrid CPU+GPU Evolution (Independent Paths Working Together)

**Problem**: The CPU evolution path (`src/evaluator.rs`) and GPU path (`src/gpu_evolver/`) currently run independently -- `--gpu` mode uses only the GPU, and the default mode uses only CPU worker threads. The CPU cores sit idle during GPU evolution except for the main loop thread.

**Proposal**: Run CPU worker threads *alongside* the GPU evolver. CPU workers evolve independently using the same reference image and broadcast channel pattern as `cpu_main()`. When a CPU worker finds an improvement, the main loop uploads it to the GPU via `reinit_chains()` (or a more targeted single-chain injection). When the GPU finds an improvement, it broadcasts to CPU workers.

This is architecturally complex because:
1. CPU and GPU use different fitness metrics (CPU fitness comes from `evaluator.rs` with potential floating-point differences)
2. CPU uses multi-vertex polygons; GPU uses triangulated versions
3. `reinit_chains()` is expensive (uploads all chains + resets everything)

A simpler hybrid approach: use 1-2 CPU threads for **crossover exploration**. The GPU is bad at crossover (it disrupts all chains' cached framebuffers for incremental eval). Instead, periodically read back top-K chain drawings from the GPU, perform CPU-side crossover between them, and inject promising crossover offspring back into the GPU population.

```rust
// CPU crossover thread:
loop {
    let parents = gpu_evolver.readback_chains(&top_k_chain_ids);
    for _ in 0..crossover_attempts {
        let offspring = crossover(&parents[rng.gen_range(0..k)], &parents[rng.gen_range(0..k)]);
        let fitness = cpu_evaluate(&offspring, &ref_image);
        if fitness > worst_chain_fitness {
            inject_queue.send(offspring);
        }
    }
    sleep(Duration::from_secs(5));
}
```

**Impact**: Medium. Crossover is the one mutation type that incremental eval cannot accelerate (it requires full re-rasterization). Offloading it to CPU while GPU focuses on single-polygon mutations exploits each architecture's strength.

### 10. Multi-GPU Support

**Problem**: The system uses a single GPU device. The adapter enumeration code in `src/gpu_evolver/pipeline.rs` (lines 104-119) already lists all adapters but selects only the best one.

**Proposal**: Create one `GpuEvolver` per adapter, each running on its own thread with its own batch loop. A coordinator thread:
1. Collects results from all evolvers via channels
2. Identifies the global best across all GPUs
3. Periodically migrates the global best to all GPUs via `queue.write_buffer` to a single chain slot

```rust
struct MultiGpuCoordinator {
    evolvers: Vec<(GpuEvolver, JoinHandle<()>)>,
    result_rx: mpsc::Receiver<(usize, Option<Drawing>)>, // (gpu_id, new_best)
    global_best: Drawing,
}
```

Each GPU runs independently with its own chain count (proportional to its compute capability). Migration is lightweight: one 16KB write per GPU per migration interval.

**Key design decisions**:
- Each GPU gets its own wgpu `Device` and `Queue` (separate Vulkan device contexts)
- The coordinator thread does NOT touch wgpu at all -- it only handles Drawing structs
- Migration frequency: every 5-10 seconds (slow enough to not disrupt local exploration)

**Impact**: Near-linear throughput scaling. A system with both an RTX 5090 and an integrated GPU would get the 5090's full throughput plus ~10% from the iGPU. Primarily useful for multi-GPU workstations.

### 11. Reduce `readback_chain()` Stall on Improvement via Pre-allocated Staging

**Problem**: The current `readback_chain()` creates a new command encoder, submits, and blocks (`src/gpu_evolver/mod.rs` lines 529-564). This allocates Vulkan command buffer objects on the hot path.

Even if we adopt proposal 3 (async best readback), there are other callers of `readback_chain()` and `readback_chains()`:
- `evaluate_chain_fitness()` (benchmarks)
- `prepare_for_benchmark()`
- Project switching

These are not hot-path but could benefit from reusing pre-allocated command encoders.

**Proposal**: Minimal -- just pre-allocate the staging buffer once (already done as `readback_staging_buf`). The main improvement here is proposal 3.

**Impact**: Low. Readback for non-hot-path operations is acceptable as-is.

### 12. Buffer Memory Optimization for Incremental Eval

**Problem**: The `chain_framebuffers_buf` is `chain_count * W * H * 4` bytes. At 384x384 with 1024 max chains, this is 1024 * 384 * 384 * 4 = 603 MB. This is clamped by `max_storage_buffer_binding_size` (typically 2GB on desktop Vulkan), but it still dominates GPU memory usage.

In practice, the active chain count is typically 4-16, meaning 99%+ of the framebuffer memory is wasted.

**Proposal**: Allocate `chain_framebuffers_buf` sized for `active_chain_count` (not `max_chain_count`). When `active_chain_count` changes (user adjusts chain count), reallocate the buffer and recreate the affected bind groups. This requires:

1. Add `set_active_chain_count()` to `GpuPipeline` that recreates `chain_framebuffers_buf`, `chain_total_errors_buf`, and their bind groups
2. Call it when `mutation_params.chain_count` changes
3. Reset `framebuffers_initialized = false` to trigger re-initialization

**Impact**: Medium. Reduces GPU memory usage by 90%+ for typical configs (4-16 active chains vs 1024 max). Frees VRAM for other purposes and may improve memory bandwidth due to less cache pressure.

---

## Priority Ranking

| # | Optimization | Effort | Impact | Risk |
|---|-------------|--------|--------|------|
| 1 | Split submit/collect for GPU overlap | Medium | **High** | Low |
| 2 | CPU worker thread for rendering/encoding | Medium | **High** | Low |
| 4 | Adaptive batch size controller | Medium | **Medium-High** | Low |
| 3 | Async best-chain readback | Low-Medium | **Medium** | Low |
| 5 | Conditional timestamp collection | Low | **Low-Medium** | None |
| 6 | Mutex consolidation | Low | **Low-Medium** | None |
| 9 | Hybrid CPU+GPU crossover | High | **Medium** | Medium |
| 12 | Dynamic framebuffer allocation | Medium | **Medium** (memory) | Medium |
| 10 | Multi-GPU support | High | **High** | Medium |
| 8 | Params dedup | Low | **Low** | None |

## Recommended Implementation Order

1. **Split submit/collect (1) + CPU worker (2)**: These two together give the biggest single improvement. Implement 1 first, then 2 builds on top. Expected 15-30% throughput improvement combined.

2. **Quick wins (5, 6)**: Conditional timestamps and mutex consolidation are 30-minute changes. Do them alongside or immediately after.

3. **Adaptive batching (4)**: Requires timing infrastructure from the submit/collect split. Natural follow-on.

4. **Async best readback (3)**: Clean up the last synchronous stall in the hot path.

5. **Hybrid crossover (9)**: Exploratory -- try it after the pipeline is well-optimized. May or may not help depending on how often crossover is beneficial at convergence.

6. **Multi-GPU (10)**: Only when single-GPU throughput is fully optimized and the user has multiple GPUs.

7. **Dynamic framebuffer allocation (12)**: Nice memory optimization but not throughput-critical unless VRAM is constrained.
