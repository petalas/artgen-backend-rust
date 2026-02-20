# CPU-GPU Hybrid Architecture Analysis

This document analyzes the CPU-GPU interaction patterns in artgen-backend-rust and proposes optimization opportunities. The analysis focuses on the `--gpu --headless` path (`gpu_main_loop_headless`) which is the primary production mode, and the `GpuEvolver` subsystem in `src/gpu_evolver/`.

## Current Architecture Summary

### GPU Pipeline (4 passes per iteration, batched 50x)

```
Per batch (GPU_ITERATIONS_PER_BATCH = 50):

  for i in 0..50:
    [Mutate] -> [Rasterize+Error] -> [Select] -> [Migrate?]

  Copy control_flags, fitness_packed, timestamps -> staging buffers
  Submit single command buffer
  Issue map_async on staging
```

### CPU-GPU Synchronization Flow

```
run_batch() call N:
  1. finish_pending_readback() for batch N-1   <-- device.poll(Wait) stall
  2. Write params + control flags via queue.write_buffer()
  3. Encode 50 iterations into one command buffer
  4. queue.submit()
  5. Issue map_async for staging buffers
  6. Return batch N-1's result
```

The double-buffered staging (alternating index 0/1) means the CPU reads results from the *previous* batch while the GPU executes the *current* batch. This is the existing overlap mechanism.

### Main Loop Cadence (headless mode)

```
loop {
  lock mutex -> check switch_request, pause, benchmark
  lock mutex -> clone mutation_params
  evolver.run_batch()                     <-- blocks on previous batch
  if new_best -> lock mutex, update WsState
  every 200ms -> CPU rasterize + PNG encode + WsState update
  every 2s -> build_gpu_stats() + readback island drawings
  every 10s -> JSON/PNG save to disk
}
```

## Identified Optimization Opportunities

### 1. Eliminate Main-Loop Mutex Contention During GPU Batch

**Problem**: The main evolution loop acquires and releases `ws_state.0.lock()` multiple times per iteration:
- Check `switch_request` (line 1347)
- Check `benchmark_request` (line 1384)
- Check `paused` (line 1398)
- Clone `mutation_params` (line 1439)
- Update `WsState` on improvement (line 1466-1484)
- Update stats (line 1506-1517)

Each lock acquisition can contend with WS server threads that hold the lock while serializing JSON (potentially tens of milliseconds for large GPU stats payloads with 1024 chain fitness values).

**Proposal**: Consolidate all reads from `WsState` into a single lock acquisition at the top of each loop iteration. Extract `switch_request`, `benchmark_request`, `paused`, and `mutation_params` in one critical section. This reduces from 4+ lock acquisitions per loop iteration to 1 read + 1 write (for results).

```rust
// Consolidated read
let (should_switch, bench_req, is_paused, mp) = {
    let mut s = ws_state.0.lock().unwrap();
    let switch = s.switch_request.is_some();
    let bench = s.benchmark_request.take();
    let paused = s.paused;
    let mp = s.mutation_params.clone();
    (switch, bench, paused, mp)
};
```

**Impact**: Reduces lock contention by ~75%. Most beneficial when multiple WS clients are connected and generating serialization work.

### 2. Overlap CPU Work with GPU Execution

**Problem**: The current structure is:

```
[GPU batch N-1 readback + stall] -> [GPU batch N submit] -> [CPU idle/polling]
```

Between submitting batch N and calling `run_batch()` again, the CPU does useful work (PNG encoding, stats building, file I/O) but this work is *not* systematically overlapped with GPU execution. When there is no improvement and no stats interval, the CPU immediately calls `run_batch()` again, which stalls in `finish_pending_readback()`.

**Proposal**: Restructure the main loop into an explicit pipelined model where CPU work is *guaranteed* to overlap with GPU execution:

```rust
loop {
    // 1. Submit next GPU batch (non-blocking after first call)
    evolver.submit_batch(&mp);  // encodes + submits, returns immediately

    // 2. Do ALL CPU work while GPU runs
    do_cpu_work();  // PNG encode, stats, file I/O, WS updates

    // 3. Harvest previous batch results (may need brief poll)
    let result = evolver.collect_results();  // minimal stall if GPU finished during step 2
}
```

This means splitting `run_batch()` into two explicit phases: `submit_batch()` (encode + submit + map_async) and `collect_results()` (poll + readback). The current implementation already does this conceptually via `pending_batch`, but the API forces them into the same call, meaning the CPU must do all its work *after* the blocking readback.

**Impact**: On a typical iteration where GPU batch takes ~5ms and CPU work takes ~1-2ms, this could eliminate ~1-2ms of idle GPU time per batch. Over millions of batches, this adds up to measurably higher evals/sec.

### 3. Adaptive Batch Size Based on GPU Utilization

**Problem**: `GPU_ITERATIONS_PER_BATCH = 50` is a compile-time constant. The optimal batch size depends on:
- Image resolution (higher resolution = more work per rasterize_error pass)
- Chain count (more chains = more GPU work)
- Lambda (more offspring = larger rasterize dispatch)
- Whether the CPU needs to do work (stats, PNG, saves)

With small images (64x64) and few chains, 50 iterations may complete in <1ms, making the overhead of map_async/poll/readback proportionally large. With large images (512x512) and many chains, 50 iterations may take >50ms, during which the CPU is blocked and cannot respond to WS commands or update the display.

**Proposal**: Implement adaptive batch sizing based on measured GPU execution time:

```rust
struct BatchSizeController {
    current_batch_size: u32,
    target_batch_time_ms: f64,  // e.g., 10ms
    min_batch_size: u32,        // e.g., 10
    max_batch_size: u32,        // e.g., 500
}

impl BatchSizeController {
    fn adjust(&mut self, actual_time_ms: f64) {
        let ratio = self.target_batch_time_ms / actual_time_ms;
        let new_size = (self.current_batch_size as f64 * ratio) as u32;
        self.current_batch_size = new_size.clamp(self.min_batch_size, self.max_batch_size);
    }
}
```

The target batch time should be tuned to balance:
- **Throughput**: Larger batches amortize submission overhead
- **Responsiveness**: Smaller batches let the CPU respond to WS commands sooner
- **Overlap efficiency**: Batch time should be >= CPU work time for full overlap

A good starting target is 8-12ms (matching display refresh intervals).

**Impact**: Could improve throughput by 10-30% for extreme configurations (very small or very large images) and significantly improve UI responsiveness for large configurations.

### 4. Deferred and Batched Readback of Island Drawings

**Problem**: `build_gpu_stats()` calls `evolver.readback_chains()` which creates a temporary staging buffer, issues copy commands, submits, polls, and maps — all synchronously. This happens every 2 seconds and reads back one full `GpuDrawingState` (16KB) per island. With 4 islands, that is a 64KB synchronous GPU round-trip that blocks the evolution loop.

This readback calls `device.poll(Maintain::Wait)` which forces the GPU to drain *all* pending work, potentially stalling a running evolution batch.

**Proposal A — Async island readback**: Pre-allocate a persistent staging buffer for island readbacks (sized for max islands) and use the same double-buffered async pattern as the main readback. Piggyback the island copy commands onto the next evolution batch's command encoder:

```rust
// During batch encoding, if stats are due:
if stats_due {
    for (i, &chain_id) in island_chain_ids.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &p.chain_states_buf,
            chain_id as u64 * state_size,
            &p.island_staging_buf,
            i as u64 * state_size,
            state_size,
        );
    }
}
// Read island data from the *previous* batch's staging in finish_pending_readback()
```

**Proposal B — Reduce readback frequency**: Island thumbnails update at ~0.5 Hz in the UI. The 2-second stats interval could skip island readback on alternating cycles (every 4 seconds) since island best chains change slowly.

**Proposal C — Lightweight island readback**: Instead of reading back full 16KB `GpuDrawingState` per island, add a small GPU-side buffer that stores only the JSON-serializable fields needed for display (fitness, polygon count, mutation scale). This reduces the readback to ~32 bytes per island instead of 16KB.

**Impact**: Proposal A eliminates the synchronous stall entirely. Proposal C reduces data transfer by ~99.8%.

### 5. Non-blocking `readback_chain()` for New Best Drawing

**Problem**: When the control flags indicate `new_best_found`, `finish_pending_readback()` calls `readback_chain()` which performs a full synchronous GPU round-trip: encode copy -> submit -> poll(Wait) -> map -> read -> unmap. This 16KB readback blocks the main loop.

Since a new global best is found relatively rarely (once every few hundred batches for mature drawings), this is not a hot path — but it can cause UI jank because it adds 1-3ms of latency to the batch that found an improvement.

**Proposal**: Pre-allocate a dedicated staging buffer for best-chain readback. When `new_best_found` is detected during `finish_pending_readback()`, issue the copy command as part of the *next* batch's command encoder, and read the result in the *following* `finish_pending_readback()` call. This adds one batch of latency to improvement detection (50 iterations x ~0.1ms = ~5ms) but eliminates the synchronous stall.

```rust
struct GpuEvolver {
    // ... existing fields
    pending_best_readback: Option<u32>,  // chain_id to read back next batch
    best_readback_staging: Buffer,       // persistent staging buffer
}
```

**Impact**: Eliminates 1-3ms synchronous stall on improvement detection. Negligible latency cost (one batch delay).

### 6. CPU-Side Work Queue for Heavy Operations

**Problem**: PNG encoding (`encode_rgba_as_png`), CPU rasterization (`drawing.draw()`), JSON serialization (`serde_json::to_string`), and file I/O (`to_file`, `std::fs::write`) all run on the main evolution thread. These operations can take 5-20ms each, during which the GPU may be idle.

**Proposal**: Offload heavy CPU work to a dedicated worker thread, communicating via a bounded channel:

```rust
enum CpuWork {
    RenderAndEncodePng { drawing: Drawing, w: usize, h: usize },
    SaveToFile { drawing: Drawing, path: String },
    BuildGpuStats { /* ... */ },
}

// In main loop:
cpu_worker_tx.try_send(CpuWork::RenderAndEncodePng { ... }).ok();

// Worker thread:
while let Ok(work) = rx.recv() {
    match work {
        CpuWork::RenderAndEncodePng { drawing, w, h } => {
            let mut buf = vec![0u8; w * h * 4];
            drawing.draw(&mut buf, w, h, Rasterizer::HalfSpace);
            let png = encode_rgba_as_png(&buf, w, h);
            // Update WsState with new PNG
        }
        // ...
    }
}
```

Use a bounded channel (capacity 1-2) so the worker naturally back-pressures without unbounded queue growth.

**Impact**: Could improve effective GPU utilization by 10-20% by ensuring the main thread always has a batch ready to submit. Most impactful at higher resolutions where CPU rasterization for display is expensive.

### 7. Conditional Timestamp Collection

**Problem**: Timestamp queries are collected for every batch (`collect_timestamps: true` is always passed). Each batch resolves 8 timestamps into a staging buffer, copies to a double-buffered staging buffer, maps it, reads it, and unmaps it. The timestamp data is only consumed every 2 seconds when `print_averages()` is called, but the GPU/driver overhead of timestamp queries applies to every batch.

**Proposal**: Only collect timestamps for a small fraction of batches — enough to get statistically meaningful averages but without the per-batch overhead:

```rust
let collect_timestamps = batches % 10 == 0;  // every 10th batch
evolver.run_batch(&mp, collect_timestamps);
```

This still provides 5 samples per second at typical batch rates (50+ batches/sec), which is more than enough for meaningful timing averages.

**Impact**: Reduces per-batch GPU overhead by eliminating timestamp query set writes for 90% of batches. On some drivers, timestamp queries can add measurable overhead to compute dispatch latency.

### 8. Multi-Queue / Async Compute Exploitation

**Problem**: All GPU work (evolution compute + readback copies + island readback copies) is submitted to a single queue. wgpu currently exposes only one queue per device, but the underlying Vulkan driver may support multiple queue families (compute + transfer).

**Proposal (Future)**: When wgpu gains multi-queue support (tracked in [wgpu#1716](https://github.com/gfx-rs/wgpu/issues/1716)), use a dedicated transfer queue for staging copies while compute work continues on the compute queue:

```
Compute Queue: [Mutate] -> [Rasterize] -> [Select] -> [Migrate] ...
Transfer Queue:                    [Copy staging] -> [Map]
```

This is a long-term architectural consideration. For now, the single-queue constraint means all copies are serialized with compute dispatches.

**Near-term alternative**: Use `queue.write_buffer()` for small uniform updates (params, control flags) instead of staging + copy, as wgpu may internally use a transfer queue for `write_buffer()` on some backends. This is *already done* in the current code, so no change needed here.

**Impact**: Multi-queue would theoretically eliminate copy stalls entirely, but this is blocked on wgpu API support.

### 9. Multi-GPU Support Architecture

**Problem**: The system currently uses a single GPU device. Systems with multiple GPUs (e.g., discrete + integrated, or multi-GPU workstations) leave hardware unused.

**Proposal**: A multi-GPU architecture could partition chains across GPUs:

```rust
struct MultiGpuEvolver {
    evolvers: Vec<GpuEvolver>,  // one per GPU
    chains_per_gpu: Vec<u32>,
}
```

**Design considerations**:
- **Island mapping**: Each GPU runs a set of islands. Inter-island migration would require CPU-mediated data transfer (readback from GPU A, upload to GPU B).
- **Migration cost**: A full `GpuDrawingState` is 16KB. Migrating 1 chain between GPUs costs ~32KB of PCIe bandwidth (readback + upload). At 1000 iterations/migration interval, this is negligible.
- **Load balancing**: GPUs with different capabilities could run different chain counts. The adaptive batch sizing (proposal 3) would handle this naturally.
- **Synchronization**: Each GPU runs independently with its own `run_batch()` loop. The CPU collects results from all GPUs and identifies the global best.
- **wgpu support**: `Instance::enumerate_adapters()` already lists all GPUs (the code already does this in `pipeline.rs`). Creating a separate `GpuPipeline` per adapter is straightforward.

**Implementation sketch**:
1. Enumerate adapters, create one `GpuEvolver` per GPU
2. Partition total chain count across GPUs
3. Run independent batch loops (could be separate threads, one per GPU)
4. Periodically exchange best drawings between GPUs via CPU
5. Report combined evals/sec and global best

**Impact**: Near-linear throughput scaling for 2-GPU systems. Diminishing returns beyond 2 GPUs due to migration overhead and CPU bottlenecks in result collection.

### 10. Headless Mode GPU Scheduling Optimization

**Problem**: In headless mode, the only consumers of GPU results are:
- Evolution (continuous)
- WS stats updates (every 2s)
- WS image updates (every 200ms, only when improved)
- File saves (every 10s)

The GPU runs flat-out with no vsync or frame pacing, which is correct for throughput, but the polling strategy could be refined.

**Proposal**: Use `device.poll(Maintain::Poll)` (non-blocking) instead of `Maintain::Wait` when the CPU has other work to do, falling back to `Maintain::Wait` only when the CPU is idle:

```rust
fn finish_pending_readback_nonblocking(&mut self) -> Option<Option<Drawing>> {
    // Try non-blocking poll first
    self.pipeline.device.poll(Maintain::Poll);

    // Check if map_async completed
    match self.pending_map_receivers.as_ref() {
        Some(r) => match r.control_rx.try_recv() {
            Ok(_) => {
                // Data is ready, proceed with readback
                Some(self.do_readback())
            }
            Err(TryRecvError::Empty) => None,  // Not ready yet, CPU can do other work
            Err(TryRecvError::Disconnected) => panic!("map_async callback dropped"),
        }
        None => None,
    }
}
```

This converts the blocking wait into a polling loop that interleaves CPU work:

```rust
loop {
    evolver.submit_batch(&mp);

    // Try to harvest results while doing CPU work
    loop {
        if let Some(result) = evolver.try_collect_results() {
            break result;
        }
        // Do a chunk of CPU work
        do_next_cpu_task();
    }
}
```

**Impact**: Better CPU utilization during GPU execution. Most beneficial when there is significant CPU work to do (multiple WS clients, frequent stats updates, large images requiring expensive CPU rasterization).

### 11. Params Update Coalescing

**Problem**: `mutation_params` is read from the mutex and uploaded to the GPU via `queue.write_buffer()` on *every* batch, even when params haven't changed. The `GpuParams` struct is 128 bytes, so the per-batch cost is small, but the mutex acquisition adds latency.

**Proposal**: Track a generation counter on `MutationParams` and only re-upload when it changes:

```rust
struct GpuEvolver {
    // ... existing
    last_params_generation: u64,
    cached_params: GpuParams,
}

// In run_batch:
if params_generation != self.last_params_generation {
    self.cached_params = gpu_params_from(mp, ...);
    p.queue.write_buffer(&p.params_buf, 0, bytemuck::bytes_of(&self.cached_params));
    self.last_params_generation = params_generation;
}
```

**Impact**: Eliminates ~99% of params uploads (params change only on user interaction). Marginal throughput improvement, but cleaner separation of concerns.

## Priority Ranking

| # | Optimization | Effort | Impact | Risk |
|---|-------------|--------|--------|------|
| 2 | Overlap CPU work with GPU execution | Medium | High | Low |
| 3 | Adaptive batch sizing | Medium | High | Low |
| 6 | CPU work queue for heavy operations | Medium | High | Low |
| 1 | Mutex contention reduction | Low | Medium | Low |
| 4 | Deferred island readback | Medium | Medium | Low |
| 7 | Conditional timestamp collection | Low | Low-Medium | None |
| 11 | Params update coalescing | Low | Low | None |
| 5 | Non-blocking best readback | Medium | Low | Low |
| 10 | Non-blocking poll strategy | Medium | Medium | Medium |
| 9 | Multi-GPU support | High | High | Medium |
| 8 | Multi-queue compute | N/A (blocked) | High | N/A |

## Recommended Implementation Order

1. **Quick wins** (1, 7, 11): Low effort, low risk, measurable improvement
2. **Core pipeline improvement** (2, 3): Restructure the main loop for proper CPU-GPU overlap and adaptive batching
3. **Background workers** (6, 4): Offload CPU-heavy work to keep the GPU fed
4. **Advanced optimizations** (5, 10): Non-blocking readback patterns
5. **Hardware scaling** (9): Multi-GPU support when single-GPU throughput plateaus
