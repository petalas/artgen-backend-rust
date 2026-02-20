# CPU-GPU Hybrid Architecture: Performance Analysis

Analysis of the CPU-GPU interaction patterns in the GPU evolution pipeline, focused on identifying bottlenecks and suggesting concrete improvements.

## Architecture Overview

The GPU evolution pipeline (`src/gpu_evolver/mod.rs`) runs a batch loop:

1. CPU writes `GpuParams` + `ControlFlags` + error accumulator zeros via `queue.write_buffer`
2. CPU encodes 50 iterations of (mutate -> rasterize_error -> select -> migrate) into **one command buffer**
3. CPU submits the command buffer
4. CPU **synchronously blocks** waiting for three sequential readbacks:
   - `control_staging_buf` (16 bytes) -- determines if a new global best was found
   - `timestamp_staging_buf` (64 bytes) -- GPU pass timings
   - `fitness_staging_buf` (N*4 bytes) -- per-chain fitness values
5. If `new_best_found`, CPU issues a **second submit** to copy one chain's `GpuDrawingState` (48,032 bytes) and blocks again
6. CPU returns to step 1

This means the GPU is **idle** between every batch while the CPU reads back results and prepares the next submission.

---

## Issue 1: Triple Synchronous Map/Unmap Creates a Pipeline Bubble

**Current code** (`mod.rs` lines 304-317):
```rust
let control = self.read_control_flags();   // map + device.poll(Wait) + unmap
self.read_timestamps(migrate_ran);          // map + device.poll(Wait) + unmap
self.read_chain_fitness();                  // map + device.poll(Wait) + unmap
```

Each `read_*` method calls `map_async` then `device.poll(Maintain::Wait)`, which blocks the CPU thread until the GPU finishes the entire submission AND the buffer is mapped. Since timestamps and fitness are already staged by the same command buffer, a single `device.poll(Wait)` call would suffice for all three buffers.

**Suggested fix -- batch all readbacks into one poll:**

```rust
// Map all three staging buffers before polling
let ctrl_slice = p.control_staging_buf.slice(..);
let ts_slice = p.timestamp_staging_buf.slice(..);
let fit_slice = p.fitness_staging_buf.slice(..((active * 4) as u64));

let (tx1, rx1) = std::sync::mpsc::channel();
let (tx2, rx2) = std::sync::mpsc::channel();
let (tx3, rx3) = std::sync::mpsc::channel();

ctrl_slice.map_async(MapMode::Read, move |r| { tx1.send(r).unwrap(); });
ts_slice.map_async(MapMode::Read, move |r| { tx2.send(r).unwrap(); });
fit_slice.map_async(MapMode::Read, move |r| { tx3.send(r).unwrap(); });

// ONE poll wakes all three
p.device.poll(Maintain::Wait);

rx1.recv().unwrap().unwrap();
rx2.recv().unwrap().unwrap();
rx3.recv().unwrap().unwrap();

// Read all three, then unmap all three
let ctrl_data = ctrl_slice.get_mapped_range();
let ts_data = ts_slice.get_mapped_range();
let fit_data = fit_slice.get_mapped_range();
// ... process ...
drop(ctrl_data); drop(ts_data); drop(fit_data);
p.control_staging_buf.unmap();
p.timestamp_staging_buf.unmap();
p.fitness_staging_buf.unmap();
```

**Impact:** Eliminates 2 of 3 `device.poll(Wait)` calls per batch. On some drivers, each poll has non-trivial overhead (kernel transition, fence wait). This alone could shave 0.1-0.5ms per batch.

---

## Issue 2: GPU Idle During CPU Processing (No Double-Buffering)

The biggest performance issue. The current flow is strictly sequential:

```
GPU: [=== batch N compute ===]...........[=== batch N+1 compute ===]
CPU: .........................[readback N][prepare N+1]
                              ^--- GPU idle here ---^
```

With double-buffered staging buffers, the CPU could submit batch N+1 while reading back batch N:

```
GPU: [=== batch N compute ===][=== batch N+1 compute ===][=== batch N+2 ===]
CPU: [readback N-1]...........[readback N]................[readback N+1]
```

**Implementation sketch:**

Create two sets of staging buffers (`control_staging_buf_a/b`, `timestamp_staging_buf_a/b`, `fitness_staging_buf_a/b`). Alternate which set the command encoder copies into. After submitting batch N (into staging set A), immediately start encoding batch N+1 (into staging set B) while mapping set A asynchronously.

```rust
struct DoubleBuffered {
    staging: [StagingSet; 2],
    current: usize,    // which set the GPU is writing to
}

fn run_batch(&mut self, ...) {
    let read_idx = self.current;       // staging set from PREVIOUS batch
    let write_idx = 1 - self.current;  // staging set for THIS batch

    // 1. Start mapping previous batch's staging buffers (non-blocking)
    self.staging[read_idx].map_all_async();

    // 2. Encode + submit current batch (writes to staging[write_idx])
    self.encode_and_submit(write_idx, ...);

    // 3. NOW block to read previous batch's results
    self.staging[read_idx].poll_and_read();

    self.current = write_idx;
}
```

**Impact:** The GPU stays busy while the CPU processes readback + WS state updates + PNG encoding. With 128 chains at 512x512, the rasterize_error pass alone takes several milliseconds per batch -- currently wasted waiting for CPU.

**Cost:** Doubles staging buffer memory (currently ~48KB for one chain readback + 16 bytes control + 64 bytes timestamps + N*4 fitness). At 512 chains this is ~50KB extra, negligible.

---

## Issue 3: Unnecessary Per-Batch Readback of Timestamps

Timestamp queries are resolved and read back on **every** batch (50 iterations), but they're only printed every 2 seconds (`last_stats_timestamp` check in `main.rs` lines 1471-1495). The GPU pass timings are averaged over many batches.

**Current:** The `read_timestamps` method maps the timestamp staging buffer every batch even when the data won't be displayed.

**Suggested fix:** Only resolve timestamps on the last iteration of each stats window (or every Nth batch). Set a flag before the batch:

```rust
let need_timestamps = last_stats_timestamp.elapsed().as_secs() >= 2;
// In encode loop, only add timestamp_writes when need_timestamps is true
// Only resolve_query_set + copy to staging when need_timestamps
// Only call read_timestamps when need_timestamps
```

The timestamp query set slots are still written (they're part of the compute pass descriptor), but the `resolve_query_set` + `copy_buffer_to_buffer` + map/read cycle can be skipped. This eliminates the staging buffer map overhead for ~99% of batches.

**Impact:** Small per-batch savings (~0.05ms), but it simplifies the double-buffering implementation.

---

## Issue 4: `build_gpu_stats` Does N Island Readbacks

In `main.rs` lines 937-939:
```rust
let island_drawings: Vec<Drawing> = island_stats.iter()
    .map(|is| evolver.readback_chain(is.best_chain_id))
    .collect();
```

Each `readback_chain` call creates a new command encoder, submits it, maps a staging buffer, blocks on `device.poll(Wait)`, reads 48,032 bytes, and unmaps. With 8 islands, that's **8 sequential GPU submissions** just for the stats readback every 2 seconds.

**Suggested fix -- batch all island readbacks into one submission:**

Create a staging buffer large enough for `island_count` drawing states (or reuse a pre-allocated one). Copy all N chains' states in a single command encoder, submit once, map once, read all N.

```rust
fn readback_chains(&self, chain_ids: &[u32]) -> Vec<Drawing> {
    let mut encoder = ...;
    for (i, &chain_id) in chain_ids.iter().enumerate() {
        let src_offset = chain_id as u64 * GPU_DRAWING_STATE_SIZE as u64;
        let dst_offset = i as u64 * GPU_DRAWING_STATE_SIZE as u64;
        encoder.copy_buffer_to_buffer(
            &self.chain_states_buf, src_offset,
            &self.multi_readback_staging_buf, dst_offset,
            GPU_DRAWING_STATE_SIZE as u64,
        );
    }
    p.queue.submit(std::iter::once(encoder.finish()));
    // Single map + poll + unmap
    // ...
}
```

**Impact:** Reduces 8 submit+poll cycles to 1. At 8 islands, this saves ~7 round-trips to the GPU driver, potentially 1-5ms every 2 seconds. More importantly, it prevents GPU pipeline stalls during the stats collection window.

---

## Issue 5: Error Accumulator Reset via `write_buffer`

In `mod.rs` lines 165-166:
```rust
let zeros = vec![0u8; active as usize * 4];
p.queue.write_buffer(&p.error_accumulators_buf, 0, &zeros);
```

This allocates a `Vec` and does a CPU-to-GPU transfer every batch. Since the select shader already does `atomicExchange(&error_accumulators[chain_id], 0u)` (line 88 of `select.wgsl`), the error accumulators are already zeroed after select runs. The `write_buffer` call is redundant for iterations 2-50 within a batch -- it only matters for iteration 1.

But since the command buffer contains 50 iterations with select clearing the accumulators each time, the only accumulator values that matter at submission time are the ones before iteration 1. The `atomicExchange` in select already resets them.

**Verification needed:** Confirm that after the select pass of the previous batch's last iteration, all error accumulators are 0. If so, the `write_buffer` for zeros can be removed entirely (saving a small CPU allocation + GPU DMA per batch). The first batch after `reinit_chains` would need the buffer zeroed, but that's already handled by the buffer being freshly created.

**Impact:** Removes one `queue.write_buffer` call and one `Vec` allocation per batch. Minor but free.

---

## Issue 6: CPU-GPU Co-Evolution Opportunity

Currently, the CPU path and GPU path are mutually exclusive (`--gpu` flag). The CPU worker threads sit idle when GPU mode is active, yet the CPU has `num_cpus` cores available.

**Strategy: CPU seeds the GPU with improved candidates**

While the GPU runs batches of (1+1) evolution across 128+ chains, the CPU could run a small number of worker threads doing higher-quality evolution (e.g., larger mutations, multi-step hill climbing) on the current global best. When a CPU worker finds an improvement:

1. Convert the CPU `Drawing` to `GpuDrawingState` via `drawing_to_gpu`
2. `queue.write_buffer` to inject it into the worst-performing chain's slot in `chain_states_buf`
3. The GPU pipeline naturally picks it up on the next iteration

This is essentially **heterogeneous island migration**: CPU islands evolve with different strategies (broader search, different mutation distributions) while GPU islands do fast narrow search.

**Implementation complexity:** Medium. Requires:
- Spawning CPU evaluator threads alongside the GPU evolver
- Using the existing `Evaluator` struct with CPU rasterization
- A small channel to send CPU improvements to the GPU main loop
- Periodic injection of CPU-found drawings into GPU chain states via `write_buffer`

**Impact:** Could significantly improve exploration diversity. The CPU workers cost no GPU time and the `write_buffer` injection is a single DMA of 48KB per improvement found.

---

## Issue 7: `write_buffer` for Params Every Batch

In `mod.rs` lines 150-152:
```rust
let mut params = gpu_params_from(mutation_params, ...);
params.iteration_number = self.iteration;
p.queue.write_buffer(&p.params_buf, 0, bytemuck::bytes_of(&params));
```

This reconstructs the full 128-byte `GpuParams` struct and writes the entire uniform buffer every batch, even when only `iteration_number` changes. When mutation params haven't changed (the common case during active evolution), this is wasteful.

**Suggested fix:** Cache the previous `MutationParams` and only rebuild + write when it actually changes. When only `iteration_number` changes, use a partial write:

```rust
// Only write the 4-byte iteration_number at its offset
let offset = offset_of!(GpuParams, iteration_number) as u64;
p.queue.write_buffer(&p.params_buf, offset, &self.iteration.to_le_bytes());
```

However, `write_buffer` has alignment requirements (offset must be a multiple of 4 for wgpu). The `iteration_number` field is at byte offset 24 (6th u32), which satisfies this. The minimum write size is also 4 bytes.

**Impact:** Eliminates 128 bytes of CPU-to-GPU transfer per batch in the common case. Marginal, but demonstrates the principle.

---

## Issue 8: Batch Size Optimization

The current `GPU_ITERATIONS_PER_BATCH = 50` (in `settings.rs` line 43) was likely chosen as a reasonable default. The tradeoffs:

- **Larger batches** (100-200): Amortize submission + readback overhead over more iterations. GPU stays busy longer per submission. But: longer latency before detecting a new global best, and the `ControlFlags` mechanism can only report ONE best per batch (the last chain to improve via `atomicMax`). Improvements within the batch that are later superseded are lost to the CPU.
- **Smaller batches** (10-25): Lower latency for detecting improvements, more responsive UI updates. But: higher per-iteration overhead from submission + readback.

**Analysis:** With the current synchronous readback, the overhead per batch is dominated by 3 `device.poll(Wait)` calls + map/unmap cycles. If Issue 2 (double-buffering) is implemented, the readback cost effectively overlaps with compute, making larger batches less important.

**Recommendation:** After implementing double-buffering, profile with batch sizes of 25, 50, and 100. The optimal value depends on the ratio of compute time to readback latency. For 128 chains at 512x512, the rasterize_error pass dominates, so 50 iterations likely keeps the GPU busy for 10-50ms -- well above the readback latency.

---

## Issue 9: Mutex Lock Contention on WS State

In the headless main loop (`main.rs` lines 1410-1412):
```rust
let mp = ws_state.0.lock().unwrap().mutation_params.clone();
if let Some(new_best) = evolver.run_batch(&mp) {
```

The `mutation_params` is cloned from a `Mutex<WsState>` **every batch**. The WsState mutex is also acquired by the WS server thread for reads and command handling. While the lock is held briefly, this creates potential contention:

- The evolution loop acquires the lock ~20-100 times per second (once per batch)
- Each WS client thread acquires it ~30 times per second for stats updates
- Command handling (pause, parameter changes) also acquires it

**Suggested fix:** Use an `Arc<AtomicBool>` dirty flag for mutation params. Only re-read params when the flag is set:

```rust
// In the main loop:
if params_dirty.load(Ordering::Relaxed) {
    mp = ws_state.0.lock().unwrap().mutation_params.clone();
    params_dirty.store(false, Ordering::Relaxed);
}
```

Or use `tokio::sync::watch` to broadcast parameter changes without polling a mutex.

**Impact:** Reduces mutex acquisitions in the hot loop from every-batch to only-when-changed. The `Mutex` is not heavily contended in practice (held for microseconds), so this is a minor optimization.

---

## Issue 10: Chain State Readback Only When Global Best Found

Currently, the full `GpuDrawingState` readback (48,032 bytes + a separate GPU submission) only happens when `control.new_best_found != 0`. This is already well-optimized. However, there's a subtlety: the `ControlFlags` mechanism uses `atomicMax` on `best_fitness_bits`, so it tracks the single highest-fitness chain across the entire batch. If two chains both improve past the previous global best, only the last one to write wins the `atomicMax`.

This is fine for correctness (the CPU gets the best of the best), but it means the `best_chain_id` stored via `atomicStore` may be stale if two chains race. Chain A sets `best_fitness_bits = 100`, then chain B sets it to `105`, but `best_chain_id` might still point to A if B's `atomicStore` happened before A's.

**Current mitigation:** The `atomicStore` for `best_chain_id` is guarded by `if best_bits > old_best_bits`, so only the true winner writes its ID. But since there's no lock between the `atomicMax` and the `atomicStore`, a very narrow race exists where:
1. Chain A does `atomicMax(100)`, gets `old=50`, writes `best_chain_id = A`
2. Chain B does `atomicMax(105)`, gets `old=100`, writes `best_chain_id = B`

This is correct -- B wins. The race only matters if both happen in the same warp/thread, which they can't since each chain is a separate workgroup.

**Verdict:** The current approach is correct. No change needed.

---

## Priority-Ordered Recommendations

| Priority | Issue | Estimated Impact | Effort |
|----------|-------|-----------------|--------|
| 1 | **Batch staging buffer readbacks** (Issue 1) | 0.1-0.5ms/batch | Small (1-2 hours) |
| 2 | **Double-buffer staging for GPU overlap** (Issue 2) | 10-30% throughput increase | Medium (4-8 hours) |
| 3 | **Batch island readbacks in `build_gpu_stats`** (Issue 4) | 1-5ms every 2s | Small (1-2 hours) |
| 4 | **Skip timestamp readback most batches** (Issue 3) | Minor per-batch | Small (30 min) |
| 5 | **Remove redundant error accumulator reset** (Issue 5) | Minor per-batch | Tiny (verify + delete) |
| 6 | **CPU-GPU co-evolution** (Issue 6) | Potentially large | Large (multi-day) |
| 7 | **Partial params write** (Issue 7) | Negligible | Small |
| 8 | **Profile batch sizes** (Issue 8) | Data-dependent | Small |
| 9 | **Reduce mutex contention** (Issue 9) | Minor | Small |

Issues 1-3 are the highest-value, lowest-effort improvements. Issue 2 (double-buffering) is the single biggest win because it eliminates the GPU idle period during CPU readback processing.
