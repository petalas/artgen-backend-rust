# CPU-GPU Hybrid Architecture

**Expert focus**: Utilizing idle CPU cores alongside GPU, async pipeline, data sharing, PNG offload

---

## Executive Summary

CPU cores are completely idle during GPU mode. The existing `Evaluator` and CPU rasterization code is production-ready and can run in parallel with GPU evolution. CPU and GPU can evolve independently, sharing only the global best via `Arc<RwLock<Drawing>>`.

---

## 1. CPU Workers Alongside GPU

### Architecture

```
                     +----------------+
                     |  Global Best   |  Arc<RwLock<Drawing>>
                     |  + fitness     |
                     +-------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |   GPU Evolver    |          |   CPU Workers   |
     |   512 chains     |          |   N threads     |
     |   batch loop     |          |   Evaluator     |
     +------------------+          +-----------------+
```

### Synchronization

Both sides do (1+1) evolution independently. Global best only moves upward (monotonic), so no ABA problem.

- **GPU side**: After `run_batch()` returns `Some(new_best)`, write-lock to update global best. Before each batch, read-lock to inject CPU improvements into GPU chain 0 via `queue.write_buffer` (48032 bytes — negligible PCIe overhead).
- **CPU side**: Each worker reads global best via read-lock at start of `produce_new_best()`. When it finds improvement, write-locks and updates.

Write contention is extremely rare since improvements happen infrequently. `Drawing` is `Clone`, so workers clone and work on their own copy.

### Injecting CPU Improvements into GPU

```rust
pub fn inject_best(&mut self, drawing: &Drawing) {
    let seed = 0xCAFE_BABE_u64.wrapping_add(self.iteration as u64);
    let gpu_state = drawing_to_gpu(drawing, seed);
    self.pipeline.queue.write_buffer(
        &self.pipeline.chain_states_buf,
        0, // chain 0
        bytemuck::bytes_of(&gpu_state),
    );
    self.best_fitness_bits = gpu_state.fitness_bits;
}
```

### Expected Performance

- **CPU workers** at 384x384: ~500-2000 evals/sec per core
- With 14 CPU cores: ~7K-28K additional evals/sec
- Raw eval contribution: ~2-5%
- **Practical improvement rate**: 10-30% (especially at high fitness)
  - Every CPU evaluation sent back IS an improvement (adaptive search)
  - CPU excels at micro-adjustments that dominate at high fitness
  - GPU chains often spin finding nothing at fitness >85%

---

## 2. Work Partitioning: CPU Fine-Tuning, GPU Exploration

Run CPU workers with **fine-tuning parameters**:
- High `micro_adjust_prob` (0.5 instead of 0.01)
- Low `add_polygon_prob` and `remove_polygon_prob` (0)
- Small `micro_adjust_delta` (0.002 instead of 0.01)

Run GPU chains with **default exploration parameters**:
- Current mutation probabilities
- Structural changes (add/remove/reorder)

Analogous to broad MCTS search + deep evaluation. CPU does focused hill climbing while GPU explores the landscape.

---

## 3. Double-Buffered Readback

### Current bottleneck

```
[GPU batch N] → [CPU poll+read] → [GPU batch N+1] → [CPU poll+read] → ...
                 ^^^^ GPU idle ^^^^                   ^^^^ GPU idle ^^^^
```

### Fix: Two staging buffer sets, alternating each batch

```
Batch N:   GPU executes passes    |  CPU reads results from batch N-1
Batch N+1: GPU executes passes    |  CPU reads results from batch N
```

Use `Maintain::WaitForSubmissionIndex` to poll only the previous batch. The current batch's staging buffer is submitted and the GPU starts work immediately.

Expected: eliminates ~100-200 microseconds of CPU stall per batch. Combined with larger batch sizes, the overlap becomes significant.

---

## 4. Data Sharing: Minimizing PCIe Transfers

Current transfers are already minimal:
- Reference image: uploaded once
- Control flags: 16 bytes per batch
- Best drawing readback: 48032 bytes only when new best found
- Params: 96 bytes per batch

With hybrid mode, the new transfer is injecting CPU improvements: one `queue.write_buffer` of 48032 bytes per injection. Even at one per second, 48 KB/s is trivially small on PCIe 5.0 (64 GB/s).

**Optimization**: Batch injection — only inject at the start of each GPU batch, checking if global best changed since last injection. At most one 48 KB write per batch.

---

## 5. PNG Encoding Offload

### Current problem

In the inner loop, PNG encoding blocks the GPU evolution thread:

```rust
global_best.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);  // CPU rasterize
let png = encode_rgba_as_png(&render_buf, w, h);                  // PNG encode
```

This can block for ~200ms at 384x384.

### Fix: Dedicated PNG encoding thread

```rust
let (png_tx, png_rx) = mpsc::sync_channel::<(Drawing, usize, usize)>(1);

thread::spawn(move || {
    let mut render_buf = vec![0u8; 384 * 384 * 4];
    while let Ok((drawing, w, h)) = png_rx.recv() {
        drawing.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);
        let png = encode_rgba_as_png(&render_buf, w, h);
        // update WsState...
    }
});

// In main loop (non-blocking):
png_tx.try_send((global_best.clone(), w, h)).ok(); // drops frame if encoder busy
```

### Alternative: Send raw RGBA over WebSocket

For local connections, skip PNG entirely:
- Raw RGBA: 384x384x4 = 590 KB (base64 ~787 KB)
- PNG: ~100-200 KB but costs CPU time

For remote connections, consider `zstd`/`lz4` which are 10-50x faster than PNG with 3-4x compression.

---

## 6. Practical Implementation Summary

| File | Change |
|------|--------|
| `src/main.rs` | Add CPU worker spawning in `gpu_main_loop_headless`, drain CPU results in inner loop, update shared state |
| `src/gpu_evolver/mod.rs` | Add `inject_best()` method to `GpuEvolver` |
| `src/evaluator.rs` | Minor: allow constructing Evaluator without broadcast channel |

The PNG offload is independent and can be done separately — spawning one more thread and replacing inline encode with `try_send`.
