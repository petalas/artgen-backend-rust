# Engine module ownership

The compute-only engine is `GpuEvolver`. The older `Engine` retains its distinct
CPU and render-pipeline contracts.

## Capacity and dispatch planning

`gpu_evolver::execution` owns allocation sizes, logical raster geometry, mutation
dispatch and required storage. Construction, ordinary batches, full evaluation
and benchmark capacity checks use those rules. `GpuPipeline` supplies actual
buffer sizes to validation. Planning adds no allocation to the batch loop and
does not change shader bytes or compute-pass separation.

Allocation preserves the existing independent storage clamps. Ordinary evolution
still clamps requested chains to drawing capacity; benchmark preparation rejects
configurations that do not fit every required buffer. Reporting a smaller
framebuffer capacity without enforcing it in ordinary configuration is a separate
behavior issue, not silently changed by this refactor.

`ResourceSizes` includes the reserved diagnostic word in required tile-count
storage. Allocation preserves the existing worst-case tile-count allocation;
that word must fit inside it for a requested workload to qualify.

## Shader installation

`gpu_evolver::programs` owns shader variant identity, source specialization,
compilation policy and compiled handles. Mutation's logical group width and
initialization policy stay attached to its selected program. The raster, binning,
initial framebuffer and framebuffer-update programs form one installed tile
group. A failed change must preserve the entire preceding group.

`GpuPipeline` owns storage, binding and the concrete device. Its program accessors
borrow the installed handles; ordinary batches do not look up caches or allocate
program state. Stage-specific policies remain explicit. In particular, historical
mutation programs retain their initialization default, and raster initialization
continues to depend on the backend.

The binner writes counts and indices only for tiles that meet each offspring's
dirty rectangle, the tiles rasterize_error and update_framebuffers read. Other
tiles keep stale words from earlier iterations, so a new consumer of the tile
lists must apply the same dirty-tile test. Validate binner changes with
`check-gpu-binning --scope consumed-only`; `full-buffers` compares stale words
with the frozen full binner and no longer applies to the production shader.
Each offspring is dispatched as `BIN_ROW_BANDS` workgroups over bands of tile
rows, so full-image (crossover) offspring do not bound the dispatch alone.

After binning its band, each bin workgroup appends the consumed tiles that a
changed polygon may meet (or that have the shortcut off) to `raster_work`.
`GpuPipeline::record_raster` then dispatches 8192 raster workgroups that loop
over the list. On the v1 fixtures, about 7–10% of the former 73.7k-workgroup grid
had work. The test must match the raster shortcut exactly, because skipped tiles
used to return before adding to the error sums. update_framebuffers empties the
list at the end of every iteration, and `evaluate_chain_fitness` resets it from
the host because it runs no update. A benchmark binning source without
`raster_work` gets a separate `raster_worklist.wgsl` pass with the same test.
A benchmark raster source without it keeps the full grid. That extra pass made
512-batch runs host-bound and 5% slower, so production keeps the list in the
binner.

Bundled programs, historical programs and diagnostic binners are existing reasons
for the installation seam. A generic cache abstraction or another GPU adapter
would add no needed variation.

## Pending readback

`gpu_evolver::readback` owns one pending batch with its map completions, staging
slot, queued winning chain and collected observations. The optional drawing map
is the authority for whether that batch contains a drawing copy. Batch metadata
and receivers can no longer be detached or paired with a separate deferred flag.

`GpuEvolver` retains command recording. Each staging slot has its own drawing
staging buffer, so the new submission always precedes prior collection, and map
registration follows collection. A winning chain reported by batch N-1 is copied
at the end of batch N+1, because batch N is encoded before N-1 is collected. The
single-buffer design drained N-1 before encoding N whenever both batches carried
a drawing (every third batch); removing that drain raised throughput about 2%
(hillclimb, three cases). The copied drawing is still a real chain state. Against
a single-buffer reference, `check-gpu-lifecycle` differs only in event `drawing`
fields: the single-buffer build returned no drawing every third call, because
its early collection consumed the copy and left the next batch none to copy.
Copy source selection remains at the same position in the encoder.

The host picks the winning chain from the batch's `fitness_packed` readback:
highest fitness, ties to the lowest chain id. The control word `best_chain_id`
is not used. Select writes it after an `atomicMax`, so concurrent workgroups
race, and the last store can name a chain other than the best, or a different
tied chain on each run. That race made `check-gpu-lifecycle` deliveries differ
between runs of one build. `new_best_found` and `best_fitness_bits` are
deterministic and still gate delivery. Chain fitness never decreases, so the
copy taken a batch later is at least as good as the reported best. Terminal
flush still prefers a newer synchronous drawing over an older deferred drawing.

Ordinary reinitialization preserves staging alternation; seeded preparation resets
it to zero after the GPU reset. The ownership change preserves existing channel
and drawing allocations and the existing blocking/error policy.

`check-gpu-lifecycle` records ordinary returns without per-batch drains. Both
timestamp modes cover ordinary evolution, reset with pending work and seeded
repreparation with pending work. Compare its `comparison.json` exactly between
reference and candidate; the companion summary and snapshots retain numerical
quality, terminal maximum-parent checks and errors. It is an unscored correctness
check, not a throughput measurement.

## Trial qualification

`benchmark::qualification` owns the measured batch clock, completed-work rule,
canonical state identity and scoring purpose. Isolated trials and the resident
lab share those rules while retaining their preparation and numerical checks.
An unscored qualification remains distinct from a scored timing even when both
have `diagnostic:false`. See [the result contract](trial-qualification.md).

## Verification

Run `cargo test --lib --locked` and the relevant example tests. Real GPU checks
must run serially under `.cache/evolution-bench/benchmark.lock`, with a parent
process timeout. Use the standalone `check-gpu-programs` and
`check-gpu-rasterization` examples; Dozen teardown crashes in threaded Rust GPU
tests, as recorded in `docs/LEARNINGS.md`. Compare complete canonical state,
cached framebuffer byte-L1, fitness and completed work before interpreting
throughput. Compilation and snapshots stay outside engine timing; ordinary readback and final drain stay
inside it.

The [verification record](engine-architecture-validation.md) reports the fixed
comparisons, their limits and the separate longer ff check.

## Power of Ten deviations retained by the refactor

- Rules 2 and 7. Existing GPU construction, backend compilation and ordinary
  readback contain driver waits without an in-process timeout. Real verification
  uses an external process timeout and the benchmark lock. A driver-lifetime
  redesign is separate from the ownership changes here.
- Rule 3. Wgpu command encoding, map callbacks and returned drawings allocate
  during ordinary batches. The refactor introduces no new heap allocation in
  that path. Existing chain, offspring, image and batch limits bound live work.
- Rule 4. The existing pipeline constructor and command-recording function exceed
  one page. Their cohesive allocation order and synchronization are preserved;
  capacity arithmetic moves out without splitting the GPU command schedule.
  Subsequent ownership changes must keep those synchronization decisions local.
- Rules 8 and 9. Existing serde/bytemuck derives and wgpu map callbacks remain.
  Callbacks target one pending operation and existing fixed staging storage.
- Rule 10. Run all/pedantic Clippy and retain its output. Existing warnings outside
  the changed path are not suppressed or rewritten as part of these refactors.
