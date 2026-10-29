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

`gpu_evolver::programs` owns source specialization, compilation policy and
compiled handles, and installs the bundled sources. Each stage file in
`src/shaders/` holds only its own logic. `programs/bundled.rs` appends the shared
prelude: constants generated from Rust (polygon capacity, `TILE_MAX_POLYS`,
`CHANGED_INDICES`, and for the tile group `TILE_W`/`TILE_H` and the
`raster_work` header words), then `common.wgsl` (the `Polygon`, `DrawingState`
and `Params` structs and unpack helpers), `changes.wgsl` (the change-record
encoding, below), for the stages that read offspring records
`offspring_changes.wgsl`, and, for the tile group, `tiles.wgsl` (tile geometry,
the consumed-tile test and `pack_fb_pixel`). The stage body comes first, so compiler line numbers
match its file. A tile change regenerates the constants; bundled sources have no
replacement markers. CPU tests validate every assembled program at every
supported tile through naga and its SPIR-V writer, and compare the reflected
struct layouts with `gpu_evolver::buffers`. `examples/emit-bundled-shaders`
writes the complete sources for tools that read a shader file; it emits tile
stages at 16x16, the variants' marker form. Shader variants (historical
and candidate sources, diagnostic instrumentation, their caches and the worklist
pass older binners need) sit behind one seam, `programs/variants.rs`, reached
through `GpuPipeline::shader_variants()`. Production never installs a variant.
Variants are complete sources without the prelude, specialized by their
16x16 tile-size markers: `WG_X`/`WG_Y` or `TILE_W`/`TILE_H` for raster,
`TILE_W`/`TILE_H` for binning. An emitted bundled raster or binner specializes to
exactly the installed source. Mutation's logical group width and initialization
policy stay attached to its selected program. The raster, binning, initial framebuffer and
framebuffer-update programs form one installed tile group. A failed change must
preserve the entire preceding group.

`GpuPipeline` owns storage, binding and the concrete device. Its program accessors
borrow the installed handles; ordinary batches do not look up caches or allocate
program state. Stage-specific policies remain explicit. In particular, historical
mutation programs retain their initialization default, and raster initialization
continues to depend on the backend.

The binner writes counts and indices only for consumed tiles, those that meet
each offspring's dirty rectangle, the tiles rasterize_error and
update_framebuffers read. Other tiles keep stale words from earlier iterations,
so a new consumer of the tile lists must use `tile_consumed` (or its per-axis
forms) from `tiles.wgsl`, which the binner, worklist pass and framebuffer update
share. Validate binner changes with `check-gpu-binning --scope consumed-only`; `full-buffers` compares stale words
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
the host because it runs no update. A binning variant that declares no
`raster_work` binding gets a separate `raster_worklist.wgsl` pass with the same
test. A raster variant without that binding keeps the full grid. The variant
seam decides this `RasterSchedule` from the declared bindings when a variant is
installed; command recording only matches on it. That extra pass made
512-batch runs host-bound and 5% slower, so production keeps the list in the
binner.

Bundled programs, historical programs and diagnostic binners are existing reasons
for the installation seam. A generic cache abstraction or another GPU adapter
would add no needed variation.

## Change records

A change record says which polygons an evaluated drawing changed relative to
the drawing whose pixels are cached. `gpu_evolver::changes` names it in Rust;
`src/shaders/changes.wgsl` owns its shader encoding: `CHANGED_INDICES`,
`COVER_MARGIN`, the changed-index words and their decoding, the coverage tests
and `changes_shortcut`, the tile-list overflow guard. Every bundled stage uses
those; no stage repeats them. CPU tests check a Rust reference of the encoding
and compare the shader constants and the `ChainChange` layout.

- The offspring record lives in a working state: the dirty rectangle in
  `fitness_bits`/`stagnation_counter` and the changed-index words in
  `rng_state[1..3]`. mutate writes it; select, the binner, the worklist pass and
  raster read it through `offspring_changes.wgsl`. Historical shader variants
  read and write these words in place, so they stay where they are.
- The chain record is a `GpuChainChange` per chain in `chain_changes`: the
  accepted winner's changed-index words, a mask and the parent versions of those
  polygons. select writes it before copying the winner; update_framebuffers
  reads it in the same iteration. Only those two programs bind it, and neither
  can be replaced by a shader variant. Parent states carry nothing in
  `rng_state[1..4]` or past `polygon_count`, so the update shortcut works at the
  full polygon limit (it used to stash parents in slots 996–999 and stop above
  996 polygons).

An all-zero record is unknown and turns every shortcut off. The host writes
records only through two `GpuPipeline` methods. `write_chain_states`, the only
way chain states are replaced after construction (reinitialization, seeded
benchmark reset), forgets every chain record with them, through
`forget_chain_changes`, which `evaluate_chain_fitness` also calls.
`mark_full_evaluation` gives the parent copies of a full evaluation a
whole-image dirty rectangle and unknown changed indices. The two stay separate:
the offspring record is rewritten by mutate every iteration and needs the host
only when mutate does not run; the chain record is read only in the iteration
select writes it, so forgetting it is a guard, not a batch-loop step.

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
