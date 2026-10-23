# Learnings

Project-specific gotchas and hard-won knowledge. Search here when you hit unexpected behavior.

<!-- Use this format for new entries:

### [Short descriptive title]

**Context**: What you were doing when you hit this.
**Gotcha**: What went wrong or what's surprising.
**Fix**: The correct approach (1-3 lines).

-->

### Comparison wrappers must handle termination around the shared benchmark lock

**Context**: Reusing `scripts/evolution-bench.py` process helpers for paired engine experiments.
**Gotcha**: `run_process` cleans up its direct child after timeouts or Python exceptions. A supervisor receiving default SIGTERM exits without that cleanup, releases the shared benchmark lock and can leave its worker running. Also, `verify_saved_state` verifies the final snapshot only; it does not verify `initial-state.json`.
**Fix**: The paired comparator installs a scoped SIGTERM-to-cancellation handler and restores the previous handler after cleanup. It separately hashes the initial canonical state and associates its parents with checked quality. Regression tests cover owned-worker termination, unrelated-process survival and missing/corrupt initial artifacts.

### Reconstruct reference bytes before integer GPU error accumulation

**Context**: Qualifying mature project benchmarks against framebuffer byte L1.
**Gotcha**: A reference byte sampled from `Rgba8Unorm` and multiplied by 255 can be slightly noninteger. Casting the sum of absolute errors to `u32` can then undercount. On the mature ff fixture, initialization cached 2,923,600 while its framebuffer's exact byte L1 was 2,966,665. Black-only references did not expose this.
**Fix**: Round reconstructed reference channels before subtraction in both error shaders. The standalone GPU check now visits every reference byte with empty, opaque and translucent drawings and compares cached, full and framebuffer errors.

### Frozen floating-point metadata needs JSON roundtrip parsing

**Context**: Comparing frozen geometry percentiles with benchmark trial metadata.
**Gotcha**: Serde JSON's default fast parser changed an observed `f64` bbox percentile from `0.19420557263631666` to `0.19420557263631663`. Strict supervisor comparisons rejected all otherwise valid trials.
**Fix**: Enable `serde_json`'s `float_roundtrip` feature and preserve exact metadata comparisons. The fixture tests include this value; the real supervisor validates imported metadata across Rust loading and Python comparison.

### Do not change selected geometry after its error was evaluated

**Context**: The 64-batch mature darwin benchmark failed exact cached/framebuffer L1 even though shorter trials passed.
**Gotcha**: Selection used to remove triangles whose normalized cross magnitude was at most `1e-5` after caching error and exporting fitness. Such triangles can still cover pixels. At iteration 2997, candidate 97 improved one pixel by 118 error units, then cleanup removed that triangle while retaining the improved total. A tiny two-layer regression reproduced cached error 0 versus actual framebuffer error 765 after neutral acceptance.
**Fix**: Preserve the evaluated candidate's geometry and polygon count through selection. The unscored post-selection cleanup is removed. Any future cleanup must form part of a proposal before evaluation, with its cost and quality measured separately.

### Automatic workgroup zeroing makes Dozen shader compilation slow

**Context**: Project loading waits for `GpuEvolver::new`, initial fitness evaluation, and the first preview.
**Gotcha**: Loading a pipeline cache does not skip all compilation. On the WSL Dozen driver, automatic initialization of the large shared polygon arrays dominates compilation of the rasterize/error and framebuffer pipelines.
**Fix**: These three shaders explicitly initialize every shared value they consume before a barrier. Their Vulkan pipelines disable redundant automatic zeroing. The reductions use `subgroup_id` and `num_subgroups`, since WGSL does not guarantee a mapping from local invocation indices to subgroup IDs. Keep automatic zeroing for mutation, whose empty-parent crossover reads polygon slot zero, and for DX12, where Naga emulates subgroup IDs using local indices.

### Run Dozen GPU checks on the application's main-thread path

**Context**: Checking rasterization across workgroup sizes and offspring counts.
**Gotcha**: On this WSL/NVIDIA setup, the threaded Rust test runner crashes during GPU teardown after the numerical checks finish. The same crash reproduces against the unchanged engine; the standalone main-thread runner exits successfully.
**Fix**: Run `cargo run --release --example check-gpu-rasterization`. It checks empty drawings, partial image tiles, multiple shared polygon chunks, and incremental fitness against a full recomputation.

### Rasterization fixtures must respect the tile-bin capacity

**Context**: Testing polygons that span more than one shared-memory chunk.
**Gotcha**: Each screen tile stores at most `TILE_MAX_POLYS` polygon indices, currently 256. A fixture with more overlapping polygons silently omits later layers, so its analytical expected image is wrong.
**Fix**: The GPU check uses 194 overlapping polygons. This exceeds the 192-polygon shared chunk at an 8x8 workgroup, stays below the bin limit, and puts visible red layers in the final chunk.

### Trunk 0.21 requires a boolean NO_COLOR value

**Context**: Building the WASM viewer from a terminal that exports `NO_COLOR=1`.
**Gotcha**: Trunk exits before building because its argument parser requires `true` or `false` for `NO_COLOR`. Passing `--no-color` does not bypass the environment parsing error.
**Fix**: Run `NO_COLOR=true trunk build --locked`. The dependency verification script sets this value for Trunk.

### naga does not support `enable subgroups;` WGSL directive

**Context**: Tried to use `enable subgroups;` + `subgroupAdd()` in rasterize_error.wgsl.
**Gotcha**: Naga's WGSL frontend does not recognize the `enable` directive — it fails with "expected global item" (22.x) or lists `subgroups` as `UnimplementedEnableExtension` (28.x). However, subgroup builtins (`subgroupAdd`, `subgroup_invocation_id`, etc.) work fine WITHOUT the `enable` directive when `Features::SUBGROUP` is requested on the device.
**Fix**: Just use subgroup builtins directly (no `enable` directive). Requires `Features::SUBGROUP` on the device and 1D workgroup layout (`@workgroup_size(N, 1, 1)`) — naga rejects subgroup builtins on multi-dimensional workgroups.

### GPU timestamp stages use a different pass schedule from bulk iterations

**Context**: Explaining why a large sampled bin interval did not predict large optimization gains.
**Gotcha**: Some `run_batch` comments describe older pass layouts. The executable code uses two passes per bulk iteration and four for the final instrumented iteration. The sampled bin interval can therefore include different dependency and transition effects; it is not a whole-run share.
**Check**: Follow the actual `begin_compute_pass` and dispatch calls. Compare full-batch envelope and completed wall time, and use a repeated-body diagnostic before attributing a precise share to bin construction.


## Bin-cache initialization is not a free throughput assumption

Candidate 5's tile-parallel bin shader writes every consumed shared bounds slot
before its barrier. Candidate 6 disabled only automatic bin workgroup zeroing in
both normal and diagnostic factories. Full-buffer GPU equality, rasterization,
50 Rust tests and all mature trajectory checks passed. Five paired full-budget
runs per project measured 39.83% less benchmark preparation, but mona failed the
frozen throughput non-regression gate. Keep default zeroing in the best archived
candidate rather than assuming fewer initialization instructions guarantee an
engine gain. The slow pair remains included and background isolation is
unverified. This does not establish a general hardware cause.

`setup_seconds` includes initial snapshot serialization and numerical
qualification as well as construction and warmup. It is not a direct browser
loading timer or an isolated pipeline-compilation timer. See
[experiment 6](../benchmarks/evolution/v1/experiments/06-bin-cache-initialization.md).

## Live acceptance needs copy provenance and explicit shutdown

An ordinary returned drawing keeps the fitness from its own mapped GPU header.
The immediately observed chain-fitness vector describes the same copy batch,
but the source chain was selected from an earlier batch's best signal. Another
chain can overtake it before the copy. Preserve the copy origin for cached
publications; maximum equality at acquisition is telemetry, not a correctness
requirement. After the final flush, require the published drawing to match a
final packed maximum. Capture `flush_pending()` before `benchmark_drain()`,
which discards that return value.

The WebSocket handler can already be sending its next stats packet when the
last image receipt arrives. The first actual acceptance smoke delivered both
images and matched the terminal maximum but failed with BrokenPipe when its
owned sockets were deliberately shut down. Only known transport-close errors
after explicit stop are expected. Pre-stop disconnects, protocol failures,
validation errors and joined-thread panics remain errors. Acceptance receive,
Close and command-reply exits need the same classification as image sends.

Normal image-generation coalescing cannot retain every scheduled publication.
The opt-in workload uses a bounded queue and the shared serializers/throttles.
Receipt timestamps precede extra CPU rendering audits. Retain actual client
receipt bytes after timing, then bind each frame to its recorded identity,
work counters and payload hashes. A producer's `payload_valid` flag alone is
not independent artifact evidence. See [acceptance policy 1](../benchmarks/evolution/v1/acceptance-policy.md).

## Live result merging must preserve typed float serialization

The ordinary trial writes typed `f32` parameters directly through serde_json.
The live formatter initially merged `serde_json::to_value(TrialRecord)` into its
extra fields. That conversion widened the parameters to exact `f64` values,
changing 25 JSON values, including `0.2` to `0.20000000298023224`. The first
full live worker completed its work and payload checks but correctly failed the
collector's exact parameter gate.

Serialize the typed trial to bytes before parsing the merge object. Keep
`float_roundtrip` for existing `f64` complexity metadata. Test the actual
`Run::finish` result against the ordinary typed JSON and the literal default
parameter map. Live protocol 2 requires an explicit version field and fresh
A/A measurements. Preserve the old invalid records; do not add a parameter
tolerance or reuse their timings. Short two-publication smokes cannot pass the
full 128-publication collector contract, so verify that boundary with a full
unscored worker before a new measurement package.
