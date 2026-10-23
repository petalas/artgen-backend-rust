# Learnings

Project-specific gotchas and hard-won knowledge. Search here when you hit unexpected behavior.

<!-- Use this format for new entries:

### [Short descriptive title]

**Context**: What you were doing when you hit this.
**Gotcha**: What went wrong or what's surprising.
**Fix**: The correct approach (1-3 lines).

-->

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
