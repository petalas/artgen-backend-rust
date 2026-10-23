# Learnings

Project-specific gotchas and hard-won knowledge. Search here when you hit unexpected behavior.

<!-- Use this format for new entries:

### [Short descriptive title]

**Context**: What you were doing when you hit this.
**Gotcha**: What went wrong or what's surprising.
**Fix**: The correct approach (1-3 lines).

-->

### Trunk 0.21 requires a boolean NO_COLOR value

**Context**: Building the WASM viewer from a terminal that exports `NO_COLOR=1`.
**Gotcha**: Trunk exits before building because its argument parser requires `true` or `false` for `NO_COLOR`. Passing `--no-color` does not bypass the environment parsing error.
**Fix**: Run `NO_COLOR=true trunk build --locked`. The dependency verification script sets this value for Trunk.

### naga does not support `enable subgroups;` WGSL directive

**Context**: Tried to use `enable subgroups;` + `subgroupAdd()` in rasterize_error.wgsl.
**Gotcha**: Naga's WGSL frontend does not recognize the `enable` directive — it fails with "expected global item" (22.x) or lists `subgroups` as `UnimplementedEnableExtension` (28.x). However, subgroup builtins (`subgroupAdd`, `subgroup_invocation_id`, etc.) work fine WITHOUT the `enable` directive when `Features::SUBGROUP` is requested on the device.
**Fix**: Just use subgroup builtins directly (no `enable` directive). Requires `Features::SUBGROUP` on the device and 1D workgroup layout (`@workgroup_size(N, 1, 1)`) — naga rejects subgroup builtins on multi-dimensional workgroups.
