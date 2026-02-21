# Learnings

Project-specific gotchas and hard-won knowledge. Search here when you hit unexpected behavior.

<!-- Use this format for new entries:

### [Short descriptive title]

**Context**: What you were doing when you hit this.
**Gotcha**: What went wrong or what's surprising.
**Fix**: The correct approach (1-3 lines).

-->

### naga does not support `enable subgroups;` WGSL directive

**Context**: Tried to use `enable subgroups;` + `subgroupAdd()` in rasterize_error.wgsl.
**Gotcha**: Naga's WGSL frontend does not recognize the `enable` directive — it fails with "expected global item" (22.x) or lists `subgroups` as `UnimplementedEnableExtension` (28.x). However, subgroup builtins (`subgroupAdd`, `subgroup_invocation_id`, etc.) work fine WITHOUT the `enable` directive when `Features::SUBGROUP` is requested on the device.
**Fix**: Just use subgroup builtins directly (no `enable` directive). Requires `Features::SUBGROUP` on the device and 1D workgroup layout (`@workgroup_size(N, 1, 1)`) — naga rejects subgroup builtins on multi-dimensional workgroups.
