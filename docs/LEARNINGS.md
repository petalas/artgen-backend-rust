# Learnings

Project-specific gotchas and hard-won knowledge. Search here when you hit unexpected behavior.

<!-- Use this format for new entries:

### [Short descriptive title]

**Context**: What you were doing when you hit this.
**Gotcha**: What went wrong or what's surprising.
**Fix**: The correct approach (1-3 lines).

-->

### wgpu 22.1.0 naga does not support WGSL subgroup operations

**Context**: Tried to use `enable subgroups;` + `subgroupAdd()` in rasterize_error.wgsl for faster workgroup reduction.
**Gotcha**: Naga's WGSL frontend in wgpu 22.x does not recognize the `enable` directive at all -- it fails with "expected global item". Even in naga 28.0.0, `subgroups` is listed as `UnimplementedEnableExtension`. There is no cargo feature flag or workaround.
**Fix**: Use shared-memory binary tree reduction (8-step `stride >>= 1` loop with `workgroupBarrier()`) instead. Do not request `Features::SUBGROUP` on the device.
