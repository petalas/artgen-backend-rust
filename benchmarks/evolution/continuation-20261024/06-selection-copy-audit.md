# Selection copying already uses all lanes

NO-GO for another cooperative winner-copy prototype. The accepted selection
shader already publishes acceptance, polygon count and offspring ID on its leader,
crosses an unconditional workgroup barrier, and copies the counted prefix with
lane-striped indices. At 1000 polygons, each of its 64 lanes copies at most 16.
Rejection preserves the geometry; unused tail words remain untouched. There is
no serial winner-copy loop to replace and no new installer or benchmark is built.

Direct comparison against a6eedb1 confirms unchanged selection SHA256
bfd33b7bb029ded82387dfa9ad0ed8159c0a8b0977cb13509c18821dc6dbb279.
The existing reduction's strict comparison preserves tree-dependent equal-fitness
ties. Its comment about lowest-index preference is not a basis for changing that
behavior. Error exchange/reset, penalties, adaptive scaling, RNG parity use,
control exports and both selection/framebuffer-update dispatch paths remain
unchanged. The historical selection/migration description does not establish a
current GPU migration dispatch.

The sampled select-plus-framebuffer-update interval does not isolate copying or
prove a hardware limiter. Source ownership and barrier evidence rule out this
specific proposal without another build, test suite or GPU run.

The bounded source report and future correctness requirements are retained in
`.cache/evolution-continuation-20261024/select-copy-audit-72/report.md`.
Prove It Works required inspecting the actual copy and comparing baseline bytes;
Explain the Number prevented treating the combined sampled interval as copy time.
