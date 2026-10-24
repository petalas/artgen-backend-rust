# Reject matched-policy compact raster dispatch

REJECT. The cache-only matched-options probe gives a short aggregate throughput
ratio of 0.665429245342, with conditional interval
[0.660861784705, 0.670028273392].
Matching compilation policy did not make this implementation competitive.
The prefix/indirect family is closed; its runtime remains removed from the engine.

The historical prototypes accidentally enabled automatic compact-raster shared
initialization while ordinary Vulkan raster disabled it. This new probe copied
the exact v7 Cargo source snapshot into a private cache project and changed only
compact raster's compilation descriptor to the ordinary backend policy.
The prefix stayed default TRUE. Ordinary pipeline/orchestration/lab source,
all WGSL, lockfile, workspace and release profiles remained byte-identical.
The copied checker also compiled its actual instrumented raster with the matched
descriptor and recorded the effective value. Both arms used FALSE on Vulkan.

The reference remains decoded C with historical G64 mutation and binner 598.
This is deliberately not a comparison against the newer AE1/G1 default. Equal
settings, logical evaluations, image resolution and numerical operations were
preserved. The result neither reuses old timing samples nor attributes the old
loss specifically to initialization: no contemporaneous compact-TRUE arm ran.

Before timing, the new FALSE configuration passed 75 actual coverage/error
fixtures and both real missing/duplicate tile faults. Each fault kept both error
arrays equal while coverage detected the wrong work. Every recorded compilation
option event says raster FALSE and prefix TRUE. This is fresh FALSE evidence,
not an inherited TRUE proof. The four later CPU-only cases remain unrun; this
does not upgrade the record to 79 GPU fixtures.

The frozen filter used two alternating pairs per ff/mona/darwin case, eight
batches, seed 7 and two matching warmups. All twelve normal trials and six
overflow replays passed exact seven-field state, raw framebuffer L1, fitness,
work, timestamps and device checks. Timing work was 1,572,864 evaluations;
diagnostics contributed 786,432 with null rates, and three qualifications added
393,216. Every pair remains included. No full-budget or reserved run followed.

Two initial CPU attempts exposed stale shared-target cache linkage to the other
workspace. Their logs remain retained. Only the copied packages' build cache
was evicted; source, profiles and lockfile were unchanged. Both final artifacts
are non-fresh Cargo builds with the copied manifest paths and were archived
immediately. No assumed target path was used. The lab SHA is
9af3824cb358dba4d76179956c7c0ad8a49a21509f7ae3abf47cd7f28d2b67f6;
the checker SHA is
4d7b9408aceea9bed3abc3547874979dd322e64f03b629d4565fa181f1f931f6.
Five compact, thirteen lab and seven checker CPU tests, scoped check and
all/pedantic analysis completed before the release builds.

Evidence under .cache/evolution-overnight-20261024/compact-matched-options/:
HANDOFF.json, source-preservation.json, build/immutable-build.json, gpu-spec.json,
gpu-proof/, gpu-outcome.json, screen/output/analysis.json and filter-outcome.json.
Patch 39 applies only to the archived v7 project, not the current engine, where
those files are removed. The source snapshot and real commands remain retained.

The large loss remains a result of the tested implementation. Prefix work,
mapping, indirect validation, pass boundaries and scheduling are possible
contributors, not measured causes. These exploratory intervals establish no
hardware bottleneck. Background isolation remains unverified.
