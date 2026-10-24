# Observe compiled reference shaders before another microexperiment

Status is FROZEN_UNRUN. No optimization is proposed or adopted here. Recent
source-level cache and bbox experiments remain rejected. This investigation
asks what the installed compiler actually emits for the accepted engine.

A bounded source/tool audit finds Nsight Systems and Nsight Compute installed,
but does not establish hardware-counter support for Vulkan translated through
Dozen. The old 80–93 microsecond selection-plus-framebuffer samples also cannot
justify a selection rewrite. The current device does not request the feature
required for a same-pass midpoint timestamp. That separate observation is
conditional on runtime adapter support and would need a private device/build;
it is deferred here. No capability probe was executed.

## Source-supported compiler route

The installed Vulkan driver contains the `DXIL_DEBUG`, `verbose`, `dump_module`,
`dump_blob` and `shader_%s_%d.blob` strings. Its binary hash is frozen. The
[official Mesa 25.3.6 archive](https://archive.mesa3d.org/mesa-25.3.6.tar.xz)
implements those flags in `src/microsoft/compiler/nir_to_dxil.c`: verbose prints
final pre-emission NIR, dump_module prints Mesa's DXIL module tree, and dump_blob
writes shader containers. This upstream version matches the installed package;
Kisak patch parity remains unverified. Runtime output must confirm the route.
`DZN_DEBUG=dxil` is guarded by `_WIN32` in the matching source and is unsuitable
for Linux disassembly.

The [frozen protocol](profiles/compiler-observation-spec.json) requests one
reference-only Mona worker, seed 7, 64 measured batches, two warmups, diagnostic
false and ordinary Stages timestamps. It uses the unchanged accepted v12 release
executable, not a rejected prototype or instrumented build. All 83 archived
source entries match root and archive; all six full production shader strings
are verified inside the executable. No Rust build, shader, precision, dispatch,
pass, parameter, timestamp feature or ordinary executable change is needed.

The [one-shot runner](attempts/22-compiler-observation-runner.patch) sets only
`DXIL_DEBUG=verbose,dump_module,dump_blob` and a fresh private
`MESA_SHADER_CACHE_DIR`. It starts from an empty working directory, excluding
application pipeline-cache hits and the ordinary Mesa disk cache. Native/global
GPU caches and background isolation remain unverified.

There is one native workload, no retries, a 600-second package deadline and a
300-second child limit with 30 seconds reserved for cleanup. The pinned supervisor
applies a 64 MiB per-file process limit and kills/reaps its owned child on timeout
or cancellation. Post-run bounds are 512 files, 256 MiB in worker/cache outputs,
and at most 32 blobs. Missing diagnostics, process failure, replay failure or
exhaustion is retained and stops this route. No candidate or paired timing runs.

The ordinary frozen producer can still emit a rate in its raw TrialRecord.
That raw record remains unchanged. This bundle is `compiler_observation` with
`score_eligible:false`; every elapsed time and rate is excluded from performance
scoring and comparisons. Logging/compiler diagnostics are uncalibrated overhead.

## Required evidence and limits

Before interpreting the compiler output, verify complete counters, 64 timestamp
samples, exact parameters/device, initial/final seven-field canonical replay,
cached errors versus actual framebuffer RGB-byte-L1 and packed fitness under the
existing 2e-5 policy. Expected measured work is 1,048,576 evaluations. Two warmups
add 32,768 source-accounted evaluations; full-fitness/reset work stays separate.

Retain stdout/stderr, all blobs, pipeline/Mesa caches, sizes and hashes. Associate
compiler sections with stages through actual thread-group/resource properties;
container order alone is insufficient. NIR and DXIL are intermediate compiler
evidence, not final NVIDIA ISA, dynamic instruction counts, occupancy, register
spills, bandwidth saturation or utilization. One reference capture establishes
neither repeatability nor an engine speedup.

Prove It Works requires actual output and replay. Attack the Premise changes the
next step from another guessed edit to observation. Explain the Number keeps
compiler structure separate from a hardware limiter. The capability audits and
capture artifacts are under `.cache/evolution-observation-20261024/`.
