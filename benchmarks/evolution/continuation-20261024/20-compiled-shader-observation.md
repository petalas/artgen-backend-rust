# The compiled reference cache differs from the source struct

The accepted AE1 raster cache is split into field arrays in the captured Mesa
NIR and DXIL module. Its source-level 48-byte record stride does not survive at
this IR level. This directly challenges the padding-saving premise behind the
closed scalar44 experiment. It does not measure hardware allocation or explain
that experiment's slowdown. No new optimization is proposed or adopted.

The same capture confirms eager shared loads of six vertex floats, four bounds
and packed color before the polygon AABB branch. Edge arithmetic follows that
branch; color normalization remains after triangle coverage. These are static
compiler facts, not dynamic cost or a reason to reopen the rejected deferred-load,
RGBA or bbox experiments. No supported new performance hypothesis emerged.

## Actual compiler evidence

The [bounded raw excerpts](profiles/compiler-raster-excerpts.txt) retain exact log
lines and the source-log SHA256. NIR lines 55641–55648 and DXIL globals
56593–56600 declare three float[512] vertex arrays, four float[256] bounds arrays
and one uint[256] color array. Their declared element storage totals 11,264 bytes.
Vertices index 2*i and 2*i+1; bounds and color index i. There is no retained array
of 48-byte records. Final hardware alignment, allocation and bank behavior remain
unknown. The scalar44 result remains REJECT with its original data unchanged.

NIR lines 56140–56157 and DXIL lines 57031–57062 place all eleven shared-load sites
before the AABB branch. No cross-pixel hoisting or loop unrolling appears in this
output. The loader steps by 256, the pixel ordinal bound is 512 with step 256,
and polygon traversal steps by one with the existing 256 clamp. Per-hit color
byte extraction and floating normalization/rescaling remain represented. The
DXIL tree's printed division mnemonic alone is not a final hardware instruction.

The analyst associates six application sections with blobs through independent
workgroup/resource/texture/UAV properties, rather than their file order:

| Blob | Application stage | Physical workgroup |
| --- | --- | ---: |
| shader_cs_18.blob | mutation | 64 |
| shader_cs_19.blob | raster/error | 256 |
| shader_cs_20.blob | binning | 512 |
| shader_cs_21.blob | selection | 64 |
| shader_cs_22.blob | framebuffer initialization | 512 |
| shader_cs_23.blob | framebuffer update | 512 |

The raster fingerprint also establishes logical 32 by 16 through shifts/masks and
its pixel-loop bound. Six application associations have high static confidence;
compilation presence alone is not a runtime dispatch trace. Sections 0–15 name
Dozen indirect-draw helpers, with some property-equivalent blob associations
ambiguous. Sections 16/17 are auxiliary programs of unproven exact purpose.
All 24 blobs are structurally valid DXBC containers with DXIL chunks. The parent
checks the raster's 256,1,1 thread dimensions directly in its blob. All 42 bounded
analysis inputs retain their hashes.

## One validated, unscored capture

The [protocol](profiles/compiler-observation-spec.json) and
[runner](attempts/22-compiler-observation-runner.patch) were frozen in fefc1fa
before execution. One unchanged v12 reference executable runs Mona, seed 7,
64 batches, two warmups, diagnostic false and ordinary Stages timestamps. It
completes 1,048,576 measured evaluations, 4096 iterations, 64 batches and 64
samples with zero recorded errors. Source-accounted warmups add
32,768 evaluations. Full-fitness/reset GPU work remains separate.

The parent offline audit passes all 339 predicates. Both saved canonical states
match accepted reference in all seven fields. All eight parents match exact
cached-versus-framebuffer RGB-byte-L1; maximum independent f32 fitness difference
is zero under the unchanged 2e-5 policy. Previously independently audited numerical
functions are reused unchanged. No scalar rerender or fresh independent numerical
audit is claimed. All 83 archived source entries still match root/archive, three
ordinary binaries and two helpers retain their identities, and six complete shader
strings are verified inside the immutable release executable.

The run captures 24 blobs and a 5,377,711-byte compiler log, among 32 files totaling
26,658,154 bytes. It returns zero; controller duration is 22.627 seconds, within
the declared bounds. These are capture/resource facts, not engine performance.
There is no retry, second worker, candidate source, Rust build, feature change,
new pass or production modification.

The raw frozen TrialRecord still contains its ordinary rate. It remains unchanged;
the surrounding bundle is compiler_observation with score_eligible:false. All
elapsed/rate fields are excluded from performance scoring and comparisons.
Compiler logging can perturb execution. This single observation establishes no
speedup or repeatability and supplies no final NVIDIA ISA, dynamic instruction
counts, utilization, occupancy, register spills, bandwidth limit or hardware cause.

## Repeatable diagnostic route

The installed driver accepts `DXIL_DEBUG=verbose,dump_module,dump_blob` in this
run. Fresh private working and Mesa-cache directories prevent the known application
cache from suppressing compilation. Global/native GPU caches remain unverified.
Matching flags are implemented by the
[official Mesa 25.3.6 source](https://archive.mesa3d.org/mesa-25.3.6.tar.xz) in
src/microsoft/compiler/nir_to_dxil.c. Kisak patch parity remains unverified; actual
captured output establishes this installed route. DZN_DEBUG=dxil disassembly is
behind _WIN32 in that upstream source and is unsuitable on Linux.

Nsight tools are installed, but hardware-counter support for the translated
Dozen workload is not verified. Selection and framebuffer costs remain combined
in the ordinary timer. A same-pass midpoint requires an additional enabled feature;
its adapter capability was not queried and no private observer was built here.
The [machine record](profiles/compiler-observation-result.json) and
`.cache/evolution-observation-20261024/` retain source/tool audits, complete raw
capture, mappings, excerpts, containers, correctness evidence and hashes.

Attack the Premise replaced another layout guess with emitted-code evidence.
Prove It Works required actual dumps, container checks and replay. Explain the
Number keeps compiler structure separate from a measured hardware limiter.
