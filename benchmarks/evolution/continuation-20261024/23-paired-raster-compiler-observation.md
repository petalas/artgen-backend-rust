# The kept raster shares polygon loads across both pixels

The actual Mesa NIR and DXIL output retains one polygon traversal and eleven
shared cache loads for the two pixels owned by each lane. Six vertex values,
four bounds and packed color feed both pixel paths. The source-level reuse in
the [kept paired shader](22-paired-pixel-raster.md) survives this lowering.
This observation supplies no new speed measurement or hardware explanation.

No further optimization is proposed from this capture. Some arithmetic remains
duplicated, but moving it across coverage guards can add unused work or change
compiled floating-point behavior. The closed scalar44, normalized-color,
deferred-load and bbox experiments remain closed. Production sources, normal
executables and evolution parameters are unchanged.

## Emitted code

The [exact excerpts](profiles/paired-compiler-raster-excerpts.txt) retain original
log line numbers and its SHA256. NIR lines 56253-56270 and DXIL lines 57379-57400
show eleven shared loads before both guarded pixel paths. Both consume the same
loaded values. The polygon ordinal advances once after those paths. AE1's
[prior compiler capture](20-compiled-shader-observation.md) placed these loads
inside its separate traversal for each pixel. Its outer pixel loop is absent
from the paired output; the remaining 512 constants guard pixel validity.

| Work | Actual paired compiler output |
| --- | --- |
| Polygon traversal and cached geometry/color loads | One set feeds both pixels |
| Tile-origin shifts and cache indices | Shared |
| Coordinate division, reference fetch and old-framebuffer read | Separate per pixel |
| Coverage, color normalization, RGB rescaling and blend | Separate per pixel |
| Shared field arrays, cooperative loader and reduction | Retained |

Each covered-pixel branch still contains four color-normalization divisions and
three RGB rescalings. Six polygon-only edge differences also remain duplicated.
These are static sites in this compiler output, not dynamic instruction counts
or measured costs. The DXIL tree's printed division mnemonic is not a final
device instruction. There is no evidence here for register spills, occupancy,
bandwidth saturation, utilization or a hardware cause of the earlier gain.

Shared declarations still contain three float[512] vertex arrays, four float[256]
bounds arrays and uint[256] packed color. Their element storage totals 11,264
bytes, without establishing final hardware allocation. The cooperative-loader
text is identical to AE1. Three barriers, both subgroup integer sums and both
atomic error additions remain. All five other application blobs are byte-identical
to their counterparts in the prior capture.

Workgroup 256,1,1, the reference texture, storage bindings 0/2/3/4/6 and dual error
sums identify the raster section independently of file order. Its properties
match shader_cs_19.blob, SHA256 8204dcc8..., with thread dimensions checked directly
at bytes [148,160). The section occupies original log lines 55616-58352, bytes
[3215305,3338247). Full mappings and exact hashes are in the
[machine record](profiles/paired-compiler-observation-result.json).

## One unscored default-path capture

Commit 143b6f6 freezes the [protocol](profiles/paired-compiler-observation-spec.json)
and [runner](attempts/25-paired-compiler-observation-runner.patch) before execution.
One immutable rebuilt default evolution-bench executable, SHA256 1cf30fea...,
runs Mona, seed 7, 64 batches, two warmups, ordinary Stages timestamps and
diagnostic false. It uses the already-kept bundled raster 83c505bc..., without a
source override. Fresh private working and Mesa-cache directories expose the
compiler output through DXIL_DEBUG=verbose,dump_module,dump_blob.

The worker completes 4096 iterations and 1,048,576 measured evaluations with
64 timestamp samples and zero recorded errors. Source-accounted warmups add
32,768 evaluations; full-fitness/reset work is separate. It returns zero and
captures 24 DXBC/DXIL blobs among 32 files totaling 26,687,094 bytes. The compiler
log is 5,400,641 bytes. There is one worker, no retry and no Rust build.

The [offline validator](attempts/25-paired-compiler-data-validation.patch) passes
594 predicates, reusing the previously audited numerical functions unchanged.
Both saved canonical states match the accepted trajectory in all seven fields.
All eight parents have exact cached-versus-framebuffer RGB-byte-L1; independent
f32 fitness differences are zero under the unchanged 2e-5 policy. No scalar
rerender or fresh independent numerical audit is claimed. The 83 source files,
three normal binaries, two helpers and prior capture inputs retain their hashes.
All six full shader strings match the frozen executable. An additional 82 parent
checks verify analysis inputs, exact excerpt slices, actual blob dimensions,
shared SSA uses, loader equality and the six application-blob comparisons.

The raw TrialRecord retains its ordinary rate. The enclosing record explicitly
sets score_eligible:false and excludes all elapsed/rate fields from comparison.
Compiler logging can perturb execution. The previously qualified 5.49% reserved
gain keeps its original samples and limits; this capture adds none.

The initial CPU preflight had an incorrect expected argument count. It was
replaced with exact frozen-command equality before any native execution. Ruff,
exact patch reconstruction and scoped whitespace checks pass. All raw files,
the bounded 44-input analysis inventory, repeatable extractor and final hashes
remain under .cache/evolution-paired-observation-20261024/.

Prove It Works required actual compiler bytes and numerical replay. Explain the
Number kept static reuse separate from dynamic costs and throughput causality.
No source change follows from this observation alone.
