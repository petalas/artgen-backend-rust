# Keep the original readback scheduling

REJECT the later-collection prototype at the frozen short filter. Its aggregate
throughput ratio is 0.997520819, conditional 95% interval
[0.991751437, 1.003323765]. It does not clear the 1.005 advancement gate. All
correctness checks pass, but this experiment establishes no engine gain. No
full-budget or reserved extension runs. Runtime prototype code is removed.

| Case | Paired median | Paired range |
| --- | ---: | --- |
| ff | 1.000483122 | [0.988151213, 1.012815031] |
| mona | 0.996485587 | [0.993377466, 0.999593709] |
| darwin | 0.995693832 | [0.990640771, 1.000746893] |

| Case | Original evaluations/s, median [range] | Later collection, median [range] |
| --- | ---: | ---: |
| ff | 336025 [332598, 339453] | 336145 [335431, 336860] |
| mona | 269001 [267632, 270370] | 268051 [267523, 268580] |
| darwin | 269706 [268879, 270532] | 268540 [268000, 269080] |

The prototype encodes the current compute passes before collecting the same
pending drawing-buffer conflict. Collection still precedes drawing-copy selection
and submission. Both parameter upload and control-reset construction/upload move
with collection. Finish can update the CPU best-fitness threshold; moving finish
alone would change reset bytes and ordinary delivery. GPU dispatches, pass
boundaries, copies, queries, shaders, precision, RNG and parameters are unchanged.
The default path reconstructs byte for byte from the shared helpers.

The actual ordinary-API checker passes eight unscored mature-mona runs, seed 7,
eight batches and two warmups. Both timestamp-disabled and timestamp-enabled A/A
pairs pass before either A/B pair runs. All four pairs agree exactly on ordinary
Some/None positions, returned drawing geometry and own-header fitness, copied
counters and chain fitness. Delivered headers remain members of their copied
chain fitness, including legitimate nonmaximum scores. Captured terminal flush
results match an actual final maximum parent in geometry and fitness. Seven-field
canonical states, numerical qualification, integer byte-L1, completed work and
timestamp sample counts pass. The checker captures flush before the checked drain
and adds no per-batch snapshot or drain. It reports no throughput score.

The resident screen uses two alternating pairs per original project, eight
batches, seed 7, two warmups and unchanged settings. Twelve normal trials, six
separate overflow replays and three reference qualifications pass with zero
errors. Normal work totals 1,572,864 completed evaluations, diagnostics add
786,432 and qualifications add 393,216, excluded from timing statistics. The
separate API check adds 1,048,576. Actual timing and diagnostic records identify
original collection as false and later collection as true, with identical AE1
shader bytes, G1 dispatch metadata and TRUE mutation initialization. Every pair
is retained. Background isolation remains unverified.

An independent offline audit reproduces the ratio and interval, verifies all
eight API runs, 26 retained canonical snapshots and 104 parent byte-L1 totals,
and hashes 205 retained inputs. Its 650 checks pass. The largest CPU-fitness
difference is 7.62939453125e-6, within the unchanged 2e-5 numerical policy.

The sampled final-iteration raster intervals remain about 0.41/0.56/0.55 ms for
ff/mona/darwin in both arms. Those samples neither measure the whole-run raster
share nor isolate CPU stalls. The earlier observer's roughly one-second sum of
finish-call durations included GPU completion and mapping. This experiment
rejects one scheduling change; it does not show that readbacks are free, identify
a GPU hardware limiter, or turn that observed sum into recoverable time.

Scoped compilation, thirty focused CPU tests and actual help/invalid-CLI checks
pass. Independent Standards and Spec reviews find no material issue. Strict
Clippy retains nine existing library blockers. The full all/pedantic pass has no
example findings; the required fourth private evolver bool crosses its existing
struct-size heuristic without a suppression. That bool and its runtime API are
removed with the rejected prototype. Cleanup restores all three tracked files
exactly to a6eedb1 and removes the new checker after archiving its bytes. Scoped check and 42 existing CPU tests pass
on the restored code. All six production shader hashes remain unchanged.

Immutable release v13 archives 84 source files and the complete include graph.
Its engine-lab SHA256 is
d7c3f4aacebc464775c33e6a356a8cb38cf47086f07109bd9979e433b7b6774c;
the actual API checker is
1a830245951c7c3ed54422b0363701b440f6be6269d40f511d409febc002e68f.
Optimization is 3 and debug assertions are false. All three production shader
bytes are verified inside the executables.

Evidence is under `.cache/evolution-continuation-20261024/late-readback-66/`,
`build-v13/`, `late-readback-api-proof/`, `late-readback-screen-run/`,
`late-result-audit-70/` and `late-readback-rejected-cleanup-69/`. The pre-data
measurement specification, raw events, process records, source reconstruction,
archived patch and all timing rows are retained. The short interval is conditional
on same-seed repeats and is a screening result, not independent-seed acceptance.

Explain the Number kept observed finish duration separate from recoverable
stall. Prove It Works required the actual ordinary API and final population
checks before timing. The failed advancement gate ends this prototype.
