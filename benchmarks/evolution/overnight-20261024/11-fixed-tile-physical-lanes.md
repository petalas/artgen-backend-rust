# Adopt fixed-tile physical raster lanes

ADOPTED. Attempt 26, raster256, improves measured resident engine throughput
against decoded C on the frozen workload. The reserved aggregate ratio is
1.0626356401569936, with conditional 95% interval
[1.0582641999915285, 1.0671493731764186]. This is 6.2636% higher throughput
on the recorded backend. Training independently passed at 1.0637084947108335,
with interval [1.059230238728394, 1.0682130848812341]. Both clear the unchanged
campaign gates. AE1 is the production source after these checks. These ratios compare directly with C and must not be multiplied by
historical gains or the preliminary screen ratio.

## Frozen identity and work

The adopted source is
`.cache/evolution-overnight-20261024/raster-lanes/raster-lanes-256.wgsl`,
SHA-256 `ae1fd75fdc118debf2a50cbad699e513b5af0dcbfa3754b45966f7a8ee925af3`.
The decoded-C reference is
`raster-decoded/raster-c-decoded-aabb.wgsl` under the same cache root,
SHA-256 `a292287ae6d9c212c7970dd3e712001a1b5a827975e6216355384fc0106eb3af`.
Production `src/shaders/rasterize_error.wgsl` matches the adopted source at report
freeze. The predeclared contract is `raster-lanes-confirmation/spec.json`.
Its SHA-256 is `f93891c4e1cdd5b050271946de75ca03cd15d197f503ab6ec230ddbb0f93ef63`.

All MutationParams remain fixed, including logical 32x16 tiles, four chains,
lambda 64, 64 iterations per batch, mutation probabilities and crossover.
Each logical tile still assigns all 512 pixel ordinals. Physical invocation count
is `min(256, WG_X*WG_Y)`, so the measured tile uses 256 lanes and two pixels per
lane. Each pixel retains its floating-point calculation and byte quantization.
Only already-integer new and old errors are summed across a lane's pixels.
Compaction was disabled throughout measurement and its rejected runtime has since
been removed. The binner, mutation shader, selection, image
resolution, fixture identities, RNG and reset trajectories remain fixed.

Training used seed 7. Reserved confirmation used the distinct mature corpus and
seeds ff 11, mona 13 and darwin 17. Each phase ran five alternating pairs per case,
64 measured batches per trial and two matching warmup batches. Every normal trial
completed 4,096 iterations and 1,048,576 evaluations after the checked drain.
The timed resident engine loop includes ordinary encoding, submission, polling,
drawing readback and drain. Setup, reset, compilation, extra verification,
snapshots, hashing and report writes are outside the timing region.

## Per-case rates and ranges

Rates below are evaluations per second, rounded to one decimal. Each rate is the
median of five trials, followed by the minimum and maximum. Ratios are medians of
five paired throughput ratios, followed by their minimum and maximum. A paired
ratio need not equal the ratio of the two independently reported medians.
Every timed sample, including slow pairs, remains included.

| Phase | Case | C evaluations/s, median [min, max] | AE1 evaluations/s, median [min, max] | Paired ratio, median [min, max] |
| --- | --- | ---: | ---: | ---: |
| Training | ff | 262,168.2 [253,896.2, 268,539.1] | 282,786.6 [272,970.6, 291,183.5] | 1.084324 [1.074823, 1.103110] |
| Training | mona | 214,852.9 [212,088.9, 216,162.8] | 226,898.6 [226,164.2, 228,505.9] | 1.055944 [1.049922, 1.066365] |
| Training | darwin | 219,337.0 [217,718.5, 220,793.3] | 229,388.7 [227,689.7, 231,431.2] | 1.048835 [1.031235, 1.051667] |
| Reserved | ff | 249,475.5 [245,045.3, 258,097.5] | 268,112.2 [263,876.2, 276,181.7] | 1.076846 [1.070067, 1.097940] |
| Reserved | mona | 221,448.6 [218,102.8, 226,630.9] | 233,911.0 [230,823.7, 238,295.5] | 1.058325 [1.045975, 1.069884] |
| Reserved | darwin | 223,838.4 [218,193.4, 230,200.8] | 233,196.1 [227,774.4, 238,953.2] | 1.043910 [1.038021, 1.057042] |

The aggregate statistic uses equal-project geometric means of paired ratios.
The recorded intervals use 10,000 paired-log bootstrap resamples with seed
20261024. They are conditional on timing repeats of these fixed trajectories,
not independent search-seed or hardware-population intervals. Each case passes
the median gate of 0.995 and interval lower-bound gate of 0.99 in both phases.

## Sampled raster attribution

The analysis field `sampled_stage_median_ns` contains nanoseconds per sampled
iteration, not milliseconds or whole-engine wall time. The raster component below
is converted to microseconds by division by 1,000. Each entry is the median across
five trials of the trial's sampled raster total divided by its sample count.

| Phase | Case | C raster microseconds | AE1 raster microseconds |
| --- | --- | ---: | ---: |
| Training | ff | 510.7570 | 436.4930 |
| Training | mona | 621.9630 | 560.4270 |
| Training | darwin | 603.1260 | 554.3980 |
| Reserved | ff | 516.7065 | 439.1645 |
| Reserved | mona | 623.5390 | 563.2860 |
| Reserved | darwin | 600.2125 | 555.2760 |

The sampled raster stage remains the largest of the four sampled GPU stages.
Its lower time supports attribution of the measured engine gain to this raster
change. Those samples do not measure occupancy, hardware utilization or the
raster share of the whole resident loop. Fewer physical invocations are a source
mechanism; an occupancy or scheduling explanation remains unverified.

## Exactness and independent audit

`raster-lanes-confirmation/independent-audit.json` records `valid=true` and
`error_count=0`. The independent audit verified all 60 normal trials and 12 diagnostic
replays, plus six qualification runs. Completed work is 62,914,560 normal,
12,582,912 diagnostic and 6,291,456 qualification evaluations. Diagnostic rates
are null and excluded from throughput statistics. No process, device, replay,
capacity, overflow or numerical errors were accepted as timing samples.

The audit checks exact initial and final replay of all seven canonical fields,
complete parameter hashes, source identity, GPU identity, submitted/completed
work, alternating order and warmup counts. The fields are parents, offspring,
parent total errors, parent framebuffers, offspring errors, chain update flags
and packed fitness. It independently verifies 144 raw framebuffer-byte RGB-L1
totals against cached errors, checks finite fitness and geometry association,
and matches the historical training and reserved endpoint anchors. The audit script and its output remain retained with the raw artifacts.

The candidate-specific release executable is
`raster-lanes-confirmation/build/check-gpu-rasterization`. Its recorded numerical
run passed both fixture loops at all five supported logical shapes, 8x8, 16x8,
16x16, 32x8 and 32x16. The loops cover odd image bounds, layered colors, reference
bytes and the visible-layer regression. `numerical/process-record.json` records
exit 0, and `numerical/process/stdout.log` ends with
`GPU rasterization checks passed`. Both loops were expanded
in `examples/check-gpu-rasterization.rs` from three shapes to these five.
This is candidate-specific evidence, not a result inherited from C.

## Source ownership and remaining risk

The source proof and review record in `raster-lanes/README.md`, its manifest and
`lanes-256-vs-c.diff` explain the lane ownership and preservation obligations.
For tile size T and physical lanes P, ordinal o has unique owner `o mod P` and
iteration `floor(o/P)`. Coordinates still derive from o and the logical tile.
The cooperative loader uses the same ownership argument for each readable cache
slot. All lanes in a non-skipped workgroup reach the cache and reduction barriers,
even when their image pixels are invalid. Invalid pixels contribute integer zero
without leaving the entry point. Pixel blending retains ascending polygon order.

The decoded cache retains 256 entries at 48 bytes each, or 12,288 bytes.
Reduction arrays and dirty metadata retain their capacities. The declared
workgroup storage total is 13,332 bytes, or 13,392 with each global conservatively
rounded to 16 bytes. These source totals do not establish physical allocation.
Separate new and old errors are each at most 765 per pixel, 1,530 per lane at
32x16, and 391,680 per tile. All fit u32. The reduction uses actual subgroup IDs
and subgroup count, preserving C's existing subgroup support contract.

The source-preservation guards retain the ABI, blend expressions, loader body and
reduction suffix. Source identity alone cannot establish machine floating-point
identity after changed helper boundaries or invocation count. The focused
numerical checker and exact trajectory/state/L1 replay supply that runtime
evidence on the recorded backend. Serial pixel work, compiler choices and
register pressure can behave differently elsewhere.

The recorded adapter is `Microsoft Direct3D12 (NVIDIA GeForce RTX 5090)`, exposed
through Vulkan by Dozen, Mesa 25.3.6. Both sides use that same backend and device
metadata. The logs retain the dzn nonconformance warning. Background isolation
remains unverified. No broad hardware, utilization, cross-backend or evolution
quality claim follows from this confirmation.

## Dispositions and restoration history

[Report 10](10-raster-edge-and-lane-screens.md) retains the preliminary evidence.
The 72-byte edge cache was rejected before GPU execution for exceeding capacity.
The 56-byte edge cache and lanes128 passed the short filter but were not selected
for further samples. Only lanes256 advanced, selected by the highest preliminary
aggregate point estimate. No gain from an unselected candidate is combined here.

The temporary build wrapper succeeded, then shadowed its restoration Path with a
CompletedProcess value. `raster-lanes-confirmation/restoration.json` records the
verified restoration from immutable C and the absence of a build rerun. That
incident remains part of the record. It does not describe the final source state;
the later intentional adoption installs AE1. The cached prototype manifest and
`outcome.json` preserve their earlier pending-adoption status. The final decision
in this report follows the passed confirmation and independent audit.

Post-cleanup CPU checks passed 31 focused library, 12 lab and seven mutation
checker tests. All/pedantic Clippy completed successfully; repository diagnostics
remain outside the changed lines. These checks verify the execution seam used
for the next experiments; they are separate from the candidate-specific GPU proof.

Recorded evidence lives under `.cache/evolution-overnight-20261024/` in
`raster-lanes/`, `raster-lanes-train-full/output/`,
`raster-lanes-confirmation/`, and `raster-lanes-reserved-{ff,mona,darwin}/output/`.
The final training and reserved statistics are respectively
`raster-lanes-train-full/output/analysis.json` and
`raster-lanes-confirmation/reserved-analysis/analysis.json`.
