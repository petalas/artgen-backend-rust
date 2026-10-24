# Full-batch profile of the accepted engine

The full-batch GPU queue compute span occupies a median 78.45% to 81.38% of
measured evolution wall time in batch-envelope mode. GPU compute remains the
largest measured interval. The remaining 0.73 to 0.84 seconds per 64-batch run
needs attribution before another host optimization. It includes uncovered GPU
copies, queue/submission gaps and completion handling, so it is neither CPU time
nor recoverable stall. This is a profile, not an optimization result.

| Case | Evolution wall seconds | GPU span seconds | Wall minus span seconds | Median span / wall |
| --- | ---: | ---: | ---: | ---: |
| ff | 3.409 [3.367, 3.473] | 2.656 [2.642, 2.707] | 0.729 [0.709, 0.766] | 78.45% |
| mona | 3.985 [3.971, 4.002] | 3.243 [3.240, 3.243] | 0.742 [0.729, 0.762] | 81.38% |
| darwin | 4.087 [4.069, 4.214] | 3.243 [3.242, 3.244] | 0.844 [0.825, 0.970] | 79.35% |

Each cell is the median and minimum–maximum across five envelope workers.
Differences and fractions are computed per worker before aggregation; medians
are not assumed additive. Query 6 begins the first bulk compute pass, and query
7 ends the final selection/framebuffer-update pass. Their difference covers all
64 iterations in each batch, including barriers and GPU stalls. It excludes
preceding uploads, following copies and readback. It is elapsed queue time,
not GPU utilization or evidence of a bandwidth/occupancy bottleneck.

## Matched collection

Five alternating Stages/BatchEnvelope pairs run on each original mature drawing,
seed 7, 64 measured batches and two warmups. All settings, precision, RNG,
compiled shaders, initialization policies and GPU identity remain fixed. The
same immutable accepted v12 executable runs all workers. Its 83 frozen source
files match the current engine. Both modes receive copies of one immutable
warm current-engine cache from an excluded mona pilot. Every worker has a private
working directory. No builds or independent heavy audits run during worker timing.

Thirty workers complete 31,457,280 measured evaluations with zero process,
device, numerical, capacity, replay or artifact errors. Each completes 64
batches, 4096 iterations and 1,048,576 evaluations, with 64 timestamp samples.
Saved initial/final canonical fields, byte-L1, quality and parameters match the
frozen reference. Envelope records retain null throughput scores, and the entire
profiling bundle is excluded from primary performance scoring.

Normal Stages wall seconds are ff 3.557 [3.379, 4.183], mona 4.024
[3.963, 4.054] and darwin 4.070 [3.935, 4.099]. In particular, ff variability
prevents assuming instrumentation is free or transferring the envelope fraction
unchanged to every normal run. No slow pair is removed. Background isolation
remains unverified. Setup and post-run qualification/export are outside the
measured loop; normal readback and checked final drain are inside.

The final-iteration Stages samples remain dominated by rasterization:

| Case | Mutation ms | Binning ms | Rasterization ms | Selection + update ms |
| --- | ---: | ---: | ---: | ---: |
| ff | 0.0901 | 0.0501 | 0.4329 | 0.0827 |
| mona | 0.1083 | 0.0616 | 0.5566 | 0.0821 |
| darwin | 0.0991 | 0.0556 | 0.5504 | 0.0916 |

These are medians of worker-average sampled intervals. They cover the last
iteration of each batch and do not establish whole-run stage shares. Envelope
mode keeps the first three sampled stages but uses the selection query pair for
the whole batch, exporting selection as unavailable/zero.

## Separate host observation

One unscored current-engine mona worker uses the same seed, budget and warmup.
Its 3.998-second loop passes exact replay and quality. A guarded 1.514-second
interior contains 61 `/proc` observations: 45 report `do_sys_poll`, and 16 report
the main thread running. Process and main-thread CPU both total 0.39 seconds,
about 25.76% of one core in that window. This is one cropped observation, not a
whole-loop CPU fraction, and cannot be added to the GPU span percentage.

The user-CPU profile contains 195 samples and zero lost samples. Inclusive
stacks establish deferred backend encoding inside `encoder.finish()` through
`encode_commands` and `encode_compute_pass`. Visible finish ancestry accounts
for 6.67% of sampled user CPU, not wall time. The largest 34.87% self entry is an
unresolved NVIDIA driver offset without usable caller ancestry. Submission,
front-end recording and finalization therefore lack complete phase attribution.
There is no supported optimization claim yet.

The existing observer's whole finish-call durations are not reused as savings.
The rejected later-readback prototype moved front-end recording but left
`encoder.finish()` after collection. Current stacks distinguish those operations;
they do not by themselves justify reopening that experiment. A separate private,
buffered phase observation is the next diagnostic, with all production code
remaining unchanged.

## Validation and provenance

Independent offline audit passes 775 checks. It reproduces raw results,
statistics, pair order, 64-sample accounting and input/source identities; 248
saved-parent associations pass byte-L1/f32 fitness recomputation with maximum
CPU-fitness difference zero under the unchanged 2e-5 policy. Saved scalar/pixel
oracle records pass; scalar rasterization is not rerun. Host perf data reproduces
the cropped report. The parent's offline analysis finishes 7.1467 seconds before
the guarded host window, so its reads do not overlap that crop.

The accepted executable SHA256 is
0cdf0caa4cb361f997fd42902e042ab3463a40049727047eecd4c0ddf74be732.
It is a locked release build with optimization 3 and debug assertions false.
All three production shader bytes remain unchanged and embedded. The adapter is
RTX 5090 through Vulkan/Dozen, Mesa 25.3.6. No evolution parameter is changed.

Evidence is under `.cache/evolution-profiling-20261024/`: `matched/`,
`analysis.json`, `host-mona/`, `host-inclusive-report/`, `audit-75/`,
`seams-73/` and `host-cost-76/`. The excluded pilot remains in `pilot-envelope/`.
The executed local collectors and immutable pre-data protocol are retained.
Explain the Number separates queue time, elapsed residual, on-CPU sampling and
nested waits. Prove It Works requires matching actual work and saved states
before interpreting any timing.
