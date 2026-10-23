# First evolution profile

The first diagnostic profiles support focusing on GPU compute before host
encoding. Polygon binning is the largest sampled stage and the first source
investigation target. No optimization has been attempted or accepted.

The measurement setup was committed as `21eb782`. These profiles use the same
frozen ff, mona and darwin mature-drawing restarts as the [reference](baseline.md),
with seed 7, four chains, 64 offspring per chain, 64 iterations per batch,
64 measured batches and two warmup batches. Each process completes 1,048,576
evaluations. Every initial parent exceeds 97% fitness.

## Observations

Each project has one host profile and a separate GPU timestamp profile. These
are diagnostic observations, not repeated reference/candidate timing trials.

| Project | Host CPU, fraction of one core | Main-thread poll observations | GPU compute span / evolution wall time |
|---|---:|---:|---:|
| ff | 6.37% | 92.86% | 95.97% |
| mona | 5.32% | 94.76% | 96.83% |
| darwin | 5.50% | 95.53% | 96.19% |

Host CPU includes user and system time across all worker threads. The cropped
steady windows contain 504, 649 and 627 samples respectively, over approximately
12.71, 16.36 and 15.81 seconds. The poll column counts observations of the main
thread in `do_sys_poll`; it does not count GPU operations. CPU activity and GPU
execution overlap, so these columns must not be added.

The host runs use the archived reference executable. `perf record` samples
user CPU at 499 Hz, and `/proc` supplies CPU counters and main-thread wait state
at a target 25 ms interval. Analysis starts one second after the first completed
measured-batch marker. It ends one second before a conservative estimate of
measurement completion. Setup, warmup, numerical validation and artifact writes
stay outside that cropped window. All three steady `perf report` windows report
zero lost samples. Closed NVIDIA driver code accounts for roughly one third of
their sampled user CPU time, which is a share of the small on-CPU interval,
not of evolution wall time.

The separate timestamp mode moves query 6 to the beginning of the first bulk
compute pass and retains query 7 at the end of the last selection/framebuffer
update pass. Their difference covers every iteration in each batch, including
pass boundaries. It excludes preceding uploads and following copies/readback.
It measures elapsed GPU queue compute span, including stalls, rather than GPU
utilization. It does not establish a bandwidth, occupancy or other hardware
bottleneck.

All three timestamp runs contain exactly 64 samples after warmup. Last-iteration
mutation, binning and rasterization samples remain available. Selection timing
is unavailable and records zero because its query pair measures the full batch.
The profiling result's `evals_per_second` is null, and the reference supervisor
rejects these records from its primary score.

## Verification and evidence

All six processes completed with zero errors and 6,291,456 completed evaluations
in total. Initial/final canonical state hashes, parameters, counters, quality and
adapter identity match the reference exactly. Independent framebuffer byte-L1
checks match cached totals for all 24 final parents. Profiling changes neither
the corpus nor the archived throughput reference.

Local retained evidence:

- [Combined checks and measurements](../../../.cache/evolution-profile/analysis.json).
- Host raw samples, `perf.data`, steady reports and worker states under
  `.cache/evolution-profile/host-{ff,mona,darwin}/`.
- Timestamp logs and worker states under
  `.cache/evolution-profile/envelope-{ff,mona,darwin}/`.
- [Archived timestamp executable and provenance](../../../.cache/evolution-profile/envelope-provenance.json),
  with its source diff and SHA-256. The host executable remains in the original
  reference bundle.

These are local ignored artifacts. Preserve their directories when retaining or
moving the reference evidence. One profile per case does not qualify a speedup,
a future noise bound or a clean-machine reference capture.

## Repeat the profiles

```sh
python3 scripts/profile-evolution.py \
  --baseline .cache/evolution-bench/baseline-20261023T172054.919294Z-95742/summary.json
```

The command serially runs the archived binary for host sampling, builds and
archives the current release example for timestamp sampling, and clones frozen
fixtures and cache seeds into a new bundle. It verifies replay, numerical checks,
saved state hashes, completed work and sample counts. It records no primary
throughput score. `--case ff` selects one project; `--host-only` and
`--envelope-only` select one mode. `--binary PATH` supplies the timestamp binary.

The script supports Linux host sampling with readable `/proc` and usable `perf`.
Recognized perf startup permission/event failures produce unavailable evidence;
worker crashes and invalid results fail the run. Process groups, output sizes,
sampling work and execution time are bounded. Zero-warmup reference captures are
explicitly unsupported. A real combined ff command is retained in
[the CLI proof bundle](../../../.cache/evolution-profile/cli-ff-proof/summary.json).

## First optimization hypothesis

The existing reference attributes approximately 72 to 73% of summed sampled GPU
stage time to binning. That attribution covers only the final iteration of each
batch; it cannot be applied directly to the full compute span or wall time.

`bin_polygons.wgsl` serially resets every tile count and scatters every polygon
into all overlapping tiles for each offspring. `rasterize_error.wgsl` returns
before reading those bins when a tile lies outside the offspring's dirty box.
The first proposed experiment is to avoid constructing bins that rasterization
cannot consume, preserving polygon order and all consumed tile contents.

This is a source-based hypothesis. Before accepting it, verify empty/inverted
dirty boxes, partial edge tiles, full-image mutations, changing dirty regions,
all supported workgroups, overflow diagnostics and exact evolution replay.
Measure paired reference/candidate runs across all three projects, keeping fixed
work, settings and quality gates. An accepted result still needs the frozen
comparison policy and live-path non-regression gate described in the campaign
plan. Do not reduce image resolution, polygon complexity or candidate evaluation
to obtain a throughput gain.

Explain the Number drove the separate CPU/wait and full-compute measurements.
Build the Lever drove the rerunnable profiling command. Prove It Works drove its
real CLI replay and the independent state/error checks.
