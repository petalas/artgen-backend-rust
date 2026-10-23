# First mature-project reference capture

The user captured this reference on 23 October 2026 after the benchmark setup and
numerical repairs. It is usable as the reference for this fixed three-project,
seed-7 suite. It does not freeze the broader optimization campaign.

Raw evidence: [complete capture](../../../.cache/evolution-bench/baseline-20261023T172054.919294Z-95742/summary.json).
The bundle includes the binary, source snapshot, fixtures, 45 worker records,
quality/state artifacts and before/after machine observations. The local
[recomputed analysis](../../../.cache/hillclimb/baseline-analysis.json) checks the
recorded statistics, work counts, fixture/binary hashes and worker results.

## Throughput and repeatability

The primary reference score is **60,333 completed evaluations/second**, the
equal-weight geometric mean of the three per-project median rates. It includes
CPU command encoding, ordinary result collection and the final GPU drain.

| Project | Initial GPU similarity | Initial triangles | Median evals/s | Min to max evals/s | CV |
|---|---:|---:|---:|---:|---:|
| ff | 97.34536% | 423 | 69,409 | 69,111 to 69,493 | 0.144% |
| mona | 97.18222% | 539 | 55,572 | 55,388 to 55,662 | 0.136% |
| darwin | 97.40385% | 474 | 56,936 | 56,765 to 57,101 | 0.180% |

Each project has ten normal trials arranged as five A/A pairs, plus five
overflow-diagnostic replays. Every trial completes 1,048,576 logical evaluations.
The normal trials account for 31,457,280 completed evaluations; diagnostics add
15,728,640 and contribute no timing score. All 45 trials pass with zero errors.
Initial/final canonical evolution hashes agree across normal and diagnostic
runs for each project. Cached RGB L1, independent numerical checks, initial
similarity and overflow limits pass.

The workloads keep evolving during measurement. Triangle counts finish at 481
for ff, 546 for mona and 520 for darwin, with improved fitness in every chain.
Rates describe these matched mature trajectories, not a fixed polygon count.

The largest observed absolute A/A pair difference is 0.463%. This is an observed
control spread, not a guaranteed noise bound for future sessions. The planned
3% practical improvement threshold has headroom over this capture's variation;
a future comparison still needs paired reference/candidate runs, its frozen
confidence procedure, per-case non-regression and quality gates.

## Identity and measurement scope

- Binary SHA-256: `ecaafba649c45c3fdcb952e1e8dbd90974042c0b32a0e2f780aaec3ddf2d0c41`.
- Fixture manifest SHA-256: `82dfe207ecd20ae964edf4528f1a90fe4176b21f4ac0911b305da1e1edf1c5aa`.
- Optimized locked release build, opt-level 3, debug assertions disabled.
- RTX 5090 through Vulkan/Dozen, Mesa 25.3.6; reported NVIDIA driver 610.88.
- Seed 7, four active chains, 64 offspring per chain, 64 iterations per batch,
  64 measured batches, two warmup batches, 32x16 raster workgroup.
- Production timestamps enabled; longest reference side 384px; default mutation
  settings frozen in the manifest. These are mature-drawing restarts.

The source is an archived working-tree snapshot based on `543b766`, not a new
commit. Before comparing another build, match this binary/fixture/protocol
identity and preserve the capture bundle.

The operator intended a quiet session. CPU one-minute load point samples range
from 0.170 to 0.704 on 16 logical CPUs. GPU point samples report 0 to 1% utilization
between worker runs, with temperatures 41 to 49 degrees Celsius. These observations
are compatible with low background activity; they cannot prove isolation during
every measured interval. The backend connection was unavailable in all 90
observations, with connection-refused errors, not an observed evolving backend.

`clean_baseline: null` intentionally preserves that uncertainty. It is not a
failed numerical or accounting check. `verdict: no throughput verdict` means
there is no optimization candidate to compare. Do not edit these raw fields to
assert stronger evidence than the capture contains.

## Next profiling target

Polygon binning accounts for approximately 72 to 73% of the summed sampled GPU
stage intervals in all three cases. Mutation, rasterization/error and
selection/framebuffer updates account for the rest. Samples cover only the last
iteration of each batch; these percentages are not shares of total application
wall time and are not GPU utilization measurements.

The next investigation should profile binning and host submission/wait behavior
separately from reported timings. Confirm the end-to-end limiting resource before
choosing an implementation change. Serial AABB scattering per offspring is a
source-based hypothesis to investigate, not a demonstrated optimization win.
Live-path acceptance, true continuation workloads and broader seed/held-out
validation remain later campaign gates.
