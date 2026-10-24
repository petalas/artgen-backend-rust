# Adopt one offspring per mutation workgroup

ADOPTED. Attempt 28, G1, increases measured resident engine throughput against
original mutation e133 on the frozen workload. Reserved confirmation gives
1.1098016465320737, with conditional 95% interval
[1.1035078897417134, 1.1163566547613246], or 10.9802% higher throughput on the
recorded backend. Training passed separately at 1.1018080529719705, with interval
[1.0983005697485595, 1.1054331648520803]. Both clear the unchanged campaign gates.
`mutation-group-confirmation/adoption.json` records KEEP after the independent
audit. Production `src/shaders/mutate.wgsl` matches G1 at this report's freeze.
These ratios compare directly with e133 while using the same binner and raster.
They must not be multiplied by historical gains or the short-screen ratio.

## Frozen mechanism and work

Paths below are relative to `.cache/evolution-overnight-20261024/` unless stated
otherwise. The exact candidate is `mutation-split/mutate-g1-wg64-copy64.wgsl`,
SHA-256 `3a36417746ab19b0c800c718e8c87795c17d9bff78114fd65bc8369b016578d8`.
The reference is `mutation-split/sources/mutate-original.wgsl`, SHA-256
`e13308a7cc59071043a3f5750a7f25214933ae4645cd016875f2dfa12c3dfe73`.
The fixed binner is `598b32eab7ae998f2d644b36a5d71d17997772ed06dbd5c0b726130ef51d13d9`.
The fixed raster AE1 is
`ae1fd75fdc118debf2a50cbad699e513b5af0dcbfa3754b45966f7a8ee925af3`.
Both production hashes still match at freeze.

G1 uses 64 physical lanes and one logical offspring per workgroup. All 64 lanes
cooperate on the parent copy. Only the leader executes the original RNG and
mutation body. The host dispatch is `[active, ceil(lambda / 1), 1]`. Four chains
and lambda 64 still produce 256 evaluations each iteration. The entire
MutationParams object remains unchanged, including all mutation probabilities,
crossover, adaptive behavior, raster dimensions and 64 iterations per batch.
The recorded parameter hash is
`755575d7c391eaa45ba8358724d0c66ae499c445a388badb5d465fb07a824f60`.

Default mutation workgroup-memory zero initialization remains TRUE. It is
required for the empty-parent crossover fallback, which reads shared slot zero
when no parent polygon was loaded. This is part of the correctness contract.

`mutation-group-confirmation/spec.json` froze the schedule before confirmation
at 2026-10-24T03:35:22.721074+00:00. Each phase uses five alternating pairs per
ff/mona/darwin case, 64 measured batches and two matching warmup batches.
Training uses seed 7. Reserved confirmation uses distinct mature states with
seeds ff 11, mona 13 and darwin 17. Every normal trial submits and completes
4,096 iterations, 64 batches and 1,048,576 evaluations after the checked drain.
All 30 training and all 30 reserved timing trials remain included.

Timing covers the resident engine loop, including ordinary encoding, submission,
polling, drawing readback and the checked drain. Setup, reset, compilation,
preparation, extra qualification, snapshots, hashing and report writes are
outside the timed region. This is an engine-loop result, not an application or
startup latency measurement.

## Rates and repeatability

Rates are evaluations per second. Each cell gives the median of five trials
and its minimum and maximum, rounded to one decimal. Ratios give the median of
five paired throughput ratios and their minimum and maximum. Paired medians
need not equal the ratio of the two independent rate medians. Slow pairs remain
included.

| Phase | Case | e133 evaluations/s, median [min, max] | G1 evaluations/s, median [min, max] | Paired ratio, median [min, max] |
| --- | --- | ---: | ---: | ---: |
| Training | ff | 282,492.7 [278,052.8, 292,394.6] | 311,356.9 [303,931.8, 320,303.9] | 1.102177 [1.093072, 1.120153] |
| Training | mona | 224,382.1 [223,602.7, 227,376.3] | 248,762.3 [246,827.0, 250,977.3] | 1.102928 [1.100646, 1.112519] |
| Training | darwin | 224,895.7 [224,015.6, 226,656.0] | 246,282.2 [244,923.5, 247,842.0] | 1.093920 [1.090482, 1.099873] |
| Reserved | ff | 266,198.8 [261,354.6, 278,791.1] | 293,624.5 [291,657.0, 305,015.1] | 1.108447 [1.092290, 1.140655] |
| Reserved | mona | 232,485.3 [227,125.9, 239,951.3] | 257,234.5 [250,824.7, 264,445.7] | 1.104342 [1.100663, 1.120586] |
| Reserved | darwin | 232,884.5 [224,013.4, 237,711.3] | 258,145.1 [249,396.0, 261,031.8] | 1.112427 [1.090548, 1.117014] |

The aggregate is the equal-project geometric mean of paired ratios. The
conditional intervals use 10,000 paired-log bootstrap resamples with seed
20261024. They describe repeats of these fixed trajectories, not independent
search seeds or a hardware population. Both phases pass the aggregate ratio
gate of 1.005 and lower-bound gate above 1.0. All cases pass the paired-median
gate of 0.995 and case interval lower-bound gate of 0.99.

| Phase | Case | Case geometric-mean ratio | Conditional 95% interval |
| --- | --- | ---: | --- |
| Training | ff | 1.105382893 | [1.095839440, 1.115009459] |
| Training | mona | 1.105266364 | [1.101305492, 1.109560664] |
| Training | darwin | 1.094808433 | [1.092275285, 1.097738191] |
| Reserved | ff | 1.111635477 | [1.096569368, 1.127751812] |
| Reserved | mona | 1.109458230 | [1.101964924, 1.117002490] |
| Reserved | darwin | 1.108313797 | [1.099247407, 1.114612624] |

## Sampled stage attribution

`sampled_stage_median_ns` is in nanoseconds per sampled iteration. Each value
is the median across five trials of that trial's sampled stage total divided
by its 64 timestamp samples. The stage order is mutation, binning, raster/error
and selection. These samples cover one iteration per measured batch, not every
iteration in the resident loop.

| Phase | Case | Source | Mutation ns | Binning ns | Raster/error ns | Selection ns |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Training | ff | e133 | 176,090.5 | 50,501.5 | 433,608.0 | 81,383.5 |
| Training | ff | G1 | 90,018.5 | 50,380.0 | 431,677.5 | 82,705.0 |
| Training | mona | e133 | 215,126.5 | 61,737.0 | 568,865.5 | 82,060.0 |
| Training | mona | G1 | 108,762.0 | 61,823.0 | 563,456.0 | 81,843.0 |
| Training | darwin | e133 | 195,261.5 | 55,550.5 | 556,465.0 | 91,900.0 |
| Training | darwin | G1 | 99,373.5 | 55,895.5 | 554,539.0 | 92,528.0 |
| Reserved | ff | e133 | 193,060.0 | 53,395.0 | 442,543.0 | 85,827.5 |
| Reserved | ff | G1 | 97,954.0 | 53,394.5 | 441,915.0 | 86,635.5 |
| Reserved | mona | e133 | 223,731.5 | 63,380.5 | 562,968.0 | 86,067.5 |
| Reserved | mona | G1 | 113,063.5 | 63,886.5 | 561,697.0 | 85,030.0 |
| Reserved | darwin | e133 | 212,812.0 | 58,243.5 | 555,185.0 | 99,382.0 |
| Reserved | darwin | G1 | 108,180.0 | 59,000.5 | 557,374.5 | 100,793.0 |

The sampled mutation time falls in all six comparisons. For example, reserved
ff changes from 193,060 ns to 97,954 ns, or 193.060 to 97.954 microseconds.
This supports attribution to the mutation change alongside the measured
whole-engine improvement. Raster/error remains the largest sampled GPU stage.
That suggests it limits further gains from mutation changes. The samples do not establish
whole-loop stage shares, occupancy or hardware utilization. A scheduling or
compiler explanation beyond the source mechanism remains unverified.

## Exactness and independent audit

`mutation-group-confirmation/independent-audit.json` records `valid=true` and
`error_count=0`. It covers all 60 normal trials and 12 separate diagnostic
replays. Submitted and completed normal work is 62,914,560 evaluations;
diagnostic work is 12,582,912 evaluations. Six reference qualification runs
contribute 6,291,456 evaluations. Diagnostics have no throughput score.

The audit confirms exact initial and final replay of all seven canonical fields,
complete parameters and source identity, device identity, work counters,
alternating order and warmups. The fields are packed parents, packed offspring,
parent errors, parent framebuffers, offspring errors, chain update flags and
packed fitness. All 144 independent raw framebuffer-byte RGB-L1 totals match
cached errors. Fitness remains finite and associated with the correct geometry.
All six historical initial/final trajectory anchors match exactly. Separate
actual-source overflow diagnostics match the normal trajectories. No process,
device, capacity, replay, work, fitness, overflow or numerical error became a
timing sample. `audit-group.py` and its completed output retain the audit record.

The candidate-specific direct proof is
`mutation-group-confirmation/numerical/output/result.json`. It records 368
completed fixtures, 368 candidate comparisons, 736 submitted and completed
dispatches, zero failed results and `passed=true`. This checker has no evolution
throughput score. Its fixtures cover every integer lambda 1 through 64,
single/multiple mutation, adaptive behavior, empty-parent crossover,
polygon-count boundaries through 1000, source-isolated retained histories and
unchanged unproduced slots. Inactive padding is reported separately.

## Source preservation and remaining limits

`mutation-split/manifest.json` and `proof.md` retain the source preservation and
ownership evidence. Review found all 16 original helpers byte-exact and the
original leader body byte-exact after removing only its serial copy. Each
readable parent-prefix slot has one loader and each copied child slot has one
writer. Both unconditional barriers precede early returns. The storage barrier
orders the cooperative copy before leader mutation. Added parent-load and
child-copy loops take at most 16 iterations per lane for G1. Helpers consume no
RNG and write no child header.

Bounds checks preserve unproduced slots. Logical child zero in each chain retains its
adaptive exception. Tournament selection still reads the complete 1024-parent
binding, including inactive donors. Empty-parent crossover retains the default
zeroed slot-zero fallback. No probability, precision, fitness objective or
polygon order changes. The long mutation entry point remains the documented
Power of Ten deviation because its RNG-consuming branch sequence is kept
byte-exact. Source preservation does not replace the direct runtime proof.

The source-coupled host width selects both dispatch sites and travels with the
exact-source pipeline cache entry. Logical offspring width is distinct from
physical lanes and from MutationParams. Marker-free historical sources retain
width 64. Malformed, duplicated or commented declarations fail closed.

The timing binaries and source inventory are frozen under `build-v8/`, with
actual executable paths, release configuration and hashes. The adopted source's
focused CPU logs now record 31 library, 12 lab and seven mutation-checker tests
passed, all with zero failures, in `g1-adoption-lib.log`,
`g1-adoption-lab.log` and `g1-adoption-mutation-checker.log`. These are separate
from the direct GPU proof. The subsequent release `build-v9/` also completed;
its resident lab SHA-256 is
`c6e06c42b518a560d35898472952ecb8e8f6624655f066df67f8dfe3d73b9aee`.
That build bundles G1 as the default and retains the tested binner and raster.

The recorded adapter is `Microsoft Direct3D12 (NVIDIA GeForce RTX 5090)`,
exposed through Vulkan by Dozen, Mesa 25.3.6. Both sides use the same device
metadata. Background isolation remains unverified. This result supports the
recorded resident loop and backend; it establishes no cross-backend speed,
hardware utilization or evolution-quality improvement.

[Report 12](12-mutation-group-screens.md) retains attempts 28 through 31.
Only G1 advanced by the highest passing preliminary point estimate. G4 and G16
passed their short filters but remain unselected. G32 failed the practical gate.
No runner-up promotion, sample extensions or exclusions occurred. The cached
source-preparation README and earlier `outcome.json` retain their historical
pre-adoption wording. The final decision follows the passed audit and KEEP
record, not those earlier status snapshots.

## Evidence freeze

Report evidence frozen at `2026-10-24T03:47:40.502186+00:00`. Production mutation, binner and raster
hashes match the identities above. The following SHA-256 hashes pin the decision
and its evidence without changing the cached artifacts.

| Artifact | SHA-256 |
| --- | --- |
| `mutation-group-confirmation/spec.json` | `f38282fd0c5f6967974849de13812b6a8475539050bf7ef23b9115de08e52eac` |
| `mutation-group-train-full/output/analysis.json` | `3bc3366dc44456f956c8315b73868935355c94e35fc8a42330aa0217003a9925` |
| `mutation-group-confirmation/reserved-analysis/analysis.json` | `94ff8b8c9c979572526449c2310cda8ba8ade821f8dc8d17814d2ec4ba213def` |
| `mutation-group-confirmation/independent-audit.json` | `3ca87393823fac931cfd43a881667f6eb40c4741265b259dd5e45860fa4113be` |
| `mutation-group-confirmation/adoption.json` | `ecc9f8a7c7638536168987eb9dcfed9da3eab76a8ebaab8f3b93645909249ae4` |
| `mutation-group-confirmation/numerical/output/result.json` | `e1e029edd651aa799d418bf785a7c60ef5239c01519a445017165ed59fb70ae7` |
