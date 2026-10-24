# Triangle edge microexperiments

Two independent candidates use the accepted AE1 raster, the accepted G1 mutation
and wide local-count binner. Evolution settings, images, logical tiles, precision,
RNG streams, polygon visitation and the integer error objective stay fixed.

Attempt 43 returns before computing the third triangle edge when the first two
edges already have incompatible signs under both inclusive winding tests. Covered
pixels retain the original three edge expressions and blend arithmetic. Attempt
44 combines the earlier compact edge-delta cache with AE1's physical 256-lane
execution. It recomputes the original pixel AABBs and stores the original six
edge differences, with a 56-byte record. Conservative shared storage is 15,440
bytes, below the unchanged 16,352-byte request. These candidates are independent.

The fixed short filter uses two alternating pairs per project, eight batches,
seed 7, two warmups and separate overflow replays. The frozen specification is
`.cache/evolution-overnight-20261024/raster-micros-02-spec.json`; source proofs,
hashes and diffs are in `next-audit-54/`. Build-v11 is an immutable fresh release
with the accepted defaults. A passing filter alone does not authorize adoption.

A separate unscored current-default mona worker completed 1,048,576 evaluations
and reproduced the original captured endpoint during host profiling. Its guarded
steady window contains 65 samples over about 1.62 seconds. Main-thread CPU time
was 0.45 seconds, about 28% of one core; 42 of 65 samples observed polling sleep.
The user-CPU profile's largest reported symbol is inside the proprietary GPU
driver. This single profile does not establish hardware utilization or a
recoverable CPU share. It supports testing small raster arithmetic changes before
adding a host scheduling mechanism. Full evidence is in
`current-host-profile-01/`, including perf data and the bounded process record.

Both candidates passed the fixed short filter with zero errors. Lazy third-edge
evaluation measured an aggregate ratio of 1.023547551403 with conditional 95%
interval [1.009117862546, 1.038183574849]. The independent compact-edge56
composition measured 1.021599428862 with interval [1.015376844491, 1.027860147407].
The intervals overlap; this screen does not establish which is generally faster.
The predeclared highest-point rule selects lazyEdge alone for further work.
Compact-edge56 receives no full-budget samples. No default changes follow yet.

All 24 timing trajectories and nine separate overflow replays completed with
exact canonical state, cached raw framebuffer L1, packed fitness, settings,
adapter and work gates passing. Normal work totals 3,145,728 evaluations;
diagnostics add 1,179,648 and reference qualification adds 393,216, all excluded
from scoring. Every slow pair remains included. Independent source review found
no material issue in either candidate. A source review and an eight-batch screen
do not replace candidate-specific numerical fixtures or the frozen five-pair
64-batch training and reserved confirmation. The selected package is frozen in
`raster-lazy-confirmation/spec.json`.

Per-case paired ratios below are medians with observed minimum and maximum.

| Candidate | Case | Median [min, max] |
| --- | --- | --- |
| edge56 | ff | 1.019837278 [1.016832267, 1.022842289] |
| edge56 | mona | 1.012159878 [0.996591683, 1.027728073] |
| edge56 | darwin | 1.033038768 [1.030385938, 1.035691598] |
| lazyEdge | ff | 1.004047547 [0.990711085, 1.017384010] |
| lazyEdge | mona | 1.047788068 [1.017085526, 1.078490610] |
| lazyEdge | darwin | 1.019815909 [1.018124206, 1.021507613] |

## Interrupted confirmation

The candidate-specific release numerical checker completed successfully across
all five supported workgroup shapes. Its binary SHA256 is
`9658a41df134b50111a40690140e023a1a68431bb03e9419da4e3b715c578dd7`.
The temporary build restored the accepted AE1 production source before runtime
confirmation. Source snapshots and restoration hashes are retained in the package.

The subsequent full confirmation was interrupted. The user reported a possible
computer restart; recovery found no worker or session. The incremental summary contains
ten of thirty expected timing trials and two diagnostics, plus two completed
reference qualifications. These records retain 14,680,064 completed evaluations
excluding warmup and numerical-checker work. The aggregate package never
completed, reserved confirmation never started, and no full-budget timing score
is claimed. The candidate remains unadopted. The short result and completed
numerical proof remain historical evidence, not permission to extend or promote
the partial series. `raster-lazy-confirmation/interrupted.json` records recovery.

The proposed host readback observer also stopped during cache-only preparation.
It was not compiled or run and changed no root engine code. Its unfinished cache
files remain available; they support no timing conclusion.
