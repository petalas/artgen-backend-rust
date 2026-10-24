# Reject the scalar geometry-cache layout

Reject this candidate. Completed-engine throughput is 0.910216 times reference,
with a conditional 95% bootstrap interval of [0.907813, 0.912492]. That is about
9.0% lower throughput in this fixed-work screen. Every measured pair is slower.
The full matrix passes its recorded correctness and work checks with zero
errors. No reserved run, all-workgroup checker build, retry or extra sample
follows. Accepted AE1 and all production settings remain unchanged.

| Case | Paired ratio median | Minimum to maximum | 95% interval for paired geometric mean |
| --- | ---: | ---: | ---: |
| ff | 0.909554 | 0.900417 to 0.917199 | 0.905176 to 0.915263 |
| mona | 0.911944 | 0.911175 to 0.918561 | 0.911550 to 0.916623 |
| darwin | 0.907238 | 0.900942 to 0.912520 | 0.902638 to 0.910335 |

Each case has five alternating pairs, 64 measured batches per arm, seed 7 and
eight fixed warmups. The aggregate equally weights the three cases and uses
10,000 stratified bootstrap draws with seed 20261024. All fifteen pairs remain
included. The frozen aggregate threshold was strictly above 1.005, its lower
bound strictly above 1, each case median at least 0.995 and lower bound at least
0.99. This result fails all of those guards. Its interval is conditional on
same-seed timing repeats, not independent search seeds or other hardware.

The [source patch](attempts/16-scalar-geometry-cache.patch) replaces three vec2
vertex members with six f32 members. The existing loader projects its original
vertex values into those scalars. The by-value pixel helper reconstructs each
vec2 eagerly before its existing AABB check. Packed color, per-covered-pixel
unpacking, blend arithmetic, ordering, barriers and integer reductions remain.
The exact inverse recovers AE1. The patch reconstructs the candidate with zero
fuzz. Candidate SHA256 is
`7f551b7f5455430b6d2cb38f879fefb05096f33689a71d37528e87122efdb2d2`.

Source layout inference changes alignment from 8 to 4 and stride from 48 to 44.
The 256-record cache shrinks from 12,288 to 11,264 bytes. Including unchanged
reductions and conservative global padding gives 12,368 bytes, below the existing
16,352-byte request. These are inferred layout bounds. Actual allocation, bank
conflicts and occupancy were not measured. The source proof alone cannot establish
compiled floating-point equality; runtime checks cover the recorded training
endpoints rather than every possible input or workgroup.

The [frozen protocol](profiles/scalar-geometry-cache-spec.json) uses the immutable
prepared release executable, Stages timestamps and explicit finalization-overlap
false in both arms. The earlier A/A64 supplies calibration and GPU identities
only. Its rates are excluded, and it is not fresh noise evidence. The
[one-shot runner](attempts/17-scalar-cache-training-runner.patch) reuses the pinned
26-test validator. Syntax, pure gate controls, scoped Ruff and exact archived
patch reconstruction pass. This screen requires no Rust build.

## Measured path and limits

The worker completed 30 timing trials, six unscored diagnostics and three
unscored reference qualifications. Timing contains 31,457,280 completed
evaluations. Diagnostics add 6,291,456 and qualifications 3,145,728. Thirty-nine
successful preparations account for at least 5,111,808 warmup evaluations,
bringing the source-accounted logical lower bound to 46,006,272. Constructor,
full-fitness and reset work stay separate. The native process took 271.235
seconds; the controller, including validation, took 273.666 seconds within the
1,800-second budget. The process exited zero; rejection is the controller's
statistical disposition, not an execution error.

Completed-engine timing includes final drain. Device/source setup, preparation
and oracle exports remain outside throughput timing. The sampled raster averages
increase from 0.433 to 0.511 ms for ff, 0.561 to 0.656 ms for mona, and 0.555 to
0.650 ms for darwin. These are medians of five trial averages, each sampling only
64 final iterations. They are not whole-run stage shares. Other sampled stages
remain close. This localizes an observed regression to raster samples without
proving the hardware limiter, bank conflicts, occupancy or compiled instruction
cause. Smaller source allocation does not establish less executed work.

The machine had load average 0.26 on 16 logical CPUs before launch, with no other
known benchmark or build. WSL host/GPU background isolation is unverified. Both
arms use the same immutable release, Vulkan/Dozen backend and fixed configuration
in alternating order.

The independent offline audit passes all 446 checks. Eighteen initial/final
snapshots reproduce their seven canonical hashes. All 72 parent framebuffers
match cached integer RGB-byte-L1 exactly; independent f32 fitness has maximum
difference zero. The existing 2e-5 bound remains unchanged. All 78 runtime source
files, three ordinary binaries, two helpers and 86 private build source files
retain their hashes. Earlier audit-bookkeeping exceptions remain archived and
are separate from native workload errors. No scalar rerender occurred.

The [machine record](profiles/scalar-geometry-cache-result.json) and
`.cache/evolution-layout-20261024/` retain raw rows, clocks, saved states, source
proof, setup distributions, accounting and the independent offline audit.
Training covered the actual 32 by 16 raster workgroup. The candidate never
advanced to the planned five-workgroup fixture or reserved-data gates and has
no broader runtime qualification.

Prove It Works requires canonical replay and raw framebuffer RGB-byte-L1 checks.
Explain the Number separates the observed raster slowdown from an unmeasured
hardware cause. Attack the Premise keeps the rejected RGBA cohort closed and
preserves the original packed-color boundary. Power of Ten leaves all loop and
initialization bounds intact.
