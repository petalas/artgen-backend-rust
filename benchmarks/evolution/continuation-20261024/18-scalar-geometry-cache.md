# Test a scalar geometry-cache layout

Status is SOURCE_FROZEN_UNQUALIFIED. No candidate workload has run. This is one
independent raster layout experiment. Production retains AE1 and every evolution
parameter remains fixed. The prior normalized-RGBA numerical failure and deferred
load throughput rejection stay closed.

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
conflicts, occupancy and throughput benefits have not been measured. Compiled
floating-point equality remains unproven despite unchanged arithmetic.

The [frozen protocol](profiles/scalar-geometry-cache-spec.json) uses the immutable
prepared release executable, Stages timestamps, eight warmups, seed 7, 64 measured
batches and five alternating pairs for each of ff, mona and darwin. Both arms set
finalization overlap false. The earlier A/A64 result supplies calibration and GPU
identities only. It is not fresh noise evidence and none of its rates enters A/B.
The [one-shot runner](attempts/17-scalar-cache-training-runner.patch) reuses the
hash-pinned 26-test validator. Its syntax, pure statistical controls and scoped
Ruff pass. The 78 accepted source files, three ordinary binaries and two helpers
match the preservation inventory. No Rust rebuild is needed for this screen.

The predeclared aggregate ratio must exceed 1.005 with a 95% lower bound above
1. Each case median must be at least 0.995 and its lower bound at least 0.99.
Every pair stays included. Invalid work closes performance INCONCLUSIVE; a valid
training failure closes REJECT. No retry, variant, extra sample or exclusion is
allowed. A training pass requires actual candidate-specific numerical fixtures
across all five workgroups before the fixed reserved-drawing confirmation. The
single package has a 1,800-second deadline and at most four serial workers.

Existing whole-batch GPU spans and host-phase records locate raster work as a
plausible investigation target. They do not measure this cache's limiter.
Completed-engine timing includes final drain; setup and oracle export remain
outside. A tiny result cannot establish a bank-conflict or occupancy mechanism.

Prove It Works requires actual canonical replay and raw framebuffer RGB-byte-L1
checks. Explain the Number keeps layout savings separate from speed. Attack the
Premise preserves the original packed-color materialization boundary after the
RGBA failure. Power of Ten keeps all existing loop bounds and initialization
ownership unchanged. The source proof and local evidence are retained under
`.cache/evolution-layout-20261024/`.
