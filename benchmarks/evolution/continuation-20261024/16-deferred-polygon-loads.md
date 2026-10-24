# Defer polygon payload reads until needed

This new source-only raster experiment changes a by-value polygon helper argument
to its existing slot index. It reads the AABB first, vertices only after the
unchanged AABB rejection, and packed color only after the unchanged coverage
test. Every original float expression, edge evaluation, blend operand order,
polygon order and integer error operation remains. Shared memory is unchanged
and immutable after the existing loader barrier. Dispatch, settings and RNG
remain fixed. The [exact candidate patch](attempts/11-deferred-polygon-loads.patch)
applies independently to accepted AE1.

The source hypothesis is that a backend might otherwise materialize the full
48-byte by-value record before rejection. Compiler load sinking may already
avoid it; repeated indexed loads or bounds checks may cost more. No measured
memory traffic, hardware limiter or gain is claimed from source operation counts.
The source-selection report and inverse/hash proof are in
`.cache/evolution-next-20261024/selection-107/`. Candidate SHA256 is
`c2c484608af8faa6f3e22ce1ab4d735bf4f936498c9ff9c253f315476f4b435b`.

The fresh [64-batch control](15-engine-kernel-control.md) passes its frozen bands.
The [comparison specification](profiles/deferred-polygon-loads-spec.json) fixes
one candidate, five alternating pairs per image, 64 measured batches, eight
preparation batches, seed 7, ordinary Stages timestamps, separate diagnostics and
all original engine parameters. The immutable prepared lab needs no Rust rebuild.
Finalization overlap stays explicitly false in both arms.

The complete training matrix must have aggregate ratio strictly above 1.005,
aggregate lower 95% bound strictly above 1, every paired case median at least
0.995 and every case lower bound at least 0.99. The existing 10,000-draw paired
log bootstrap uses seed 20261024 and equal case weights. Valid failure is REJECT;
wrong work, errors or incomplete evidence are INCONCLUSIVE. Every pair is kept.
No retries, added samples, alternate candidates or warmup variants follow a stop.

Only a training pass can receive a candidate-specific five-workgroup numerical
proof. Only that proof can permit the three fixed reserved processes, using
distinct mature drawings and seeds 11/13/17. Their complete combined matrix must
clear the same gate. At most four GPU workers share one fresh 1,800-second clock,
with child limits capped at 600 seconds and remaining time minus 30 seconds.
Proof/build work stays outside measured timing. No formal policy1 KEEP follows.

The [validation adapter](attempts/12-kernel-validation-adapter.patch) binds every
arm to its shader bytes and each phase to its manifest. All 26 CPU tests pass,
including the actual completed A/A64 records and the unscored qualification
contract. Scoring, numerical, scalar, strict typed-parameter, work, device and
seven-field replay checks are retained. Source/path swaps are rejected. The
[one-shot training runner](attempts/13-deferred-training-runner.patch) passes
syntax, scoped Ruff and pure gate controls. Runtime candidate proof is pending.

Prove It Works requires compiled numerical equality. Explain the Number keeps
source-byte reasoning separate from measured whole-engine throughput. Boundary
Discipline binds JSON evidence to the actual source and workload.
