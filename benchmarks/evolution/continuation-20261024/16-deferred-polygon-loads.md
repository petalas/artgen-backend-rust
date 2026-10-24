# Reject deferred shared-polygon payload loads

REJECT at the frozen practical gate. Aggregate throughput ratio is
1.0037759369034747, conditional 95% interval
[1.00073828225412,1.006734175066504]. The estimate is about +0.38%, below the
required strict +0.5%. The lower bound and all case guards pass; they do not
override the failed point threshold. No reserved process, dedicated numerical
checker build or production shader change follows.

| Case | Paired median ratio | Paired minimum to maximum | 95% interval for geometric mean ratio |
| --- | ---: | ---: | ---: |
| ff | 1.001960946 | 0.990199731 to 1.014560766 | 0.996051555 to 1.011285639 |
| mona | 1.003368382 | 1.000004727 to 1.007187116 | 1.001231732 to 1.005290846 |
| darwin | 1.003929507 | 0.999592538 to 1.011824315 | 1.001014512 to 1.008546260 |

Five alternating pairs per image use 64 measured batches, eight fixed preparation
batches, seed 7, ordinary Stages timestamps and all original engine parameters.
The interval estimates equal-case geometric mean ratios, not the median column.
The paired-log bootstrap uses 10,000 draws and seed 20261024. Every pair remains
included. The [frozen specification](profiles/deferred-polygon-loads-spec.json)
records all gates and stop rules before candidate measurement.

The fresh [control](15-engine-kernel-control.md) passed its unchanged bands. Its
aggregate ratio was 0.995709390, an offset of similar magnitude to this small
candidate gap. No control correction, subtraction or old timing sample is used.
Background isolation and a causal hardware limiter remain unverified. This is
a conditional observation, not a universal intrinsic gain or loss. The
alternative full-color-cache idea receives no samples in this experiment.

## The source change

The candidate changes a by-value polygon argument to its existing slot index.
It fetches AABB fields first, vertices after the unchanged AABB rejection, and
packed color only inside the unchanged coverage branch. Every original float
expression, edge evaluation, blend operand order, polygon order and integer error
operation remains. Shared allocation, immutable publication barrier, logical
tiles, dispatch, settings and RNG stay unchanged. The
[exact source patch](attempts/11-deferred-polygon-loads.patch) is independent
of previously rejected arithmetic experiments. Candidate SHA256 is
`c2c484608af8faa6f3e22ce1ab4d735bf4f936498c9ff9c253f315476f4b435b`.

The hypothesis was that the backend might materialize the entire by-value record
before rejection. Source reasoning predicts no hardware result. Compiler load
sinking may already avoid that traffic, and repeated indexing or checks may add
work. The unchanged allocation and this small end-to-end estimate establish
neither actual memory traffic nor the cause of the result.

## Work and verification

Both native workers complete with zero process, device, timing-row or numerical
errors. Together they retain 60 normal rows, twelve diagnostics and six reference
qualifications. Exact seven-field initial/final endpoint associations, strict
typed parameters, G1/TRUE mutation, overlap false, Stages samples, work counters
and actual arm/source hashes pass. The
[validation adapter](attempts/12-kernel-validation-adapter.patch) passes 26 CPU
tests, including real completed control records and source/path/rate/work
rejections. Unscored qualification rates remain null and normal rates remain
finite completed-work rates. Scoped Ruff and pure runner gate controls pass.

Independent audit passes 916 scoped checks, 36 saved snapshots and 144 parent
RGB-byte-L1/f32 fitness comparisons against frozen source-image bytes. Maximum
fitness difference is zero under the unchanged 2e-5 policy. Normal rows retain
canonical hashes associated with those saved endpoints rather than separate
full snapshots. Existing scalar uncertainty records are checked without a scalar
rerender. Execution uses the ordinary 32 by 16 tile; broader all-five-workgroup
and AABB/edge/partial-image fixture proof was not built for this rejected source.

Total work is 62,914,560 normal, 12,582,912 diagnostic and 6,291,456 qualification
evaluations, plus 10,223,616 source-accounted warmup evaluations. The combined
logical lower bound is 92,012,544. Full-fitness/reset setup is separate. Both jobs
reuse the existing release binary with no Rust build. Control and candidate
controllers finish in 261.087 and 265.026 seconds. Setup is outside measured
throughput; drain is already inside it. Last-iteration stage samples are not
whole-run shares.

The [machine result](profiles/deferred-polygon-loads-result.json) retains all
ratios, setup clocks, counters and audit metadata. Raw files, immutable clocks,
source selection and proofs are under `.cache/evolution-next-20261024/`.
All 376 protected hashes remain unchanged, including 78 accepted root Rust/WGSL
files, normal executables, helpers and raw evidence. The audit explicitly allows
parent result-document edits rather than claiming a clean worktree during
recording. Its initial cleanliness assumptions are retained as observations;
no numerical or scoring result was changed by that bookkeeping correction.
Production remains AE1 and every earlier disposition stays unchanged.

Explain the Number separates the small conditional estimate from source traffic
reasoning and the control offset. Prove It Works requires actual compiled replay
and saved byte-L1 checks. Boundary Discipline binds evidence to exact source,
manifest and work identities.
