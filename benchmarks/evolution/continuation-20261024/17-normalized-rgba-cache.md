# Reject normalized RGBA caching for numerical inconsistency

Reject the candidate for correctness. The campaign closes INCONCLUSIVE for
performance: numerical validation stopped before any of the 30 planned timing
rows. There is no throughput ratio or confidence interval. The worker returned
one after 41.153 seconds; the controller retained 41.391 seconds. No retry, extra
sample, source variant, reserved process or checker build follows. Production
remains accepted AE1 with all evolution parameters unchanged.

The first ff reference qualification and reference diagnostic pass. The candidate
diagnostic completes 64 batches but records nine errors: six final canonical
fields differ and three cached error totals disagree with its own framebuffer.
Initial canonical fields match exactly. Only final `offspring_errors` remains
equal; parents, offspring, total errors, framebuffers, update flags and packed
fitness differ.

| Candidate chain | Cached total | Actual framebuffer RGB L1 | Cached minus actual |
| --- | ---: | ---: | ---: |
| 0 | 2,717,604 | 2,717,585 | +19 |
| 1 | 2,717,851 | 2,717,842 | +9 |
| 2 | 2,715,810 | 2,715,788 | +22 |
| 3 | 2,716,615 | 2,716,615 | 0 |

Every reference delta is zero. These integer equalities are independent of the
existing 2e-5 CPU/GPU fitness tolerance. Native expected fitness derives from the
cached total and stays within that tolerance. Independently deriving fitness
from actual framebuffer L1 reveals a chain-2 difference of
0.00002288818359375, above 0.00002. That is an additional offline finding, not a
tenth native error. No acceptance threshold was changed.

## Source change and consistency boundary

The [exact patch](attempts/14-normalized-rgba-cache.patch) starts directly from
AE1. It replaces packed color and padding with four scalar f32 fields. The
cooperative loader invokes the unchanged `unpack_color` builtin, stores normalized
RGBA, and the original by-value pixel helper reconstructs its color vector. Every
original RGB multiplication by 255, inverse-alpha expression and blend operand
order remains. Geometry, AABB/edge math, ownership, barriers, integer error
operations, dispatch and RNG are unchanged. It does not compose either rejected
alpha-only or deferred-load work.

The stride increases from 48 to 56 bytes. A 256-record cache uses 14,336 bytes;
unchanged reductions and conservatively padded globals bring the bound to
15,440, below the existing 16,352-byte request. Cache allocation grows by 2,048
bytes. The source proof reverses exactly to AE1. Candidate SHA256 is
`892c0019bbfa3f4a2e27388427c7fc7635308591f83b8a1f46dbff8a6856d6c9`.

The override changes the error-scoring pipeline. Initialization and accepted
framebuffer-update kernels retain their original per-pixel color decoding.
Selection caches `parent_total - old_dirty + new_dirty`, then the update kernel
renders accepted geometry without recomputing that total. Therefore the scored
and committed quantized pixels must agree. The endpoints establish a broken
consistency contract. They do not identify the first faulty pixel/iteration,
rounding, reassociation, driver behavior or compiled instruction cause. Identical
helper text alone cannot prove identical effective output across these paths.

## Method and retained work

The [frozen protocol](profiles/normalized-rgba-cache-spec.json) declared five
alternating pairs per image, 64 measured batches, seed 7, eight fixed warmups and
Stages timestamps, using the existing immutable release binary. The previous
64-batch A/A result was calibration only. None of its samples enters this test,
and no new A/A or A/B timing result is claimed. The
[runner](attempts/15-rgba-cache-training-runner.patch) reuses the hash-pinned
26-test validator and passes syntax, pure gate controls and scoped Ruff. No Rust
build or production-code edit was needed.

Retained work is one reference qualification and two diagnostics, including the
incorrect candidate: 192 batches, 12,288 iterations and 3,145,728 evaluations.
The unchanged controller conservatively counts two correctness-true preparations,
yielding a 3,407,872 logical lower bound. Source order proves that all three
preparations returned successfully before their initial snapshot and completed
batches. Its separate refined bound includes 393,216 warmup evaluations and
3,538,944 logical evaluations. Constructor/full-fitness/reset work stays separate.
No failed-candidate work is discarded or converted into a scoring rate.

The independent measurement audit passes with all failures retained: 238 of 250
recorded predicates pass, twelve reproduce expected process/candidate failures,
and none is unexpected. Six snapshots and 24 parent byte-L1/f32 checks confirm
the mismatch. All 232 protected inputs remain unchanged, including accepted
Rust/WGSL, normal binaries, helpers, private sources and raw evidence. No scalar
rerender, additional GPU job or compiled-code inspection was performed.

The [machine record](profiles/normalized-rgba-cache-result.json) and
`.cache/evolution-rgba-20261024/` retain source proofs, exact patch reconstruction,
process logs, clocks, snapshots, accounting and both offline reports. The earlier
experiments retain their original dispositions. The
[project learning](../../../docs/LEARNINGS.md) records the scoring/framebuffer
contract so future performance work keeps the integer checks.

Prove It Works required actual framebuffer byte comparisons. Explain the Number
forbids performance claims from this invalid candidate. Boundary Discipline
identifies the scored-pixel versus committed-pixel contract.
