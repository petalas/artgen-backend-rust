# Test exact mutation bbox reuse

Status is SOURCE_FROZEN_UNQUALIFIED. This is one independent mutation experiment;
no candidate workload has run. Production G1, bin598 and raster AE1 remain
unchanged, with every evolution parameter fixed. Earlier experiments stay closed.

The [source patch](attempts/18-mutation-bbox-reuse.patch) captures the three original
packed vertex words in the final per-polygon path of `single_mutate_offspring`.
After the original mutation, repack and polygon store, it compares the resulting
words. Exact equality returns the existing bbox. A changed word follows the
original new-bbox calculation and union. Color, vertices, polygon writes and RNG
calls keep their original order. Empty drawings, earlier returns, crossover and
multi-mutation remain unchanged.

This does not assume that vertex unpack/repack round trips exactly. Equal packed
words give the bbox helper identical inputs; its integer min/max union is
idempotent. The source inverse restores every G1 byte, and the archived patch
reconstructs the candidate without fuzz. Candidate SHA256 is
`f3dd47b2dc6c82e12c925c4d4cb41b2f5aeda28a016dc246796a61d9d17988dc`.
No shared allocation, barrier, dispatch or parameter changes. Capturing three
private words, testing equality and branching may cost more than the removed
calculation. Guard frequency, compiled instruction savings and speed are unknown.

The [frozen protocol](profiles/mutation-bbox-reuse-spec.json) uses the existing
prepared release binary and hot mutation installer. Both sources retain G1,
WG64 and automatic workgroup initialization TRUE. Both arms explicitly disable
finalization overlap. There are five alternating pairs for each of ff, mona and
darwin, seed 7, 64 measured batches and eight preparation batches with Stages
timestamps. Previous A/A64 provides calibration and GPU identities only, not
fresh noise evidence or samples for this comparison.

The [training runner](attempts/20-bbox-training-runner.patch) keeps the existing
complete-matrix gate. Aggregate ratio must exceed 1.005 and its 95% lower bound
must exceed 1. Each case median must be at least 0.995 and lower bound at least
0.99. Statistics retain all pairs, equally weight cases and use 10,000 stratified
bootstrap draws with seed 20261024. Invalid work closes INCONCLUSIVE. A valid
training failure closes REJECT. No retry, additional variant, sample or exclusion.
A training pass requires the actual 368-fixture mutation checker with this source,
G1 and initialization TRUE, then the fixed reserved-drawing confirmation. This
single package has a 1,800-second deadline and at most four serial workers.

The import-only [stage contract](attempts/19-explicit-shader-stage-contract.patch)
now checks the declared bin, raster or mutation stage. Mutation-source metadata
must explicitly retain G1 and TRUE initialization; raster/bin sources retain
null mutation metadata. No rates, parameters, quality, canonical replay, work,
occupancy, clock or statistical checks change. Thirty CPU tests pass, including
the original 26 and four [stage regressions](attempts/21-shader-stage-contract-tests.patch).
The converted wire fixtures are CPU tests, not native mutation qualification.
The first discovery invocation found no hyphenated tests and is retained; the
explicit loader ran all 30 successfully. Scoped Ruff passes without suppressions.
The cohesive phase validator is 62 physical lines after stage propagation, with
its existing bounded evidence association preserved.

Mutation takes less sampled GPU time than raster. A faster bbox helper therefore
has a limited whole-engine opportunity. The sampled intervals do not establish
its execution frequency or whole-run share. Completed-engine timing includes
normal readback and final drain, while setup and oracle exports remain outside.

Prove It Works requires exact replay, actual framebuffer RGB-byte-L1 and the
existing fitness policy before a performance claim. Attack the Premise moved
this experiment away from the failed cache-layout assumptions. Explain the
Number keeps source-level redundancy separate from measured savings. Source,
proofs and bounded CPU evidence are in `.cache/evolution-mechanism-20261024/`.
