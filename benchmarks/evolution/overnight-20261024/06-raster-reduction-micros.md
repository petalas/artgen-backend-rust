# Reject raster reduction microoptimizations

Both candidates build independently on the kept decoded-AABB cache, using the
same committed binner and complete fixed evolution parameters. Two alternating
pairs per project ran eight batches with seed 7, two warmups and matching stage
instrumentation. All 24 normal trials and nine timing-excluded diagnostic replays
passed exact seven-field state replay, byte-L1, fitness, completed work and device
checks. All timed pairs remain included.

| Attempt | Change | Preliminary aggregate ratio | Conditional 95% interval | Decision |
| --- | --- | ---: | --- | --- |
| 15 | Pack new/old errors into one subgroup sum when subgroup size is at most 64 | 0.998946 | [0.996568, 1.001330] | Reject filter |
| 16 | Compute exact byte RGB L1 with integer arithmetic | 0.979164 | [0.961942, 0.996695] | Reject filter |

The first change bounds each sum by 64 times 765, or 48,960, preventing carry
between packed 16-bit fields. Larger subgroups retain separate reductions. The
second keeps reference rounding, blending and coverage unchanged, replacing the
final nonnegative floor/cast and byte differences. Production validation and the
actual GPU trajectory checks accepted both sources, but neither cleared the
unchanged 1.005 preliminary throughput gate. Their composition was not run.
Neither source remains installed in production.

Normal completed work was 3,145,728 evaluations. Diagnostic rates are null and
excluded. Full per-project rates, ranges, stage intervals and overflow evidence
are retained in `.cache/evolution-overnight-20261024/raster-reduction-screen-03b/`.
The archived lab is `build-v3/`; source proofs and complete hashes are in
`raster-reduction/`. Patches 15 and 16 apply to the kept decoded cache.

The first launch used relative shader paths inside the isolated worker directory
and failed before GPU setup or work. That evidence remains in
`raster-reduction-screen-03/`. The throwaway supervisor now resolves input paths
before switching directory; the replacement ran the unchanged frozen workload.
Background isolation remains unverified, and the two-pair intervals are filter
observations rather than adoption evidence. No full confirmation was requested
for these losing filters.
