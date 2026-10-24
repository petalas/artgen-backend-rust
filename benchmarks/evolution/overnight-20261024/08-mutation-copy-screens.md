# Reject cooperative mutation copying after full confirmation

Mutation still uses its original WG64 shader. None of these experimental sources
was adopted. All complete evolution parameters, logical offspring IDs, RNG draw
order, tournament donor range and default workgroup zeroing stayed fixed.

Six candidates load the parent with wider physical workgroups, cooperatively copy
its active prefix in two/four/eight lanes per offspring, synchronize storage, then
run the original mutation body in compact logical leaders. The additional prefix
copy for crossover can change inactive padding. Source dataflow review and the
finite direct checker cover its lack of observable tail reads. The separate
header candidate only removes stores overwritten by dirty-bbox outputs.

The short filter used two alternating pairs per project, eight batches, seed 7,
two matching warmups and the kept binner/raster. All 84 normal trials and 24
separate diagnostic replays passed. Normal work totaled 11,010,048 evaluations;
diagnostics contributed no rates. No pair was excluded.

| Attempt | Candidate | Preliminary aggregate ratio | Filter |
| --- | --- | ---: | --- |
| 18 | copy128x2 | 1.003681 | Reject filter |
| 19 | copy256x2 | 1.001722 | Reject filter |
| 20 | copy256x4 | 1.009051 | Advance |
| 21 | copy512x2 | 1.000772 | Reject filter |
| 22 | copy512x4 | 1.003511 | Reject filter |
| 23 | copy512x8 | 1.008768 | Advance |
| 24 | elideHeaders | 0.997761 | Reject filter |

Only the highest aggregate point estimate, `copy256x4`, advanced to the frozen
full package. The 512x8 alternative remains an unadopted short-filter observation;
it received no additional samples after selection. The candidate passed the
368-fixture physical mutation checker before confirmation.

Full confirmation used five alternating pairs per project at 64 batches, two
warmups, seed 7 on training and reserved seeds 11/13/17. Both sides use the kept
512-lane binner and decoded-AABB raster cache. The training aggregate ratio was
**1.005154**, conditional 95% interval **[1.001819, 1.008698]**, passing its gates.
Reserved ratio was **1.004616**, interval **[1.000671, 1.008435]**. The reserved
point estimate fails the unchanged practical threshold of at least 1.005. The
result is **REJECT**. There is no sample extension, tolerance change or promotion
of the runner-up.

All 60 full normal workers and twelve overflow replays passed numerical, exact
state, fitness, work and device checks. Their work totals were 62,914,560 and
12,582,912 evaluations respectively; six reference qualification workers added
6,291,456. The independent endpoint/byte-L1 audit is retained with the raw evidence.
The confirmation supervisor's generic `invalid_or_rejected` outcome reports its
failed statistical assertion; the workers and analysis records themselves have
zero errors. This is a failed performance gate, not a numerical failure.

Evidence is under `.cache/evolution-overnight-20261024/` in `mutation-copy-screen-01`,
`mutation-train-full`, `mutation-reserved-{ff,mona,darwin}` and
`mutation-confirmation/`. The frozen specification predates full execution.
Sources, exact diffs and ownership/tail proofs are in `mutation-copy/`; the
immutable lab binary/source inventory is `build-v4/`. Patches 18–24 preserve all
seven independent attempts. Background isolation remains unverified, and timing
intervals are conditional on the frozen trajectories. No evolution-quality or
throughput gain is claimed from a rejected source.
