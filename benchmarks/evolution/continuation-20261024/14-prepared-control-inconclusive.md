# Prepared control stops before candidate qualification

The [frozen campaign](13-prepared-qualification-protocol.md) is closed
**INCONCLUSIVE**. One A/A process completed all 30 timing rows, six diagnostic
rows and three reference qualifications. Both arms had finalization overlap
explicitly disabled. No training or reserved-seed A/B process ran. There were no
retries, excluded rows, added samples or changed warmup policies. The earlier
two-warmup experiment remains REJECT, and production remains unchanged.

The native process returned zero with zero reported errors. The supervisor then
rejected the first reference qualification's null `evals_per_second`. The native
qualification path calls `measure` and `complete_record` directly, rather than
the ordinary trial runner that assigns a scoring rate. It retains the default
null rate even with `diagnostic:false`. The controller incorrectly passed that
unscored record to the ordinary scored-result validator. Synthetic controller
tests and source review missed this producer/consumer mismatch.

The original clock, result, controller and raw artifacts remain immutable.
Offline checks do not reopen that campaign or manufacture qualification scores.

## Retained control measurements

The complete A/A matrix also misses the predeclared stability gate when examined
offline using the unchanged 10,000-draw paired bootstrap and seed 20261024.
Its aggregate ratio is 0.9989241108482665 with 95% interval
[0.9948900023527549, 1.0032328981026781]. The Darwin paired median is below the
required 0.99 boundary. This result supplies no candidate comparison.

| Case | Paired median ratio | Paired minimum to maximum | 95% interval for geometric mean ratio |
| --- | ---: | ---: | ---: |
| ff | 0.995356721 | 0.994268646 to 1.013987839 | 0.994567506 to 1.008429189 |
| mona | 1.004875542 | 0.994498352 to 1.013110338 | 0.998320352 to 1.009488863 |
| darwin | 0.987320987 | 0.979814758 to 1.010246560 | 0.984095368 to 1.001202425 |

Ratios are control B throughput divided by control A throughput. The median
column and bootstrap interval estimate different statistics. Five alternating
pairs per image use one repeated seed; these are timing repetitions, not
independent search trajectories. No background isolation or cause for the
control variation is established.

## Setup and measured work

| Case | Shared constructor, seconds | Shared precompile, seconds | Normal preparation median A / B, seconds |
| --- | ---: | ---: | ---: |
| ff | 15.049543 | 0.787196 | 0.790103 / 0.799886 |
| mona | 10.464680 | 0.784356 | 0.891798 / 0.888391 |
| darwin | 10.763074 | 0.723322 | 0.809384 / 0.806791 |

Constructor and precompile occur once per case and are not charged twice to the
arms. Preparation includes seeded setup, full fitness, eight warmups, drain and
reset. It is outside measured throughput. These brackets do not measure whole
application startup or isolate lazy driver compilation. First, median, minimum
and maximum values, source-install clocks and separate diagnostic history are
retained in the [machine result](profiles/prepared-qualification-result.json).
Drain is already included in the scored elapsed clock.

Retained work is 3,932,160 normal, 786,432 diagnostic and 393,216 qualification
evaluations. Thirty-nine successful eight-batch preparations account for at
least another 5,111,808, totaling 10,223,616 logical evaluations. Full-fitness
and reset GPU setup is separate. This is completed work, not valid candidate
performance evidence. The controller took 125.065 seconds and the child
124.734 seconds, within the frozen bounds.

## Evidence and disposition

Raw output, process evidence, immutable clock/result and offline summaries are
under `.cache/evolution-qualification-prepared-20261024/campaign/`. The
[qualification alias patch](attempts/07-prepared-qualification.patch) and
[original controller patch](attempts/08-prepared-campaign-controller.patch)
retain the exact attempted implementation, including the validation defect.
The machine result records source and executable identities.

Independent audit reproduced the statistics and all completed-work counts. All
430 predicates passed, including expected validator rejections. Eighteen saved
snapshots and 72 parent RGB-byte-L1/f32 fitness checks agree with the pinned
source-image bytes. Maximum fitness difference is 7.62939453125e-6, below 2e-5.
Every normal row associates with those checked canonical endpoints; normal rows
do not each retain a separate full snapshot. No scalar rerender was performed.
The audit independently confirmed the producer/validator defect. Its complete
checks and 231 protected-file hashes remain in `result-audit-105/`.

The [CPU-only correction](attempts/09-qualification-contract-correction.patch)
adds an explicit unscored qualification guard and actual-record regressions.
All nine unchanged and ten new CPU tests pass, with Ruff clean. It requires
null qualification rates and preserves protocol, typed parameters, completed
work, elapsed/drain, quality/scalar/fitness, GPU and replay checks. Normal timing
rows still require finite correct rates. No raw records, producer or old helpers
were changed. The corrected whole-phase validator succeeds offline and the
unchanged A/A gate still fails. The correction preserves 321 pre-existing files.
It retains the consumed clock path and cannot reopen or resume this campaign.
Neither the correction nor offline analysis authorizes more timing here.
All 78 accepted Rust/WGSL files, three normal executables and two comparison
helpers still match the pre-campaign inventory. No engine mechanism was adopted.
A fresh ordinary API64 check remains required before any later adoption.

Explain the Number keeps the failed control separate from candidate throughput
and setup clocks. Prove It Works requires actual saved-state and byte-L1 checks.
Boundary Discipline requires an explicit unscored qualification validator.
