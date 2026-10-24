# Fresh prepared-throughput qualification

This campaign tests repeatability and conditional prepared throughput after the
single [eight-warmup observation](12-full-path-preparation.md). It uses fresh
outputs, a new clock and no historical timing samples. The original two-warmup
overlap experiment remains REJECT. This protocol cannot produce formal policy1
KEEP, an application-startup claim or a production default change.

The [frozen specification](profiles/prepared-qualification-spec.json) fixes the
original ff, mona and darwin drawings, reference bytes, all mutation parameters,
G1 mutation with workgroup zeroing enabled, and identical AE1 raster source in
both arms. The only compared engine mechanism is the archived default-off
finalization-overlap flag. Reference is explicitly false. Candidate is false
for A/A and true for A/B. All arms receive eight warmup batches followed by
seeded reset. Ordinary GPU evolution and the untouched measured loop both use
Stages timestamps.

## Finite schedule and gates

| Phase | Cases and seeds | Alternating pairs per case | Measured batches |
| --- | --- | ---: | ---: |
| A/A | All three, seed 7 | 5 | 8 |
| Training, only after A/A passes | All three, seed 7 | 5 | 64 |
| Reserved, only after training passes | ff 11, mona 13, darwin 17 | 5 | 64 |

Pair order is AB, BA, AB, BA, AB. Reserved cases use three separate processes;
their complete matrix is reconstructed before applying statistical gates. These
are fixed reserved RNG seeds on the same corpus, not unseen drawings or fifteen
independent search seeds. Diagnostic and qualification work stays separate from
timing rows.

Statistics use the existing `compare.statistics` paired log-mean bootstrap math,
10,000 resamples, seed 20261024, independent case strata and equal case weights.
The caller pins that seed explicitly; the helper's historical default differs.
Training and reserved statistics remain separate. Native lab analysis and old
policy helpers do not decide this campaign's gates.

A/A requires its aggregate 95% interval inside [0.99, 1.01], every case interval
inside [0.98, 1.02], and every case paired median inside [0.99, 1.01]. An interval
need not contain 1 if wholly inside its practical equivalence band. This control
qualifies the eight-batch comparison, not every 64-batch variance pattern.

Each complete A/B phase requires aggregate throughput ratio strictly above
1.03, aggregate lower 95% bound strictly above 1, every case paired median at
least 0.995, and every case lower bound at least 0.99. Failure of a valid complete
A/B phase stops this campaign as REJECT. It does not prove intrinsic slowdown.
A/A failure, wrong work, incomplete data, lock contention, cancellation or timeout
stops INCONCLUSIVE. No retries, extra samples, exclusions or warmup variants
follow a stop.

At most five processes run serially under the benchmark lock. Each has at most
600 seconds, capped by the remaining fresh 1,800-second monotonic package budget
minus a 30-second cleanup reserve. The clock starts before lock acquisition and
includes validation, analysis and export. All partial files and completed-work
counters remain available after failure. Cargo, heavy audits and browser
rendering stay outside timing windows.

## Separate setup from measured work

The private `engine-lab-prepared-v2` alias leaves the original measured function,
clock, checked drain, Drawing lifetime and all archived library/shader bytes
unchanged. Setup metrics bracket existing operations outside that clock:

| Metric | Boundary |
| --- | --- |
| Per-case device initialization | Existing GPU evolver constructor |
| Per-case source precompile | Existing precompile helper, including zero-warmup setup |
| Per-row source installation | Existing source installation and mode selection |
| Per-row preparation | Whole seeded preparation, eight warmups, drain and reset |

Report the first value, median and range by arm, case and phase. Normal and
diagnostic timings remain separate. Shared setup is not charged twice to the
arms. Qualification, snapshots and extra CPU verification are harness work.
First scored preparation already follows precompile, reference qualification
and diagnostics. Source installation can exclude deferred native compilation.
These metrics therefore do not measure whole-application startup or pure warmup
cost. Drain is already included in measured elapsed time.

The new qualification JSON producer serializes typed TrialRecord bytes before
parsing into Value and adding metadata. This preserves strict f32 parameter
spellings and nonzero f64 values. It does not alter GPU computation or loosen
the existing 2e-5 numerical policy. Old widened JSON remains untouched; only
canonical state associations, not old floating-point spellings or rates, may
support historical correctness checks.

## Proof required before interpretation

Require the exact planned case/pair/order matrix, requested modes, complete
submitted/completed work, finite rates derived from completed evaluations, zero
errors, pinned inputs/typed parameters and consistent GPU identities. Initial
and final seven-field canonical endpoints must match each phase's own reference
qualification. Reserved seeds have their own endpoints. Retain the complete
existing numerical/scalar uncertainty policy and overflow checks.

If every phase completes, scored work is 66,846,720 logical evaluations.
Diagnostics add 13,369,344; qualification adds 6,684,672. The source-accounted
117 eight-batch preparations add 15,335,424. Total expected logical evolution
work is 102,236,160 evaluations in 6,240 batches. Other full-fitness/reset GPU
setup remains separate. Partial runs report actual completed work or explicit
lower bounds rather than these planned totals.

A full pass is `PASS_PREPARED_CONDITIONAL`, with same-seed timing uncertainty,
unverified background isolation and unresolved cold/application costs. It does
not rescue the old screen or flip the engine default. The preserved ordinary
API8 proof applies to the unchanged archived engine; a fresh API64 return/flush
check is required before any production adoption.

The private lab's source freeze is
`9cc71d312fe5c2358c4b32eef4b00681f5e2884242169b6c26777ea3baae6fc5`;
binary SHA256 is
`97d176f3b5f97c818d19a5e25b2f75e9e587bf1925a948c330b78e2022512ab1`.
It passes 19 focused CPU tests, with zero owned Clippy findings. Source and
controller review must finish before runtime. Evidence and immutable build
records are under `.cache/evolution-qualification-prepared-20261024/`.

Explain the Number separates prepared work from setup and same-seed timing
uncertainty. Prove It Works requires exact completed work and state replay.
Boundary Discipline keeps the gates in the campaign supervisor.
