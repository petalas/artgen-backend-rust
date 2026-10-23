# Paired experiment screening, policy 1

This policy is fixed before measuring the first dirty-region binning candidate.
It screens an isolated engine change on the three qualified mature restarts.
It cannot accept a campaign winner. Live preview acceptance, reserved states,
independent evolution seeds and the broader campaign qualification remain open.

The reference executable is built from `dd52d50`, which adds profiling support
to the engine used for the recorded reference. The candidate differs only in
polygon bin construction. Both executables and the shader patch are archived
before timing. Supplied executables retain an unverified-build label in the
generic comparison command; adjacent Cargo artifacts and source hashes provide
this experiment's build evidence.

## Work and order

Use the immutable corpus and pipeline-cache seed from the user capture at
`.cache/evolution-bench/baseline-20261023T172054.919294Z-95742/summary.json`.
Preserve seed 7, two warmup batches, 64 measured batches and every frozen setting.
Each normal worker completes 1,048,576 evaluations. ff, mona and darwin have equal
weight, and every initial active parent must exceed 97% fitness.

Run five reference/candidate pairs per case. Reverse order each pair, starting
with reference then candidate. Run one separate candidate overflow-diagnostic
replay per case after its timing pairs. Five pairs across three projects produce
30 normal workers and three diagnostics. Diagnostics do not contribute to timing
scores. A shorter or smaller invocation is explicitly a preliminary screen.
The hard cap is ten pairs per case. Do not add samples after seeing a favorable
or unfavorable interval; a larger confirmation schedule needs a separate frozen
invocation.

Every worker runs serially under the benchmark lock, with copied fixtures,
isolated working directories, checked exits and bounded execution. Record the
machine observations and all failures. The user allowed development runs despite
possible background GPU activity. Preserve isolation as unverified, leave other
processes untouched, and make no clean-machine claim. Never exclude a slow pair
because it looks noisy.

## Correctness and accounting

Require complete submitted/completed counters, valid numerical checks, exact
initial/final canonical replay between reference and candidate, and matching
quality, complexity, parameters and adapter identity. For the captured seed and
budget, also require agreement with the archived reference trajectory. Retain
drawings as export artifacts. Recompute canonical hashes from both
`initial-state.json` and `state.json`, associate their parents with the checked
initial/final quality, and compare them with each result. Scoring uses the packed
states. Exported drawing JSON is retained for inspection rather than used as a
separate numerical authority.

The candidate diagnostic must match its normal trajectory and observe no consumed
tile bin above 256 entries. The dirty-region candidate builds only bins that the
rasterizer can read; its diagnostic covers those actual append operations over
every iteration. The initial drawing still receives full-image binning and full
complexity checks. No overflow may be excused on a consumed tile.

Any worker, replay, capacity, overflow or artifact error invalidates the entire
comparison. Retain its work in accounting and exclude invalid pairs from any
score. Do not selectively retain successful pairs from an invalid invocation.
Profiling modes cannot contribute primary throughput samples.

## Statistics and provisional decision

For pair i in case c, compute the rate ratio R[c,i] as candidate completed eval/s
divided by reference completed eval/s. The aggregate is the exponential of the
equal-weight mean of the three case means of log R. Keep raw rates, per-case
medians and ranges, and each paired ratio visible. A faster case cannot receive
more influence merely because its workers finish sooner.

Use 10,000 bootstrap resamples with random seed 20261023. Resample paired log
ratios independently within each case, preserving the three-case structure.
Use the 2.5th and 97.5th percentile endpoints for each case and the aggregate.
Each case interval estimates exp(mean(log R)); its median-ratio gate below is a
separate statistic.
These are exploratory intervals conditional on the observed same-seed timing
repeats. Five pairs cannot establish future noise limits, independent search
quality or reliable coverage under arbitrary machine drift.

A provisional timing screen passes only when all of these hold:

- At least five complete pairs exist for every case.
- Aggregate ratio exceeds 1.03 and its interval lower bound exceeds 1.0.
- Improvement also exceeds the captured maximum absolute A/A deviation,
  approximately 0.4631%, recorded directly from the original control pairs.
- Every case has a median paired ratio of at least 1.0 and an interval lower
  bound of at least 0.99. This tolerance does not permit a known slowdown.
- All numerical, replay, overflow, work and artifact gates pass.

An insufficient sample budget or interval is inconclusive. A demonstrated
per-case slowdown or a practical gain below the frozen threshold rejects the
candidate screen. No result from this command is `KEEP`. Even a passing timing
screen records `INCONCLUSIVE` for adoption, with missing live/held-out/clean-run
qualification gates listed. Retain a promising shader as an experimental patch
and archived executable while the production shader remains the reference.

The [comparison command](../../../scripts/compare-evolution.py) owns this analysis
and records its exact constants and script identity in each bundle. Its tests
use synthetic workers to check work, failure, pairing and statistics contracts.
Real GPU checks and independent artifact review remain required for the result.

Explain the Number keeps these paired engine measurements separate from sampled
stage attribution. Build the Lever makes the comparison rerunnable. Prove It
Works requires exact state and numerical agreement before interpreting rates.
