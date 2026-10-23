# Live acceptance protocol 2

Protocol 1's new-state engine confirmation passed. Its live package ended
invalid on the first worker, and the other 19 live workers were not run.
The ordinary trial contract and immutable corpus remain version 1.

The live worker completed all required GPU work, received all 128 publications
and passed endpoint checks. Its result formatter first converted a typed
`TrialRecord` through `serde_json::to_value`. That widened `f32` parameters to
their exact `f64` values. For example, the agreed JSON value `0.2` became
`0.20000000298023224`. The strict parameter comparison correctly rejected the
reported values. The data cannot be relabeled as an accepted timing sample.

The fix serializes the typed trial record to JSON bytes before parsing the
object used to merge live fields. This preserves the ordinary trial's wire
representation. It also preserves the existing handling of `f64` complexity
metadata. Parameter comparison remains exact; no tolerance or score gate changes.

Live results now contain `live_protocol_version: 2` in addition to the ordinary
`protocol_version: 1`. The live collector's policy version is 2 and requires
that explicit value. Missing or legacy live versions are rejected. Engine-mode
semantics, fixtures, seeds and the original comparison policy are unchanged.

All live workload, timing, correctness, deadline and statistical rules in
[acceptance policy 1](acceptance-policy.md) still apply. A protocol-2 comparison
needs a new A/A baseline and new output directory. It cannot reuse the failed
worker, continue the closed package ledger, pool samples or reinterpret old
metadata. No automatic retry of the ended package is permitted.

Before that baseline, validate one full 256-batch worker through the protocol-2
collector boundary, including exact parameters, ordinary trial fields, raw
receipt shards, state evidence and all 128 delivered images. This is an
unscored integration check, not A/A equivalence or a candidate speed verdict.

The existing new-state engine result remains separately reported. It does not
establish live latency non-regression. The original runtime binner stays installed
until the required acceptance checks pass.

The full-size integration gate has passed on the corrected reference release:
4,194,304 completed evaluations, 128 received publications, exact parameters,
zero errors and matching endpoint state/error checks. The report sets
`timing_eligible: false`. See [the retained evidence and rerun command](acceptance-result.md#corrected-full-size-integration-passed).
Fresh live A/A qualification is still pending.
