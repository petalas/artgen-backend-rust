# New kernel investigation control

This is a fresh control for independent GPU-kernel work. The previous
prepared-overlap campaign remains closed INCONCLUSIVE. No old timing samples or
ledger are reused, and no overlap candidate runs here.

The [frozen control specification](profiles/kernel-control64-spec.json) uses the
already-built uninstrumented lab, accepted AE1/G1/bin sources and identical
reference/control arms. Finalization overlap is explicitly false. The original
manifest and all evolution parameters remain fixed. There are five alternating
pairs per ff, mona and darwin, seed 7, 64 measured batches, eight preparation
batches and ordinary Stages timestamps. This tests a longer measurement window,
not a changed engine.

A/A bands are unchanged: aggregate 95% interval within [0.99,1.01], each case
interval within [0.98,1.02], each paired median within [0.99,1.01]. Statistics
use the pinned helper's 10,000 paired-log bootstrap draws and seed 20261024.
One process runs under the nonblocking benchmark lock with a new exclusive
clock, a 900-second total deadline, a 600-second child limit and 30-second cleanup
reserve. No retries or extra samples follow a failure.

Validation uses the corrected explicit unscored qualification contract and the
unchanged normal timing, numerical, scalar, typed-parameter, work, device, saved
state and seven-field replay checks. The validator is imported for its functions
and immutable executable/input identity metadata only; its old campaign entry
point is never called. The one-shot runner is under
`.cache/evolution-next-20261024/control-64/`. Source inspection, preflight and
scoped Ruff pass; no Rust rebuild is needed.

Full completion would retain 31,457,280 timing, 6,291,456 diagnostic and 3,145,728
qualification evaluations, plus 5,111,808 source-accounted warmup evaluations.
Their sum is 46,006,272; full-fitness/reset GPU setup remains separate.

A valid control pass can support a separately frozen new kernel comparison. It
cannot demonstrate an engine speedup, rescue a rejected experiment, establish
background isolation or authorize a production change. A failed control stops
this investigation's candidate timing. Source selection proceeds independently
without candidate measurements.

Explain the Number requires stable controls and keeps setup separate. Prove It
Works requires the actual completed work and saved numerical states.
