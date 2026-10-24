# Locate the first-trial penalty before another scheduling change

The [finalization-overlap experiment](09-finalization-overlap.md) remains rejected.
The next task is attribution, with no production change or adoption claim.
One private observer will record the first and later timed trials using the
original mona, seed 7, eight-batch, two-warmup workload.

The earlier two-warmup screen and separate four-warmup diagnostic both retained
a slow first candidate. In the latter, its first trial took 2.142444 times its
later trial; the reference first/later ratio was 0.980247. Those are two-pair
observations, not proof of a steady-state gain, a universal startup cost, or a
driver/JIT cause. An offline detector retains this retrospective signature;
it is not a replacement for the original performance gate.

## Distinguishing observations

Installed wgpu 30.0.1 source does not create a compute pipeline merely because
the experiment splits a submission. It can acquire an encoder pool, allocate
native command buffers, encode backend commands during `finish()`, prepare each
submitted buffer and call the Vulkan driver. The actual backend is Vulkan/Dozen
over the NVIDIA adapter. The accepted single-buffer baseline also had unusually
long first measured finishes, so the penalty cannot be assumed unique to the
split. Archived logs contain no explicit lazy-compilation message.

| Hypothesis | Observation that would support its location |
| --- | --- |
| Encoder or backend finalization adds the time | Excess in encoder acquisition, prefix finish or tail finish |
| Submission or completion adds the time | Excess in the isolated submit or collection/poll bracket |
| Warmup misses the guarded path | Zero actual guarded executions between preparation snapshots |

An excess before the first true guard would contradict a guarded-execution
location. A positive lifetime count before the first timed candidate would show
that the path had already executed during untimed work. Neither elapsed-time
observation alone identifies CPU work, GPU work or descheduling.

## Bounded private recorder

The [frozen specification](profiles/first-trial-observer-spec.json) permits one
unscored instrumented invocation. A fresh copy of the archived prototype adds
the unique `first-use-engine-lab` example, so its shared-cache build does not
replace the normal benchmark binaries. Root and historical archives stay
unchanged. The validated host-phase recorder supplies bounded storage,
ordering checks and capped post-measurement export.

Four sidecars correspond to the actual timed order, reference, candidate,
candidate, reference. Each records actual guard operands, split ordinals,
successful two-buffer submissions, direct host phases and nested readback
phases. Before/after preparation snapshots isolate warmup guard counts without
timing warmup. Lifetime counters survive seeded resets and include previous
qualification and diagnostic work.

The original clock, command order, work, callbacks, Drawing lifetimes and single
checked drain remain intact. No event logs, file writes, additional GPU calls or
extra drains occur during measurement. Export follows elapsed capture, with a
2 MiB limit per sidecar and a 120-second outer process bound. Invalid or partial
evidence remains invalid and is retained. A killed process cannot export its
current RAM buffer.

The actual exporter must reject missing, reversed, escaped or unfinished spans,
incorrect targets and exhausted storage. Source inverse checks, include/input
inventories, compiler/profile provenance, actual binary identities and exact
endpoint replay precede interpretation. Nested readbacks are already contained
in collection brackets; CPU and GPU elapsed spans overlap.

If instrumentation loses the symptom, at most one unmodified control invocation
may check whether it still reproduces. Both remain unscored. No slow sample is
dropped, merged into the old result or used to extend its failed screen. A fix
requires a separate mechanism and fresh declared comparison after attribution.

Source reports and design are retained under
`.cache/evolution-profiling-20261024/first-use-source-91/` and
`first-use-observer-design-92/`. The continuation's inputs, retrospective detector
and decisions are under `first-use-20261024/`.

Attack the Premise keeps warmup and first-use explanations as testable claims.
Explain the Number separates elapsed phase locations from causal explanations.
Prove It Works requires valid exported rows and actual completed work.
