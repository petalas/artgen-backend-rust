# Live latency repeatability diagnosis

The operator ran the frozen protocol-2 script on an otherwise quiet machine.
The package completed ten valid A/A workers with zero errors, then stopped
INCONCLUSIVE at its unchanged preview-latency equivalence gate. Every executed
worker used the original reference executable, including rows labelled
`candidate`. The optimized binner's ten planned A/B workers were not run.

Evidence is retained at
`.cache/evolution-qualification/live-protocol2-20261023T232627Z-210491/summary.json`.
The package accounts for 41,943,040 completed evaluations and 1,280 delivered
images. The parent independently checked 60 artifact hashes, 20 canonical
endpoints, 80 framebuffer byte-L1 totals, all receipts and the paired bootstrap.
All initial and final canonical states agree across workers.

| A/A metric | Paired geometric ratio | Conditional 95% interval | Frozen equivalence gate |
| --- | --- | --- | --- |
| Completed evaluations/sec | 1.00013544 | 0.99953152 to 1.00073972 | Passed |
| Per-run p95 preview latency | 0.98087004 | 0.96454406 to 1.00060706 | Inconclusive |

Both intervals must fit within 0.99 to 1.01. Five run pairs are the sampling
units. These are repeated measurements of one trajectory, not independent
evolution seeds. No slow pair was removed and no extra sample was added.

Across the ten workers, p95 preview latency ranged from 35.34 to 37.19ms.
Its existing timestamp decomposition narrows the investigation:

| Existing interval | Range of per-run p95 values |
| --- | --- |
| Publication due to legacy send marker | 34.71 to 36.58ms |
| Legacy send marker to complete client receipt | 0.67 to 0.72ms |

The legacy send marker precedes message construction and serialization. Its
second interval is therefore not isolated network time. Component quantiles
cannot be added to reconstruct an end-to-end quantile.

## Hypothesis and diagnostic boundary

The shared WebSocket handler checks publications after its 33ms sender sleep.
Images also retain their 200ms eligibility rule. The existing due timestamp
precedes drawing, PNG encoding, drawing JSON, locks, hashing and enqueueing.
The data is consistent with a polling-phase effect, but does not distinguish
that effect from variable preparation, lock waits or sender wakeup delays.

The next diagnostic records those boundaries on the actual workload. An
optional `--latency-trace v1` on `--live-acceptance` captures integer nanoseconds
relative to the session clock. Trace data stays off the wire and is written
after connection shutdown and thread joins. The ordinary protocol-2 output,
publication schedule, renderer, GPU work and sender/image cadences stay unchanged
when tracing is omitted.

The trace separates producer drawing/PNG/JSON work, state and ledger locks,
enqueueing, sender poll targets and wakeups, image eligibility, dequeueing,
message serialization, socket writes and complete receipt. Stats turns are
included because a slow stats send can delay the next publication check.
Receipt can precede the sender thread's return from `ws.send`; that is a valid
interleaving, not a negative transport duration.

Storage is bounded to the existing publication count and 8,192 sender turns,
with the existing 64MiB artifact limit. Capacity errors retain partial evidence
and invalidate the diagnostic. Disabled tracing allocates no trace storage and
takes no extra clock samples.

CPU tests exercise the trace and real loopback handler first. One separately
identified, unscored reference worker then uses reserved mona, seed 13, 256
batches, two warmups, a publication every two completed batches and the
unchanged 300ms deadline. It must retain 4,194,304 completed evaluations, all
128 receipts, ordinary numerical checks and exact endpoint replay.

Its result uses `mode: live_latency_diagnostic` and `score_eligible: false`.
The existing collector rejects that mode for scoring. This worker locates
latency episodes; it cannot establish repeatability or a speedup. The diagnostic
does not resume the stopped A/A package, alter its result, loosen the 1% gate or
install the optimized binner.

Prove It Works requires actual receipts and retained trace rows. Fix Root Causes
keeps the investigation at the earliest varying stage. Explain the Number keeps
the traced observation separate from the accepted engine throughput evidence.

## Captured trace

One locked-release reference worker completed with zero errors, 4,194,304
submitted and completed evaluations, all 128 receipts and no missed deadline.
Its initial and final canonical states exactly match the operator's A/A
trajectory. The parent recomputed both hashes and all eight framebuffer byte-L1
totals, checked every receipt, and verified that the ordinary scoring collector
rejects the diagnostic mode. No second worker was run.

The bundle is `.cache/evolution-diagnostics/live-latency-09/`. The executable,
Cargo artifact/profile, source snapshots and identities are under `build-v2/`.
`worker/output/latency-trace-v1.json` contains 128 complete publication rows and
2,304 sender turns. Its 1,578,833 bytes hash to
`3337b4b3fe3dfff7ea0ced7dc77b4e4cba11b7a29711ba57dae1ce02a65b7ed0`.
The verified analysis is `worker/analysis.json` and the additional per-message
checks are `worker/additional-analysis.json`.

| Interval | Median | p95 | Maximum |
| --- | --- | --- | --- |
| Drawing, PNG, drawing JSON, hashing/copying/locks combined | 4.289ms | 4.530ms | 5.973ms |
| Enqueue to dequeue | 15.055ms | 30.325ms | 32.478ms |
| Message preparation and wire serialization | 0.373ms | 0.428ms | 0.447ms |
| Socket-write start to complete client receipt | 0.155ms | 0.190ms | 0.320ms |

The individual components in this table are distributions over the same 128
messages. Their p95 values must not be added. The measured end-to-end p95 was
35.201ms in this one instrumented run, not a before/after performance result.

Every publication was enqueued during the sender's timed sleep, after the
preceding send had returned. Every publication left on the first eligible check.
Queue depth never exceeded one. No queued message encountered the image throttle.
Sender wake overshoot had p95 0.101ms; stats writes had p95 0.047ms. Producer and
sender state-lock waits had p95 0.000130ms and 0.000331ms. These observations
locate the large delay at waiting for the next sender poll, rather than backlog,
image eligibility, a long lock wait or a long socket write in this run.

Publication 54 is the observed message closest to the run's p95. Its components
add to its actual 35.34ms total, shown rounded here:

| Publication 54 component | Duration |
| --- | --- |
| Drawing/PNG/drawing JSON/hash/copy/lock preparation | 4.19ms |
| Queue residence until the next sender check | 30.64ms |
| Message preparation, serialization, write and receipt | 0.51ms |

Its arrival phase explains the long queue residence. The trace establishes this
mechanism on the recorded run. It does not prove that every part of the earlier
A/A confidence interval came from polling phase, nor quantify instrumentation
overhead or identify a hardware bottleneck.

## Verification and next step

All 38 live CPU tests passed, including the prior 26 and real producer/loopback
coverage, ordering, disabled tracing, capacity exhaustion and partial-error
artifacts. Check, scoped formatting and all/pedantic Clippy passed without new
owned diagnostics. The specification review found no material defect. Standards
review found two oversized functions; cohesive extraction reduced them to 49
and 54 lines. Re-inlining the helpers reproduces the reviewed source byte for
byte, and the same 38 tests passed after extraction.

The existing 33ms polling cadence is intentional production behavior. Removing
it just to pass this qualification would change the workload being qualified.
The next measurement design should test control of startup phase and a justified
sample budget while retaining the original 1% bound. Freeze any new schedule
before collecting another A/A package. The old result remains INCONCLUSIVE.

No new baseline rerun is requested yet. The normal engine still uses the original
binner; candidate 5's separately confirmed engine gain remains archived.
