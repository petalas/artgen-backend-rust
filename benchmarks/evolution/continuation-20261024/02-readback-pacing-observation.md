# Observe pre-submit drawing readback collection

One unscored private mona worker uses seed 7, 64 batches, two warmups and unchanged
parameters. It completes 1,048,576 evaluations with zero errors and exact original
canonical state, quality and completed-work replay. The root engine is unchanged.
The private source adds a timer around the existing pre-submit finish call and
markers outside the existing measured-loop clock. No GPU work is added.

The complete log records 21 pre-submit finish calls, at completed batches
3, 6, 9 through 63. Observed call time sums to 1064.229859 ms; median is
50.739484 ms, p95 51.190265 ms and maximum 51.548564 ms. The instrumented loop's
wall time is 3.992433534 seconds. These are observations from one modified worker,
not a throughput comparison. Call durations include prior GPU completion,
mapping, decoding and bookkeeping. Their sum is not recoverable stall or a
promised optimization share. Diagnostic writes also perturb whole-loop timing.

The private locked release build has optimization 3, the same flags and all three
accepted shaders embedded. Its binary SHA256 is
b08097e044683727a17a7e4c024551d26b6547b9f1a7ae4607a6d1c7e86ca49c.
All 83 source files and 30 literal includes are archived. Removing the two-file
observer insertion restores the root files byte for byte. The parent validates
complete markers, strict unique fields, bounded line sizes, monotonic counters,
pending staging/iteration/evaluation metadata and actual end-state artifacts.

A separate source audit suggests one smaller scheduling prototype before adding
another staging buffer. Collect the same pending readback after encoding compute
passes but before current drawing-copy selection and queue submission. The control
reset must move with it: finish can update best_fitness_bits, so constructing reset
bytes before finish changes the ordinary readback threshold and cadence. The
source proposal adds no buffer, dispatch, parameter or skipped readback. At this
observation stage it was uncompiled and unmeasured. It required ordinary
Some/None/content/flush qualification as well as exact population replay. A bare finish-call move is rejected.

Evidence is in `.cache/evolution-continuation-20261024/readback-observer-58/` and
`readback-order-audit-60/`. The historical incomplete observer57 remains unchanged.
This result selects a question for further work; it establishes no speedup or GPU
utilization. Explain the Number prevented treating the full finish-call duration
as available headroom. Prove It Works required actual completed-work counters
and exact saved-state replay.

The subsequent [bounded prototype](04-late-readback.md) passes ordinary API
qualification but fails its throughput advancement gate. Its runtime code is
removed; the observer does not justify another pacing change by itself.
