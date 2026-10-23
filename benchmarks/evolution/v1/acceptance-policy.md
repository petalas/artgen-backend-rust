# Bounded restart and live acceptance, protocol 1

This package qualifies the unchanged experiment-5 candidate beyond the seed-7
timing screen. Experiment 6 remains rejected. Do not change candidate settings,
combine scores from earlier runs or install a runtime optimization from this
package's exploratory verdict alone.

Freeze executable, source, helper, policy, cache and fixture identities before
the first package worker. Background video activity is permitted under the
user's instruction. Record isolation as unverified, retain every slow pair and
leave other processes untouched.

## Distinct mature states

Current project drawings duplicate the original fixtures. Generate exactly one
new state per project using the original reference executable, seed 101 and
64 batches. Select parent 0 in advance. Preserve its packed source, exported
drawing and construction record. Do not select a different parent or repeat
construction to obtain a favorable input.

Import ff, mona and darwin into a separate immutable corpus. Source images,
normalized RGBA, dimensions, resolution and mutation parameters must match the
original capture. Each drawing hash must differ from its original counterpart.
Reject duplicate, invalid, overflowing or below-97% starts without replacement.
These are mature-drawing restarts, not continuation checkpoints or held-out
image families.

Use ff seed 11, mona seed 13 and darwin seed 17. For each, run five alternating
reference/candidate pairs at 64 batches, followed by one candidate overflow
replay. Run an additional reference/candidate correctness comparison at each
prefix of 8 and 32 batches. Prefix and diagnostic runs do not enter timing.
All trials use two warmups, four chains, 64 offspring per chain and 64 iterations
per batch. The 45 engine workers complete 38,535,168 evaluations if all finish.

At start and each declared endpoint require exact canonical state replay,
effective settings and adapter identity; checked cached RGB-L1 versus framebuffer
bytes; the existing scalar uncertainty policy; strict initial similarity above
97%; identical quality/complexity; completed submitted work; and occupancy at
most 256 in diagnostic replay. Reuse the existing trial implementation and
comparison statistics. Apply the unchanged policy-1 aggregate and per-project
throughput gates, with equal project weights and no timing exclusions.

## Real backend publication workload

Use the actual backend renderer, PNG encoder, drawing serialization, WebSocket
handler and throttles. An opt-in acceptance mode uses an isolated output and
loopback ephemeral port. Ordinary application evolution stays unchanged.
The receiver and sender share a monotonic clock.

Run reserved mona, seed 13, two warmups and 256 measured batches. Complete
4,194,304 evaluations and schedule one publication per two completed batches,
including the final drain. Exactly 128 due, sent and received publications are
required. Publish repeated content when the drawing has not improved. An
acceptance-only bounded queue preserves identities that normal coalescing
would discard. Receipt acknowledgements must not pace evolution.
The shared stats serializer also runs through the existing 33ms send throttle
between queued images. This acceptance workload fixes message production
independently of normal state coalescing; it is not a claim about the normal
application's observed message frequency. The local receiver measures transport
receipt and validates PNG consistency, not browser rendering or JavaScript time.

Each publication records its run ID, sequence, completed-work origin,
due/send/receive times, drawing/PNG hashes, score and polygon count. Validate
payload identity and decoded PNG dimensions and bytes through the production
CPU renderer. GPU pixels follow a different rasterization contract. Reuse the
ordinary trial's numerical endpoint checks. Historical score binding preserves
the drawing's own mapped header score and acquisition copy batch. It must be
finite and belong to that copied population's score vector. A previously
selected winner may have been overtaken before its copy, so equality with the
acquisition maximum is telemetry only. Repeated publications retain the original
binding. This does not independently prove historical pixel fitness or diagnose
the earlier best-ID signal race. Do not repair that race in the observer.
The terminal published drawing and GPU score must attain a packed parent
maximum after the checked final drain.

Use a fixed 300ms due-to-receipt deadline. Source inspection shows a 200ms image
throttle plus a 33ms server poll, so the earlier proposed 200ms deadline had no
scheduling margin. The 300ms value is selected before any live data. Keep the
actual production throttle. Timing includes command encoding, submission,
ordinary readback, publication work, final GPU drain and receipt of all due
publications. Extra endpoint audits and artifact writes remain outside it.

First run five reference A/A pairs. Qualify 128 publications, the deadline and
paired throughput and per-run p95-latency equivalence intervals within
0.99 to 1.01. If this fails or is inconclusive, stop without A/B. Otherwise run
five alternating A/B pairs. Require throughput median ratio at least 1 and
lower bound at least 0.99. Require latency candidate/reference median at most 1
and upper bound at most 1.01. Missing publications, device errors or an increased
missed-deadline count fail acceptance. Resample run pairs, not individual
publications, using 10,000 paired log-mean bootstrap draws and seed 20261023.

## Bounds and disposition

Build and prove the infrastructure before starting the package clock. Freeze a
same-boot monotonic ledger at the first construction worker. All staged commands
share its hard 3,600-second deadline, including lock acquisition. Enforce at
most 120 seconds per worker in the supervisor. Cooperative in-process checks
do not bound a stuck GPU driver.

The maximum package is three construction workers, 45 engine workers and
20 live workers: 68 workers and 125,566,976 completed evaluations. A failure,
unqualified A/A or inconclusive interval ends the relevant package. Retain
partial work and planned-but-unrun stages. Do not extend samples, retry until
favorable, filter a slow pair or change gates after data.

Every process, device, numerical, capacity or artifact error remains visible.
Cancellation must stop and reap owned workers and write a partial summary.
Keep the original runtime default until the required acceptance gates pass.
Policy 1 remains an exploratory screen and cannot return KEEP. This package
does not establish true continuation, broad image-family generalization or
portable behavior on untested GPU backends.

Prove It Works requires real client receipts and packed-state evidence.
Explain the Number separates engine work, live delivery and setup.
