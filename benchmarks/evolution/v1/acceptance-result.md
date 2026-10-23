# Restart and live acceptance evidence

This unit adds a bounded qualification path for the archived experiment-5
tile-parallel binner. Experiment 6 remains rejected. The original runtime binner
is unchanged while this qualification runs. The frozen
[acceptance policy](acceptance-policy.md) specifies the measurement and stop rules.

## Infrastructure proof

The backend's opt-in `--live-acceptance` mode shares production rendering,
PNG encoding, drawing/message serialization and WebSocket transport. It uses
one local receiver, a bounded immutable publication queue and the existing
33ms send and 200ms image throttles. The queue fixes the workload independently
of normal image coalescing. Browser JavaScript and presentation time are outside
this transport-receipt measurement.

An initial four-batch GPU smoke completed all work, delivered both images and
matched the terminal maximum, but its next stats send raced deliberate socket
shutdown and reported BrokenPipe. That failed run remains retained. The fix
classifies only known transport-close errors after explicit stop as expected.
Subsequent review and real WebSocket regressions closed receive, Close,
command-reply, malformed-JSON and unexpected-binary error paths. Genuine failures
remain invalid even after the final required image has arrived.

The protocol-1 code passed 24 backend tests, 50 library tests and 133 Python tests.
The new live module and changed message guards have no Clippy diagnostics under
all/pedantic checks. Existing diagnostics remain elsewhere. Three malformed
message regressions failed on the prior source and passed after correction.
Independent source and collector reviews have no remaining material findings.

Both protocol-1 reference and candidate executables passed an additional GPU smoke:
65,536 completed evaluations and two received publications each, with zero
errors, no missed deadlines and an exact terminal population maximum. These
short functional runs do not contribute a timing score.

Protocol-1 reference backend SHA-256:
`555109620c3e948f96e13a426cfb33e43d147796a3bfd5f79723e232604d5f8c`.
Protocol-1 candidate backend SHA-256:
`ea1cd0855b0cba1360f6a5efcad4507aefb5046038ff271b825adee5009ce9de`.
Their archived source identities differ only in `bin_polygons.wgsl`. The
candidate keeps experiment 5's original automatic workgroup initialization.
All builds use locked release dependencies and common acceptance instrumentation.

## Distinct mature corpus

The original reference generated one new drawing per project using seed 101,
64 batches and the preselected parent 0. All three construction workers passed
the existing numerical, completed-work and overflow gates. No replacement
drawing or extra construction run was selected.

| Project | Original triangles | New triangles | Imported scalar fitness | Construction maximum occupancy |
| --- | --- | --- | --- | --- |
| ff | 423 | 474 | 97.57231% | 71 |
| mona | 539 | 559 | 97.23995% | 80 |
| darwin | 474 | 526 | 97.85356% | 82 |

Normalized reference bytes, source-image identities, image dimensions and all
mutation parameters match the original capture. Every drawing hash and packed
geometry differs. GPU trial preparation independently requires every active
initial parent to exceed 97%. These are new mature-drawing restarts, not
continuation checkpoints or new image families.

Manifest SHA-256:
`880115ff3e75703abf23370fe67b21052c7be302196683532efc959207f9f841`.

The fixed engine matrix uses seeds 11/13/17, five alternating 64-batch pairs per
project, one overflow replay per project, and separate 8/32-batch correctness
prefixes. The 45 workers account for 38,535,168 completed evaluations when all
finish; diagnostic and prefix work is excluded from the throughput score.

## Run bundle and current scope

All source snapshots, release identities, red/green logs, raw client receipts,
state files, inputs and decisions are retained under
`.cache/evolution-qualification/acceptance-07/`. Staged commands share the
original monotonic 3,600-second package ledger. No timer reset, sample extension,
slow-pair exclusion or retry is allowed. Background activity remains permitted
and isolation is unverified under the user's instruction.

The engine confirmation passed. The live package stopped invalid on its first
worker, so no A/A equivalence or candidate latency verdict exists. The remaining
19 live workers were not run. No adoption or broader campaign completion is
claimed.

Prove It Works required actual GPU work and client receipt bytes. Explain the
Number separated completed engine work, transport latency and extra audits.


## Engine confirmation passed on the new states

All 45 workers completed with zero errors. Every original policy-1 gate passed,
with exact seeded endpoint and prefix replay. Completed submitted work agreed
at 38,535,168 evaluations, including 3,145,728 diagnostic and 3,932,160 prefix
evaluations excluded from the timing score. The 30 timed workers contributed
31,457,280 completed evaluations. No slow pair was excluded.

The aggregate paired throughput ratio was **2.30456151**, with conditional 95%
interval **2.29967860 to 2.30992064**. This is a separate confirmation on the new mature
states and seeds, not a pooled estimate with the original seed-7 result.

| Project | Reference eval/s median, range | Candidate eval/s median, range | Paired geometric ratio | Conditional 95% interval |
| --- | --- | --- | --- | --- |
| darwin | 55,701.45, 55,593.17 to 55,784.09 | 131,782.84, 131,658.78 to 133,742.13 | 2.37993875 | 2.36598295 to 2.39467580 |
| ff | 66,626.79, 66,488.31 to 66,814.97 | 147,406.20, 146,846.77 to 147,954.62 | 2.21123295 | 2.20869401 to 2.21377481 |
| mona | 54,776.54, 54,537.90 to 54,890.55 | 127,454.13, 126,658.88 to 127,788.03 | 2.32575858 | 2.32011255 to 2.33162052 |

The parent independently checked 180 output artifact hashes, 90 canonical
endpoints and 360 framebuffer byte-L1 totals. The result is retained at
`.cache/evolution-qualification/acceptance-07/engine/summary.json` and the
recomputable audit at `engine/parent-verification.json`. The generic policy-1
verdict remains INCONCLUSIVE for adoption while `reserved_confirmation` is
passed. These timing intervals describe five repeats of one trajectory per
project; the three project seeds differ from the development seed 7.

## Live package stopped on result serialization

The first reference A/A worker physically completed all 4,194,304 evaluations,
received all 128 scheduled publications, missed no 300ms deadline and matched
the terminal packed maximum. Its process exited zero and its internal payload
and endpoint checks passed. The collector nevertheless marked the worker and
package INVALID because reported parameters did not exactly match the frozen
manifest.

`serde_json::to_value` widened the typed trial's f32 parameters before writing
JSON. Twenty-five fields changed their wire values; for example, 0.2 became
0.20000000298023224. Input and GPU settings had not changed. Exact comparison
was retained, and the raw result was not edited or reclassified as an eligible
sample. The other 19 live workers remained unrun. No candidate live comparison
or A/A equivalence claim follows from that worker.

The package closed after 45,875,200 reported evaluations across three
construction, 45 engine and one live worker. Earlier short functional smokes
are separate, unscored evidence outside that package. The old monotonic ledger
will not be reset or reused to continue the failed measurement.

[Live protocol 2](live-protocol-2.md) corrects typed float serialization and
requires an explicit new live version and fresh A/A baseline. Its preparation
and full integration check are separate from the closed package. The existing
new-state engine result remains valid; no kernel, mutation, precision or
selection change is part of the serialization repair.

## Corrected full-size integration passed

The corrected live protocol-2 reference executable completed one separate,
unscored worker on reserved mona at seed 13. It completed all 256 batches,
16,384 iterations and 4,194,304 evaluations after two warmup batches. Submitted
and completed counters agree. All 128 scheduled images were sent and received,
with zero errors or missed 300ms deadlines. The terminal publication matched
both the score and geometry of a final packed maximum parent.

The current collector accepted the actual result with exact manifest parameters
and nonzero complexity metadata. It checked ordinary protocol 1 plus live
protocol 2, all raw receipt identities and payload hashes, and the original
cache inventory. The parent recomputed both canonical endpoints and all eight
initial/final framebuffer byte-L1 totals. The receipt shard contains 128 actual
frames and 35,569,172 bytes. Its contents and the result/state artifacts remain
under `protocol2-validation/reference/output/`.

An independent read-only audit confirmed the raw receipts, both canonical
hashes, all eight error totals, terminal maximum and corrected release identities.
It found no material defect in the integration evidence or fresh-run script.

The validation report is
`.cache/evolution-qualification/acceptance-07/protocol2-validation/summary.json`,
with `valid: true`, `error_count: 0` and `timing_eligible: false`. This worker
establishes integration, not live A/A equivalence, latency non-regression or
candidate performance. It is separate from the closed protocol-1 package.
No timing sample or ledger is reused.

The repair passed 26 backend tests and 35 collector tests. The original 24
backend and 34 collector tests remain intact. Fresh source review found no
material issue and confirmed zero new Clippy diagnostics. Existing library
and full-suite results above predate this narrow serialization repair.

Both corrected locked-release backends are archived under `v5/`, sharing the
same acceptance source. Reference SHA-256:
`ba2e484e27bd0316de381e01f1f273eadaae8570dc325fbfc5c1284ad1e8dd84`.
Candidate SHA-256:
`c0802ff5b474b79ed3b5d9e2e43bc2ae2f8e0314689c45129fd8ff4956dac41d`.
The candidate retains experiment 5's automatic workgroup initialization.
The normal executable and shader were restored to the original binner.

## Next measurement

When the machine is otherwise quiet, run from the repository root:

```sh
bash .cache/evolution-qualification/acceptance-07/rerun-live-protocol2.sh
```

The script verifies frozen collector/helper, binary and manifest hashes, then
creates a new output directory and a fresh 3,600-second clock. It runs five
live A/A pairs first. A/B runs only if both the throughput and p95-latency A/A
intervals fit the unchanged 0.99 to 1.01 bounds. A failure or inconclusive gate
ends that package without retries or extra samples. The output is
`.cache/evolution-qualification/live-protocol2-<UTC timestamp>-<PID>/summary.json`.
Quiet-machine operation does not itself prove the equivalence bounds.
