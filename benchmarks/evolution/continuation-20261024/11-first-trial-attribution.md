# The extra elapsed time is inside command-buffer finalization

One qualified private observation reproduces the first-candidate penalty and
locates its largest pauses in `encoder.finish()`. It does not identify the
lower-level cause. The original overlap experiment remains REJECT, with no
production or evolution-parameter change.

The [frozen diagnostic](10-first-trial-diagnostic.md) used mona, seed 7, eight
measured batches and two warmups. Four timed rows completed 524,288 evaluations;
separate diagnostics completed 262,144 and qualification completed 131,072.
Every row completed its expected work with zero errors and eight timestamp
samples. Initial/final state replay matches the original accepted trajectory.

| Timed row | Wall seconds | Prefix finish ms | Final finish ms |
| --- | ---: | ---: | ---: |
| First reference | 0.472492559 | 0 | 114.122636 |
| First candidate | 0.919391229 | 220.534923 | 477.773195 |
| Later candidate | 0.435652217 | 21.691827 | 79.791553 |
| Later reference | 0.478428782 | 0 | 117.756378 |

The first candidate accumulates 596.824738 ms more finish elapsed than the later
candidate. Its result wall is only 483.739012 ms longer. Direct collection
brackets are 113.905816 ms shorter, with further differences in drain and other
phases. CPU and GPU work overlap, so the finish difference is not recoverable
savings. These are host-thread elapsed intervals including off-CPU waits, not
CPU-cycle counts or a steady-state throughput comparison.

## The first guarded call is not the large pause

Both candidate trials actually split at batches 4 and 7. In the first trial,
batch 4's prefix finish takes 10.648844 ms. The three largest direct spans occur
later:

| Batch | Phase | Elapsed ms |
| --- | --- | ---: |
| 5 | Final finish | 221.940208 |
| 7 | Prefix finish | 209.886079 |
| 8 | Final finish | 204.246266 |

Tail encoder acquisition totals just 0.002554 ms in that trial. Compute recording
totals 0.478298 ms and queue submission 6.760804 ms. Per-buffer work inside the
submit brackets cannot directly explain the large earlier finish spans.

The first timed candidate starts with two successful split submissions already
recorded during untimed work. Its measured splits have global execution ordinals
3 and 4; the later candidate has ordinals 5 and 6. Each preparation window shows
two guard evaluations and zero splits. This confirms that the two warmups did
not exercise the branch, but disproves the claim that it had never executed.
Earlier validation diagnostics use a different instrumented bin pipeline, so
those prior executions do not establish identical preparation.

## Pool recycling is a hypothesis

Installed wgpu 30.0.1 source acquires encoders from a shared LIFO allocator.
Each Vulkan encoder owns a command pool. Native command buffers are allocated
in groups of 16; completion maintenance resets and returns encoders. Pending
uploads also use this allocator. Extra copy-only tails can therefore change
pool availability and histories.

The sidecars do not record native handles, free-stack contents or completed
fence values. They cannot show that tail 4 becomes compute 5, or prove driver/JIT
work. Backend encoding, native allocation and descheduling inside finish remain
possible explanations. No source-supported production fix follows yet.

The source review recommends one new, unscored preparation check with eight
warmups in both arms, because the measured pauses occur at batches 5, 7 and 8.
Four warmups reach only the earlier nonspiking guard. The
[separate prospective specification](profiles/full-path-preparation-spec.json)
retains all measurements, unchanged timed work and exact endpoint checks, with
the extra preparation work reported separately. Disappearance would support
preparation sensitivity only. Persistence would reject this preparation's
sufficiency. No further warmup sweep or old-result promotion is authorized by
that diagnostic.

## Evidence and verification

All four real sidecars pass native validation. Independent audit passes 373
checks over 324 pinned inputs, including 32 batches, 36 readbacks, 264 direct
and 316 nested spans. Six saved snapshots reproduce all seven canonical state
fields; 24 RGB-byte L1/fitness checks pass with maximum difference
7.62939453125e-6 against the unchanged 2e-5 tolerance. No scalar rerender was
performed. The retained symptom makes the conditional unmodified control
unnecessary; none was run.

Direct spans, drain and explicit gaps close to each sidecar wall. Its end marker
follows original elapsed capture by 170, 221, 150 and 181 ns. These offsets are
retained rather than silently clamped. Nested readback/poll intervals are already
inside collection/drain and must not be added again. Instrumentation overhead,
background isolation and repeatability remain uncalibrated.

Ten actual exporter CPU tests pass, and independent source review finds no
material issue. Inverse reconstruction restores all 84 archived sources.
One locked release build uses optimization 3 and debug assertions false, with
the exact shader and provenance bytes verified. The unique example target
leaves all 78 accepted runtime Rust/WGSL files and all three normal executables
unchanged.

The [observer patch](attempts/05-first-trial-observer.patch) applies after
[the archived prototype](attempts/04-finalize-overlap-rejected.patch) in a private
copy. A reconstruction check reproduces all four observer postimages exactly.
The [result data](profiles/first-trial-attribution.json) retain the per-row and
per-batch accounting. Raw source, binary, CPU artifacts, four physical sidecars,
audit and lifecycle report remain under `.cache/evolution-profiling-20261024/`
in `first-use-observer-93/`, `first-use-review-94/`, `first-use-20261024/`,
`first-use-result-audit-95/` and `first-use-lifecycle-96/`.

Explain the Number separates phase location from resource cause and savings.
Attack the Premise rejects the first-guard explanation. Prove It Works requires
actual guard counters and exact state/work checks before interpreting timings.
