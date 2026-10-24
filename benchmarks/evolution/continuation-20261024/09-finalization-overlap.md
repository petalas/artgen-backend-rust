# Finalization overlap failed the short screen

REJECT. Finalizing the compute command buffer before an existing readback wait
preserved correctness but failed the frozen throughput gate. The prototype is
removed. All evolution parameters, shaders and accepted engine behavior remain
unchanged. No full-training or reserved-seed confirmation was run.

The [host-phase observation](08-host-phase-attribution.md) located 390.249 ms
of backend finalization on 21 early-conflict caller batches. That was a
conditional overlap ceiling, not measured savings. The experiment finalized
the independent compute/timestamp prefix before collecting the previous result,
then uploaded the same parameters and control reset. A second command buffer
retained the original control, fitness and drawing-copy order. Both buffers were
submitted together once. Calls without the exact existing conflict retained the
original single-buffer path.

## Correctness passed before timing

The default-off prototype passed source review, 27 source-preservation checks
and 30 focused CPU tests. Reconstructing its false path restores the entire
original engine byte for byte. True-path commands concatenate to the original
sequence; readback, map, unmap, counter and flush bodies are unchanged.

The physical ordinary-API checker completed both timestamp-mode A/A pairs before
either A/B pair. Eight runs preserved every Some/None return, drawing header and
geometry, chain fitness, completed-work counter and terminal flush. Terminal
geometry and fitness match an actual maximum packed parent. Independent offline
audit reproduced 26 canonical snapshots and 104 parent RGB-byte L1/fitness checks.
Maximum CPU-fitness difference was 7.62939453125e-6, within the unchanged 2e-5
policy. No scalar rerender was performed.

## Frozen two-warmup screen

The predeclared screen used two alternating pairs per drawing, eight batches,
seed 7 and two warmups. Twelve timing trials completed 1,572,864 evaluations.
Six separate diagnostics completed 786,432; three qualification workers completed
393,216. All work and correctness gates passed, with zero errors. Diagnostics
and qualification workers were excluded from rates.

Candidate/reference throughput was **0.750009094**, conditional paired-bootstrap
95% interval **[0.565639497, 0.994473766]**. The aggregate misses the 1.005 advance
threshold, and every case median misses its 0.98 guard. All samples remain in
the result.

| Drawing | First pair ratio | Second pair ratio | Median pair ratio |
| --- | ---: | ---: | ---: |
| ff | 0.450432 | 1.132216 | 0.791324 |
| mona | 0.514625 | 1.112633 | 0.813629 |
| darwin | 0.546593 | 1.115149 | 0.830871 |

The first candidate trial is slow for every drawing; the later trial is faster
than its reference. Two repeats do not establish a steady-state gain or a uniform
intrinsic slowdown. The driver/JIT explanation remains unproved. Independent
audit reproduced the complete statistics and passed 675 checks over 209 retained
artifacts. Background isolation remains unverified.

## One separate four-warmup diagnostic

Source inspection places the first possible split at batch 4, conditional on
prior improvements and deferred drawing state. Two warmups may miss it. A private
copy changed only four existing warmup policy/metadata literals from 2 to 4.
Its inverse restores the exact original lab source. No production code changed.

One unscored mona invocation used two pairs, eight batches and four warmups in
both arms. Actual output metadata confirms four warmups in all six timing and
diagnostic rows. Qualification replay and every timed endpoint match the
original two-warmup trajectory. It completed 524,288 timing evaluations,
262,144 diagnostic evaluations and 131,072 qualification evaluations with zero
errors. The original screen and its statistics remain unchanged.

| Diagnostic pair | Reference seconds | Candidate seconds | Throughput ratio |
| --- | ---: | ---: | ---: |
| First | 0.467903525 | 0.928277958 | 0.504055408 |
| Second | 0.477332132 | 0.433279925 | 1.101671470 |

Four warmups did not remove the observed first-trial penalty. Guard execution
during warmup was not instrumented, so this does not identify its cause. No
additional warmup variants, sample exclusions or confirmation runs followed.
This diagnostic cannot promote the rejected screen or establish repeatability.

## Restoration and retained evidence

The three modified runtime/tool files were restored exactly to `6b7c2d5`, whose
runtime also matches `801695a` and `a6eedb1`. The new checker was archived and
removed. All 78 tracked Rust/WGSL files match the accepted sources. After
restoration, scoped Cargo check and 42 CPU tests passed. Both normal benchmark
tools were rebuilt from root and match the accepted v12 binaries byte for byte.
The pre-existing application executable remained untouched.

The exact rejected prototype is retained in
[attempt 04](attempts/04-finalize-overlap-rejected.patch). The frozen specification
and result, including binary and artifact identities, are in
[the specification](profiles/finalize-overlap-spec.json) and
[the result](profiles/finalize-overlap-result.json).
Raw evidence remains under `.cache/evolution-profiling-20261024/` in
`finalize-overlap-85/`, `build-finalize-v1/`, `finalize-api-proof/`,
`finalize-screen/`, `finalize-result-audit-90/`,
`finalize-warmup-diagnostic-88/`, `finalize-rejected-cleanup-89/` and
`restored-default-build/`.
The [full-batch profile chart](profiles/full-batch-profile.html) retains the
accepted-engine GPU-span and wall-time comparison that led to this investigation.

Explain the Number keeps first-use behavior, conditional uncertainty and the
separate diagnostic visible. Test Behavior preserves ordinary return and flush
semantics. Prove It Works requires exact restored sources and completed work.
The next investigation would need to locate the first-trial penalty directly;
these data justify neither changing evolution parameters nor adopting overlap.
