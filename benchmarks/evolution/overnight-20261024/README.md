# Overnight engine performance work

Evolution settings, mutation probabilities, images, resolutions, precision,
selection rules and RNG streams remain fixed. Only engine implementation changes
were adopted. Each accepted change passed the full original-corpus and distinct
mature-state confirmation, exact replay, cached framebuffer L1 and fitness checks,
completed-work accounting and separate overflow qualification.

| Accepted change | Commit | Training ratio | Reserved ratio |
| --- | --- | ---: | ---: |
| 512-lane local-count binner | `9e9727c` | 1.362658 | 1.386012 |
| Decoded raster geometry cache | `afde923` | 1.235967 | 1.238173 |
| 256 physical raster lanes, same logical tile | `de680df` | 1.063708 | 1.062636 |
| One offspring per cooperative mutation group | `fccc4e0` | 1.101808 | 1.109802 |

Each row compares against its preceding implementation. These ratios are not
multiplied into a combined claim. Reports 03, 05, 11 and 13 retain the full rates,
ranges, conditional intervals, source identities, work counters and limits.
A fresh direct comparison measures **4.58 times original throughput**,
with conditional 95% interval [4.52, 4.65], across five alternating
pairs per project. All 33 workers pass with exact canonical outputs and zero
reported errors. [Report 22](22-original-to-final.md) records the rates, audit,
build identities and the unchanged broader policy-1 INCONCLUSIVE disposition.

Forty-four source attempts are retained under `attempts/`, including rejected
capacity designs, short filters and unadopted combinations. Rejected raster
compaction and direct tile grouping were removed from runtime. The remaining
triangle-edge candidate passed a short filter and numerical fixtures, but its
full confirmation stopped during an interruption. It remains unadopted. Report
21 records the partial evidence without a full-budget score.

## Fast local shader checks

The resident `examples/engine-lab.rs` loop reuses the device, buffers and compiled
shader handles. It supports binning, raster and mutation sources. Up to sixteen
independent variants can share one reference qualification per case and budget.
Compilation, reset, numerical qualification and reporting stay outside timing.
Ordinary evolution, drawing readback and checked GPU completion stay inside it.

The existing immutable local release can test a WGSL change without another
Cargo build. From the repository root, choose an unused run name and supply the
candidate's real absolute path:

```sh
ENGINE_LAB_BUILD=build-v11 python3 -B \
  .cache/evolution-overnight-20261024/run-lab.py my-raster-screen \
  --shader-stage raster \
  --reference-shader src/shaders/rasterize_error.wgsl \
  --variant trial=/absolute/path/to/candidate.wgsl \
  --pairs 2 --batches 8 --diagnostics true
```

This local wrapper uses the frozen mature corpus, an isolated working directory,
the shared GPU lock and an outer timeout. It keeps raw rows and errors under the
named cache directory. The following offline command applies the frozen filter:

```sh
python3 -B .cache/evolution-overnight-20261024/analyze-lab.py \
  .cache/evolution-overnight-20261024/my-raster-screen/output
```

A short pass is a reason for confirmation, not adoption. A failed correctness,
work or overflow check never becomes a fast timing sample. Reference/reference
and deliberately broken-source runs established the loop's basic behavior.
Rebuild and archive a new executable for Rust changes; the archived build cannot
test code written after it. The portable CLI, compiler-cache contracts and proof
history are described in [engine-lab.md](engine-lab.md).

Binning candidates also have a source-hot full-buffer checker. Its immutable
local executable is `hot-bin-checker/build/check-gpu-binning` under the same
campaign cache. `--scope full-buffers --shader PATH --output NEWDIR` runs 835
paired checks without rebuilding. Run it on the main thread, in a private working
directory, under the shared benchmark lock and an outer timeout. Its positive and
broken-source proofs are recorded in report 17.

GPU jobs run serially. Avoid Cargo, large CPU audits and other owned heavy work
while scoring. Use the actual Cargo-reported executable and retain its hash;
shared target paths can contain binaries from a different experimental project.
Background isolation remains unverified, and performance conclusions apply to
the recorded fixtures, backend and budgets.
