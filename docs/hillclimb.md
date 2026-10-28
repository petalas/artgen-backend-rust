# Hill-climbing engine throughput

`scripts/hillclimb.py` is the inner loop for engine performance work: one command
builds two sides, alternates timed trials on the frozen v1 fixtures, checks
correctness and replay identity, prints a verdict and appends a row to
[`benchmarks/hillclimb/log.tsv`](../benchmarks/hillclimb/log.tsv).

```sh
scripts/hillclimb.py --note "what I changed"          # working tree vs HEAD
scripts/hillclimb.py --cases ff --pairs 3 --batches 16  # quick screen, ~1 min
scripts/hillclimb.py --control HEAD~1 --candidate HEAD  # two commits
scripts/hillclimb.py --candidate HEAD                   # A/A noise check
```

Each side is a git ref, `WORKTREE` (uncommitted tree, default candidate) or a
prebuilt `evolution-bench` executable. Fixtures come from
`scripts/evolution-bench.py freeze`.

## Why it is fast

Engine construction costs 11–15 s per process. `perf` attributes 83% of it to the
NVIDIA shader compiler behind Dozen (`libnvidia-gpucomp.so`); the wgpu pipeline
cache saves only ~5 s of it. A 64-batch trial times ~3.2 s of work. So a fresh
process per trial spends about 80% of its wall time compiling.

`evolution-bench serve` keeps one engine per case and runs trials from stdin
(`@result {json}` replies). `prepare_seeded_benchmark` restarts each trial from
the seeded fixture, and resident trials reproduce the fresh-process replay hash.
A resident trial takes ~4.7 s of wall time instead of 16–21 s.

- Builds are cached by git tree hash under `.cache/hillclimb/bin` (newest 8 kept).
  Refs build in one reusable worktree (`.cache/hillclimb/ref-src`) with its own
  incremental target dir; `WORKTREE` builds in the normal `target/`.
- Executables built before `serve` existed fall back to one process per trial.
- Runs take the shared `.cache/evolution-bench/benchmark.lock`.

## Gates and verdict

- Every trial must complete all work with zero numerical errors.
- Every trial of a case must end in the same canonical replay hash on both sides.
  A change meant to alter evolution passes `--allow-behavior-change`.
- Verdict uses the equal-case geometric mean of paired ratios with a stratified
  bootstrap 95% interval. `FASTER`/`SLOWER` need the interval to exclude 1.0 and
  a move of at least `--min-effect` (default 0.5%). Fewer than three pairs give
  `SCREEN`, which is a direction, not a result.
- Output also shows sampled GPU µs per batch for mutate/bin/raster/select, so a
  kernel change can be told apart from a host-side change.

## Noise

Measured on a quiet host (no game or video on the Windows side; host load
produced a false 3.4% SLOWER in an earlier A/A):

- Default A/A (3 cases × 5 pairs × 64 batches, ~4 minutes): 0.9975, 95%
  [0.9947, 1.0003], NOISE. mona and darwin pairs stay within ±0.5%; ff pairs
  spread 0.975–1.002 and are the noisiest case.
- An idle second server does not bias the first (single-process A/A 0.994–0.998,
  with an idle sibling 0.997–0.998).
- Throughput drifts down inside every process as it runs more batches, from the
  driver's submit path, not the kernels (see LEARNINGS, "Host submit time grows").
  Both servers run the same batches at each pair, so AB/BA mostly cancels it.
  Compare only within one run; absolute evals/s from a long run read low.
- Below ~1% effects, rerun before trusting a verdict, and run an A/A when the
  machine state changes.

## Relation to the campaign protocol

This is a screen. Adoption still uses the reserved-seed confirmation, audits
and reports described in `benchmarks/evolution/`. Shader-only variants can also
be screened in one process with `examples/engine-lab.rs`.
