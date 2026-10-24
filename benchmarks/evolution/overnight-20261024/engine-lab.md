# Resident engine comparisons

`examples/engine-lab.rs` retains one GPU device and its buffers per case while
switching among compiled shader handles. Compilation, seeded reset, two warmup
batches, snapshots, numerical qualification and report writes are outside the
engine timer. Ordinary batch work, normal drawing readback and the final checked
drain are inside it. Every trial records submitted and completed work and errors.

The reference is numerically qualified once per case and budget with the existing
independent renderer policy. Each timed execution must then match all seven
canonical state fields exactly at both endpoints. It also checks raw framebuffer
RGB L1, fitness association, all work counters and timestamp sample counts.
Reporting-only best-chain IDs remain excluded from canonical replay.

Run the release executable on the main thread under a parent timeout, in an
isolated working directory while holding `.cache/evolution-bench/benchmark.lock`.
Use absolute paths for the manifest, sources, executable and new output directory.
For example, the executable accepts these arguments:

```text
--manifest MANIFEST --reference-shader REFERENCE_WGSL
--variant name=CANDIDATE_WGSL --case all --seed 7
--pairs 2 --batches 8 --output NEW_DIRECTORY
```

The manifest's complete parameters remain fixed. The example writes an incremental
`summary.json`, one row per timed trial in `trials.jsonl`, copied shader sources
and separate reference qualification artifacts. A failed check stops the job;
invalid rows have no throughput score. It accepts up to sixteen variants. Each
comparison alternates reference/candidate order across pairs.

`--shader-stage bin|raster` selects which pipeline the supplied sources replace;
the default is `bin`. `--diagnostics true` adds one untimed overflow replay per
source and case. Those rows have null throughput, separate counters and saved
snapshots. They must match the same qualified trajectory and remain below the
256-entry bin capacity. Diagnostics instrument the actual installed binner,
including variants that accumulate a local count and store it once.

## Initial physical proof

The first resident A/A used identical tile-parallel shader bytes on both sides,
five pairs per project, eight batches, seed 7 and two warmups. All thirty timed
trials passed, completing 3,932,160 evaluations. The three qualified initial and
final hashes also matched fresh executions of the archived candidate 05 binary.

The equal-project geometric mean A/A ratio was 1.00237, with a conditional 95%
interval of 0.99859 to 1.00711. Per-case median paired ratios were 0.99892 for ff,
1.00024 for mona and 1.00108 for darwin. Every slow pair is retained. This is
observed repeatability for this short session, not a future noise ceiling.
Small candidate differences still need the frozen full-budget confirmation.

A separate deliberately broken shader reset all bins without appending polygons.
The actual source switch failed the initial state and fitness gates, stopped the
job and returned a null candidate throughput score. It did not qualify as a fast
sample. Its valid reference work remains retained.

Evidence is under `.cache/evolution-overnight-20261024/aa-resident-01/` and
`negative-source-gate/`. Both use the archived executable, source inventory and
Cargo artifact record under `.cache/evolution-overnight-20261024/build/`.
The first failed launcher setup is retained as `aa-resident/`; it ran no GPU work.

The expanded loop subsequently passed a reference/reference integration check
with two normal trials and two overflow replays on ff. Normal work and diagnostic
work each completed 262,144 evaluations. Occupancy was 66 and all replay, numeric
and device checks passed. The evidence is in `v2-overflow-smoke/`. Its timings are
integration evidence, not a candidate comparison. The expanded full-bin checker
also passed 832 paired comparisons and 1,664 dispatches in
`baseline-full-buffers/`.

The retained focused CPU checks and two independent source reviews cover the
installation caches, workgroup recreation, timer and error gates. Existing
production GPU waits are bounded by the supervisor's process timeout. The
pre-existing large pipeline constructor remains intact to keep resource order
and layout unchanged. Its added fields are empty caches and optional overrides;
splitting that constructor belongs to a separate validated refactor. The lab's
64-line main and 66-line schema test are mostly typed data declarations and one
cohesive result path. Other new operations stay within a page. Wgpu allocation
inside ordinary engine work remains included in elapsed time.
