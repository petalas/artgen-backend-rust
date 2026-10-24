# Adopt the confirmed engine baseline

The engine-only campaign adopts the unchanged tile-parallel binner from earlier
candidate 05. Each of 32 lanes computes polygon bounds; each tile has one owner
that appends polygon indices in their original order. Evolution parameters,
rendering arithmetic, raw occupancy counts and the first 256 ordered entries
remain unchanged. Automatic shared-memory initialization remains enabled.

Shader SHA-256 is
`a219175f3fd395600f0ecbef3ac099b7450aeaf972e1e5c91dfc5dcde1fb901b`.

Earlier full-budget confirmation on distinct mature states and seeds measured
2.30456 times the original engine throughput, with a conditional 95% interval
of 2.29968 to 2.30992. All 45 workers passed canonical replay, numerical and
completed-work gates. See the retained
`.cache/evolution-qualification/acceptance-07/engine/summary.json` and the
candidate 05 report in `benchmarks/evolution/v1/experiments/05-tile-parallel-binning.md`.

Before adoption, three fresh eight-batch replays on the original ff, mona and
darwin corpus matched the previously saved trajectories exactly. Each completed
131,072 evaluations with zero errors. Their timings are excluded from new
speedup claims because setup work could overlap them. Evidence is retained in
`.cache/evolution-overnight-20261024/baseline-recheck/summary.json`.

The current release build also passed. Earlier full-bin-buffer and rasterization
checks remain applicable to these identical shader bytes. This adoption follows
the user's new engine-only scope. It does not change the disposition of the
earlier live qualification or claim that its separate gates passed.
