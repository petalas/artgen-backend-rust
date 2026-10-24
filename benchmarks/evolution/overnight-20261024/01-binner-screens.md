# First four binner screens

These are preliminary filters, not accepted performance changes. Every trial
uses the same mature corpus and complete parameters as the starting baseline.
Each candidate has two alternating pairs per project at eight batches, seed 7
and two warmups. All 48 timed trials passed exact initial/final canonical replay,
framebuffer RGB L1, fitness, completed-work and device-error gates. They completed
6,291,456 evaluations. No pair was excluded.

| Attempt | Isolated implementation change | Aggregate candidate/reference ratio | Next decision |
| --- | --- | ---: | --- |
| 01 | Bin workgroup 32 to 64 lanes, both striped strides updated | 1.15821 | Width sweep passed; compare with leading width |
| 02 | Bin workgroup 32 to 128 lanes, both striped strides updated | 1.25078 | Width sweep passed; compare with leading width |
| 03 | Bin workgroup 32 to 256 lanes, both striped strides updated | 1.31768 | Leading measured width; full-budget qualification required |
| 04 | Tile-local raw occupancy, one final atomic count store | 1.17748 | Test composition with the leading width |

The ratio is the equal-project geometric mean of paired throughput ratios.
Two pairs cannot support adoption. Full-budget confirmation and distinct-state,
disjoint-seed validation remain required by `policy.md`.

Sampled bin intervals decreased on all three projects. For example, ff's median
last-iteration bin interval changed from approximately 0.471 ms to 0.081 ms in
attempt 03. This supports the intended binning mechanism; it is not whole-run
GPU utilization or a separately measured hardware limit. The earlier causal
bin-body experiment remains the evidence that bin work affects completion time.

The exact patches are in `attempts/01-wg64.patch` through
`attempts/04-local-count32.patch`. They apply independently to the committed
32-lane tile-parallel baseline. Bounds arithmetic, layer order, occupancy above
256 and untouched data suffixes remain unchanged by source inspection. Before
acceptance, candidate-specific full-buffer and overflow replays remain required.

Raw source identities, launch/build provenance, qualification states and all rows
are retained in `.cache/evolution-overnight-20261024/bin-screen-01/`. The complete
paired rates, medians, ranges and stage samples are in `output/analysis.json`.
An earlier analysis label incorrectly evaluated confirmation thresholds without
requiring the full budget. It is retained as `analysis-v1-threshold-labels.json`;
the corrected analysis marks every preliminary result ineligible for confirmation.
No raw measurements or frozen thresholds changed.
