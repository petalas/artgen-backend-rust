# Binner follow-up filters

The next five candidates used the unchanged 32-lane tile-parallel reference,
two alternating pairs per project at eight batches, seed 7 and two warmups.
All sixty normal trials and eighteen untimed overflow replays passed exact state,
byte-L1, fitness, work and device checks. Timed work was 7,864,320 evaluations;
2,359,296 diagnostic evaluations are separate and excluded from timing statistics.

| Attempt | Implementation | Preliminary aggregate ratio | Filter disposition |
| --- | --- | ---: | --- |
| 05 | Internal bin width 16 | 0.82046 | Reject at the declared budget |
| 06 | Internal bin width 512 | 1.33605 | Compare leading configurations |
| 07 | Two packed integers instead of four cached tile bounds | 0.99695 | Reject at the declared budget |
| 08 | Skip unconsumed tile owners after the bounds barrier | 1.00297 | Reject below 1.005 promotion threshold |
| 09 | Width 256 with tile-local counting | 1.34748 | Compare leading configurations |

These are filters, not acceptance results. Every row remains in the evidence;
there are no sample exclusions or extensions. The local-counter/width combination
was tested only after both mechanisms had been measured separately. A further
direct comparison selects one implementation for full-budget confirmation and
reserved-state validation. None of these new candidates is installed by this record.

Sampled bin intervals explain why the broad width/count changes are promising.
Attempt 09's per-project median sampled bin intervals were approximately
0.063/0.081/0.054 ms for ff/mona/darwin, against 0.471/0.597/0.515 ms in its
paired references. Packed bounds did not reduce these intervals; dirty-owner
skipping changed them little. Less source work did not imply a throughput win.
These samples cover the last iteration of each batch, not the whole GPU run.

The packed cache uses two 16-bit values per word. The requested texture-dimension
limit bounds every cached tile coordinate well below 65,536. Original floating
point bounds remain unchanged. The dirty-owner candidate uses the rasterizer's
literal strict overlap predicate after the shared-cache barrier. It intentionally
leaves unread bins stale, so its focused proof requires consumed-bin equality.
The other candidates retain full-buffer equality contracts. Candidate-specific
focused GPU checks are still required for any accepted implementation.

Patches `attempts/05-wg16.patch` through `attempts/09-local-count256.patch` apply
independently to the committed baseline. Source proofs and SHA inventories are
under `.cache/evolution-overnight-20261024/bin-followups/` and `variants/`.
Raw outcomes, diagnostic snapshots, complete distributions and stage samples are
under `bin-screen-02/`, including `output/analysis.json`. No full-budget result
or broader quality claim follows from these two-pair filters.
