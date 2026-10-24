# Keep the 512-lane local-count binner

The engine-only campaign keeps internal bin workgroups of 512 lanes and one
tile-local occupancy counter followed by a final count store. This improves
completed engine evaluations per second over the adopted 32-lane tile-parallel
baseline. The complete evolution parameters, raster tile size, genotype,
precision, fitness, selection and RNG rules remain unchanged.

Each polygon-bound slot has one writer before the uniform workgroup barrier.
Each tile then has one owner scanning polygons in ascending order. Its local
counter counts every overlap, including those above 256, while storing the same
ordered first 256 indices. Empty tiles store zero; unused index words remain
untouched. The 16,000-byte cache and default memory initialization are unchanged.

## Full-budget results

Both frozen corpora passed all adoption gates. Each corpus has five alternating
pairs per project, 64 batches and 1,048,576 completed evaluations per trial.
Every slow sample remains included.

| Corpus | Aggregate throughput ratio | Conditional 95% interval | Normal trials |
| --- | ---: | --- | ---: |
| Original mature ff/mona/darwin, seed 7 | 1.36266 | 1.35871 to 1.36621 | 30 |
| Reserved mature states, seeds 11/13/17 | 1.38601 | 1.38223 to 1.38991 | 30 |

The aggregate is the equal-project geometric mean of paired ratios. Intervals
use the frozen stratified log bootstrap, 10,000 draws and seed 20261024. Every
per-case median and lower-bound non-regression gate passed. These are separate
confirmations rather than pooled timing samples.

Reserved-case rates below are evaluations per second, median and range of five
runs per side. The initial states remain above 97% fitness.

| Project | 32-lane reference | 512-lane local count |
| --- | --- | --- |
| ff | 145,381; 143,330 to 148,117 | 201,310; 197,602 to 205,196 |
| mona | 126,273; 124,979 to 128,647 | 176,229; 173,969 to 179,814 |
| darwin | 131,371; 130,682 to 133,525 | 180,362; 177,780 to 182,456 |

The isolated width and local-count screens preceded composition. The final
direct short comparison selected 512 lanes with local counting over 256 lanes
with local counting. Sampled bin intervals decreased alongside whole-engine
time. Stage timestamps cover only the last iteration of each batch; they support
the binning mechanism without measuring utilization or a whole-run bin share.

## Correctness and identities

All sixty normal trials and twelve overflow replays passed exact seven-field
initial/final replay, framebuffer-byte L1, fitness, work-counter and device gates.
Normal score work totals 62,914,560 evaluations. Diagnostic work totals
12,582,912 evaluations and is excluded from timing statistics. Six reference
qualifications add 6,291,456 separately recorded evaluations.

The independent audit recomputed 144 framebuffer RGB L1 totals, all saved
canonical endpoints and their associations with every normal row. Initial and
final hashes also match the historical original and reserved trajectories.
The selected source passed all 835 full-buffer GPU comparisons, including
occupancy above 256 and 512-lane boundary cases. All checks reported zero errors.

Measured shader SHA-256:
`b851f25680f2a61110cddd8683b1cafea221b683a1ea822c90a46bba9c5522da`.
The production source corrects one stale header comment and has SHA-256
`598b32eab7ae998f2d644b36a5d71d17997772ed06dbd5c0b726130ef51d13d9`.
An explicit comparison proves every executable WGSL token remains identical;
the proof is retained as `binner-confirmation/comment-equivalence.json`.

Frozen specification, build identities, direct-comparison results, full-buffer
logs, aggregate analysis and independent audit are retained under
`.cache/evolution-overnight-20261024/`, chiefly `bin-leader-screen-03/`,
`binner-train-full/`, `binner-reserved-{ff,mona,darwin}/` and
`binner-confirmation/`. The audit is `binner-confirmation/independent-audit.json`.
Its first attempt used the ordinary final-state filename for diagnostic files;
that failed audit is retained, and the corrected audit reads `final-state.json`.

This evidence covers mature drawing restarts on the recorded RTX 5090 through
Vulkan/Dozen. Repetitions share a seed within each case; reserved states and seeds
are distinct. Background isolation remains unverified. The next target is the
rasterizer's shared cache and repeated vertex decoding, with the same fixed work.
