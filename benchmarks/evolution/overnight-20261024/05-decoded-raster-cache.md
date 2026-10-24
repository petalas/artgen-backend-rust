# Keep decoded tile geometry in the raster cache

The rasterizer now decodes each tile polygon's vertices and computes its AABB
once during the cooperative load. Pixel lanes reuse those values and retain the
original edge tests, color decoding, blend order, channel quantization and error
reduction. The cache contains at most 256 entries with a 48-byte stride, using
12,288 bytes. Every consumed entry has one writer before the uniform barrier;
partial-image lanes still load and synchronize. No cache reuse follows blending.

This is a throughput change with all evolution parameters fixed. Both sides use
the committed 512-lane local-count binner. The reference raster is the unchanged
original `7bacb583…`; the kept source is `a292287a…`. Earlier cache filters and the
independent decoded-vertices-only alternative remain separate observations.

The two-pair, eight-batch filter against cooperative packed loading measured
1.11194 for decoded vertices plus AABBs and 1.10212 for vertices alone. Both
passed exact replay, but only the leading complete cache advanced. Patches 13
and 14 preserve those isolated candidates; patch 14 is unadopted.

Full confirmation froze five alternating pairs per project at 64 batches and
two warmups, using seed 7 on training and seeds 11/13/17 on the distinct reserved
ff/mona/darwin drawings. Each measured worker completed 1,048,576 evaluations.
No pair was excluded or added. Timing includes ordinary readback and the final
checked drain, while compilation, reset, warmup and extra verification stay outside.

Training aggregate ratio: **1.235967**, conditional 95% interval
**[1.232809, 1.239085]**. Reserved aggregate ratio: **1.238173**, interval
**[1.234483, 1.241748]**. All project median and lower-bound gates pass the frozen
engine-only policy. These are equal-weight geometric means of paired throughput
ratios, with 10,000 stratified log-bootstrap draws and seed 20261024.

| Corpus/project | Reference eval/s median (range) | Candidate eval/s median (range) |
| --- | ---: | ---: |
| Training ff | 213,927 (208,847–219,512) | 264,066 (256,540–271,764) |
| Training mona | 175,293 (174,204–176,512) | 216,918 (214,601–218,859) |
| Training darwin | 176,132 (175,549–177,736) | 217,409 (215,620–218,265) |
| Reserved ff | 204,123 (202,364–208,944) | 251,393 (248,129–257,457) |
| Reserved mona | 177,992 (174,466–180,543) | 219,831 (215,544–223,579) |
| Reserved darwin | 180,229 (178,609–185,485) | 223,566 (219,072–229,449) |

The 60 normal workers completed 62,914,560 evaluations. Twelve separate overflow
replays completed 12,582,912 evaluations and contributed no timing samples.
Six scalar-qualification workers completed another 6,291,456 evaluations. All
had zero numerical, canonical-state, work-accounting, process and device errors.
An independent offline audit reconstructed all endpoint associations and 144
framebuffer RGB-byte L1 totals, including the original captured trajectories.
Cached totals and fitness remained exact. The actual candidate also passed the
main-thread GPU rasterization checker, including mixed reference bytes, partial
tiles, blending and the visible-layer regression. Production Naga accepted every
supported raster workgroup. Source review found no material issue.

Sampled raster intervals moved from 0.741/0.911/0.876 ms to
0.511/0.629/0.604 ms on the training projects. Those timestamps cover the final
iteration of each batch; they support attribution and do not establish whole-run
shares, hardware utilization or a universal backend result.

Evidence is retained under `.cache/evolution-overnight-20261024/`:
`raster-decoded-screen-02`, `raster-train-full`, `raster-reserved-{ff,mona,darwin}`,
and `raster-confirmation/{spec.json,outcome.json,independent-audit.json}`.
The release binary and source inventory are archived in `build-v3/`.
Background isolation remains unverified. Bootstrap uncertainty is conditional on
the recorded seed trajectories and timing session. This keep decision applies to
the engine-only policy; older live and campaign dispositions remain unchanged.
No combined speedup is inferred by multiplying separate experiment ratios.

Power of Ten retains the existing cohesive raster `main` rather than splitting
its floating-point path during this change; its body becomes shorter. All new
storage is bounded by the existing tile clamp. Prove It Works required physical
GPU checks and exact saved-state/L1 replay before keeping the optimization.
