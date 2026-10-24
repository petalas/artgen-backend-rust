# Reject packed bounds in the wide binner

REJECT. The independent packed-cache candidate gives aggregate ratio
0.994871465250, with exploratory conditional interval
[0.989725178178, 1.000044511539].
It did not clear the fixed short practical gate. Production remains binner 598.

Only the shared bounds representation changes: four u32 coordinates become
two u32 words, each storing two 16-bit integers. The requested device texture
limit is 2048; every supported tile axis is at least eight pixels, and the
original helper clamps upper bounds to the grid. All four outputs therefore
fit within 0..256. No valid integer bound loses a bit. Source-declared shared
storage falls from 16,000 to 8,000 bytes. The original float helper, WG512,
barrier, polygon order, uncapped count, first-256 writes and stale suffixes
remain unchanged. This is not float16 or a resolution specialization.

The predeclared screen used two alternating pairs per case, eight batches,
two warmups and seed 7. All twelve normal trials, six separate overflow replays
and three reference qualifications passed exact state, raw-L1, fitness, work,
device and capacity checks. Normal work was 1,572,864 evaluations; diagnostic
work was 786,432 with null rates; qualification added 393,216. No pair was
excluded, and no full-budget or reserved run followed.

The candidate subsequently passed the new source-hot binning proof as a
checker-integration case: 835 full-buffer comparisons and 1,670 dispatches.
That broadens its recorded correctness evidence, not its performance verdict.
Four extraction operations, shared-memory traffic and compiler choices remain
unmeasured attribution possibilities; smaller source storage did not produce
a useful engine gain on this screen.

The exact source SHA is
0aba8b878dbe081d044a965005369a0cf79c22523146b3079be6e0f26b02632b.
Evidence under .cache/evolution-overnight-20261024/: bin-packed512/ has the
generator/proof/hashes, bin-packed512-screen-spec.json froze the schedule,
bin-packed512-screen-01/output/ retains raw rows/ranges/stage samples, and
hot-bin-checker/gpu/packed-positive/ records the direct buffer proof.
Timing uses immutable build-v9 with kept G1/AE1 defaults. Patch 40 applies to
598. Background isolation remains unverified.
