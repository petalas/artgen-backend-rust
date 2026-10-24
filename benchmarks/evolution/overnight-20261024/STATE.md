# Campaign checkpoint

Scope and fixed gates are in [policy.md](policy.md). The user authorized engine
throughput work until morning, fixed evolution parameters and frequent commits.

- Starting commit: `374a470`, containing the completed live diagnostic checkpoint.
- Stop after 07:00 UTC on 2026-10-24, 08:00 Europe/London.
- Mutable state: `.cache/evolution-overnight-20261024/state.json`.
- Full local decision trail: `.cache/hillclimb/decisions.tsv`.
- GPU jobs run serially under `.cache/evolution-bench/benchmark.lock`.
- Training uses the original ff/mona/darwin mature corpus; reserved confirmation
  uses the distinct mature corpus and seeds 11/13/17.

The starting tile-parallel baseline is recorded in
[00-tile-parallel-baseline.md](00-tile-parallel-baseline.md). The resident loop
passed physical A/A, deliberately invalid-source and overflow proofs. Ten
binner configurations have been screened. The retained 512-lane local-count
implementation passed full original and reserved confirmations, measuring
1.36266 and 1.38601 times the starting fast baseline's throughput. See
[03-512-lane-local-count.md](03-512-lane-local-count.md).

The decoded tile geometry cache also passed full training and reserved confirmation,
measuring 1.23597 and 1.23817 times its preceding engine throughput. See
[05-decoded-raster-cache.md](05-decoded-raster-cache.md).

The fixed-tile 256-lane raster also passed training and reserved confirmation,
measuring 1.06371 and 1.06264 times decoded C throughput. It preserves the logical
32×16 tile and every pixel calculation; see
[11-fixed-tile-physical-lanes.md](11-fixed-tile-physical-lanes.md).

Both compact-raster prefix experiments failed their short filters and their
runtime path was removed. One logical offspring per mutation workgroup also passed full confirmation,
measuring 1.10181 and 1.10980 times the preceding engine throughput. It is now
the default; see [13-one-offspring-workgroups.md](13-one-offspring-workgroups.md).
Every evolution setting remains pinned. Conservative masks, cached alpha, small
binning operations, packed bounds and alternate G1 lane widths failed their short
filters. Matched compilation options did not rescue indirect raster compaction.
Direct adjacent-tile grouping also failed and was removed in `fe5e829`; its fresh
coverage, fault and host-edge proofs remain archived in
[report 20](20-direct-raster-coarsening.md). Two triangle-edge microexperiments passed their short filter. The sole selected
candidate passed numerical fixtures, but its full confirmation was interrupted
before completion. It remains unadopted; see [report 21](21-raster-edge-micros.md).
After recovery, the accepted defaults are being compared directly against the
original archived engine. No partial candidate timings are promoted.
Earlier live-preview results retain their original dispositions and are outside
this engine-only score. Work continues until the morning cutoff.
