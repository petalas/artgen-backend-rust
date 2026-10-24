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

The exact previously confirmed tile-parallel shader is installed in the working
tree for an engine recheck. A resident comparison example is being implemented
to amortize device, pipeline and scalar-qualification setup. No new optimization
has been accepted under this campaign yet. Earlier live-preview results retain
their original dispositions and are outside the new score.
