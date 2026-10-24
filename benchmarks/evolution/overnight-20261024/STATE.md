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

The exact previously confirmed tile-parallel shader passed three fresh seeded
replays and is adopted as the engine baseline. Its record is
[00-tile-parallel-baseline.md](00-tile-parallel-baseline.md). The resident
comparison example has passed its CPU checks and release build; real A/A and
candidate verification follow. Earlier live-preview results retain their
original dispositions and are outside the new score.
