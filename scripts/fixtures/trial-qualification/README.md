# Native qualification record

`qualification.json` is a byte-for-byte copy of the retained native
`campaign/aa/output/qualification-ff/result.json` under
`.cache/evolution-qualification-prepared-20261024/`. Its SHA-256 is
`2e80e8fd4c6da54705ef70d0436290692a75beb5877cd4289db7e73696656ad9`.
`case.json` contains the dimensions, complexity, parameters and case identifier
from that campaign's frozen manifest, normalized for the supervisor interface.

This record has `diagnostic:false` and `evals_per_second:null`. It proves the
qualification producer's shape. It is not an accepted timing score. The original
campaign remains inconclusive, as recorded in
`benchmarks/evolution/continuation-20261024/14-prepared-control-inconclusive.md`.
Tests mutate copies in memory. They never run the campaign or modify its source
artifacts.
