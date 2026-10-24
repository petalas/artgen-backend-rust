# Direct mutation equivalence checks

`examples/check-gpu-mutation.rs` compares supplied WGSL sources through the actual
mutation bind-group layouts and default pipeline options. Run the release example
on the main thread under an outer 600-second timeout. It creates a new evidence
directory, copies and hashes sources, records incremental events, then writes
`result.json`. It emits no performance score.

The 368 fixtures cover every integer lambda from 1 through 64, single/multiple
mutation modes, adaptive mutation on/off, crossover probabilities 0/.2/1, empty
parents, dense boundaries through 1000 polygons and immediate smaller-state history
probes. The 1024-parent binding is retained because tournament selection uses its
array length. Only four parent workgroups dispatch; logical offspring IDs and RNG
streams remain authoritative when a source uses additional physical helper lanes.

Comparison requires identical active headers and polygon prefixes, unchanged error
accumulators, and untouched unproduced slots against each variant's own persistent
history. Inactive polygon padding may differ and is reported. A later active tail
is always checked. The checker does not independently render those drawings; the
resident engine's full numerical/replay gates remain separate.

The leading preliminary copy candidate, WG256 with four copy lanes per child,
passed all 368 comparisons and 736 submitted/completed dispatches with zero
failures. Source SHA is `46452ae3…`; reference is `e13308a7…`. Raw results, complete
events, binary/source identities and process records are retained in
`.cache/evolution-overnight-20261024/mutation-confirmation/numerical/`. The first
throwaway launcher mistakenly expected a `status` field instead of `passed` after
the successful process exit. Its error remains retained; `verified-result.json`
checks the original result and event totals. No GPU work was repeated to repair
aggregation.

Focused Cargo check, four unit tests, help and fifteen invalid-input CLI cases
passed. All/pedantic Clippy reports zero diagnostics in the checker. Independent
source inspection found no material issue. Fixed buffer capacities, bounded reads,
outer/native-call timeout requirements and framework-allocation deviations are
documented beside the tool. This physical proof establishes mutation equivalence
for these finite fixtures, not a throughput gain or candidate adoption.
