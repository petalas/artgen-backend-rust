# Keep G1 mutation at 64 physical lanes

Both physical-width alternatives failed the preliminary practical gate. The
kept G1 shader retains 64 physical and copy lanes per workgroup. Logical group
width remains one in every arm; lambda, offspring IDs, RNG branches, default
zero initialization and the entire mutation parameter object remain unchanged.

| Attempt | Mechanism | Aggregate ratio | Conditional 95% interval | Decision |
| --- | --- | ---: | --- | --- |
| 37 | 32 physical/copy lanes | 1.000559679 | [0.996135795, 1.005003210] | REJECT |
| 38 | 128 physical/copy lanes | 1.000530926 | [0.995736593, 1.005348344] | REJECT |

Each source changes exactly the two physical-width/copy-stride declarations.
The host still dispatches [active, lambda, 1]. Unique loader/copy ownership and
both unconditional barriers remain; only leader zero enters the original body.
Maximum prefix-loop iterations per lane become 32 and eight instead of sixteen.
Shared storage is unchanged, including the empty-parent zeroing contract.

The pre-data screen used two alternating pairs per project, eight measured
batches, two warmups and seed 7. All 24 normal trials, nine separate overflow
replays and three reference qualifications passed exact canonical/L1/fitness,
complete work, device and capacity gates. Normal work was 3,145,728 evaluations;
diagnostic work was 1,179,648 with null rates; qualification added 393,216.
No pair was excluded. Neither candidate received full or reserved runs.

The near-one ratios provide no useful throughput gain on this screen. They do
not establish a future noise ceiling or a hardware-occupancy explanation. The
copy/idle-lane tradeoff was a hypothesis, not a measured limiter. Binner 598
and raster AE1 remain fixed. No evolution parameter was tuned.

Evidence under .cache/evolution-overnight-20261024/: source-only generators,
hashes and ownership proof in mutation-g1-lanes/; frozen schedule in
mutation-g1-lanes-screen-spec.json; raw rows, medians/ranges and stage samples
in mutation-g1-lanes-screen-01/output/, using immutable build-v9/. Patches
37-38 apply independently to kept G1. The 368-case proof qualifies G1 at
64 lanes; it was not rerun for these rejected widths. Background isolation
remains unverified.
