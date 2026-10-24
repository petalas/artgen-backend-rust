# Keep the existing reference-byte conversion

REJECT the source-only reference-packing candidate at the frozen short filter.
Throughput ratio is 0.992301398, conditional 95% interval
[0.981884425, 1.002828887]. It fails the 1.005 aggregate gate. Mona's paired median
also fails the 0.98 case guard. There is no full-budget or reserved extension,
and production AE1 stays unchanged.

| Case | Paired median | Paired range |
| --- | ---: | --- |
| ff | 1.000440218 | [0.997161490, 1.003718946] |
| mona | 0.975575745 | [0.947893851, 1.003257640] |
| darwin | 1.001516280 | [0.998399076, 1.004633483] |

The candidate replaces only three `round(ref_color.channel * 255.0)` reference
assignments with `pack4x8unorm(ref_color)`, byte extraction and exact u8-to-f32
conversion. It preserves texture format/upload, bindings, initialization, blend
math, downstream old/new L1 arithmetic, reductions and dispatch. A Uint texture
would require host and layout changes, so it was not implemented.

The scalar source proof exhausts all 256 original bytes, adjacent f32 rounding
directions and a wider neighbor envelope, covering 1,786 normalized values and
3,580 multiply/add paths. Both conversions recover the same original byte. This
proves the arithmetic premise, not driver execution or a speedup. Builtin lowering
can emit the same work or additional work.

The existing immutable v12 resident lab loads the two WGSL files through its
raster installer. No Rust source edit, new option, interface or build is needed.
The screen completes in 98.683 seconds including setup and verification; this
is supervisor duration, not the scored evolution metric. The measured loop
includes normal readback and checked drain, excluding preparation and snapshots.

Two alternating pairs, eight batches, seed 7 and two warmups run on all three
original projects. All twelve timing trials, six overflow replays and three
reference qualifications pass exact seven-field state, integer byte-L1, fitness,
work and device checks. Actual rows record G1 and TRUE mutation initialization,
with the expected distinct reference/candidate source hashes. Normal work is
1,572,864 evaluations; diagnostic work is 786,432 and qualification work is
393,216, excluded from statistics. Parameters remain byte-identical.

The offline parent audit reuses the previously independent audit's numerical
functions, reproduces the paired bootstrap separately, and verifies 18 snapshots
and 72 parent byte-L1 totals. All 294 checks pass. Maximum CPU-fitness difference
is 7.62939453125e-6, within the unchanged 2e-5 policy. Forty output files are
hashed. Existing scalar-quality records are checked; scalar rasterization is not
rerun. A standalone exhaustive-byte, all-five-workgroup GPU checker was not built
for this failed candidate.

Every slow pair is retained, including mona's 0.947893851 pair. Background
isolation remains unverified. The short screen cannot distinguish intrinsic
builtin cost from uncontrolled variation or establish a hardware limiter. It
provides no demonstrated whole-engine benefit, so the candidate stops here.

Candidate SHA256 is
0689e08a0b0ad119a4986c3b91083bbe6c8f55e22e0ea671a63f6ce39420c02c.
Reference SHA256 is the unchanged AE1 hash ae1fd75f.... Evidence is under
`.cache/evolution-continuation-20261024/reference-bytes-audit-71/`,
`reference-pack-screen/` and `reference-pack-screen-run/`. The exact candidate
patch is retained beside this report's sibling attempts.

Prove It Works required actual compiled replay and byte-L1 checks. Explain the
Number kept total supervisor duration, measured engine throughput and scalar
arithmetic proof separate. The failed gate prevents a longer campaign.
