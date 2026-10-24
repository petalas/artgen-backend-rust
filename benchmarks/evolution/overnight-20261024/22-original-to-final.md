# Direct original-to-final engine comparison

The accepted defaults deliver **4.584441 times the original engine's
completed evaluations/second** on the fixed mature ff/mona/darwin corpus. The
conditional 95% paired interval is [4.524493, 4.645300].
This is a fresh direct measurement, not multiplication of component gains.
All five statistical screen gates pass. The unchanged policy-1 tool still labels
broader adoption INCONCLUSIVE; this result does not qualify live previews,
independent image families, clean isolation or a broader campaign.

The run uses five alternating pairs per project, seed 7, 64 measured batches,
64 iterations per batch, four chains, lambda 64 and two matching warmup batches.
Every trial completes 1,048,576 logical evaluations. All mutation probabilities,
complete MutationParams, normalized images, resolutions, precision, objective,
selection and RNG streams remain fixed. Initial active parents all exceed 97%.
Every pair, including slower samples, remains included.

Rates are evaluations/second, reported as median and observed minimum/maximum.
The ratio column is the paired geometric mean, not a ratio of rate medians.

| Case | Original median [min, max] | Final median [min, max] | Paired ratio [conditional 95% interval] |
| --- | --- | --- | --- |
| ff | 68734.4 [66303.6, 69105.7] | 306882.5 [287364.7, 311135.0] | 4.412381 [4.257296, 4.570796] |
| mona | 55315.8 [54042.9, 55349.8] | 262074.0 [261750.7, 264779.9] | 4.784817 [4.750791, 4.821453] |
| darwin | 56751.1 [56486.2, 57016.8] | 257103.9 [255189.7, 264981.0] | 4.563738 [4.508684, 4.619465] |

The frozen comparator applies 10,000 paired log-ratio resamples, seed 20261023,
independently within each project, with equal project weights. These intervals
condition on the same search seed and this recorded timing session. They are
not independent search-seed or cross-device uncertainty.

## Correctness and completed work

All 33 workers complete successfully with zero process, device, numerical,
replay or artifact errors. Thirty normal workers contribute 31,457,280 measured
evaluations. Three separate candidate overflow replays contribute 3,145,728,
excluded from timing statistics. Total reported submitted and completed work is
34,603,008 evaluations, 135,168 iterations and 2,112 batches. All workers report
64 timestamp samples. Preparation, compilation, two warmups and numerical audits
are outside engine elapsed time; ordinary batches, drawing readback and the
checked final GPU drain remain inside it.

The comparison enforces exact initial/final equality of seven canonical fields:
parents, offspring, parent errors, parent framebuffers, offspring errors, chain
update flags and packed fitness. Current reference trajectories also reproduce
the original captured endpoints. Reporting-only best-chain control IDs retain
the documented exclusion. No evolutionary setting is changed to obtain a rate.

The separate offline audit verifies all 132 output hashes and 66 canonical
endpoint hashes. Six distinct canonical state values cover the repeated
endpoints. It recomputes all 24 distinct parent framebuffer RGB-byte L1 totals
and verifies all 264 parent quality associations through exact canonical
identity. Every associated stored fitness also matches the independent f32
penalty arithmetic bit for bit. All 132 exported final parent drawings match
their packed geometry, colors and fitness. Existing scalar numerical envelopes
still require zero unexplained pixels; scalar byte identity is not assumed.
Overflow maxima are ff 74, mona 80 and darwin 84, below capacity 256.

## Sources and delivery

The original executable is the immutable captured reference, SHA256
`ecaafba649c45c3fdcb952e1e8dbd90974042c0b32a0e2f780aaec3ddf2d0c41`.
The measured accepted-default executable is SHA256
`f42b9b0794f814a50a80d10caafe00909bcb17c5ed40dde8584a5df9f1fe1a8b`.
It contains the accepted wide local-count binner, decoded physical256 raster and
cooperative G1 mutation. Reports 03, 05, 11 and 13 retain their individual full
training, distinct mature-state/seed confirmation and source/runtime proofs.

The final locked production release build passes with optimization 3 and debug
assertions disabled. Its benchmark executable is byte-identical to the measured
one. The full final source inventory includes both target entry points, all
literal source fixtures, workspace manifests, lockfile, profiles and toolchain.
All three accepted shader bytes are verified inside both rebuilt executables.
The production application executable SHA256 is
`ec0f13fcade64c3e235f24d93a427e8e17290d0c86ce48aeddc06cd6a937ece2`.
The normal application was not launched to exercise UI or WebSocket behavior.

The earlier build-v11 inventory omitted the benchmark target entry-point file.
The original inventory is retained unchanged. A supplementary record confirms
that file matches its Git blob at the recorded build commit. The complete final
inventory and byte-identical rebuild provide the stronger artifact connection;
no frozen provenance was silently rewritten. The first final archive also lacked
a historical binner reference used only by unbuilt checker examples. That archive
is retained. The final artifact-only rebuild archives 83 source files and all
30 literal include dependencies, reproducing both executables byte for byte.

The user reported a possible computer restart before recovery. The interrupted
lazy-edge confirmation was closed without a full score or adoption. This final
comparison is a fresh package with both sides alternating after recovery.
The hourly scheduler was deleted at 07:00 UTC. Only this already-started bounded
comparison, its audit, release build and final record were finished afterward.
No new optimization experiment began after the cutoff.

## Limits and next bottleneck

The recorded backend is Vulkan through Dozen, Mesa 25.3.6, on an RTX 5090.
Background isolation remains unverified. This is engine throughput on the fixed
workloads, not application latency, GPU utilization, broad hardware scaling or
search-quality improvement. The default renderer/error stage remains the
largest sampled stage in the latest profiles. Last-iteration samples are not a
whole-run cost fraction. A single unscored host profile observes substantial
polling and about 28% of one CPU core, so the stage/wall residual cannot be
assigned to recoverable CPU time.

The cache-only readback pacing observer remains uncompiled and unrun. It is a
possible next diagnostic, not an optimization result. The lazy-edge candidate
and compact-edge composition remain unadopted despite their promising short
screen. No partial confirmation or runner-up receives a retroactive promotion.

Raw evidence is retained under
`.cache/evolution-overnight-20261024/final-original-v-default/`.
The offline audit is `final-original-v-default-audit.json`; final source/build
records and immutable executables are in `final-default-build-v2/`. The rerunnable
resident shader loop and source-hot bin checker are described in
[the campaign overview](README.md) and [engine-lab.md](engine-lab.md).
