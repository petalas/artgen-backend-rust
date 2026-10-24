# Paired-pixel traversal stops at a compiler failure

The two-pixel raster experiment is **INCONCLUSIVE**. Its single native attempt
aborts in Naga 30.0.1's SPIR-V writer before any timing, diagnostic or qualification
record completes. There is no performance ratio, numerical result or adoption.
The original cohort is closed; no GPU retry or reserved run follows.

The [protocol](profiles/pixel-pair-spec.json),
[candidate patch](attempts/23-raster-pixel-pair.patch) and
[runner](attempts/23-pixel-pair-runner.patch) were committed in ce36b70 before
launch. The hypothesis was to traverse the ordered polygon prefix once for the
two pixels already owned by a physical lane. The candidate retains one logical
32 by 16 tile per physical workgroup of 256 lanes. It changes neither dispatch
counts nor evolution settings. Each pixel retains its ascending polygon order
and the original blend function. The four one-pixel workgroup shapes retain
their original path. This differs from the rejected T2/T4 tile-grouping changes.

The preceding compiler observation showed repeated shared polygon loads in
the serial pixel loop. It supplied no dynamic instruction counts or hardware
limiter. Reusing those loads was a guess; retaining two live pixel states could
also increase costs. The experiment establishes neither outcome.

The immutable prepared lab executable is
97d176f3b5f97c818d19a5e25b2f75e9e587bf1925a948c330b78e2022512ab1.
Its frozen invocation specifies five alternating pairs for each of ff, Mona and
Darwin, seed 7, 64 batches and eight warmups, ordinary Stages timestamps,
diagnostics enabled and finalization overlap false. The candidate source is
9996fe7ceb50a9b3bc70d0289a2c291b50689937146be0f4a61466089ebd8142.
All evolution parameters and compilation policies remain fixed.

The child exits with signal 6. Its retained stderr reports:

```text
naga-30.0.1/src/back/spv/block.rs:3847:56
internal error: entered unreachable code: Expression [90] is not cached!
```

Both JSONL streams are empty. The partial summary still says running and has
zero recorded errors; that unfinished field cannot count the external abort.
The supervisor records the failed process and the controller closes INCONCLUSIVE.
No successful preparation is recorded. Zero recorded evaluations is a lower
bound, not proof that setup executed no GPU work. All partial files remain intact.
The [machine record](profiles/pixel-pair-result.json) retains exact identities,
the failure signature and the CPU regression evidence.

A parent offline audit passes 193 failure-accounting and preservation checks.
The 78 protected Rust/WGSL sources, three normal executables and two benchmark
helpers retain their hashes. The copied sources match the frozen inputs. These
checks establish failure accounting and preservation, not numerical correctness.
No candidate state snapshots exist to audit.

A CPU-only reproducer now exercises the existing Naga 30.0.1 parser, validator
and SPIR-V writer without device initialization. The reference succeeds; the
original candidate reproduces the same panic twice. A five-line input reduces the
trigger to passing `&pixel.r` to a function accepting `ptr<function, f32>`.
Replacing that struct-member pointer with a standalone local scalar makes the
minimal control succeed. Neither a returned pixel struct nor the polygon loop
is required for this reduced failure. The writer's function-call argument lookup
is the observed failing site; this does not diagnose every possible Naga failure.

One [source-only workaround](attempts/23-pixel-pair-compiler-workaround.patch)
passes local RGB scalars to the unchanged blend function, then copies the values
back. Reversing those six declarations, six stores and two argument-list changes
restores the original candidate exactly. Its source hash is
83c505bcc9220f39c11354692cdec3828e4d76681df8468ea93d597b4f4ceb7b.
The final CPU matrix has ten expected outcomes: the reference, scalar control
and five specialized workaround shapes succeed; both original-candidate shapes
and the minimal struct-pointer case retain the expected panic. GPU correctness
and throughput of the workaround remain unqualified.

The [probe source](attempts/23-wgsl-spv-probe.patch) and cached executable are
retained under `.cache/evolution-pixel-pair-20261024/compiler-parent/`.
`spv-probe-final PATH.wgsl` supplies a fast compiler check before a future GPU
attempt. It reuses pinned dependency artifacts and needs no Cargo or engine
rebuild. Its SPIR-V 1.3 descriptor disables workgroup zeroing but uses the writer's
default binding map and capabilities, rather than the full adapter-specific wgpu
descriptor. It is a compiler regression check, not a GPU numerical oracle.

Prove It Works replaced a source-only assumption with an exact compiler
reproduction. Explain the Number keeps this failed launch out of performance
statistics. The fast loop required no dependency upgrade or production change.
