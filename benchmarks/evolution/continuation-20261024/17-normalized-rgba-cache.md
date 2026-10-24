# Cache normalized polygon color once per tile

This independent source-only candidate moves the original `unpack_color` call
into the cooperative polygon loader. Four scalar f32 fields replace the packed
color word and padding. The pixel helper reconstructs its normalized RGBA vector,
then retains every original RGB multiplication by 255, inverse-alpha expression
and blend operand order. It starts directly from accepted AE1. No deferred-load
changes, alpha-only rewrite or rejected source is composed or reopened.

The [exact patch](attempts/14-normalized-rgba-cache.patch) changes workgroup-only
records from 48 to 56 bytes. At 256 records the cache is 14,336 bytes. Unchanged
reductions and conservatively padded globals bring the bound to 15,440, below
the existing 16,352-byte request. This adds 2,048 cache bytes. Every consumed
component has the same unique cooperative writer before the existing barrier.
Partial-image lanes retain loading and barrier participation. Geometry, AABB and
edge math, pixel ownership, integer errors, reductions, dispatch, settings and
RNG remain fixed. The inverse source proof reconstructs every AE1 byte.

The hypothesis is that cooperative decode can remove repeated covered-pixel
unpacking. Extra shared allocation and field reads, unconditional loader decode
and compiled float materialization may offset or invalidate it. Source arithmetic
and layout reasoning establish no hardware limiter, numerical equality or gain.
Candidate SHA256 is
`892c0019bbfa3f4a2e27388427c7fc7635308591f83b8a1f46dbff8a6856d6c9`.
Source, generator, inverse, layout and initialization proof are under
`.cache/evolution-rgba-20261024/source-110/`.

## Frozen comparison

The [new specification](profiles/normalized-rgba-cache-spec.json) fixes one
candidate, five alternating pairs per ff/mona/darwin, 64 measured batches, seed 7,
eight fixed preparation batches and ordinary Stages timestamps. The same
immutable release lab compiles both shaders with the existing production options.
Overlap stays false, mutation G1 and automatic mutation initialization stay fixed,
and the full manifest parameters remain unchanged. No Rust rebuild is needed.

The earlier [64-batch control](15-engine-kernel-control.md) is calibration evidence
only. Its rates do not enter this comparison. It is an earlier same-session
observation and does not guarantee current background isolation or noise. Current
preflight confirms the same accepted source files, helpers, binary identities and
quiet host load. Fresh paired reference/candidate runs supply every A/B sample;
no prior candidate samples or control correction are used.

Training must exceed aggregate ratio 1.005, have aggregate lower 95% bound strictly
above 1, each paired case median at least 0.995 and each case lower bound at least
0.99. The pinned paired-log bootstrap uses 10,000 draws, seed 20261024 and equal
case weights. All pairs remain included. A valid failure stops REJECT; errors,
incomplete work or numerical failure stop without a performance conclusion.
There are no retries, extra samples, warmup changes or follow-up source variants.

Only a training pass can receive the candidate-specific five-workgroup numerical
checker. Only that proof can permit three reserved-seed processes, using distinct
mature drawings and seeds 11/13/17. Their complete matrix must clear the same gate.
At most four GPU workers share a fresh 1,800-second package clock, with children
capped at 600 seconds and remaining time minus 30 seconds. No formal policy1 KEEP
or general driver claim follows.

The [runner](attempts/15-rgba-cache-training-runner.patch) reuses the hash-pinned,
26-test import-only validator. Only paths, arm label and the new specification
change; numerical/work/rate/source guards are retained. Syntax, pure gate controls
and scoped Ruff pass. Preflight verifies 78 accepted Rust/WGSL files, all three
normal binaries and two helpers. GPU compilation and numerical replay are the
next gate, not consequences of the source proof.

Power of Ten fixes cache capacity and initialization ownership. Prove It Works
requires actual compiled replay. Explain the Number keeps shared-memory and
source-operation hypotheses separate from completed-engine throughput.
