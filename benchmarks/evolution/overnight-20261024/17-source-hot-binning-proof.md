# Load binning sources into the full-buffer proof

The binning checker now accepts a shader file without rebuilding for each
candidate. It installs the bounded source through the existing engine factory,
then keeps it across every workgroup change. No fixture, oracle, buffer history,
or dispatch-loop byte changed.

The only library change exposes the existing setter with an exclusive-ownership
and drain contract. Shader specialization, cached handles, limits, default
compilation options and failure preservation remain unchanged. The checker
validates arguments, UTF-8, the 256 KiB source bound, WGSL semantics, marker
cardinality and output exclusion before GPU initialization. Source runs require
a fresh output directory and retain the exact source, SHA, scope and completion
result. A metadata file without a successful result is not a pass.

No arguments retain the legacy strict candidate-01/04 contract and its 255
dispatches. That contract intentionally rejects the restored full-bin baseline.
The paired full-buffer mode retains all 835 comparisons and 1,670 dispatches,
including poisoned/stale data, all five tile shapes, partial image boundaries,
empty draws, 255/256/257/500/999/1000 counts, and the 528-tile stripe history.

Nine CPU tests, scoped Cargo check, and owned-code all/pedantic diagnostics pass.
The library's 576 existing diagnostics remain recorded. Independent source
review found no material issue. The immutable release checker is
.cache/evolution-overnight-20261024/hot-bin-checker/build/check-gpu-binning,
SHA-256 d581446d03656b9b6aa80c74ad36b2ae3c402642b86a56bb02d844eaf749a10d.
Its actual Cargo artifact reports optimization 3 and disabled debug assertions;
all three production shader bytes and the frozen reference are embedded.

Actual main-thread GPU proof passed the packed-512 source through every full
buffer case: 835 comparisons and 1,670 dispatches. That source has a different
shared representation from the compiled default; its rejected performance
disposition is independent of this correctness proof. A separate valid shader
forced its polygon count to zero. The checker rejected its first full-image
fixture with actual count 0 versus reference 5, exited 1 and wrote no completed
result. Source bytes and hash were retained. This demonstrates that the new input
path is installed and checked.

The initial wrapper expected the words assertion or mismatch, so it misclassified
the descriptive count error. Its failed aggregation is retained alongside the
corrected offline summary. No GPU or compilation work was repeated for that
reporting error.

Evidence is under .cache/evolution-overnight-20261024/hot-bin-checker/:
final-source-proof.json, build/manifest.json, the CPU logs and source inventory,
gpu/packed-positive/, gpu/zero-counts-negative/, and gpu/corrected-summary.json.
Runs use a private working directory, the shared benchmark lock and an outer
timeout. The public command is:

    timeout 600s CHECKER --scope full-buffers --shader SOURCE.wgsl --output NEWDIR

The output parent must exist. This is an untimed behavior proof, not a throughput
measurement. Legacy GPU waits remain bounded by the outer timeout; each map/poll
also retains its 120-second bound.
