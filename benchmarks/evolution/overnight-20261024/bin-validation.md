# Binner boundary verification

The checker retains all earlier 832 paired cases. Three new cases use a
384 by 704 image at raster tile size 32 by 16, giving 528 tiles. They exercise
the second tile stripe of a 512-lane binner, and polygon preload counts
511/512/513, followed by smaller and empty drawings in the same buffers.
Each readback is 1,628,352 bytes, bounded below 2 MiB. Strict mode remains
unchanged at 255 dispatches.

The selected 512-lane, local-count shader passed all 835 paired full-buffer
comparisons, or 1,670 GPU dispatches. Counts and every index word match the
frozen original GPU binner, including occupancy above 256 and unwritten suffixes.
The exact source SHA-256 is
`b851f25680f2a61110cddd8683b1cafea221b683a1ea822c90a46bba9c5522da`.
Evidence and the checked executable identity are under
`.cache/evolution-overnight-20261024/binner-confirmation/full-buffers/` and `build/`.
This is a correctness result, not throughput acceptance.

The instrumentation tests now pin the 32-lane atomic-add reference fixture,
whose SHA-256 is
`a219175f3fd395600f0ecbef3ac099b7450aeaf972e1e5c91dfc5dcde1fb901b`.
That keeps the legacy factory test meaningful when the production winner uses
another workgroup width or local counting. A separate existing test validates
the actual production shader and its diagnostic variant at all five raster
workgroup sizes. All six focused CPU tests pass. The checker release build and
all/pedantic Clippy checks pass with no checker diagnostics; existing library
warnings remain outside this change.
