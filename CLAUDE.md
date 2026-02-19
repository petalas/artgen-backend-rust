# artgen-backend-rust

Genetic art generator in Rust. Evolves polygon-based drawings to approximate a reference image. Worker threads (one per CPU core) mutate drawings and evaluate fitness via CPU rasterization (half-space triangles + scanline fill) with SIMD alpha blending. A GPU path (wgpu) exists for rendering + compute-shader error diff but has known issues (errors returning 0). SDL2 window displays upscaled best result. Drawings are serialized to/from JSON (`*.best.json`).

## Active Warnings

<!-- Temporary alerts for cross-cutting concerns. Remove when resolved. -->

*(None currently.)*

## Knowledge System

This project uses a routing table (below) to map file patterns to docs you must read before editing. When you struggle with something, capture what you learned (see [When You Struggle](#when-you-struggle-mandatory)).

### Required Reading Before Editing

| File pattern you are editing | Read first |
|------------------------------|-----------|
| `src/engine.rs` | Engine struct, wgpu pipeline setup, fitness calculation, GPU rasterization path |
| `src/evaluator.rs` | Evolutionary loop — worker threads call `produce_new_best()` until improvement found |
| `src/utils.rs` | CPU rasterization (half-space `fill_triangle`, scanline `fill_shape`), SIMD color blending, helpers |
| `src/models/**` | Data models: Drawing (serialize/mutate), Polygon, Point, Color, Line. Coords are normalized 0.0–1.0 |
| `src/shader.wgsl`, `src/error.compute2.wgsl` | GPU shaders — vertex/fragment passthrough + compute error diff (workgroup 8x8) |
| `src/settings.rs` | Global constants (image dims, mutation probabilities, alpha range, polygon limits) |
| `src/main.rs` | SDL2 display, worker thread spawning (`num_cpus`), broadcast channel for new best, CLI args |
| `src/texture_wrapper.rs`, `src/buffer_dimensions.rs` | wgpu texture/buffer helpers (row alignment padding) |
| `benches/**` | Benchmarks use `divan` crate, not `criterion` |
| `Cargo.toml`, `build.sh`, `rust-toolchain.toml` | Build config — uses nightly (`portable_simd` feature) |
| Weird bug or unexpected behavior | [LEARNINGS.md](docs/LEARNINGS.md) — search for the symptom |

### When You Struggle (Mandatory)

If a fix takes more than one attempt:

1. **Check if documented** — search `docs/` for the key terms
2. **If documented**: improve the entry if it wasn't clear enough
3. **If new**: add to the appropriate doc (or LEARNINGS.md if unsure). Only capture things **specific to this project** or that contradict reasonable assumptions — not general programming knowledge.
4. **Add a routing entry** if no file pattern covers this area yet
5. **Consider code prevention**: can a wrapper, type guard, lint rule, or validator prevent this?
6. **Prune while you're there**: if you spot any outdated entries in the doc, fix or remove them
