# artgen-backend-rust

Genetic art generator in Rust. Evolves polygon-based drawings to approximate a reference image. Worker threads (one per CPU core) mutate drawings and evaluate fitness via CPU rasterization (half-space triangles + scanline fill) with SIMD alpha blending. A GPU path (wgpu) exists for rendering + compute-shader error diff but has known issues (errors returning 0). SDL2 window displays upscaled best result. Drawings are serialized to/from JSON (`*.best.json`).

## Active Warnings

<!-- Temporary alerts for cross-cutting concerns. Remove when resolved. -->

*(None currently.)*

## Agent Delegation Hints

This codebase has several distinct subsystems — use parallel agents to investigate them independently rather than reading everything into the main context:

- **GPU pipeline** (`src/gpu_evolver/`, `src/shaders/`): complex, lots of files — always delegate exploration to an agent
- **CPU rasterization** (`src/utils.rs`, `src/evaluator.rs`): SIMD-heavy, benefits from focused agent analysis
- **Models & settings** (`src/models/`, `src/settings.rs`): small files, OK to read directly if you just need a quick lookup
- **Cross-cutting changes** (e.g., adding a new mutation type): spawn parallel agents for GPU shader side and CPU/model side simultaneously

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
| `src/gpu_evolver/**` | Read [engine module ownership](docs/engine-architecture.md) before changing capacity, shader installation, command scheduling or readback. |
| `src/benchmark/**`, `examples/engine-lab.rs`, `scripts/evolution-bench.py` | Read [trial qualification](docs/trial-qualification.md) before changing record purpose, canonical identity or timed work. |
| Engine performance work, `scripts/hillclimb.py`, `examples/evolution-bench.rs` | Read [hill-climbing](docs/hillclimb.md). Screen every engine change with `scripts/hillclimb.py` before and after; it logs to `benchmarks/hillclimb/log.tsv`. |
| `src/benchmark/quality.rs`, `scripts/quality.py`, evolution-settings tuning | Read [quality benchmark](docs/quality-benchmark.md). Judge settings and search changes by remaining error over minutes from a blank start, not by evals/s. |
| `src/shaders/*.wgsl` | GPU compute shaders — `mutate.wgsl` (PCG RNG + all mutations), `rasterize_error.wgsl` (half-space triangle per pixel + error reduction), `select.wgsl` (selection + migration). Stage files are not complete WGSL: `common.wgsl`, `tiles.wgsl` and Rust-generated constants are appended (see "Shader installation" in [engine module ownership](docs/engine-architecture.md)) |
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
