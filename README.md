# Polygon renderer from scratch

## Requirements

[SDL2](https://wiki.libsdl.org/SDL2/Installation)

## Build and run

- `./build.sh`
- `./target/release/argten-backend-rust`

## Benchmarks

`cargo bench`

## Last benchmark results (on M1 Max)

```text
bench                 fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ calculate_fitness                │               │               │               │         │
│  ├─ GPU             10.48 ms      │ 22.96 ms      │ 12.65 ms      │ 12.63 ms      │ 100     │ 100
│  ├─ HalfSpace       2.535 ms      │ 2.971 ms      │ 2.546 ms      │ 2.573 ms      │ 100     │ 100
│  ╰─ Scanline        6.628 ms      │ 8.025 ms      │ 6.987 ms      │ 7.059 ms      │ 100     │ 100
├─ draw                             │               │               │               │         │
│  ├─ HalfSpace       2.364 ms      │ 2.656 ms      │ 2.374 ms      │ 2.402 ms      │ 100     │ 100
│  ╰─ Scanline        6.285 ms      │ 7.744 ms      │ 6.832 ms      │ 6.958 ms      │ 100     │ 100
╰─ fill_color         454.7 µs      │ 546.3 µs      │ 457.1 µs      │ 461 µs        │ 100     │ 100
```

## To generate a flamegraph

- Make sure you have perf installed if on linux.
- `cargo install flamegraph`
- `./flamegraph.sh`

To generate one manually, while the app is running run:

- `perf record --call-graph dwarf -p $(pgrep artgen)` (might have to run with sudo).
- Ctrl+C after a few seconds.
- Might have to fix permissions of perf.data if you had to run perf with sudo.
- `perf script | inferno-collapse-perf | inferno-flamegraph > perf.svg`
- open perf.svg in a browser

## To look at asm

`cargo install cargo-show-asm --force`
`cargo-asm --lib utils::fill_pixel`
