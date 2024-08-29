# Polygon renderer from scratch

## Build and run

- `./build.sh`
- `./target/release/argten-backend-rust`

## Benchmarks

`cargo bench`

## Last benchmark results (on M1 Max)

```text
bench            fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ draw                        │               │               │               │         │
   ├─ HalfSpace  2.508 ms      │ 3.147 ms      │ 2.513 ms      │ 2.53 ms       │ 100     │ 1000
   ╰─ Scanline   6.092 ms      │ 6.421 ms      │ 6.148 ms      │ 6.163 ms      │ 100     │ 1000
```
