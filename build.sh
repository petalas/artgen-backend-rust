#!/bin/sh
# RUSTFLAGS="-O -C target-feature=+aes,+neon,+fcma"
# RUSTFLAGS="--emit asm" cargo build --release
RUSTFLAGS="--emit asm -C embed-bitcode -C target-cpu=native" cargo build --release
echo "produced: ./target/release/artgen-backend-rust"