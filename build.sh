#!/bin/sh
# RUSTFLAGS="-C target-feature=+aes,+neon,+fcma" 
cargo build --release
echo "produced: ./target/release/artgen-backend-rust"