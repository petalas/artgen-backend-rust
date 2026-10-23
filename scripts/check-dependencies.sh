#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

cargo check --locked -p artgen-backend-rust --all-targets
cargo test --locked -p artgen-backend-rust -p artgen-shared
cargo check --locked -p artgen-viewer --target wasm32-unknown-unknown

cd viewer
NO_COLOR=true trunk build --locked
