FROM rust:latest AS builder
RUN cargo install trunk
WORKDIR /app
COPY Cargo.toml Cargo.lock* rust-toolchain.toml ./
COPY viewer/ ./viewer/
# Create stubs for root package so workspace resolves
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && \
    mkdir -p benches && echo "fn main() {}" > benches/bench.rs
# Add wasm target AFTER rust-toolchain.toml is in place (it forces nightly)
RUN rustup target add wasm32-unknown-unknown
WORKDIR /app/viewer
RUN trunk build --release

FROM nginx:alpine
COPY --from=builder /app/viewer/dist /usr/share/nginx/html
COPY viewer/nginx.conf /etc/nginx/conf.d/default.conf
