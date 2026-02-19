FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libsdl2-dev \
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml Cargo.lock* rust-toolchain.toml ./
COPY src/ src/
COPY benches/ benches/

RUN cargo build --release

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# libnvidia-gl provides the Vulkan ICD (nvidia_icd.json + libGLX_nvidia.so)
# The NVIDIA Container Toolkit overrides these with host-matched driver libs at runtime.
# Pin to a version close to the host driver — the container toolkit will handle mismatches.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libvulkan1 \
    libnvidia-gl-590 \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /app
COPY --from=builder /app/target/release/artgen-backend-rust .

ENTRYPOINT ["./artgen-backend-rust"]
CMD ["--gpu", "--headless"]
