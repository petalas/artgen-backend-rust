FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

# Install build dependencies + kisak-mesa PPA for dozen (Vulkan-on-D3D12) driver
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        software-properties-common \
        curl \
        build-essential \
        pkg-config \
        libsdl2-dev \
        libvulkan-dev \
        ca-certificates && \
    add-apt-repository -y ppa:kisak/kisak-mesa && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends mesa-vulkan-drivers && \
    rm -rf /var/lib/apt/lists/*

# Install Rust nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

# Build the project
WORKDIR /app
COPY Cargo.toml Cargo.lock* rust-toolchain.toml ./
COPY src/ src/
COPY benches/ benches/
RUN cargo build --release

# --- Runtime stage ---
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        software-properties-common \
        libsdl2-2.0-0 \
        libvulkan1 && \
    add-apt-repository -y ppa:kisak/kisak-mesa && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends mesa-vulkan-drivers && \
    rm -rf /var/lib/apt/lists/*

# Configure WSL lib path for d3d12
RUN echo '/usr/lib/wsl/lib' > /etc/ld.so.conf.d/wsl.conf && ldconfig 2>/dev/null || true

COPY --from=builder /app/target/release/artgen-backend-rust /usr/local/bin/artgen

# Use only the dozen (Vulkan-on-D3D12) ICD to access the discrete GPU
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /data
ENTRYPOINT ["artgen"]
