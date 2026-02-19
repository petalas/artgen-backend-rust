#!/bin/bash
# Run the artgen GPU pipeline in Docker with access to the NVIDIA GPU
# Usage: ./docker-run.sh <image_file> [--gpu]
#
# Requires: nvidia-container-toolkit, Docker

set -e

IMAGE_FILE="${1:?Usage: ./docker-run.sh <image_file> [--gpu]}"
shift

# Build if needed
docker build -t artgen .

# Run with GPU access and WSL lib mounts for D3D12
docker run --rm -it \
    --gpus all \
    -v /usr/lib/wsl:/usr/lib/wsl:ro \
    --device /dev/dxg:/dev/dxg \
    -v "$(pwd):/data" \
    artgen "$IMAGE_FILE" "$@"
