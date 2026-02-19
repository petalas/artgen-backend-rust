#!/bin/bash
# Run the artgen GPU pipeline in Docker with access to the NVIDIA GPU via WSL2/D3D12
# Usage: ./docker-run.sh <image_file> [extra args...]
#
# Examples:
#   ./docker-run.sh ff.jpg --gpu --headless
#   ./docker-run.sh ff.jpg --gpu
#
# Output files (*.best.json, *.best.png) are written to the current directory.
# Requires: nvidia-container-toolkit, Docker

set -e

IMAGE_FILE="${1:?Usage: ./docker-run.sh <image_file> [--gpu] [--headless]}"
shift

# Build if needed
docker build -t artgen .

# Run with GPU access and WSL lib mounts for D3D12 (dozen/dzn Vulkan driver)
docker run --rm -it \
    --gpus all \
    -v /usr/lib/wsl:/usr/lib/wsl:ro \
    --device /dev/dxg:/dev/dxg \
    -v "$(pwd):/data" \
    artgen "$IMAGE_FILE" "$@"
