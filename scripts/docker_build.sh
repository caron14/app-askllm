#!/bin/bash

# Docker build script for HLE Screener

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building HLE Screener Docker images...${NC}"

# Parse arguments
TARGET=${1:-production}
TAG=${2:-latest}

# Build base image
echo -e "${YELLOW}Building base image...${NC}"
docker build \
    --target base \
    --tag hle-screener:base \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

# Build specified target
echo -e "${YELLOW}Building ${TARGET} image...${NC}"
docker build \
    --target ${TARGET} \
    --tag hle-screener:${TAG} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

# Tag images appropriately
if [ "$TARGET" == "production" ]; then
    docker tag hle-screener:${TAG} hle-screener:latest
    echo -e "${GREEN}Tagged as hle-screener:latest${NC}"
elif [ "$TARGET" == "development" ]; then
    docker tag hle-screener:${TAG} hle-screener:dev
    echo -e "${GREEN}Tagged as hle-screener:dev${NC}"
fi

# Show image sizes
echo -e "${YELLOW}Image sizes:${NC}"
docker images | grep hle-screener

echo -e "${GREEN}Build complete!${NC}"

# Instructions
echo -e "\n${YELLOW}To run the container:${NC}"
echo "  Production: docker-compose up hle-screener"
echo "  Demo: docker-compose up hle-demo"
echo "  Analysis: docker-compose up hle-analysis"
echo "  Development: docker-compose up hle-dev"
echo "  All services: docker-compose up"

# Check for NVIDIA Docker runtime
if ! docker info | grep -q nvidia; then
    echo -e "\n${RED}Warning: NVIDIA Docker runtime not detected${NC}"
    echo "GPU support may not be available. Install nvidia-docker2 for GPU support."
fi