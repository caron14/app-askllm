#!/bin/bash

# Docker run helper script for HLE Screener

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
SERVICE=${1:-hle-screener}
COMMAND=${2:-}

# Function to check prerequisites
check_prerequisites() {
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi

    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}docker-compose is not installed${NC}"
        exit 1
    fi

    # Check if .env file exists
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Creating .env file from example...${NC}"
        if [ -f .env.example ]; then
            cp .env.example .env
            echo -e "${GREEN}.env file created. Please update with your values.${NC}"
        else
            echo -e "${YELLOW}Creating basic .env file...${NC}"
            cat > .env <<EOF
# Hugging Face token for accessing gated datasets
HF_TOKEN=your_token_here

# CUDA device configuration
CUDA_VISIBLE_DEVICES=0

# Model quantization (none, 8bit, 4bit)
QUANTIZATION=none

# Number of workers for distributed processing
NUM_WORKERS=4
EOF
            echo -e "${GREEN}.env file created. Please update HF_TOKEN.${NC}"
        fi
    fi
}

# Function to create necessary directories
create_directories() {
    echo -e "${YELLOW}Creating necessary directories...${NC}"
    mkdir -p data/hle data/gsm8k
    mkdir -p artifacts/indexes artifacts/eval_only/DO_NOT_TRAIN artifacts/umap artifacts/results
    mkdir -p logs
    echo -e "${GREEN}Directories created${NC}"
}

# Function to download datasets if needed
download_datasets() {
    if [ ! -d "data/hle" ] || [ -z "$(ls -A data/hle)" ]; then
        echo -e "${YELLOW}HLE data not found. Run download script after starting container.${NC}"
    fi
    if [ ! -d "data/gsm8k" ] || [ -z "$(ls -A data/gsm8k)" ]; then
        echo -e "${YELLOW}GSM8K data not found. Run download script after starting container.${NC}"
    fi
}

# Main execution
echo -e "${GREEN}HLE Screener Docker Runner${NC}"

# Check prerequisites
check_prerequisites

# Create directories
create_directories

# Check datasets
download_datasets

# Run based on service
case $SERVICE in
    "build")
        echo -e "${YELLOW}Building Docker images...${NC}"
        ./scripts/docker_build.sh production latest
        ;;
    
    "up")
        echo -e "${YELLOW}Starting all services...${NC}"
        docker-compose up -d
        echo -e "${GREEN}All services started${NC}"
        echo "Main app: http://localhost:8501"
        echo "Demo app: http://localhost:8502"
        echo "Analysis app: http://localhost:8503"
        echo "Ray dashboard: http://localhost:8265"
        ;;
    
    "down")
        echo -e "${YELLOW}Stopping all services...${NC}"
        docker-compose down
        echo -e "${GREEN}All services stopped${NC}"
        ;;
    
    "hle-screener")
        echo -e "${YELLOW}Starting main HLE Screener app...${NC}"
        docker-compose up -d hle-screener
        echo -e "${GREEN}Main app started at http://localhost:8501${NC}"
        ;;
    
    "demo")
        echo -e "${YELLOW}Starting demo app...${NC}"
        docker-compose up -d hle-demo
        echo -e "${GREEN}Demo app started at http://localhost:8502${NC}"
        ;;
    
    "analysis")
        echo -e "${YELLOW}Starting analysis app...${NC}"
        docker-compose up -d hle-analysis
        echo -e "${GREEN}Analysis app started at http://localhost:8503${NC}"
        ;;
    
    "dev")
        echo -e "${YELLOW}Starting development environment...${NC}"
        docker-compose run --rm hle-dev /bin/bash
        ;;
    
    "cli")
        echo -e "${YELLOW}Running CLI command: ${COMMAND}${NC}"
        docker-compose run --rm hle-screener python3 -m hle_screener.cli ${COMMAND}
        ;;
    
    "build-index")
        echo -e "${YELLOW}Building FAISS index...${NC}"
        docker-compose run --rm hle-screener python3 -m hle_screener.cli build-index
        ;;
    
    "score-batch")
        echo -e "${YELLOW}Running batch scoring...${NC}"
        docker-compose run --rm hle-screener python3 -m hle_screener.cli score-batch --limit 100
        ;;
    
    "distributed")
        echo -e "${YELLOW}Starting distributed scoring with Ray...${NC}"
        docker-compose up -d ray-head ray-worker
        sleep 5
        docker-compose run --rm hle-screener python3 -m hle_screener.cli score-distributed \
            --use-ray --ray-address ray-head:10001 --limit 100
        ;;
    
    "logs")
        SERVICE_NAME=${COMMAND:-hle-screener}
        echo -e "${YELLOW}Showing logs for ${SERVICE_NAME}...${NC}"
        docker-compose logs -f ${SERVICE_NAME}
        ;;
    
    "shell")
        SERVICE_NAME=${COMMAND:-hle-screener}
        echo -e "${YELLOW}Opening shell in ${SERVICE_NAME}...${NC}"
        docker-compose exec ${SERVICE_NAME} /bin/bash
        ;;
    
    "test")
        echo -e "${YELLOW}Running tests...${NC}"
        docker-compose run --rm hle-dev pytest tests/ -v
        ;;
    
    "clean")
        echo -e "${YELLOW}Cleaning up...${NC}"
        docker-compose down -v
        docker system prune -f
        echo -e "${GREEN}Cleanup complete${NC}"
        ;;
    
    *)
        echo -e "${YELLOW}Usage: $0 [command] [options]${NC}"
        echo ""
        echo "Commands:"
        echo "  build         - Build Docker images"
        echo "  up            - Start all services"
        echo "  down          - Stop all services"
        echo "  hle-screener  - Start main app only"
        echo "  demo          - Start demo app only"
        echo "  analysis      - Start analysis app only"
        echo "  dev           - Start development shell"
        echo "  cli [cmd]     - Run CLI command"
        echo "  build-index   - Build FAISS index"
        echo "  score-batch   - Run batch scoring"
        echo "  distributed   - Run distributed scoring"
        echo "  logs [service]- Show logs for service"
        echo "  shell [service]- Open shell in service"
        echo "  test          - Run tests"
        echo "  clean         - Clean up containers and volumes"
        ;;
esac