# Multi-stage build for optimized image size
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
COPY README.md .

# Install uv for package management
RUN pip install uv

# Install project dependencies
RUN uv pip install --system -e .

# Install additional dependencies for quantization and distributed processing
RUN uv pip install --system \
    bitsandbytes \
    accelerate \
    ray[default] \
    tqdm \
    psutil

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY CLAUDE.md .
COPY TODO.md .

# Create necessary directories
RUN mkdir -p \
    artifacts/indexes \
    artifacts/eval_only/DO_NOT_TRAIN \
    artifacts/umap \
    artifacts/results \
    data/hle \
    data/gsm8k \
    .cache/huggingface \
    .cache/transformers

# Set permissions
RUN chmod -R 755 /app

# Expose ports
EXPOSE 8501 8502 8503

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python3", "-m", "hle_screener.cli", "--help"]

# ----------------------------
# Development image with additional tools
FROM base as development

# Install development tools
RUN uv pip install --system \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    ipython \
    jupyter

# Set development environment
ENV ENVIRONMENT=development

# ----------------------------
# Production image (optimized)
FROM base as production

# Remove unnecessary files
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
RUN find /app -type f -name "*.pyc" -delete

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set production environment
ENV ENVIRONMENT=production

# Default to Streamlit app
CMD ["python3", "-m", "streamlit", "run", "src/hle_screener/app_streamlit.py", "--server.port", "8501"]