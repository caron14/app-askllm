# HLE Quality Screener - Development Manual

## Overview

This manual provides detailed instructions for developing, extending, and maintaining the HLE Quality Screener system.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended) or Poetry
- Git
- CUDA toolkit (optional, for GPU acceleration)

### Installation from Source

```bash
# Clone repository
git clone <repo-url>
cd hle-screener

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Environment Variables

Create a `.env` file for local development:
```bash
CUDA_VISIBLE_DEVICES=0  # GPU selection
HF_HOME=/path/to/models  # Hugging Face cache directory
TRANSFORMERS_CACHE=/path/to/cache
```

## Architecture

### Core Components

#### 1. Data Loading (`io.py`)
- Loads HLE data from Hugging Face datasets
- Handles GSM8K dataset for testing
- Manages eval-only data persistence

#### 2. Embeddings (`embed.py`)
- `EmbeddingClient` class for pluggable embedding models
- Default: BAAI/bge-m3 (1024 dimensions)
- Alternative: intfloat/e5-large-v2
- L2 normalization for cosine similarity

#### 3. Indexing (`index.py`)
- FAISS IndexFlatIP for cosine similarity search
- Metadata management for HLE items
- Save/load functionality with safety checks

#### 4. Ask-LLM Judge (`askllm.py`)
- Qwen2.5-3B-Instruct as default judge
- Temperature=0 for deterministic outputs
- JSON-structured responses with retry logic

#### 5. Scoring (`score.py`)
- Composite quality score calculation
- Weighted similarity scoring
- Batch processing with checkpointing

#### 6. Visualization (`viz_umap.py`)
- UMAP dimensionality reduction
- Interactive Plotly visualizations
- Query projection capabilities

### Data Flow

```
Input Question
    ↓
Embedding Model
    ↓
FAISS Search → Top-5 HLE Items
    ↓
Ask-LLM Judge (with context)
    ↓
Quality Score Calculation
    ↓
Results (JSON/CSV)
```

## CLI Usage Examples

### Building Indexes

```bash
# Full HLE dataset
uv run python -m hle_screener.cli build-index

# Limited dataset for testing
uv run python -m hle_screener.cli build-index --limit 100

# Custom config
uv run python -m hle_screener.cli build-index --config configs/custom.yaml
```

### Scoring Operations

```bash
# Score single text
uv run python -m hle_screener.cli score-one \
  --text "If a train travels 60 mph for 2 hours, how far does it go?" \
  --id "train_001"

# Batch scoring with resume
uv run python -m hle_screener.cli score-batch \
  --limit 500 \
  --resume-from "gsm8k_0250" \
  --output-dir results/batch_001

# Score from file
cat questions.txt | while read line; do
  uv run python -m hle_screener.cli score-one --text "$line"
done
```

### UMAP Visualization

```bash
# Generate UMAP projection
uv run python -m hle_screener.cli prep-umap

# Custom parameters
uv run python -m hle_screener.cli prep-umap \
  --output artifacts/umap/custom.pkl
```

## Streamlit Application

### Running the App

```bash
# Default port (8501)
uv run streamlit run src/hle_screener/app_streamlit.py

# Custom port
uv run streamlit run src/hle_screener/app_streamlit.py \
  -- --server.port 8080

# Via CLI
uv run python -m hle_screener.cli serve --port 8080
```

### Features

1. **Search & Score Tab**
   - Text or ID-based input
   - Real-time scoring
   - CSV export functionality

2. **UMAP Visualization Tab**
   - Interactive 2D projections
   - Query point overlay
   - Top-5 highlighting

3. **Settings Tab**
   - Model configuration display
   - Safety compliance info
   - Configuration reload

4. **Batch Results Tab**
   - View saved batch results
   - Statistics and metrics
   - Export capabilities

## API Usage

### Python API

```python
from hle_screener import (
    create_embedding_client,
    FAISSIndex,
    create_askllm_judge,
    QualityScorer
)
from hle_screener.utils import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Initialize components
embedding_client = create_embedding_client(config=config)
index = FAISSIndex.load(
    Path("artifacts/indexes/faiss.index"),
    Path("artifacts/eval_only/DO_NOT_TRAIN/hle_metadata.json")
)
judge = create_askllm_judge(config=config)

# Create scorer
scorer = QualityScorer(index, embedding_client, judge, config)

# Score an item
result = scorer.score(
    item_id="test_001",
    item_text="What is the capital of France?"
)

print(f"Quality Score: {result.quality_score}")
print(f"Top Similar: {result.top5_references[0].hle_id}")
```

### Batch Processing

```python
# Load items
items = [
    {"id": f"item_{i}", "question": f"Question {i}"}
    for i in range(100)
]

# Score batch with checkpointing
results = scorer.score_batch(
    items,
    output_path=Path("results"),
    resume_from="item_50"  # Resume from specific ID
)

# Process results
quality_scores = [r.quality_score for r in results]
print(f"Average Quality: {sum(quality_scores)/len(quality_scores):.1f}")
```

## Configuration Management

### Custom Configuration

Create `configs/custom.yaml`:

```yaml
# Model settings
embed_model: "intfloat/e5-large-v2"
judge_model: "Qwen/Qwen2.5-7B-Instruct"  # Larger model

# Scoring
alpha: 0.7  # More weight on similarity
top_k: 10   # More references

# UMAP
umap_n_neighbors: 30
umap_min_dist: 0.05

# Processing
batch_size: 16  # Smaller for larger models
checkpoint_frequency: 5
```

### Environment-specific Settings

```yaml
# configs/gpu.yaml
device: "cuda:0"
torch_dtype: "float16"
batch_size: 64

# configs/cpu.yaml  
device: "cpu"
torch_dtype: "float32"
batch_size: 8
```

## Testing

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_index.py -v

# With coverage
uv run pytest tests/ --cov=src/hle_screener --cov-report=html

# Parallel execution
uv run pytest tests/ -n auto
```

### Writing Tests

Example test structure:

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    def test_basic_functionality(self):
        # Test implementation
        pass
    
    @pytest.mark.parametrize("input,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
    ])
    def test_parametrized(self, input, expected):
        # Parametrized test
        pass
    
    @patch('module.external_dependency')
    def test_with_mock(self, mock_dep):
        # Test with mocked dependencies
        pass
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=8
   
   # Use CPU instead
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Model Download Failures**
   ```bash
   # Set custom cache directory
   export HF_HOME=/path/to/cache
   
   # Use offline mode
   export HF_DATASETS_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

3. **FAISS Index Errors**
   ```python
   # Rebuild index
   uv run python -m hle_screener.cli build-index --force
   ```

4. **Streamlit Port Conflicts**
   ```bash
   # Use different port
   uv run streamlit run src/hle_screener/app_streamlit.py \
     --server.port 8502
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:
```bash
export LOG_LEVEL=DEBUG
```

## Performance Optimization

### Embedding Optimization

```python
# Batch processing
embedding_client.encode_batch(texts, batch_size=64)

# GPU acceleration
embedding_client = EmbeddingClient(
    model_id="BAAI/bge-m3",
    device="cuda:0"
)
```

### FAISS Optimization

```python
# Use GPU index (requires faiss-gpu)
import faiss
gpu_index = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(),
    0,  # GPU ID
    cpu_index
)
```

### Model Quantization

```python
# 8-bit quantization for judge
judge = AskLLMJudge(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    load_in_8bit=True
)
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync

# Copy application
COPY . .

# Run app
CMD ["uv", "run", "streamlit", "run", "src/hle_screener/app_streamlit.py"]
```

### Production Checklist

- [ ] Set `PYTHONUNBUFFERED=1` for logging
- [ ] Configure model caching directory
- [ ] Set appropriate batch sizes for hardware
- [ ] Enable SSL for Streamlit if exposed
- [ ] Configure monitoring (logs, metrics)
- [ ] Set up regular index updates
- [ ] Implement backup strategy for artifacts

## Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Keep functions focused and small

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit PR with clear description

### Adding New Models

1. Update `embed.py` or `askllm.py`
2. Add configuration in `default.yaml`
3. Write tests for new model
4. Update documentation

## Safety Compliance

### Checklist

- [ ] Never write HLE text to training datasets
- [ ] Store HLE artifacts under `eval_only/DO_NOT_TRAIN/`
- [ ] Include canary GUID in all HLE outputs
- [ ] Use temperature=0 for reproducibility
- [ ] Log all model configurations

### Audit Trail

All operations are logged with:
- Timestamp
- Model versions
- Configuration hash
- Input/output samples
- Canary GUID

## Support

For issues or questions:
1. Check this manual
2. Review test cases for examples
3. Check GitHub issues
4. Contact maintainers

## Version History

- **0.1.0**: Initial release
  - Core scoring functionality
  - Streamlit UI
  - CLI interface
  - UMAP visualization