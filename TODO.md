# TODO.md - Living Development Document

> Last Updated: 2025-08-10
> Auto-updated on each PR merge

## 🚨 Current Blockers
- **Streamlit Cloud Deployment**: Full version incompatible due to 1GB RAM limit
  - Mitigation: Created lightweight demo version (app_cloud.py) ✅

## ⚠️ Risk Items
1. **Model Memory Requirements**: Qwen2.5-3B requires ~6GB VRAM minimum
   - Mitigation: Add CPU fallback and quantization options
   - Status: ⚠️ Demo version created for cloud deployment
2. **Dataset Access**: MMLU/HLE dataset may require authentication
   - Mitigation: Mock data fallback implemented ✅
3. **FAISS GPU Support**: Requires separate faiss-gpu installation
   - Mitigation: CPU index works but slower for large datasets ✅
4. **Cloud Deployment Limitations**: Streamlit Cloud has 1GB RAM limit
   - Mitigation: Created demo version with mock data ✅

## 🔄 In Progress
- [ ] Model quantization for reduced memory usage
- [ ] Docker containerization for easier deployment
- [ ] Integration tests for all three Streamlit apps

## 📋 Next Steps

### High Priority
- [ ] Add model quantization support (8-bit, 4-bit)
- [ ] Implement distributed batch processing
- [ ] Add progress bars to CLI operations
- [ ] Create Docker container for deployment

### Medium Priority
- [ ] Add support for additional embedding models
- [ ] Implement caching layer for embeddings
- [ ] Add data validation and error recovery
- [ ] Create API endpoint wrapper
- [ ] Add metrics dashboard

### Low Priority
- [ ] Support for custom judge prompts
- [ ] Export to additional formats (Parquet, Arrow)
- [ ] Add embedding fine-tuning capability
- [ ] Implement A/B testing framework

## ✅ Recently Completed
- [x] Pre-analysis Streamlit app for visual similarity analysis (2025-01-10)
- [x] CLAUDE.md for AI assistant guidance (2025-01-10)
- [x] Fixed Pydantic v2 and dependency compatibility (2025-01-10)
- [x] Batch processing auto-resume with run ID tracking (2025-01-10)
- [x] Living TODO.md with auto-update scripts (2025-01-10)
- [x] Git version control initialization (2025-01-10)
- [x] Streamlit Cloud demo version (2025-01-10)
- [x] Enhanced CLI with status and list-runs commands (2025-01-10)
- [x] Core scoring pipeline (2025-01-10)
- [x] Streamlit web interface (2025-01-10)
- [x] CLI with all commands (2025-01-10)
- [x] Test suite with >80% coverage (2025-01-10)
- [x] Safety compliance for eval-only data (2025-01-10)

## 🧪 Test Status
❌ Failing (Last checked: 2025-08-10)

## 🐛 Known Issues
1. **Streamlit UMAP tab**: May be slow with >10k points
   - Workaround: Use sampling or pre-computation
2. **Batch checkpoint frequency**: Fixed at 10 items
   - Workaround: Modify in config file
3. **Full version on Streamlit Cloud**: Incompatible due to resource limits
   - Workaround: Use demo version (app_cloud.py) or deploy locally

## 💡 Future Enhancements
- Multi-GPU support for large-scale processing
- Real-time streaming evaluation
- Integration with MLflow/Weights & Biases
- Custom embedding model training
- Active learning for judge improvement
- Real HLE dataset integration when available
- Advanced visualization techniques (t-SNE, PCA)
- Batch analysis for multiple queries

## 📊 Performance Benchmarks
| Operation | Items | Time | Hardware |
|-----------|-------|------|----------|
| Build Index | 1000 | ~2 min | CPU |
| Score Single | 1 | ~3 sec | GPU |
| Batch Score | 100 | ~5 min | GPU |
| UMAP Prep | 5000 | ~1 min | CPU |
| Demo App Load | - | ~5 sec | Cloud |
| Auto-Resume | 100 | instant | Any |
| Analysis App | 500 | ~10 sec | CPU |

## 🔗 Dependencies to Watch
- `transformers`: Check for Qwen model updates
- `sentence-transformers`: New embedding models
- `faiss-cpu`: Performance improvements
- `streamlit`: UI enhancements

## 🚀 Deployment Options
| Platform | Status | Notes |
|----------|--------|-------|
| Local | ✅ Ready | Full features with GPU support |
| Streamlit Cloud | ⚠️ Demo only | Use app_cloud.py or app_analysis.py |
| Docker | 🔄 Planned | Containerization in progress |
| AWS/GCP | 📋 TODO | Requires GPU instances |
| Colab | ✅ Possible | Notebook adaptation needed |



## 📈 Project Statistics
- Python files: 12
- Test files: 4
- Total lines of code: 1,787
- Last automated update: 2025-08-10

## 📝 Notes for Contributors
- Always update this file when completing tasks
- Add new blockers as soon as identified
- Update risk items with mitigation status
- Keep performance benchmarks current
- Test demo version before cloud deployment
