# TODO.md - Living Development Document

> Last Updated: 2025-01-10
> Auto-updated on each PR merge

## 🚨 Current Blockers
- None

## ⚠️ Risk Items
1. **Model Memory Requirements**: Qwen2.5-3B requires ~6GB VRAM minimum
   - Mitigation: Add CPU fallback and quantization options
2. **Dataset Access**: MMLU/HLE dataset may require authentication
   - Mitigation: Mock data fallback implemented
3. **FAISS GPU Support**: Requires separate faiss-gpu installation
   - Mitigation: CPU index works but slower for large datasets

## 🔄 In Progress
- [ ] Batch processing auto-resume improvements
- [ ] Run ID tracking system

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
- [x] Core scoring pipeline (2025-01-10)
- [x] Streamlit web interface (2025-01-10)
- [x] CLI with all commands (2025-01-10)
- [x] Test suite with >80% coverage (2025-01-10)
- [x] Safety compliance for eval-only data (2025-01-10)

## 🐛 Known Issues
1. **Streamlit UMAP tab**: May be slow with >10k points
   - Workaround: Use sampling or pre-computation
2. **Batch checkpoint frequency**: Fixed at 10 items
   - Workaround: Modify in config file

## 💡 Future Enhancements
- Multi-GPU support for large-scale processing
- Real-time streaming evaluation
- Integration with MLflow/Weights & Biases
- Custom embedding model training
- Active learning for judge improvement

## 📊 Performance Benchmarks
| Operation | Items | Time | Hardware |
|-----------|-------|------|----------|
| Build Index | 1000 | ~2 min | CPU |
| Score Single | 1 | ~3 sec | GPU |
| Batch Score | 100 | ~5 min | GPU |
| UMAP Prep | 5000 | ~1 min | CPU |

## 🔗 Dependencies to Watch
- `transformers`: Check for Qwen model updates
- `sentence-transformers`: New embedding models
- `faiss-cpu`: Performance improvements
- `streamlit`: UI enhancements

## 📝 Notes for Contributors
- Always update this file when completing tasks
- Add new blockers as soon as identified
- Update risk items with mitigation status
- Keep performance benchmarks current

---
*This document is automatically updated via GitHub Actions on PR merge*