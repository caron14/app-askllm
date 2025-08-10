import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.hle_screener.schema import HLEItem
from src.hle_screener.index import FAISSIndex, build_hle_index
from src.hle_screener.embed import EmbeddingClient


class TestFAISSIndex:
    def test_index_creation(self):
        index = FAISSIndex(dimension=768)
        assert index.dimension == 768
        assert index.index.ntotal == 0
        assert len(index.metadata) == 0
    
    def test_add_and_search(self):
        index = FAISSIndex(dimension=3)
        
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float32)
        
        metadata = [
            HLEItem(hle_id=f"test_{i}", subject="math", question_text=f"Question {i}")
            for i in range(5)
        ]
        
        index.add(embeddings, metadata)
        assert index.index.ntotal == 5
        assert len(index.metadata) == 5
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=3)
        
        assert len(results) == 3
        assert results[0].hle_id == "test_0"
        assert results[0].rank == 1
        assert 0.99 <= results[0].cosine_similarity <= 1.01
    
    def test_cosine_similarity_monotonicity(self):
        index = FAISSIndex(dimension=3)
        
        embeddings = np.random.randn(100, 3).astype(np.float32)
        metadata = [
            HLEItem(hle_id=f"test_{i}", subject="math", question_text=f"Question {i}")
            for i in range(100)
        ]
        
        index.add(embeddings, metadata)
        
        query = np.random.randn(1, 3).astype(np.float32)
        results = index.search(query, k=10)
        
        similarities = [r.cosine_similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_top_k_count(self):
        index = FAISSIndex(dimension=3)
        
        embeddings = np.random.randn(10, 3).astype(np.float32)
        metadata = [
            HLEItem(hle_id=f"test_{i}", subject="math", question_text=f"Question {i}")
            for i in range(10)
        ]
        
        index.add(embeddings, metadata)
        
        query = np.random.randn(1, 3).astype(np.float32)
        
        for k in [1, 3, 5, 10]:
            results = index.search(query, k=k)
            assert len(results) == k
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            index = FAISSIndex(dimension=3)
            embeddings = np.random.randn(5, 3).astype(np.float32)
            metadata = [
                HLEItem(hle_id=f"test_{i}", subject="math", question_text=f"Question {i}")
                for i in range(5)
            ]
            
            index.add(embeddings, metadata)
            
            index_path = tmppath / "test.index"
            metadata_path = tmppath / "eval_only/DO_NOT_TRAIN/metadata.json"
            index.save(index_path, metadata_path)
            
            loaded_index = FAISSIndex.load(index_path, metadata_path)
            
            assert loaded_index.dimension == 3
            assert loaded_index.index.ntotal == 5
            assert len(loaded_index.metadata) == 5
            assert loaded_index.metadata[0].hle_id == "test_0"


class TestIndexDeterminism:
    def test_deterministic_search(self):
        np.random.seed(42)
        index = FAISSIndex(dimension=10)
        
        embeddings = np.random.randn(50, 10).astype(np.float32)
        metadata = [
            HLEItem(hle_id=f"test_{i}", subject="math", question_text=f"Question {i}")
            for i in range(50)
        ]
        
        index.add(embeddings, metadata)
        
        query = np.random.randn(1, 10).astype(np.float32)
        
        results1 = index.search(query, k=5)
        results2 = index.search(query, k=5)
        
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.hle_id == r2.hle_id
            assert abs(r1.cosine_similarity - r2.cosine_similarity) < 1e-6