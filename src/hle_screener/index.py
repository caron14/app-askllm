import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback dummy tqdm
    def tqdm(iterable, **kwargs):
        return iterable

from .schema import HLEItem, RetrievalResult
from .embed import EmbeddingClient
from .io import save_embeddings, load_embeddings, save_hle_metadata, load_hle_metadata

logger = logging.getLogger(__name__)


class FAISSIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[HLEItem] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add(self, embeddings: np.ndarray, metadata: List[HLEItem]):
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata items")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(embeddings_normalized.astype('float32'))
        self.metadata.extend(metadata)
        
        if self.embeddings is None:
            self.embeddings = embeddings_normalized
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings_normalized])
        
        logger.info(f"Added {len(metadata)} items to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        if query_embedding.shape[-1] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[-1]} doesn't match index dimension {self.dimension}")
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        similarities, indices = self.index.search(query_normalized.astype('float32'), k)
        
        results = []
        for rank, (idx, sim) in enumerate(zip(indices[0], similarities[0]), 1):
            if idx >= 0 and idx < len(self.metadata):
                item = self.metadata[idx]
                results.append(RetrievalResult(
                    hle_id=item.hle_id,
                    subject=item.subject,
                    question_text=item.question_text,
                    cosine_similarity=float(sim),
                    rank=rank
                ))
        
        return results
    
    def save(self, index_path: Path, metadata_path: Path):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        save_hle_metadata(self.metadata, metadata_path)
        
        if self.embeddings is not None:
            embeddings_path = index_path.parent / "embeddings.pkl"
            save_embeddings(self.embeddings, {"dimension": self.dimension}, embeddings_path)
    
    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> 'FAISSIndex':
        index = faiss.read_index(str(index_path))
        metadata = load_hle_metadata(metadata_path)
        
        dimension = index.d
        faiss_index = cls(dimension)
        faiss_index.index = index
        faiss_index.metadata = metadata
        
        embeddings_path = index_path.parent / "embeddings.pkl"
        if embeddings_path.exists():
            embeddings, _ = load_embeddings(embeddings_path)
            faiss_index.embeddings = embeddings
        
        logger.info(f"Loaded FAISS index with {len(metadata)} items")
        return faiss_index


def build_hle_index(
    hle_items: List[HLEItem],
    embedding_client: EmbeddingClient,
    save_path: Optional[Path] = None
) -> FAISSIndex:
    logger.info(f"Building index for {len(hle_items)} HLE items")
    
    texts = [item.question_text for item in hle_items]
    embeddings = embedding_client.encode_batch(texts, batch_size=32)
    
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.add(embeddings, hle_items)
    
    if save_path:
        index_path = save_path / "faiss.index"
        metadata_path = save_path.parent / "eval_only/DO_NOT_TRAIN/hle_metadata.json"
        index.save(index_path, metadata_path)
    
    return index


def retrieve_similar_hle(
    query_text: str,
    index: FAISSIndex,
    embedding_client: EmbeddingClient,
    k: int = 5
) -> List[RetrievalResult]:
    query_embedding = embedding_client.encode(query_text, show_progress=False)
    results = index.search(query_embedding, k=k)
    
    logger.info(f"Retrieved top-{k} similar HLE items for query")
    return results