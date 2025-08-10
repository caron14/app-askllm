import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import torch
import logging
from tqdm import tqdm

from .utils import ModelCache

logger = logging.getLogger(__name__)


class EmbeddingClient:
    # Supported embedding models with their dimensions
    SUPPORTED_MODELS = {
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-small-v2": 384,
        "intfloat/multilingual-e5-large": 1024,
        "thenlper/gte-large": 1024,
        "thenlper/gte-base": 768,
        "thenlper/gte-small": 384,
    }
    
    def __init__(self, model_id: str = "BAAI/bge-m3", device: Optional[str] = None):
        if model_id not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_id} not in supported list, using anyway")
        
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dimension = self.SUPPORTED_MODELS.get(model_id, None)
        self._load_model()
    
    def _load_model(self):
        logger.info(f"Loading embedding model: {self.model_id}")
        self.model = ModelCache.get_or_create("embedding", self.model_id)
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts with {self.model_id}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        if normalize_embeddings:
            embeddings = normalize(embeddings, norm='l2', axis=1)
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)
    
    @property
    def embedding_dim(self) -> int:
        test_embedding = self.encode("test", show_progress=False)
        return test_embedding.shape[1]


def create_embedding_client(model_id: Optional[str] = None, config: Optional[dict] = None) -> EmbeddingClient:
    if config and not model_id:
        model_id = config.get("embed_model", "BAAI/bge-m3")
    
    model_id = model_id or "BAAI/bge-m3"
    
    return EmbeddingClient(model_id=model_id)


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return float(np.dot(vec1_norm, vec2_norm))


def batch_compute_similarities(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    
    corpus_norms = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    
    similarities = np.dot(corpus_norms, query_norm)
    
    return similarities