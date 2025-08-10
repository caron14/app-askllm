"""
Caching layer for embeddings to avoid recomputation.
Supports file-based and Redis-based caching.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import Redis for distributed caching
HAS_REDIS = False
try:
    import redis
    HAS_REDIS = True
except ImportError:
    logger.info("Redis not installed, using file-based caching only")


class EmbeddingCache:
    """Cache for storing and retrieving embeddings."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_redis: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        ttl: int = 86400  # 24 hours default TTL
    ):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for file-based cache
            use_redis: Use Redis for caching
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            ttl: Time-to-live for cache entries in seconds
        """
        self.cache_dir = cache_dir or Path("artifacts/cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_redis = use_redis and HAS_REDIS
        self.ttl = ttl
        
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False
                )
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, falling back to file cache")
                self.use_redis = False
                self.redis_client = None
        else:
            self.redis_client = None
        
        # In-memory cache for current session
        self.memory_cache: Dict[str, np.ndarray] = {}
    
    def _get_cache_key(self, text: str, model_id: str) -> str:
        """Generate cache key from text and model ID."""
        content = f"{model_id}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model_id: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Text that was embedded
            model_id: Model used for embedding
        
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text, model_id)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {cache_key[:8]}")
            return self.memory_cache[cache_key]
        
        # Check Redis if available
        if self.use_redis and self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    embedding = pickle.loads(cached_data)
                    self.memory_cache[cache_key] = embedding
                    logger.debug(f"Cache hit (Redis): {cache_key[:8]}")
                    return embedding
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                self.memory_cache[cache_key] = embedding
                logger.debug(f"Cache hit (file): {cache_key[:8]}")
                return embedding
            except Exception as e:
                logger.warning(f"File cache read error: {e}")
        
        logger.debug(f"Cache miss: {cache_key[:8]}")
        return None
    
    def set(self, text: str, model_id: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            text: Text that was embedded
            model_id: Model used for embedding
            embedding: The embedding vector
        """
        cache_key = self._get_cache_key(text, model_id)
        
        # Store in memory cache
        self.memory_cache[cache_key] = embedding
        
        # Store in Redis if available
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.ttl,
                    pickle.dumps(embedding)
                )
                logger.debug(f"Cached to Redis: {cache_key[:8]}")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        # Store in file cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
            logger.debug(f"Cached to file: {cache_key[:8]}")
        except Exception as e:
            logger.warning(f"File cache write error: {e}")
    
    def get_batch(self, texts: list, model_id: str) -> Dict[int, np.ndarray]:
        """
        Get embeddings for multiple texts from cache.
        
        Args:
            texts: List of texts
            model_id: Model used for embedding
        
        Returns:
            Dictionary mapping indices to cached embeddings
        """
        cached = {}
        for i, text in enumerate(texts):
            embedding = self.get(text, model_id)
            if embedding is not None:
                cached[i] = embedding
        
        return cached
    
    def set_batch(self, texts: list, model_id: str, embeddings: np.ndarray):
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of texts
            model_id: Model used for embedding
            embeddings: Array of embeddings
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, model_id, embedding)
    
    def clear(self):
        """Clear all caches."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear Redis cache
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("Cleared Redis cache")
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")
        
        logger.info("Cleared all embedding caches")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_entries": len(self.memory_cache),
            "memory_size_mb": sum(
                e.nbytes for e in self.memory_cache.values()
            ) / (1024 * 1024),
            "file_entries": len(list(self.cache_dir.glob("*.npy"))),
            "redis_available": self.use_redis,
        }
        
        if self.use_redis and self.redis_client:
            try:
                stats["redis_entries"] = self.redis_client.dbsize()
            except:
                stats["redis_entries"] = 0
        
        return stats


class CachedEmbeddingClient:
    """Wrapper for EmbeddingClient with caching."""
    
    def __init__(
        self,
        embedding_client,
        cache: Optional[EmbeddingCache] = None
    ):
        """
        Initialize cached embedding client.
        
        Args:
            embedding_client: Base embedding client
            cache: EmbeddingCache instance
        """
        self.client = embedding_client
        self.cache = cache or EmbeddingCache()
    
    def encode(
        self,
        texts: Union[str, list],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode texts with caching.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Normalize embeddings
        
        Returns:
            Embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache for all texts
        cached = self.cache.get_batch(texts, self.client.model_id)
        
        # Find texts that need encoding
        to_encode = []
        to_encode_indices = []
        
        for i, text in enumerate(texts):
            if i not in cached:
                to_encode.append(text)
                to_encode_indices.append(i)
        
        # Encode missing texts
        if to_encode:
            logger.info(f"Encoding {len(to_encode)} texts (cached: {len(cached)})")
            new_embeddings = self.client.encode(
                to_encode,
                batch_size=batch_size,
                show_progress=show_progress,
                normalize_embeddings=normalize_embeddings
            )
            
            # Cache new embeddings
            self.cache.set_batch(to_encode, self.client.model_id, new_embeddings)
            
            # Add to cached dict
            for idx, embedding in zip(to_encode_indices, new_embeddings):
                cached[idx] = embedding
        else:
            logger.info(f"All {len(texts)} texts found in cache")
        
        # Reconstruct full embeddings array in original order
        embeddings = np.array([cached[i] for i in range(len(texts))])
        
        return embeddings
    
    def encode_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Encode batch of texts with caching."""
        return self.encode(texts, batch_size=batch_size)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


if __name__ == "__main__":
    # Test caching
    cache = EmbeddingCache()
    
    # Test basic operations
    test_text = "This is a test sentence."
    test_model = "test-model"
    test_embedding = np.random.randn(384)
    
    # Test set and get
    cache.set(test_text, test_model, test_embedding)
    retrieved = cache.get(test_text, test_model)
    
    assert retrieved is not None
    assert np.allclose(retrieved, test_embedding)
    
    print("Cache test passed!")
    print(f"Cache stats: {cache.get_stats()}")