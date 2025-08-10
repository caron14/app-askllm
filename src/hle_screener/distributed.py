"""
Distributed batch processing for parallel scoring.
Uses multiprocessing for CPU-based parallelism and Ray for distributed computing.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Ray for distributed computing
HAS_RAY = False
try:
    import ray
    HAS_RAY = True
except ImportError:
    logger.info("Ray not installed, distributed computing not available")


@dataclass
class WorkItem:
    """A single work item for processing."""
    item_id: str
    text: str
    chunk_id: int
    total_chunks: int


class DistributedScorer:
    """Distributed scoring using multiprocessing or Ray."""
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_ray: bool = False,
        ray_address: Optional[str] = None
    ):
        """
        Initialize distributed scorer.
        
        Args:
            num_workers: Number of parallel workers (defaults to CPU count)
            use_ray: Use Ray for distributed computing
            ray_address: Ray cluster address (for remote clusters)
        """
        self.num_workers = num_workers or cpu_count()
        self.use_ray = use_ray and HAS_RAY
        self.ray_address = ray_address
        
        if self.use_ray:
            self._init_ray()
    
    def _init_ray(self):
        """Initialize Ray runtime."""
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address)
                logger.info(f"Connected to Ray cluster at {self.ray_address}")
            else:
                ray.init(num_cpus=self.num_workers)
                logger.info(f"Initialized local Ray with {self.num_workers} CPUs")
    
    def score_batch_distributed(
        self,
        items: List[Dict[str, Any]],
        config_path: str,
        output_dir: Path,
        quantization: str = "none",
        checkpoint_frequency: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Score items in parallel using distributed processing.
        
        Args:
            items: List of items to score
            config_path: Path to configuration file
            output_dir: Output directory for results
            quantization: Model quantization level
            checkpoint_frequency: Save checkpoint every N items
        
        Returns:
            List of scoring results
        """
        if self.use_ray:
            return self._score_batch_ray(
                items, config_path, output_dir, quantization, checkpoint_frequency
            )
        else:
            return self._score_batch_multiprocessing(
                items, config_path, output_dir, quantization, checkpoint_frequency
            )
    
    def _score_batch_multiprocessing(
        self,
        items: List[Dict[str, Any]],
        config_path: str,
        output_dir: Path,
        quantization: str,
        checkpoint_frequency: int
    ) -> List[Dict[str, Any]]:
        """Score batch using multiprocessing."""
        
        # Create work chunks
        chunk_size = max(1, len(items) // self.num_workers)
        chunks = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunks.append((chunk, i // chunk_size, len(chunks)))
        
        logger.info(f"Processing {len(items)} items in {len(chunks)} chunks with {self.num_workers} workers")
        
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _process_chunk,
                    chunk,
                    config_path,
                    output_dir,
                    quantization,
                    chunk_id,
                    total_chunks
                ): chunk_id
                for chunk, chunk_id, total_chunks in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    completed += len(chunk_results)
                    
                    logger.info(f"Completed chunk {chunk_id + 1}/{len(chunks)} ({completed}/{len(items)} items)")
                    
                    # Save checkpoint if needed
                    if completed % checkpoint_frequency == 0:
                        self._save_checkpoint(results, output_dir, completed, len(items))
                
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id}: {e}")
        
        return results
    
    def _score_batch_ray(
        self,
        items: List[Dict[str, Any]],
        config_path: str,
        output_dir: Path,
        quantization: str,
        checkpoint_frequency: int
    ) -> List[Dict[str, Any]]:
        """Score batch using Ray distributed computing."""
        
        if not HAS_RAY:
            raise ImportError("Ray is not installed. Install with: pip install ray")
        
        # Create Ray remote function
        @ray.remote
        def process_chunk_ray(chunk, config_path, output_dir, quantization, chunk_id, total_chunks):
            return _process_chunk(chunk, config_path, output_dir, quantization, chunk_id, total_chunks)
        
        # Create work chunks
        chunk_size = max(1, len(items) // self.num_workers)
        chunks = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunks.append((chunk, i // chunk_size))
        
        logger.info(f"Processing {len(items)} items in {len(chunks)} chunks using Ray")
        
        # Submit all tasks to Ray
        futures = [
            process_chunk_ray.remote(
                chunk,
                config_path,
                output_dir,
                quantization,
                chunk_id,
                len(chunks)
            )
            for chunk, chunk_id in chunks
        ]
        
        # Collect results
        results = []
        completed = 0
        
        while futures:
            # Wait for any task to complete
            ready, futures = ray.wait(futures, num_returns=1)
            
            for ref in ready:
                chunk_results = ray.get(ref)
                results.extend(chunk_results)
                completed += len(chunk_results)
                
                logger.info(f"Completed {completed}/{len(items)} items")
                
                # Save checkpoint if needed
                if completed % checkpoint_frequency == 0:
                    self._save_checkpoint(results, output_dir, completed, len(items))
        
        return results
    
    def _save_checkpoint(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path,
        completed: int,
        total: int
    ):
        """Save checkpoint file."""
        checkpoint_path = output_dir / f"checkpoint_{completed}_{total}.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump({
                "completed": completed,
                "total": total,
                "results": results
            }, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def cleanup(self):
        """Clean up distributed resources."""
        if self.use_ray and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")


def _process_chunk(
    chunk: List[Dict[str, Any]],
    config_path: str,
    output_dir: Path,
    quantization: str,
    chunk_id: int,
    total_chunks: int
) -> List[Dict[str, Any]]:
    """
    Process a chunk of items (runs in separate process).
    
    Args:
        chunk: Items to process
        config_path: Path to configuration
        output_dir: Output directory
        quantization: Model quantization level
        chunk_id: ID of this chunk
        total_chunks: Total number of chunks
    
    Returns:
        List of results
    """
    # Import here to avoid circular imports and ensure clean process
    from .config import load_config
    from .index import FAISSIndex
    from .embed import create_embedding_client
    from .askllm import create_askllm_judge
    from .score import QualityScorer
    
    logger.info(f"Worker {chunk_id + 1}/{total_chunks}: Processing {len(chunk)} items")
    
    # Load configuration
    cfg = load_config(config_path)
    
    # Load index
    index = FAISSIndex.load(
        Path("artifacts/indexes/faiss.index"),
        Path("artifacts/eval_only/DO_NOT_TRAIN/hle_metadata.json")
    )
    
    # Create models
    embedding_client = create_embedding_client(config=cfg)
    judge = create_askllm_judge(config=cfg, quantization=quantization)
    
    # Create scorer
    scorer = QualityScorer(index, embedding_client, judge, cfg)
    
    # Score items
    results = []
    for i, item in enumerate(chunk):
        try:
            result = scorer.score(item["id"], item["question"])
            results.append(result.dict())
            
            if (i + 1) % 10 == 0:
                logger.info(f"Worker {chunk_id + 1}: Completed {i + 1}/{len(chunk)} items")
        
        except Exception as e:
            logger.error(f"Worker {chunk_id + 1}: Error scoring item {item['id']}: {e}")
            results.append({
                "item_id": item["id"],
                "error": str(e),
                "quality_score": 0.0
            })
    
    logger.info(f"Worker {chunk_id + 1}/{total_chunks}: Completed all {len(chunk)} items")
    return results


def estimate_optimal_workers(
    num_items: int,
    has_gpu: bool = False,
    memory_per_worker_gb: float = 2.0
) -> int:
    """
    Estimate optimal number of workers based on system resources.
    
    Args:
        num_items: Number of items to process
        has_gpu: Whether GPU is available
        memory_per_worker_gb: Estimated memory per worker in GB
    
    Returns:
        Recommended number of workers
    """
    import psutil
    
    # Get system resources
    num_cpus = cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Calculate based on memory
    max_workers_memory = int(available_memory_gb / memory_per_worker_gb)
    
    # If GPU is available, use fewer workers (GPU operations don't parallelize well)
    if has_gpu:
        max_workers = min(2, max_workers_memory, num_cpus // 2)
    else:
        max_workers = min(max_workers_memory, num_cpus)
    
    # Don't use more workers than items
    max_workers = min(max_workers, num_items)
    
    # Ensure at least 1 worker
    max_workers = max(1, max_workers)
    
    logger.info(
        f"System: {num_cpus} CPUs, {total_memory_gb:.1f}GB RAM, "
        f"{available_memory_gb:.1f}GB available"
    )
    logger.info(f"Recommended workers: {max_workers}")
    
    return max_workers


if __name__ == "__main__":
    # Test distributed scorer
    import torch
    
    print("Testing distributed scorer...")
    print(f"Has Ray: {HAS_RAY}")
    print(f"Has GPU: {torch.cuda.is_available()}")
    
    # Estimate workers
    workers = estimate_optimal_workers(100, torch.cuda.is_available())
    print(f"Optimal workers for 100 items: {workers}")