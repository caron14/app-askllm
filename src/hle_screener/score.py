import math
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
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

from .schema import RetrievalResult, ScoringResult, AskLLMResponse
from .index import FAISSIndex, retrieve_similar_hle
from .embed import EmbeddingClient
from .askllm import AskLLMJudge
from .io import save_results_batch
from .utils import get_reproducibility_info, save_json_with_canary

logger = logging.getLogger(__name__)


def calculate_similarity_score(references: List[RetrievalResult]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for i, ref in enumerate(references):
        weight = 1.0 / math.log2(i + 2)
        weighted_sum += weight * ref.cosine_similarity
        weight_sum += weight
    
    if weight_sum > 0:
        return weighted_sum / weight_sum
    return 0.0


def calculate_quality_score(
    askllm_score: float,
    sim_score: float,
    alpha: float = 0.8
) -> float:
    quality = alpha * askllm_score + (1 - alpha) * (100 * sim_score)
    return round(quality, 1)


def score_single_item(
    item_id: str,
    item_text: str,
    index: FAISSIndex,
    embedding_client: EmbeddingClient,
    judge: AskLLMJudge,
    config: Dict[str, Any],
    k: int = 5
) -> ScoringResult:
    logger.info(f"Scoring item: {item_id}")
    
    references = retrieve_similar_hle(item_text, index, embedding_client, k=k)
    
    askllm_response = judge.judge(references, item_text)
    
    sim_score = calculate_similarity_score(references)
    askllm_score = round(askllm_response.yes_probability * 100, 1)
    quality_score = calculate_quality_score(
        askllm_score,
        sim_score,
        alpha=config.get("alpha", 0.8)
    )
    
    model_conf = {
        **get_reproducibility_info(config),
        **judge.get_model_info()
    }
    
    result = ScoringResult(
        item_id=item_id,
        item_text=item_text,
        top5_references=references,
        sim_score=sim_score,
        askllm_score=askllm_score,
        quality_score=quality_score,
        askllm_response=askllm_response,
        model_configuration=model_conf
    )
    
    logger.info(f"Scored {item_id}: quality={quality_score:.1f}, askllm={askllm_score:.1f}, sim={sim_score:.3f}")
    
    return result


def load_or_create_run_state(output_path: Path, run_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    state_file = output_path / ".run_state.json"
    
    if run_id and state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
            if state.get("run_id") == run_id:
                logger.info(f"Resuming run {run_id} from item {state.get('last_completed_idx', -1) + 1}")
                return run_id, state
    
    if not run_id and state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
            if state.get("status") != "completed":
                run_id = state.get("run_id")
                logger.info(f"Auto-resuming previous run {run_id} from item {state.get('last_completed_idx', -1) + 1}")
                return run_id, state
    
    new_run_id = run_id or str(uuid.uuid4())[:8]
    new_state = {
        "run_id": new_run_id,
        "started_at": datetime.utcnow().isoformat(),
        "last_completed_idx": -1,
        "last_completed_id": None,
        "total_items": 0,
        "completed_items": [],
        "failed_items": [],
        "status": "in_progress"
    }
    
    logger.info(f"Starting new run: {new_run_id}")
    return new_run_id, new_state


def save_run_state(output_path: Path, state: Dict[str, Any]):
    state_file = output_path / ".run_state.json"
    state["updated_at"] = datetime.utcnow().isoformat()
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def score_batch(
    items: List[Dict[str, str]],
    index: FAISSIndex,
    embedding_client: EmbeddingClient,
    judge: AskLLMJudge,
    config: Dict[str, Any],
    output_path: Optional[Path] = None,
    resume_from: Optional[str] = None,
    run_id: Optional[str] = None
) -> List[ScoringResult]:
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    run_id, state = load_or_create_run_state(output_path, run_id) if output_path else (str(uuid.uuid4())[:8], {})
    
    state["total_items"] = len(items)
    results = []
    start_idx = 0
    
    if resume_from:
        for i, item in enumerate(items):
            if item.get("id") == resume_from:
                start_idx = i
                logger.info(f"Manual resume from item {start_idx}: {resume_from}")
                break
    elif state.get("last_completed_idx", -1) >= 0:
        start_idx = state["last_completed_idx"] + 1
        
        checkpoint_file = output_path / f"run_{run_id}_checkpoint_{state['last_completed_idx'] + 1}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = [ScoringResult(**r) for r in checkpoint_data.get("results", [])]
                logger.info(f"Loaded {len(results)} previous results from checkpoint")
    
    checkpoint_frequency = config.get("checkpoint_frequency", 10)
    
    # Create progress bar
    progress_bar = tqdm(
        enumerate(items[start_idx:], start_idx),
        total=len(items) - start_idx,
        desc="Scoring items",
        unit="item",
        initial=0,
        disable=not HAS_TQDM
    )
    
    for i, item in progress_bar:
        item_id = item.get("id", f"item_{i}")
        
        if item_id in state.get("completed_items", []):
            logger.info(f"Skipping already completed item: {item_id}")
            continue
        
        try:
            # Update progress bar description
            if HAS_TQDM:
                progress_bar.set_description(f"Scoring {item_id}")
            else:
                logger.info(f"Processing item {i+1}/{len(items)}: {item_id}")
            
            result = score_single_item(
                item_id=item_id,
                item_text=item.get("question", item.get("text", "")),
                index=index,
                embedding_client=embedding_client,
                judge=judge,
                config=config
            )
            results.append(result)
            
            state["last_completed_idx"] = i
            state["last_completed_id"] = item_id
            state["completed_items"].append(item_id)
            
            if output_path:
                if (i + 1) % checkpoint_frequency == 0 or (i + 1) == len(items):
                    checkpoint_file = output_path / f"run_{run_id}_checkpoint_{i+1}.json"
                    save_results_batch(
                        [r.dict() for r in results],
                        checkpoint_file
                    )
                    logger.info(f"Saved checkpoint at item {i+1}")
                
                save_run_state(output_path, state)
                
        except Exception as e:
            logger.error(f"Error scoring item {i} ({item_id}): {e}")
            state["failed_items"].append({"id": item_id, "error": str(e)})
            if output_path:
                save_run_state(output_path, state)
            continue
    
    if output_path:
        state["status"] = "completed"
        state["completed_at"] = datetime.utcnow().isoformat()
        save_run_state(output_path, state)
        
        final_path = output_path / f"run_{run_id}_final.json"
        save_results_batch([r.dict() for r in results], final_path)
        logger.info(f"Saved final results to {final_path}")
        
        summary_path = output_path / f"run_{run_id}_summary.json"
        summary = {
            "run_id": run_id,
            "total_items": len(items),
            "completed_items": len(results),
            "failed_items": len(state.get("failed_items", [])),
            "average_quality_score": sum(r.quality_score for r in results) / len(results) if results else 0,
            "started_at": state.get("started_at"),
            "completed_at": state.get("completed_at"),
            "config": config
        }
        save_json_with_canary(summary, summary_path)
        logger.info(f"Run {run_id} completed: {len(results)}/{len(items)} items processed")
    
    return results


class QualityScorer:
    def __init__(
        self,
        index: FAISSIndex,
        embedding_client: EmbeddingClient,
        judge: AskLLMJudge,
        config: Dict[str, Any]
    ):
        self.index = index
        self.embedding_client = embedding_client
        self.judge = judge
        self.config = config
    
    def score(self, item_id: str, item_text: str) -> ScoringResult:
        return score_single_item(
            item_id=item_id,
            item_text=item_text,
            index=self.index,
            embedding_client=self.embedding_client,
            judge=self.judge,
            config=self.config
        )
    
    def score_batch(
        self,
        items: List[Dict[str, str]],
        output_path: Optional[Path] = None,
        resume_from: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> List[ScoringResult]:
        return score_batch(
            items=items,
            index=self.index,
            embedding_client=self.embedding_client,
            judge=self.judge,
            config=self.config,
            output_path=output_path,
            resume_from=resume_from,
            run_id=run_id
        )