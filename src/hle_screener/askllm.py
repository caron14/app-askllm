import json
import torch
from typing import Dict, Any, List, Optional, Tuple, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

from .schema import RetrievalResult, AskLLMResponse
from .utils import ModelCache, format_hle_context
from .quantization import load_quantized_model, check_quantization_support

logger = logging.getLogger(__name__)


class AskLLMJudge:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 64,
        quantization: Optional[Literal["8bit", "4bit", "none"]] = "none"
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        logger.info(f"Loading LLM judge model: {self.model_id} with quantization={self.quantization}")
        
        # Check if quantization is supported
        if self.quantization != "none":
            support = check_quantization_support()
            if not support[self.quantization]:
                logger.warning(f"Quantization {self.quantization} not supported, falling back to none")
                self.quantization = "none"
        
        # Try to get from cache first
        cache_key = f"llm_{self.quantization}"
        cached = ModelCache.get(cache_key)
        
        if cached:
            self.model, self.tokenizer = cached
        else:
            # Load with quantization if specified
            if self.quantization != "none":
                self.model, self.tokenizer = load_quantized_model(
                    self.model_id,
                    quantization=self.quantization
                )
                ModelCache.set(cache_key, (self.model, self.tokenizer))
            else:
                # Use original loading method
                self.model, self.tokenizer = ModelCache.get_or_create(
                    "llm",
                    self.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
    
    def _create_prompt(self, references: List[RetrievalResult], candidate_text: str) -> str:
        system_prompt = """You are a data quality rater evaluating whether a synthetic question would be useful for improving performance on high-quality evaluation questions.

Output strict JSON only with this exact format:
{"yes_probability": 0.0-1.0, "rationale_short": "max 50 words"}

Do not include any other text, only the JSON object."""
        
        reference_context = "Reference HLE Questions:\n\n"
        for i, ref in enumerate(references, 1):
            reference_context += f"{i}. Subject: {ref.subject}\n"
            reference_context += f"   Question: {ref.question_text}\n\n"
        
        user_prompt = f"""{reference_context}

Candidate Question to Evaluate:
{candidate_text}

Evaluation Rubric:
- Does the candidate question test similar concepts or skills as the reference questions?
- Would training on this candidate improve performance on questions like the references?
- Consider topic relevance, difficulty level, and problem-solving approach.

Rate the usefulness of this candidate question for improving performance on the reference questions."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
        
        return prompt
    
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    def _generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else 1e-7,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _parse_json_response(self, response: str) -> AskLLMResponse:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            data = json.loads(response)
            
            yes_prob = float(data.get("yes_probability", 0.5))
            yes_prob = max(0.0, min(1.0, yes_prob))
            
            rationale = data.get("rationale_short", "No rationale provided")
            if len(rationale) > 200:
                rationale = rationale[:197] + "..."
            
            return AskLLMResponse(
                yes_probability=yes_prob,
                rationale_short=rationale
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}. Response: {response[:200]}")
            return AskLLMResponse(
                yes_probability=0.5,
                rationale_short="Failed to parse model response"
            )
    
    def judge(
        self,
        references: List[RetrievalResult],
        candidate_text: str
    ) -> AskLLMResponse:
        prompt = self._create_prompt(references, candidate_text)
        
        try:
            response = self._generate_response(prompt)
            result = self._parse_json_response(response)
            
            logger.info(f"Judge decision: p={result.yes_probability:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during judgment: {e}")
            return AskLLMResponse(
                yes_probability=0.5,
                rationale_short=f"Error during judgment: {str(e)[:100]}"
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "device": str(self.device)
        }


def create_askllm_judge(config: Optional[Dict[str, Any]] = None, quantization: Optional[str] = None) -> AskLLMJudge:
    if config:
        return AskLLMJudge(
            model_id=config.get("judge_model", "Qwen/Qwen2.5-3B-Instruct"),
            temperature=config.get("temperature", 0.0),
            max_new_tokens=config.get("max_new_tokens", 64),
            quantization=quantization or config.get("quantization", "none")
        )
    return AskLLMJudge(quantization=quantization or "none")