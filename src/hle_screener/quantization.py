"""
Model quantization support for reduced memory usage.
Supports 8-bit and 4-bit quantization using bitsandbytes.
"""

import logging
from typing import Optional, Literal, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import quantization libraries
HAS_BITSANDBYTES = False
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    logger.warning("bitsandbytes not installed, quantization not available")

HAS_ACCELERATE = False
try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    logger.warning("accelerate not installed, some quantization features unavailable")


class QuantizationConfig:
    """Configuration for model quantization."""
    
    def __init__(
        self,
        quantization: Optional[Literal["8bit", "4bit", "none"]] = "none",
        compute_dtype: str = "float16",
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ):
        """
        Initialize quantization configuration.
        
        Args:
            quantization: Quantization level ("8bit", "4bit", or "none")
            compute_dtype: Compute dtype for quantized models
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
            bnb_4bit_quant_type: Quantization type for 4-bit ("nf4" or "fp4")
            bnb_4bit_use_double_quant: Use double quantization for 4-bit
        """
        self.quantization = quantization
        self.compute_dtype = compute_dtype
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        
        # Validate configuration
        if quantization in ["8bit", "4bit"] and not HAS_BITSANDBYTES:
            raise ValueError(
                f"Quantization {quantization} requested but bitsandbytes not installed. "
                "Install with: pip install bitsandbytes"
            )
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model loading kwargs for transformers."""
        import torch
        
        kwargs = {}
        
        if self.quantization == "8bit":
            kwargs.update({
                "load_in_8bit": True,
                "device_map": "auto",
                "torch_dtype": getattr(torch, self.compute_dtype),
            })
        elif self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )
            kwargs.update({
                "quantization_config": bnb_config,
                "device_map": "auto",
                "torch_dtype": getattr(torch, self.compute_dtype),
            })
        else:
            # No quantization
            kwargs.update({
                "torch_dtype": getattr(torch, self.compute_dtype),
            })
        
        return kwargs


def load_quantized_model(
    model_id: str,
    quantization: Optional[Literal["8bit", "4bit", "none"]] = "none",
    cache_dir: Optional[Path] = None,
    **kwargs
):
    """
    Load a model with optional quantization.
    
    Args:
        model_id: HuggingFace model ID
        quantization: Quantization level
        cache_dir: Cache directory for models
        **kwargs: Additional arguments for model loading
    
    Returns:
        Loaded model with quantization applied
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create quantization config
    quant_config = QuantizationConfig(quantization=quantization)
    model_kwargs = quant_config.get_model_kwargs()
    model_kwargs.update(kwargs)
    
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)
    
    logger.info(f"Loading model {model_id} with quantization={quantization}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # Log memory usage
    if quantization != "none":
        import torch
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Model loaded with {quantization} quantization, using {memory_used:.2f} GB VRAM")
    
    return model, tokenizer


def estimate_memory_usage(
    model_id: str,
    quantization: Literal["8bit", "4bit", "none"] = "none"
) -> float:
    """
    Estimate memory usage for a model with given quantization.
    
    Args:
        model_id: HuggingFace model ID
        quantization: Quantization level
    
    Returns:
        Estimated memory usage in GB
    """
    # Base model sizes (approximate)
    model_sizes = {
        "Qwen/Qwen2.5-3B-Instruct": 3.0,  # 3B parameters
        "Qwen/Qwen2.5-7B-Instruct": 7.0,  # 7B parameters
        "Qwen/Qwen2.5-14B-Instruct": 14.0,  # 14B parameters
    }
    
    # Get base size
    base_size = model_sizes.get(model_id, 7.0)  # Default to 7B
    
    # Apply quantization factor
    if quantization == "8bit":
        # 8-bit reduces to ~1 byte per parameter
        estimated = base_size * 1.2  # Add 20% overhead
    elif quantization == "4bit":
        # 4-bit reduces to ~0.5 bytes per parameter
        estimated = base_size * 0.6  # Add 20% overhead
    else:
        # FP16 is 2 bytes per parameter
        estimated = base_size * 2.4  # Add 20% overhead
    
    return estimated


def check_quantization_support() -> Dict[str, bool]:
    """Check which quantization options are available."""
    import torch
    
    support = {
        "cuda_available": torch.cuda.is_available(),
        "bitsandbytes": HAS_BITSANDBYTES,
        "accelerate": HAS_ACCELERATE,
        "8bit": HAS_BITSANDBYTES and torch.cuda.is_available(),
        "4bit": HAS_BITSANDBYTES and torch.cuda.is_available(),
    }
    
    # Check CUDA compute capability for 4-bit
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        # 4-bit requires compute capability >= 7.5
        support["4bit"] = support["4bit"] and (capability[0] >= 7 and capability[1] >= 5)
    
    return support


if __name__ == "__main__":
    # Test quantization support
    support = check_quantization_support()
    print("Quantization Support:")
    for key, value in support.items():
        print(f"  {key}: {value}")
    
    # Test memory estimation
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    for quant in ["none", "8bit", "4bit"]:
        mem = estimate_memory_usage(model_id, quant)
        print(f"\n{model_id} with {quant}: ~{mem:.1f} GB")