"""
Model architectures including LLaMA and Mixture of Experts implementations.
"""

# Model components
from .models.llama import (
    LlamaConfig,
    LlamaForCausalLM,
    create_llama_7b_parallel,
    create_llama_13b_parallel,
    create_llama_30b_parallel,
    create_llama_65b_parallel,
    create_llama2_7b_parallel,
    create_code_llama_7b_parallel,
    # LLaMA 3 models
    create_llama3_8b_parallel,
    create_llama3_8b_instruct_parallel,
    create_llama3_70b_parallel,
    create_llama3_70b_instruct_parallel,
    create_llama3_405b_parallel,
    # Tiny LLaMA 3 for development
    create_tiny_llama3_150m,
    create_tiny_llama3_50m,
    optimize_model_for_training,
    estimate_model_memory
)

from .moe import (
    LlamaMoEModel,
    MoETransformerLayer,
    MoELlamaLayer,
    TopKRouter,
    create_llama_moe_7b,
    create_switch_transformer,
    create_glam_model
)

__all__ = [
    # LLaMA models
    "LlamaModel",
    "LlamaConfig", 
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaRMSNorm",
    "create_llama_7b",
    "create_llama_13b",
    "create_llama_30b",
    "create_llama_65b",
    "create_llama2_7b",
    "create_code_llama_7b",
    
    # MoE models
    "LlamaMoEModel",
    "MoETransformerLayer",
    "SparseExpertLayer",
    "Router",
    "create_llama_moe_7b",
    "create_switch_transformer",
    "create_glam_model",
]

# Model registry for dynamic creation
MODEL_REGISTRY = {
    "llama_7b": create_llama_7b,
    "llama_13b": create_llama_13b,
    "llama_30b": create_llama_30b,
    "llama_65b": create_llama_65b,
    "llama_2_7b": create_llama2_7b,
    "code_llama": create_code_llama_7b,
    "llama_moe_7b": create_llama_moe_7b,
}

def create_model(model_name: str, **kwargs):
    """Create a model by name from the registry."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    return MODEL_REGISTRY[model_name](**kwargs)

def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())