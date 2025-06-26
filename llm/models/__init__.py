"""
Model architectures including LLaMA and Mixture of Experts implementations.
"""

from .llama import (
    LlamaModel,
    LlamaConfig,
    LlamaAttention,
    LlamaDecoderLayer,
    RMSNorm,
    create_llama_7b,
    create_llama_13b,
    create_llama_30b,
    create_llama_65b,
    create_llama2_7b,
    create_code_llama_7b
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