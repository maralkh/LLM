# training_infra/models/__init__.py
"""
Model architectures and implementations for LLaMA training.

This module provides LLaMA model implementations with support for distributed
training, including all variants from original LLaMA to LLaMA 3 and Tiny models.
"""

# Try importing base model components first
try:
    from .base import (
        BaseModel,
        LanguageModel,
        ModelConfig
    )
    _BASE_AVAILABLE = True
except ImportError:
    _BASE_AVAILABLE = False

# Try importing LLaMA models
try:
    from .llama import (
        LLaMAConfig,
        RMSNorm,
        RotaryPositionalEmbedding,
        LLaMAAttention,
        LLaMAMLP,
        LLaMADecoderLayer,
        LLaMAModel,
        LLaMAForCausalLM,
        
        # Model creation functions
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
        
        # Tiny LLaMA 3 models
        create_tiny_llama3_150m,
        create_tiny_llama3_50m,
        
        # Utilities
        optimize_model_for_training,
        estimate_model_memory,
        apply_rotary_pos_emb,
        rotate_half
    )
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False

# Try importing MoE models
try:
    from .moe import (
        # Configuration
        LlamaMoEConfig,
        create_tiny_llama_moe,
        create_llama_moe_7b,
        create_llama_moe_13b,
        create_llama_moe_30b,
        create_llama_moe_65b,
        create_code_llama_moe_7b,
        get_llama_moe_config,
        LLAMA_MOE_CONFIGS,
        
        # Components (reuse LLaMA attention)
        LlamaMoEAttention,
        LlamaMoEMLP,
        LlamaMoEExpertMLP,
        
        # MoE specific
        LlamaMoEOutput,
        LlamaMoETopKRouter,
        LlamaMoESwitchRouter,
        LlamaMoELayer,
        LlamaMoEEfficientLayer,
        
        # Models
        LlamaMoEModelOutput,
        LlamaMoECausalLMOutput,
        LlamaMoEDecoderLayer,
        LlamaMoEModel,
        LlamaMoEForCausalLM,
        create_tiny_llama_moe_model,
        create_llama_moe_7b_model,
        create_llama_moe_13b_model,
        create_code_llama_moe_7b_model,
        
        # Utilities
        compute_load_balancing_loss,
        analyze_expert_utilization,
        compute_moe_efficiency,
    )
    _MOE_AVAILABLE = True
except ImportError:
    _MOE_AVAILABLE = False

# Try importing model utilities
try:
    from .utils import (
        WeightInitializer,
        init_linear_layer,
        init_embedding_layer,
        apply_initialization,
        create_cache
    )
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False

# Core exports - start with empty list and add based on availability
__all__ = []

# Add base exports if available
if _BASE_AVAILABLE:
    __all__.extend([
        "BaseModel",
        "LanguageModel", 
        "ModelConfig"
    ])

# Add LLaMA exports if available
if _LLAMA_AVAILABLE:
    __all__.extend([
        # Configuration
        "LLaMAConfig",
        
        # Model components
        "RMSNorm",
        "RotaryPositionalEmbedding", 
        "LLaMAAttention",
        "LLaMAMLP",
        "LLaMADecoderLayer",
        "LLaMAModel",
        "LLaMAForCausalLM",
        
        # Model creation functions
        "create_llama_7b_parallel",
        "create_llama_13b_parallel",
        "create_llama_30b_parallel", 
        "create_llama_65b_parallel",
        "create_llama2_7b_parallel",
        "create_code_llama_7b_parallel",
        
        # LLaMA 3 models
        "create_llama3_8b_parallel",
        "create_llama3_8b_instruct_parallel",
        "create_llama3_70b_parallel",
        "create_llama3_70b_instruct_parallel", 
        "create_llama3_405b_parallel",
        
        # Tiny LLaMA 3 models
        "create_tiny_llama3_150m",
        "create_tiny_llama3_50m",
        
        # Utilities
        "optimize_model_for_training",
        "estimate_model_memory",
        "apply_rotary_pos_emb",
        "rotate_half",
    ])

# Add MoE exports if available
if _MOE_AVAILABLE:
    __all__.extend([
        # MoE Configuration
        "LlamaMoEConfig",
        "create_tiny_llama_moe",
        "create_llama_moe_7b",
        "create_llama_moe_13b",
        "create_llama_moe_30b",
        "create_llama_moe_65b",
        "create_code_llama_moe_7b",
        "get_llama_moe_config",
        "LLAMA_MOE_CONFIGS",
        
        # MoE Components
        "LlamaMoEAttention",
        "LlamaMoEMLP",
        "LlamaMoEExpertMLP",
        
        # MoE Routing
        "LlamaMoEOutput",
        "LlamaMoETopKRouter",
        "LlamaMoESwitchRouter",
        "LlamaMoELayer",
        "LlamaMoEEfficientLayer",
        
        # MoE Models
        "LlamaMoEModelOutput",
        "LlamaMoECausalLMOutput",
        "LlamaMoEDecoderLayer",
        "LlamaMoEModel",
        "LlamaMoEForCausalLM",
        "create_tiny_llama_moe_model",
        "create_llama_moe_7b_model",
        "create_llama_moe_13b_model",
        "create_code_llama_moe_7b_model",
        
        # MoE Utilities
        "compute_load_balancing_loss",
        "analyze_expert_utilization",
        "compute_moe_efficiency",
    ])

# Add utils exports if available
if _UTILS_AVAILABLE:
    __all__.extend([
        "WeightInitializer",
        "init_linear_layer",
        "init_embedding_layer",
        "apply_initialization",
        "create_cache"
    ])


def info():
    """Print information about available models."""
    print("\n🤖 Training Infrastructure - Models")
    print("=" * 50)
    
    print(f"Base models: {'✅ Available' if _BASE_AVAILABLE else '❌ Not available'}")
    print(f"LLaMA models: {'✅ Available' if _LLAMA_AVAILABLE else '❌ Not available'}")
    print(f"MoE models: {'✅ Available' if _MOE_AVAILABLE else '❌ Not available'}")
    print(f"Model utilities: {'✅ Available' if _UTILS_AVAILABLE else '❌ Not available'}")
    
    if _LLAMA_AVAILABLE:
        print("\n📋 Available LLaMA variants:")
        print("  - Tiny LLaMA (150M, 50M)")
        print("  - Standard LLaMA (7B, 13B, 30B, 65B)")
        print("  - LLaMA 2 (7B, 13B, 70B)")
        print("  - LLaMA 3 (8B, 70B, 405B)")
        print("  - Code LLaMA variants")
    
    if _MOE_AVAILABLE:
        print("\n📋 Available MoE variants:")
        print("  - LLaMA-MoE (7B, 13B, 30B, 65B)")
        print("  - Tiny LLaMA-MoE")
        print("  - Code LLaMA-MoE")


def get_available_models():
    """Get list of available model types."""
    models = []
    
    if _LLAMA_AVAILABLE:
        models.append("llama")
    
    if _MOE_AVAILABLE:
        models.append("llama-moe")
    
    return models


def create_model(model_type: str, **kwargs):
    """Create a model instance.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model configuration arguments
    
    Returns:
        Model instance
    """
    if model_type == "llama":
        if not _LLAMA_AVAILABLE:
            raise ImportError("LLaMA models not available. Check dependencies.")
        # Use the existing create functions from llama module
        return create_tiny_llama3_150m(**kwargs)
    
    elif model_type == "llama-moe":
        if not _MOE_AVAILABLE:
            raise ImportError("MoE models not available. Check dependencies.")
        return create_tiny_llama_moe_model(**kwargs)
    
    else:
        available = get_available_models()
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")