# training_infra/models/__init__.py
"""
Model architectures and implementations for LLaMA training.

This module provides LLaMA model implementations with support for distributed
training, including all variants from original LLaMA to LLaMA 3 and Tiny models.
"""

from .llama import (
    LlamaConfig,
    RMSNorm,
    RotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    
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

# Core exports - always available
__all__ = [
    # Configuration
    "LlamaConfig",
    
    # Model components
    "RMSNorm",
    "RotaryEmbedding", 
    "LlamaAttention",
    "LlamaMLP",
    "LlamaDecoderLayer",
    "LlamaModel",
    "LlamaForCausalLM",
    
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
]

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


def info():
    """Print information about available models."""
    print("🦙 Training Infrastructure - Models")
    print()
    print("Available LLaMA models:")
    print("  ✅ LLaMA 7B/13B/30B/65B")
    print("  ✅ LLaMA 2 7B (extended context)")
    print("  ✅ Code LLaMA 7B (16k context)")
    print("  ✅ LLaMA 3 8B/70B/405B")
    print("  ✅ Tiny LLaMA 3 50M/150M (development)")
    
    if _MOE_AVAILABLE:
        print()
        print("Available LLaMA-MoE models:")
        print("  ✅ Tiny LLaMA-MoE (4 experts)")
        print("  ✅ LLaMA-MoE 7B (8 experts)")
        print("  ✅ LLaMA-MoE 13B (16 experts)")
        print("  ✅ LLaMA-MoE 30B (32 experts)")
        print("  ✅ LLaMA-MoE 65B (64 experts)")
        print("  ✅ Code LLaMA-MoE 7B (8 experts)")
    else:
        print()
        print("LLaMA-MoE models: ❌ Not available (import error)")


def list_models():
    """List all available model creation functions."""
    models = [
        "create_llama_7b_parallel",
        "create_llama_13b_parallel", 
        "create_llama_30b_parallel",
        "create_llama_65b_parallel",
        "create_llama2_7b_parallel",
        "create_code_llama_7b_parallel",
        "create_llama3_8b_parallel",
        "create_llama3_8b_instruct_parallel",
        "create_llama3_70b_parallel",
        "create_llama3_70b_instruct_parallel",
        "create_llama3_405b_parallel",
        "create_tiny_llama3_150m",
        "create_tiny_llama3_50m",
    ]
    
    if _MOE_AVAILABLE:
        models.extend([
            "create_tiny_llama_moe_model",
            "create_llama_moe_7b_model",
            "create_llama_moe_13b_model",
            "create_code_llama_moe_7b_model",
        ])
    
    return models


def quick_model_create(model_name: str, **kwargs):
    """Quick model creation with reasonable defaults."""
    model_creators = {
        "tiny_llama3_150m": create_tiny_llama3_150m,
        "tiny_llama3_50m": create_tiny_llama3_50m,
        "llama_7b": create_llama_7b_parallel,
        "llama_13b": create_llama_13b_parallel,
        "llama2_7b": create_llama2_7b_parallel,
        "code_llama_7b": create_code_llama_7b_parallel,
        "llama3_8b": create_llama3_8b_parallel,
        "llama3_70b": create_llama3_70b_parallel,
    }
    
    if _MOE_AVAILABLE:
        model_creators.update({
            "tiny_llama_moe": create_tiny_llama_moe_model,
            "llama_moe_7b": create_llama_moe_7b_model,
            "llama_moe_13b": create_llama_moe_13b_model,
            "code_llama_moe_7b": create_code_llama_moe_7b_model,
        })
    
    if model_name not in model_creators:
        available = ", ".join(model_creators.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    return model_creators[model_name](**kwargs)


# Model presets for common configurations
MODEL_PRESETS = {
    "development": {
        "model": "tiny_llama3_150m",
        "description": "Tiny model for development and testing",
        "config": {}
    },
    "small_production": {
        "model": "llama3_8b",
        "description": "Small production model",
        "config": {"tensor_parallel_size": 1}
    },
    "large_production": {
        "model": "llama3_70b", 
        "description": "Large production model",
        "config": {"tensor_parallel_size": 4}
    },
}

if _MOE_AVAILABLE:
    MODEL_PRESETS.update({
        "development_moe": {
            "model": "tiny_llama_moe",
            "description": "Tiny MoE model for development",
            "config": {}
        },
        "production_moe": {
            "model": "llama_moe_7b",
            "description": "Production MoE model",
            "config": {"tensor_parallel_size": 2}
        },
    })


def create_model_from_preset(preset_name: str, overrides: dict = None) -> any:
    """Create a model from a preset configuration."""
    overrides = overrides or {}
    
    if preset_name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    preset = MODEL_PRESETS[preset_name]
    config = preset["config"].copy()
    config.update(overrides)
    
    return quick_model_create(preset["model"], **config)


def list_model_presets():
    """List available model presets"""
    print("📋 Model Presets:")
    print("=" * 40)
    
    for name, preset in MODEL_PRESETS.items():
        print(f"  • {name}: {preset['description']}")
        print(f"    Model: {preset['model']}")


# Add preset functions to exports
__all__.extend([
    "info",
    "list_models",
    "quick_model_create",
    "create_model_from_preset",
    "list_model_presets",
    "MODEL_PRESETS"
])


if __name__ == "__main__":
    info()
    print()
    list_model_presets()