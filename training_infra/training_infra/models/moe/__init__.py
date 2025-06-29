# training_infra/models/moe/__init__.py
"""
LLaMA-MoE (Mixture of Experts) models for scalable training.

This module provides a complete LLaMA-MoE implementation with:
- LLaMA-compatible architecture with MoE layers
- Configurable expert routing (Top-K, Switch)
- Load balancing and auxiliary losses
- Tensor parallel support
- Various model sizes from tiny to large
"""

# Version info
__version__ = "0.1.0"

# Try importing configuration first
try:
    from .config import (
        LlamaMoEConfig,
        create_tiny_llama_moe,
        create_llama_moe_7b,
        create_llama_moe_13b,
        create_llama_moe_30b,
        create_llama_moe_65b,
        create_code_llama_moe_7b,
        get_llama_moe_config,
        LLAMA_MOE_CONFIGS,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

# Try importing attention components (same as LLaMA)
try:
    from .attention import (
        RMSNorm,
        RotaryEmbedding,
        LlamaMoEAttention,
        rotate_half,
        apply_rotary_pos_emb,
        repeat_kv,
    )
    _ATTENTION_AVAILABLE = True
except ImportError:
    _ATTENTION_AVAILABLE = False

# Try importing MLP components
try:
    from .mlp import (
        LlamaMoEMLP,
        LlamaMoEExpertMLP,
        LlamaMoEParallelExpertMLP,
        create_llama_moe_expert_mlp,
    )
    _MLP_AVAILABLE = True
except ImportError:
    _MLP_AVAILABLE = False

# Try importing expert routing and MoE layers
try:
    from .experts import (
        LlamaMoEOutput,
        LlamaMoETopKRouter,
        LlamaMoESwitchRouter,
        LlamaMoELayer,
        LlamaMoEEfficientLayer,
        create_llama_moe_layer,
        compute_load_balancing_loss,
    )
    _EXPERTS_AVAILABLE = True
except ImportError:
    _EXPERTS_AVAILABLE = False

# Import flash attention (optional)
try:
    from .flash_attention import (
        flash_attention_forward,
        get_flash_attention_backend,
        is_flash_attention_available,
        FlashAttentionConfig
    )
    _FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    _FLASH_ATTENTION_AVAILABLE = False

# Try importing main models
try:
    from .modeling import (
        LlamaMoEModelOutput,
        LlamaMoECausalLMOutput,
        LlamaMoEDecoderLayer,
        LlamaMoEModel,
        LlamaMoEForCausalLM,
        create_tiny_llama_moe_model,
        create_llama_moe_7b_model,
        create_llama_moe_13b_model,
        create_code_llama_moe_7b_model,
    )
    _MODELING_AVAILABLE = True
except ImportError:
    _MODELING_AVAILABLE = False

# Try importing utilities
try:
    from .utils import (
        init_moe_weights,
        count_moe_parameters,
        analyze_expert_utilization,
        calculate_gini_coefficient,
        visualize_expert_usage,
        get_memory_usage,
        compute_moe_efficiency,
        create_balanced_expert_config,
    )
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False

# Build exports based on availability
__all__ = []

# Add config exports if available
if _CONFIG_AVAILABLE:
    __all__.extend([
        # Configuration
        "LlamaMoEConfig",
        "create_tiny_llama_moe",
        "create_llama_moe_7b",
        "create_llama_moe_13b", 
        "create_llama_moe_30b",
        "create_llama_moe_65b",
        "create_code_llama_moe_7b",
        "get_llama_moe_config",
        "LLAMA_MOE_CONFIGS",
    ])

# Add attention exports if available
if _ATTENTION_AVAILABLE:
    __all__.extend([
        # Attention (same as LLaMA)
        "RMSNorm",
        "RotaryEmbedding",
        "LlamaMoEAttention",
        "rotate_half",
        "apply_rotary_pos_emb",
        "repeat_kv",
    ])

# Add flash attention exports if available
if _FLASH_ATTENTION_AVAILABLE:
    __all__.extend([
        'flash_attention_forward',
        'get_flash_attention_backend',
        'is_flash_attention_available',
        'FlashAttentionConfig'
    ])

# Add MLP exports if available
if _MLP_AVAILABLE:
    __all__.extend([
        # MLP components
        "LlamaMoEMLP",
        "LlamaMoEExpertMLP",
        "LlamaMoEParallelExpertMLP",
        "create_llama_moe_expert_mlp",
    ])

# Add expert exports if available
if _EXPERTS_AVAILABLE:
    __all__.extend([
        # Expert routing and MoE layers
        "LlamaMoEOutput",
        "LlamaMoETopKRouter",
        "LlamaMoESwitchRouter",
        "LlamaMoELayer",
        "LlamaMoEEfficientLayer",
        "create_llama_moe_layer",
        "compute_load_balancing_loss",
    ])

# Add modeling exports if available
if _MODELING_AVAILABLE:
    __all__.extend([
        # Main models
        "LlamaMoEModelOutput",
        "LlamaMoECausalLMOutput",
        "LlamaMoEDecoderLayer",
        "LlamaMoEModel",
        "LlamaMoEForCausalLM",
        "create_tiny_llama_moe_model",
        "create_llama_moe_7b_model",
        "create_llama_moe_13b_model",
        "create_code_llama_moe_7b_model",
    ])

# Add utils exports if available
if _UTILS_AVAILABLE:
    __all__.extend([
        # Utilities
        "init_moe_weights",
        "count_moe_parameters",
        "analyze_expert_utilization",
        "calculate_gini_coefficient",
        "visualize_expert_usage",
        "get_memory_usage",
        "compute_moe_efficiency",
        "create_balanced_expert_config",
    ])


def get_available_features():
    """Get information about available MoE features."""
    features = {
        'config': _CONFIG_AVAILABLE,
        'attention': _ATTENTION_AVAILABLE,
        'flash_attention': _FLASH_ATTENTION_AVAILABLE,
        'mlp': _MLP_AVAILABLE,
        'experts': _EXPERTS_AVAILABLE,
        'modeling': _MODELING_AVAILABLE,
        'utils': _UTILS_AVAILABLE
    }
    return features


def info():
    """Print information about available MoE components."""
    print("\n🎯 LLaMA-MoE Components")
    print("=" * 40)
    
    features = get_available_features()
    
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"{status} {feature.title()}")
    
    if _CONFIG_AVAILABLE:
        print("\n📋 Available MoE variants:")
        print("  - Tiny MoE: For development")
        print("  - Standard MoE: 7B, 13B, 30B, 65B")
        print("  - Code LLaMA MoE variants")
    
    print(f"\n📊 Features available: {sum(features.values())}/{len(features)}")


def test_imports():
    """Test that all available MoE components can be imported."""
    print("🧪 Testing MoE imports...")
    
    try:
        features = get_available_features()
        
        for feature, available in features.items():
            if available:
                print(f"✅ {feature} imports successful")
            else:
                print(f"⚠️  {feature} not available")
        
        total_available = sum(features.values())
        print(f"\n✅ MoE module tested: {total_available}/{len(features)} components available")
        
        return total_available > 0
        
    except Exception as e:
        print(f"❌ MoE import test failed: {e}")
        return False

def info():
    # Check if all imports were successful
    _ALL_IMPORTS_SUCCESSFUL = all([
        _CONFIG_AVAILABLE,
        _ATTENTION_AVAILABLE,
        _FLASH_ATTENTION_AVAILABLE,
        _MLP_AVAILABLE,
        _EXPERTS_AVAILABLE,
        _MODELING_AVAILABLE,
        _UTILS_AVAILABLE
    ])

    if not _ALL_IMPORTS_SUCCESSFUL:
        print(f"Warning: Some LLaMA-MoE components could not be imported.")
        print(f"Available features: {sum(get_available_features().values())}/6")

if __name__ == "__main__":
    info()
    test_imports()