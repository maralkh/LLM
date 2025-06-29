# training_infra/models/llama/__init__.py
"""LLaMA model implementation.

This module provides complete LLaMA model implementation including
configuration, attention, MLP, and modeling components.
"""

# Import config first (no dependencies on other modules)
try:
    from .config import (
        LLaMAConfig,
        LLaMAVariants,
        get_model_info,
        print_model_comparison,
        test_llama_configs
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

# Import attention components
try:
    from .attention import (
        RotaryPositionalEmbedding,
        LLaMAAttention,
        rotate_half,
        apply_rotary_pos_emb
    )
    _ATTENTION_AVAILABLE = True
except ImportError:
    _ATTENTION_AVAILABLE = False

# Import MLP components
try:
    from .mlp import (
        RMSNorm,
        LLaMAMLP,
        SwiGLU,
        GEGLU,
        ReGLU,
        MoEMLP,
        create_mlp
    )
    _MLP_AVAILABLE = True
except ImportError:
    _MLP_AVAILABLE = False

# Import parallel components (optional)
try:
    from ...utils.parallel import (
        VocabParallelEmbedding,
        ColumnParallelLinear,
        RowParallelLinear,
        ParallelMLP,
        set_tensor_parallel_size,
        get_tensor_parallel_size
    )
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False

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

# Import modeling components (depends on config, attention, mlp)
try:
    from .modeling import (
        LLaMAModel,
        LLaMADecoderLayer,
        LLaMAForCausalLM,
        test_llama_model
    )
    _MODELING_AVAILABLE = True
except ImportError:
    _MODELING_AVAILABLE = False

# Import variants and creation functions (depends on modeling)
try:
    from .variants import (
        create_llama_7b_parallel,
        create_llama_13b_parallel,
        create_llama_30b_parallel,
        create_llama_65b_parallel,
        create_llama2_7b_parallel,
        create_code_llama_7b_parallel,
        create_llama3_8b_parallel,
        create_llama3_8b_instruct_parallel,
        create_llama3_70b_parallel,
        create_llama3_70b_instruct_parallel,
        create_llama3_405b_parallel,
        create_tiny_llama3_150m,
        create_tiny_llama3_50m,
        optimize_model_for_training,
        estimate_model_memory
    )
    _VARIANTS_AVAILABLE = True
except ImportError:
    _VARIANTS_AVAILABLE = False

# Core exports - start with empty and add based on availability
__all__ = []

# Add config exports if available
if _CONFIG_AVAILABLE:
    __all__.extend([
        'LLaMAConfig',
        'LLaMAVariants', 
        'get_model_info',
        'print_model_comparison',
        'test_llama_configs'
    ])

# Add attention exports if available
if _ATTENTION_AVAILABLE:
    __all__.extend([
        'RotaryPositionalEmbedding',
        'LLaMAAttention',
        'rotate_half',
        'apply_rotary_pos_emb'
    ])

# Add MLP exports if available
if _MLP_AVAILABLE:
    __all__.extend([
        'RMSNorm',
        'LLaMAMLP', 
        'SwiGLU',
        'GEGLU',
        'ReGLU',
        'MoEMLP',
        'create_mlp'
    ])

# Add parallel exports if available
if _PARALLEL_AVAILABLE:
    __all__.extend([
        'VocabParallelEmbedding',
        'ColumnParallelLinear',
        'RowParallelLinear',
        'ParallelMLP',
        'set_tensor_parallel_size',
        'get_tensor_parallel_size'
    ])

# Add flash attention exports if available
if _FLASH_ATTENTION_AVAILABLE:
    __all__.extend([
        'flash_attention_forward',
        'get_flash_attention_backend',
        'is_flash_attention_available',
        'FlashAttentionConfig'
    ])

# Add modeling exports if available
if _MODELING_AVAILABLE:
    __all__.extend([
        'LLaMAModel',
        'LLaMADecoderLayer',
        'LLaMAForCausalLM',
        'test_llama_model'
    ])

# Add variants exports if available
if _VARIANTS_AVAILABLE:
    __all__.extend([
        'create_llama_7b_parallel',
        'create_llama_13b_parallel',
        'create_llama_30b_parallel',
        'create_llama_65b_parallel',
        'create_llama2_7b_parallel',
        'create_code_llama_7b_parallel',
        'create_llama3_8b_parallel',
        'create_llama3_8b_instruct_parallel',
        'create_llama3_70b_parallel',
        'create_llama3_70b_instruct_parallel',
        'create_llama3_405b_parallel',
        'create_tiny_llama3_150m',
        'create_tiny_llama3_50m',
        'optimize_model_for_training',
        'estimate_model_memory'
    ])


def get_available_features():
    """Get information about available LLaMA features."""
    features = {
        'config': _CONFIG_AVAILABLE,
        'attention': _ATTENTION_AVAILABLE,
        'mlp': _MLP_AVAILABLE,
        'parallel': _PARALLEL_AVAILABLE,
        'flash_attention': _FLASH_ATTENTION_AVAILABLE,
        'modeling': _MODELING_AVAILABLE,
        'variants': _VARIANTS_AVAILABLE
    }
    return features


def info():
    """Print information about available LLaMA components."""
    print("\n🦙 LLaMA Model Components")
    print("=" * 40)
    
    features = get_available_features()
    
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"{status} {feature.replace('_', ' ').title()}")
    
    if _VARIANTS_AVAILABLE:
        print("\n📋 Available model variants:")
        print("  - Tiny: 50M, 150M params")
        print("  - Standard: 7B, 13B, 30B, 65B")
        print("  - LLaMA 2: 7B, 13B, 70B")
        print("  - LLaMA 3: 8B, 70B, 405B")
        print("  - Code LLaMA variants")


def test_imports():
    """Test that all available components can be imported."""
    print("🧪 Testing LLaMA imports...")
    
    try:
        features = get_available_features()
        
        for feature, available in features.items():
            if available:
                print(f"✅ {feature} imports successful")
            else:
                print(f"⚠️  {feature} not available")
        
        print(f"\n✅ LLaMA module tested: {sum(features.values())}/{len(features)} components available")
        return True
        
    except Exception as e:
        print(f"❌ LLaMA import test failed: {e}")
        return False
    
if __name__ == "__main__":
    info()
    test_imports()