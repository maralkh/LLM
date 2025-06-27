"""LLaMA model implementation.

This module provides complete LLaMA model implementation including
configuration, attention, MLP, and modeling components.
"""

# Import config first (no dependencies)
from .config import (
    LLaMAConfig,
    LLaMAVariants,
    get_model_info,
    print_model_comparison,
    test_llama_configs
)

# Import components if dependencies are available
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

# Import modeling components if dependencies are available
try:
    from .modeling import (
        LLaMAModel,
        LLaMADecoderLayer,
        test_llama_model
    )
    _MODELING_AVAILABLE = True
except ImportError:
    _MODELING_AVAILABLE = False

__all__ = [
    # Configuration (always available)
    'LLaMAConfig',
    'LLaMAVariants', 
    'get_model_info',
    'print_model_comparison',
    'test_llama_configs'
]

# Add component exports if available
if _ATTENTION_AVAILABLE:
    __all__.extend([
        'RotaryPositionalEmbedding',
        'LLaMAAttention',
        'rotate_half',
        'apply_rotary_pos_emb'
    ])

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

if _PARALLEL_AVAILABLE:
    __all__.extend([
        'VocabParallelEmbedding',
        'ColumnParallelLinear',
        'RowParallelLinear',
        'ParallelMLP',
        'set_tensor_parallel_size',
        'get_tensor_parallel_size'
    ])

if _FLASH_ATTENTION_AVAILABLE:
    __all__.extend([
        'flash_attention_forward',
        'get_flash_attention_backend',
        'is_flash_attention_available',
        'FlashAttentionConfig'
    ])

if _MODELING_AVAILABLE:
    __all__.extend([
        'LLaMAModel',
        'LLaMADecoderLayer',
        'test_llama_model'
    ])


def get_available_features():
    """Get information about available LLaMA features."""
    features = {
        'config': True,
        'attention': _ATTENTION_AVAILABLE,
        'mlp': _MLP_AVAILABLE,
        'parallel': _PARALLEL_AVAILABLE,
        'flash_attention': _FLASH_ATTENTION_AVAILABLE,
        'modeling': _MODELING_AVAILABLE
    }
    return features


def print_feature_status():
    """Print status of available LLaMA features."""
    features = get_available_features()
    
    print("🔍 LLaMA Feature Status:")
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"   {status} {feature}")
        
    if not all(features.values()):
        print("\n💡 Some features unavailable due to missing dependencies.")
        print("   This is normal during initial setup.")