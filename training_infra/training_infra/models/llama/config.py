"""LLaMA model configuration.

This module defines configuration classes for LLaMA model variants,
supporting both tiny models for development and large models for production.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import math


@dataclass
class LLaMAConfig:
    """Configuration for LLaMA models.
    
    Supports various model sizes from tiny (150M) to large (70B+) parameters.
    Designed for both development and production use cases.
    """
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # For GQA (Grouped Query Attention)
    max_position_embeddings: int = 2048
    
    # Model behavior
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None  # {"type": "linear", "factor": 2.0}
    attention_bias: bool = False
    mlp_bias: bool = False
    
    # Training configuration
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Advanced features
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    
    # Precision and performance
    torch_dtype: str = "float32"  # "float32", "float16", "bfloat16"
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    
    # Model variant
    model_type: str = "llama"
    
    def __post_init__(self):
        """Validate and set derived configurations."""
        # Set num_key_value_heads for GQA if not specified
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
            
        # Validate attention head configuration
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
            
        # Calculate head dimensions
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Validate key-value heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
            
        # Validate torch_dtype
        valid_dtypes = ["float32", "float16", "bfloat16"]
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(
                f"torch_dtype must be one of {valid_dtypes}, got {self.torch_dtype}"
            )
            
        # Set intermediate size if not provided (standard LLaMA ratio)
        if hasattr(self, '_auto_intermediate_size'):
            self.intermediate_size = int(2 * self.hidden_size * 2 / 3)
            # Round to nearest multiple of 256 for efficiency
            self.intermediate_size = ((self.intermediate_size + 255) // 256) * 256
    
    def get_torch_dtype(self):
        """Get the corresponding torch dtype."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16, 
            "bfloat16": torch.bfloat16
        }
        return dtype_map[self.torch_dtype]

    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads
    
    @property
    def num_q_heads(self) -> int:
        """Number of query heads."""
        return self.num_attention_heads
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLaMAConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class LLaMAVariants:
    """Predefined LLaMA model configurations for different use cases."""
    
    @staticmethod
    def tiny_llama(precision: str = "float32") -> LLaMAConfig:
        """Tiny LLaMA for development and testing (~150M parameters).
        
        Perfect for:
        - Fast development and testing
        - CI/CD pipelines
        - Local development on limited hardware
        - Unit testing
        
        Args:
            precision: Model precision ("float32", "float16", "bfloat16")
        """
        return LLaMAConfig(
            vocab_size=32000,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=3,
            max_position_embeddings=2048,
            torch_dtype=precision,
            model_type="tiny_llama"
        )
    
    @staticmethod
    def small_llama(precision: str = "bfloat16") -> LLaMAConfig:
        """Small LLaMA for research and experimentation (~1B parameters).
        
        Perfect for:
        - Research experiments
        - Fine-tuning on single GPU
        - Proof of concept development
        
        Args:
            precision: Model precision ("float32", "float16", "bfloat16")
        """
        return LLaMAConfig(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=4096,
            torch_dtype=precision,
            model_type="small_llama"
        )
    
    @staticmethod
    def llama_7b(precision: str = "bfloat16") -> LLaMAConfig:
        """LLaMA 7B configuration for production use.
        
        Perfect for:
        - Production deployments
        - Fine-tuning with multiple GPUs
        - High-quality inference
        
        Args:
            precision: Model precision ("float32", "float16", "bfloat16")
        """
        return LLaMAConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=4096,
            torch_dtype=precision,
            model_type="llama_7b"
        )
    
    @staticmethod
    def llama_13b(precision: str = "bfloat16") -> LLaMAConfig:
        """LLaMA 13B configuration with GQA.
        
        Perfect for:
        - High-performance applications
        - Advanced fine-tuning
        - Research requiring larger models
        
        Args:
            precision: Model precision ("float32", "float16", "bfloat16")
        """
        return LLaMAConfig(
            vocab_size=32000,
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,
            max_position_embeddings=4096,
            torch_dtype=precision,
            model_type="llama_13b"
        )
    
    @staticmethod
    def llama_70b(precision: str = "bfloat16") -> LLaMAConfig:
        """LLaMA 70B configuration with GQA for maximum performance.
        
        Perfect for:
        - State-of-the-art applications
        - Large-scale research
        - Multi-node distributed training
        
        Args:
            precision: Model precision ("float32", "float16", "bfloat16")
        """
        return LLaMAConfig(
            vocab_size=32000,
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,  # Grouped Query Attention
            max_position_embeddings=4096,
            torch_dtype=precision,
            model_type="llama_70b"
        )
    
    @staticmethod
    def llama3_8b(
        precision: str = "bfloat16",
        max_position_embeddings: int = 8192,
        rope_scaling_type: str = "linear",
        rope_scaling_factor: float = 1.0
    ) -> LLaMAConfig:
        """LLaMA 3 8B with RoPE scaling support.
        
        Args:
            precision: Model precision ("float32", "float16", "bfloat16")
            max_position_embeddings: Maximum sequence length (8192 for LLaMA 3)
            rope_scaling_type: RoPE scaling type ("linear", "dynamic", "yarn", "su")
            rope_scaling_factor: Scaling factor for RoPE
        """
        rope_scaling = None
        if rope_scaling_factor != 1.0:
            rope_scaling = {
                "type": rope_scaling_type,
                "factor": rope_scaling_factor,
                "original_max_position_embeddings": 8192
            }
            
        return LLaMAConfig(
            vocab_size=128256,  # LLaMA 3 vocab size
            hidden_size=4096,
            intermediate_size=14336,  # LLaMA 3 specific
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA in LLaMA 3
            max_position_embeddings=max_position_embeddings,
            rope_theta=500000.0,  # LLaMA 3 specific
            rope_scaling=rope_scaling,
            torch_dtype=precision,
            model_type="llama3_8b"
        )
    
    @staticmethod
    def llama3_70b(
        precision: str = "bfloat16",
        max_position_embeddings: int = 8192,
        rope_scaling_type: str = "linear",
        rope_scaling_factor: float = 1.0
    ) -> LLaMAConfig:
        """LLaMA 3 70B with RoPE scaling support."""
        rope_scaling = None
        if rope_scaling_factor != 1.0:
            rope_scaling = {
                "type": rope_scaling_type,
                "factor": rope_scaling_factor,
                "original_max_position_embeddings": 8192
            }
            
        return LLaMAConfig(
            vocab_size=128256,
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,  # GQA
            max_position_embeddings=max_position_embeddings,
            rope_theta=500000.0,
            rope_scaling=rope_scaling,
            torch_dtype=precision,
            model_type="llama3_70b"
        )
    
    @staticmethod
    def llama4_example(
        precision: str = "bfloat16",
        max_position_embeddings: int = 32768,
        rope_scaling_type: str = "yarn",
        rope_scaling_factor: float = 4.0
    ) -> LLaMAConfig:
        """Example LLaMA 4-style config with advanced RoPE scaling.
        
        This is speculative based on trends, not official LLaMA 4.
        """
        rope_scaling = {
            "type": rope_scaling_type,
            "factor": rope_scaling_factor,
            "original_max_position_embeddings": 8192
        }
        
        return LLaMAConfig(
            vocab_size=128256,
            hidden_size=5120,
            intermediate_size=16384,
            num_hidden_layers=48,
            num_attention_heads=40,
            num_key_value_heads=8,  # More aggressive GQA
            max_position_embeddings=max_position_embeddings,
            rope_theta=1000000.0,  # Higher base for longer contexts
            rope_scaling=rope_scaling,
            torch_dtype=precision,
            use_sliding_window=False,
            model_type="llama4_example"
        )


def get_model_info(config: LLaMAConfig) -> Dict[str, Any]:
    """Calculate model information and statistics.
    
    Args:
        config: LLaMA configuration
        
    Returns:
        Dictionary with model statistics
    """
    # Embedding parameters
    embedding_params = config.vocab_size * config.hidden_size
    
    # Attention parameters per layer
    q_params = config.hidden_size * config.hidden_size
    k_params = config.num_key_value_heads * config.head_dim * config.hidden_size
    v_params = config.num_key_value_heads * config.head_dim * config.hidden_size
    o_params = config.hidden_size * config.hidden_size
    attention_params_per_layer = q_params + k_params + v_params + o_params
    
    # MLP parameters per layer
    gate_params = config.hidden_size * config.intermediate_size
    up_params = config.hidden_size * config.intermediate_size
    down_params = config.intermediate_size * config.hidden_size
    mlp_params_per_layer = gate_params + up_params + down_params
    
    # LayerNorm parameters per layer (2 layer norms per layer)
    layernorm_params_per_layer = 2 * config.hidden_size
    
    # Total parameters per layer
    params_per_layer = (
        attention_params_per_layer + 
        mlp_params_per_layer + 
        layernorm_params_per_layer
    )
    
    # Total transformer parameters
    transformer_params = params_per_layer * config.num_hidden_layers
    
    # Final layer norm
    final_layernorm_params = config.hidden_size
    
    # Output layer (lm_head)
    if config.tie_word_embeddings:
        output_params = 0  # Tied with embeddings
    else:
        output_params = config.vocab_size * config.hidden_size
    
    # Total parameters
    total_params = (
        embedding_params + 
        transformer_params + 
        final_layernorm_params + 
        output_params
    )
    
    # Memory estimation (different precisions)
    memory_fp32_gb = total_params * 4 / (1024**3)  # 4 bytes per parameter
    memory_fp16_gb = total_params * 2 / (1024**3)  # 2 bytes per parameter  
    memory_bf16_gb = total_params * 2 / (1024**3)  # 2 bytes per parameter
    memory_int8_gb = total_params * 1 / (1024**3)  # 1 byte per parameter
    
    return {
        'total_parameters': total_params,
        'embedding_parameters': embedding_params,
        'transformer_parameters': transformer_params,
        'attention_parameters_per_layer': attention_params_per_layer,
        'mlp_parameters_per_layer': mlp_params_per_layer,
        'parameters_per_layer': params_per_layer,
        'memory_fp32_gb': memory_fp32_gb,
        'memory_fp16_gb': memory_fp16_gb,
        'memory_bf16_gb': memory_bf16_gb,
        'memory_int8_gb': memory_int8_gb,
        'dtype': config.torch_dtype,
        'config': config.to_dict()
    }


def print_model_comparison():
    """Print comparison of different LLaMA variants."""
    variants = {
        'Tiny (FP32)': LLaMAVariants.tiny_llama("float32"),
        'Tiny (BF16)': LLaMAVariants.tiny_llama("bfloat16"),
        'Small (BF16)': LLaMAVariants.small_llama("bfloat16"),
        '7B (BF16)': LLaMAVariants.llama_7b("bfloat16"),
        '13B (BF16)': LLaMAVariants.llama_13b("bfloat16"),
        '70B (BF16)': LLaMAVariants.llama_70b("bfloat16"),
        'LLaMA3 8B': LLaMAVariants.llama3_8b("bfloat16"),
        'LLaMA3 70B': LLaMAVariants.llama3_70b("bfloat16")
    }
    
    print("\n🔍 LLaMA Model Variants Comparison")
    print("=" * 100)
    print(f"{'Model':<12} {'Params':<12} {'Layers':<8} {'Hidden':<8} {'Heads':<8} {'KV Heads':<8} {'Memory':<10} {'Context':<8}")
    print("-" * 100)
    
    for name, config in variants.items():
        info = get_model_info(config)
        params = info['total_parameters']
        
        # Choose memory based on dtype
        if config.torch_dtype == "float32":
            memory = info['memory_fp32_gb']
        elif config.torch_dtype == "bfloat16":
            memory = info['memory_bf16_gb']
        else:  # float16
            memory = info['memory_fp16_gb']
        
        if params >= 1e9:
            params_str = f"{params/1e9:.1f}B"
        elif params >= 1e6:
            params_str = f"{params/1e6:.0f}M"
        else:
            params_str = f"{params/1e3:.0f}K"
        
        context_str = f"{config.max_position_embeddings//1000}k" if config.max_position_embeddings >= 1000 else str(config.max_position_embeddings)
            
        print(f"{name:<12} {params_str:<12} {config.num_hidden_layers:<8} "
              f"{config.hidden_size:<8} {config.num_attention_heads:<8} {config.num_key_value_heads:<8} "
              f"{memory:.1f} GB{'':<3} {context_str:<8}")
    
    print("\n💡 Precision Recommendations:")
    print("  - FP32: Maximum precision, debugging")
    print("  - BF16: Best balance for most use cases") 
    print("  - FP16: Memory efficient, watch for overflow")
    print("  - INT8: Maximum efficiency, quantized inference")
    
    print("\n🚀 Use Case Recommendations:")
    print("  - Tiny: Development, testing, CI/CD")
    print("  - Small: Research, single-GPU fine-tuning")
    print("  - 7B: Production, multi-GPU training")
    print("  - 13B: High-performance applications")
    print("  - 70B: State-of-the-art, distributed training")
    print("  - LLaMA3: Latest with improved performance")


# Testing function
def test_llama_configs():
    """Test LLaMA configuration creation and validation."""
    print("🧪 Testing LLaMA Configurations...")
    
    # Test predefined variants
    variants = [
        ('Tiny', LLaMAVariants.tiny_llama()),
        ('Small', LLaMAVariants.small_llama()),
        ('7B', LLaMAVariants.llama_7b()),
        ('LLaMA3 8B', LLaMAVariants.llama3_8b()),
        ('Custom', LLaMAVariants.custom_llama(1024, 16, 16, use_gqa=True))
    ]
    
    for name, config in variants:
        info = get_model_info(config)
        params = info['total_parameters']
        
        print(f"   {name}: {params:,} parameters, {config.max_position_embeddings} context")
        
        # Validate configuration
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0
        
        # Test dtype conversion
        torch_dtype = config.get_torch_dtype()
        assert torch_dtype is not None
        
    # Test RoPE scaling
    print("   Testing RoPE scaling...")
    config_scaled = LLaMAVariants.llama3_8b(
        max_position_embeddings=32768,
        rope_scaling_type="dynamic",
        rope_scaling_factor=4.0
    )
    assert config_scaled.rope_scaling is not None
    assert config_scaled.rope_scaling["factor"] == 4.0
    
    # Test serialization
    config = LLaMAVariants.tiny_llama()
    config_dict = config.to_dict()
    restored_config = LLaMAConfig.from_dict(config_dict)
    assert config.hidden_size == restored_config.hidden_size
    
    print("✅ LLaMA configuration tests passed!")


if __name__ == "__main__":
    test_llama_configs()
    print_model_comparison()