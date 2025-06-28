# training_infra/models/moe/config.py
"""
LLaMA-MoE model configuration classes.
Based on LLaMA format with Mixture of Experts extensions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LlamaMoEConfig:
    """
    Configuration for LLaMA-MoE models.
    Extends LLaMA architecture with Mixture of Experts layers.
    """
    
    # Base LLaMA parameters (keeping exact same names)
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # For GQA
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    
    # MoE specific parameters
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    moe_layers: Optional[List[int]] = None  # Which layers use MoE
    router_aux_loss_coef: float = 0.02
    router_type: str = "top_k"  # "top_k", "switch", "soft"
    router_jitter_noise: float = 0.01
    use_auxiliary_loss: bool = True
    expert_dropout: float = 0.1
    
    def __post_init__(self):
        # Set GQA heads if not specified
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
            
        # Set MoE layers if not specified (use all layers)
        if self.moe_layers is None:
            self.moe_layers = list(range(self.num_hidden_layers))
            
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Base LLaMA validation
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot be "
                f"larger than num_attention_heads ({self.num_attention_heads})"
            )
        
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        
        # MoE validation
        if self.num_experts < 2:
            raise ValueError(f"num_experts ({self.num_experts}) must be at least 2")
        
        if self.num_experts_per_token < 1 or self.num_experts_per_token > self.num_experts:
            raise ValueError(
                f"num_experts_per_token ({self.num_experts_per_token}) must be "
                f"between 1 and num_experts ({self.num_experts})"
            )
        
        if any(layer >= self.num_hidden_layers for layer in self.moe_layers):
            raise ValueError(
                f"MoE layers {self.moe_layers} contain indices >= num_hidden_layers "
                f"({self.num_hidden_layers})"
            )
        
        if self.router_type not in ["top_k", "switch", "soft"]:
            raise ValueError(f"router_type must be one of ['top_k', 'switch', 'soft']")


# Predefined configurations following LLaMA naming pattern
def create_llama_moe_7b():
    """Create LLaMA-MoE 7B configuration"""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        num_experts=8,
        num_experts_per_token=2,
        moe_layers=list(range(8, 24)),  # MoE in middle layers
    )


def create_llama_moe_13b():
    """Create LLaMA-MoE 13B configuration"""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        num_experts=16,
        num_experts_per_token=2,
        moe_layers=list(range(10, 30)),  # MoE in middle layers
    )


def create_llama_moe_30b():
    """Create LLaMA-MoE 30B configuration"""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=6656,
        intermediate_size=17920,
        num_hidden_layers=60,
        num_attention_heads=52,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        num_experts=32,
        num_experts_per_token=2,
        moe_layers=list(range(15, 45)),  # MoE in middle layers
    )


def create_llama_moe_65b():
    """Create LLaMA-MoE 65B configuration"""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=22016,
        num_hidden_layers=80,
        num_attention_heads=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        num_experts=64,
        num_experts_per_token=1,  # Switch routing for large models
        moe_layers=list(range(20, 60)),  # MoE in middle layers
        router_type="switch",
    )


def create_tiny_llama_moe():
    """Create tiny LLaMA-MoE for development"""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        num_experts=4,
        num_experts_per_token=2,
        moe_layers=[1, 3],  # MoE in layers 1 and 3
    )


def create_code_llama_moe_7b():
    """Create Code LLaMA-MoE 7B configuration"""
    return LlamaMoEConfig(
        vocab_size=32016,  # Different vocab for code
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=16384,  # Long context for code
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,  # Different RoPE base
        num_experts=8,
        num_experts_per_token=2,
        moe_layers=list(range(8, 24)),
    )


# Configuration registry following LLaMA pattern
LLAMA_MOE_CONFIGS = {
    "tiny_llama_moe": create_tiny_llama_moe,
    "llama_moe_7b": create_llama_moe_7b,
    "llama_moe_13b": create_llama_moe_13b,
    "llama_moe_30b": create_llama_moe_30b,
    "llama_moe_65b": create_llama_moe_65b,
    "code_llama_moe_7b": create_code_llama_moe_7b,
}


def get_llama_moe_config(config_name: str) -> LlamaMoEConfig:
    """Get a predefined LLaMA-MoE configuration by name."""
    if config_name not in LLAMA_MOE_CONFIGS:
        available = ", ".join(LLAMA_MOE_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return LLAMA_MOE_CONFIGS[config_name]()


def test_llama_moe_config():
    """Test LLaMA-MoE configuration functionality."""
    print("Testing LLaMA-MoE Configuration...")
    
    # Test basic configuration
    config = create_tiny_llama_moe()
    print(f"✅ Tiny LLaMA-MoE Config created")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Experts: {config.num_experts}")
    print(f"   MoE layers: {config.moe_layers}")
    
    # Test all predefined configs
    print("\n--- Testing All Predefined Configs ---")
    for name in LLAMA_MOE_CONFIGS:
        cfg = get_llama_moe_config(name)
        print(f"✅ {name}: {cfg.num_hidden_layers} layers, {cfg.num_experts} experts")
    
    # Test validation
    print("\n--- Testing Validation ---")
    try:
        # This should fail
        bad_config = LlamaMoEConfig(
            hidden_size=512,
            num_attention_heads=7,  # 512 not divisible by 7
        )
        print("❌ Validation should have failed")
    except ValueError as e:
        print(f"✅ Validation correctly caught error: {str(e)[:50]}...")
    
    print("\n✅ All LLaMA-MoE config tests passed!")


if __name__ == "__main__":
    test_llama_moe_config()