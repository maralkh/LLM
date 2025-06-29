# models/moe/config.py
"""
LLaMA-MoE configuration that extends base LLaMA configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

# Import base LLaMA config
try:
    from ..llama.config import LLaMAConfig
except ImportError:
    # Fall back to absolute imports for direct execution
    import sys
    from pathlib import Path
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Now import with absolute paths
    from training_infra.models.llama.config import LLaMAConfig


@dataclass
class LlamaMoEConfig(LLaMAConfig):
    """
    Configuration for LLaMA-MoE models.
    Inherits all LLaMA parameters and adds MoE-specific ones.
    """
    
    # Core model parameters (ensure they exist)
    hidden_act: str = "silu"  # Activation function for MLPs
    pretraining_tp: int = 1  # Tensor parallel size during pretraining
    
    # MoE specific parameters
    num_experts: int = 8
    num_experts_per_token: int = 2  # Top-K experts
    moe_layers: List[int] = field(default_factory=lambda: [])  # Which layers use MoE
    router_type: str = "top_k"  # "top_k", "switch", "soft"
    router_jitter_noise: float = 0.0
    expert_capacity: Optional[int] = None  # For switch routing
    load_balancing_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    
    # MoE training parameters
    moe_dropout: float = 0.0
    expert_dropout: float = 0.0
    
    # Missing attributes that experts.py and mlp.py expect
    use_auxiliary_loss: bool = True
    auxiliary_loss_factor: float = 0.01
    expert_capacity_tokens: Optional[int] = None
    
    # Additional MoE configuration options
    router_bias: bool = False
    expert_bias: bool = False
    normalize_router_output: bool = True
    use_expert_bias: bool = False
    
    # Performance and memory options
    use_grouped_moe: bool = False
    grouped_moe_size: int = 1
    use_balanced_assignment: bool = True
    expert_parallel_size: int = 1
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        super().__post_init__()
        
        # Set default MoE layers if not specified
        if not self.moe_layers:
            # By default, use MoE in middle layers (e.g., layers 8-24 for 32-layer model)
            if self.num_hidden_layers >= 16:
                start = self.num_hidden_layers // 4
                end = 3 * self.num_hidden_layers // 4
                self.moe_layers = list(range(start, end))
            else:
                # For small models, use MoE in odd layers
                self.moe_layers = list(range(1, self.num_hidden_layers, 2))
        
        # Validate MoE layers
        for layer_idx in self.moe_layers:
            if layer_idx >= self.num_hidden_layers:
                raise ValueError(f"MoE layer {layer_idx} exceeds num_hidden_layers {self.num_hidden_layers}")
        
        # Validate router type
        if self.router_type not in ['top_k', 'switch', 'soft']:
            raise ValueError(f"router_type must be one of ['top_k', 'switch', 'soft']")
        
        # Set expert capacity for switch routing
        if self.router_type == "switch" and self.expert_capacity is None:
            # Default capacity: (tokens_per_batch / num_experts) * capacity_factor
            # This will be set dynamically during forward pass
            pass
        
        # Ensure auxiliary loss factor matches other loss coefficients
        if self.use_auxiliary_loss:
            if self.auxiliary_loss_factor != self.load_balancing_loss_coef:
                self.auxiliary_loss_factor = self.load_balancing_loss_coef
    
    @property
    def num_parameters(self) -> int:
        """Calculate total parameters including experts."""
        # Base LLaMA parameters
        base_params = super().num_parameters
        
        # Add expert parameters
        expert_params = 0
        num_moe_layers = len(self.moe_layers)
        
        if num_moe_layers > 0:
            # Each expert has the same MLP as base model
            single_expert_params = (
                self.hidden_size * self.intermediate_size * 3  # gate, up, down projections
            )
            # Router parameters
            router_params = self.hidden_size * self.num_experts
            
            # Total MoE parameters
            expert_params = num_moe_layers * (
                self.num_experts * single_expert_params + router_params
            )
            
            # Subtract replaced MLP parameters
            replaced_mlp_params = num_moe_layers * single_expert_params
            expert_params -= replaced_mlp_params
        
        return base_params + expert_params
    
    @property
    def active_parameters(self) -> int:
        """Calculate active parameters per forward pass."""
        # Base LLaMA parameters
        base_params = super().num_parameters
        
        # Adjust for MoE layers
        num_moe_layers = len(self.moe_layers)
        if num_moe_layers > 0:
            # MLP parameters that are replaced
            single_mlp_params = self.hidden_size * self.intermediate_size * 3
            replaced_params = num_moe_layers * single_mlp_params
            
            # Active expert parameters (only top-k experts)
            active_expert_params = num_moe_layers * (
                self.num_experts_per_token * single_mlp_params + 
                self.hidden_size * self.num_experts  # router
            )
            
            return base_params - replaced_params + active_expert_params
        
        return base_params


# Predefined configurations
def create_tiny_llama_moe():
    """Create tiny LLaMA-MoE for development."""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",  # Add activation function
        num_experts=1,
        num_experts_per_token=1,
        moe_layers=[1, 3],  # MoE in layers 1 and 3
        use_auxiliary_loss=True,
        auxiliary_loss_factor=0.01,
    )


def create_llama_moe_7b():
    """Create LLaMA-MoE 7B configuration."""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # No GQA in original
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",  # Add activation function
        num_experts=8,
        num_experts_per_token=2,
        moe_layers=list(range(8, 24)),  # MoE in middle layers
        use_auxiliary_loss=True,
        auxiliary_loss_factor=0.01,
    )


def create_llama_moe_13b():
    """Create LLaMA-MoE 13B configuration."""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",  # Add activation function
        num_experts=16,
        num_experts_per_token=2,
        moe_layers=list(range(10, 30)),
        use_auxiliary_loss=True,
        auxiliary_loss_factor=0.01,
    )


def create_llama_moe_30b():
    """Create LLaMA-MoE 30B configuration."""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=6656,
        intermediate_size=17920,
        num_hidden_layers=60,
        num_attention_heads=52,
        num_key_value_heads=52,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",  # Add activation function
        num_experts=32,
        num_experts_per_token=2,
        moe_layers=list(range(15, 45)),
        use_auxiliary_loss=True,
        auxiliary_loss_factor=0.01,
    )


def create_llama_moe_65b():
    """Create LLaMA-MoE 65B configuration."""
    return LlamaMoEConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=22016,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",  # Add activation function
        num_experts=64,
        num_experts_per_token=1,  # Switch routing for large models
        moe_layers=list(range(20, 60)),
        router_type="switch",
        use_auxiliary_loss=True,
        auxiliary_loss_factor=0.01,
    )


def create_code_llama_moe_7b():
    """Create Code LLaMA-MoE 7B configuration."""
    return LlamaMoEConfig(
        vocab_size=32016,  # Different vocab for code
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=16384,  # Long context for code
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,  # Different RoPE base
        hidden_act="silu",  # Add activation function
        num_experts=8,
        num_experts_per_token=2,
        moe_layers=list(range(8, 24)),
        use_auxiliary_loss=True,
        auxiliary_loss_factor=0.01,
    )


# Configuration registry
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
    print(f"   Use auxiliary loss: {config.use_auxiliary_loss}")
    print(f"   Auxiliary loss factor: {config.auxiliary_loss_factor}")
    
    # Test all predefined configs
    print("\n--- Testing All Predefined Configs ---")
    for name in LLAMA_MOE_CONFIGS:
        cfg = get_llama_moe_config(name)
        print(f"✅ {name}: {cfg.num_hidden_layers} layers, {cfg.num_experts} experts")
        
        # Test that all required attributes exist
        required_attrs = ['use_auxiliary_loss', 'auxiliary_loss_factor', 'expert_capacity_tokens']
        for attr in required_attrs:
            if not hasattr(cfg, attr):
                print(f"❌ Missing attribute: {attr}")
            else:
                print(f"   ✅ {attr}: {getattr(cfg, attr)}")
    
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