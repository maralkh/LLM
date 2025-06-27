"""LLaMA MLP components.

This module contains MLP and activation function implementations
for LLaMA models.
"""


"""LLaMA MLP components.

This module contains MLP and activation function implementations
for LLaMA models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle imports for different execution contexts
try:
    from .config import LLaMAConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from training_infra.models.llama.config import LLaMAConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More stable and efficient than LayerNorm for large models.
    Used in LLaMA instead of standard LayerNorm.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype)


class LLaMAMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation."""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Import parallel layers if available
        try:
            from .parallel import ColumnParallelLinear, RowParallelLinear
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size, self.intermediate_size, 
                bias=config.mlp_bias, gather_output=False
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size, self.intermediate_size, 
                bias=config.mlp_bias, gather_output=False
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size, self.hidden_size, 
                bias=config.mlp_bias, input_is_parallel=True
            )
        except ImportError:
            # Fallback to standard linear layers
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
        # Activation function
        self.act_fn = F.silu  # SiLU (Swish) activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(gate) * up
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


class SwiGLU(nn.Module):
    """SwiGLU activation function.
    
    Combines Swish (SiLU) activation with GLU (Gated Linear Unit).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)


class GEGLU(nn.Module):
    """GEGLU activation function.
    
    Combines GELU activation with GLU (Gated Linear Unit).
    Alternative to SwiGLU used in some transformer variants.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.gate_proj(x)) * self.up_proj(x)


class ReGLU(nn.Module):
    """ReGLU activation function.
    
    Combines ReLU activation with GLU (Gated Linear Unit).
    Simpler alternative to SwiGLU and GEGLU.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gate_proj(x)) * self.up_proj(x)


class MoEMLP(nn.Module):
    """Mixture of Experts MLP.
    
    Routes inputs to different expert MLPs based on a gating mechanism.
    Useful for scaling model capacity without proportional compute increase.
    """
    
    def __init__(
        self, 
        config: LLaMAConfig, 
        num_experts: int = 8,
        top_k: int = 2,
        expert_capacity_factor: float = 1.0
    ):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        
        # Gating network
        self.gate = nn.Linear(config.hidden_size, num_experts, bias=False)
        
        # Expert MLPs
        self.experts = nn.ModuleList([
            LLaMAMLP(config) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Compute gating scores
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)
        
        # Route to experts
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i:i+1]
            
            for expert_id in range(self.num_experts):
                expert_mask = (expert_indices == expert_id)
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[expert_mask] += expert_probs[expert_mask] * expert_output
        
        return output.view(batch_size, seq_len, hidden_size)


class GLU(nn.Module):
    """Generic Gated Linear Unit.
    
    Supports different activation functions in the gating mechanism.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        activation: str = "silu",
        bias: bool = False
    ):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        
        # Activation function
        if activation == "silu" or activation == "swish":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.gate_proj(x)) * self.up_proj(x)


def create_mlp(
    config: LLaMAConfig, 
    mlp_type: str = "standard",
    **kwargs
) -> nn.Module:
    """Factory function to create different types of MLPs.
    
    Args:
        config: LLaMA configuration
        mlp_type: Type of MLP ("standard", "moe", "swiglu", "geglu", "reglu")
        **kwargs: Additional arguments for specific MLP types
        
    Returns:
        MLP module
    """
    if mlp_type == "standard":
        return LLaMAMLP(config)
    elif mlp_type == "moe":
        return MoEMLP(config, **kwargs)
    elif mlp_type == "swiglu":
        return SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
    elif mlp_type == "geglu":
        return GEGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
    elif mlp_type == "reglu":
        return ReGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")


# Testing function
def test_mlp_components():
    """Test MLP components."""
    print("🧪 Testing MLP Components...")
    
    # Test RMSNorm
    rms_norm = RMSNorm(256)
    x = torch.randn(4, 16, 256)
    normed = rms_norm(x)
    assert normed.shape == x.shape
    print("   ✅ RMSNorm working")
    
    # Test LLaMAMLP
    from .config import LLaMAVariants
    config = LLaMAVariants.tiny_llama()
    mlp = LLaMAMLP(config)
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output = mlp(hidden_states)
    assert output.shape == hidden_states.shape
    print("   ✅ LLaMAMLP working")
    
    # Test SwiGLU
    swiglu = SwiGLU(config.hidden_size, config.intermediate_size)
    glu_output = swiglu(hidden_states)
    assert glu_output.shape == (batch_size, seq_len, config.intermediate_size)
    print("   ✅ SwiGLU working")
    
    # Test GEGLU
    geglu = GEGLU(config.hidden_size, config.intermediate_size)
    geglu_output = geglu(hidden_states)
    assert geglu_output.shape == (batch_size, seq_len, config.intermediate_size)
    print("   ✅ GEGLU working")
    
    # Test MLP factory
    standard_mlp = create_mlp(config, "standard")
    factory_output = standard_mlp(hidden_states)
    assert factory_output.shape == hidden_states.shape
    print("   ✅ MLP factory working")
    
    # Test MoE MLP
    moe_mlp = create_mlp(config, "moe", num_experts=4, top_k=2)
    moe_output = moe_mlp(hidden_states)
    assert moe_output.shape == hidden_states.shape
    print("   ✅ MoE MLP working")
    
    print("✅ MLP components tests passed!")


if __name__ == "__main__":
    test_mlp_components()