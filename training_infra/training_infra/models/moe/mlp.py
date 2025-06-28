# training_infra/models/moe/mlp.py
"""
LLaMA-MoE MLP implementations.
Based on LLaMA MLP format with tensor parallelism support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    # Try relative import first (when imported as module)
    from .config import LlamaMoEConfig
except ImportError:
    # Fallback for direct execution
    try:
        from config import LlamaMoEConfig
    except ImportError:
        print("Warning: Could not import LlamaMoEConfig. Make sure the module is properly installed.")
        raise

# Import tensor parallel components from main LLaMA implementation
try:
    from ..parallelism import TensorParallelLinear
    _TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    _TENSOR_PARALLEL_AVAILABLE = False
    # Fallback to regular Linear
    TensorParallelLinear = nn.Linear


class LlamaMoEMLP(nn.Module):
    """Standard LLaMA MLP with SwiGLU activation (for non-MoE layers)"""
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_config = getattr(config, 'tensor_parallel_config', None)
        
        # Use tensor parallel linear layers for MLP
        if _TENSOR_PARALLEL_AVAILABLE and self.tensor_parallel_config:
            self.gate_proj = TensorParallelLinear(
                self.hidden_size, 
                self.intermediate_size, 
                bias=False,
                config=self.tensor_parallel_config,
                gather_output=False  # Keep sharded for elementwise ops
            )
            self.up_proj = TensorParallelLinear(
                self.hidden_size, 
                self.intermediate_size, 
                bias=False,
                config=self.tensor_parallel_config,
                gather_output=False
            )
            self.down_proj = TensorParallelLinear(
                self.intermediate_size, 
                self.hidden_size, 
                bias=False,
                config=self.tensor_parallel_config,
                scatter_input=True,  # Input is sharded
                gather_output=True   # Gather final output
            )
        else:
            # Fallback to regular linear layers
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = self._get_activation_function(config.hidden_act)

    def _get_activation_function(self, hidden_act):
        if hidden_act == "silu":
            return F.silu
        elif hidden_act == "gelu":
            return F.gelu
        elif hidden_act == "relu":
            return F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x):
        # SwiGLU: gate_proj(x) * silu(up_proj(x))
        # Both projections are sharded, so elementwise ops work correctly
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate = self.act_fn(gate_output) * up_output
        return self.down_proj(intermediate)


class LlamaMoEExpertMLP(nn.Module):
    """Individual expert MLP - identical structure to LlamaMoEMLP"""
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_config = getattr(config, 'tensor_parallel_config', None)
        
        # Same structure as LlamaMoEMLP but individual expert
        if _TENSOR_PARALLEL_AVAILABLE and self.tensor_parallel_config:
            self.gate_proj = TensorParallelLinear(
                self.hidden_size, 
                self.intermediate_size, 
                bias=False,
                config=self.tensor_parallel_config,
                gather_output=False
            )
            self.up_proj = TensorParallelLinear(
                self.hidden_size, 
                self.intermediate_size, 
                bias=False,
                config=self.tensor_parallel_config,
                gather_output=False
            )
            self.down_proj = TensorParallelLinear(
                self.intermediate_size, 
                self.hidden_size, 
                bias=False,
                config=self.tensor_parallel_config,
                scatter_input=True,
                gather_output=True
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            
        self.act_fn = self._get_activation_function(config.hidden_act)
        
        # Expert-specific dropout
        self.dropout = nn.Dropout(config.expert_dropout) if config.expert_dropout > 0 else None

    def _get_activation_function(self, hidden_act):
        if hidden_act == "silu":
            return F.silu
        elif hidden_act == "gelu":
            return F.gelu
        elif hidden_act == "relu":
            return F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x):
        # SwiGLU activation - same as LlamaMoEMLP
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate = self.act_fn(gate_output) * up_output
        
        # Apply expert dropout if specified
        if self.dropout is not None:
            intermediate = self.dropout(intermediate)
        
        return self.down_proj(intermediate)


class LlamaMoEParallelExpertMLP(nn.Module):
    """
    Parallel implementation of multiple experts for efficiency.
    All experts computed in parallel then routed.
    """
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        
        # Parallel expert weights: [num_experts, hidden_size, intermediate_size]
        self.gate_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_size)
        )
        
        self.act_fn = self._get_activation_function(config.hidden_act)
        
        # Initialize weights
        self._init_weights()
        
        # Expert dropout
        self.dropout = nn.Dropout(config.expert_dropout) if config.expert_dropout > 0 else None

    def _get_activation_function(self, hidden_act):
        if hidden_act == "silu":
            return F.silu
        elif hidden_act == "gelu":
            return F.gelu
        elif hidden_act == "relu":
            return F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def _init_weights(self):
        """Initialize expert weights."""
        import math
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all experts in parallel.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Expert outputs [num_experts, batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Reshape input for batch matrix multiply: [batch_size * seq_len, hidden_size]
        x_flat = x.view(-1, hidden_size)
        
        # Parallel computation across all experts
        # gate_outputs: [num_experts, batch_size * seq_len, intermediate_size]
        gate_outputs = torch.bmm(
            x_flat.unsqueeze(0).expand(self.num_experts, -1, -1),
            self.gate_proj
        )
        
        # up_outputs: [num_experts, batch_size * seq_len, intermediate_size]
        up_outputs = torch.bmm(
            x_flat.unsqueeze(0).expand(self.num_experts, -1, -1),
            self.up_proj
        )
        
        # Apply SwiGLU activation
        intermediate = self.act_fn(gate_outputs) * up_outputs
        
        # Apply dropout if specified
        if self.dropout is not None:
            intermediate = self.dropout(intermediate)
        
        # Final projection: [num_experts, batch_size * seq_len, hidden_size]
        expert_outputs = torch.bmm(intermediate, self.down_proj)
        
        # Reshape back: [num_experts, batch_size, seq_len, hidden_size]
        expert_outputs = expert_outputs.view(self.num_experts, batch_size, seq_len, hidden_size)
        
        return expert_outputs


def create_llama_moe_expert_mlp(config: LlamaMoEConfig, expert_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create different types of expert MLPs.
    
    Args:
        config: LLaMA-MoE configuration
        expert_type: Type of expert ("standard", "parallel")
        **kwargs: Additional arguments for specific expert types
        
    Returns:
        Expert MLP module
    """
    if expert_type == "standard":
        return LlamaMoEExpertMLP(config)
    elif expert_type == "parallel":
        return LlamaMoEParallelExpertMLP(config)
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


def test_llama_moe_mlp_modules():
    """Test LLaMA-MoE MLP modules."""
    print("Testing LLaMA-MoE MLP modules...")
    
    try:
        from .config import create_tiny_llama_moe
    except ImportError:
        from config import create_tiny_llama_moe
    
    config = create_tiny_llama_moe()
    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test Standard MLP
    print("\n--- Testing LLaMA-MoE Standard MLP ---")
    standard_mlp = LlamaMoEMLP(config)
    standard_output = standard_mlp(hidden_states)
    assert standard_output.shape == hidden_states.shape
    print(f"✅ Standard MLP output shape: {standard_output.shape}")
    
    standard_params = sum(p.numel() for p in standard_mlp.parameters())
    print(f"✅ Standard MLP parameters: {standard_params:,}")
    
    # Test Expert MLP
    print("\n--- Testing Expert MLP ---")
    expert_mlp = LlamaMoEExpertMLP(config)
    expert_output = expert_mlp(hidden_states)
    assert expert_output.shape == hidden_states.shape
    print(f"✅ Expert MLP output shape: {expert_output.shape}")
    
    expert_params = sum(p.numel() for p in expert_mlp.parameters())
    print(f"✅ Expert MLP parameters: {expert_params:,}")
    
    # Test Parallel Expert MLP
    print("\n--- Testing Parallel Expert MLP ---")
    parallel_mlp = LlamaMoEParallelExpertMLP(config)
    parallel_output = parallel_mlp(hidden_states)
    expected_shape = (config.num_experts, batch_size, seq_len, config.hidden_size)
    assert parallel_output.shape == expected_shape
    print(f"✅ Parallel Expert MLP output shape: {parallel_output.shape}")
    
    parallel_params = sum(p.numel() for p in parallel_mlp.parameters())
    print(f"✅ Parallel Expert MLP parameters: {parallel_params:,}")
    
    # Test factory function
    print("\n--- Testing Factory Function ---")
    for expert_type in ["standard", "parallel"]:
        expert = create_llama_moe_expert_mlp(config, expert_type)
        print(f"✅ Created {expert_type} expert: {type(expert).__name__}")
    
    # Performance comparison
    print("\n--- Performance Comparison ---")
    with torch.no_grad():
        # Standard MLP
        import time
        start = time.time()
        for _ in range(100):
            _ = standard_mlp(hidden_states)
        standard_time = time.time() - start
        
        # Single Expert
        start = time.time()
        for _ in range(100):
            _ = expert_mlp(hidden_states)
        expert_time = time.time() - start
        
        print(f"✅ Standard MLP time (100 runs): {standard_time:.4f}s")
        print(f"✅ Expert MLP time (100 runs): {expert_time:.4f}s")
    
    print("\n✅ All LLaMA-MoE MLP tests passed!")


if __name__ == "__main__":
    test_llama_moe_mlp_modules()