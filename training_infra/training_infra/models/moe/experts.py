# training_infra/models/moe/experts.py
"""
LLaMA-MoE expert routing and MoE layer implementations.
Handles routing tokens to appropriate experts and load balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    # Try relative import first (when imported as module)
    from .config import LlamaMoEConfig
    from .mlp import LlamaMoEExpertMLP, LlamaMoEParallelExpertMLP, LlamaMoEMLP
except ImportError:
    # Fallback for direct execution
    try:
        from config import LlamaMoEConfig
        from mlp import LlamaMoEExpertMLP, LlamaMoEParallelExpertMLP, LlamaMoEMLP
    except ImportError:
        print("Warning: Could not import MoE components. Make sure the module is properly installed.")
        raise


@dataclass
class LlamaMoEOutput:
    """Output from LLaMA-MoE layer with routing information."""
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    aux_loss: Optional[torch.Tensor] = None
    load_balancing_loss: Optional[torch.Tensor] = None
    routing_weights: Optional[torch.Tensor] = None


class LlamaMoETopKRouter(nn.Module):
    """
    Top-K router for selecting experts.
    Routes each token to top-k experts based on learned routing weights.
    """
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.jitter_noise = config.router_jitter_noise
        
        # Router network: linear layer that maps hidden states to expert weights
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Initialize router weights (smaller std for stability)
        torch.nn.init.normal_(self.router.weight, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using Top-K selection.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            expert_weights: [batch_size, seq_len, num_experts_per_token] - routing weights
            expert_indices: [batch_size, seq_len, num_experts_per_token] - selected expert indices
            router_logits: [batch_size, seq_len, num_experts] - raw router outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing: [batch_size * seq_len, hidden_size]
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Compute router logits: [batch_size * seq_len, num_experts]
        router_logits = self.router(hidden_states_flat)
        
        # Add jitter noise during training for stability
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
        
        # Apply softmax to get routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            routing_weights, 
            self.num_experts_per_token, 
            dim=-1
        )
        
        # Normalize expert weights to sum to 1 (important for stability!)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Reshape back to original batch dimensions
        expert_weights = expert_weights.view(batch_size, seq_len, self.num_experts_per_token)
        expert_indices = expert_indices.view(batch_size, seq_len, self.num_experts_per_token)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)
        
        return expert_weights, expert_indices, router_logits


class LlamaMoESwitchRouter(nn.Module):
    """
    Switch router that routes each token to a single expert.
    Based on Switch Transformer architecture.
    """
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.capacity_factor = config.expert_capacity_factor
        
        # Router network
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        torch.nn.init.normal_(self.router.weight, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route each token to single best expert (Switch style)."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten and route
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        router_logits = self.router(hidden_states_flat)
        
        # Apply softmax and select best expert
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.max(routing_weights, dim=-1)
        
        # Reshape for Switch routing (single expert per token)
        expert_weights = expert_weights.view(batch_size, seq_len, 1)
        expert_indices = expert_indices.view(batch_size, seq_len, 1)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)
        
        return expert_weights, expert_indices, router_logits


def compute_load_balancing_loss(router_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute load balancing loss to encourage equal expert utilization.
    
    This is very important! Without this, all tokens go to one or two experts.
    
    Args:
        router_logits: [batch_size, seq_len, num_experts]
        expert_indices: [batch_size, seq_len, num_experts_per_token]
        
    Returns:
        Load balancing loss scalar
    """
    batch_size, seq_len, num_experts = router_logits.shape
    
    # Calculate percentage of usage for each expert
    routing_weights = F.softmax(router_logits, dim=-1)
    expert_usage = routing_weights.mean(dim=(0, 1))  # [num_experts]
    
    # Define uniform distribution (goal: all experts used equally)
    uniform_distribution = torch.ones_like(expert_usage) / num_experts
    
    # MSE loss to bring closer to uniform distribution
    aux_loss = F.mse_loss(expert_usage, uniform_distribution)
    
    return aux_loss


class LlamaMoELayer(nn.Module):
    """
    Complete LLaMA-MoE layer with routing and expert computation.
    """
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.use_auxiliary_loss = config.use_auxiliary_loss
        self.aux_loss_coef = config.router_aux_loss_coef
        
        # Create router based on type
        if config.router_type == "top_k":
            self.router = LlamaMoETopKRouter(config)
        elif config.router_type == "switch":
            self.router = LlamaMoESwitchRouter(config)
        else:
            raise ValueError(f"Unknown router type: {config.router_type}")
        
        # Create individual experts (each expert is a LlamaMoEExpertMLP)
        self.experts = nn.ModuleList([
            LlamaMoEExpertMLP(config) for _ in range(self.num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> LlamaMoEOutput:
        """
        Forward pass through LLaMA-MoE layer.
        
        Complete process:
        1. Router decides which experts to use
        2. Each expert applies SwiGLU on its assigned tokens  
        3. Results are combined with weights
        4. Load balancing loss is computed
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            LlamaMoEOutput with routed expert outputs and routing information
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Step 1: Route tokens to experts
        expert_weights, expert_indices, router_logits = self.router(hidden_states)
        
        # Step 2: Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Step 3: Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)  # [batch, seq, num_experts_per_token]
            
            if expert_mask.any():
                # Extract tokens for this expert
                # Need to flatten because expert_mask is multi-dimensional
                token_mask = expert_mask.any(dim=-1)  # [batch, seq]
                
                if token_mask.any():
                    expert_tokens = hidden_states[token_mask]  # [num_tokens, hidden_size]
                    
                    if expert_tokens.numel() > 0:
                        # Apply SwiGLU in expert
                        expert_output = self.experts[expert_idx](expert_tokens)
                        
                        # Get weights corresponding to this expert
                        expert_token_weights = expert_weights[expert_mask]  # [num_assignments]
                        
                        # Weight the output
                        weighted_output = expert_output * expert_token_weights.unsqueeze(-1)
                        
                        # Add to final output
                        output[token_mask] += weighted_output
        
        # Step 4: Compute auxiliary loss for load balancing
        aux_loss = None
        if self.use_auxiliary_loss and self.training:
            aux_loss = compute_load_balancing_loss(router_logits, expert_indices)
            aux_loss = aux_loss * self.aux_loss_coef
        
        return LlamaMoEOutput(
            hidden_states=output,
            router_logits=router_logits,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            aux_loss=aux_loss,
            load_balancing_loss=aux_loss,
            routing_weights=F.softmax(router_logits, dim=-1)
        )


class LlamaMoEEfficientLayer(nn.Module):
    """
    More efficient LLaMA-MoE layer using parallel expert computation.
    Faster for large models.
    """
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.use_auxiliary_loss = config.use_auxiliary_loss
        self.aux_loss_coef = config.router_aux_loss_coef
        
        # Router
        if config.router_type == "top_k":
            self.router = LlamaMoETopKRouter(config)
        elif config.router_type == "switch":
            self.router = LlamaMoESwitchRouter(config)
        else:
            raise ValueError(f"Unknown router type: {config.router_type}")
        
        # Parallel experts for efficiency (all experts computed in parallel)
        self.parallel_experts = LlamaMoEParallelExpertMLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> LlamaMoEOutput:
        """Efficient forward pass using parallel expert computation."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens
        expert_weights, expert_indices, router_logits = self.router(hidden_states)
        
        # Compute all expert outputs in parallel
        # expert_outputs: [num_experts, batch_size, seq_len, hidden_size]
        expert_outputs = self.parallel_experts(hidden_states)
        
        # Create routing mask for each expert
        # routing_mask: [num_experts, batch_size, seq_len, num_experts_per_token]
        routing_mask = expert_indices.unsqueeze(0) == torch.arange(
            self.num_experts, device=expert_indices.device
        ).view(-1, 1, 1, 1)
        
        # Apply routing weights
        # expert_weights: [batch_size, seq_len, num_experts_per_token, 1]
        weighted_expert_outputs = expert_outputs.unsqueeze(-2) * expert_weights.unsqueeze(0).unsqueeze(-1)
        
        # Sum over selected experts
        # output: [batch_size, seq_len, hidden_size]
        output = (weighted_expert_outputs * routing_mask.unsqueeze(-1)).sum(dim=(0, 2))
        
        # Auxiliary loss
        aux_loss = None
        if self.use_auxiliary_loss and self.training:
            aux_loss = compute_load_balancing_loss(router_logits, expert_indices)
            aux_loss = aux_loss * self.aux_loss_coef
        
        return LlamaMoEOutput(
            hidden_states=output,
            router_logits=router_logits,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            aux_loss=aux_loss,
            load_balancing_loss=aux_loss,
            routing_weights=F.softmax(router_logits, dim=-1)
        )


def create_llama_moe_layer(config: LlamaMoEConfig, efficient: bool = True) -> nn.Module:
    """
    Factory function to create LLaMA-MoE layers.
    
    Args:
        config: LLaMA-MoE configuration
        efficient: Whether to use efficient parallel implementation
        
    Returns:
        MoE layer module
    """
    if efficient:
        return LlamaMoEEfficientLayer(config)
    else:
        return LlamaMoELayer(config)


def test_llama_moe_experts():
    """Test LLaMA-MoE expert routing and layers."""
    print("Testing LLaMA-MoE Expert Routing...")
    
    try:
        from .config import create_tiny_llama_moe
    except ImportError:
        from config import create_tiny_llama_moe
    
    config = create_tiny_llama_moe()
    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test Top-K Router
    print("\n--- Testing Top-K Router ---")
    topk_router = LlamaMoETopKRouter(config)
    expert_weights, expert_indices, router_logits = topk_router(hidden_states)
    
    print(f"✅ Expert weights shape: {expert_weights.shape}")
    print(f"✅ Expert indices shape: {expert_indices.shape}")
    print(f"✅ Router logits shape: {router_logits.shape}")
    
    # Verify routing constraints
    assert expert_weights.shape == (batch_size, seq_len, config.num_experts_per_token)
    assert expert_indices.shape == (batch_size, seq_len, config.num_experts_per_token)
    assert router_logits.shape == (batch_size, seq_len, config.num_experts)
    
    # Test weight normalization (important!)
    weight_sums = expert_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
    print("✅ Expert weights properly normalized")
    
    # Test Switch Router
    print("\n--- Testing Switch Router ---")
    switch_config = create_tiny_llama_moe()
    switch_config.router_type = "switch"
    switch_config.num_experts_per_token = 1
    
    switch_router = LlamaMoESwitchRouter(switch_config)
    switch_weights, switch_indices, switch_logits = switch_router(hidden_states)
    
    assert switch_weights.shape == (batch_size, seq_len, 1)
    assert switch_indices.shape == (batch_size, seq_len, 1)
    print("✅ Switch router works correctly")
    
    # Test MoE Layer
    print("\n--- Testing LLaMA-MoE Layer ---")
    moe_layer = LlamaMoELayer(config)
    moe_output = moe_layer(hidden_states)
    
    assert moe_output.hidden_states.shape == hidden_states.shape
    assert moe_output.aux_loss is not None
    print(f"✅ MoE output shape: {moe_output.hidden_states.shape}")
    print(f"✅ Auxiliary loss: {moe_output.aux_loss.item():.6f}")
    
    # Test Efficient MoE Layer
    print("\n--- Testing Efficient LLaMA-MoE Layer ---")
    efficient_moe = LlamaMoEEfficientLayer(config)
    efficient_output = efficient_moe(hidden_states)
    
    assert efficient_output.hidden_states.shape == hidden_states.shape
    print(f"✅ Efficient MoE output shape: {efficient_output.hidden_states.shape}")
    
    # Test load balancing
    print("\n--- Testing Load Balancing ---")
    aux_loss = compute_load_balancing_loss(router_logits, expert_indices)
    print(f"✅ Load balancing loss: {aux_loss.item():.6f}")
    
    # Check expert utilization
    routing_weights = F.softmax(router_logits, dim=-1)
    expert_usage = routing_weights.mean(dim=(0, 1))
    print(f"✅ Expert usage distribution: {expert_usage.numpy()}")
    
    # Performance comparison
    print("\n--- Performance Comparison ---")
    with torch.no_grad():
        import time
        
        # Standard MoE
        start = time.time()
        for _ in range(50):
            _ = moe_layer(hidden_states)
        standard_time = time.time() - start
        
        # Efficient MoE
        start = time.time()
        for _ in range(50):
            _ = efficient_moe(hidden_states)
        efficient_time = time.time() - start
        
        print(f"✅ Standard MoE time (50 runs): {standard_time:.4f}s")
        print(f"✅ Efficient MoE time (50 runs): {efficient_time:.4f}s")
        if efficient_time > 0:
            print(f"✅ Speedup: {standard_time/efficient_time:.2f}x")
    
    # Parameter counting
    print("\n--- Parameter Analysis ---")
    moe_params = sum(p.numel() for p in moe_layer.parameters())
    efficient_params = sum(p.numel() for p in efficient_moe.parameters())
    
    print(f"✅ Standard MoE parameters: {moe_params:,}")
    print(f"✅ Efficient MoE parameters: {efficient_params:,}")
    
    print("\n✅ All LLaMA-MoE expert tests passed!")


if __name__ == "__main__":
    test_llama_moe_experts()