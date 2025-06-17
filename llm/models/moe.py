# training_infra/models/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

@dataclass 
class MoEConfig:
    """Configuration for Mixture of Experts"""
    hidden_size: int = 4096
    num_experts: int = 8
    num_experts_per_tok: int = 2  # Top-k routing
    intermediate_size: int = 11008
    hidden_act: str = "silu"
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_bias: bool = False
    use_aux_loss: bool = True
    expert_capacity_factor: float = 1.0
    expert_dropout: float = 0.0
    router_jitter_noise: float = 0.0
    normalize_router_prob_before_dropping: bool = False

class TopKRouter(nn.Module):
    """Top-K Router for Mixture of Experts"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_jitter_noise = config.router_jitter_noise
        self.normalize_router_prob_before_dropping = config.normalize_router_prob_before_dropping
        
        self.classifier = nn.Linear(config.hidden_size, config.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute router probabilities"""
        # hidden_states: [batch_size, sequence_length, hidden_dim]
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        
        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(hidden_states) * self.jitter_noise
            hidden_states = hidden_states + noise
        
        # Compute router logits
        router_logits = self.classifier(hidden_states)
        router_probabilities = F.softmax(router_logits, dim=-1)
        
        return router_logits, router_probabilities

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the router
        
        Returns:
            routing_weights: [batch_size * seq_len, num_experts_per_tok]
            selected_experts: [batch_size * seq_len, num_experts_per_tok] 
            router_logits: [batch_size * seq_len, num_experts]
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Compute router probabilities
        router_logits, router_probs = self._compute_router_probabilities(hidden_states)
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize routing weights
        if self.normalize_router_prob_before_dropping:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        else:
            routing_weights = F.softmax(
                torch.gather(router_logits, -1, selected_experts), dim=-1
            )
        
        return routing_weights, selected_experts, router_logits

class MoELayer(nn.Module):
    """Mixture of Experts Layer"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Router
        self.router = TopKRouter(config)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])
        
        self.expert_dropout = nn.Dropout(config.expert_dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_losses: Dictionary of auxiliary losses
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Flatten for easier processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Route tokens to experts
        routing_weights, selected_experts, router_logits = self.router(hidden_states)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get tokens and weights for this expert
                expert_tokens = hidden_states_flat[expert_mask]
                
                # Get routing weights for this expert
                expert_weights = routing_weights[expert_mask]
                expert_positions = (selected_experts[expert_mask] == expert_idx).nonzero(as_tuple=True)[1]
                expert_weights = expert_weights[range(len(expert_weights)), expert_positions]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                expert_output = self.expert_dropout(expert_output)
                
                # Weight the output
                expert_output = expert_output * expert_weights.unsqueeze(-1)
                
                # Add to final output
                final_hidden_states[expert_mask] += expert_output
        
        # Reshape back
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        
        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(router_logits, routing_weights, selected_experts)
        
        return final_hidden_states, aux_losses

    def _compute_aux_losses(self, router_logits: torch.Tensor, routing_weights: torch.Tensor, 
                           selected_experts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training stability"""
        aux_losses = {}
        
        if self.config.use_aux_loss:
            # Load balancing loss (encourages equal expert usage)
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Expert usage frequency
            expert_usage = torch.zeros(self.num_experts, device=router_logits.device)
            for expert_idx in range(self.num_experts):
                expert_mask = (selected_experts == expert_idx).any(dim=-1)
                expert_usage[expert_idx] = expert_mask.float().mean()
            
            # Load balancing loss
            load_balancing_loss = self.num_experts * torch.var(expert_usage)
            aux_losses['load_balancing_loss'] = self.config.router_aux_loss_coef * load_balancing_loss
            
            # Z-loss (encourages sparsity in router outputs)
            if self.config.router_z_loss_coef > 0:
                z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
                aux_losses['z_loss'] = self.config.router_z_loss_coef * z_loss
        
        return aux_losses

class Expert(nn.Module):
    """Individual Expert (Feed-Forward Network)"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Standard FFN with gating (like LLaMA)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert forward pass (SwiGLU)"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class SparseMoEBlock(nn.Module):
    """Sparse Mixture of Experts Block (replacement for standard MLP)"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.moe = MoELayer(config)
        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through sparse MoE block"""
        output, aux_losses = self.moe(hidden_states)
        return output, aux_losses

# Advanced MoE variants
class SwitchTransformerMoE(MoELayer):
    """Switch Transformer style MoE (top-1 routing)"""
    
    def __init__(self, config: MoEConfig):
        # Force top-1 routing for Switch Transformer
        config.num_experts_per_tok = 1
        super().__init__(config)
        
        # Switch transformer uses different capacity and load balancing
        self.capacity_factor = config.expert_capacity_factor
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Switch Transformer forward with capacity-based routing"""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        
        # Compute capacity per expert
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        
        # Route tokens
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        routing_weights, selected_experts, router_logits = self.router(hidden_states)
        
        # Track expert assignments and capacity
        expert_counts = torch.zeros(self.num_experts, device=hidden_states.device)
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # Process tokens with capacity constraints
        for token_idx in range(num_tokens):
            expert_idx = selected_experts[token_idx, 0].item()
            
            # Check if expert has capacity
            if expert_counts[expert_idx] < capacity:
                expert_weight = routing_weights[token_idx, 0]
                token_input = hidden_states_flat[token_idx:token_idx+1]
                
                expert_output = self.experts[expert_idx](token_input)
                final_hidden_states[token_idx] = expert_output.squeeze(0) * expert_weight
                
                expert_counts[expert_idx] += 1
            # else: token is dropped (capacity exceeded)
        
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        aux_losses = self._compute_aux_losses(router_logits, routing_weights, selected_experts)
        
        return final_hidden_states, aux_losses

class GLaMRouter(nn.Module):
    """GLaM-style router with learned gating"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        
        # Learnable gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.num_experts)
        )
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GLaM router forward pass"""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Compute gating scores
        gate_scores = self.gate(hidden_states_flat)
        gate_scores = gate_scores / self.temperature
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            F.softmax(gate_scores, dim=-1), self.num_experts_per_tok, dim=-1
        )
        
        return routing_weights, selected_experts, gate_scores

class MoETransformerLayer(nn.Module):
    """Transformer layer with MoE instead of standard MLP"""
    
    def __init__(self, attention_module, config: MoEConfig):
        super().__init__()
        self.attention = attention_module
        self.moe_block = SparseMoEBlock(config)
        
        # Layer norms
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with attention + MoE"""
        
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_output = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
            **kwargs
        )
        
        if isinstance(attention_output, tuple):
            attention_output = attention_output[0]
        
        hidden_states = residual + attention_output
        
        # MoE block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        moe_output, aux_losses = self.moe_block(hidden_states)
        hidden_states = residual + moe_output
        
        return hidden_states, aux_losses

# Utility functions for MoE models
def create_moe_config(
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    router_aux_loss_coef: float = 0.01
) -> MoEConfig:
    """Create MoE configuration"""
    return MoEConfig(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        router_aux_loss_coef=router_aux_loss_coef
    )

def compute_moe_loss(model_outputs: torch.Tensor, aux_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute total loss including MoE auxiliary losses"""
    total_aux_loss = torch.tensor(0.0, device=model_outputs.device)
    
    for loss_name, loss_value in aux_losses.items():
        total_aux_loss += loss_value
    
    return total_aux_loss

def get_expert_utilization(aux_losses: Dict[str, torch.Tensor], num_experts: int) -> Dict[str, float]:
    """Get expert utilization statistics"""
    stats = {}
    
    if 'load_balancing_loss' in aux_losses:
        # Higher load balancing loss indicates more imbalanced expert usage
        stats['load_balance_score'] = aux_losses['load_balancing_loss'].item()
        stats['expert_balance'] = 1.0 / (1.0 + stats['load_balance_score'])
    
    return stats

# Example usage and integration
class MoELlamaLayer(nn.Module):
    """LLaMA layer with MoE instead of standard MLP"""
    
    def __init__(self, llama_attention, moe_config: MoEConfig):
        super().__init__()
        self.self_attn = llama_attention
        self.moe_block = SparseMoEBlock(moe_config)
        
        # Use RMSNorm like LLaMA
        from .llama import RMSNorm
        self.input_layernorm = RMSNorm(moe_config.hidden_size)
        self.post_attention_layernorm = RMSNorm(moe_config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass combining LLaMA attention with MoE"""
        
        # Self attention (same as LLaMA)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_outputs = self.self_attn(hidden_states, **kwargs)
        hidden_states = attention_outputs[0]
        hidden_states = residual + hidden_states
        
        # MoE block (replaces standard MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        moe_output, aux_losses = self.moe_block(hidden_states)
        hidden_states = residual + moe_output
        
        # Return format compatible with LLaMA
        outputs = (hidden_states,)
        if len(attention_outputs) > 1:
            outputs += attention_outputs[1:]
        
        return outputs, aux_losses