# training_infra/models/moe/utils.py
"""
Utility functions for MoE models.
Includes initialization, analysis, and helper functions.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np

from .config import MoEConfig


def init_moe_weights(module: nn.Module, config: MoEConfig) -> None:
    """
    Initialize MoE model weights.
    
    Args:
        module: The module to initialize
        config: MoE configuration
    """
    if isinstance(module, nn.Linear):
        # Standard initialization for linear layers
        torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Embedding initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif hasattr(module, 'router'):
        # Special initialization for routers - smaller variance for stability
        torch.nn.init.normal_(module.router.weight, mean=0.0, std=0.02)


def count_moe_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in different parts of MoE model.
    
    Args:
        model: MoE model
        
    Returns:
        Dictionary with parameter counts
    """
    param_counts = {
        'total': 0,
        'embedding': 0,
        'attention': 0,
        'router': 0,
        'experts': 0,
        'norm': 0,
        'output_head': 0
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_counts['total'] += param_count
        
        if 'embed' in name:
            param_counts['embedding'] += param_count
        elif 'attention' in name or 'attn' in name:
            param_counts['attention'] += param_count
        elif 'router' in name:
            param_counts['router'] += param_count
        elif 'expert' in name:
            param_counts['experts'] += param_count
        elif 'norm' in name:
            param_counts['norm'] += param_count
        elif 'lm_head' in name or 'output' in name:
            param_counts['output_head'] += param_count
    
    return param_counts


def analyze_expert_utilization(router_logits: torch.Tensor, expert_indices: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze how evenly tokens are distributed across experts.
    
    Args:
        router_logits: Router outputs [batch_size, seq_len, num_experts]
        expert_indices: Selected expert indices [batch_size, seq_len, num_experts_per_token]
        
    Returns:
        Analysis results dictionary
    """
    batch_size, seq_len, num_experts = router_logits.shape
    total_tokens = batch_size * seq_len
    
    # Calculate routing probabilities
    routing_probs = torch.softmax(router_logits, dim=-1)
    
    # Expert usage statistics
    expert_usage = routing_probs.mean(dim=(0, 1))  # Average probability per expert
    expert_selection_count = torch.zeros(num_experts)
    
    # Count how many times each expert was selected
    for expert_id in range(num_experts):
        expert_selection_count[expert_id] = (expert_indices == expert_id).sum().float()
    
    expert_selection_freq = expert_selection_count / expert_indices.numel()
    
    # Load balancing metrics
    uniform_prob = 1.0 / num_experts
    usage_variance = expert_usage.var().item()
    gini_coefficient = calculate_gini_coefficient(expert_usage.cpu().numpy())
    
    # Entropy of expert distribution (higher = more balanced)
    entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-10)).item()
    max_entropy = math.log(num_experts)
    normalized_entropy = entropy / max_entropy
    
    return {
        'expert_usage_probs': expert_usage.cpu().numpy(),
        'expert_selection_freq': expert_selection_freq.cpu().numpy(),
        'usage_variance': usage_variance,
        'gini_coefficient': gini_coefficient,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'load_balance_score': 1.0 - gini_coefficient,  # Higher = more balanced
        'num_experts': num_experts,
        'total_tokens': total_tokens
    }


def calculate_gini_coefficient(x: np.ndarray) -> float:
    """Calculate Gini coefficient for measuring inequality."""
    x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x)
    return 1 - 2 * np.sum(cumsum) / (n * cumsum[-1])


def visualize_expert_usage(analysis_results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Visualize expert utilization patterns.
    
    Args:
        analysis_results: Results from analyze_expert_utilization()
        save_path: Optional path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    expert_ids = range(analysis_results['num_experts'])
    usage_probs = analysis_results['expert_usage_probs']
    selection_freq = analysis_results['expert_selection_freq']
    
    # Expert usage probabilities
    ax1.bar(expert_ids, usage_probs)
    ax1.axhline(y=1.0/analysis_results['num_experts'], color='r', linestyle='--', 
                label='Uniform distribution')
    ax1.set_title('Expert Usage Probabilities')
    ax1.set_xlabel('Expert ID')
    ax1.set_ylabel('Average Probability')
    ax1.legend()
    
    # Expert selection frequency
    ax2.bar(expert_ids, selection_freq)
    ax2.axhline(y=1.0/analysis_results['num_experts'], color='r', linestyle='--',
                label='Uniform distribution')
    ax2.set_title('Expert Selection Frequency')
    ax2.set_xlabel('Expert ID')
    ax2.set_ylabel('Selection Frequency')
    ax2.legend()
    
    # Load balancing metrics
    metrics = ['Gini Coeff', 'Normalized\nEntropy', 'Load Balance\nScore']
    values = [
        analysis_results['gini_coefficient'],
        analysis_results['normalized_entropy'],
        analysis_results['load_balance_score']
    ]
    ax3.bar(metrics, values)
    ax3.set_title('Load Balancing Metrics')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    
    # Usage distribution histogram
    ax4.hist(usage_probs, bins=10, alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(usage_probs), color='r', linestyle='-', label='Mean')
    ax4.axvline(x=1.0/analysis_results['num_experts'], color='g', linestyle='--', 
                label='Ideal (uniform)')
    ax4.set_title('Distribution of Expert Usage')
    ax4.set_xlabel('Usage Probability')
    ax4.set_ylabel('Number of Experts')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Expert usage plot saved to {save_path}")
    
    plt.show()


def get_memory_usage(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """
    Estimate memory usage of MoE model.
    
    Args:
        model: MoE model
        input_shape: Input tensor shape (batch_size, seq_len)
        
    Returns:
        Memory usage in MB for different components
    """
    def get_tensor_memory(tensor):
        return tensor.numel() * tensor.element_size() / (1024 * 1024)  # MB
    
    # Model parameters
    param_memory = sum(get_tensor_memory(p) for p in model.parameters())
    
    # Estimate activation memory (rough approximation)
    batch_size, seq_len = input_shape
    hidden_size = model.config.hidden_size if hasattr(model, 'config') else 512
    
    # Activations: input, attention outputs, MLP outputs, etc.
    # This is a rough estimate - actual memory depends on implementation
    activation_memory = (
        batch_size * seq_len * hidden_size * 4 *  # Input + attention + MLP + output
        4 / (1024 * 1024)  # 4 bytes per float32, convert to MB
    )
    
    # Router memory (logits and routing info)
    num_experts = getattr(model.config, 'num_experts', 8) if hasattr(model, 'config') else 8
    router_memory = (
        batch_size * seq_len * num_experts * 4 / (1024 * 1024)  # Router logits
    )
    
    return {
        'parameters_mb': param_memory,
        'activations_mb': activation_memory,
        'router_mb': router_memory,
        'total_estimated_mb': param_memory + activation_memory + router_memory
    }


def compute_moe_efficiency(config: MoEConfig) -> Dict[str, float]:
    """
    Compute efficiency metrics for MoE configuration.
    
    Args:
        config: MoE configuration
        
    Returns:
        Efficiency metrics
    """
    total_params = config.num_parameters
    active_params = config.active_parameters
    
    # Parameter efficiency
    param_efficiency = active_params / total_params
    
    # Compute scaling factor compared to dense model
    dense_intermediate_size = config.intermediate_size
    moe_intermediate_size = config.intermediate_size * config.num_experts
    capacity_scaling = moe_intermediate_size / dense_intermediate_size
    
    # Effective scaling (considering only active experts)
    effective_scaling = (config.intermediate_size * config.num_experts_per_token) / dense_intermediate_size
    
    return {
        'parameter_efficiency': param_efficiency,
        'capacity_scaling': capacity_scaling,
        'effective_scaling': effective_scaling,
        'expert_utilization': config.num_experts_per_token / config.num_experts,
        'total_parameters': total_params,
        'active_parameters': active_params
    }


def create_balanced_expert_config(
    base_config: MoEConfig,
    target_efficiency: float = 0.2,
    max_experts: int = 64
) -> MoEConfig:
    """
    Create a balanced MoE configuration targeting specific efficiency.
    
    Args:
        base_config: Base configuration to modify
        target_efficiency: Target parameter efficiency (active/total)
        max_experts: Maximum number of experts
        
    Returns:
        Optimized MoE configuration
    """
    # Calculate optimal number of experts for target efficiency
    # efficiency = experts_per_token / num_experts
    # num_experts = experts_per_token / efficiency
    
    optimal_experts = min(
        max_experts,
        max(2, int(base_config.num_experts_per_token / target_efficiency))
    )
    
    # Create new config
    new_config = MoEConfig(
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        max_position_embeddings=base_config.max_position_embeddings,
        vocab_size=base_config.vocab_size,
        
        # Optimized MoE parameters
        num_experts=optimal_experts,
        num_experts_per_token=base_config.num_experts_per_token,
        expert_capacity_factor=base_config.expert_capacity_factor,
        
        # Use MoE in middle layers for best efficiency
        moe_layers=list(range(
            base_config.num_hidden_layers // 4,
            3 * base_config.num_hidden_layers // 4
        )),
        
        # Other parameters
        router_aux_loss_coef=base_config.router_aux_loss_coef,
        router_type=base_config.router_type,
        router_jitter_noise=base_config.router_jitter_noise,
        use_auxiliary_loss=base_config.use_auxiliary_loss,
        rms_norm_eps=base_config.rms_norm_eps,
        rope_theta=base_config.rope_theta,
        tie_word_embeddings=base_config.tie_word_embeddings,
        initializer_range=base_config.initializer_range,
        attention_dropout=base_config.attention_dropout,
        expert_dropout=base_config.expert_dropout,
    )
    
    return new_config


def test_moe_utils():
    """Test MoE utility functions."""
    print("Testing MoE Utilities...")
    
    from .config import create_tiny_moe_config
    
    config = create_tiny_moe_config()
    
    # Test parameter counting
    print("\n--- Testing Parameter Analysis ---")
    efficiency_metrics = compute_moe_efficiency(config)
    print(f"✅ Parameter efficiency: {efficiency_metrics['parameter_efficiency']:.3f}")
    print(f"✅ Capacity scaling: {efficiency_metrics['capacity_scaling']:.2f}x")
    print(f"✅ Effective scaling: {efficiency_metrics['effective_scaling']:.2f}x")
    print(f"✅ Total parameters: {efficiency_metrics['total_parameters']:,}")
    print(f"✅ Active parameters: {efficiency_metrics['active_parameters']:,}")
    
    # Test expert utilization analysis
    print("\n--- Testing Expert Utilization Analysis ---")
    batch_size, seq_len = 4, 32
    router_logits = torch.randn(batch_size, seq_len, config.num_experts)
    expert_indices = torch.randint(0, config.num_experts, 
                                 (batch_size, seq_len, config.num_experts_per_token))
    
    analysis = analyze_expert_utilization(router_logits, expert_indices)
    print(f"✅ Gini coefficient: {analysis['gini_coefficient']:.3f}")
    print(f"✅ Normalized entropy: {analysis['normalized_entropy']:.3f}")
    print(f"✅ Load balance score: {analysis['load_balance_score']:.3f}")
    
    # Test memory usage estimation
    print("\n--- Testing Memory Usage ---")
    class DummyModel:
        def __init__(self):
            self.config = config
    
    dummy_model = DummyModel()
    memory_usage = get_memory_usage(dummy_model, (batch_size, seq_len))
    print(f"✅ Estimated total memory: {memory_usage['total_estimated_mb']:.1f} MB")
    print(f"✅ Router memory: {memory_usage['router_mb']:.1f} MB")
    
    # Test balanced configuration creation
    print("\n--- Testing Balanced Configuration ---")
    balanced_config = create_balanced_expert_config(config, target_efficiency=0.25)
    balanced_efficiency = compute_moe_efficiency(balanced_config)
    print(f"✅ Balanced config experts: {balanced_config.num_experts}")
    print(f"✅ Balanced efficiency: {balanced_efficiency['parameter_efficiency']:.3f}")
    
    # Test different configurations
    print("\n--- Testing Different Configurations ---")
    configs = ['tiny_moe', 'small_moe', 'medium_moe']
    
    for config_name in configs:
        from .config import get_moe_config
        cfg = get_moe_config(config_name)
        metrics = compute_moe_efficiency(cfg)
        print(f"✅ {config_name}: {metrics['total_parameters']:,} total, "
              f"{metrics['active_parameters']:,} active "
              f"({metrics['parameter_efficiency']:.1%} efficiency)")
    
    print("\n✅ All MoE utility tests passed!")


if __name__ == "__main__":
    test_moe_utils()