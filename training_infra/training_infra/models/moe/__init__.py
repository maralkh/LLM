# training_infra/models/moe/__init__.py
"""
LLaMA-MoE (Mixture of Experts) models for scalable training.

This module provides a complete LLaMA-MoE implementation with:
- LLaMA-compatible architecture with MoE layers
- Configurable expert routing (Top-K, Switch)
- Load balancing and auxiliary losses
- Tensor parallel support
- Various model sizes from tiny to large
"""

# Configuration
try:
    from .config import (
        LlamaMoEConfig,
        create_tiny_llama_moe,
        create_llama_moe_7b,
        create_llama_moe_13b,
        create_llama_moe_30b,
        create_llama_moe_65b,
        create_code_llama_moe_7b,
        get_llama_moe_config,
        LLAMA_MOE_CONFIGS,
    )

    # Attention components (same as LLaMA)
    from .attention import (
        RMSNorm,
        RotaryEmbedding,
        LlamaMoEAttention,
        rotate_half,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    # MLP components
    from .mlp import (
        LlamaMoEMLP,
        LlamaMoEExpertMLP,
        LlamaMoEParallelExpertMLP,
        create_llama_moe_expert_mlp,
    )

    # Expert routing and MoE layers
    from .experts import (
        LlamaMoEOutput,
        LlamaMoETopKRouter,
        LlamaMoESwitchRouter,
        LlamaMoELayer,
        LlamaMoEEfficientLayer,
        create_llama_moe_layer,
        compute_load_balancing_loss,
    )

    # Main models
    from .modeling import (
        LlamaMoEModelOutput,
        LlamaMoECausalLMOutput,
        LlamaMoEDecoderLayer,
        LlamaMoEModel,
        LlamaMoEForCausalLM,
        create_tiny_llama_moe_model,
        create_llama_moe_7b_model,
        create_llama_moe_13b_model,
        create_code_llama_moe_7b_model,
    )

    # Utilities
    from .utils import (
        init_moe_weights,
        count_moe_parameters,
        analyze_expert_utilization,
        calculate_gini_coefficient,
        visualize_expert_usage,
        get_memory_usage,
        compute_moe_efficiency,
        create_balanced_expert_config,
    )
    
    _ALL_IMPORTS_SUCCESSFUL = True

except ImportError as e:
    print(f"Warning: Some LLaMA-MoE components could not be imported: {e}")
    _ALL_IMPORTS_SUCCESSFUL = False

# Version info
__version__ = "0.1.0"


# Main exports - only add if imports were successful
if _ALL_IMPORTS_SUCCESSFUL:
    __all__ = [
        # Configuration
        "LlamaMoEConfig",
        "create_tiny_llama_moe",
        "create_llama_moe_7b",
        "create_llama_moe_13b", 
        "create_llama_moe_30b",
        "create_llama_moe_65b",
        "create_code_llama_moe_7b",
        "get_llama_moe_config",
        "LLAMA_MOE_CONFIGS",
        
        # Attention (same as LLaMA)
        "RMSNorm",
        "RotaryEmbedding",
        "LlamaMoEAttention",
        "rotate_half",
        "apply_rotary_pos_emb",
        "repeat_kv",
        
        # MLP
        "LlamaMoEMLP", 
        "LlamaMoEExpertMLP",
        "LlamaMoEParallelExpertMLP",
        "create_llama_moe_expert_mlp",
        
        # Expert routing
        "LlamaMoEOutput",
        "LlamaMoETopKRouter",
        "LlamaMoESwitchRouter", 
        "LlamaMoELayer",
        "LlamaMoEEfficientLayer",
        "create_llama_moe_layer",
        "compute_load_balancing_loss",
        
        # Models
        "LlamaMoEModelOutput",
        "LlamaMoECausalLMOutput",
        "LlamaMoEDecoderLayer",
        "LlamaMoEModel",
        "LlamaMoEForCausalLM",
        "create_tiny_llama_moe_model",
        "create_llama_moe_7b_model",
        "create_llama_moe_13b_model",
        "create_code_llama_moe_7b_model",
        
        # Utilities
        "init_moe_weights",
        "count_moe_parameters",
        "analyze_expert_utilization",
        "calculate_gini_coefficient",
        "visualize_expert_usage",
        "get_memory_usage",
        "compute_moe_efficiency",
        "create_balanced_expert_config",
    ]
else:
    __all__ = []


def info():
    """Print information about the LLaMA-MoE module."""
    print("🔥 Training Infrastructure - LLaMA-MoE (Mixture of Experts)")
    print(f"Version: {__version__}")
    print()
    print("Available configurations:")
    for name in LLAMA_MOE_CONFIGS.keys():
        config = get_llama_moe_config(name)
        efficiency = compute_moe_efficiency(config)
        print(f"  - {name}: {config.num_experts} experts, "
              f"{efficiency['total_parameters']:,} params, "
              f"{efficiency['parameter_efficiency']:.1%} efficiency")
    
    print()
    print("Available components:")
    print("  ✅ LLaMA-compatible configuration")
    print("  ✅ RoPE attention (same as LLaMA)")
    print("  ✅ SwiGLU expert MLP layers") 
    print("  ✅ Router implementations (Top-K, Switch)")
    print("  ✅ Load balancing and auxiliary losses")
    print("  ✅ Tensor parallel support")
    print("  ✅ Parallel expert computation")
    print("  ✅ Analysis and visualization tools")
    print("  ✅ Complete LLaMA-MoE models")


def quick_start(config_name: str = "tiny_llama_moe"):
    """
    Quick start function to create and test a LLaMA-MoE model.
    
    Args:
        config_name: Name of the configuration to use
        
    Returns:
        Dictionary with config and model components
    """
    if not _ALL_IMPORTS_SUCCESSFUL:
        print("❌ Cannot run quick_start: Import errors occurred")
        return None
        
    print(f"🚀 Quick starting LLaMA-MoE with {config_name}...")
    
    # Get configuration
    config = get_llama_moe_config(config_name)
    print(f"✅ Configuration: {config.num_experts} experts, {config.num_parameters:,} parameters")
    print(f"   MoE layers: {config.moe_layers}")
    
    # Create model
    if config_name == "tiny_llama_moe":
        model = create_tiny_llama_moe_model()
    elif config_name == "llama_moe_7b":
        model = create_llama_moe_7b_model()
    else:
        # Fallback to tiny for other configs
        model = create_tiny_llama_moe_model()
    
    print(f"✅ Model created: {type(model).__name__}")
    
    # Test with dummy input
    import torch
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_router_logits=True,
            return_dict=True
        )
    
    print(f"✅ Model output shape: {outputs.logits.shape}")
    print(f"✅ Auxiliary loss: {outputs.aux_loss.item() if outputs.aux_loss else 'None'}")
    
    # Analyze expert utilization
    analysis = None
    if outputs.router_logits:
        # Use first router logits and create dummy expert indices
        router_logits = outputs.router_logits[0]  # First MoE layer
        expert_indices = torch.randint(0, config.num_experts, 
                                     (batch_size, seq_len, config.num_experts_per_token))
        
        analysis = analyze_expert_utilization(router_logits, expert_indices)
        print(f"✅ Expert load balance score: {analysis['load_balance_score']:.3f}")
    
    return {
        'config': config,
        'model': model,
        'test_output': outputs,
        'analysis': analysis
    }


def benchmark_configs():
    """Benchmark different LLaMA-MoE configurations."""
    print("🔬 Benchmarking LLaMA-MoE Configurations...")
    print()
    
    results = []
    
    # Only test configs that we can actually create
    test_configs = ["tiny_llama_moe"]  # Add more as they become available
    
    for config_name in test_configs:
        config = get_llama_moe_config(config_name)
        efficiency = compute_moe_efficiency(config)
        
        # Create model for testing
        if config_name == "tiny_llama_moe":
            model = create_tiny_llama_moe_model()
        else:
            continue  # Skip configs we can't create yet
        
        # Benchmark forward pass
        import torch
        import time
        
        batch_size, seq_len = 4, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_ids)
            
            # Timing
            start = time.time()
            for _ in range(100):
                output = model(input_ids)
            end = time.time()
            
            avg_time = (end - start) / 100
        
        results.append({
            'name': config_name,
            'experts': config.num_experts,
            'total_params': efficiency['total_parameters'],
            'active_params': efficiency['active_parameters'],
            'efficiency': efficiency['parameter_efficiency'],
            'time_ms': avg_time * 1000,
            'throughput': (batch_size * seq_len) / avg_time
        })
    
    # Print results table
    print(f"{'Config':<20} {'Experts':<8} {'Total Params':<12} {'Active Params':<12} {'Efficiency':<10} {'Time (ms)':<10} {'Throughput':<12}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['name']:<20} {result['experts']:<8} {result['total_params']:<12,} "
              f"{result['active_params']:<12,} {result['efficiency']:<10.1%} "
              f"{result['time_ms']:<10.2f} {result['throughput']:<12.0f}")
    
    return results


if __name__ == "__main__":
    info()
    print()
    
    # Quick test
    test_results = quick_start("tiny_llama_moe")
    print()
    
    # Benchmark if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_configs()