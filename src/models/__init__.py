# training_infra/models/__init__.py
"""
Model architectures and implementations for LLaMA training.

This module provides LLaMA model implementations with support for distributed
training, including all variants from original LLaMA to LLaMA 3 and Tiny models.
"""

from .llama import (
    LlamaConfig,
    RMSNorm,
    RotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    
    # Model creation functions
    create_llama_7b_parallel,
    create_llama_13b_parallel,
    create_llama_30b_parallel,
    create_llama_65b_parallel,
    create_llama2_7b_parallel,
    create_code_llama_7b_parallel,
    
    # LLaMA 3 models
    create_llama3_8b_parallel,
    create_llama3_8b_instruct_parallel,
    create_llama3_70b_parallel,
    create_llama3_70b_instruct_parallel,
    create_llama3_405b_parallel,
    
    # Tiny LLaMA 3 models
    create_tiny_llama3_150m,
    create_tiny_llama3_50m,
    
    # Utilities
    optimize_model_for_training,
    estimate_model_memory,
    apply_rotary_pos_emb,
    rotate_half
)

try:
    from .moe import (
        MoEConfig,
        MoELayer,
        SparseMoEBlock,
        MoETransformerLayer,
        MoELlamaLayer,
        Router,
        Expert,
        TopKGating
    )
    _MOE_AVAILABLE = True
except ImportError:
    _MOE_AVAILABLE = False

# Core exports - always available
__all__ = [
    # Configuration
    "LlamaConfig",
    
    # Model components
    "RMSNorm",
    "RotaryEmbedding", 
    "LlamaAttention",
    "LlamaMLP",
    "LlamaDecoderLayer",
    "LlamaModel",
    "LlamaForCausalLM",
    
    # Model creation functions
    "create_llama_7b_parallel",
    "create_llama_13b_parallel",
    "create_llama_30b_parallel", 
    "create_llama_65b_parallel",
    "create_llama2_7b_parallel",
    "create_code_llama_7b_parallel",
    
    # LLaMA 3 models
    "create_llama3_8b_parallel",
    "create_llama3_8b_instruct_parallel",
    "create_llama3_70b_parallel",
    "create_llama3_70b_instruct_parallel", 
    "create_llama3_405b_parallel",
    
    # Tiny LLaMA 3 models
    "create_tiny_llama3_150m",
    "create_tiny_llama3_50m",
    
    # Utilities
    "optimize_model_for_training",
    "estimate_model_memory",
    "apply_rotary_pos_emb",
    "rotate_half",
]

# Conditional exports for MoE
if _MOE_AVAILABLE:
    __all__.extend([
        "MoEConfig",
        "MoELayer",
        "SparseMoEBlock", 
        "MoETransformerLayer",
        "MoELlamaLayer",
        "Router",
        "Expert",
        "TopKGating"
    ])

# Module info
__version__ = "1.0.0"
__description__ = "Model architectures and implementations for LLaMA training"

def print_models_info():
    """Print information about models module"""
    print("🧠 LLaMA Models Module")
    print("=" * 35)
    
    print("Available model variants:")
    variants = {
        "LLaMA 1": ["7B"],
        "LLaMA 2": ["7B", "13B", "30B", "65B", "70B"], 
        "LLaMA 3": ["8B", "8B-Instruct", "70B", "70B-Instruct", "405B"],
        "Tiny LLaMA 3": ["50M", "150M"],
        "Code LLaMA": ["7B"]
    }
    
    for family, sizes in variants.items():
        print(f"  • {family}: {', '.join(sizes)}")
    
    print(f"\nModel components:")
    components = ["LlamaConfig", "RMSNorm", "RotaryEmbedding", "LlamaAttention", "LlamaMLP"]
    for component in components:
        print(f"  • {component}")
    
    print(f"\nOptional features:")
    print(f"  • MoE (Mixture of Experts): {'✅' if _MOE_AVAILABLE else '❌'}")

def list_model_creators():
    """List all available model creation functions"""
    print("🏗️  Model Creation Functions:")
    print("=" * 40)
    
    creators = [name for name in __all__ if name.startswith("create_")]
    
    # Group by family
    families = {
        "LLaMA 1": [],
        "LLaMA 2": [],
        "LLaMA 3": [],
        "Tiny LLaMA 3": [],
        "Code LLaMA": []
    }
    
    for creator in creators:
        if "llama1" in creator:
            families["LLaMA 1"].append(creator)
        elif "llama2" in creator:
            families["LLaMA 2"].append(creator)
        elif "llama3" in creator and "tiny" not in creator:
            families["LLaMA 3"].append(creator)
        elif "tiny" in creator:
            families["Tiny LLaMA 3"].append(creator)
        elif "code" in creator:
            families["Code LLaMA"].append(creator)
    
    for family, funcs in families.items():
        if funcs:
            print(f"\n{family}:")
            for func in funcs:
                print(f"  • {func}")

def get_model_info(model_variant: str) -> dict:
    """Get information about a specific model variant"""
    
    model_specs = {
        # LLaMA 1
        "llama1_7b": {
            "parameters": "7B",
            "vocab_size": 32000,
            "context_length": 2048,
            "architecture": "Original LLaMA",
            "min_gpus": 1,
            "recommended_memory_gb": 16
        },
        
        # LLaMA 2  
        "llama2_7b": {
            "parameters": "7B", 
            "vocab_size": 32000,
            "context_length": 4096,
            "architecture": "LLaMA 2 with extended context",
            "min_gpus": 1,
            "recommended_memory_gb": 18
        },
        "llama2_13b": {
            "parameters": "13B",
            "vocab_size": 32000, 
            "context_length": 4096,
            "architecture": "LLaMA 2",
            "min_gpus": 2,
            "recommended_memory_gb": 32
        },
        "llama2_70b": {
            "parameters": "70B",
            "vocab_size": 32000,
            "context_length": 4096, 
            "architecture": "LLaMA 2 with GQA",
            "min_gpus": 8,
            "recommended_memory_gb": 200
        },
        
        # LLaMA 3
        "llama3_8b": {
            "parameters": "8B",
            "vocab_size": 128256,
            "context_length": 8192,
            "architecture": "LLaMA 3 enhanced",
            "min_gpus": 1, 
            "recommended_memory_gb": 20
        },
        "llama3_70b": {
            "parameters": "70B",
            "vocab_size": 128256,
            "context_length": 8192,
            "architecture": "LLaMA 3 with improved GQA", 
            "min_gpus": 8,
            "recommended_memory_gb": 220
        },
        "llama3_405b": {
            "parameters": "405B",
            "vocab_size": 128256,
            "context_length": 8192,
            "architecture": "LLaMA 3 mega model",
            "min_gpus": 32,
            "recommended_memory_gb": 1000
        },
        
        # Tiny LLaMA 3
        "tiny_llama3_50m": {
            "parameters": "50M",
            "vocab_size": 128256,
            "context_length": 2048,
            "architecture": "Tiny LLaMA 3 for development",
            "min_gpus": 0,  # Can run on CPU
            "recommended_memory_gb": 2
        },
        "tiny_llama3_150m": {
            "parameters": "150M", 
            "vocab_size": 128256,
            "context_length": 8192,
            "architecture": "Tiny LLaMA 3 for development",
            "min_gpus": 1,
            "recommended_memory_gb": 4
        },
        
        # Code LLaMA
        "code_llama_7b": {
            "parameters": "7B",
            "vocab_size": 32016,
            "context_length": 16384,
            "architecture": "Code-specialized LLaMA",
            "min_gpus": 1,
            "recommended_memory_gb": 20
        }
    }
    
    if model_variant not in model_specs:
        available = ", ".join(model_specs.keys())
        raise ValueError(f"Unknown model variant: {model_variant}. Available: {available}")
    
    return model_specs[model_variant]

def recommend_model_for_hardware(
    gpu_count: int = 1,
    gpu_memory_gb: float = 8,
    use_case: str = "training"  # training, inference, development
) -> list:
    """Recommend suitable models based on available hardware"""
    
    recommendations = []
    
    # Development models (always good for testing)
    if use_case == "development":
        recommendations.extend([
            ("tiny_llama3_50m", "Ultra-fast development and testing"),
            ("tiny_llama3_150m", "Development with full architecture")
        ])
    
    # Production recommendations based on hardware
    if gpu_count == 0 or gpu_memory_gb < 4:
        recommendations.append(("tiny_llama3_50m", "CPU or very low memory"))
    
    elif gpu_memory_gb >= 4 and gpu_memory_gb < 16:
        recommendations.append(("tiny_llama3_150m", "Low memory GPU"))
    
    elif gpu_memory_gb >= 16 and gpu_count >= 1:
        recommendations.extend([
            ("llama3_8b", "Good balance of capability and efficiency"),
            ("llama2_7b", "Proven model with good performance")
        ])
        
        if use_case == "inference" and gpu_memory_gb >= 20:
            recommendations.append(("llama3_8b_instruct", "Best for chat/instruction following"))
    
    elif gpu_memory_gb >= 40 and gpu_count >= 8:
        recommendations.extend([
            ("llama3_70b", "High capability model"),
            ("llama3_70b_instruct", "Best instruction following")
        ])
    
    elif gpu_memory_gb >= 80 and gpu_count >= 32:
        recommendations.append(("llama3_405b", "State-of-the-art mega model"))
    
    return recommendations

def create_model_comparison_table():
    """Create a comparison table of all models"""
    print("📊 LLaMA Model Comparison")
    print("=" * 80)
    print(f"{'Model':<20} {'Parameters':<12} {'Context':<8} {'Vocab':<8} {'Min GPUs':<9} {'Memory (GB)'}")
    print("-" * 80)
    
    models = [
        "tiny_llama3_50m", "tiny_llama3_150m", "llama2_7b", "llama3_8b", 
        "llama2_13b", "llama2_70b", "llama3_70b", "llama3_405b", "code_llama_7b"
    ]
    
    for model in models:
        try:
            info = get_model_info(model)
            print(f"{model:<20} {info['parameters']:<12} {info['context_length']:<8} "
                  f"{info['vocab_size']:<8} {info['min_gpus']:<9} {info['recommended_memory_gb']}")
        except:
            continue

def quick_model_create(model_variant: str, **kwargs):
    """Quick model creation with sensible defaults"""
    
    creators = {
        "tiny_llama3_50m": create_tiny_llama3_50m,
        "tiny_llama3_150m": create_tiny_llama3_150m,
        "llama1_7b": create_llama_7b_parallel,
        "llama2_7b": create_llama2_7b_parallel,
        "llama2_13b": create_llama_13b_parallel,
        "llama2_30b": create_llama_30b_parallel,
        "llama2_65b": create_llama_65b_parallel,
        "llama2_70b": create_llama_65b_parallel,  # Using 65b as proxy
        "llama3_8b": create_llama3_8b_parallel,
        "llama3_8b_instruct": create_llama3_8b_instruct_parallel,
        "llama3_70b": create_llama3_70b_parallel,
        "llama3_70b_instruct": create_llama3_70b_instruct_parallel,
        "llama3_405b": create_llama3_405b_parallel,
        "code_llama_7b": create_code_llama_7b_parallel,
    }
    
    if model_variant not in creators:
        available = ", ".join(creators.keys())
        raise ValueError(f"Unknown model variant: {model_variant}. Available: {available}")
    
    # Set sensible defaults
    defaults = {
        "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
        "use_flash_attention": kwargs.get("use_flash_attention", True),
        "use_checkpointing": kwargs.get("use_checkpointing", False)
    }
    
    # Override defaults with kwargs
    defaults.update(kwargs)
    
    return creators[model_variant](**defaults)

# Add convenience functions to exports
__all__.extend([
    "print_models_info",
    "list_model_creators",
    "get_model_info", 
    "recommend_model_for_hardware",
    "create_model_comparison_table",
    "quick_model_create"
])

# Model configuration presets
MODEL_PRESETS = {
    "development": {
        "model": "tiny_llama3_150m",
        "description": "Fast development and testing",
        "config": {
            "tensor_parallel_size": 1,
            "use_flash_attention": True,
            "use_checkpointing": False
        }
    },
    "production_small": {
        "model": "llama3_8b", 
        "description": "Production ready, efficient",
        "config": {
            "tensor_parallel_size": 1,
            "use_flash_attention": True,
            "use_checkpointing": True
        }
    },
    "production_large": {
        "model": "llama3_70b",
        "description": "High capability production model",
        "config": {
            "tensor_parallel_size": 8,
            "use_flash_attention": True,
            "use_checkpointing": True
        }
    },
    "chat_optimized": {
        "model": "llama3_8b_instruct",
        "description": "Optimized for conversation",
        "config": {
            "tensor_parallel_size": 1,
            "use_flash_attention": True,
            "use_checkpointing": False
        }
    },
    "code_specialized": {
        "model": "code_llama_7b",
        "description": "Specialized for code generation",
        "config": {
            "tensor_parallel_size": 1,
            "use_flash_attention": True,
            "use_checkpointing": True
        }
    }
}

def create_model_from_preset(preset_name: str, **overrides):
    """Create model from preset configuration"""
    
    if preset_name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    preset = MODEL_PRESETS[preset_name]
    config = preset["config"].copy()
    config.update(overrides)
    
    return quick_model_create(preset["model"], **config)

def list_model_presets():
    """List available model presets"""
    print("📋 Model Presets:")
    print("=" * 40)
    
    for name, preset in MODEL_PRESETS.items():
        print(f"  • {name}: {preset['description']}")
        print(f"    Model: {preset['model']}")

# Add preset functions to exports
__all__.extend([
    "create_model_from_preset",
    "list_model_presets",
    "MODEL_PRESETS"
])

# Model utilities
class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def count_parameters(model) -> dict:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "total_parameters_M": total_params / 1_000_000,
            "trainable_parameters_M": trainable_params / 1_000_000
        }
    
    @staticmethod
    def get_model_size_mb(model) -> float:
        """Get model size in MB"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / 1024 / 1024
    
    @staticmethod
    def print_model_summary(model, model_name: str = "Model"):
        """Print detailed model summary"""
        param_info = ModelUtils.count_parameters(model)
        size_mb = ModelUtils.get_model_size_mb(model)
        
        print(f"🧠 {model_name} Summary")
        print("=" * 40)
        print(f"Total Parameters: {param_info['total_parameters']:,} ({param_info['total_parameters_M']:.1f}M)")
        print(f"Trainable Parameters: {param_info['trainable_parameters']:,} ({param_info['trainable_parameters_M']:.1f}M)")
        print(f"Model Size: {size_mb:.1f} MB")
        
        if hasattr(model, 'config'):
            config = model.config
            print(f"\nArchitecture:")
            print(f"  • Vocab Size: {config.vocab_size:,}")
            print(f"  • Hidden Size: {config.hidden_size}")
            print(f"  • Layers: {config.num_hidden_layers}")
            print(f"  • Attention Heads: {config.num_attention_heads}")
            if hasattr(config, 'num_key_value_heads') and config.num_key_value_heads:
                print(f"  • KV Heads: {config.num_key_value_heads} (GQA)")
            print(f"  • Max Position: {config.max_position_embeddings}")
    
    @staticmethod
    def compare_models(*models, names=None) -> dict:
        """Compare multiple models"""
        if names is None:
            names = [f"Model_{i+1}" for i in range(len(models))]
        
        comparison = {}
        for model, name in zip(models, names):
            comparison[name] = ModelUtils.count_parameters(model)
            comparison[name]["size_mb"] = ModelUtils.get_model_size_mb(model)
        
        return comparison

# Add ModelUtils to exports
__all__.append("ModelUtils")

# Architecture validation
def validate_model_architecture(model_variant: str) -> dict:
    """Validate model architecture and requirements"""
    
    try:
        info = get_model_info(model_variant)
        
        validation = {
            "valid": True,
            "warnings": [],
            "requirements": info,
            "recommendations": []
        }
        
        # Check memory requirements
        if info["recommended_memory_gb"] > 80:
            validation["warnings"].append("High memory requirements")
            validation["recommendations"].append("Consider using tensor parallelism")
        
        # Check GPU requirements
        if info["min_gpus"] > 1:
            validation["recommendations"].append(f"Requires at least {info['min_gpus']} GPUs")
        
        # Check for development models
        if "tiny" in model_variant:
            validation["recommendations"].append("Perfect for development and testing")
        
        return validation
        
    except ValueError as e:
        return {
            "valid": False,
            "error": str(e),
            "warnings": ["Invalid model variant"],
            "recommendations": ["Check available models with list_model_creators()"]
        }

# Add validation function to exports
__all__.append("validate_model_architecture")