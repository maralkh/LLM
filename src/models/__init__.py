# training_infra/models/__init__.py
"""
Model architectures and implementations for LLaMA training.

This module provides LLaMA model implementations with support for distributed
training, including all variants from original LLaMA to LLaMA 3 and Tiny models.
"""

# Core LLaMA imports - these should always be available
try:
    from .llama import (
        LlamaConfig,
        RMSNorm,
        RotaryEmbedding,
        LlamaAttention,
        LlamaMLP,
        LlamaDecoderLayer,
        LlamaModel,
        LlamaForCausalLM,
        
        # Utilities
        apply_rotary_pos_emb,
        rotate_half
    )
    _LLAMA_CORE_AVAILABLE = True
except ImportError:
    _LLAMA_CORE_AVAILABLE = False

# Model creation functions - might not all be implemented
try:
    from .llama import (
        create_llama_7b_parallel,
        create_llama_13b_parallel,
        create_llama_30b_parallel,
        create_llama_65b_parallel,
        create_llama2_7b_parallel,
        create_code_llama_7b_parallel,
    )
    _LLAMA_CREATORS_AVAILABLE = True
except ImportError:
    _LLAMA_CREATORS_AVAILABLE = False

# LLaMA 3 models - might be newer implementations
try:
    from .llama import (
        create_llama3_8b_parallel,
        create_llama3_8b_instruct_parallel,
        create_llama3_70b_parallel,
        create_llama3_70b_instruct_parallel,
        create_llama3_405b_parallel,
    )
    _LLAMA3_AVAILABLE = True
except ImportError:
    _LLAMA3_AVAILABLE = False

# Tiny LLaMA 3 models - might be development only
try:
    from .llama import (
        create_tiny_llama3_150m,
        create_tiny_llama3_50m,
    )
    _TINY_LLAMA_AVAILABLE = True
except ImportError:
    _TINY_LLAMA_AVAILABLE = False

# Utility functions - might have dependencies
try:
    from .llama import (
        optimize_model_for_training,
        estimate_model_memory,
    )
    _LLAMA_UTILS_AVAILABLE = True
except ImportError:
    _LLAMA_UTILS_AVAILABLE = False

# MoE imports - completely optional
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

# Advanced MoE - even more optional
try:
    from .moe import (
        SwitchTransformerMoE,
        GLaMRouter,
        LlamaMoEModel,
        create_llama_moe_7b,
        create_switch_transformer,
        create_glam_model
    )
    _ADVANCED_MOE_AVAILABLE = True
except ImportError:
    _ADVANCED_MOE_AVAILABLE = False

# Core exports - only what's definitely available
__all__ = []

# Add core LLaMA exports if available
if _LLAMA_CORE_AVAILABLE:
    __all__.extend([
        "LlamaConfig",
        "RMSNorm",
        "RotaryEmbedding", 
        "LlamaAttention",
        "LlamaMLP",
        "LlamaDecoderLayer",
        "LlamaModel",
        "LlamaForCausalLM",
        "apply_rotary_pos_emb",
        "rotate_half",
    ])

# Add model creators if available
if _LLAMA_CREATORS_AVAILABLE:
    __all__.extend([
        "create_llama_7b_parallel",
        "create_llama_13b_parallel",
        "create_llama_30b_parallel", 
        "create_llama_65b_parallel",
        "create_llama2_7b_parallel",
        "create_code_llama_7b_parallel",
    ])

if _LLAMA3_AVAILABLE:
    __all__.extend([
        "create_llama3_8b_parallel",
        "create_llama3_8b_instruct_parallel",
        "create_llama3_70b_parallel",
        "create_llama3_70b_instruct_parallel", 
        "create_llama3_405b_parallel",
    ])

if _TINY_LLAMA_AVAILABLE:
    __all__.extend([
        "create_tiny_llama3_150m",
        "create_tiny_llama3_50m",
    ])

if _LLAMA_UTILS_AVAILABLE:
    __all__.extend([
        "optimize_model_for_training",
        "estimate_model_memory",
    ])

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

if _ADVANCED_MOE_AVAILABLE:
    __all__.extend([
        "SwitchTransformerMoE",
        "GLaMRouter",
        "LlamaMoEModel",
        "create_llama_moe_7b",
        "create_switch_transformer",
        "create_glam_model"
    ])

# Module info
__version__ = "1.0.0"
__description__ = "Model architectures and implementations for LLaMA training"

def print_models_info():
    """Print information about models module"""
    print("🧠 LLaMA Models Module")
    print("=" * 35)
    
    print("Available components:")
    print(f"  • Core LLaMA: {'✅' if _LLAMA_CORE_AVAILABLE else '❌'}")
    print(f"  • Model Creators: {'✅' if _LLAMA_CREATORS_AVAILABLE else '❌'}")
    print(f"  • LLaMA 3: {'✅' if _LLAMA3_AVAILABLE else '❌'}")
    print(f"  • Tiny LLaMA: {'✅' if _TINY_LLAMA_AVAILABLE else '❌'}")
    print(f"  • Utilities: {'✅' if _LLAMA_UTILS_AVAILABLE else '❌'}")
    print(f"  • MoE Support: {'✅' if _MOE_AVAILABLE else '❌'}")
    print(f"  • Advanced MoE: {'✅' if _ADVANCED_MOE_AVAILABLE else '❌'}")
    
    if _LLAMA_CORE_AVAILABLE:
        print("\nAvailable model variants:")
        variants = []
        if _LLAMA_CREATORS_AVAILABLE:
            variants.extend(["LLaMA 1: 7B", "LLaMA 2: 7B, 13B, 30B, 65B", "Code LLaMA: 7B"])
        if _LLAMA3_AVAILABLE:
            variants.extend(["LLaMA 3: 8B, 8B-Instruct, 70B, 70B-Instruct, 405B"])
        if _TINY_LLAMA_AVAILABLE:
            variants.append("Tiny LLaMA 3: 50M, 150M")
        
        for variant in variants:
            print(f"  • {variant}")

def list_model_creators():
    """List all available model creation functions"""
    print("🏗️  Model Creation Functions:")
    print("=" * 40)
    
    if not any([_LLAMA_CREATORS_AVAILABLE, _LLAMA3_AVAILABLE, _TINY_LLAMA_AVAILABLE]):
        print("❌ No model creators available")
        return
    
    creators = [name for name in __all__ if name.startswith("create_")]
    
    if creators:
        for creator in sorted(creators):
            print(f"  • {creator}")
    else:
        print("❌ No creators found in exports")

def get_model_info(model_variant: str) -> dict:
    """Get information about a specific model variant"""
    
    # Safe model specs - only include what we're confident about
    model_specs = {
        # LLaMA 2  
        "llama2_7b": {
            "parameters": "7B", 
            "vocab_size": 32000,
            "context_length": 4096,
            "architecture": "LLaMA 2",
            "min_gpus": 1,
            "recommended_memory_gb": 18
        },
        
        # LLaMA 3
        "llama3_8b": {
            "parameters": "8B",
            "vocab_size": 128256,
            "context_length": 8192,
            "architecture": "LLaMA 3",
            "min_gpus": 1, 
            "recommended_memory_gb": 20
        },
        
        # Tiny LLaMA 3
        "tiny_llama3_50m": {
            "parameters": "50M",
            "vocab_size": 128256,
            "context_length": 2048,
            "architecture": "Tiny LLaMA 3",
            "min_gpus": 0,  # Can run on CPU
            "recommended_memory_gb": 2
        },
        "tiny_llama3_150m": {
            "parameters": "150M", 
            "vocab_size": 128256,
            "context_length": 8192,
            "architecture": "Tiny LLaMA 3",
            "min_gpus": 1,
            "recommended_memory_gb": 4
        },
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
    if use_case == "development" and _TINY_LLAMA_AVAILABLE:
        recommendations.extend([
            ("tiny_llama3_50m", "Ultra-fast development and testing"),
            ("tiny_llama3_150m", "Development with full architecture")
        ])
    
    # Production recommendations based on hardware
    if gpu_count == 0 or gpu_memory_gb < 4:
        if _TINY_LLAMA_AVAILABLE:
            recommendations.append(("tiny_llama3_50m", "CPU or very low memory"))
    
    elif gpu_memory_gb >= 4 and gpu_memory_gb < 16:
        if _TINY_LLAMA_AVAILABLE:
            recommendations.append(("tiny_llama3_150m", "Low memory GPU"))
    
    elif gpu_memory_gb >= 16 and gpu_count >= 1:
        if _LLAMA3_AVAILABLE:
            recommendations.append(("llama3_8b", "Good balance of capability and efficiency"))
        if _LLAMA_CREATORS_AVAILABLE:
            recommendations.append(("llama2_7b", "Proven model with good performance"))
    
    if not recommendations:
        recommendations.append(("manual_setup", "Please check available model creators"))
    
    return recommendations

def quick_model_create(model_variant: str, **kwargs):
    """Quick model creation with error handling"""
    
    # Map variants to creators (only include what we know exists)
    creators = {}
    
    if _TINY_LLAMA_AVAILABLE:
        creators.update({
            "tiny_llama3_50m": create_tiny_llama3_50m,
            "tiny_llama3_150m": create_tiny_llama3_150m,
        })
    
    if _LLAMA_CREATORS_AVAILABLE:
        creators.update({
            "llama2_7b": create_llama2_7b_parallel,
            "llama_7b": create_llama_7b_parallel,
            "llama_13b": create_llama_13b_parallel,
        })
    
    if _LLAMA3_AVAILABLE:
        creators.update({
            "llama3_8b": create_llama3_8b_parallel,
            "llama3_8b_instruct": create_llama3_8b_instruct_parallel,
        })
    
    if model_variant not in creators:
        available = ", ".join(creators.keys()) if creators else "None available"
        raise ValueError(f"Model {model_variant} not available. Available: {available}")
    
    # Set sensible defaults
    defaults = {
        "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
        "use_flash_attention": kwargs.get("use_flash_attention", True),
        "use_checkpointing": kwargs.get("use_checkpointing", False)
    }
    
    # Override defaults with kwargs
    defaults.update(kwargs)
    
    try:
        return creators[model_variant](**defaults)
    except Exception as e:
        raise RuntimeError(f"Failed to create {model_variant}: {e}")

# Model utilities with error handling
class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def count_parameters(model) -> dict:
        """Count model parameters"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params,
                "total_parameters_M": total_params / 1_000_000,
                "trainable_parameters_M": trainable_params / 1_000_000
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def get_model_size_mb(model) -> float:
        """Get model size in MB"""
        try:
            total_size = 0
            for param in model.parameters():
                total_size += param.numel() * param.element_size()
            return total_size / 1024 / 1024
        except:
            return 0.0
    
    @staticmethod
    def print_model_summary(model, model_name: str = "Model"):
        """Print detailed model summary"""
        try:
            param_info = ModelUtils.count_parameters(model)
            size_mb = ModelUtils.get_model_size_mb(model)
            
            print(f"🧠 {model_name} Summary")
            print("=" * 40)
            
            if "error" not in param_info:
                print(f"Total Parameters: {param_info['total_parameters']:,} ({param_info['total_parameters_M']:.1f}M)")
                print(f"Trainable Parameters: {param_info['trainable_parameters']:,} ({param_info['trainable_parameters_M']:.1f}M)")
                print(f"Model Size: {size_mb:.1f} MB")
            else:
                print(f"Error getting model info: {param_info['error']}")
            
            if hasattr(model, 'config'):
                config = model.config
                print(f"\nArchitecture:")
                print(f"  • Vocab Size: {getattr(config, 'vocab_size', 'Unknown'):,}")
                print(f"  • Hidden Size: {getattr(config, 'hidden_size', 'Unknown')}")
                print(f"  • Layers: {getattr(config, 'num_hidden_layers', 'Unknown')}")
                print(f"  • Attention Heads: {getattr(config, 'num_attention_heads', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Error printing model summary: {e}")

# Safe function creation with availability checks
def create_model_from_preset(preset_name: str, **overrides):
    """Create model from preset configuration (if creators available)"""
    
    presets = {}
    
    if _TINY_LLAMA_AVAILABLE:
        presets["development"] = {
            "model": "tiny_llama3_150m",
            "description": "Fast development and testing",
        }
    
    if _LLAMA3_AVAILABLE:
        presets["production_small"] = {
            "model": "llama3_8b", 
            "description": "Production ready, efficient",
        }
    
    if not presets:
        raise RuntimeError("No model creators available for presets")
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    preset = presets[preset_name]
    return quick_model_create(preset["model"], **overrides)

# Add safe convenience functions to exports
__all__.extend([
    "print_models_info",
    "list_model_creators",
    "get_model_info", 
    "recommend_model_for_hardware",
    "ModelUtils"
])

# Only add functions that have their dependencies available
if any([_LLAMA_CREATORS_AVAILABLE, _LLAMA3_AVAILABLE, _TINY_LLAMA_AVAILABLE]):
    __all__.extend([
        "quick_model_create",
        "create_model_from_preset"
    ])

# Status check function
def check_models_status():
    """Check which model components are available"""
    status = {
        "core_llama": _LLAMA_CORE_AVAILABLE,
        "model_creators": _LLAMA_CREATORS_AVAILABLE,
        "llama3": _LLAMA3_AVAILABLE,
        "tiny_llama": _TINY_LLAMA_AVAILABLE,
        "utilities": _LLAMA_UTILS_AVAILABLE,
        "moe": _MOE_AVAILABLE,
        "advanced_moe": _ADVANCED_MOE_AVAILABLE,
    }
    
    return status

__all__.append("check_models_status")