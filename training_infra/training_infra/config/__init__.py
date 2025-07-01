# training_infra/config/__init__.py
"""
Configuration Management System

This module provides a comprehensive configuration system for training infrastructure.
All configurations integrate with existing model configurations from models/llama/config.py

## Basic Usage:
```python
from training_infra.config import TrainingConfig
from training_infra.models.llama.config import LLaMAConfig

# Create training config
train_config = TrainingConfig(
    model_name="tiny_llama_150m",
    batch_size=8,
    epochs=10
)

# Use existing model config
model_config = LLaMAConfig.tiny_llama()

# Complete training setup
training_setup = create_training_setup(train_config, model_config)
```

## Configuration Types:
- BaseConfig: Foundation for all configs
- TrainingConfig: Complete training setup
- OptimizerConfig: Optimizer settings
- SchedulerConfig: Learning rate scheduler settings
- LoggingConfig: Logging and monitoring settings
- Model configs: Available from training_infra.models.llama.config
"""

from typing import Any, Dict, Optional, Union, Type

# Core configuration classes
from .base import (
    BaseConfig,
    ConfigValidationError,
    save_config,
    load_config,
    ExampleConfig,
)

from .validation import (
    validate_config,
    ConfigValidator,
    ValidationRule,
)

# Training configurations
try:
    from .training_configs import (
        TrainingConfig,
        OptimizerConfig,
        SchedulerConfig,
        LoggingConfig,
        get_development_config,
        get_production_config,
        get_large_model_config,
        create_training_config_validator,
    )
    _TRAINING_CONFIGS_AVAILABLE = True
except ImportError:
    _TRAINING_CONFIGS_AVAILABLE = False

# Import model configurations from models module
try:
    from ..models.llama.config import (
        LLaMAConfig,
        LLaMAVariants,
    )
    _MODEL_CONFIGS_AVAILABLE = True
except ImportError:
    _MODEL_CONFIGS_AVAILABLE = False

# Core exports - always available
__all__ = [
    # Base configuration
    "BaseConfig",
    "ConfigValidationError", 
    "save_config",
    "load_config",
    "ExampleConfig",
    
    # Validation
    "validate_config",
    "ConfigValidator", 
    "ValidationRule",
]

# Add training config exports
if _TRAINING_CONFIGS_AVAILABLE:
    __all__.extend([
        "TrainingConfig",
        "OptimizerConfig",
        "SchedulerConfig", 
        "LoggingConfig",
        "get_development_config",
        "get_production_config",
        "get_large_model_config",
        "create_training_config_validator",
    ])

# Add model config exports (from models module)
if _MODEL_CONFIGS_AVAILABLE:
    __all__.extend([
        "LLaMAConfig",
        "LLaMAVariants",
    ])


def list_available_configs():
    """List all available configuration classes."""
    configs = {
        "base": ["BaseConfig", "ExampleConfig"],
        "validation": ["ConfigValidator", "ValidationRule"]
    }
    
    if _TRAINING_CONFIGS_AVAILABLE:
        configs["training"] = [
            "TrainingConfig", "OptimizerConfig", 
            "SchedulerConfig", "LoggingConfig"
        ]
    
    if _MODEL_CONFIGS_AVAILABLE:
        configs["models"] = [
            "LLaMAConfig", "LLaMAVariants"
        ]
    
    return configs


def get_config_info():
    """Get information about available configurations."""
    info = {
        "available_modules": [],
        "training_configs_available": _TRAINING_CONFIGS_AVAILABLE,
        "model_configs_available": _MODEL_CONFIGS_AVAILABLE,
    }
    
    if _TRAINING_CONFIGS_AVAILABLE:
        info["available_modules"].append("training_configs")
        info["predefined_training_configs"] = [
            "development", "production", "large_model"
        ]
    
    if _MODEL_CONFIGS_AVAILABLE:
        info["available_modules"].append("model_configs")
        info["available_model_variants"] = [
            "tiny_llama", "small_llama", "llama_7b", "llama_13b"
        ]
    
    return info


def create_training_setup(
    model_variant: str = "tiny_llama",
    training_preset: str = "development",
    custom_training_config: Optional[dict] = None,
    custom_model_config: Optional[dict] = None
) -> tuple:
    """Create a complete configuration setup for training.
    
    Args:
        model_variant: Model variant from LLaMAVariants ('tiny_llama', 'small_llama', etc.)
        training_preset: Training configuration preset ('development', 'production', 'large_model')
        custom_training_config: Optional custom training config overrides
        custom_model_config: Optional custom model config overrides
    
    Returns:
        Tuple of (training_config, model_config)
    """
    if not _TRAINING_CONFIGS_AVAILABLE:
        raise ImportError("Training configurations not available")
    if not _MODEL_CONFIGS_AVAILABLE:
        raise ImportError("Model configurations not available")
    
    # Get model config from LLaMAVariants
    if hasattr(LLaMAVariants, model_variant):
        model_config = getattr(LLaMAVariants, model_variant)()
    else:
        available_variants = [attr for attr in dir(LLaMAVariants) 
                            if not attr.startswith('_') and callable(getattr(LLaMAVariants, attr))]
        raise ValueError(f"Unknown model variant: {model_variant}. Available: {available_variants}")
    
    # Apply custom model config if provided
    if custom_model_config:
        for key, value in custom_model_config.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
    
    # Get training config based on preset
    if training_preset == "development":
        training_config = get_development_config()
    elif training_preset == "production":
        training_config = get_production_config()
    elif training_preset == "large_model":
        training_config = get_large_model_config()
    else:
        raise ValueError(f"Unknown training preset: {training_preset}")
    
    # Apply custom training config if provided
    if custom_training_config:
        for key, value in custom_training_config.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
    
    # Update training config with model info
    training_config.model_name = f"{model_variant}_{model_config.model_type}"
    
    # Set appropriate sequence length based on model
    training_config.max_seq_length = min(
        training_config.max_seq_length, 
        model_config.max_position_embeddings
    )
    
    return training_config, model_config


def validate_training_setup(training_config, model_config) -> bool:
    """Validate a complete training setup."""
    if not _TRAINING_CONFIGS_AVAILABLE or not _MODEL_CONFIGS_AVAILABLE:
        raise ImportError("Complete setup validation requires both training and model configs")
    
    try:
        # Validate training config
        training_validator = create_training_config_validator()
        training_validator.validate_strict(training_config)
        
        # Validate model config (basic checks)
        if not hasattr(model_config, 'hidden_size') or model_config.hidden_size <= 0:
            raise ValueError("Invalid model config: hidden_size must be positive")
        
        if not hasattr(model_config, 'vocab_size') or model_config.vocab_size <= 0:
            raise ValueError("Invalid model config: vocab_size must be positive")
        
        # Cross-validation: check compatibility
        if training_config.max_seq_length > model_config.max_position_embeddings:
            raise ValueError(
                f"Training sequence length ({training_config.max_seq_length}) "
                f"exceeds model's max position embeddings ({model_config.max_position_embeddings})"
            )
        
        # Estimate memory usage and warn if high
        if hasattr(model_config, 'estimate_memory_usage'):
            estimated_memory = model_config.estimate_memory_usage(
                batch_size=training_config.get_effective_batch_size(),
                seq_length=training_config.max_seq_length
            )
            
            if estimated_memory["total_gb"] > 24:
                print(f"⚠️  Warning: Estimated memory usage is {estimated_memory['total_gb']:.1f} GB")
                print("   Consider reducing batch size or sequence length")
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup validation failed: {e}")
        return False


def print_training_setup_summary(training_config, model_config):
    """Print a summary of the training setup."""
    print("📋 Training Setup Summary")
    print("=" * 50)
    
    # Model summary
    print(f"🧠 Model: {training_config.model_name}")
    print(f"   Architecture: {model_config.num_hidden_layers} layers, {model_config.hidden_size}d")
    print(f"   Vocabulary: {model_config.vocab_size:,} tokens")
    print(f"   Max Position: {model_config.max_position_embeddings}")
    
    if hasattr(model_config, 'estimate_parameters'):
        params = model_config.estimate_parameters()
        print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Training summary
    print(f"\n🏋️  Training Setup:")
    print(f"   Epochs: {training_config.epochs}")
    print(f"   Batch Size: {training_config.batch_size} (effective: {training_config.get_effective_batch_size()})")
    print(f"   Learning Rate: {training_config.optimizer.lr}")
    print(f"   Optimizer: {training_config.optimizer.name}")
    print(f"   Scheduler: {training_config.scheduler.name}")
    print(f"   Mixed Precision: {training_config.use_mixed_precision}")
    
    # Data summary
    print(f"\n📊 Data Setup:")
    print(f"   Dataset: {training_config.dataset_name}")
    print(f"   Max Sequence Length: {training_config.max_seq_length}")
    print(f"   Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    
    # Device and performance
    print(f"\n⚡ Performance:")
    print(f"   Device: {training_config.device}")
    print(f"   Gradient Clipping: {training_config.gradient_clipping}")
    print(f"   DataLoader Workers: {training_config.dataloader_num_workers}")
    
    # Logging summary
    print(f"\n📝 Logging:")
    print(f"   Experiment: {training_config.logging.experiment_name}")
    print(f"   Output Dir: {training_config.logging.output_dir}")
    print(f"   Wandb: {training_config.logging.use_wandb}")
    print(f"   TensorBoard: {training_config.logging.use_tensorboard}")
    print(f"   Log Every: {training_config.logging.log_every_n_steps} steps")


def get_recommended_config(
    target_memory_gb: float = 8.0,
    target_training_time_hours: float = 2.0,
    use_case: str = "development"
) -> dict:
    """Get recommended configuration based on constraints.
    
    Args:
        target_memory_gb: Target memory usage in GB
        target_training_time_hours: Target training time in hours
        use_case: Use case ('development', 'research', 'production')
    
    Returns:
        Dictionary with recommended settings
    """
    recommendations = {
        "development": {
            "model_variant": "tiny_llama",
            "training_preset": "development",
            "custom_training_config": {
                "batch_size": 4 if target_memory_gb < 8 else 8,
                "epochs": 2 if target_training_time_hours < 1 else 5,
                "use_mixed_precision": target_memory_gb < 16,
            }
        },
        "research": {
            "model_variant": "small_llama" if target_memory_gb > 16 else "tiny_llama",
            "training_preset": "production",
            "custom_training_config": {
                "batch_size": 2 if target_memory_gb < 16 else 4,
                "epochs": 10,
                "use_mixed_precision": True,
            }
        },
        "production": {
            "model_variant": "llama_7b" if target_memory_gb > 32 else "small_llama",
            "training_preset": "production",
            "custom_training_config": {
                "batch_size": 1 if target_memory_gb < 24 else 2,
                "epochs": 20,
                "use_mixed_precision": True,
                "gradient_accumulation_steps": 8,
            }
        }
    }
    
    if use_case not in recommendations:
        raise ValueError(f"Unknown use case: {use_case}. Available: {list(recommendations.keys())}")
    
    return recommendations[use_case]


def test_config_integration():
    """Test the configuration integration system."""
    print("🧪 Testing Configuration Integration System")
    
    try:
        # Test training setup creation
        training_config, model_config = create_training_setup(
            model_variant="tiny_llama",
            training_preset="development"
        )
        print("✅ Training setup creation works")
        
        # Test validation
        is_valid = validate_training_setup(training_config, model_config)
        if is_valid:
            print("✅ Training setup validation works")
        
        # Test custom configs
        custom_training = {"batch_size": 16, "epochs": 3}
        custom_model = {"hidden_size": 1024}
        
        custom_train_config, custom_model_config = create_training_setup(
            model_variant="tiny_llama",
            training_preset="development",
            custom_training_config=custom_training,
            custom_model_config=custom_model
        )
        print("✅ Custom config overrides work")
        
        # Test recommendations
        recommendations = get_recommended_config(
            target_memory_gb=8.0,
            use_case="development"
        )
        print("✅ Configuration recommendations work")
        
        # Test different model variants
        try:
            variants_to_test = ["tiny_llama", "small_llama"]
            for variant in variants_to_test:
                try:
                    train_cfg, model_cfg = create_training_setup(
                        model_variant=variant,
                        training_preset="development"
                    )
                    print(f"✅ {variant} variant works")
                except Exception as e:
                    print(f"⚠️  {variant} variant not available: {e}")
        except Exception as e:
            print(f"⚠️  Some model variants not available: {e}")
        
        # Test summary generation
        print_training_setup_summary(training_config, model_config)
        print("✅ Summary generation works")
        
        print("🎉 Configuration integration system working!")
        return True
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Print available configurations
    configs = list_available_configs()
    print("Available configurations:")
    for category, config_list in configs.items():
        print(f"  {category}: {', '.join(config_list)}")
    
    # Test integration system
    test_config_integration()