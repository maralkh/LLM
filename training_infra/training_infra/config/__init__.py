# training_infra/config/__init__.py
"""
Configuration Management System

This module provides a comprehensive configuration system for training infrastructure.
All configurations are based on dataclasses with validation and serialization support.

## Basic Usage:
```python
from training_infra.config import BaseConfig, TrainingConfig

# Create basic config
config = BaseConfig()

# Create training config
train_config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    epochs=10
)

# Save/load from YAML
train_config.save_yaml("config.yaml")
loaded_config = TrainingConfig.from_yaml("config.yaml")
```

## Configuration Hierarchy:
- BaseConfig: Foundation for all configs
- TrainingConfig: Training-specific settings
- ModelConfig: Model architecture settings  
- DistributedConfig: Distributed training settings (Phase 8)
"""

from .base import (
    BaseConfig,
    ConfigValidationError,
    save_config,
    load_config,
    ExampleConfig,  # Add ExampleConfig for testing
)

try:
    from .training_configs import (
        TrainingConfig,
        OptimizerConfig,
        SchedulerConfig,
        LoggingConfig,
    )
    _TRAINING_CONFIGS_AVAILABLE = True
except ImportError:
    _TRAINING_CONFIGS_AVAILABLE = False

try:
    from .model_configs import (
        ModelConfig,
        LlamaConfig,
        TinyLlamaConfig,
    )
    _MODEL_CONFIGS_AVAILABLE = True
except ImportError:
    _MODEL_CONFIGS_AVAILABLE = False

try:
    from .validation import (
        validate_config,
        ConfigValidator,
        ValidationRule,
    )
    _VALIDATION_AVAILABLE = True
except ImportError:
    _VALIDATION_AVAILABLE = False

# Core exports - always available
__all__ = [
    # Base configuration
    "BaseConfig",
    "ConfigValidationError",
    "save_config", 
    "load_config",
    "ExampleConfig",  # Add for testing
]

# Add conditional exports
if _TRAINING_CONFIGS_AVAILABLE:
    __all__.extend([
        "TrainingConfig",
        "OptimizerConfig", 
        "SchedulerConfig",
        "LoggingConfig",
    ])

if _MODEL_CONFIGS_AVAILABLE:
    __all__.extend([
        "ModelConfig",
        "LlamaConfig",
        "TinyLlamaConfig",
    ])

if _VALIDATION_AVAILABLE:
    __all__.extend([
        "validate_config",
        "ConfigValidator",
        "ValidationRule",
    ])

def list_available_configs():
    """List all available configuration classes."""
    available = ["BaseConfig"]
    
    if _TRAINING_CONFIGS_AVAILABLE:
        available.extend(["TrainingConfig", "OptimizerConfig", "SchedulerConfig"])
    
    if _MODEL_CONFIGS_AVAILABLE:
        available.extend(["ModelConfig", "LlamaConfig", "TinyLlamaConfig"])
    
    print("📋 Available Configuration Classes:")
    for config in available:
        print(f"  ✅ {config}")
    
    if not _TRAINING_CONFIGS_AVAILABLE:
        print(f"  ⏳ TrainingConfig (Phase 3)")
    if not _MODEL_CONFIGS_AVAILABLE:
        print(f"  ⏳ ModelConfig (Phase 2)")
    if not _VALIDATION_AVAILABLE:
        print(f"  ⏳ Validation (Phase 1.2)")
    
    return available