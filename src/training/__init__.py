# training_infra/training/__init__.py
"""
Training orchestration and configuration components.

This module provides high-level training orchestration, configuration management,
and advanced training strategies for LLaMA models.
"""

from .config import TrainingConfig
from .trainer import Trainer
from .orchestrator import (
    LlamaTrainingOrchestrator,
    TrainingStrategy,
    create_llama_7b_orchestrator,
    create_llama_13b_orchestrator,
    create_llama_70b_orchestrator,
    create_code_llama_orchestrator,
    create_llama3_8b_orchestrator,
    create_llama3_70b_orchestrator,
    create_llama3_405b_orchestrator,
    create_tiny_llama3_orchestrator,
    get_llama_variant_config
)

try:
    from .advanced import (
        AdvancedLlamaTrainer,
        LlamaMoEModel,
        create_llama_moe_7b,
        AdvancedInferenceEngine,
        create_llama_training_config,
        create_moe_training_config,
        LlamaWithLoRA,
        HybridMoEModel,
        CurriculumLearningTrainer,
        GradientAccumulationTrainer
    )
    _ADVANCED_AVAILABLE = True
except ImportError:
    _ADVANCED_AVAILABLE = False

try:
    from .cli import (
        LlamaCLI,
        InteractiveCLI,
        ExtendedLlamaCLI,
        create_sample_data,
        validate_cli_setup,
        quick_test_cli
    )
    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False

try:
    from .benchmarking import PerformanceBenchmark
    _BENCHMARKING_AVAILABLE = True
except ImportError:
    _BENCHMARKING_AVAILABLE = False

# Core exports - always available
__all__ = [
    # Configuration
    "TrainingConfig",
    
    # Base trainer
    "Trainer",
    
    # Orchestration
    "LlamaTrainingOrchestrator",
    "TrainingStrategy",
    
    # Orchestrator factories
    "create_llama_7b_orchestrator",
    "create_llama_13b_orchestrator", 
    "create_llama_70b_orchestrator",
    "create_code_llama_orchestrator",
    "create_llama3_8b_orchestrator",
    "create_llama3_70b_orchestrator",
    "create_llama3_405b_orchestrator",
    "create_tiny_llama3_orchestrator",
    
    # Utilities
    "get_llama_variant_config",
]

# Conditional exports
if _ADVANCED_AVAILABLE:
    __all__.extend([
        "AdvancedLlamaTrainer",
        "LlamaMoEModel", 
        "create_llama_moe_7b",
        "AdvancedInferenceEngine",
        "create_llama_training_config",
        "create_moe_training_config",
        "LlamaWithLoRA",
        "HybridMoEModel",
        "CurriculumLearningTrainer",
        "GradientAccumulationTrainer"
    ])

if _CLI_AVAILABLE:
    __all__.extend([
        "LlamaCLI",
        "InteractiveCLI", 
        "ExtendedLlamaCLI",
        "create_sample_data",
        "validate_cli_setup",
        "quick_test_cli"
    ])

if _BENCHMARKING_AVAILABLE:
    __all__.append("PerformanceBenchmark")

# Module info
__version__ = "1.0.0"
__description__ = "Training orchestration and configuration for LLaMA models"

def print_training_info():
    """Print information about training module"""
    print("🎯 LLaMA Training Module")
    print("=" * 40)
    print("Core components:")
    print("  • TrainingConfig - Training configuration")
    print("  • Trainer - Base trainer class") 
    print("  • LlamaTrainingOrchestrator - High-level orchestration")
    print("  • TrainingStrategy - Strategy configuration")
    
    print(f"\nOrchestrator factories:")
    factories = [name for name in __all__ if name.startswith("create_")]
    for factory in factories:
        print(f"  • {factory}")
    
    print(f"\nOptional components:")
    print(f"  • Advanced training: {'✅' if _ADVANCED_AVAILABLE else '❌'}")
    print(f"  • CLI interface: {'✅' if _CLI_AVAILABLE else '❌'}")
    print(f"  • Benchmarking: {'✅' if _BENCHMARKING_AVAILABLE else '❌'}")

# Convenience functions
def create_training_config(
    model_name: str = "llama3_8b",
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    **kwargs
) -> TrainingConfig:
    """Create a basic training configuration"""
    
    return TrainingConfig(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=TrainingConfig.OptimizerConfig(
            name="adamw",
            lr=learning_rate,
            weight_decay=kwargs.get("weight_decay", 0.01)
        ),
        scheduler=TrainingConfig.SchedulerConfig(
            name="cosine",
            warmup_steps=kwargs.get("warmup_steps", 1000)
        ),
        use_amp=kwargs.get("use_amp", True),
        amp_dtype=kwargs.get("amp_dtype", "bfloat16"),
        **kwargs
    )

def quick_orchestrator(
    model_variant: str,
    num_gpus: int = 1,
    strategy: str = "standard"
) -> LlamaTrainingOrchestrator:
    """Quick orchestrator creation"""
    
    return LlamaTrainingOrchestrator(
        model_variant=model_variant,
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

# Add convenience functions to exports
__all__.extend([
    "create_training_config",
    "quick_orchestrator", 
    "print_training_info"
])