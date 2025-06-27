# training_infra/training/__init__.py
"""
Training orchestration and configuration components.

This module provides high-level training orchestration, configuration management,
and advanced training strategies for LLaMA models.
"""

import logging
from typing import Optional, Dict, Any

# Core configuration imports with error handling
try:
    from .config import TrainingConfig
    _CONFIG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Training config not available: {e}")
    _CONFIG_AVAILABLE = False
    TrainingConfig = None

# Orchestrator imports with error handling
try:
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
    _ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Training orchestrator not available: {e}")
    _ORCHESTRATOR_AVAILABLE = False
    
    # Fallback classes
    class LlamaTrainingOrchestrator:
        def __init__(self, *args, **kwargs):
            raise ImportError("Orchestrator not available")
    
    class TrainingStrategy:
        def __init__(self, *args, **kwargs):
            self.name = "standard"
            self.parameters = {}

# CLI imports with error handling
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
except ImportError as e:
    logging.warning(f"CLI not available: {e}")
    _CLI_AVAILABLE = False

# Core exports - always available
__all__ = []

# Add available components to exports
if _CONFIG_AVAILABLE:
    __all__.extend([
        "TrainingConfig",
    ])

if _ORCHESTRATOR_AVAILABLE:
    __all__.extend([
        "LlamaTrainingOrchestrator",
        "TrainingStrategy",
        "create_llama_7b_orchestrator",
        "create_llama_13b_orchestrator", 
        "create_llama_70b_orchestrator",
        "create_code_llama_orchestrator",
        "create_llama3_8b_orchestrator",
        "create_llama3_70b_orchestrator",
        "create_llama3_405b_orchestrator",
        "create_tiny_llama3_orchestrator",
        "get_llama_variant_config",
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

# Module info
__version__ = "2.0.0"
__description__ = "Training orchestration and configuration for LLaMA models"

def print_training_info():
    """Print information about training module"""
    print("🎯 LLaMA Training Module v2.0")
    print("=" * 45)
    
    print("Core components:")
    print(f"  • TrainingConfig: {'✅' if _CONFIG_AVAILABLE else '❌'}")
    print(f"  • LlamaTrainingOrchestrator: {'✅' if _ORCHESTRATOR_AVAILABLE else '❌'}")
    print(f"  • CLI Interface: {'✅' if _CLI_AVAILABLE else '❌'}")
    
    if _ORCHESTRATOR_AVAILABLE:
        print(f"\nOrchestrator factories:")
        factories = [name for name in __all__ if name.startswith("create_")]
        for factory in factories:
            print(f"  • {factory}")
    
    print(f"\nAvailability Summary:")
    available_count = sum([_CONFIG_AVAILABLE, _ORCHESTRATOR_AVAILABLE, _CLI_AVAILABLE])
    total_count = 3
    print(f"  {available_count}/{total_count} components available")
    
    if not _CONFIG_AVAILABLE:
        print("  ⚠️  TrainingConfig missing - basic training configuration unavailable")
    if not _ORCHESTRATOR_AVAILABLE:
        print("  ⚠️  Orchestrator missing - high-level training management unavailable")
    if not _CLI_AVAILABLE:
        print("  ⚠️  CLI missing - command line interface unavailable")

# Convenience functions with error handling
def create_training_config(
    model_name: str = "llama3_8b",
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    **kwargs
):
    """Create a basic training configuration"""
    
    if not _CONFIG_AVAILABLE:
        raise ImportError("TrainingConfig not available. Check training.config module.")
    
    return TrainingConfig(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_config={
            "name": "adamw",
            "lr": learning_rate,
            "weight_decay": kwargs.get("weight_decay", 0.01)
        },
        scheduler_config={
            "name": "cosine",
            "warmup_steps": kwargs.get("warmup_steps", 1000)
        },
        use_amp=kwargs.get("use_amp", True),
        amp_dtype=kwargs.get("amp_dtype", "bfloat16"),
        **kwargs
    )

def quick_orchestrator(
    model_variant: str,
    num_gpus: int = 1,
    strategy: str = "standard",
    **kwargs
):
    """Quick orchestrator creation with error handling"""
    
    if not _ORCHESTRATOR_AVAILABLE:
        raise ImportError("LlamaTrainingOrchestrator not available. Check training.orchestrator module.")
    
    return LlamaTrainingOrchestrator(
        model_variant=model_variant,
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True,
        **kwargs
    )

def setup_training_environment():
    """Setup training environment and validate dependencies"""
    
    issues = []
    recommendations = []
    
    # Check core dependencies
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
        else:
            issues.append("CUDA not available")
            recommendations.append("Install CUDA for GPU acceleration")
    except ImportError:
        issues.append("PyTorch not installed")
        return issues, recommendations
    
    # Check training components
    if not _CONFIG_AVAILABLE:
        issues.append("TrainingConfig not available")
        recommendations.append("Check training.config module implementation")
    
    if not _ORCHESTRATOR_AVAILABLE:
        issues.append("Training orchestrator not available")
        recommendations.append("Check training.orchestrator module and dependencies")
    
    if not _CLI_AVAILABLE:
        recommendations.append("CLI not available - install for command line interface")
    
    # Check distributed training availability
    try:
        from ..distributed import setup_distributed_training
        print("✅ Distributed training support available")
    except ImportError:
        issues.append("Distributed training not available")
        recommendations.append("Check distributed module implementation")
    
    # Check model implementations
    try:
        from ..models.llama import LlamaForCausalLM
        print("✅ LLaMA model implementations available")
    except ImportError:
        issues.append("LLaMA models not available")
        recommendations.append("Check models.llama module implementation")
    
    return issues, recommendations

def validate_training_setup() -> Dict[str, Any]:
    """Validate complete training setup"""
    
    validation = {
        "valid": True,
        "issues": [],
        "recommendations": [],
        "component_status": {
            "config": _CONFIG_AVAILABLE,
            "orchestrator": _ORCHESTRATOR_AVAILABLE,
            "cli": _CLI_AVAILABLE
        }
    }
    
    issues, recommendations = setup_training_environment()
    validation["issues"] = issues
    validation["recommendations"] = recommendations
    
    if issues:
        validation["valid"] = False
    
    # Check integration with distributed training
    try:
        from ..distributed import health_check
        dist_health = health_check()
        if not dist_health.get("overall_healthy", False):
            validation["issues"].append("Distributed training system not healthy")
            validation["recommendations"].append("Run distributed health check for details")
    except Exception as e:
        validation["issues"].append(f"Could not check distributed system: {e}")
    
    return validation

# Add convenience functions to exports
__all__.extend([
    "create_training_config",
    "quick_orchestrator", 
    "print_training_info",
    "setup_training_environment",
    "validate_training_setup"
])

# Integration utilities
def create_integrated_training_setup(
    model_variant: str = "llama3_8b",
    strategy: str = "auto",
    num_gpus: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create integrated setup with both training and distributed components"""
    
    setup_result = {
        "success": False,
        "orchestrator": None,
        "distributed_config": None,
        "training_config": None,
        "errors": []
    }
    
    try:
        # Validate setup first
        validation = validate_training_setup()
        if not validation["valid"]:
            setup_result["errors"] = validation["issues"]
            return setup_result
        
        # Auto-detect GPU count
        if num_gpus is None:
            try:
                import torch
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            except:
                num_gpus = 1
        
        # Create distributed config using the unified system
        try:
            from ..distributed import AutoConfigurator
            distributed_config = AutoConfigurator.auto_configure(model_variant, num_gpus)
            setup_result["distributed_config"] = distributed_config
        except Exception as e:
            setup_result["errors"].append(f"Failed to create distributed config: {e}")
            return setup_result
        
        # Create training config
        if _CONFIG_AVAILABLE:
            training_config = create_training_config(
                model_name=model_variant,
                **kwargs
            )
            setup_result["training_config"] = training_config
        
        # Create orchestrator
        if _ORCHESTRATOR_AVAILABLE:
            orchestrator = quick_orchestrator(
                model_variant=model_variant,
                strategy=strategy,
                num_gpus=num_gpus,
                **kwargs
            )
            setup_result["orchestrator"] = orchestrator
        
        setup_result["success"] = True
        
        if setup_result["orchestrator"]:
            print("✅ Integrated training setup completed successfully!")
            setup_result["orchestrator"].print_configuration()
        
    except Exception as e:
        setup_result["errors"].append(f"Setup failed: {str(e)}")
    
    return setup_result

__all__.append("create_integrated_training_setup")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Module health check on import
def _module_health_check():
    """Quick health check on module import"""
    issues = []
    
    if not _CONFIG_AVAILABLE:
        issues.append("TrainingConfig")
    if not _ORCHESTRATOR_AVAILABLE:
        issues.append("Orchestrator")
    
    if issues:
        logging.warning(f"Training module issues: {', '.join(issues)} not available")
    else:
        logging.info("Training module loaded successfully")

# Run health check
_module_health_check()