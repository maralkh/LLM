# training_infra/__init__.py
"""
🦙 Training Infrastructure

A comprehensive, production-ready framework for training LLaMA models with
advanced distributed parallelism and modern optimization techniques.

## Quick Start:
```python
from training_infra import quick_start

# Ultra-fast development with Tiny LLaMA 3
orchestrator = quick_start("tiny_llama3_150m", "standard", 1)

# Production training with LLaMA 3 8B  
orchestrator = quick_start("llama3_8b", "standard", 4)
```

## CLI Usage:
```bash
# Development workflow
python -m training_infra.cli train --model tiny_llama3_150m --data dev_data.jsonl

# Production training
python -m training_infra.cli train --model llama3_8b --gpus 4 --data data.jsonl
```
"""

__version__ = "0.1.0"
__author__ = "Training Infrastructure Team"
__description__ = "Advanced distributed training framework for LLaMA models"

# Basic imports - will expand as we build
from typing import Optional, Union
import warnings

def quick_start(
    model_name: str,
    strategy: str = "standard", 
    num_gpus: int = 1,
    **kwargs
) -> 'LlamaTrainingOrchestrator':
    """
    Quick start function for easy training setup.
    
    Args:
        model_name: Model to use ("tiny_llama3_150m", "llama3_8b", etc.)
        strategy: Training strategy ("standard", "lora", "moe", etc.)
        num_gpus: Number of GPUs to use
        **kwargs: Additional configuration options
    
    Returns:
        LlamaTrainingOrchestrator: Ready-to-use training orchestrator
    
    Example:
        >>> orchestrator = quick_start("tiny_llama3_150m", "standard", 1)
        >>> # orchestrator.launch_training(train_dataloader, val_dataloader)
    """
    try:
        from .training import LlamaTrainingOrchestrator
        
        orchestrator = LlamaTrainingOrchestrator(
            model_name=model_name,
            strategy=strategy,
            num_gpus=num_gpus,
            **kwargs
        )
        return orchestrator
        
    except ImportError as e:
        warnings.warn(
            f"Training components not yet available: {e}\n"
            f"This will be implemented in Phase 4. For now, this is a placeholder.",
            UserWarning
        )
        return None

def load_model(checkpoint_path: str, **kwargs):
    """
    Load a model from checkpoint for fine-tuning.
    
    Args:
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional loading options
    
    Returns:
        Model instance loaded from checkpoint
    """
    try:
        from .models.utils import loading
        return loading.load_model_from_checkpoint(checkpoint_path, **kwargs)
    except ImportError as e:
        warnings.warn(
            f"Model loading not yet available: {e}\n"
            f"This will be implemented in Phase 2-3.",
            UserWarning  
        )
        return None

# Version check function
def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "accelerate",
        "peft",
        "wandb"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"💡 Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All dependencies are installed!")
        return True

# Basic info function
def info():
    """Print basic information about the training infrastructure."""
    print(f"""
🦙 Training Infrastructure v{__version__}
{__description__}

📦 Available Components (will expand as we build):
  - Configuration System (Phase 1)
  - Model Implementations (Phase 2) 
  - Training Pipeline (Phase 3)
  - CLI Interface (Phase 4)
  - Advanced Features (Phase 5+)

🚀 Quick Start:
  from training_infra import quick_start
  orchestrator = quick_start("tiny_llama3_150m", "standard", 1)

💡 Check dependencies:
  from training_infra import check_dependencies
  check_dependencies()
""")

# Make key functions available at package level
__all__ = [
    "quick_start",
    "load_model", 
    "check_dependencies",
    "info",
    "__version__"
]