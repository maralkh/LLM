# training_infra/__init__.py
"""
🦙 LLaMA Distributed Training Infrastructure

A comprehensive, production-ready framework for training LLaMA models with
advanced distributed parallelism and modern optimization techniques.

## Key Features:
- 🚀 Full 3D Parallelism (Tensor + Pipeline + Data)
- 🧠 Advanced Training Strategies (Standard, MoE, LoRA, Hybrid)  
- ⚡ Memory Optimizations (Checkpointing, ZeRO, Flash Attention)
- 🔄 Pipeline Parallelism with Microbatching
- 📊 Automatic Configuration and Resource Management
- 🎯 High-level Orchestrator for Easy Training
- 💾 Memory Estimation and Performance Benchmarking
- 🖥️ Production CLI Interface
- 🛠️ Tiny Models for Development

## Quick Start:
```python
from training_infra import quick_start

# Ultra-fast development with Tiny LLaMA 3
orchestrator = quick_start("tiny_llama3_150m", "standard", 1)

# Production training with LLaMA 3 8B
orchestrator = quick_start("llama3_8b", "standard", 4)

# Launch training
orchestrator.launch_training(train_dataloader, val_dataloader)
```

## CLI Usage:
```bash
# Development workflow
python -m training_infra.training.cli train --model tiny_llama3_150m --data dev_data.jsonl

# Production training  
python -m training_infra.training.cli train --model llama3_8b --gpus 4 --data data.jsonl

# Interactive mode
python -m training_infra.training.cli interactive
```
"""

__version__ = "1.0.0"
__author__ = "LLaMA Training Infrastructure Team"
__description__ = "Advanced distributed training framework for LLaMA models"

# Import submodules first
from . import training
from . import distributed  
from . import models
from . import parallelism

# High-level interface imports
from .training.orchestrator import (
    # Core orchestration
    LlamaTrainingOrchestrator,
    TrainingStrategy,
    
    # Orchestrator factories
    create_llama_7b_orchestrator,
    create_llama_13b_orchestrator, 
    create_llama_70b_orchestrator,
    create_code_llama_orchestrator,
    create_llama3_8b_orchestrator,
    create_llama3_70b_orchestrator,
    create_llama3_405b_orchestrator,
    create_tiny_llama3_orchestrator,
)

from .training.config import (
    TrainingConfig,
    # Convenience functions
    create_training_config,
)

from .distributed.config import (
    # Configuration classes
    DistributedConfig,
    TensorParallelConfig,
    PipelineParallelConfig, 
    DataParallelConfig,
    ZeROConfig,
    MixedPrecisionConfig,
    CommunicationConfig,
    MemoryOptimizationConfig,
    
    # Factories and auto-configuration
    ConfigurationFactory,
    AutoConfigurator,
)

from .distributed.trainer import (
    # Trainers
    DistributedTrainer,
    LlamaDistributedTrainer,
    PipelineDistributedTrainer,
    AdaptiveDistributedTrainer,
    create_distributed_trainer,
)

from .distributed.microbatch_scheduler import (
    # Pipeline components
    MicrobatchScheduler,
    ScheduleStep,
    PipelineSchedule,
    GPipeSchedule,
    OneForwardOneBackwardSchedule,
    ChimeraSchedule,
    InterleavedSchedule,
    MicrobatchMetrics,
    BatchSplitter,
)

from .models.llama import (
    # LLaMA model components
    LlamaConfig,
    RMSNorm,
    RotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    
    # Utilities
    optimize_model_for_training,
    estimate_model_memory,
    apply_rotary_pos_emb,
    rotate_half,
)

from .parallelism.tensor_parallelism import (
    # Tensor parallelism components
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    ParallelEmbedding,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

from .parallelism.pipeline_parallelism import (
    # Pipeline parallelism components
    PipelineParallel,
    PipelineModule,
    PipelineStage,
    LayerSpec,
    TiedLayerSpec,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)

# Model creation functions (distributed across modules)
try:
    from .models.llama import (
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
    )
except ImportError:
    # These functions might not be implemented yet
    pass

# Convenience function for quick setup
def quick_start(model_name: str, strategy: str = "standard", num_gpus: int = 1):
    """
    Quick start function for LLaMA training setup.
    
    Args:
        model_name: Model variant (e.g., "tiny_llama3_150m", "llama3_8b")
        strategy: Training strategy ("standard", "moe", "lora", "hybrid")
        num_gpus: Number of GPUs to use
        
    Returns:
        LlamaTrainingOrchestrator: Configured orchestrator ready for training
    """
    # Map model names to orchestrator factory functions
    orchestrator_factories = {
        "tiny_llama3_50m": lambda: create_tiny_llama3_orchestrator("50m", num_gpus, strategy),
        "tiny_llama3_150m": lambda: create_tiny_llama3_orchestrator("150m", num_gpus, strategy),
        "llama_7b": lambda: create_llama_7b_orchestrator(num_gpus, strategy),
        "llama_13b": lambda: create_llama_13b_orchestrator(num_gpus, strategy),
        "llama_70b": lambda: create_llama_70b_orchestrator(num_gpus, strategy),
        "code_llama": lambda: create_code_llama_orchestrator("7b", num_gpus, strategy),
        "llama3_8b": lambda: create_llama3_8b_orchestrator(num_gpus, strategy),
        "llama3_70b": lambda: create_llama3_70b_orchestrator(num_gpus, strategy),
        "llama3_405b": lambda: create_llama3_405b_orchestrator(num_gpus, strategy),
    }
    
    if model_name not in orchestrator_factories:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(orchestrator_factories.keys())}")
    
    return orchestrator_factories[model_name]()

# Add convenient alias for quick orchestrator creation
quick_orchestrator = quick_start

# Export all important symbols
__all__ = [
    # Version info
    "__version__", "__author__", "__description__",
    
    # Submodules
    "training", "distributed", "models", "parallelism",
    
    # Core orchestration
    "LlamaTrainingOrchestrator", "TrainingStrategy", "TrainingConfig",
    
    # Configuration
    "DistributedConfig", "TensorParallelConfig", "PipelineParallelConfig",
    "DataParallelConfig", "ZeROConfig", "MixedPrecisionConfig",
    "CommunicationConfig", "MemoryOptimizationConfig",
    "ConfigurationFactory", "AutoConfigurator",
    
    # Trainers
    "DistributedTrainer", "LlamaDistributedTrainer", "PipelineDistributedTrainer",
    "AdaptiveDistributedTrainer", "create_distributed_trainer",
    
    # Pipeline components
    "MicrobatchScheduler", "ScheduleStep", "PipelineSchedule",
    "GPipeSchedule", "OneForwardOneBackwardSchedule", "ChimeraSchedule",
    "InterleavedSchedule", "MicrobatchMetrics", "BatchSplitter",
    
    # Model components
    "LlamaConfig", "RMSNorm", "RotaryEmbedding", "LlamaAttention",
    "LlamaMLP", "LlamaDecoderLayer", "LlamaModel", "LlamaForCausalLM",
    
    # Parallelism components
    "ColumnParallelLinear", "RowParallelLinear", "VocabParallelEmbedding",
    "ParallelEmbedding", "PipelineParallel", "PipelineModule", "PipelineStage",
    
    # Orchestrator factories
    "create_llama_7b_orchestrator", "create_llama_13b_orchestrator",
    "create_llama_70b_orchestrator", "create_code_llama_orchestrator",
    "create_llama3_8b_orchestrator", "create_llama3_70b_orchestrator", 
    "create_llama3_405b_orchestrator", "create_tiny_llama3_orchestrator",
    
    # Convenience functions
    "quick_start", "quick_orchestrator", "create_training_config",
    
    # Utilities
    "optimize_model_for_training", "estimate_model_memory",
    "apply_rotary_pos_emb", "rotate_half",
]