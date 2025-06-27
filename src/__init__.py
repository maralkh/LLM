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

# Import from submodules
from . import training
from . import distributed  
from . import models
from . import parallelism

# High-level interface imports
from .training import (
    # Core orchestration
    LlamaTrainingOrchestrator,
    TrainingStrategy,
    TrainingConfig,
    
    # Orchestrator factories
    create_llama_7b_orchestrator,
    create_llama_13b_orchestrator, 
    create_llama_70b_orchestrator,
    create_code_llama_orchestrator,
    create_llama3_8b_orchestrator,
    create_llama3_70b_orchestrator,
    create_llama3_405b_orchestrator,
    create_tiny_llama3_orchestrator,
    
    # Base trainer
    Trainer,
    
    # Convenience functions
    create_training_config,
    quick_orchestrator
)

from .distributed import (
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
    
    # Trainers
    DistributedTrainer,
    LlamaDistributedTrainer,
    PipelineDistributedTrainer,
    AdaptiveDistributedTrainer,
    create_distributed_trainer,
    
    # Pipeline components
    MicrobatchScheduler,
    ScheduleStep,
    PipelineSchedule,
    GPipeSchedule,
    OneForwardOneBackwardSchedule,
    ChimeraSchedule,
    InterleavedSchedule,
    MicrobatchMetrics,
    BatchSplitter
)

from models import (
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

from parallelism import (
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
    rotate_half,

)