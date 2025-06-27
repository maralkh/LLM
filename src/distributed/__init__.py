# training_infra/distributed/__init__.py
"""
Distributed training components for LLaMA models.

This module provides distributed training capabilities including tensor parallelism,
pipeline parallelism, data parallelism, and ZeRO optimizer support.
"""

from .config import (
    DistributedConfig,
    TensorParallelConfig,
    PipelineParallelConfig,
    DataParallelConfig,
    ZeROConfig,
    MixedPrecisionConfig,
    CommunicationConfig,
    MemoryOptimizationConfig,
    ConfigurationFactory,
    AutoConfigurator
)

from .trainer import (
    DistributedTrainer,
    LlamaDistributedTrainer,
    PipelineDistributedTrainer,
    AdaptiveDistributedTrainer,
    DistributedPerformanceTracker,
    MemoryOptimizer,
    BatchSizeAdapter,
    create_distributed_trainer
)

from .microbatch_scheduler import (
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

# Core exports
__all__ = [
    # Configuration classes
    "DistributedConfig",
    "TensorParallelConfig", 
    "PipelineParallelConfig",
    "DataParallelConfig",
    "ZeROConfig",
    "MixedPrecisionConfig",
    "CommunicationConfig",
    "MemoryOptimizationConfig",
    
    # Configuration factories
    "ConfigurationFactory",
    "AutoConfigurator",
    
    # Trainer classes
    "DistributedTrainer",
    "LlamaDistributedTrainer",
    "PipelineDistributedTrainer", 
    "AdaptiveDistributedTrainer",
    "DistributedPerformanceTracker",
    "MemoryOptimizer",
    "BatchSizeAdapter",
    "create_distributed_trainer",
    
    # Pipeline and microbatch components
    "MicrobatchScheduler",
    "ScheduleStep",
    "PipelineSchedule",
    "GPipeSchedule", 
    "OneForwardOneBackwardSchedule",
    "ChimeraSchedule",
    "InterleavedSchedule",
    "MicrobatchMetrics",
    "BatchSplitter",
]

# Module info
__version__ = "1.0.0" 
__description__ = "Distributed training components for LLaMA models"

def print_distributed_info():
    """Print information about distributed module"""
    print("🌐 LLaMA Distributed Training Module")
    print("=" * 45)
    
    print("Parallelism strategies:")
    print("  • Tensor Parallelism - Split model layers across GPUs")
    print("  • Pipeline Parallelism - Split model into stages")
    print("  • Data Parallelism - Replicate model across GPUs")
    print("  • ZeRO Optimizer - Memory-efficient optimizer")
    
    print("\nConfiguration classes:")
    configs = [name for name in __all__ if name.endswith("Config")]
    for config in configs:
        print(f"  • {config}")
    
    print(f"\nTrainer classes:")
    trainers = [name for name in __all__ if "Trainer" in name]
    for trainer in trainers:
        print(f"  • {trainer}")
    
    print(f"\nPipeline components:")
    pipeline = [name for name in __all__ if any(x in name for x in ["Schedule", "Microbatch", "Splitter"])]
    for component in pipeline:
        print(f"  • {component}")

def create_simple_distributed_config(
    strategy: str = "data_parallel",
    num_gpus: int = 1,
    **kwargs
) -> DistributedConfig:
    """Create a simple distributed configuration"""
    
    if strategy == "data_parallel":
        return ConfigurationFactory.create_data_parallel_config(num_gpus)
    elif strategy == "tensor_parallel":
        return ConfigurationFactory.create_tensor_parallel_config(num_gpus)
    elif strategy == "pipeline":
        microbatch_size = kwargs.get("microbatch_size", 2)
        schedule = kwargs.get("schedule", "1f1b")
        return ConfigurationFactory.create_pipeline_parallel_config(
            num_gpus, microbatch_size, schedule
        )
    elif strategy == "hybrid":
        tp_size = kwargs.get("tensor_parallel_size", 2)
        pp_size = kwargs.get("pipeline_parallel_size", 2)
        dp_size = kwargs.get("data_parallel_size", num_gpus // (tp_size * pp_size))
        return ConfigurationFactory.create_hybrid_parallel_config(
            tp_size, pp_size, dp_size
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def estimate_distributed_memory(
    model_size: str,
    batch_size: int = 8,
    sequence_length: int = 2048,
    config: DistributedConfig = None,
    **kwargs
) -> dict:
    """Estimate memory requirements for distributed training"""
    
    if config is None:
        config = AutoConfigurator.auto_configure(model_size, 1)
    
    return AutoConfigurator.estimate_memory_requirements(
        model_size=model_size,
        batch_size=batch_size,
        sequence_length=sequence_length,
        config=config
    )

def create_microbatch_scheduler(
    num_microbatches: int,
    microbatch_size: int,
    num_pipeline_stages: int,
    schedule_type: str = "1f1b",
    **kwargs
) -> MicrobatchScheduler:
    """Create a microbatch scheduler for pipeline parallelism"""
    
    return MicrobatchScheduler(
        num_microbatches=num_microbatches,
        microbatch_size=microbatch_size,
        num_pipeline_stages=num_pipeline_stages,
        current_stage=kwargs.get("current_stage", 0),
        schedule_type=schedule_type,
        virtual_stages=kwargs.get("virtual_stages", None),
        enable_profiling=kwargs.get("enable_profiling", True)
    )

# Add convenience functions to exports
__all__.extend([
    "create_simple_distributed_config",
    "estimate_distributed_memory",
    "create_microbatch_scheduler",
    "print_distributed_info"
])

# Distributed training utilities
class DistributedUtils:
    """Utility class for distributed training operations"""
    
    @staticmethod
    def get_world_size() -> int:
        """Get distributed world size"""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_world_size()
        except:
            pass
        return 1
    
    @staticmethod
    def get_rank() -> int:
        """Get distributed rank"""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank()
        except:
            pass
        return 0
    
    @staticmethod
    def is_main_process() -> bool:
        """Check if this is the main process"""
        return DistributedUtils.get_rank() == 0
    
    @staticmethod
    def print_on_main(message: str):
        """Print message only on main process"""
        if DistributedUtils.is_main_process():
            print(message)
    
    @staticmethod
    def get_device_info() -> dict:
        """Get device information"""
        info = {
            "world_size": DistributedUtils.get_world_size(),
            "rank": DistributedUtils.get_rank(),
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_memory_gb": []
        }
        
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["gpu_count"] = torch.cuda.device_count()
                for i in range(info["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    info["gpu_memory_gb"].append(memory_gb)
        except:
            pass
        
        return info

# Add utility class to exports
__all__.append("DistributedUtils")

# Configuration presets for common scenarios
PRESET_CONFIGS = {
    "single_gpu": lambda: ConfigurationFactory.create_single_gpu_config(),
    "dual_gpu_dp": lambda: ConfigurationFactory.create_data_parallel_config(2),
    "quad_gpu_dp": lambda: ConfigurationFactory.create_data_parallel_config(4),
    "dual_gpu_tp": lambda: ConfigurationFactory.create_tensor_parallel_config(2),
    "quad_gpu_tp": lambda: ConfigurationFactory.create_tensor_parallel_config(4),
    "octo_gpu_hybrid": lambda: ConfigurationFactory.create_hybrid_parallel_config(2, 2, 2),
    "tiny_model": lambda: ConfigurationFactory.create_tiny_llama3_config(1),
    "llama3_8b_optimal": lambda: ConfigurationFactory.create_llama3_8b_config(4),
    "llama3_70b_optimal": lambda: ConfigurationFactory.create_llama3_70b_config(8),
    "llama3_405b_optimal": lambda: ConfigurationFactory.create_llama3_405b_config(64),
}

def get_preset_config(preset_name: str) -> DistributedConfig:
    """Get a preset distributed configuration"""
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return PRESET_CONFIGS[preset_name]()

def list_preset_configs():
    """List available preset configurations"""
    print("📋 Available Distributed Configuration Presets:")
    print("=" * 50)
    
    descriptions = {
        "single_gpu": "Single GPU training",
        "dual_gpu_dp": "2 GPU data parallel", 
        "quad_gpu_dp": "4 GPU data parallel",
        "dual_gpu_tp": "2 GPU tensor parallel",
        "quad_gpu_tp": "4 GPU tensor parallel", 
        "octo_gpu_hybrid": "8 GPU hybrid (2x2x2)",
        "tiny_model": "Tiny model configuration",
        "llama3_8b_optimal": "Optimal for LLaMA 3 8B",
        "llama3_70b_optimal": "Optimal for LLaMA 3 70B",
        "llama3_405b_optimal": "Optimal for LLaMA 3 405B",
    }
    
    for preset, desc in descriptions.items():
        print(f"  • {preset}: {desc}")

# Add preset functions to exports
__all__.extend([
    "get_preset_config",
    "list_preset_configs", 
    "PRESET_CONFIGS"
])