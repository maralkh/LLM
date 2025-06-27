# training_infra/__init__.py
"""
Unified Training Infrastructure for LLaMA models
Merges parallelism and distributed components
"""

# Import from distributed_init (our fixed initialization)
try:
    from .distributed_init import (
        setup_distributed_training,
        cleanup_distributed,
        get_world_size,
        get_rank,
        get_local_rank,
        get_tensor_parallel_rank,
        get_pipeline_parallel_rank,
        get_data_parallel_rank,
        is_main_process,
        barrier,
        print_distributed_info,
        get_distributed_state,
        ParallelismConfig,
        DistributedContext,
        health_check
    )
    _DISTRIBUTED_INIT_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_INIT_AVAILABLE = False

# Import tensor parallelism
try:
    from .tensor_parallel import (
        TensorParallelLinear,
        TensorParallelEmbedding,
        TensorParallelConfig,
        TensorParallelMultiHeadAttention,
        ColumnParallelLinear,
        RowParallelLinear,
        convert_model_to_tensor_parallel,
        all_gather_tensor_parallel,
        reduce_scatter_tensor_parallel,
        tensor_parallel_all_reduce
    )
    _TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    _TENSOR_PARALLEL_AVAILABLE = False

# Import pipeline parallelism
try:
    from .pipeline_parallel import (
        PipelineConfig,
        PipelineStage,
        PipelineParallelModel,
        AsyncPipelineParallelModel,
        PipelineParallelTrainer,
        MicroBatchScheduler,
        create_pipeline_stages,
        setup_pipeline_parallel_model
    )
    _PIPELINE_PARALLEL_AVAILABLE = True
except ImportError:
    _PIPELINE_PARALLEL_AVAILABLE = False

# Import distributed config
try:
    from .distributed_config import (
        DistributedConfig,
        TensorParallelConfig as DistTensorParallelConfig,
        PipelineParallelConfig,
        DataParallelConfig,
        ZeROConfig,
        MixedPrecisionConfig,
        CommunicationConfig,
        MemoryOptimizationConfig,
        ConfigurationFactory,
        AutoConfigurator
    )
    _DISTRIBUTED_CONFIG_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_CONFIG_AVAILABLE = False

# Import microbatch scheduler
try:
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
    _MICROBATCH_SCHEDULER_AVAILABLE = True
except ImportError:
    _MICROBATCH_SCHEDULER_AVAILABLE = False

# Import distributed trainers
try:
    from .distributed_trainer import (
        DistributedTrainer,
        LlamaDistributedTrainer,
        PipelineDistributedTrainer,
        AdaptiveDistributedTrainer,
        create_distributed_trainer,
        DistributedPerformanceTracker,
        MemoryOptimizer,
        BatchSizeAdapter
    )
    _DISTRIBUTED_TRAINER_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_TRAINER_AVAILABLE = False

# Module info
__version__ = "1.0.0"
__description__ = "Unified training infrastructure for LLaMA models"

# Core exports
__all__ = []

# Add available components to exports
if _DISTRIBUTED_INIT_AVAILABLE:
    __all__.extend([
        "setup_distributed_training", "cleanup_distributed",
        "get_world_size", "get_rank", "get_local_rank",
        "get_tensor_parallel_rank", "get_pipeline_parallel_rank", "get_data_parallel_rank",
        "is_main_process", "barrier", "print_distributed_info", "get_distributed_state",
        "ParallelismConfig", "DistributedContext", "health_check"
    ])

if _TENSOR_PARALLEL_AVAILABLE:
    __all__.extend([
        "TensorParallelLinear", "TensorParallelEmbedding", "TensorParallelConfig",
        "TensorParallelMultiHeadAttention", "ColumnParallelLinear", "RowParallelLinear",
        "convert_model_to_tensor_parallel", "all_gather_tensor_parallel",
        "reduce_scatter_tensor_parallel", "tensor_parallel_all_reduce"
    ])

if _PIPELINE_PARALLEL_AVAILABLE:
    __all__.extend([
        "PipelineConfig", "PipelineStage", "PipelineParallelModel",
        "AsyncPipelineParallelModel", "PipelineParallelTrainer", "MicroBatchScheduler",
        "create_pipeline_stages", "setup_pipeline_parallel_model"
    ])

if _DISTRIBUTED_CONFIG_AVAILABLE:
    __all__.extend([
        "DistributedConfig", "DistTensorParallelConfig", "PipelineParallelConfig",
        "DataParallelConfig", "ZeROConfig", "MixedPrecisionConfig",
        "CommunicationConfig", "MemoryOptimizationConfig",
        "ConfigurationFactory", "AutoConfigurator"
    ])

if _MICROBATCH_SCHEDULER_AVAILABLE:
    __all__.extend([
        "MicrobatchScheduler", "ScheduleStep", "PipelineSchedule",
        "GPipeSchedule", "OneForwardOneBackwardSchedule", "ChimeraSchedule",
        "InterleavedSchedule", "MicrobatchMetrics", "BatchSplitter"
    ])

if _DISTRIBUTED_TRAINER_AVAILABLE:
    __all__.extend([
        "DistributedTrainer", "LlamaDistributedTrainer", "PipelineDistributedTrainer",
        "AdaptiveDistributedTrainer", "create_distributed_trainer",
        "DistributedPerformanceTracker", "MemoryOptimizer", "BatchSizeAdapter"
    ])

def print_module_info():
    """Print information about available components"""
    print("🚀 Training Infrastructure Module")
    print("=" * 40)
    print("Available components:")
    print(f"  • Distributed Init: {'✅' if _DISTRIBUTED_INIT_AVAILABLE else '❌'}")
    print(f"  • Tensor Parallel: {'✅' if _TENSOR_PARALLEL_AVAILABLE else '❌'}")
    print(f"  • Pipeline Parallel: {'✅' if _PIPELINE_PARALLEL_AVAILABLE else '❌'}")
    print(f"  • Distributed Config: {'✅' if _DISTRIBUTED_CONFIG_AVAILABLE else '❌'}")
    print(f"  • Microbatch Scheduler: {'✅' if _MICROBATCH_SCHEDULER_AVAILABLE else '❌'}")
    print(f"  • Distributed Trainer: {'✅' if _DISTRIBUTED_TRAINER_AVAILABLE else '❌'}")
    
    if _DISTRIBUTED_INIT_AVAILABLE:
        try:
            state = get_distributed_state()
            if state["is_initialized"]:
                print(f"\nDistributed Status: ✅ Initialized")
                print(f"World Size: {state['world_size']}")
            else:
                print(f"\nDistributed Status: ⚠️ Not initialized")
        except:
            pass

__all__.append("print_module_info")


# Quick setup function
def quick_setup(model_size="7B", strategy="auto", **kwargs):
    """Quick setup for distributed training"""
    
    if not _DISTRIBUTED_INIT_AVAILABLE:
        print("❌ Distributed initialization not available")
        return False
    
    if not _DISTRIBUTED_CONFIG_AVAILABLE:
        print("❌ Distributed configuration not available")
        return False
    
    try:
        # Auto-detect GPU count
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Create config
        if strategy == "auto":
            config = AutoConfigurator.auto_configure(model_size, num_gpus)
        else:
            config = getattr(ConfigurationFactory, f'create_{strategy}_config')(num_gpus)
        
        # Setup distributed
        success = setup_distributed_training(
            tensor_parallel_size=config.tensor_parallel.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel.pipeline_parallel_size,
            data_parallel_size=config.data_parallel.data_parallel_size,
            **kwargs
        )
        
        if success:
            print("✅ Quick setup completed!")
            print_distributed_info()
            return True
        else:
            print("❌ Quick setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick setup error: {e}")
        return False

__all__.append("quick_setup")