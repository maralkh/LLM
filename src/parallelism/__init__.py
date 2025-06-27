# training_infra/parallelism/__init__.py
"""
Low-level parallelism primitives for distributed LLaMA training.

This module provides the foundational parallelism components including
tensor parallelism, pipeline parallelism, and communication utilities.
"""

import logging
from typing import Optional, Dict, Any

# First, try to import the centralized distributed initialization
try:
    from .distributed_init import (
        setup_distributed_training,
        initialize_distributed,
        cleanup_distributed,
        ParallelismConfig,
        DistributedContext,
        get_world_size,
        get_rank,
        get_local_rank,
        get_tensor_parallel_rank,
        get_pipeline_parallel_rank,
        get_data_parallel_rank,
        get_expert_parallel_rank,
        get_tensor_parallel_group,
        get_pipeline_parallel_group,
        get_data_parallel_group,
        get_expert_parallel_group,
        is_pipeline_first_stage,
        is_pipeline_last_stage,
        is_main_process,
        barrier,
        print_distributed_info,
        get_distributed_state
    )
    _DISTRIBUTED_INIT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Distributed initialization not available: {e}")
    _DISTRIBUTED_INIT_AVAILABLE = False

# Import tensor parallelism components
try:
    from .tensor_parallel import (
        TensorParallelLinear,
        TensorParallelEmbedding,
        TensorParallelConfig,
        TensorParallelAttention,
        ColumnParallelLinear,
        RowParallelLinear,
        convert_to_tensor_parallel,
        all_gather_tensor_parallel,
        reduce_scatter_tensor_parallel,
        tensor_parallel_all_reduce
    )
    _TENSOR_PARALLEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Tensor parallelism not available: {e}")
    _TENSOR_PARALLEL_AVAILABLE = False

# Import pipeline parallelism components
try:
    from .pipeline_parallel import (
        PipelineConfig,
        PipelineStage,
        PipelineParallelModel,
        AsyncPipelineParallelModel,
        PipelineParallelTrainer,
        MicroBatchScheduler,
        create_pipeline_stages,
        setup_pipeline_parallel_model,
        all_reduce_pipeline_parallel_gradients,
        checkpoint_pipeline_stage
    )
    _PIPELINE_PARALLEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Pipeline parallelism not available: {e}")
    _PIPELINE_PARALLEL_AVAILABLE = False

# Import memory optimization components
try:
    from .memory_optimization import (
        ActivationCheckpointing,
        GradientCompression,
        MemoryProfiler,
        optimize_memory_usage,
        gradient_checkpointing_enable,
        gradient_checkpointing_disable
    )
    _MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Memory optimization not available: {e}")
    _MEMORY_OPTIMIZATION_AVAILABLE = False

# Import flash attention components
try:
    from .flash_attention import (
        FlashAttention,
        FlashMHA,
        enable_flash_attention,
        disable_flash_attention,
        is_flash_attention_available
    )
    _FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Flash attention not available: {e}")
    _FLASH_ATTENTION_AVAILABLE = False

# Core exports with centralized initialization
__all__ = []

# Distributed initialization exports (highest priority)
if _DISTRIBUTED_INIT_AVAILABLE:
    __all__.extend([
        "setup_distributed_training",
        "initialize_distributed", 
        "cleanup_distributed",
        "ParallelismConfig",
        "DistributedContext",
        "get_world_size",
        "get_rank",
        "get_local_rank",
        "get_tensor_parallel_rank",
        "get_pipeline_parallel_rank", 
        "get_data_parallel_rank",
        "get_expert_parallel_rank",
        "barrier",
        "print_distributed_info",
        "get_distributed_state"
    ])

# Tensor parallelism exports
if _TENSOR_PARALLEL_AVAILABLE:
    __all__.extend([
        "TensorParallelLinear",
        "TensorParallelEmbedding",
        "TensorParallelConfig", 
        "TensorParallelAttention",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "convert_to_tensor_parallel",
        "all_gather_tensor_parallel",
        "reduce_scatter_tensor_parallel",
        "tensor_parallel_all_reduce"
    ])

# Pipeline parallelism exports
if _PIPELINE_PARALLEL_AVAILABLE:
    __all__.extend([
        "PipelineConfig",
        "PipelineStage",
        "PipelineParallelModel",
        "AsyncPipelineParallelModel", 
        "PipelineParallelTrainer",
        "MicroBatchScheduler",
        "create_pipeline_stages",
        "setup_pipeline_parallel_model",
        "all_reduce_pipeline_parallel_gradients",
        "checkpoint_pipeline_stage"
    ])

# Memory optimization exports
if _MEMORY_OPTIMIZATION_AVAILABLE:
    __all__.extend([
        "ActivationCheckpointing",
        "GradientCompression",
        "MemoryProfiler", 
        "optimize_memory_usage",
        "gradient_checkpointing_enable",
        "gradient_checkpointing_disable"
    ])

# Flash attention exports
if _FLASH_ATTENTION_AVAILABLE:
    __all__.extend([
        "FlashAttention",
        "FlashMHA",
        "enable_flash_attention",
        "disable_flash_attention",
        "is_flash_attention_available"
    ])

# Module info
__version__ = "2.0.0"
__description__ = "Centralized parallelism primitives for distributed LLaMA training"

def print_parallelism_info():
    """Print information about parallelism module with improved structure"""
    print("⚡ LLaMA Parallelism Module v2.0")
    print("=" * 50)
    
    print("Available components:")
    print(f"  • Distributed Init: {'✅' if _DISTRIBUTED_INIT_AVAILABLE else '❌'}")
    print(f"  • Tensor Parallelism: {'✅' if _TENSOR_PARALLEL_AVAILABLE else '❌'}")
    print(f"  • Pipeline Parallelism: {'✅' if _PIPELINE_PARALLEL_AVAILABLE else '❌'}")
    print(f"  • Memory Optimization: {'✅' if _MEMORY_OPTIMIZATION_AVAILABLE else '❌'}")
    print(f"  • Flash Attention: {'✅' if _FLASH_ATTENTION_AVAILABLE else '❌'}")
    
    # Show distributed state if available
    if _DISTRIBUTED_INIT_AVAILABLE:
        try:
            state = get_distributed_state()
            if state["is_initialized"]:
                print("\nCurrent distributed state:")
                print(f"  • World Size: {state['world_size']}")
                print(f"  • Current Rank: {state['rank']}")
                print(f"  • TP Size: {state['tensor_parallel']['size']}")
                print(f"  • PP Size: {state['pipeline_parallel']['size']}")
                print(f"  • DP Size: {state['data_parallel']['size']}")
            else:
                print("\n⚠️  Distributed not initialized")
        except Exception:
            print("\n⚠️  Could not get distributed state")
    
    if not any([_DISTRIBUTED_INIT_AVAILABLE, _TENSOR_PARALLEL_AVAILABLE, 
                _PIPELINE_PARALLEL_AVAILABLE, _MEMORY_OPTIMIZATION_AVAILABLE]):
        print("\n⚠️  No parallelism components available")
        print("This may indicate missing dependencies or incomplete installation")

def get_parallelism_capabilities() -> Dict[str, bool]:
    """Get available parallelism capabilities"""
    return {
        "distributed_init": _DISTRIBUTED_INIT_AVAILABLE,
        "tensor_parallel": _TENSOR_PARALLEL_AVAILABLE,
        "pipeline_parallel": _PIPELINE_PARALLEL_AVAILABLE,
        "memory_optimization": _MEMORY_OPTIMIZATION_AVAILABLE,
        "flash_attention": _FLASH_ATTENTION_AVAILABLE
    }

def validate_parallelism_setup() -> Dict[str, Any]:
    """Enhanced parallelism setup validation"""
    
    validation = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    # Check PyTorch distributed
    try:
        import torch.distributed as dist
        if dist.is_available():
            validation["torch_distributed"] = True
        else:
            validation["warnings"].append("PyTorch distributed not available")
            validation["torch_distributed"] = False
    except ImportError:
        validation["errors"].append("PyTorch not available")
        validation["valid"] = False
        return validation
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            validation["cuda_available"] = True
            validation["gpu_count"] = torch.cuda.device_count()
            validation["cuda_version"] = torch.version.cuda
        else:
            validation["warnings"].append("CUDA not available - limited parallelism")
            validation["cuda_available"] = False
            validation["gpu_count"] = 0
    except Exception:
        validation["cuda_available"] = False
        validation["gpu_count"] = 0
    
    # Check parallelism components
    capabilities = get_parallelism_capabilities()
    missing_components = [k for k, v in capabilities.items() if not v]
    
    if missing_components:
        validation["warnings"].extend([
            f"Missing parallelism component: {comp}" for comp in missing_components
        ])
    
    # Check if distributed is initialized
    if _DISTRIBUTED_INIT_AVAILABLE:
        try:
            state = get_distributed_state()
            validation["distributed_initialized"] = state["is_initialized"]
            if state["is_initialized"]:
                validation["distributed_backend"] = state["backend"]
                validation["current_parallelism"] = {
                    "tp": state["tensor_parallel"]["size"],
                    "pp": state["pipeline_parallel"]["size"], 
                    "dp": state["data_parallel"]["size"]
                }
        except Exception:
            validation["distributed_initialized"] = False
    
    # Generate recommendations
    if validation["gpu_count"] == 0:
        validation["recommendations"].append("Use CPU training or install CUDA")
    elif validation["gpu_count"] == 1:
        validation["recommendations"].append("Single GPU - consider data parallelism with multiple nodes")
    elif validation["gpu_count"] >= 2:
        validation["recommendations"].append("Multiple GPUs available - tensor/pipeline parallelism beneficial")
        
        if validation["gpu_count"] >= 8:
            validation["recommendations"].append("8+ GPUs - consider 3D parallelism (TP+PP+DP)")
    
    # Check for optimal configuration
    if validation.get("distributed_initialized", False):
        current = validation.get("current_parallelism", {})
        total_parallel = current.get("tp", 1) * current.get("pp", 1) * current.get("dp", 1)
        if total_parallel != validation["gpu_count"]:
            validation["warnings"].append(f"Parallelism mismatch: using {total_parallel} of {validation['gpu_count']} GPUs")
    
    return validation

# Enhanced utility class
class ParallelismManager:
    """High-level parallelism management utility"""
    
    @staticmethod
    def auto_setup(
        model_size_params: Optional[int] = None,
        num_gpus: Optional[int] = None,
        memory_per_gpu_gb: float = 40.0,
        target_batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Automatically setup optimal parallelism configuration"""
        
        if not _DISTRIBUTED_INIT_AVAILABLE:
            return {"error": "Distributed initialization not available"}
        
        # Get GPU count if not provided
        if num_gpus is None:
            try:
                import torch
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            except:
                num_gpus = 1
        
        # Default model size if not provided
        if model_size_params is None:
            model_size_params = 7_000_000_000  # 7B default
        
        # Calculate optimal strategy
        strategy = ParallelismManager._calculate_optimal_strategy(
            model_size_params, num_gpus, memory_per_gpu_gb
        )
        
        # Setup distributed training
        success = setup_distributed_training(
            tensor_parallel_size=strategy["tensor_parallel_size"],
            pipeline_parallel_size=strategy["pipeline_parallel_size"],
            data_parallel_size=strategy["data_parallel_size"]
        )
        
        strategy["setup_success"] = success
        return strategy
    
    @staticmethod
    def _calculate_optimal_strategy(
        model_size_params: int,
        num_gpus: int,
        memory_per_gpu_gb: float
    ) -> Dict[str, Any]:
        """Calculate optimal parallelism strategy"""
        
        strategy = {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
            "reasoning": [],
            "memory_estimate": {}
        }
        
        # Rough memory estimation (in GB)
        # Model weights: params * 4 bytes (FP32) or 2 bytes (FP16)
        # Gradients: same as weights
        # Optimizer states: 2x weights for Adam
        # Activations: depends on batch size and sequence length
        
        model_memory_gb = (model_size_params * 6) / (1024**3)  # Conservative estimate with gradients + optimizer
        memory_per_gpu_available = memory_per_gpu_gb * 0.8  # 80% utilization
        
        strategy["memory_estimate"] = {
            "model_memory_gb": model_memory_gb,
            "available_per_gpu_gb": memory_per_gpu_available,
            "total_available_gb": memory_per_gpu_available * num_gpus
        }
        
        # Strategy selection based on model size and memory constraints
        if model_size_params < 1_000_000_000:  # < 1B parameters
            strategy["reasoning"].append("Small model - data parallelism sufficient")
            
        elif model_size_params < 10_000_000_000:  # 1B - 10B parameters
            if model_memory_gb > memory_per_gpu_available:
                # Need tensor parallelism
                tp_size = min(8, num_gpus, int(model_memory_gb / memory_per_gpu_available) + 1)
                strategy["tensor_parallel_size"] = tp_size
                strategy["data_parallel_size"] = num_gpus // tp_size
                strategy["reasoning"].append(f"Medium model requiring TP={tp_size} for memory")
            else:
                strategy["reasoning"].append("Medium model fits on single GPU - using data parallelism")
                
        elif model_size_params < 50_000_000_000:  # 10B - 50B parameters
            # Definitely need tensor parallelism
            tp_size = min(8, num_gpus)
            if num_gpus >= 16 and model_size_params > 30_000_000_000:
                # Also use pipeline parallelism for very large models
                pp_size = min(4, num_gpus // tp_size)
                strategy["pipeline_parallel_size"] = pp_size
                strategy["data_parallel_size"] = num_gpus // (tp_size * pp_size)
                strategy["reasoning"].append(f"Large model using 3D parallelism: TP={tp_size}, PP={pp_size}")
            else:
                strategy["data_parallel_size"] = num_gpus // tp_size
                strategy["reasoning"].append(f"Large model using TP={tp_size}")
            strategy["tensor_parallel_size"] = tp_size
            
        else:  # > 50B parameters
            # Very large model - need aggressive parallelism
            tp_size = min(8, num_gpus)
            pp_size = min(8, num_gpus // tp_size)
            
            if pp_size == 1 and num_gpus >= 16:
                pp_size = min(4, num_gpus // tp_size)
            
            strategy["tensor_parallel_size"] = tp_size
            strategy["pipeline_parallel_size"] = pp_size
            strategy["data_parallel_size"] = max(1, num_gpus // (tp_size * pp_size))
            strategy["reasoning"].append(f"Very large model requiring 3D parallelism: TP={tp_size}, PP={pp_size}")
        
        # Validate and adjust strategy
        total_gpus_used = (strategy["tensor_parallel_size"] * 
                          strategy["pipeline_parallel_size"] * 
                          strategy["data_parallel_size"])
        
        if total_gpus_used != num_gpus:
            # Adjust data parallel size to use all GPUs
            strategy["data_parallel_size"] = num_gpus // (
                strategy["tensor_parallel_size"] * strategy["pipeline_parallel_size"]
            )
            strategy["reasoning"].append(f"Adjusted to use all {num_gpus} GPUs")
        
        return strategy
    
    @staticmethod
    def get_memory_estimate(model_size_params: int, config: Dict[str, int]) -> Dict[str, float]:
        """Estimate memory usage with given parallelism configuration"""
        
        tp_size = config.get("tensor_parallel_size", 1)
        pp_size = config.get("pipeline_parallel_size", 1) 
        
        # Model is split across TP and PP dimensions
        model_params_per_gpu = model_size_params / (tp_size * pp_size)
        
        # Memory breakdown (in GB)
        memory_breakdown = {
            "model_weights": (model_params_per_gpu * 2) / (1024**3),  # FP16
            "gradients": (model_params_per_gpu * 2) / (1024**3),      # FP16
            "optimizer_states": (model_params_per_gpu * 8) / (1024**3),  # Adam: 2x FP32
            "activations": 2.0,  # Rough estimate for activations
            "overhead": 1.0      # Framework overhead
        }
        
        memory_breakdown["total_per_gpu"] = sum(memory_breakdown.values())
        
        return memory_breakdown

# Convenience functions with improved error handling
def quick_setup(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    **kwargs
) -> bool:
    """Quick setup for common parallelism configurations"""
    
    if not _DISTRIBUTED_INIT_AVAILABLE:
        logging.error("❌ Distributed initialization not available")
        return False
    
    try:
        success = setup_distributed_training(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            **kwargs
        )
        
        if success:
            print_distributed_info()
            return True
        else:
            logging.error("❌ Failed to setup distributed training")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error in quick setup: {e}")
        return False

def auto_setup_for_model(model_size: str = "7B") -> bool:
    """Auto setup parallelism for common model sizes"""
    
    size_map = {
        "1B": 1_000_000_000,
        "3B": 3_000_000_000,
        "7B": 7_000_000_000,
        "13B": 13_000_000_000,
        "30B": 30_000_000_000,
        "65B": 65_000_000_000,
        "175B": 175_000_000_000
    }
    
    if model_size not in size_map:
        logging.error(f"❌ Unknown model size: {model_size}. Available: {list(size_map.keys())}")
        return False
    
    model_params = size_map[model_size]
    result = ParallelismManager.auto_setup(model_size_params=model_params)
    
    if result.get("setup_success", False):
        print(f"✅ Auto-setup completed for {model_size} model")
        print("Strategy:", result.get("reasoning", []))
        return True
    else:
        logging.error(f"❌ Auto-setup failed for {model_size} model")
        return False

# Fallback implementations for missing components
class _FallbackConfig:
    """Fallback config when components not available"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Provide fallback configs if components not available
if not _TENSOR_PARALLEL_AVAILABLE:
    TensorParallelConfig = _FallbackConfig
    __all__.append("TensorParallelConfig")

if not _PIPELINE_PARALLEL_AVAILABLE:
    PipelineConfig = _FallbackConfig
    __all__.append("PipelineConfig")

# Add utility functions to exports
__all__.extend([
    "print_parallelism_info",
    "get_parallelism_capabilities", 
    "validate_parallelism_setup",
    "ParallelismManager",
    "quick_setup",
    "auto_setup_for_model"
])

# Module status and health check
def get_module_status() -> Dict[str, Any]:
    """Get comprehensive status of parallelism module"""
    
    status = {
        "version": __version__,
        "description": __description__,
        "components_available": get_parallelism_capabilities(),
        "total_exports": len(__all__),
        "validation": validate_parallelism_setup()
    }
    
    # Check if ready for production
    critical_components = ["distributed_init", "tensor_parallel"]
    status["ready_for_production"] = all(
        status["components_available"].get(comp, False) 
        for comp in critical_components
    )
    
    # Add recommendations
    status["recommendations"] = []
    if not status["ready_for_production"]:
        missing = [comp for comp in critical_components 
                  if not status["components_available"].get(comp, False)]
        status["recommendations"].append(f"Install missing components: {missing}")
    
    if status["validation"]["gpu_count"] == 0:
        status["recommendations"].append("Install CUDA for GPU acceleration")
    
    return status

def health_check() -> bool:
    """Perform health check of parallelism module"""
    
    print("🔍 Parallelism Module Health Check")
    print("=" * 40)
    
    status = get_module_status()
    
    # Check components
    components = status["components_available"]
    all_available = all(components.values())
    
    print("Component Status:")
    for component, available in components.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}")
    
    # Check validation
    validation = status["validation"]
    print(f"\nValidation: {'✅ PASS' if validation['valid'] else '❌ FAIL'}")
    
    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  ⚠️  {warning}")
    
    if validation["errors"]:
        print("Errors:")
        for error in validation["errors"]:
            print(f"  ❌ {error}")
    
    if validation["recommendations"]:
        print("Recommendations:")
        for rec in validation["recommendations"]:
            print(f"  💡 {rec}")
    
    # Overall status
    overall_healthy = (all_available and validation["valid"] and 
                      len(validation["errors"]) == 0)
    
    print(f"\nOverall Health: {'✅ HEALTHY' if overall_healthy else '⚠️  NEEDS ATTENTION'}")
    
    return overall_healthy

__all__.extend(["get_module_status", "health_check"])

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)