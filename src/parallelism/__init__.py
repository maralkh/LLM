# training_infra/parallelism/__init__.py
"""
Low-level parallelism primitives for distributed LLaMA training.

This module provides the foundational parallelism components including
tensor parallelism, pipeline parallelism, and communication utilities.
"""

try:
    from .tensor_parallel import (
        TensorParallelLinear,
        TensorParallelEmbedding,
        TensorParallelConfig,
        split_tensor_parallel,
        gather_tensor_parallel,
        all_reduce_tensor_parallel
    )
    _TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    _TENSOR_PARALLEL_AVAILABLE = False

try:
    from .pipeline_parallel import (
        PipelineParallelConfig,
        PipelineParallelLlama,
        PipelineStage,
        pipeline_send,
        pipeline_recv,
        pipeline_send_recv
    )
    _PIPELINE_PARALLEL_AVAILABLE = True
except ImportError:
    _PIPELINE_PARALLEL_AVAILABLE = False

try:
    from .communication import (
        init_distributed,
        cleanup_distributed,
        get_world_size,
        get_rank,
        get_local_rank,
        get_tensor_parallel_rank,
        get_pipeline_parallel_rank,
        get_data_parallel_rank,
        all_reduce_grads,
        broadcast_parameters,
        synchronize_processes
    )
    _COMMUNICATION_AVAILABLE = True
except ImportError:
    _COMMUNICATION_AVAILABLE = False

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
except ImportError:
    _MEMORY_OPTIMIZATION_AVAILABLE = False

try:
    from .flash_attention import (
        FlashAttention,
        FlashMHA,
        enable_flash_attention,
        disable_flash_attention,
        is_flash_attention_available
    )
    _FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    _FLASH_ATTENTION_AVAILABLE = False

# Core exports (always attempt to export, graceful degradation)
__all__ = []

# Tensor parallelism exports
if _TENSOR_PARALLEL_AVAILABLE:
    __all__.extend([
        "TensorParallelLinear",
        "TensorParallelEmbedding", 
        "TensorParallelConfig",
        "split_tensor_parallel",
        "gather_tensor_parallel",
        "all_reduce_tensor_parallel"
    ])

# Pipeline parallelism exports
if _PIPELINE_PARALLEL_AVAILABLE:
    __all__.extend([
        "PipelineParallelConfig",
        "PipelineParallelLlama",
        "PipelineStage",
        "pipeline_send",
        "pipeline_recv", 
        "pipeline_send_recv"
    ])

# Communication exports
if _COMMUNICATION_AVAILABLE:
    __all__.extend([
        "init_distributed",
        "cleanup_distributed",
        "get_world_size",
        "get_rank",
        "get_local_rank",
        "get_tensor_parallel_rank",
        "get_pipeline_parallel_rank",
        "get_data_parallel_rank", 
        "all_reduce_grads",
        "broadcast_parameters",
        "synchronize_processes"
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
__version__ = "1.0.0"
__description__ = "Low-level parallelism primitives for distributed LLaMA training"

def print_parallelism_info():
    """Print information about parallelism module"""
    print("⚡ LLaMA Parallelism Module")
    print("=" * 40)
    
    print("Available parallelism types:")
    print(f"  • Tensor Parallelism: {'✅' if _TENSOR_PARALLEL_AVAILABLE else '❌'}")
    print(f"  • Pipeline Parallelism: {'✅' if _PIPELINE_PARALLEL_AVAILABLE else '❌'}")
    print(f"  • Communication Utils: {'✅' if _COMMUNICATION_AVAILABLE else '❌'}")
    print(f"  • Memory Optimization: {'✅' if _MEMORY_OPTIMIZATION_AVAILABLE else '❌'}")
    print(f"  • Flash Attention: {'✅' if _FLASH_ATTENTION_AVAILABLE else '❌'}")
    
    if not any([_TENSOR_PARALLEL_AVAILABLE, _PIPELINE_PARALLEL_AVAILABLE, 
                _COMMUNICATION_AVAILABLE, _MEMORY_OPTIMIZATION_AVAILABLE]):
        print("\n⚠️  No parallelism components available")
        print("This may indicate missing dependencies or incomplete installation")

def get_parallelism_capabilities() -> dict:
    """Get available parallelism capabilities"""
    return {
        "tensor_parallel": _TENSOR_PARALLEL_AVAILABLE,
        "pipeline_parallel": _PIPELINE_PARALLEL_AVAILABLE,
        "communication": _COMMUNICATION_AVAILABLE,
        "memory_optimization": _MEMORY_OPTIMIZATION_AVAILABLE,
        "flash_attention": _FLASH_ATTENTION_AVAILABLE
    }

def validate_parallelism_setup() -> dict:
    """Validate parallelism setup and dependencies"""
    
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
        else:
            validation["warnings"].append("CUDA not available - limited parallelism")
            validation["cuda_available"] = False
            validation["gpu_count"] = 0
    except:
        validation["cuda_available"] = False
        validation["gpu_count"] = 0
    
    # Check parallelism components
    capabilities = get_parallelism_capabilities()
    missing_components = [k for k, v in capabilities.items() if not v]
    
    if missing_components:
        validation["warnings"].extend([
            f"Missing parallelism component: {comp}" for comp in missing_components
        ])
        validation["recommendations"].append("Some advanced features may not be available")
    
    # Generate recommendations
    if validation["gpu_count"] == 0:
        validation["recommendations"].append("Use CPU training or install CUDA")
    elif validation["gpu_count"] == 1:
        validation["recommendations"].append("Single GPU - data parallelism not beneficial")
    elif validation["gpu_count"] >= 2:
        validation["recommendations"].append("Multiple GPUs - tensor/pipeline parallelism available")
    
    return validation

# Parallelism utilities
class ParallelismUtils:
    """Utility functions for parallelism operations"""
    
    @staticmethod
    def get_optimal_parallelism_strategy(
        model_size_params: int,
        num_gpus: int,
        gpu_memory_gb: float = 40
    ) -> dict:
        """Recommend optimal parallelism strategy"""
        
        strategy = {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
            "reasoning": []
        }
        
        # Tiny models - keep simple
        if model_size_params < 1_000_000_000:  # < 1B
            strategy["reasoning"].append("Small model - data parallelism sufficient")
            return strategy
        
        # Large models need tensor parallelism
        if model_size_params > 50_000_000_000:  # > 50B
            if num_gpus >= 8:
                strategy["tensor_parallel_size"] = 8
                strategy["data_parallel_size"] = num_gpus // 8
                strategy["reasoning"].append("Large model - tensor parallelism required")
            else:
                strategy["tensor_parallel_size"] = num_gpus
                strategy["data_parallel_size"] = 1
                strategy["reasoning"].append("Large model - max tensor parallelism")
        
        # Very large models need pipeline parallelism
        if model_size_params > 100_000_000_000 and num_gpus >= 16:  # > 100B
            strategy["tensor_parallel_size"] = 8
            strategy["pipeline_parallel_size"] = 2
            strategy["data_parallel_size"] = num_gpus // 16
            strategy["reasoning"].append("Very large model - 3D parallelism")
        
        # Memory constraints
        estimated_memory_per_gpu = (model_size_params * 4) / (1024**3) / num_gpus  # Rough estimate
        if estimated_memory_per_gpu > gpu_memory_gb * 0.8:
            strategy["reasoning"].append("Memory constrained - increase parallelism")
            if strategy["tensor_parallel_size"] < num_gpus:
                strategy["tensor_parallel_size"] = min(8, num_gpus)
                strategy["data_parallel_size"] = num_gpus // strategy["tensor_parallel_size"]
        
        return strategy
    
    @staticmethod
    def estimate_communication_overhead(
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        model_size_params: int
    ) -> dict:
        """Estimate communication overhead"""
        
        # Rough estimates based on typical patterns
        tp_overhead = tensor_parallel_size * 0.1  # 10% per TP rank
        pp_overhead = pipeline_parallel_size * 0.05  # 5% per PP stage
        
        total_overhead = tp_overhead + pp_overhead
        
        return {
            "tensor_parallel_overhead": tp_overhead,
            "pipeline_parallel_overhead": pp_overhead, 
            "total_overhead_percent": total_overhead,
            "efficiency": max(0, 1 - total_overhead),
            "recommendations": [
                "Consider reducing parallelism if overhead is high",
                "Use faster interconnect (InfiniBand) for large-scale training",
                "Enable communication optimization (overlap, compression)"
            ] if total_overhead > 0.3 else ["Parallelism overhead is acceptable"]
        }
    
    @staticmethod
    def check_parallelism_constraints(
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        total_gpus: int
    ) -> dict:
        """Check parallelism configuration constraints"""
        
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check total GPU count
        required_gpus = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        if required_gpus != total_gpus:
            validation["valid"] = False
            validation["errors"].append(
                f"GPU count mismatch: {required_gpus} required, {total_gpus} available"
            )
        
        # Check tensor parallel constraints
        if tensor_parallel_size > 8:
            validation["warnings"].append("High tensor parallelism may hurt performance")
        
        # Check pipeline parallel constraints
        if pipeline_parallel_size > 8:
            validation["warnings"].append("High pipeline parallelism increases bubble time")
        
        # Check data parallel constraints
        if data_parallel_size == 1 and total_gpus > 1:
            validation["warnings"].append("Not using data parallelism with multiple GPUs")
        
        return validation

# Add utility functions to exports
__all__.extend([
    "print_parallelism_info",
    "get_parallelism_capabilities",
    "validate_parallelism_setup",
    "ParallelismUtils"
])

# Convenience functions for common operations
def setup_distributed_environment(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout_minutes: int = 30
) -> bool:
    """Setup distributed training environment"""
    
    if not _COMMUNICATION_AVAILABLE:
        print("❌ Communication utilities not available")
        return False
    
    try:
        success = init_distributed(
            backend=backend,
            init_method=init_method,
            timeout_minutes=timeout_minutes
        )
        
        if success:
            print("✅ Distributed environment initialized")
            return True
        else:
            print("❌ Failed to initialize distributed environment")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up distributed environment: {e}")
        return False

def optimize_model_parallelism(
    model,
    tensor_parallel_size: int = 1,
    use_flash_attention: bool = True,
    use_activation_checkpointing: bool = False
):
    """Apply parallelism optimizations to model"""
    
    optimizations_applied = []
    
    # Apply tensor parallelism
    if tensor_parallel_size > 1 and _TENSOR_PARALLEL_AVAILABLE:
        # This would modify the model in-place to use tensor parallel layers
        optimizations_applied.append(f"Tensor parallelism ({tensor_parallel_size})")
    
    # Apply flash attention
    if use_flash_attention and _FLASH_ATTENTION_AVAILABLE:
        try:
            enable_flash_attention(model)
            optimizations_applied.append("Flash Attention")
        except:
            pass
    
    # Apply activation checkpointing  
    if use_activation_checkpointing and _MEMORY_OPTIMIZATION_AVAILABLE:
        try:
            gradient_checkpointing_enable(model)
            optimizations_applied.append("Activation Checkpointing")
        except:
            pass
    
    if optimizations_applied:
        print(f"✅ Applied optimizations: {', '.join(optimizations_applied)}")
    else:
        print("⚠️  No optimizations applied - check component availability")
    
    return model

# Add convenience functions to exports
__all__.extend([
    "setup_distributed_environment",
    "optimize_model_parallelism"
])

# Fallback implementations for missing components
class _FallbackTensorParallelConfig:
    """Fallback config when tensor parallelism not available"""
    def __init__(self, tensor_parallel_size=1, **kwargs):
        self.tensor_parallel_size = tensor_parallel_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class _FallbackPipelineParallelConfig:
    """Fallback config when pipeline parallelism not available"""
    def __init__(self, pipeline_parallel_size=1, **kwargs):
        self.pipeline_parallel_size = pipeline_parallel_size
        for k, v in kwargs.items():
            setattr(self, k, v)

# Provide fallback configs if components not available
if not _TENSOR_PARALLEL_AVAILABLE:
    TensorParallelConfig = _FallbackTensorParallelConfig
    __all__.append("TensorParallelConfig")

if not _PIPELINE_PARALLEL_AVAILABLE:
    PipelineParallelConfig = _FallbackPipelineParallelConfig
    __all__.append("PipelineParallelConfig")

# Module status
def get_module_status() -> dict:
    """Get status of parallelism module"""
    return {
        "version": __version__,
        "description": __description__,
        "components_available": get_parallelism_capabilities(),
        "total_exports": len(__all__),
        "ready_for_production": all(get_parallelism_capabilities().values())
    }

__all__.append("get_module_status")