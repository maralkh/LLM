# training_infra/distributed/config.py
"""
Distributed training configuration classes for LLaMA models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import warnings


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism"""
    tensor_parallel_size: int = 1
    sequence_parallel: bool = False
    async_communication: bool = True
    all_reduce_fusion: bool = True
    
    def __post_init__(self):
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")


@dataclass 
class PipelineParallelConfig:
    """Configuration for pipeline parallelism"""
    pipeline_parallel_size: int = 1
    microbatch_size: Optional[int] = None
    num_microbatches: Optional[int] = None
    schedule: str = "1f1b"  # 1f1b, gpipe, chimera
    virtual_pipeline_stages: Optional[int] = None
    
    def __post_init__(self):
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        
        valid_schedules = ["1f1b", "gpipe", "chimera"]
        if self.schedule not in valid_schedules:
            raise ValueError(f"schedule must be one of {valid_schedules}")


@dataclass
class DataParallelConfig:
    """Configuration for data parallelism"""
    data_parallel_size: int = 1
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = True
    
    def __post_init__(self):
        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")


@dataclass
class ZeROConfig:
    """Configuration for ZeRO optimizer"""
    stage: int = 0  # 0: disabled, 1: optimizer states, 2: + gradients, 3: + parameters
    cpu_offload: bool = False
    nvme_offload: bool = False
    overlap_communication: bool = True
    reduce_bucket_size_mb: int = 25
    allgather_bucket_size_mb: int = 25
    
    def __post_init__(self):
        if self.stage not in [0, 1, 2, 3]:
            raise ValueError("ZeRO stage must be 0, 1, 2, or 3")


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training"""
    enabled: bool = True
    dtype: str = "bfloat16"  # float16, bfloat16
    loss_scale: str = "dynamic"  # dynamic, static, or float value
    initial_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    def __post_init__(self):
        valid_dtypes = ["float16", "bfloat16"]
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")


@dataclass
class CommunicationConfig:
    """Configuration for distributed communication"""
    backend: str = "nccl"  # nccl, gloo, mpi
    timeout_minutes: int = 30
    overlap_communication: bool = True
    use_gradient_compression: bool = False
    compression_ratio: float = 0.1
    async_communication: bool = True
    
    def __post_init__(self):
        valid_backends = ["nccl", "gloo", "mpi"]
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimizations"""
    use_activation_checkpointing: bool = False
    checkpoint_every_n_layers: int = 4
    cpu_offload_optimizer: bool = False
    cpu_offload_parameters: bool = False
    pin_memory: bool = True
    empty_cache_steps: int = 100
    
    def __post_init__(self):
        if self.checkpoint_every_n_layers < 1:
            raise ValueError("checkpoint_every_n_layers must be >= 1")


@dataclass
class DistributedConfig:
    """Main distributed training configuration"""
    
    # Parallelism configurations
    tensor_parallel: TensorParallelConfig = field(default_factory=TensorParallelConfig)
    pipeline_parallel: PipelineParallelConfig = field(default_factory=PipelineParallelConfig)
    data_parallel: DataParallelConfig = field(default_factory=DataParallelConfig)
    
    # Optimization configurations
    zero: ZeROConfig = field(default_factory=ZeROConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    memory_optimization: MemoryOptimizationConfig = field(default_factory=MemoryOptimizationConfig)
    
    # Runtime settings
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        """Validate and auto-configure settings"""
        self._validate_configuration()
        self._auto_configure()
    
    def _validate_configuration(self):
        """Validate configuration consistency"""
        total_gpus = (self.tensor_parallel.tensor_parallel_size * 
                     self.pipeline_parallel.pipeline_parallel_size * 
                     self.data_parallel.data_parallel_size)
        
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if total_gpus > available_gpus:
            warnings.warn(
                f"Requested {total_gpus} GPUs but only {available_gpus} available. "
                f"Configuration may need adjustment."
            )
        
        # Validate microbatch configuration
        if self.pipeline_parallel.pipeline_parallel_size > 1:
            if (self.pipeline_parallel.microbatch_size is None and 
                self.pipeline_parallel.num_microbatches is None):
                warnings.warn(
                    "Pipeline parallelism enabled but no microbatch configuration provided. "
                    "Will auto-configure based on batch size."
                )
    
    def _auto_configure(self):
        """Auto-configure settings based on available resources"""
        
        # Auto-configure pipeline microbatches
        if (self.pipeline_parallel.pipeline_parallel_size > 1 and 
            self.pipeline_parallel.num_microbatches is None):
            # Default: 4 microbatches per pipeline stage
            self.pipeline_parallel.num_microbatches = self.pipeline_parallel.pipeline_parallel_size * 4
        
        # Auto-enable activation checkpointing for large models
        if not self.memory_optimization.use_activation_checkpointing:
            total_params_estimate = self._estimate_total_parameters()
            if total_params_estimate > 10_000_000_000:  # 10B+ parameters
                self.memory_optimization.use_activation_checkpointing = True
                warnings.warn("Auto-enabled activation checkpointing for large model")
        
        # Auto-configure communication backend
        if torch.cuda.is_available() and self.communication.backend == "nccl":
            # NCCL is optimal for CUDA
            pass
        elif not torch.cuda.is_available() and self.communication.backend == "nccl":
            self.communication.backend = "gloo"
            warnings.warn("Changed communication backend to 'gloo' for CPU training")
    
    def _estimate_total_parameters(self) -> int:
        """Estimate total parameters (placeholder)"""
        # This would typically be calculated based on model configuration
        return 7_000_000_000  # Default estimate for 7B model
    
    @property
    def world_size(self) -> int:
        """Total number of processes"""
        return (self.tensor_parallel.tensor_parallel_size * 
               self.pipeline_parallel.pipeline_parallel_size * 
               self.data_parallel.data_parallel_size)
    
    @property
    def is_distributed(self) -> bool:
        """Check if distributed training is enabled"""
        return self.world_size > 1
    
    @property
    def uses_pipeline_parallelism(self) -> bool:
        """Check if pipeline parallelism is enabled"""
        return self.pipeline_parallel.pipeline_parallel_size > 1
    
    @property
    def uses_tensor_parallelism(self) -> bool:
        """Check if tensor parallelism is enabled"""
        return self.tensor_parallel.tensor_parallel_size > 1
    
    @property
    def uses_data_parallelism(self) -> bool:
        """Check if data parallelism is enabled"""
        return self.data_parallel.data_parallel_size > 1
    
    def get_parallelism_summary(self) -> Dict[str, Any]:
        """Get summary of parallelism configuration"""
        return {
            "world_size": self.world_size,
            "tensor_parallel_size": self.tensor_parallel.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel.data_parallel_size,
            "sequence_parallel": self.tensor_parallel.sequence_parallel,
            "zero_stage": self.zero.stage,
            "mixed_precision": self.mixed_precision.enabled,
            "activation_checkpointing": self.memory_optimization.use_activation_checkpointing
        }
    
    def print_configuration(self):
        """Print detailed configuration"""
        print("🚀 Distributed Training Configuration")
        print("=" * 50)
        
        print("\n📊 Parallelism:")
        print(f"  World Size: {self.world_size}")
        print(f"  Tensor Parallel: {self.tensor_parallel.tensor_parallel_size}")
        print(f"  Pipeline Parallel: {self.pipeline_parallel.pipeline_parallel_size}")
        print(f"  Data Parallel: {self.data_parallel.data_parallel_size}")
        
        if self.uses_pipeline_parallelism:
            print(f"\n🔄 Pipeline Configuration:")
            print(f"  Schedule: {self.pipeline_parallel.schedule}")
            print(f"  Microbatch Size: {self.pipeline_parallel.microbatch_size}")
            print(f"  Num Microbatches: {self.pipeline_parallel.num_microbatches}")
        
        print(f"\n⚡ Optimizations:")
        print(f"  ZeRO Stage: {self.zero.stage}")
        print(f"  Mixed Precision: {self.mixed_precision.enabled} ({self.mixed_precision.dtype})")
        print(f"  Activation Checkpointing: {self.memory_optimization.use_activation_checkpointing}")
        print(f"  Gradient Compression: {self.communication.use_gradient_compression}")
        
        print(f"\n🌐 Communication:")
        print(f"  Backend: {self.communication.backend}")
        print(f"  Overlap Communication: {self.communication.overlap_communication}")
        print(f"  Async Communication: {self.communication.async_communication}")


class ConfigurationFactory:
    """Factory for creating common distributed configurations"""
    
    @staticmethod
    def create_single_gpu_config() -> DistributedConfig:
        """Configuration for single GPU training"""
        return DistributedConfig(
            tensor_parallel=TensorParallelConfig(tensor_parallel_size=1),
            pipeline_parallel=PipelineParallelConfig(pipeline_parallel_size=1),
            data_parallel=DataParallelConfig(data_parallel_size=1),
            mixed_precision=MixedPrecisionConfig(enabled=True, dtype="bfloat16")
        )
    
    @staticmethod
    def create_data_parallel_config(num_gpus: int) -> DistributedConfig:
        """Configuration for data parallel training"""
        return DistributedConfig(
            tensor_parallel=TensorParallelConfig(tensor_parallel_size=1),
            pipeline_parallel=PipelineParallelConfig(pipeline_parallel_size=1),
            data_parallel=DataParallelConfig(data_parallel_size=num_gpus),
            mixed_precision=MixedPrecisionConfig(enabled=True, dtype="bfloat16")
        )
    
    @staticmethod
    def create_tensor_parallel_config(num_gpus: int) -> DistributedConfig:
        """Configuration for tensor parallel training"""
        return DistributedConfig(
            tensor_parallel=TensorParallelConfig(
                tensor_parallel_size=num_gpus,
                sequence_parallel=True if num_gpus > 4 else False
            ),
            pipeline_parallel=PipelineParallelConfig(pipeline_parallel_size=1),
            data_parallel=DataParallelConfig(data_parallel_size=1),
            mixed_precision=MixedPrecisionConfig(enabled=True, dtype="bfloat16")
        )
    
    @staticmethod
    def create_pipeline_parallel_config(
        pipeline_stages: int, 
        microbatch_size: int = 2,
        schedule: str = "1f1b"
    ) -> DistributedConfig:
        """Configuration for pipeline parallel training"""
        return DistributedConfig(
            tensor_parallel=TensorParallelConfig(tensor_parallel_size=1),
            pipeline_parallel=PipelineParallelConfig(
                pipeline_parallel_size=pipeline_stages,
                microbatch_size=microbatch_size,
                schedule=schedule
            ),
            data_parallel=DataParallelConfig(data_parallel_size=1),
            mixed_precision=MixedPrecisionConfig(enabled=True, dtype="bfloat16"),
            memory_optimization=MemoryOptimizationConfig(
                use_activation_checkpointing=True,
                checkpoint_every_n_layers=2
            )
        )
    
    @staticmethod
    def create_hybrid_parallel_config(
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        use_zero: bool = True,
        zero_stage: int = 2
    ) -> DistributedConfig:
        """Configuration for hybrid 3D parallelism"""
        return DistributedConfig(
            tensor_parallel=TensorParallelConfig(
                tensor_parallel_size=tensor_parallel_size,
                sequence_parallel=True
            ),
            pipeline_parallel=PipelineParallelConfig(
                pipeline_parallel_size=pipeline_parallel_size,
                schedule="1f1b"
            ),
            data_parallel=DataParallelConfig(
                data_parallel_size=data_parallel_size,
                find_unused_parameters=True
            ),
            zero=ZeROConfig(
                stage=zero_stage if use_zero else 0,
                overlap_communication=True
            ),
            mixed_precision=MixedPrecisionConfig(enabled=True, dtype="bfloat16"),
            memory_optimization=MemoryOptimizationConfig(
                use_activation_checkpointing=True,
                checkpoint_every_n_layers=2
            ),
            communication=CommunicationConfig(
                use_gradient_compression=True,
                compression_ratio=0.1
            )
        )
    
    @staticmethod
    def create_llama_7b_config(num_gpus: int) -> DistributedConfig:
        """Optimized configuration for LLaMA 7B"""
        if num_gpus == 1:
            return ConfigurationFactory.create_single_gpu_config()
        elif num_gpus <= 4:
            return ConfigurationFactory.create_data_parallel_config(num_gpus)
        elif num_gpus <= 8:
            return ConfigurationFactory.create_tensor_parallel_config(2)
        else:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                data_parallel_size=num_gpus // 4
            )
    
    @staticmethod
    def create_llama_13b_config(num_gpus: int) -> DistributedConfig:
        """Optimized configuration for LLaMA 13B"""
        if num_gpus < 2:
            raise ValueError("LLaMA 13B requires at least 2 GPUs")
        elif num_gpus <= 4:
            return ConfigurationFactory.create_tensor_parallel_config(num_gpus)
        elif num_gpus <= 8:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                data_parallel_size=num_gpus // 4
            )
        else:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=4,
                pipeline_parallel_size=2,
                data_parallel_size=num_gpus // 8,
                use_zero=True,
    @staticmethod
    def create_tiny_llama3_config(num_gpus: int = 1) -> DistributedConfig:
        """Optimized configuration for Tiny LLaMA 3 (development/testing)"""
        # Tiny models are efficient enough for single GPU
        return DistributedConfig(
            tensor_parallel=TensorParallelConfig(tensor_parallel_size=1),
            pipeline_parallel=PipelineParallelConfig(pipeline_parallel_size=1),
            data_parallel=DataParallelConfig(
                data_parallel_size=min(num_gpus, 4),  # Max 4 for tiny models
                gradient_accumulation_steps=4
            ),
            zero=ZeROConfig(stage=0),  # No ZeRO needed for tiny models
            mixed_precision=MixedPrecisionConfig(
                enabled=True, 
                dtype="bfloat16"
            ),
            memory_optimization=MemoryOptimizationConfig(
                use_activation_checkpointing=False,  # Not needed
                cpu_offload_optimizer=False,
                pin_memory=True
            ),
            communication=CommunicationConfig(
                backend="gloo" if not torch.cuda.is_available() else "nccl",
                use_gradient_compression=False,  # Not needed for small models
                overlap_communication=False
            )
        )
    
    @staticmethod
    def create_llama_70b_config(num_gpus: int) -> DistributedConfig:
        """Optimized configuration for LLaMA 70B"""
        if num_gpus < 8:
            raise ValueError("LLaMA 70B requires at least 8 GPUs")
        elif num_gpus == 8:
            return ConfigurationFactory.create_tensor_parallel_config(8)
        elif num_gpus <= 16:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=2,
                data_parallel_size=1,
                use_zero=True,
                zero_stage=3
            )
        else:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=4,
                data_parallel_size=num_gpus // 32,
                use_zero=True,
                zero_stage=3
            )

    @staticmethod
    def create_llama3_8b_config(num_gpus: int) -> DistributedConfig:
        """Optimized configuration for LLaMA 3 8B"""
        if num_gpus == 1:
            return ConfigurationFactory.create_single_gpu_config()
        elif num_gpus <= 4:
            return ConfigurationFactory.create_data_parallel_config(num_gpus)
        elif num_gpus <= 8:
            return ConfigurationFactory.create_tensor_parallel_config(2)
        else:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                data_parallel_size=num_gpus // 4,
                use_zero=True,
                zero_stage=2
            )
    
    @staticmethod
    def create_llama3_70b_config(num_gpus: int) -> DistributedConfig:
        """Optimized configuration for LLaMA 3 70B with improved architecture"""
        if num_gpus < 8:
            raise ValueError("LLaMA 3 70B requires at least 8 GPUs")
        elif num_gpus == 8:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                use_zero=True,
                zero_stage=3
            )
        elif num_gpus <= 16:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=2,
                data_parallel_size=1,
                use_zero=True,
                zero_stage=3
            )
        else:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=4,
                data_parallel_size=num_gpus // 32,
                use_zero=True,
                zero_stage=3
            )
    
    @staticmethod
    def create_llama3_405b_config(num_gpus: int) -> DistributedConfig:
        """Optimized configuration for LLaMA 3 405B (mega model)"""
        if num_gpus < 32:
            raise ValueError("LLaMA 3 405B requires at least 32 GPUs")
        elif num_gpus <= 64:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=8,
                data_parallel_size=1,
                use_zero=True,
                zero_stage=3
            )
        else:
            return ConfigurationFactory.create_hybrid_parallel_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=8,
                data_parallel_size=num_gpus // 64,
                use_zero=True,
                zero_stage=3
            )


# Auto-configuration utilities
class AutoConfigurator:
    """Automatically configure distributed training based on resources and model"""
    
    @staticmethod
    def auto_configure(
        model_size: str,
        available_gpus: Optional[int] = None,
        memory_per_gpu_gb: Optional[float] = None,
        target_throughput: Optional[float] = None
    ) -> DistributedConfig:
        """Automatically configure distributed training"""
        
        if available_gpus is None:
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if memory_per_gpu_gb is None:
            memory_per_gpu_gb = AutoConfigurator._estimate_gpu_memory()
        
        # Select configuration based on model size
        if "tiny" in model_size.lower():
            return ConfigurationFactory.create_tiny_llama3_config(available_gpus)
        elif "7b" in model_size.lower():
            return ConfigurationFactory.create_llama_7b_config(available_gpus)
        elif "8b" in model_size.lower():
            return ConfigurationFactory.create_llama3_8b_config(available_gpus)
        elif "13b" in model_size.lower():
            return ConfigurationFactory.create_llama_13b_config(available_gpus)
        elif "70b" in model_size.lower():
            if "llama3" in model_size.lower():
                return ConfigurationFactory.create_llama3_70b_config(available_gpus)
            else:
                return ConfigurationFactory.create_llama_70b_config(available_gpus)
        elif "405b" in model_size.lower():
            return ConfigurationFactory.create_llama3_405b_config(available_gpus)
        else:
            # Default to tiny for development
            return ConfigurationFactory.create_tiny_llama3_config(available_gpus)
    
    @staticmethod
    def _estimate_gpu_memory() -> float:
        """Estimate available GPU memory"""
        if not torch.cuda.is_available():
            return 0.0
        
        # Get memory of first GPU as estimate
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    @staticmethod
    def estimate_memory_requirements(
        model_size: str,
        batch_size: int,
        sequence_length: int,
        config: DistributedConfig
    ) -> Dict[str, float]:
        """Estimate memory requirements for given configuration"""
        
        # Model parameter counts (approximate)
        param_counts = {
            "50m": 50_000_000,         # Tiny LLaMA 3 50M
            "150m": 150_000_000,       # Tiny LLaMA 3 150M
            "7b": 7_000_000_000,
            "8b": 8_000_000_000,      # LLaMA 3 8B
            "13b": 13_000_000_000, 
            "30b": 30_000_000_000,
            "70b": 70_000_000_000,
            "405b": 405_000_000_000   # LLaMA 3 405B
        }
        
        size_key = next((k for k in param_counts.keys() if k in model_size.lower()), "150m")
        
        # Special handling for LLaMA 3 variants
        if "llama3" in model_size.lower():
            if "8b" in model_size.lower():
                size_key = "8b"
            elif "405b" in model_size.lower():
                size_key = "405b"
        
        # Handle tiny models
        if "tiny" in model_size.lower():
            if "50m" in model_size.lower():
                size_key = "50m"
            else:
                size_key = "150m"
        num_params = param_counts[size_key]
        
        # Parameter memory (4 bytes per param)
        param_memory_gb = num_params * 4 / 1024**3
        
        # Optimizer memory (8 bytes per param for AdamW)
        optimizer_memory_gb = num_params * 8 / 1024**3
        
        # Gradient memory (4 bytes per param)
        gradient_memory_gb = num_params * 4 / 1024**3
        
        # Activation memory (rough estimate)
        hidden_sizes = {"7b": 4096, "13b": 5120, "30b": 6656, "70b": 8192}
        num_layers = {"7b": 32, "13b": 40, "30b": 60, "70b": 80}
        
        hidden_size = hidden_sizes.get(size_key, 4096)
        layers = num_layers.get(size_key, 32)
        
        activation_memory_gb = (batch_size * sequence_length * hidden_size * layers * 4 * 2) / 1024**3
        
        # Apply reductions based on parallelism
        if config.uses_tensor_parallelism:
            param_memory_gb /= config.tensor_parallel.tensor_parallel_size
            optimizer_memory_gb /= config.tensor_parallel.tensor_parallel_size
            gradient_memory_gb /= config.tensor_parallel.tensor_parallel_size
        
        if config.uses_pipeline_parallelism:
            activation_memory_gb /= config.pipeline_parallel.pipeline_parallel_size
        
        if config.memory_optimization.use_activation_checkpointing:
            activation_memory_gb *= 0.25  # Roughly 75% reduction
        
        # ZeRO reductions
        if config.zero.stage >= 1:
            optimizer_memory_gb /= config.data_parallel.data_parallel_size
        if config.zero.stage >= 2:
            gradient_memory_gb /= config.data_parallel.data_parallel_size
        if config.zero.stage >= 3:
            param_memory_gb /= config.data_parallel.data_parallel_size
        
        total_memory_gb = param_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb
        
        return {
            "parameter_memory_gb": param_memory_gb,
            "optimizer_memory_gb": optimizer_memory_gb,
            "gradient_memory_gb": gradient_memory_gb,
            "activation_memory_gb": activation_memory_gb,
            "total_memory_gb": total_memory_gb,
            "memory_per_gpu_gb": total_memory_gb * 1.2,  # 20% overhead
            "recommended_gpu_memory_gb": total_memory_gb * 1.5  # 50% safety margin
        }


# Example usage
if __name__ == "__main__":
    # Example 1: Auto-configuration
    config = AutoConfigurator.auto_configure("llama2_7b", available_gpus=4)
    config.print_configuration()
    
    # Example 2: Manual configuration
    config = ConfigurationFactory.create_hybrid_parallel_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=2
    )
    
    # Example 3: Memory estimation
    memory_est = AutoConfigurator.estimate_memory_requirements(
        "llama2_7b", batch_size=8, sequence_length=2048, config=config
    )
    
    print("\n💾 Memory Estimation:")
    for key, value in memory_est.items():
        print(f"  {key}: {value:.2f}")