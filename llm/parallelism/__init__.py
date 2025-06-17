# training_infra/parallelism/__init__.py
"""
Advanced parallelism support for large-scale training
"""

from .tensor_parallel import TensorParallelLinear, TensorParallelEmbedding, TensorParallelConfig
from .pipeline_parallel import PipelineParallelModel, PipelineStage, PipelineConfig
from .flash_attention import FlashAttention, FlashMHA, enable_flash_attention
from .distributed_trainer import AdvancedDistributedTrainer
from .memory_optimization import ActivationCheckpointing, GradientCompression

__all__ = [
    'TensorParallelLinear', 'TensorParallelEmbedding', 'TensorParallelConfig',
    'PipelineParallelModel', 'PipelineStage', 'PipelineConfig',
    'FlashAttention', 'FlashMHA', 'enable_flash_attention',
    'AdvancedDistributedTrainer',
    'ActivationCheckpointing', 'GradientCompression'
]