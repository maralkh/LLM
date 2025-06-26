"""
Training Infrastructure for Large Language Models

A comprehensive, production-ready training infrastructure for modern Large Language Models 
with advanced features including LLaMA architectures, Mixture of Experts, RLHF, 
reward-guided inference, synthetic data generation, and knowledge distillation.
"""

# Core components
from .config import TrainingConfig, GenerationConfig, InferenceConfig
from .trainer import Trainer
from .logger import setup_logging, get_logger
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from .utils import set_seed, get_device, save_checkpoint, load_checkpoint
from .advanced import AdvancedLlamaTrainer
from typing import List, Dict, Any, Optional, Tuple

# Model components
from .models import (
    create_llama_7b,
    create_llama_13b,
    create_llama_30b,
    create_llama_65b,
    create_llama_moe_7b,
    LlamaModel,
    LlamaMoEModel
)

# Inference components
from .inference import (
    InferenceEngine,
    create_sampler,
    SamplingConfig,
    create_reward_guided_engine,
    RewardGuidedConfig
)

# RLHF components
from .rlhf import (
    train_full_rlhf_pipeline,
    train_process_reward_model,
    train_outcome_reward_model,
    PPOTrainer,
    DPOTrainer,
    GRPOTrainer
)

# Data components
from .data import (
    create_synthetic_data_generator,
    SyntheticDataConfig,
    MathDataGenerator,
    CodeDataGenerator,
    ConstitutionalAIGenerator,
    compress_model_with_distillation,
    ProgressiveDistillationTrainer,
    DistillationConfig,
    evaluate_distillation_quality
)


# Pipeline components
from .pipeline import (
    SyntheticDistillationPipeline,
    DomainAdaptiveDistillationPipeline,
    MultiTeacherDistillationPipeline
)


# Version information
__version__ = "1.0.0"
__author__ = "Maral Khosroshahi"
__email__ = "your.email@example.com"

# Main exports
__all__ = [
    # Core training
    "TrainingConfig",
    "GenerationConfig", 
    "InferenceConfig",
    "Trainer",
    "AdvancedLlamaTrainer",
    "setup_logging",
    "get_logger",
    
    # Callbacks
    "EarlyStopping",
    "ModelCheckpoint", 
    "LearningRateScheduler",
    
    # Models
    "create_llama_7b",
    "create_llama_13b",
    "create_llama_30b", 
    "create_llama_65b",
    "create_llama_moe_7b",
    "LlamaModel",
    "LlamaMoEModel",
    
    # Inference
    "InferenceEngine",
    "create_sampler",
    "SamplingConfig",
    "create_reward_guided_engine",
    "RewardGuidedConfig",
    
    # RLHF
    "train_full_rlhf_pipeline",
    "train_process_reward_model", 
    "train_outcome_reward_model",
    "PPOTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    
    # Data generation
    "create_synthetic_data_generator",
    "SyntheticDataConfig",
    "MathDataGenerator",
    "CodeDataGenerator", 
    "ConstitutionalAIGenerator",
    
    # Distillation
    "compress_model_with_distillation",
    "ProgressiveDistillationTrainer",
    "DistillationConfig",
    "evaluate_distillation_quality",
    
    # Pipelines
    "SyntheticDistillationPipeline",
    "DomainAdaptiveDistillationPipeline",
    "MultiTeacherDistillationPipeline",
    
    # Utilities
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
]

# Convenience functions for quick access
def quick_train_llama(config_path: str, data_path: str):
    """Quick training setup for LLaMA models."""
    from .config import TrainingConfig
    from .models import create_llama_7b
    from .trainer import Trainer
    
    config = TrainingConfig.from_yaml(config_path)
    model = create_llama_7b()
    # Additional setup would go here
    return Trainer(model, config)

def quick_inference(model_path: str, prompt: str):
    """Quick inference setup."""
    from .inference import InferenceEngine
    engine = InferenceEngine.from_pretrained(model_path)
    return engine.generate(prompt)