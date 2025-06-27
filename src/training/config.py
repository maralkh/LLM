# training_infra/training/config.py
"""
Training configuration classes for LLaMA models.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
import json
import os
import warnings
from pathlib import Path


@dataclass
class TrainingConfig:
    """Main training configuration"""
    
    # Model and experiment
    model_name: str = "llama3_8b"
    experiment_name: Optional[str] = None
    
    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    
    # Optimization
    optimizer: str = "adamw"
    optimizer_config: Dict[str, Any] = field(default_factory=lambda: {
        "name": "adamw",
        "lr": 2e-5,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-8
    })
    
    scheduler: str = "cosine"
    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {
        "name": "cosine",
        "warmup_steps": 1000,
        "min_lr": 1e-6
    })
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    
    # Data
    sequence_length: int = 2048
    data_preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "tokenizer_name": None,
        "add_special_tokens": True,
        "padding": "max_length",
        "truncation": True
    })
    
    # Distributed training
    distributed_config: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "nccl",
        "find_unused_parameters": False,
        "bucket_cap_mb": 25,
        "use_ddp": True,
        "gradient_as_bucket_view": True
    })
    
    # DeepSpeed configuration
    deepspeed_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "enabled": False,
        "stage": 2,
        "offload_optimizer": False,
        "offload_parameters": False,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        }
    })
    
    # Logging and monitoring
    logging_config: Dict[str, Any] = field(default_factory=lambda: {
        "log_every": 50,
        "use_tensorboard": True,
        "use_wandb": False,
        "wandb_project": None,
        "wandb_entity": None,
        "log_level": "INFO",
        "save_logs": True
    })
    
    # Checkpointing
    checkpoint_config: Dict[str, Any] = field(default_factory=lambda: {
        "save_dir": "./checkpoints",
        "save_every": 1000,
        "keep_last": 3,
        "monitor": "val_loss",
        "mode": "min",
        "save_optimizer": True,
        "save_scheduler": True,
        "resume_from": None
    })
    
    # Evaluation
    evaluation_config: Dict[str, Any] = field(default_factory=lambda: {
        "eval_every": 1000,
        "eval_steps": 100,
        "early_stopping_patience": 3,
        "metric": "perplexity",
        "eval_datasets": [],
        "compute_metrics": True
    })
    
    # Hardware and performance
    hardware_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "compile_model": False,
        "use_flash_attention": True,
        "gradient_checkpointing": False
    })
    
    # Memory optimization
    memory_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": None,
        "empty_cache_steps": 100,
        "cpu_offload": False,
        "activation_checkpointing": False,
        "low_cpu_mem_usage": True
    })
    
    # Safety and debugging
    debug_config: Dict[str, Any] = field(default_factory=lambda: {
        "detect_anomaly": False,
        "profile_memory": False,
        "profile_compute": False,
        "check_finite": True,
        "log_nan_inf": True
    })
    
    # Other settings
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        """Post-initialization processing and validation"""
        
        # Basic validation
        self._validate_basic_params()
        
        # Update optimizer config with learning rate
        if self.optimizer_config["lr"] != self.learning_rate:
            self.optimizer_config["lr"] = self.learning_rate
        
        # Update scheduler config with warmup steps
        if self.scheduler_config["warmup_steps"] != self.warmup_steps:
            self.scheduler_config["warmup_steps"] = self.warmup_steps
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}"
        
        # Create directories
        self._create_directories()
        
        # Validate compatibility
        self._validate_compatibility()
    
    def _validate_basic_params(self):
        """Validate basic parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        
        # Validate optimizer
        valid_optimizers = ["adamw", "adam", "sgd", "adafactor"]
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
        
        # Validate scheduler
        valid_schedulers = ["cosine", "linear", "constant", "cosine_with_restarts"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}")
        
        # Validate amp_dtype
        valid_dtypes = ["float16", "bfloat16"]
        if self.amp_dtype not in valid_dtypes:
            raise ValueError(f"amp_dtype must be one of {valid_dtypes}")
    
    def _validate_compatibility(self):
        """Validate parameter combinations"""
        # Check DeepSpeed and DDP compatibility
        if self.deepspeed_config.get("enabled", False) and self.distributed_config.get("use_ddp", True):
            warnings.warn("DeepSpeed is enabled, DDP will be disabled automatically")
            self.distributed_config["use_ddp"] = False
        
        # Check gradient checkpointing and compilation
        if self.hardware_config.get("compile_model", False) and self.hardware_config.get("gradient_checkpointing", False):
            warnings.warn("Model compilation with gradient checkpointing may cause issues")
        
        # Check memory settings
        if self.memory_config.get("cpu_offload", False) and not self.deepspeed_config.get("enabled", False):
            warnings.warn("CPU offload is most effective with DeepSpeed enabled")
    
    def _create_directories(self):
        """Create necessary directories"""
        # Create checkpoint directory
        checkpoint_dir = Path(self.checkpoint_config["save_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if logging is enabled
        if self.logging_config.get("save_logs", True):
            logs_dir = checkpoint_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
    
    def estimate_memory_usage(self) -> Dict[str, str]:
        """Estimate approximate memory usage"""
        # Model size estimates (in billions of parameters)
        model_sizes = {
            "tiny_llama3_150m": 0.15,
            "llama3_1b": 1.0,
            "llama3_8b": 8.0,
            "llama3_70b": 70.0,
            "llama3_405b": 405.0
        }
        
        model_size = model_sizes.get(self.model_name, 8.0)  # Default to 8B
        
        # Rough memory estimates (GB)
        # Model weights (FP16)
        model_memory = model_size * 2  # 2 bytes per parameter
        
        # Optimizer states (AdamW needs ~2x model params for states)
        optimizer_memory = model_memory * 2 if self.optimizer == "adamw" else model_memory
        
        # Gradients
        gradient_memory = model_memory
        
        # Activations (depends on batch size and sequence length)
        activation_memory = (self.batch_size * self.sequence_length * model_size * 4) / 1e9  # Rough estimate
        
        # Total without optimizations
        total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
        
        # Apply optimizations
        if self.deepspeed_config.get("enabled", False):
            stage = self.deepspeed_config.get("stage", 2)
            if stage == 2:
                total_memory *= 0.5  # ZeRO-2 reduces memory by ~50%
            elif stage == 3:
                total_memory *= 0.3  # ZeRO-3 reduces memory by ~70%
        
        if self.hardware_config.get("gradient_checkpointing", False):
            activation_memory *= 0.5  # Roughly halves activation memory
            total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
        
        return {
            "model_params": f"{model_size:.1f}B",
            "model_memory": f"{model_memory:.1f} GB",
            "optimizer_memory": f"{optimizer_memory:.1f} GB", 
            "gradient_memory": f"{gradient_memory:.1f} GB",
            "activation_memory": f"{activation_memory:.1f} GB",
            "total_estimated": f"{total_memory:.1f} GB",
            "recommended_gpu_memory": f"{total_memory * 1.2:.1f} GB"  # 20% buffer
        }
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation"""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps"""
        steps_per_epoch = dataset_size // self.get_effective_batch_size()
        return steps_per_epoch * self.epochs
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load TrainingConfig from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainingConfig to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self, json_path: str):
        """Save TrainingConfig to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
        
        # Re-run post-init after updates
        self.__post_init__()
    
    def print_config(self):
        """Print configuration in a readable format"""
        print("⚙️ Training Configuration")
        print("=" * 50)
        
        print(f"\n📋 Model & Experiment:")
        print(f"  Model: {self.model_name}")
        print(f"  Experiment: {self.experiment_name}")
        
        print(f"\n🎯 Training Parameters:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Effective Batch Size: {self.get_effective_batch_size()}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"  Max Gradient Norm: {self.max_grad_norm}")
        print(f"  Warmup Steps: {self.warmup_steps}")
        
        print(f"\n⚡ Optimization:")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Scheduler: {self.scheduler}")
        print(f"  Mixed Precision: {self.use_amp} ({self.amp_dtype})")
        
        print(f"\n📊 Data:")
        print(f"  Sequence Length: {self.sequence_length}")
        
        print(f"\n🖥️ Hardware & Performance:")
        print(f"  Gradient Checkpointing: {self.hardware_config['gradient_checkpointing']}")
        print(f"  Flash Attention: {self.hardware_config['use_flash_attention']}")
        print(f"  Model Compilation: {self.hardware_config['compile_model']}")
        print(f"  Data Workers: {self.hardware_config['num_workers']}")
        
        print(f"\n🔄 Distributed Training:")
        print(f"  Use DDP: {self.distributed_config['use_ddp']}")
        print(f"  Backend: {self.distributed_config['backend']}")
        
        if self.deepspeed_config.get("enabled", False):
            print(f"\n🚀 DeepSpeed:")
            print(f"  Enabled: {self.deepspeed_config['enabled']}")
            print(f"  Stage: {self.deepspeed_config['stage']}")
            print(f"  Offload Optimizer: {self.deepspeed_config['offload_optimizer']}")
            print(f"  Offload Parameters: {self.deepspeed_config['offload_parameters']}")
        
        print(f"\n💾 Memory Estimation:")
        memory_info = self.estimate_memory_usage()
        for key, value in memory_info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📝 Logging:")
        print(f"  Log Every: {self.logging_config['log_every']} steps")
        print(f"  TensorBoard: {self.logging_config['use_tensorboard']}")
        print(f"  Weights & Biases: {self.logging_config['use_wandb']}")
        if self.logging_config['use_wandb']:
            print(f"  W&B Project: {self.logging_config['wandb_project']}")
        
        print(f"\n💾 Checkpointing:")
        print(f"  Save Every: {self.checkpoint_config['save_every']} steps")
        print(f"  Keep Last: {self.checkpoint_config['keep_last']} checkpoints")
        print(f"  Save Directory: {self.checkpoint_config['save_dir']}")
        if self.checkpoint_config['resume_from']:
            print(f"  Resume From: {self.checkpoint_config['resume_from']}")
        
        print(f"\n📊 Evaluation:")
        print(f"  Eval Every: {self.evaluation_config['eval_every']} steps")
        print(f"  Eval Steps: {self.evaluation_config['eval_steps']}")
        print(f"  Early Stopping Patience: {self.evaluation_config['early_stopping_patience']}")
        print(f"  Primary Metric: {self.evaluation_config['metric']}")


# Configuration factory for common setups
class TrainingConfigFactory:
    """Factory for creating common training configurations"""
    
    @staticmethod
    def create_tiny_config(model_name: str = "tiny_llama3_150m") -> TrainingConfig:
        """Configuration for tiny models (development/testing)"""
        return TrainingConfig(
            model_name=model_name,
            epochs=1,
            batch_size=32,
            learning_rate=5e-4,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            sequence_length=512,
            logging_config={
                "log_every": 10,
                "use_tensorboard": True,
                "use_wandb": False,
                "log_level": "DEBUG",
                "save_logs": True
            },
            checkpoint_config={
                "save_dir": "./checkpoints",
                "save_every": 100,
                "keep_last": 2,
                "monitor": "val_loss",
                "mode": "min",
                "save_optimizer": False,
                "save_scheduler": False
            },
            hardware_config={
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "compile_model": False,
                "use_flash_attention": False,
                "gradient_checkpointing": False
            },
            deepspeed_config={"enabled": False}
        )
    
    @staticmethod
    def create_standard_config(model_name: str = "llama3_8b") -> TrainingConfig:
        """Configuration for standard models"""
        return TrainingConfig(
            model_name=model_name,
            epochs=3,
            batch_size=8,
            learning_rate=2e-5,
            gradient_accumulation_steps=16,
            warmup_steps=2000,
            sequence_length=2048,
            logging_config={
                "log_every": 50,
                "use_tensorboard": True,
                "use_wandb": True,
                "wandb_project": f"llama_training_{model_name}",
                "log_level": "INFO",
                "save_logs": True
            },
            hardware_config={
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
                "compile_model": True,
                "use_flash_attention": True,
                "gradient_checkpointing": True
            },
            deepspeed_config={
                "enabled": True,
                "stage": 2,
                "offload_optimizer": False,
                "offload_parameters": False,
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True
                }
            }
        )
    
    @staticmethod
    def create_large_config(model_name: str = "llama3_70b") -> TrainingConfig:
        """Configuration for large models"""
        return TrainingConfig(
            model_name=model_name,
            epochs=1,
            batch_size=1,
            learning_rate=5e-6,
            gradient_accumulation_steps=128,
            warmup_steps=1000,
            sequence_length=2048,
            use_amp=True,
            amp_dtype="bfloat16",
            logging_config={
                "log_every": 25,
                "use_tensorboard": True,
                "use_wandb": True,
                "wandb_project": f"llama_training_{model_name}",
                "log_level": "INFO",
                "save_logs": True
            },
            hardware_config={
                "num_workers": 8,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
                "compile_model": False,  # May cause issues with large models
                "use_flash_attention": True,
                "gradient_checkpointing": True
            },
            deepspeed_config={
                "enabled": True,
                "stage": 3,
                "offload_optimizer": True,
                "offload_parameters": True,
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 1e9,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "gather_16bit_weights_on_model_save": True
                }
            },
            memory_config={
                "max_memory_mb": None,
                "empty_cache_steps": 50,
                "cpu_offload": True,
                "activation_checkpointing": True,
                "low_cpu_mem_usage": True
            }
        )
    
    @staticmethod
    def create_debug_config(model_name: str = "tiny_llama3_150m") -> TrainingConfig:
        """Configuration optimized for debugging"""
        config = TrainingConfigFactory.create_tiny_config(model_name)
        config.debug_config = {
            "detect_anomaly": True,
            "profile_memory": True,
            "profile_compute": True,
            "check_finite": True,
            "log_nan_inf": True
        }
        config.logging_config["log_level"] = "DEBUG"
        config.logging_config["log_every"] = 1
        config.deterministic = True
        return config
    
    @staticmethod
    def create_inference_config(model_name: str = "llama3_8b") -> TrainingConfig:
        """Configuration optimized for inference/evaluation only"""
        return TrainingConfig(
            model_name=model_name,
            epochs=0,  # No training
            batch_size=1,
            learning_rate=0.0,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            sequence_length=2048,
            use_amp=True,
            amp_dtype="bfloat16",
            logging_config={
                "log_every": 100,
                "use_tensorboard": False,
                "use_wandb": False,
                "log_level": "INFO",
                "save_logs": False
            },
            checkpoint_config={
                "save_dir": "./checkpoints",
                "save_every": 99999999,  # Never save during inference
                "keep_last": 0,
                "save_optimizer": False,
                "save_scheduler": False
            },
            evaluation_config={
                "eval_every": 1,
                "eval_steps": -1,  # Evaluate on full dataset
                "early_stopping_patience": 0,
                "metric": "perplexity",
                "compute_metrics": True
            },
            hardware_config={
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "compile_model": True,
                "use_flash_attention": True,
                "gradient_checkpointing": False  # Not needed for inference
            },
            deepspeed_config={"enabled": False}
        )


# Utility functions
def load_config_with_overrides(base_config_path: str, overrides: Dict[str, Any] = None) -> TrainingConfig:
    """Load config from file and apply overrides"""
    config = TrainingConfig.from_json(base_config_path)
    if overrides:
        config.update(overrides)
    return config


def compare_configs(config1: TrainingConfig, config2: TrainingConfig) -> Dict[str, Any]:
    """Compare two configurations and return differences"""
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    differences = {}
    for key in dict1:
        if dict1[key] != dict2[key]:
            differences[key] = {
                "config1": dict1[key],
                "config2": dict2[key]
            }
    
    return differences


# Example usage
if __name__ == "__main__":
    # Create different configurations
    tiny_config = TrainingConfigFactory.create_tiny_config()
    standard_config = TrainingConfigFactory.create_standard_config()
    large_config = TrainingConfigFactory.create_large_config("llama3_70b")
    
    # Print configurations
    print("🔍 Tiny Model Configuration:")
    tiny_config.print_config()
    
    print("\n" + "="*80 + "\n")
    
    print("🚀 Standard Model Configuration:")
    standard_config.print_config()
    
    # Save configurations
    tiny_config.to_json("tiny_config.json")
    standard_config.to_json("standard_config.json")
    large_config.to_json("large_config.json")
    
    # Example of loading and updating
    loaded_config = TrainingConfig.from_json("standard_config.json")
    loaded_config.update({"learning_rate": 1e-5, "batch_size": 16})
    print(f"\n🔄 Updated learning rate: {loaded_config.learning_rate}")
    print(f"🔄 Updated batch size: {loaded_config.batch_size}")