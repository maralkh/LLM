# training_infra/config/training_configs.py
"""
Training configuration classes for the training infrastructure.

This module provides comprehensive configuration classes for training setup,
including optimizer configuration, scheduler configuration, and logging setup.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import os

# Handle imports for different execution contexts
try:
    from .base import BaseConfig
    from .validation import ValidationRule, ConfigValidator
except ImportError:
    # Direct execution - use absolute imports
    import sys
    from pathlib import Path as PathLib
    sys.path.append(str(PathLib(__file__).parent.parent))
    
    from training_infra.config.base import BaseConfig
    from training_infra.config.validation import ValidationRule, ConfigValidator


@dataclass
class OptimizerConfig(BaseConfig):
    """Configuration for optimizers."""
    
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # AdamW specific
    amsgrad: bool = False
    
    # SGD specific  
    momentum: float = 0.9
    nesterov: bool = False
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate optimizer configuration."""
        super().__post_init__()
        
        # Validate learning rate
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        
        # Validate weight decay
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
        
        # Validate betas for Adam-based optimizers
        if self.name.lower() in ['adam', 'adamw']:
            if not (0 <= self.beta1 < 1):
                raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
            if not (0 <= self.beta2 < 1):
                raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
    
    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Get optimizer arguments based on optimizer type."""
        base_kwargs = {
            'lr': self.lr,
            'weight_decay': self.weight_decay
        }
        
        if self.name.lower() in ['adam', 'adamw']:
            base_kwargs.update({
                'betas': (self.beta1, self.beta2),
                'eps': self.eps,
                'amsgrad': self.amsgrad
            })
        elif self.name.lower() == 'sgd':
            base_kwargs.update({
                'momentum': self.momentum,
                'nesterov': self.nesterov
            })
        
        # Add custom parameters
        base_kwargs.update(self.custom_params)
        
        return base_kwargs


@dataclass
class SchedulerConfig(BaseConfig):
    """Configuration for learning rate schedulers."""
    
    name: str = "cosine"
    
    # Common parameters
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Cosine scheduler
    min_lr_ratio: float = 0.1
    
    # Step scheduler
    step_size: int = 1000
    gamma: float = 0.1
    
    # Exponential scheduler
    decay_rate: float = 0.95
    
    # Polynomial scheduler
    power: float = 1.0
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate scheduler configuration."""
        super().__post_init__()
        
        # Validate warmup
        if self.warmup_steps < 0:
            raise ValueError(f"Warmup steps must be non-negative, got {self.warmup_steps}")
        if not (0 <= self.warmup_ratio <= 1):
            raise ValueError(f"Warmup ratio must be in [0, 1], got {self.warmup_ratio}")
        
        # Validate cosine parameters
        if self.name.lower() == 'cosine':
            if not (0 <= self.min_lr_ratio <= 1):
                raise ValueError(f"Min LR ratio must be in [0, 1], got {self.min_lr_ratio}")
        
        # Validate step parameters
        elif self.name.lower() == 'step':
            if self.step_size <= 0:
                raise ValueError(f"Step size must be positive, got {self.step_size}")
            if self.gamma <= 0:
                raise ValueError(f"Gamma must be positive, got {self.gamma}")
    
    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        """Get scheduler arguments based on scheduler type."""
        base_kwargs = {
            'warmup_steps': self.warmup_steps,
        }
        
        if self.name.lower() == 'cosine':
            base_kwargs.update({
                'min_lr_ratio': self.min_lr_ratio
            })
        elif self.name.lower() == 'step':
            base_kwargs.update({
                'step_size': self.step_size,
                'gamma': self.gamma
            })
        elif self.name.lower() == 'exponential':
            base_kwargs.update({
                'gamma': self.decay_rate
            })
        elif self.name.lower() == 'polynomial':
            base_kwargs.update({
                'power': self.power
            })
        
        # Add custom parameters
        base_kwargs.update(self.custom_params)
        
        return base_kwargs


@dataclass
class LoggingConfig(BaseConfig):
    """Configuration for training logging."""
    
    # Logging frequency
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 500
    
    # Output directories
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Experiment tracking
    experiment_name: str = "training_experiment"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Wandb integration
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # TensorBoard integration
    use_tensorboard: bool = True
    
    # Console logging
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    
    # Metrics to track
    track_grad_norm: bool = True
    track_learning_rate: bool = True
    track_memory_usage: bool = True
    
    def __post_init__(self):
        """Validate and setup logging configuration."""
        super().__post_init__()
        
        # Validate step frequencies
        if self.log_every_n_steps <= 0:
            raise ValueError(f"log_every_n_steps must be positive, got {self.log_every_n_steps}")
        if self.eval_every_n_steps <= 0:
            raise ValueError(f"eval_every_n_steps must be positive, got {self.eval_every_n_steps}")
        if self.save_every_n_steps <= 0:
            raise ValueError(f"save_every_n_steps must be positive, got {self.save_every_n_steps}")
        
        # Create directories
        for dir_path in [self.output_dir, self.log_dir, self.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Set default run name if not provided
        if self.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.experiment_name}_{timestamp}"
        
        # Validate log levels
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.console_log_level not in valid_levels:
            raise ValueError(f"Invalid console log level: {self.console_log_level}")
        if self.file_log_level not in valid_levels:
            raise ValueError(f"Invalid file log level: {self.file_log_level}")


@dataclass
class TrainingConfig(BaseConfig):
    """Main training configuration class."""
    
    # Basic training parameters
    epochs: int = 10
    max_steps: Optional[int] = None
    batch_size: int = 8
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    
    # Model parameters
    model_name: str = "tiny_llama_150m"
    model_config: Optional[Dict[str, Any]] = None
    
    # Data parameters
    dataset_name: str = "synthetic"
    dataset_config: Optional[Dict[str, Any]] = None
    max_seq_length: int = 512
    
    # Training parameters
    gradient_clipping: float = 1.0
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Logging and checkpointing
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Evaluation
    do_eval: bool = True
    eval_strategy: str = "steps"  # steps, epoch, no
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Device and distributed
    device: str = "auto"
    use_cpu: bool = False
    
    # Advanced options
    resume_from_checkpoint: Optional[str] = None
    save_only_model: bool = False
    load_best_model_at_end: bool = True
    
    def __post_init__(self):
        """Validate training configuration."""
        super().__post_init__()
        
        # Convert dict configs to proper objects if needed (for YAML/JSON loading)
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        
        # Basic validation
        if self.epochs <= 0 and self.max_steps is None:
            raise ValueError("Either epochs must be positive or max_steps must be set")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"Gradient accumulation steps must be positive, got {self.gradient_accumulation_steps}")
        
        # Set eval batch size if not provided
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        
        # Validate gradient clipping
        if self.gradient_clipping <= 0:
            raise ValueError(f"Gradient clipping must be positive, got {self.gradient_clipping}")
        
        # Validate max sequence length
        if self.max_seq_length <= 0:
            raise ValueError(f"Max sequence length must be positive, got {self.max_seq_length}")
        
        # Validate evaluation strategy
        valid_eval_strategies = ['steps', 'epoch', 'no']
        if self.eval_strategy not in valid_eval_strategies:
            raise ValueError(f"Invalid eval strategy: {self.eval_strategy}")
        
        # Update logging config with training info (only if it's an object, not dict)
        if hasattr(self.logging, 'experiment_name'):
            self.logging.experiment_name = f"{self.model_name}_{self.dataset_name}"
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_training_steps(self, dataset_size: int) -> int:
        """Calculate total training steps."""
        if self.max_steps is not None:
            return self.max_steps
        
        steps_per_epoch = dataset_size // self.get_effective_batch_size()
        return steps_per_epoch * self.epochs
    
    def get_warmup_steps(self, total_steps: int) -> int:
        """Calculate warmup steps based on scheduler config."""
        if self.scheduler.warmup_steps > 0:
            return self.scheduler.warmup_steps
        elif self.scheduler.warmup_ratio > 0:
            return int(total_steps * self.scheduler.warmup_ratio)
        return 0


# Predefined training configurations
def get_development_config() -> TrainingConfig:
    """Get configuration optimized for development/testing."""
    return TrainingConfig(
        epochs=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        model_name="tiny_llama_150m",
        dataset_name="synthetic",
        max_seq_length=256,
        use_mixed_precision=False,  # Disable for debugging
        optimizer=OptimizerConfig(
            name="adamw",
            lr=1e-3,  # Higher LR for faster testing
            weight_decay=0.01
        ),
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_ratio=0.1
        ),
        logging=LoggingConfig(
            log_every_n_steps=5,
            eval_every_n_steps=20,
            save_every_n_steps=50,
            use_wandb=False,
            use_tensorboard=True
        )
    )


def get_production_config() -> TrainingConfig:
    """Get configuration optimized for production training."""
    return TrainingConfig(
        epochs=10,
        batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size: 32
        model_name="tiny_llama_150m",
        dataset_name="text",
        max_seq_length=512,
        use_mixed_precision=True,
        optimizer=OptimizerConfig(
            name="adamw",
            lr=1e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999
        ),
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_ratio=0.05,
            min_lr_ratio=0.1
        ),
        logging=LoggingConfig(
            log_every_n_steps=10,
            eval_every_n_steps=100,
            save_every_n_steps=500,
            use_wandb=True,
            use_tensorboard=True,
            track_grad_norm=True,
            track_memory_usage=True
        ),
        gradient_clipping=1.0,
        do_eval=True,
        eval_strategy="steps"
    )


def get_large_model_config() -> TrainingConfig:
    """Get configuration for larger model training."""
    return TrainingConfig(
        epochs=5,
        batch_size=2,  # Smaller batch size for larger models
        gradient_accumulation_steps=16,  # Effective batch size: 32
        model_name="llama_1b",
        dataset_name="text",
        max_seq_length=1024,
        use_mixed_precision=True,
        optimizer=OptimizerConfig(
            name="adamw",
            lr=5e-5,  # Lower LR for larger models
            weight_decay=0.01
        ),
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_ratio=0.03,
            min_lr_ratio=0.05
        ),
        logging=LoggingConfig(
            log_every_n_steps=25,
            eval_every_n_steps=200,
            save_every_n_steps=1000,
            use_wandb=True,
            track_memory_usage=True
        ),
        gradient_clipping=0.5,  # Lower gradient clipping for stability
        dataloader_num_workers=8
    )


# Configuration validation rules
def create_training_config_validator() -> ConfigValidator:
    """Create validator for training configurations."""
    validator = ConfigValidator()
    
    # Basic training rules - simple field validation
    validator.add_rule(ValidationRule(
        "batch_size",
        lambda config: config.batch_size > 0 and config.batch_size <= 128,
        "batch_size must be between 1 and 128"
    ))
    
    validator.add_rule(ValidationRule(
        "epochs", 
        lambda config: config.epochs > 0 and config.epochs <= 1000,
        "epochs must be between 1 and 1000"
    ))
    
    validator.add_rule(ValidationRule(
        "gradient_accumulation_steps",
        lambda config: config.gradient_accumulation_steps > 0 and config.gradient_accumulation_steps <= 64,
        "gradient_accumulation_steps must be between 1 and 64"
    ))
    
    validator.add_rule(ValidationRule(
        "max_seq_length",
        lambda config: config.max_seq_length > 0 and config.max_seq_length <= 4096,
        "max_seq_length must be between 1 and 4096"
    ))
    
    validator.add_rule(ValidationRule(
        "gradient_clipping",
        lambda config: config.gradient_clipping > 0 and config.gradient_clipping <= 10.0,
        "gradient_clipping must be between 0 and 10.0"
    ))
    
    # Add custom validator for complex nested validation
    def validate_optimizer(config):
        """Validate optimizer configuration."""
        if not hasattr(config, 'optimizer'):
            return "optimizer configuration is missing"
        
        optimizer = config.optimizer
        if not hasattr(optimizer, 'lr'):
            return "optimizer learning rate is missing"
        
        if not (1e-6 <= optimizer.lr <= 1e-1):
            return f"optimizer learning rate ({optimizer.lr}) must be between 1e-6 and 1e-1"
        
        return None
    
    validator.add_custom_validator(validate_optimizer)
    
    return validator


def test_training_configs():
    """Test training configuration functionality."""
    print("🧪 Testing Training Configurations")
    
    # Test basic config creation
    config = TrainingConfig()
    print(f"✅ Basic config created: {config.model_name}, batch_size={config.batch_size}")
    
    # Test predefined configs
    dev_config = get_development_config()
    print(f"✅ Development config: {dev_config.epochs} epochs, lr={dev_config.optimizer.lr}")
    
    prod_config = get_production_config()
    print(f"✅ Production config: effective_batch_size={prod_config.get_effective_batch_size()}")
    
    # Test serialization
    config_dict = config.to_dict()
    restored_config = TrainingConfig.from_dict(config_dict)
    print(f"✅ Serialization works: {restored_config.model_name}")
    
    # Test validation
    validator = create_training_config_validator()
    try:
        validator.validate_strict(config)
        print("✅ Validation passed")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Test YAML save/load
    try:
        config.save_yaml("test_config.yaml")
        loaded_config = TrainingConfig.from_yaml("test_config.yaml")
        print("✅ YAML serialization works")
        
        # Cleanup
        import os
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")
    except Exception as e:
        print(f"❌ YAML serialization failed: {e}")
    
    print("🎉 Training configurations working!")
    return True


if __name__ == "__main__":
    test_training_configs()