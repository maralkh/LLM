# training_infra/config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
import json
import yaml
from pathlib import Path

# Custom YAML representer for tuples
def tuple_representer(dumper, data):
    return dumper.represent_list(data)

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

# Register custom YAML handlers
yaml.add_representer(tuple, tuple_representer)
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

@dataclass
class GenerationConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@dataclass
class InferenceConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Union[Tuple[float, float], List[float]] = field(default_factory=lambda: [0.9, 0.999])  # Use list instead of tuple
    eps: float = 1e-8
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert list to tuple if needed for compatibility
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)

@dataclass  
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 1000
    total_steps: Optional[int] = None
    min_lr: float = 1e-6
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CheckpointConfig:
    save_dir: str = "./checkpoints"
    save_every: int = 1000
    keep_last: int = 5
    save_best: bool = True
    monitor: str = "val_loss"
    mode: str = "min"

@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    log_every: int = 100
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True

@dataclass
class TrainingConfig:
    # Model & Data
    model_name: str = "gpt2"
    dataset_path: str = ""
    batch_size: int = 32
    eval_batch_size: Optional[int] = None
    max_length: int = 512
    
    # Training
    epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Checkpointing & Logging
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Distributed Training
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # Evaluation
    eval_every: int = 1000
    eval_steps: Optional[int] = None
    
    # Mixed Precision
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Data Loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Resume Training
    resume_from: Optional[str] = None
    
    def __post_init__(self):
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
            
        # Ensure directories exist
        Path(self.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        # Handle nested configs
        if 'optimizer' in config_dict:
            config_dict['optimizer'] = OptimizerConfig(**config_dict['optimizer'])
        if 'scheduler' in config_dict:
            config_dict['scheduler'] = SchedulerConfig(**config_dict['scheduler'])
        if 'checkpoint' in config_dict:
            config_dict['checkpoint'] = CheckpointConfig(**config_dict['checkpoint'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        if 'distributed' in config_dict:
            config_dict['distributed'] = DistributedConfig(**config_dict['distributed'])
            
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        def convert_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for k, v in obj.__dict__.items():
                    converted_v = convert_dataclass(v)
                    # Convert tuples to lists for better YAML compatibility
                    if isinstance(converted_v, tuple):
                        converted_v = list(converted_v)
                    result[k] = converted_v
                return result
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        return convert_dataclass(self)
    
    def save_json(self, json_path: str):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Config saved to {json_path}")
    
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        print(f"✅ Config saved to {yaml_path}")

# Alternative safer YAML handling class
class SafeYAMLConfig:
    """A safer version of config handling that avoids tuple issues"""
    
    @staticmethod
    def save_yaml_safe(config: TrainingConfig, yaml_path: str):
        """Save config to YAML with safe handling of complex types"""
        config_dict = config.to_dict()
        
        # Custom YAML dumper that handles complex types safely
        class SafeDumper(yaml.SafeDumper):
            pass
        
        def represent_none(self, data):
            return self.represent_scalar('tag:yaml.org,2002:null', '')
        
        SafeDumper.add_representer(type(None), represent_none)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, Dumper=SafeDumper, default_flow_style=False, indent=2)
        print(f"✅ Config safely saved to {yaml_path}")
    
    @staticmethod
    def load_yaml_safe(yaml_path: str) -> TrainingConfig:
        """Load config from YAML with safe handling"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return TrainingConfig.from_dict(config_dict)

# Utility functions for config validation
def validate_config(config: TrainingConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors"""
    warnings = []
    
    # Check learning rate
    if config.optimizer.lr > 1e-2:
        warnings.append(f"Learning rate {config.optimizer.lr} seems high")
    elif config.optimizer.lr < 1e-6:
        warnings.append(f"Learning rate {config.optimizer.lr} seems very low")
    
    # Check batch size
    if config.batch_size < 1:
        warnings.append("Batch size must be positive")
    elif config.batch_size > 1024:
        warnings.append(f"Batch size {config.batch_size} is very large")
    
    # Check epochs
    if config.epochs < 1:
        warnings.append("Number of epochs must be positive")
    
    # Check gradient accumulation
    if config.gradient_accumulation_steps < 1:
        warnings.append("Gradient accumulation steps must be positive")
    
    # Check scheduler configuration
    if config.scheduler.name == "linear" and config.scheduler.total_steps is None:
        warnings.append("Linear scheduler requires total_steps to be specified")
    
    # Check distributed settings
    if config.distributed.enabled and config.distributed.backend not in ["nccl", "gloo", "mpi"]:
        warnings.append(f"Unknown distributed backend: {config.distributed.backend}")
    
    return warnings

def create_default_configs() -> Dict[str, TrainingConfig]:
    """Create a set of default configurations for common use cases"""
    
    configs = {}
    
    # Small model config
    configs["small"] = TrainingConfig(
        model_name="small_model",
        batch_size=32,
        epochs=10,
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=500)
    )
    
    # Large model config
    configs["large"] = TrainingConfig(
        model_name="large_model",
        batch_size=8,
        epochs=5,
        gradient_accumulation_steps=4,
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.01),
        scheduler=SchedulerConfig(name="linear", warmup_steps=1000),
        use_amp=True
    )
    
    # Fine-tuning config
    configs["finetune"] = TrainingConfig(
        model_name="finetuned_model",
        batch_size=16,
        epochs=3,
        optimizer=OptimizerConfig(lr=5e-5, weight_decay=0.01),
        scheduler=SchedulerConfig(name="linear", warmup_steps=100),
        checkpoint=CheckpointConfig(save_every=500, monitor="val_loss")
    )
    
    # Distributed training config
    configs["distributed"] = TrainingConfig(
        model_name="distributed_model",
        batch_size=16,
        epochs=10,
        optimizer=OptimizerConfig(lr=1e-4),
        distributed=DistributedConfig(enabled=True, backend="nccl"),
        logging=LoggingConfig(use_wandb=True, wandb_project="distributed_training")
    )
    
    return configs

# Example usage
if __name__ == "__main__":
    # Test configuration creation and serialization
    config = TrainingConfig(
        model_name="test_model",
        optimizer=OptimizerConfig(lr=1e-4, betas=[0.9, 0.999]),  # Use list instead of tuple
        scheduler=SchedulerConfig(warmup_steps=1000)
    )
    
    print("Testing configuration serialization...")
    
    # Test JSON
    config.save_json("test_config.json")
    loaded_json = TrainingConfig.from_json("test_config.json")
    print(f"JSON test passed: {loaded_json.model_name}")
    
    # Test YAML
    config.save_yaml("test_config.yaml")
    loaded_yaml = TrainingConfig.from_yaml("test_config.yaml")
    print(f"YAML test passed: {loaded_yaml.model_name}")
    
    # Test safe YAML
    SafeYAMLConfig.save_yaml_safe(config, "test_config_safe.yaml")
    loaded_safe = SafeYAMLConfig.load_yaml_safe("test_config_safe.yaml")
    print(f"Safe YAML test passed: {loaded_safe.model_name}")
    
    # Test validation
    warnings = validate_config(config)
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Configuration validation passed")
    
    print("All tests completed successfully!")