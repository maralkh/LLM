# training_infra/config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import yaml
from pathlib import Path

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    params: Dict[str, Any] = field(default_factory=dict)

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
                return {k: convert_dataclass(v) for k, v in obj.__dict__.items()}
            return obj
        
        return convert_dataclass(self)
    
    def save_json(self, json_path: str):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)