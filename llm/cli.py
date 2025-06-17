# training_infra/cli.py
import argparse
import sys
import os
from pathlib import Path
import json
import yaml

from .config import TrainingConfig

def create_config_template(output_path: str, format_type: str = "yaml"):
    """Create a template configuration file"""
    config = TrainingConfig()
    
    if format_type.lower() == "yaml":
        config.save_yaml(output_path)
    elif format_type.lower() == "json":
        config.save_json(output_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"Template configuration saved to {output_path}")

def validate_config_file(config_path: str):
    """Validate a configuration file"""
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = TrainingConfig.from_yaml(config_path)
        elif config_path.endswith('.json'):
            config = TrainingConfig.from_json(config_path)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
        
        print(f"✅ Configuration file {config_path} is valid")
        
        # Print summary
        print("\nConfiguration Summary:")
        print(f"  Model: {config.model_name}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rate: {config.optimizer.lr}")
        print(f"  Optimizer: {config.optimizer.name}")
        print(f"  Scheduler: {config.scheduler.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def compare_configs(config1_path: str, config2_path: str):
    """Compare two configuration files"""
    def load_config(path):
        if path.endswith('.yaml') or path.endswith('.yml'):
            return TrainingConfig.from_yaml(path)
        elif path.endswith('.json'):
            return TrainingConfig.from_json(path)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    try:
        config1 = load_config(config1_path)
        config2 = load_config(config2_path)
        
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        def compare_dicts(d1, d2, path=""):
            differences = []
            
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences.append(f"Key missing in config1: {current_path}")
                elif key not in d2:
                    differences.append(f"Key missing in config2: {current_path}")
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    differences.extend(compare_dicts(d1[key], d2[key], current_path))
                elif d1[key] != d2[key]:
                    differences.append(f"Different values at {current_path}: {d1[key]} vs {d2[key]}")
            
            return differences
        
        differences = compare_dicts(dict1, dict2)
        
        if differences:
            print(f"Found {len(differences)} differences between configs:")
            for diff in differences:
                print(f"  - {diff}")
        else:
            print("✅ Configurations are identical")
            
    except Exception as e:
        print(f"❌ Config comparison failed: {e}")

def list_experiments(experiments_dir: str = "./experiments"):
    """List available experiment configurations"""
    exp_dir = Path(experiments_dir)
    
    if not exp_dir.exists():
        print(f"Experiments directory {experiments_dir} does not exist")
        return
    
    config_files = []
    for pattern in ["*.yaml", "*.yml", "*.json"]:
        config_files.extend(exp_dir.glob(pattern))
    
    if not config_files:
        print(f"No configuration files found in {experiments_dir}")
        return
    
    print(f"Available experiments in {experiments_dir}:")
    
    for config_file in sorted(config_files):
        try:
            if config_file.suffix in ['.yaml', '.yml']:
                config = TrainingConfig.from_yaml(str(config_file))
            else:
                config = TrainingConfig.from_json(str(config_file))
            
            print(f"  📄 {config_file.name}")
            print(f"     Model: {config.model_name}")
            print(f"     Epochs: {config.epochs}, Batch Size: {config.batch_size}")
            print(f"     LR: {config.optimizer.lr}, Optimizer: {config.optimizer.name}")
            print()
            
        except Exception as e:
            print(f"  ❌ {config_file.name} (invalid: {e})")

def show_system_info():
    """Show system information for training"""
    import torch
    import platform
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
    except ImportError:
        memory_gb = "Unknown"
        cpu_count = "Unknown"
    
    print("System Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  RAM: {memory_gb:.1f} GB" if isinstance(memory_gb, float) else f"  RAM: {memory_gb}")
    
    print(f"\nGPU Information:")
    if torch.cuda.is_available():
        print(f"  CUDA Available: Yes")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print(f"  CUDA Available: No")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  MPS Available: Yes")
    
    print(f"\nTraining Recommendations:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"  - Use distributed training with {gpu_count} GPUs")
            print(f"  - Suggested command: torchrun --nproc_per_node={gpu_count} train.py")
        print(f"  - Enable mixed precision training (AMP)")
        
        # Memory recommendations
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 8:
            print(f"  - Use smaller batch sizes (GPU memory: {gpu_memory:.1f} GB)")
        elif gpu_memory > 16:
            print(f"  - Can use larger batch sizes (GPU memory: {gpu_memory:.1f} GB)")
    else:
        print(f"  - CPU training detected - consider using a GPU for faster training")

def estimate_training_time(config_path: str, dataset_size: int = None):
    """Estimate training time based on configuration"""
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = TrainingConfig.from_yaml(config_path)
        else:
            config = TrainingConfig.from_json(config_path)
        
        print(f"Training Time Estimation for {config.model_name}:")
        print(f"  Configuration: {config_path}")
        
        # Basic calculations
        epochs = config.epochs
        batch_size = config.batch_size
        
        if dataset_size:
            steps_per_epoch = dataset_size // batch_size
            total_steps = steps_per_epoch * epochs
            
            print(f"  Dataset Size: {dataset_size:,} samples")
            print(f"  Steps per Epoch: {steps_per_epoch:,}")
            print(f"  Total Steps: {total_steps:,}")
            
            # Rough time estimates (very approximate)
            if torch.cuda.is_available():
                seconds_per_step = 0.1  # Rough estimate for GPU
                device_type = "GPU"
            else:
                seconds_per_step = 1.0  # Rough estimate for CPU
                device_type = "CPU"
            
            total_seconds = total_steps * seconds_per_step
            hours = total_seconds / 3600
            
            print(f"  Estimated Time ({device_type}): {hours:.1f} hours")
            print(f"  Time per Epoch: {hours/epochs:.1f} hours")
            
        else:
            print(f"  Epochs: {epochs}")
            print(f"  Batch Size: {batch_size}")
            print(f"  Provide --dataset-size for time estimation")
        
        # Memory estimation
        if torch.cuda.is_available():
            print(f"\n  Memory Considerations:")
            print(f"    Batch Size: {batch_size}")
            if batch_size > 64:
                print(f"    Warning: Large batch size may require significant GPU memory")
            if config.use_amp:
                print(f"    Mixed Precision: Enabled (saves ~50% memory)")
            else:
                print(f"    Mixed Precision: Disabled (consider enabling)")
        
    except Exception as e:
        print(f"❌ Could not estimate training time: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Training Infrastructure CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a template config
  training-infra create-config config.yaml
  
  # Validate a config file
  training-infra validate config.yaml
  
  # Compare two configs
  training-infra compare config1.yaml config2.yaml
  
  # List experiments
  training-infra list-experiments ./experiments
  
  # Show system info
  training-infra system-info
  
  # Estimate training time
  training-infra estimate-time config.yaml --dataset-size 100000
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config template
    create_parser = subparsers.add_parser('create-config', help='Create configuration template')
    create_parser.add_argument('output', help='Output file path')
    create_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Output format (default: yaml)')
    
    # Validate config
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config', help='Configuration file path')
    
    # Compare configs
    compare_parser = subparsers.add_parser('compare', help='Compare two configuration files')
    compare_parser.add_argument('config1', help='First configuration file')
    compare_parser.add_argument('config2', help='Second configuration file')
    
    # List experiments
    list_parser = subparsers.add_parser('list-experiments', help='List available experiments')
    list_parser.add_argument('--dir', default='./experiments', 
                            help='Experiments directory (default: ./experiments)')
    
    # System info
    subparsers.add_parser('system-info', help='Show system information')
    
    # Estimate training time
    estimate_parser = subparsers.add_parser('estimate-time', help='Estimate training time')
    estimate_parser.add_argument('config', help='Configuration file path')
    estimate_parser.add_argument('--dataset-size', type=int, 
                                help='Dataset size for time estimation')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'create-config':
            create_config_template(args.output, args.format)
        
        elif args.command == 'validate':
            if not validate_config_file(args.config):
                sys.exit(1)
        
        elif args.command == 'compare':
            compare_configs(args.config1, args.config2)
        
        elif args.command == 'list-experiments':
            list_experiments(args.dir)
        
        elif args.command == 'system-info':
            show_system_info()
        
        elif args.command == 'estimate-time':
            estimate_training_time(args.config, args.dataset_size)
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

# ---

# training_infra/templates.py
"""Configuration templates for different use cases"""

from .config import TrainingConfig, OptimizerConfig, SchedulerConfig, LoggingConfig, CheckpointConfig

def get_classification_config():
    """Get template config for image classification"""
    return TrainingConfig(
        model_name="image_classifier",
        epochs=100,
        batch_size=128,
        eval_batch_size=256,
        max_length=224,  # Image size
        
        optimizer=OptimizerConfig(
            name="adamw",
            lr=1e-3,
            weight_decay=0.05
        ),
        
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_steps=1000,
            min_lr=1e-6
        ),
        
        logging=LoggingConfig(
            log_every=100,
            use_tensorboard=True,
            use_wandb=False
        ),
        
        checkpoint=CheckpointConfig(
            save_every=10,
            monitor="val_accuracy",
            mode="max"
        ),
        
        use_amp=True,
        max_grad_norm=1.0,
        eval_every=5
    )

def get_language_model_config():
    """Get template config for language model training"""
    return TrainingConfig(
        model_name="language_model",
        epochs=10,
        batch_size=8,
        eval_batch_size=16,
        max_length=512,
        gradient_accumulation_steps=16,  # Effective batch size = 8 * 16 = 128
        
        optimizer=OptimizerConfig(
            name="adamw",
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        ),
        
        scheduler=SchedulerConfig(
            name="linear",
            warmup_steps=2000
        ),
        
        logging=LoggingConfig(
            log_every=50,
            use_tensorboard=True,
            use_wandb=True,
            wandb_project="language_model"
        ),
        
        checkpoint=CheckpointConfig(
            save_every=2,
            monitor="val_loss",
            mode="min",
            keep_last=3
        ),
        
        use_amp=True,
        amp_dtype="bfloat16",
        max_grad_norm=1.0,
        eval_every=1000
    )

def get_distributed_config():
    """Get template config for distributed training"""
    return TrainingConfig(
        model_name="distributed_model",
        epochs=50,
        batch_size=32,  # Per GPU
        
        optimizer=OptimizerConfig(
            name="adamw",
            lr=1e-4,
            weight_decay=0.01
        ),
        
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_steps=1000
        ),
        
        logging=LoggingConfig(
            log_every=100,
            use_tensorboard=True,
            use_wandb=True
        ),
        
        checkpoint=CheckpointConfig(
            save_every=5,
            keep_last=3
        ),
        
        distributed=TrainingConfig.DistributedConfig(
            enabled=True,
            backend="nccl",
            find_unused_parameters=False
        ),
        
        use_amp=True,
        max_grad_norm=1.0
    )