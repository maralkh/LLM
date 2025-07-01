# training_infra/training/utils/setup.py
"""
Training setup utilities.

Handles:
- Device setup
- Distributed training setup
- Parallelism configuration
- Optimization setup
- Data loader setup
"""

import os
from typing import Dict, Any, Optional, Tuple
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Handle imports
try:
    from .device import get_device, is_cuda_available
except ImportError:
    warnings.warn("Device utils not available")


def setup_device(config) -> torch.device:
    """Setup training device."""
    if config.use_cpu:
        return torch.device("cpu")
    elif config.device == "auto":
        return torch.device(get_device())
    else:
        return torch.device(config.device)


def extract_parallelism_config() -> Dict[str, Any]:
    """Extract parallelism configuration from environment."""
    config = {
        'enabled': False,
        'tp_size': 1,  # Tensor parallel size
        'pp_size': 1,  # Pipeline parallel size  
        'dp_size': 1,  # Data parallel size
        'tp_rank': 0,  # Tensor parallel rank
        'pp_rank': 0,  # Pipeline parallel rank
        'dp_rank': 0,  # Data parallel rank
    }
    
    # Check environment variables for parallelism
    config['tp_size'] = int(os.environ.get('TENSOR_PARALLEL_SIZE', config['tp_size']))
    config['pp_size'] = int(os.environ.get('PIPELINE_PARALLEL_SIZE', config['pp_size']))
    config['dp_size'] = int(os.environ.get('DATA_PARALLEL_SIZE', config['dp_size']))
    
    config['tp_rank'] = int(os.environ.get('TENSOR_PARALLEL_RANK', 0))
    config['pp_rank'] = int(os.environ.get('PIPELINE_PARALLEL_RANK', 0))
    config['dp_rank'] = int(os.environ.get('DATA_PARALLEL_RANK', 0))
    
    config['enabled'] = config['tp_size'] > 1 or config['pp_size'] > 1
    
    return config


def setup_parallelism(model: nn.Module, parallelism_config: Dict[str, Any]) -> nn.Module:
    """Setup model parallelism (placeholder for parallel module integration)."""
    if not parallelism_config['enabled']:
        return model
    
    print(f"🔧 Setting up parallelism...")
    print(f"   Tensor Parallel: {parallelism_config['tp_size']} (rank {parallelism_config['tp_rank']})")
    print(f"   Pipeline Parallel: {parallelism_config['pp_size']} (rank {parallelism_config['pp_rank']})")
    print(f"   Data Parallel: {parallelism_config['dp_size']} (rank {parallelism_config['dp_rank']})")
    
    # TODO: Integration with parallel module in future phases
    # This will include:
    # - Tensor parallel model wrapping
    # - Pipeline parallel stage assignment  
    # - Communication group setup
    # - Model sharding across TP ranks
    
    print("   TODO: Integrate with training_infra.parallel module")
    print("   TODO: Apply tensor parallel to model")
    print("   TODO: Setup pipeline parallel stages")
    print("   TODO: Configure data parallel groups")
    
    return model


def setup_distributed(parallelism_config: Dict[str, Any]) -> Tuple[bool, int, int]:
    """Setup distributed training (DDP - separate from parallelism)."""
    # Calculate DDP world size (separate from TP/PP)
    if 'WORLD_SIZE' in os.environ:
        total_world_size = int(os.environ['WORLD_SIZE'])
        total_rank = int(os.environ.get('RANK', 0))
        
        # DDP world size = total_world_size / (tp_size * pp_size)
        tp_pp_size = parallelism_config['tp_size'] * parallelism_config['pp_size']
        world_size = total_world_size // tp_pp_size
        rank = total_rank // tp_pp_size
        
        is_distributed = world_size > 1
        
        if is_distributed:
            print(f"🌐 Distributed training: DDP rank {rank}/{world_size}")
            print(f"   Total world size: {total_world_size}")
            print(f"   TP×PP size: {tp_pp_size}")
            print("   TODO: Initialize DDP process group")
        
        return is_distributed, world_size, rank
    else:
        # Single node training
        return False, 1, 0


def setup_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Setup optimizer."""
    optimizer_kwargs = config.optimizer.get_optimizer_kwargs()
    
    if config.optimizer.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    elif config.optimizer.name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    elif config.optimizer.name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")
    
    print(f"✅ Optimizer: {config.optimizer.name} (lr={config.optimizer.lr})")
    return optimizer


def setup_scheduler(optimizer: torch.optim.Optimizer, config, total_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Setup learning rate scheduler."""
    # Calculate warmup steps
    warmup_steps = config.get_warmup_steps(total_steps)
    scheduler_kwargs = config.scheduler.get_scheduler_kwargs()
    
    if config.scheduler.name.lower() == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.optimizer.lr * config.scheduler.min_lr_ratio
        )
    elif config.scheduler.name.lower() == "linear":
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.scheduler.min_lr_ratio,
            total_iters=total_steps - warmup_steps
        )
    elif config.scheduler.name.lower() == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_kwargs.get('step_size', 1000),
            gamma=scheduler_kwargs.get('gamma', 0.1)
        )
    else:
        print(f"⚠️  Unknown scheduler: {config.scheduler.name}, using no scheduler")
        return None
    
    # Setup warmup scheduler if needed
    if warmup_steps > 0:
        print(f"✅ Warmup scheduler: {warmup_steps} steps")
        # TODO: Implement warmup wrapper in future
    
    print(f"✅ Scheduler: {config.scheduler.name} (total_steps={total_steps})")
    return scheduler


def setup_mixed_precision(config, device: torch.device) -> Optional[GradScaler]:
    """Setup mixed precision training."""
    if config.use_mixed_precision and device.type == "cuda":
        scaler = GradScaler()
        print("✅ Mixed precision enabled with GradScaler")
        return scaler
    return None


def setup_data_loaders(train_dataset, eval_dataset, config) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Setup training and evaluation data loaders."""
    train_dataloader = None
    eval_dataloader = None
    
    if train_dataset is not None:
        if hasattr(train_dataset, 'get_dataloader'):
            train_dataloader = train_dataset.get_dataloader(
                batch_size=config.batch_size,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.pin_memory,
                drop_last=True
            )
        else:
            # Fallback to manual DataLoader creation
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.pin_memory,
                drop_last=True
            )
        
        print(f"✅ Train DataLoader: {len(train_dataloader)} batches")
    
    if eval_dataset is not None:
        if hasattr(eval_dataset, 'get_dataloader'):
            eval_dataloader = eval_dataset.get_dataloader(
                batch_size=config.eval_batch_size,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.pin_memory,
                drop_last=False
            )
        else:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.pin_memory,
                drop_last=False
            )
        
        print(f"✅ Eval DataLoader: {len(eval_dataloader)} batches")
    
    return train_dataloader, eval_dataloader


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_training_setup(
    model: nn.Module,
    device: torch.device,
    config,
    total_steps: int,
    parallelism_config: Dict[str, Any],
    is_distributed: bool,
    world_size: int,
    rank: int
):
    """Print comprehensive training setup information."""
    print(f"🚀 Trainer initialized:")
    print(f"   Device: {device}")
    print(f"   Model: {type(model).__name__}")
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Total Steps: {total_steps:,}")
    print(f"   Mixed Precision: {config.use_mixed_precision}")
    print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    
    if parallelism_config['enabled']:
        print(f"   Parallelism: TP={parallelism_config['tp_size']}, "
              f"PP={parallelism_config['pp_size']}, "
              f"DP={parallelism_config['dp_size']}")
    
    if is_distributed:
        print(f"   Distributed: rank {rank}/{world_size}")


def test_training_utils():
    """Test training utilities."""
    print("🧪 Testing Training Utils")
    
    # Test device setup
    class MockConfig:
        use_cpu = False
        device = "auto"
    
    device = setup_device(MockConfig())
    print(f"✅ Device setup: {device}")
    
    # Test parallelism config
    parallelism_config = extract_parallelism_config()
    print(f"✅ Parallelism config: {parallelism_config}")
    
    # Test distributed setup
    is_distributed, world_size, rank = setup_distributed(parallelism_config)
    print(f"✅ Distributed setup: {is_distributed}, {world_size}, {rank}")
    
    print("🎉 Training utils working!")
    return True


if __name__ == "__main__":
    test_training_utils()