# training_infra/utils.py
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import time
from contextlib import contextmanager

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_distributed(rank: int = -1, world_size: int = -1, backend: str = 'nccl'):
    """Setup distributed training"""
    if rank == -1:
        rank = int(os.environ.get('RANK', 0))
    if world_size == -1:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(rank)
        return True
    return False

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, filepath, 
                   config=None, scaler=None, **kwargs):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'timestamp': time.time(),
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config.to_dict() if hasattr(config, 'to_dict') else config
    
    torch.save(checkpoint, filepath)
    return checkpoint

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None, device=None):
    """Load training checkpoint"""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint

def create_optimizer(model, config):
    """Create optimizer from config"""
    optimizer_name = config.optimizer.name.lower()
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
            **config.optimizer.params
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
            **config.optimizer.params
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.params.get('momentum', 0.9),
            weight_decay=config.optimizer.weight_decay,
            **{k: v for k, v in config.optimizer.params.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config"""
    scheduler_name = config.scheduler.name.lower()
    
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.total_steps or 1000,
            eta_min=config.scheduler.min_lr,
            **config.scheduler.params
        )
    elif scheduler_name == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.scheduler.min_lr / config.optimizer.lr,
            total_iters=config.scheduler.total_steps or 1000,
            **config.scheduler.params
        )
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.params.get('step_size', 1000),
            gamma=config.scheduler.params.get('gamma', 0.1),
            **{k: v for k, v in config.scheduler.params.items() if k not in ['step_size', 'gamma']}
        )
    elif scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler.params.get('factor', 0.5),
            patience=config.scheduler.params.get('patience', 5),
            **{k: v for k, v in config.scheduler.params.items() if k not in ['factor', 'patience']}
        )
    else:
        return None

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer:
    """Simple timer utility"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

@contextmanager
def timer():
    """Context manager for timing code blocks"""
    t = Timer()
    t.start()
    try:
        yield t
    finally:
        t.stop()

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"

def get_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {}

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_config(config):
    """Validate training configuration"""
    errors = []
    
    # Check required fields
    if not config.dataset_path:
        errors.append("dataset_path is required")
    
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    
    if config.epochs <= 0 and config.max_steps is None:
        errors.append("Either epochs or max_steps must be specified")
    
    if config.optimizer.lr <= 0:
        errors.append("Learning rate must be positive")
    
    # Check paths exist
    if config.dataset_path and not Path(config.dataset_path).exists():
        errors.append(f"Dataset path does not exist: {config.dataset_path}")
    
    if config.resume_from and not Path(config.resume_from).exists():
        errors.append(f"Resume checkpoint does not exist: {config.resume_from}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

def log_system_info(logger):
    """Log system information"""
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / 1024**3
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        })
    
    logger.log_text("System Information:")
    for key, value in info.items():
        logger.log_text(f"  {key}: {value}")
    
    return info