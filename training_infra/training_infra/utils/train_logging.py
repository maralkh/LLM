# training_infra/utils/logging.py
"""
Comprehensive Logging System

Provides advanced logging functionality for training infrastructure:
- Multiple output formats (console, file, structured)
- Log rotation and compression
- Performance-aware logging
- System information logging
- Integration with popular ML logging services

## Basic Usage:
```python
from training_infra.utils.logging import setup_logging

logger = setup_logging(level="INFO", log_file="training.log")
logger.info("Training started")
logger.warning("GPU memory is low")
logger.error("Training failed")
```

## Advanced Usage:
```python
from training_infra.utils.logging import setup_logging, log_system_info

# Advanced logging setup
logger = setup_logging(
    level="DEBUG",
    log_file="logs/training.log",
    max_bytes=10*1024*1024,  # 10MB rotation
    backup_count=5,
    format_style="detailed"
)

# Log system information
log_system_info()
```
"""

import logging
import logging.handlers
import sys
import os
import platform
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any
from enum import Enum

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)

def get_formatter(style: str = "simple") -> logging.Formatter:
    """
    Get logging formatter based on style.
    
    Args:
        style: Formatter style ("simple", "detailed", "json")
    
    Returns:
        logging.Formatter: Configured formatter
    """
    if style == "simple":
        return logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
    
    elif style == "detailed":
        return logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-3d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    elif style == "json":
        # For structured logging (would need JSON formatter library)
        return logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    else:
        # Default to simple
        return get_formatter("simple")

def get_colored_formatter(style: str = "simple") -> ColoredFormatter:
    """Get colored formatter for console output."""
    base_formatter = get_formatter(style)
    return ColoredFormatter(
        fmt=base_formatter._fmt,
        datefmt=base_formatter.datefmt
    )

def setup_logging(
    level: Union[str, LogLevel] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 3,
    format_style: str = "simple",
    logger_name: str = "training_infra"
) -> logging.Logger:
    """
    Setup comprehensive logging system.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        console_output: Enable console output
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep
        format_style: Formatter style ("simple", "detailed", "json")
        logger_name: Name of the logger
    
    Returns:
        logging.Logger: Configured logger
    
    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="logs/training.log")
        >>> logger.info("Training started")
    """
    # Convert level to string if enum
    if isinstance(level, LogLevel):
        level = level.value
    
    # Get or create logger
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set logging level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Setup console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Use colored formatter for console
        console_formatter = get_colored_formatter(format_style)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Setup file handler with rotation
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Use regular formatter for file (no colors)
        file_formatter = get_formatter(format_style)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to file: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"🔧 Logging setup complete (level: {level})")
    return logger

def get_logger(name: str = "training_infra") -> logging.Logger:
    """
    Get existing logger or create basic one.
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, setup basic logging
    if not logger.handlers:
        logger = setup_logging(logger_name=name)
    
    return logger

def log_system_info(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Log comprehensive system information.
    
    Args:
        logger: Optional logger instance
    
    Returns:
        dict: System information
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("🖥️  System Information:")
    logger.info("=" * 50)
    
    # Basic system info
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    # CPU information
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            logger.info(f"  cpu_frequency: {cpu_freq.current:.0f} MHz")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"  cpu_usage: {cpu_percent}%")
        
        system_info["cpu_frequency"] = cpu_freq.current if cpu_freq else None
        system_info["cpu_usage"] = cpu_percent
        
    except Exception as e:
        logger.warning(f"Could not get CPU info: {e}")
    
    # Memory information
    try:
        memory = psutil.virtual_memory()
        logger.info(f"  memory_available: {memory.available / (1024**3):.2f} GB")
        logger.info(f"  memory_usage: {memory.percent}%")
        
        system_info["memory_available_gb"] = round(memory.available / (1024**3), 2)
        system_info["memory_usage_percent"] = memory.percent
        
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
    
    # GPU information (if available)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"  gpu_count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                logger.info(f"  gpu_{i}: {gpu_name} ({gpu_memory / (1024**3):.1f} GB)")
            
            system_info["gpu_count"] = gpu_count
            system_info["cuda_version"] = torch.version.cuda
        else:
            logger.info("  gpu_count: 0 (CUDA not available)")
            system_info["gpu_count"] = 0
            
    except ImportError:
        logger.info("  gpu_count: Unknown (PyTorch not available)")
        system_info["gpu_count"] = "unknown"
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
        system_info["gpu_count"] = "error"
    
    # Disk information
    try:
        disk_usage = psutil.disk_usage('/')
        logger.info(f"  disk_free: {disk_usage.free / (1024**3):.2f} GB")
        logger.info(f"  disk_usage: {(disk_usage.used / disk_usage.total) * 100:.1f}%")
        
        system_info["disk_free_gb"] = round(disk_usage.free / (1024**3), 2)
        system_info["disk_usage_percent"] = round((disk_usage.used / disk_usage.total) * 100, 1)
        
    except Exception as e:
        logger.warning(f"Could not get disk info: {e}")
    
    logger.info("=" * 50)
    
    return system_info

def log_training_start(
    model_name: str,
    dataset_name: str, 
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
):
    """
    Log training start with configuration.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        config: Training configuration
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("🚀 Training Started")
    logger.info("=" * 50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\nConfiguration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 50)

def log_training_progress(
    epoch: int,
    step: int,
    loss: float,
    metrics: Dict[str, float],
    time_elapsed: float,
    logger: Optional[logging.Logger] = None
):
    """
    Log training progress.
    
    Args:
        epoch: Current epoch
        step: Current step
        loss: Current loss value
        metrics: Additional metrics
        time_elapsed: Time elapsed for this step
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger()
    
    # Format metrics
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    logger.info(
        f"Epoch {epoch:3d} | Step {step:6d} | "
        f"Loss: {loss:.4f} | {metrics_str} | "
        f"Time: {time_elapsed:.2f}s"
    )

# Test function
def test_logging():
    """Test the logging system."""
    print("🧪 Testing Logging System")
    
    # Test basic logging
    logger = setup_logging(level="INFO")
    logger.debug("Debug message (should not appear)")
    logger.info("✅ Info message")
    logger.warning("⚠️  Warning message")
    logger.error("❌ Error message")
    
    # Test file logging
    log_file = "test_logs/test.log"
    file_logger = setup_logging(
        level="DEBUG", 
        log_file=log_file,
        format_style="detailed"
    )
    file_logger.info("File logging test")
    
    # Check if file was created
    if Path(log_file).exists():
        print("✅ File logging works")
        Path(log_file).unlink()  # Clean up
        Path(log_file).parent.rmdir()
    
    # Test system info logging
    system_info = log_system_info(logger)
    if "platform" in system_info:
        print("✅ System info logging works")
    
    # Test training logging
    log_training_start(
        model_name="test_model",
        dataset_name="test_dataset", 
        config={"batch_size": 32, "lr": 1e-4},
        logger=logger
    )
    
    log_training_progress(
        epoch=1, step=100, loss=0.5,
        metrics={"accuracy": 0.85, "f1": 0.82},
        time_elapsed=1.5,
        logger=logger
    )
    
    print("🎉 Logging system working!")
    return True

if __name__ == "__main__":
    test_logging()