# training_infra/utils/device.py
"""
Device Management System

Smart device detection and management for training infrastructure:
- Automatic GPU/CPU detection with fallbacks
- Multi-GPU support and device selection
- CUDA version compatibility checking
- Memory-aware device selection
- Device context management

## Basic Usage:
```python
from training_infra.utils.device import get_device, set_device

# Auto-detect best device
device = get_device()  # Returns "cuda:0", "cuda", or "cpu"

# Set specific device
set_device("cuda:1")

# Check device info
gpu_info = get_gpu_info()
```

## Advanced Usage:
```python
from training_infra.utils.device import (
    get_device_count,
    is_cuda_available,
    get_best_device,
    DeviceManager
)

# Multi-GPU setup
device_count = get_device_count()
best_device = get_best_device(memory_threshold=0.8)

# Device manager
with DeviceManager("cuda:0") as device:
    # Training code here
    pass
```
"""

import os
import torch
import logging
from typing import Optional, Dict, List, Union, Any
from contextlib import contextmanager

# Global device setting
_current_device = None

def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        bool: True if CUDA is available
    """
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def get_cuda_version() -> Optional[str]:
    """
    Get CUDA version.
    
    Returns:
        str or None: CUDA version if available
    """
    try:
        if is_cuda_available():
            return torch.version.cuda
        return None
    except Exception:
        return None

def get_device_count() -> int:
    """
    Get number of available devices.
    
    Returns:
        int: Number of CUDA devices (0 if CUDA not available)
    """
    try:
        if is_cuda_available():
            return torch.cuda.device_count()
        return 0
    except Exception:
        return 0

def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get detailed information about all GPUs.
    
    Returns:
        list: List of GPU information dictionaries
    """
    gpu_info = []
    
    try:
        if not is_cuda_available():
            return gpu_info
        
        device_count = get_device_count()
        
        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                total_memory = props.total_memory
                try:
                    torch.cuda.set_device(i)
                    allocated_memory = torch.cuda.memory_allocated(i)
                    cached_memory = torch.cuda.memory_reserved(i)
                    free_memory = total_memory - allocated_memory
                except Exception:
                    allocated_memory = 0
                    cached_memory = 0
                    free_memory = total_memory
                
                info = {
                    "device_id": i,
                    "name": props.name,
                    "total_memory_gb": round(total_memory / (1024**3), 2),
                    "allocated_memory_gb": round(allocated_memory / (1024**3), 2),
                    "free_memory_gb": round(free_memory / (1024**3), 2),
                    "cached_memory_gb": round(cached_memory / (1024**3), 2),
                    "memory_usage_percent": round((allocated_memory / total_memory) * 100, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multiprocessor_count,
                }
                
                gpu_info.append(info)
                
            except Exception as e:
                logging.warning(f"Could not get info for GPU {i}: {e}")
                gpu_info.append({
                    "device_id": i,
                    "name": "Unknown",
                    "error": str(e)
                })
        
    except Exception as e:
        logging.warning(f"Could not get GPU info: {e}")
    
    return gpu_info

def get_device(
    preferred: Optional[str] = None,
    fallback_to_cpu: bool = True,
    memory_threshold: float = 0.9
) -> str:
    """
    Get the best available device.
    
    Args:
        preferred: Preferred device ("cuda", "cuda:0", "cpu", etc.)
        fallback_to_cpu: Fall back to CPU if CUDA not available
        memory_threshold: Memory usage threshold for device selection
    
    Returns:
        str: Device string ("cuda:0", "cuda", "cpu")
    
    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda:1")  # Prefer specific GPU
        >>> device = get_device(memory_threshold=0.8)  # Only GPUs with <80% usage
    """
    # Check environment variable first
    if preferred is None:
        preferred = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if preferred is not None:
            if preferred == "":
                # Empty string means no CUDA devices
                return "cpu"
            elif "," not in preferred:
                # Single device
                preferred = f"cuda:{preferred}"
    
    # If preferred device specified, try to use it
    if preferred is not None:
        if preferred == "cpu":
            return "cpu"
        
        if preferred.startswith("cuda"):
            if not is_cuda_available():
                logging.warning("CUDA requested but not available, falling back to CPU")
                return "cpu" if fallback_to_cpu else "cuda"
            
            # Check if specific device exists
            if ":" in preferred:
                device_id = int(preferred.split(":")[1])
                if device_id >= get_device_count():
                    logging.warning(f"GPU {device_id} not available, using cuda:0")
                    return "cuda:0"
            
            return preferred
    
    # Auto-detect best device
    if not is_cuda_available():
        return "cpu"
    
    device_count = get_device_count()
    if device_count == 0:
        return "cpu"
    
    # Single GPU - use it
    if device_count == 1:
        gpu_info = get_gpu_info()
        if gpu_info and len(gpu_info) > 0:
            memory_usage = gpu_info[0].get("memory_usage_percent", 0) / 100
            if memory_usage < memory_threshold:
                return "cuda:0"
            else:
                logging.warning(f"GPU 0 memory usage ({memory_usage:.1%}) exceeds threshold ({memory_threshold:.1%})")
                return "cpu" if fallback_to_cpu else "cuda:0"
        return "cuda:0"
    
    # Multiple GPUs - find best one
    gpu_info = get_gpu_info()
    best_device = None
    lowest_usage = float('inf')
    
    for info in gpu_info:
        if "error" in info:
            continue
        
        memory_usage = info.get("memory_usage_percent", 100) / 100
        
        if memory_usage < memory_threshold and memory_usage < lowest_usage:
            lowest_usage = memory_usage
            best_device = f"cuda:{info['device_id']}"
    
    if best_device is not None:
        return best_device
    
    # No GPU meets threshold, use first one or fallback
    if fallback_to_cpu:
        logging.warning(f"No GPU meets memory threshold ({memory_threshold:.1%}), using CPU")
        return "cpu"
    else:
        logging.warning(f"No GPU meets memory threshold ({memory_threshold:.1%}), using cuda:0 anyway")
        return "cuda:0"

def get_best_device(memory_threshold: float = 0.8) -> str:
    """
    Get the best available device based on memory usage.
    
    Args:
        memory_threshold: Maximum memory usage for a device to be considered
    
    Returns:
        str: Best device string
    """
    return get_device(memory_threshold=memory_threshold)

def set_device(device: str) -> str:
    """
    Set the current device.
    
    Args:
        device: Device string ("cuda:0", "cuda", "cpu")
    
    Returns:
        str: Actual device that was set
    
    Example:
        >>> actual_device = set_device("cuda:1")
        >>> print(f"Using device: {actual_device}")
    """
    global _current_device
    
    # Validate and normalize device
    if device == "cpu":
        _current_device = "cpu"
        return "cpu"
    
    if device.startswith("cuda"):
        if not is_cuda_available():
            logging.warning("CUDA not available, using CPU")
            _current_device = "cpu"
            return "cpu"
        
        # Set CUDA device
        try:
            if ":" in device:
                device_id = int(device.split(":")[1])
                if device_id >= get_device_count():
                    logging.warning(f"GPU {device_id} not available, using cuda:0")
                    device = "cuda:0"
                    device_id = 0
                
                torch.cuda.set_device(device_id)
            else:
                # Just "cuda" - use default
                torch.cuda.set_device(0)
                device = "cuda:0"
            
            _current_device = device
            return device
            
        except Exception as e:
            logging.warning(f"Could not set device {device}: {e}")
            _current_device = "cpu"
            return "cpu"
    
    # Unknown device
    logging.warning(f"Unknown device {device}, using CPU")
    _current_device = "cpu"
    return "cpu"

def get_current_device() -> Optional[str]:
    """
    Get the currently set device.
    
    Returns:
        str or None: Current device string
    """
    return _current_device

@contextmanager
def device_context(device: str):
    """
    Context manager for temporary device setting.
    
    Args:
        device: Device to use in context
    
    Example:
        >>> with device_context("cuda:1"):
        ...     # Code using cuda:1
        ...     pass
        # Automatically restored to previous device
    """
    original_device = get_current_device()
    
    try:
        actual_device = set_device(device)
        yield actual_device
    finally:
        if original_device is not None:
            set_device(original_device)

class DeviceManager:
    """
    Device manager for complex device operations.
    """
    
    def __init__(self, device: str = "auto"):
        self.requested_device = device
        self.actual_device = None
        self.original_device = None
    
    def __enter__(self):
        self.original_device = get_current_device()
        
        if self.requested_device == "auto":
            self.actual_device = get_device()
        else:
            self.actual_device = set_device(self.requested_device)
        
        return self.actual_device
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_device is not None:
            set_device(self.original_device)

def clear_cuda_cache():
    """Clear CUDA memory cache."""
    try:
        if is_cuda_available():
            torch.cuda.empty_cache()
            logging.info("✅ CUDA cache cleared")
    except Exception as e:
        logging.warning(f"Could not clear CUDA cache: {e}")

def get_memory_info(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get memory information for a device.
    
    Args:
        device: Device to check (None for current device)
    
    Returns:
        dict: Memory information in GB
    """
    if device is None:
        device = get_current_device() or get_device()
    
    if device == "cpu":
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent
            }
        except ImportError:
            return {"error": "psutil not available for CPU memory info"}
    
    if device.startswith("cuda"):
        try:
            device_id = 0
            if ":" in device:
                device_id = int(device.split(":")[1])
            
            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            cached = torch.cuda.memory_reserved(device_id)
            free = total - allocated
            
            return {
                "total_gb": round(total / (1024**3), 2),
                "allocated_gb": round(allocated / (1024**3), 2),
                "cached_gb": round(cached / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round((allocated / total) * 100, 1)
            }
        except Exception as e:
            return {"error": f"Could not get CUDA memory info: {e}"}
    
    return {"error": f"Unknown device: {device}"}

# Test function
def test_device_management():
    """Test the device management system."""
    print("🧪 Testing Device Management System")
    
    # Test CUDA availability
    cuda_available = is_cuda_available()
    print(f"✅ CUDA available: {cuda_available}")
    
    if cuda_available:
        cuda_version = get_cuda_version()
        print(f"✅ CUDA version: {cuda_version}")
        
        device_count = get_device_count()
        print(f"✅ Device count: {device_count}")
        
        gpu_info = get_gpu_info()
        print(f"✅ GPU info: {len(gpu_info)} devices")
        for info in gpu_info:
            if "error" not in info:
                print(f"    GPU {info['device_id']}: {info['name']} ({info['total_memory_gb']} GB)")
    
    # Test device selection
    device = get_device()
    print(f"✅ Auto-detected device: {device}")
    
    # Test device setting
    actual_device = set_device(device)
    print(f"✅ Set device: {actual_device}")
    
    # Test memory info
    memory_info = get_memory_info()
    if "error" not in memory_info:
        print(f"✅ Memory info: {memory_info['usage_percent']:.1f}% used")
    
    # Test device context
    with device_context("cpu"):
        current = get_current_device()
        print(f"✅ Device context: {current}")
    
    print("🎉 Device management system working!")
    return True

if __name__ == "__main__":
    test_device_management()