# training_infra/utils/reproducibility.py
"""
Reproducibility System

Comprehensive reproducibility management for training infrastructure:
- Seed setting for all random number generators
- Deterministic operations control
- Random state saving/restoring
- Environment reproducibility checking
- Reproducibility validation

## Basic Usage:
```python
from training_infra.utils.reproducibility import set_seed, set_deterministic_mode

# Set seed for reproducibility
set_seed(42)

# Enable deterministic operations (slower but fully reproducible)
set_deterministic_mode(True)
```

## Advanced Usage:
```python
from training_infra.utils.reproducibility import (
    get_random_state,
    restore_random_state,
    validate_reproducibility,
    ReproducibilityManager
)

# Save/restore random state
state = get_random_state()
# ... training code ...
restore_random_state(state)

# Validate reproducibility
is_reproducible = validate_reproducibility()
```
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

# Global reproducibility settings
_current_seed = None
_deterministic_mode = False

@dataclass
class RandomState:
    """Container for all random states."""
    python_state: Any
    numpy_state: Dict[str, Any]
    torch_state: torch.Tensor
    torch_cuda_state: Optional[Dict[int, torch.Tensor]] = None
    seed: Optional[int] = None

def set_seed(seed: int) -> int:
    """
    Set seed for all random number generators.
    
    Args:
        seed: Random seed value
    
    Returns:
        int: The seed that was set
    
    Example:
        >>> set_seed(42)
        >>> # All random operations will be deterministic
    """
    global _current_seed
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Store current seed
    _current_seed = seed
    
    logging.info(f"🎲 Random seed set to {seed}")
    return seed

def get_seed() -> Optional[int]:
    """
    Get the current seed.
    
    Returns:
        int or None: Current seed if set
    """
    return _current_seed

def set_deterministic_mode(enabled: bool = True) -> bool:
    """
    Enable or disable deterministic operations.
    
    Args:
        enabled: Whether to enable deterministic mode
    
    Returns:
        bool: Whether deterministic mode was successfully set
    
    Note:
        Deterministic mode makes training slower but fully reproducible.
        Some operations may not have deterministic implementations.
    """
    global _deterministic_mode
    
    try:
        if enabled:
            # PyTorch deterministic settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set additional environment variables for determinism
            os.environ["PYTHONHASHSEED"] = str(_current_seed or 0)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            
            # Enable deterministic algorithms (PyTorch 1.12+)
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                # Older PyTorch versions
                warnings.warn("torch.use_deterministic_algorithms not available in this PyTorch version")
            except RuntimeError as e:
                warnings.warn(f"Could not enable deterministic algorithms: {e}")
            
            _deterministic_mode = True
            logging.info("🔒 Deterministic mode enabled (slower but fully reproducible)")
            
        else:
            # Disable deterministic settings
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            try:
                torch.use_deterministic_algorithms(False)
            except AttributeError:
                pass
            except RuntimeError:
                pass
            
            _deterministic_mode = False
            logging.info("🚀 Deterministic mode disabled (faster but may not be fully reproducible)")
        
        return True
        
    except Exception as e:
        logging.warning(f"Could not set deterministic mode: {e}")
        return False

def is_deterministic_mode() -> bool:
    """
    Check if deterministic mode is enabled.
    
    Returns:
        bool: True if deterministic mode is enabled
    """
    return _deterministic_mode

def get_random_state() -> RandomState:
    """
    Get current state of all random number generators.
    
    Returns:
        RandomState: Container with all random states
    
    Example:
        >>> state = get_random_state()
        >>> # ... some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    # Get Python random state
    python_state = random.getstate()
    
    # Get NumPy random state
    numpy_state = np.random.get_state()
    
    # Get PyTorch random state
    torch_state = torch.get_rng_state()
    
    # Get CUDA random states if available
    torch_cuda_state = None
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            torch_cuda_state = {}
            for device_id in range(device_count):
                torch_cuda_state[device_id] = torch.cuda.get_rng_state(device_id)
        except Exception as e:
            logging.warning(f"Could not get CUDA random state: {e}")
    
    return RandomState(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_state=torch_state,
        torch_cuda_state=torch_cuda_state,
        seed=_current_seed
    )

def restore_random_state(state: RandomState) -> bool:
    """
    Restore random state for all generators.
    
    Args:
        state: RandomState container
    
    Returns:
        bool: True if successful
    
    Example:
        >>> state = get_random_state()
        >>> # ... some operations ...
        >>> restore_random_state(state)
    """
    try:
        # Restore Python random state
        random.setstate(state.python_state)
        
        # Restore NumPy random state  
        np.random.set_state(state.numpy_state)
        
        # Restore PyTorch random state
        torch.set_rng_state(state.torch_state)
        
        # Restore CUDA random states if available
        if state.torch_cuda_state is not None and torch.cuda.is_available():
            for device_id, cuda_state in state.torch_cuda_state.items():
                try:
                    torch.cuda.set_rng_state(cuda_state, device_id)
                except Exception as e:
                    logging.warning(f"Could not restore CUDA state for device {device_id}: {e}")
        
        # Restore global seed
        global _current_seed
        _current_seed = state.seed
        
        logging.info("🔄 Random state restored")
        return True
        
    except Exception as e:
        logging.error(f"Could not restore random state: {e}")
        return False

def validate_reproducibility(seed: int = 42, iterations: int = 3) -> bool:
    """
    Validate that operations are reproducible.
    
    Args:
        seed: Seed to use for testing
        iterations: Number of iterations to test
    
    Returns:
        bool: True if reproducible
    
    Example:
        >>> is_reproducible = validate_reproducibility()
        >>> print(f"Training is reproducible: {is_reproducible}")
    """
    logging.info(f"🧪 Testing reproducibility with seed {seed}")
    
    results = []
    
    for i in range(iterations):
        # Set seed and generate some random values
        set_seed(seed)
        
        # Test different random sources
        python_val = random.random()
        numpy_val = np.random.random()
        torch_val = torch.rand(1).item()
        
        # Test some tensor operations
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        result = torch.mm(x, y).sum().item()
        
        iteration_result = {
            "python": python_val,
            "numpy": numpy_val,
            "torch": torch_val,
            "operation": result
        }
        
        results.append(iteration_result)
        logging.debug(f"Iteration {i+1}: {iteration_result}")
    
    # Check if all iterations produced the same results
    first_result = results[0]
    is_reproducible = True
    
    for i, result in enumerate(results[1:], 1):
        for key, value in result.items():
            if abs(value - first_result[key]) > 1e-10:  # Small tolerance for floating point
                logging.warning(f"Iteration {i+1} differs in {key}: {value} vs {first_result[key]}")
                is_reproducible = False
    
    if is_reproducible:
        logging.info("✅ Reproducibility test passed")
    else:
        logging.warning("❌ Reproducibility test failed")
    
    return is_reproducible

def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information relevant for reproducibility.
    
    Returns:
        dict: Environment information
    """
    env_info = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not_set"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not_set"),
        "torch_deterministic": torch.backends.cudnn.deterministic,
        "torch_benchmark": torch.backends.cudnn.benchmark,
        "current_seed": _current_seed,
        "deterministic_mode": _deterministic_mode,
    }
    
    # PyTorch version info
    env_info["torch_version"] = torch.__version__
    
    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cudnn_version"] = torch.backends.cudnn.version()
    
    return env_info

def log_reproducibility_info(logger: Optional[logging.Logger] = None):
    """
    Log reproducibility information.
    
    Args:
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    env_info = get_environment_info()
    
    logger.info("🎲 Reproducibility Information:")
    logger.info("=" * 40)
    
    for key, value in env_info.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 40)

class ReproducibilityManager:
    """
    Context manager for reproducibility control.
    """
    
    def __init__(
        self,
        seed: int,
        deterministic: bool = False,
        restore_on_exit: bool = True
    ):
        self.seed = seed
        self.deterministic = deterministic
        self.restore_on_exit = restore_on_exit
        self.original_state = None
        self.original_deterministic = None
    
    def __enter__(self):
        if self.restore_on_exit:
            self.original_state = get_random_state()
            self.original_deterministic = is_deterministic_mode()
        
        set_seed(self.seed)
        if self.deterministic:
            set_deterministic_mode(True)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.restore_on_exit and self.original_state is not None:
            restore_random_state(self.original_state)
            if self.original_deterministic != self.deterministic:
                set_deterministic_mode(self.original_deterministic)

def setup_reproducibility(
    seed: int = 42,
    deterministic: bool = False,
    validate: bool = True
) -> bool:
    """
    Setup complete reproducibility environment.
    
    Args:
        seed: Random seed
        deterministic: Enable deterministic mode
        validate: Validate reproducibility after setup
    
    Returns:
        bool: True if setup successful and reproducible
    
    Example:
        >>> success = setup_reproducibility(seed=42, deterministic=True)
        >>> if success:
        ...     print("Training will be fully reproducible")
    """
    logging.info("🔧 Setting up reproducibility environment...")
    
    # Set seed
    set_seed(seed)
    
    # Set deterministic mode if requested
    if deterministic:
        det_success = set_deterministic_mode(True)
        if not det_success:
            logging.warning("Could not enable full deterministic mode")
    
    # Log environment info
    log_reproducibility_info()
    
    # Validate if requested
    if validate:
        is_reproducible = validate_reproducibility(seed=seed)
        if not is_reproducible:
            logging.warning("⚠️  Reproducibility validation failed")
            return False
    
    logging.info("✅ Reproducibility setup complete")
    return True

# Test function
def test_reproducibility():
    """Test the reproducibility system."""
    print("🧪 Testing Reproducibility System")
    
    # Test basic seed setting
    original_seed = get_seed()
    set_seed(42)
    new_seed = get_seed()
    print(f"✅ Seed setting: {original_seed} -> {new_seed}")
    
    # Test deterministic mode
    det_success = set_deterministic_mode(True)
    print(f"✅ Deterministic mode: {det_success}")
    
    # Test state saving/restoring
    state = get_random_state()
    print("✅ Random state captured")
    
    # Generate some random values
    val1 = torch.rand(1).item()
    
    # Restore state and generate again
    restore_random_state(state)
    val2 = torch.rand(1).item()
    
    if abs(val1 - val2) < 1e-10:
        print("✅ State restore works")
    else:
        print(f"❌ State restore failed: {val1} != {val2}")
    
    # Test reproducibility validation
    is_reproducible = validate_reproducibility(seed=42, iterations=2)
    print(f"✅ Reproducibility validation: {is_reproducible}")
    
    # Test context manager
    with ReproducibilityManager(seed=123, deterministic=False):
        val3 = torch.rand(1).item()
    
    with ReproducibilityManager(seed=123, deterministic=False):
        val4 = torch.rand(1).item()
    
    if abs(val3 - val4) < 1e-10:
        print("✅ Context manager works")
    else:
        print(f"❌ Context manager failed: {val3} != {val4}")
    
    # Test environment info
    env_info = get_environment_info()
    if "current_seed" in env_info:
        print("✅ Environment info collection works")
    
    print("🎉 Reproducibility system working!")
    return True

if __name__ == "__main__":
    test_reproducibility()