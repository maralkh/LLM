# training_infra/training/utils/__init__.py
"""
Training utilities module.

Provides setup utilities for training components.
"""

from .base import (
    TrainingState,
    BaseTrainer,
    test_base_trainer,
)

__all__ = [
    'TrainingState',
    'BaseTrainer',
    'test_base_trainer',
]

def get_utils_info():
    """Get information about available utilities."""
    return {
        #"setup_utils_available": _SETUP_UTILS_AVAILABLE,
        "total_exports": len(__all__),
    }

if __name__ == "__main__":
    info = get_utils_info()
    print(f"Training utils: {info}")