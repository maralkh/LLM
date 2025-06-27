"""Models module for training infrastructure.

This module provides various model implementations including
base classes, LLaMA variants, and utility functions.
"""

from .base import BaseModel, LanguageModel

# Import configs first (no dependencies)
try:
    from .llama import LLaMAConfig, LLaMAVariants, get_model_info, print_model_comparison
except ImportError:
    # If llama module has issues, continue without it
    pass

# Import utils if available (optional)
try:
    from .utils import (
        WeightInitializer,
        init_linear_layer,
        apply_initialization
    )
except ImportError:
    # Utils not yet available or has dependency issues
    pass

__all__ = [
    'BaseModel',
    'LanguageModel'
]

# Add llama exports if available
try:
    from .llama import LLaMAConfig
    __all__.extend(['LLaMAConfig', 'LLaMAVariants', 'get_model_info', 'print_model_comparison'])
except ImportError:
    pass