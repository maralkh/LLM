"""Model utilities module.

This module provides utilities for model operations including
initialization, activations, cache management, and memory optimization.
"""

from .initialization import (
    WeightInitializer,
    init_linear_layer,
    init_embedding_layer,
    init_attention_layer,
    init_layernorm,
    apply_initialization,
    get_initialization_info,
    print_initialization_summary,
    test_initialization
)

from .cache import (
    BaseCache,
    DynamicCache,
    StaticCache,
    SlidingWindowCache,
    QuantizedCache,
    create_cache,
    test_cache_implementations
)

from .setup import (
    setup_device,
    extract_parallelism_config,
    setup_parallelism,
    setup_distributed,
    setup_optimizer,
    setup_scheduler,
    setup_mixed_precision,
    setup_data_loaders,
    count_parameters,
    print_training_setup,
)

__all__ = [
    # Initialization utilities
    'WeightInitializer',
    'init_linear_layer',
    'init_embedding_layer', 
    'init_attention_layer',
    'init_layernorm',
    'apply_initialization',
    'get_initialization_info',
    'print_initialization_summary',
    'test_initialization',
    
    # Cache utilities
    'BaseCache',
    'DynamicCache',
    'StaticCache', 
    'SlidingWindowCache',
    'QuantizedCache',
    'create_cache',
    'test_cache_implementations',

    # trainer setup
    "setup_device",
    "extract_parallelism_config", 
    "setup_parallelism",
    "setup_distributed",
    "setup_optimizer",
    "setup_scheduler",
    "setup_mixed_precision",
    "setup_data_loaders",
    "count_parameters",
    "print_training_setup",
]