"""
Complete training and distillation pipelines.
"""
from typing import List, Dict, Any, Optional

from .synthetic_distillation import (
    SyntheticDistillationPipeline,
    DomainAdaptiveDistillationPipeline,
    MultiTeacherDistillationPipeline,
    ProductionCompressionPipeline
)

__all__ = [
    "SyntheticDistillationPipeline",
    "DomainAdaptiveDistillationPipeline",
    "MultiTeacherDistillationPipeline", 
    "ProductionCompressionPipeline",
]

# Pipeline registry
PIPELINE_REGISTRY = {
    "synthetic_distillation": SyntheticDistillationPipeline,
    "domain_adaptive": DomainAdaptiveDistillationPipeline,
    "multi_teacher": MultiTeacherDistillationPipeline,
    "production_compression": ProductionCompressionPipeline,
}

def create_pipeline(pipeline_type: str, **kwargs):
    """Create a pipeline by type."""
    if pipeline_type.lower() not in PIPELINE_REGISTRY:
        available = ", ".join(PIPELINE_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline type '{pipeline_type}'. Available: {available}")
    
    pipeline_class = PIPELINE_REGISTRY[pipeline_type.lower()]
    return pipeline_class(**kwargs)

def list_pipelines():
    """List all available pipelines."""
    return list(PIPELINE_REGISTRY.keys())