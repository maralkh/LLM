"""Base model classes for training infrastructure.

This module provides the foundation classes that all models inherit from,
implementing common functionality like save/load, device management, and
memory optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseModel(nn.Module, ABC):
    """Base class for all models in the training infrastructure.
    
    Provides common functionality:
    - Device management with automatic GPU detection
    - Model saving/loading with proper error handling
    - Memory optimization and monitoring
    - Model information and statistics
    - Type safety and validation
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        model_name: str = "base_model"
    ):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        
        # Setup logging first
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        # Device management
        self.device = self._setup_device(device)
        
        # Model metadata
        self.config = {}
        self.training_metadata = {}
        
    def _setup_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Setup device with intelligent defaults."""
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU (CUDA not available)")
        elif isinstance(device, str):
            device = torch.device(device)
            
        return device
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration - must be implemented by subclasses."""
        pass
        
    def save_model(
        self, 
        path: Union[str, Path], 
        include_optimizer: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """Save model with comprehensive metadata.
        
        Args:
            path: Path to save the model
            include_optimizer: Whether to save optimizer state
            optimizer: Optimizer to save (if include_optimizer=True)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save dictionary
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
            'model_name': self.model_name,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'training_metadata': self.training_metadata,
            'model_info': self.get_model_info()
        }
        
        # Include optimizer if requested
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['optimizer_type'] = type(optimizer).__name__
            
        try:
            torch.save(save_dict, path)
            self.logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
            
    @classmethod
    def load_model(
        cls, 
        path: Union[str, Path], 
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True
    ) -> Tuple['BaseModel', Optional[Dict[str, Any]]]:
        """Load model from checkpoint.
        
        Args:
            path: Path to model checkpoint
            device: Device to load model on
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            Tuple of (model, optimizer_state_dict)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
            
        try:
            checkpoint = torch.load(path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
            
        # Extract model info
        model_config = checkpoint.get('model_config', {})
        model_name = checkpoint.get('model_name', 'loaded_model')
        
        # Create model instance (subclass should override this)
        model = cls._from_config(model_config, device=device, model_name=model_name)
        
        # Load state dict
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load model state: {e}")
            else:
                warnings.warn(f"Some parameters could not be loaded: {e}")
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
        # Restore metadata
        model.training_metadata = checkpoint.get('training_metadata', {})
        model.to(model.device)
        
        # Get optimizer state if available
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        
        logging.getLogger(__name__).info(f"Model loaded successfully from {path}")
        return model, optimizer_state
        
    @classmethod
    def _from_config(
        cls, 
        config: Dict[str, Any], 
        device: Optional[Union[str, torch.device]] = None,
        model_name: str = "loaded_model"
    ) -> 'BaseModel':
        """Create model from config - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _from_config")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'memory_usage_mb': self.get_memory_usage(),
        }
        
        return info
        
    def get_memory_usage(self) -> float:
        """Get model memory usage in MB."""
        total_memory = 0
        for param in self.parameters():
            total_memory += param.numel() * param.element_size()
        return total_memory / (1024 * 1024)  # Convert to MB
        
    def to_device(self, device: Optional[Union[str, torch.device]] = None) -> 'BaseModel':
        """Move model to device with proper error handling."""
        if device is not None:
            self.device = torch.device(device)
            
        try:
            self.to(self.device)
            self.logger.info(f"Model moved to {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to move model to {self.device}: {e}")
            raise
            
        return self
        
    def print_model_info(self) -> None:
        """Print comprehensive model information."""
        info = self.get_model_info()
        print(f"\n📊 Model Information: {info['model_name']}")
        print(f"   Total Parameters: {info['total_parameters']:,}")
        print(f"   Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"   Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f"   Device: {info['device']}")
        print(f"   Data Type: {info['dtype']}")
        
    def set_training_metadata(self, key: str, value: Any) -> None:
        """Set training metadata."""
        self.training_metadata[key] = value
        
    def get_training_metadata(self, key: str, default: Any = None) -> Any:
        """Get training metadata."""
        return self.training_metadata.get(key, default)


class LanguageModel(BaseModel):
    """Base class for language models.
    
    Extends BaseModel with language-specific functionality:
    - Text generation capabilities
    - Tokenization handling
    - Language model metrics
    """
    
    def __init__(
        self,
        vocab_size: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        model_name: str = "language_model"
    ):
        super().__init__(device=device, dtype=dtype, model_name=model_name)
        self.vocab_size = vocab_size
        
    @abstractmethod
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text - must be implemented by subclasses."""
        pass
        
    def compute_perplexity(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Compute perplexity on given input."""
        with torch.no_grad():
            logits = self.forward(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                target_ids.view(-1), 
                ignore_index=-100
            )
            return torch.exp(loss).item()


# Testing functions for development
def test_base_model():
    """Test BaseModel functionality."""
    print("🧪 Testing BaseModel...")
    
    # Create a simple test model
    class TestModel(BaseModel):
        def __init__(self, hidden_size: int = 256):
            super().__init__(model_name="test_model")
            self.hidden_size = hidden_size
            self.linear = nn.Linear(hidden_size, hidden_size)
            
        def forward(self, x):
            return self.linear(x)
            
        def get_config(self):
            return {"hidden_size": self.hidden_size}
            
        @classmethod
        def _from_config(cls, config, device=None, model_name="loaded_model"):
            model = cls(config["hidden_size"])
            if device:
                model.device = torch.device(device)
            model.model_name = model_name
            return model
    
    # Test model creation
    model = TestModel(hidden_size=512)
    model.print_model_info()
    
    # Test forward pass
    x = torch.randn(2, 512)
    output = model(x)
    assert output.shape == (2, 512), f"Expected (2, 512), got {output.shape}"
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model.save_model(f.name)
        loaded_model, _ = TestModel.load_model(f.name)
        
    # Verify loaded model
    loaded_output = loaded_model(x)
    assert torch.allclose(output, loaded_output), "Loaded model produces different output"
    
    print("✅ BaseModel tests passed!")


if __name__ == "__main__":
    test_base_model()