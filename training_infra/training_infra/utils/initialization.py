"""Weight initialization utilities for neural networks.

This module provides various initialization strategies optimized for different
model architectures and training scenarios. Includes modern techniques like
Xavier/Glorot, He initialization, and transformer-specific methods.
"""

import math
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import Parameter

# Use standard Python logging instead of our custom logging module
import logging as python_logging

logger = python_logging.getLogger(__name__)


class WeightInitializer:
    """Comprehensive weight initialization utility.
    
    Supports multiple initialization strategies:
    - Xavier/Glorot (uniform and normal)
    - He/Kaiming (uniform and normal) 
    - Transformer-specific initialization
    - Custom initialization functions
    """
    
    @staticmethod
    def xavier_uniform(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Xavier/Glorot uniform initialization.
        
        Good for: tanh, sigmoid activations
        Formula: U(-sqrt(6/(fan_in + fan_out)) * gain, sqrt(6/(fan_in + fan_out)) * gain)
        """
        num_input = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        num_output = tensor.size(0)
        
        bound = gain * math.sqrt(6.0 / (num_input + num_output))
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
        return tensor
    
    @staticmethod
    def xavier_normal(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Xavier/Glorot normal initialization.
        
        Good for: tanh, sigmoid activations
        Formula: N(0, gain^2 * 2/(fan_in + fan_out))
        """
        num_input = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        num_output = tensor.size(0)
        
        std = gain * math.sqrt(2.0 / (num_input + num_output))
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def he_uniform(tensor: torch.Tensor, nonlinearity: str = 'relu') -> torch.Tensor:
        """He/Kaiming uniform initialization.
        
        Good for: ReLU and variants
        Formula: U(-bound, bound) where bound = sqrt(6/fan_in) * gain
        """
        gain = nn.init.calculate_gain(nonlinearity)
        fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
        
        bound = gain * math.sqrt(6.0 / fan_in)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
        return tensor
    
    @staticmethod
    def he_normal(tensor: torch.Tensor, nonlinearity: str = 'relu') -> torch.Tensor:
        """He/Kaiming normal initialization.
        
        Good for: ReLU and variants
        Formula: N(0, gain^2 * 2/fan_in)
        """
        gain = nn.init.calculate_gain(nonlinearity)
        fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
        
        std = gain * math.sqrt(2.0 / fan_in)
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def transformer_init(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        """Transformer-specific initialization.
        
        Used in BERT, GPT, and similar models.
        Formula: N(0, std^2) where std is typically 0.02
        """
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def llama_init(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """LLaMA-specific initialization.
        
        Based on the scaling used in LLaMA models.
        Formula: N(0, 1/sqrt(dim))
        """
        std = 1.0 / math.sqrt(dim)
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def mu_transfer_init(
        tensor: torch.Tensor, 
        width: int, 
        layer_type: str = 'linear',
        fan_mode: str = 'fan_in'
    ) -> torch.Tensor:
        """μTransfer (Mu Transfer) initialization.
        
        Modern initialization that enables stable transfer from small to large models.
        Based on "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
        
        Args:
            tensor: Tensor to initialize
            width: Model width (hidden dimension)
            layer_type: 'linear', 'attention_qk', 'attention_v', 'output'
            fan_mode: 'fan_in' or 'fan_out'
        """
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if fan_mode == 'fan_in' else fan_out
        
        if layer_type == 'linear':
            # Standard layers: N(0, 1/fan_in)
            std = 1.0 / math.sqrt(fan)
        elif layer_type == 'attention_qk':
            # Query/Key projections: N(0, 1/width)
            std = 1.0 / math.sqrt(width)
        elif layer_type == 'attention_v':
            # Value projection: N(0, 1/width)
            std = 1.0 / math.sqrt(width)
        elif layer_type == 'output':
            # Output projections: N(0, 1/(width * fan_in))
            std = 1.0 / math.sqrt(width * fan)
        else:
            raise ValueError(f"Unknown μTransfer layer type: {layer_type}")
            
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def deepnorm_init(tensor: torch.Tensor, num_layers: int, alpha: float = None) -> torch.Tensor:
        """DeepNorm initialization for very deep transformers.
        
        From "DeepNet: Scaling Transformers to 1,000 Layers"
        Enables stable training of extremely deep networks (1000+ layers).
        
        Args:
            tensor: Tensor to initialize
            num_layers: Total number of transformer layers
            alpha: Scaling factor (auto-computed if None)
        """
        if alpha is None:
            # Standard DeepNorm scaling
            alpha = (2 * num_layers) ** 0.25
            
        std = (2.0 / (tensor.size(0) + tensor.size(1))) ** 0.5 / alpha
        
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def rms_init(tensor: torch.Tensor, target_rms: float = 1.0) -> torch.Tensor:
        """RMS-based initialization.
        
        Initialize weights to have specific RMS (Root Mean Square) value.
        Useful for controlling activation magnitudes.
        """
        with torch.no_grad():
            tensor.normal_(0, 1.0)
            current_rms = tensor.norm() / math.sqrt(tensor.numel())
            tensor.mul_(target_rms / current_rms)
        return tensor
    
    @staticmethod
    def nqm_init(tensor: torch.Tensor, mode: str = 'geometric') -> torch.Tensor:
        """Neural Quantum Mean (NQM) initialization.
        
        Modern initialization that considers quantum-inspired scaling.
        
        Args:
            tensor: Tensor to initialize
            mode: 'geometric' or 'arithmetic' mean scaling
        """
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        
        if mode == 'geometric':
            effective_fan = math.sqrt(fan_in * fan_out)
        elif mode == 'arithmetic':
            effective_fan = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"Unknown NQM mode: {mode}")
            
        std = math.sqrt(2.0 / effective_fan)
        
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def fixup_init(tensor: torch.Tensor, num_layers: int, layer_id: int) -> torch.Tensor:
        """Fixup initialization for residual networks.
        
        From "Fixup Initialization: Residual Learning Without Normalization"
        Enables training deep residual networks without normalization layers.
        """
        # Scale by layer depth for residual connections
        scale = num_layers ** (-1.0 / (2 * (layer_id + 1)))
        
        fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
        std = math.sqrt(2.0 / fan_in) * scale
        
        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor
    
    @staticmethod
    def rezero_init(tensor: torch.Tensor) -> torch.Tensor:
        """ReZero initialization.
        
        From "ReZero is All You Need: Fast Convergence at Large Depth"
        Initialize most weights normally, but residual scaling factors to zero.
        """
        # For most layers, use standard initialization
        nn.init.xavier_normal_(tensor)
        return tensor
    
    @staticmethod
    def t5_init(tensor: torch.Tensor, d_model: int) -> torch.Tensor:
        """T5-style initialization.
        
        From "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
        Uses factor scaling based on model dimension.
        """
        factor = 1.0
        if tensor.ndim > 1:
            factor = 1.0 / math.sqrt(d_model)
            
        with torch.no_grad():
            tensor.normal_(0, factor)
        return tensor


def init_linear_layer(
    layer: nn.Linear,
    method: str = 'xavier_normal',
    bias_init: str = 'zero',
    **kwargs
) -> None:
    """Initialize a linear layer with specified method.
    
    Args:
        layer: Linear layer to initialize
        method: Initialization method - Options:
            Classical: 'xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform'
            Transformer: 'transformer', 'llama', 't5'
            Modern: 'mu_transfer', 'deepnorm', 'fixup', 'nqm', 'rms'
        bias_init: Bias initialization ('zero', 'normal', 'uniform')
        **kwargs: Additional arguments for initialization methods
    """
    # Initialize weights
    if method == 'xavier_normal':
        gain = kwargs.get('gain', 1.0)
        WeightInitializer.xavier_normal(layer.weight, gain)
    elif method == 'xavier_uniform':
        gain = kwargs.get('gain', 1.0)
        WeightInitializer.xavier_uniform(layer.weight, gain)
    elif method == 'he_normal':
        nonlinearity = kwargs.get('nonlinearity', 'relu')
        WeightInitializer.he_normal(layer.weight, nonlinearity)
    elif method == 'he_uniform':
        nonlinearity = kwargs.get('nonlinearity', 'relu')
        WeightInitializer.he_uniform(layer.weight, nonlinearity)
    elif method == 'transformer':
        std = kwargs.get('std', 0.02)
        WeightInitializer.transformer_init(layer.weight, std)
    elif method == 'llama':
        dim = kwargs.get('dim', layer.weight.size(1))
        WeightInitializer.llama_init(layer.weight, dim)
    elif method == 't5':
        d_model = kwargs.get('d_model', layer.weight.size(1))
        WeightInitializer.t5_init(layer.weight, d_model)
    elif method == 'mu_transfer':
        width = kwargs.get('width', layer.weight.size(1))
        layer_type = kwargs.get('layer_type', 'linear')
        fan_mode = kwargs.get('fan_mode', 'fan_in')
        WeightInitializer.mu_transfer_init(layer.weight, width, layer_type, fan_mode)
    elif method == 'deepnorm':
        num_layers = kwargs.get('num_layers', 12)
        alpha = kwargs.get('alpha', None)
        WeightInitializer.deepnorm_init(layer.weight, num_layers, alpha)
    elif method == 'fixup':
        num_layers = kwargs.get('num_layers', 12)
        layer_id = kwargs.get('layer_id', 0)
        WeightInitializer.fixup_init(layer.weight, num_layers, layer_id)
    elif method == 'nqm':
        mode = kwargs.get('mode', 'geometric')
        WeightInitializer.nqm_init(layer.weight, mode)
    elif method == 'rms':
        target_rms = kwargs.get('target_rms', 1.0)
        WeightInitializer.rms_init(layer.weight, target_rms)
    elif method == 'rezero':
        WeightInitializer.rezero_init(layer.weight)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    # Initialize bias
    if layer.bias is not None:
        if bias_init == 'zero':
            nn.init.zeros_(layer.bias)
        elif bias_init == 'normal':
            std = kwargs.get('bias_std', 0.01)
            nn.init.normal_(layer.bias, 0, std)
        elif bias_init == 'uniform':
            bound = kwargs.get('bias_bound', 0.01)
            nn.init.uniform_(layer.bias, -bound, bound)


def init_embedding_layer(
    layer: nn.Embedding,
    method: str = 'normal',
    std: float = 0.02
) -> None:
    """Initialize an embedding layer.
    
    Args:
        layer: Embedding layer to initialize
        method: Initialization method ('normal', 'uniform', 'xavier')
        std: Standard deviation for normal initialization
    """
    if method == 'normal':
        nn.init.normal_(layer.weight, 0, std)
    elif method == 'uniform':
        bound = std * math.sqrt(3)  # Keep same variance as normal
        nn.init.uniform_(layer.weight, -bound, bound)
    elif method == 'xavier':
        nn.init.xavier_normal_(layer.weight)
    else:
        raise ValueError(f"Unknown embedding initialization method: {method}")


def init_attention_layer(
    query: nn.Linear,
    key: nn.Linear,
    value: nn.Linear,
    output: nn.Linear,
    method: str = 'transformer',
    **kwargs
) -> None:
    """Initialize attention layers with proper scaling.
    
    Args:
        query, key, value, output: Attention projection layers
        method: Initialization method
        **kwargs: Additional arguments
    """
    # Standard initialization for Q, K, V
    for layer in [query, key, value]:
        init_linear_layer(layer, method, **kwargs)
    
    # Output projection often needs special scaling
    if method == 'transformer':
        # Scale down output projection for stability
        std = kwargs.get('std', 0.02)
        num_layers = kwargs.get('num_layers', 1)
        output_std = std / math.sqrt(2 * num_layers)
        WeightInitializer.transformer_init(output.weight, output_std)
        if output.bias is not None:
            nn.init.zeros_(output.bias)
    else:
        init_linear_layer(output, method, **kwargs)


def init_layernorm(layer: nn.LayerNorm) -> None:
    """Initialize LayerNorm with standard values.
    
    Args:
        layer: LayerNorm layer to initialize
    """
    nn.init.ones_(layer.weight)
    nn.init.zeros_(layer.bias)


def apply_initialization(
    model: nn.Module,
    method: str = 'transformer',
    **kwargs
) -> None:
    """Apply initialization to entire model recursively.
    
    Args:
        model: Model to initialize
        method: Default initialization method
        **kwargs: Additional arguments for initialization
    """
    logger.info(f"Initializing model with method: {method}")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            init_linear_layer(module, method, **kwargs)
            logger.debug(f"Initialized Linear layer: {name}")
            
        elif isinstance(module, nn.Embedding):
            embedding_method = kwargs.get('embedding_method', 'normal')
            embedding_std = kwargs.get('embedding_std', 0.02)
            init_embedding_layer(module, embedding_method, embedding_std)
            logger.debug(f"Initialized Embedding layer: {name}")
            
        elif isinstance(module, nn.LayerNorm):
            init_layernorm(module)
            logger.debug(f"Initialized LayerNorm layer: {name}")
            
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Convolutional layers
            if method in ['he_normal', 'he_uniform']:
                if 'he_normal' in method:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out')
                else:
                    nn.init.kaiming_uniform_(module.weight, mode='fan_out')
            else:
                nn.init.xavier_normal_(module.weight)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            logger.debug(f"Initialized Conv layer: {name}")


def get_initialization_info(model: nn.Module) -> dict:
    """Get information about model parameter initialization.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with initialization statistics
    """
    stats = {
        'total_parameters': 0,
        'layer_types': {},
        'parameter_stats': {}
    }
    
    for name, param in model.named_parameters():
        stats['total_parameters'] += param.numel()
        
        # Get layer type
        layer_type = name.split('.')[-1]  # Get last part (weight/bias)
        module_type = type(param).__name__
        
        if layer_type not in stats['layer_types']:
            stats['layer_types'][layer_type] = 0
        stats['layer_types'][layer_type] += 1
        
        # Parameter statistics
        with torch.no_grad():
            param_stats = {
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'shape': list(param.shape)
            }
            stats['parameter_stats'][name] = param_stats
    
    return stats


def print_initialization_summary(model: nn.Module) -> None:
    """Print summary of model initialization.
    
    Args:
        model: Model to summarize
    """
    stats = get_initialization_info(model)
    
    print(f"\n🔧 Model Initialization Summary")
    print(f"   Total Parameters: {stats['total_parameters']:,}")
    print(f"   Layer Types: {dict(stats['layer_types'])}")
    
    # Print parameter ranges
    print(f"\n📊 Parameter Statistics:")
    for name, param_stats in list(stats['parameter_stats'].items())[:5]:  # Show first 5
        print(f"   {name}: mean={param_stats['mean']:.6f}, std={param_stats['std']:.6f}")
    
    if len(stats['parameter_stats']) > 5:
        print(f"   ... and {len(stats['parameter_stats']) - 5} more layers")


# Testing function
def test_initialization():
    """Test initialization methods."""
    print("🧪 Testing Weight Initialization...")
    
    # Test linear layer initialization
    linear = nn.Linear(512, 256)
    
    # Test different methods
    methods = [
        'xavier_normal', 'he_normal', 'transformer', 'llama',
        'mu_transfer', 'deepnorm', 't5', 'nqm', 'rms'
    ]
    
    print("📊 Initialization Method Comparison:")
    for method in methods:
        linear_copy = nn.Linear(512, 256)
        
        # Set method-specific kwargs
        if method == 'mu_transfer':
            init_linear_layer(linear_copy, method, width=512, layer_type='linear')
        elif method == 'deepnorm':
            init_linear_layer(linear_copy, method, num_layers=24)
        elif method == 't5':
            init_linear_layer(linear_copy, method, d_model=512)
        elif method == 'rms':
            init_linear_layer(linear_copy, method, target_rms=0.8)
        else:
            init_linear_layer(linear_copy, method, dim=512)
        
        weight_std = linear_copy.weight.std().item()
        weight_mean = linear_copy.weight.mean().item()
        print(f"   {method:12}: mean={weight_mean:+.6f}, std={weight_std:.6f}")
    
    # Test modern initialization for attention layers
    print("\n🔍 Testing Modern Attention Initialization:")
    q_proj = nn.Linear(512, 512)
    k_proj = nn.Linear(512, 512)
    v_proj = nn.Linear(512, 512)
    o_proj = nn.Linear(512, 512)
    
    # μTransfer for attention
    init_linear_layer(q_proj, 'mu_transfer', width=512, layer_type='attention_qk')
    init_linear_layer(k_proj, 'mu_transfer', width=512, layer_type='attention_qk')
    init_linear_layer(v_proj, 'mu_transfer', width=512, layer_type='attention_v')
    init_linear_layer(o_proj, 'mu_transfer', width=512, layer_type='output')
    
    print(f"   Q projection std: {q_proj.weight.std().item():.6f}")
    print(f"   K projection std: {k_proj.weight.std().item():.6f}")
    print(f"   V projection std: {v_proj.weight.std().item():.6f}")
    print(f"   O projection std: {o_proj.weight.std().item():.6f}")
    
    # Test full model initialization
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 512)
            self.linear1 = nn.Linear(512, 1024)
            self.layernorm = nn.LayerNorm(1024)
            self.linear2 = nn.Linear(1024, 512)
    
    print("\n🏗️ Testing Full Model Initialization:")
    
    # Test different initialization strategies
    strategies = ['transformer', 'mu_transfer', 'deepnorm']
    
    for strategy in strategies:
        model = TestModel()
        if strategy == 'mu_transfer':
            apply_initialization(model, method=strategy, width=512)
        elif strategy == 'deepnorm':
            apply_initialization(model, method=strategy, num_layers=12)
        else:
            apply_initialization(model, method=strategy)
        
        print(f"\n   Strategy: {strategy}")
        print_initialization_summary(model)
    
    print("\n✅ All initialization tests passed!")


if __name__ == "__main__":
    test_initialization()