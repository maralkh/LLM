"""KV Cache utilities for efficient inference.

This module provides various caching strategies for key-value states
in transformer models, optimizing memory usage and inference speed.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings

import torch
import torch.nn as nn


class BaseCache(ABC):
    """Base class for all cache implementations.
    
    Provides common interface for different caching strategies.
    """
    
    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key and value states."""
        pass
        
    @abstractmethod
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length from cache."""
        pass
        
    @abstractmethod
    def reset(self):
        """Reset cache to empty state."""
        pass
        
    @abstractmethod
    def get_cache_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        pass


class DynamicCache(BaseCache):
    """Dynamic KV cache for efficient inference.
    
    Supports:
    - Dynamic cache growth
    - Memory-efficient storage
    - Multi-layer caching
    - Batch processing
    - Cache reuse and reset
    
    Best for:
    - Text generation with unknown length
    - Interactive chatbots
    - Development and debugging
    """
    
    def __init__(self):
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self.seen_tokens = 0  # Number of tokens processed
        
    def __getitem__(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached key and value for specific layer."""
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            return None, None
            
    def __len__(self) -> int:
        """Get number of cached layers."""
        return len(self.key_cache)
            
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key and value states."""
        # Ensure cache lists are long enough
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            
        if self.key_cache[layer_idx] is None:
            # First time caching for this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Check shapes before concatenating
            existing_key = self.key_cache[layer_idx]
            existing_value = self.value_cache[layer_idx]
            
            # Ensure compatible shapes (batch_size, num_heads, seq_len, head_dim)
            if (existing_key.shape[0] != key_states.shape[0] or 
                existing_key.shape[1] != key_states.shape[1] or
                existing_key.shape[3] != key_states.shape[3]):
                # Shape mismatch - replace instead of concatenate
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                # Concatenate along sequence dimension (dim=-2)
                self.key_cache[layer_idx] = torch.cat([existing_key, key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([existing_value, value_states], dim=-2)
            
        self.seen_tokens += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length from cache."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-2]
        return 0
        
    def get_max_length(self) -> int:
        """Get maximum sequence length across all layers."""
        max_len = 0
        for key_tensor in self.key_cache:
            if key_tensor is not None:
                max_len = max(max_len, key_tensor.shape[-2])
        return max_len
        
    def reset(self):
        """Reset cache to empty state."""
        self.key_cache = []
        self.value_cache = []
        self.seen_tokens = 0
        
    def clear_layer(self, layer_idx: int):
        """Clear cache for specific layer."""
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = None
            self.value_cache[layer_idx] = None
            
    def trim_to_length(self, max_length: int):
        """Trim cache to specified maximum length."""
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None:
                current_length = self.key_cache[i].shape[-2]
                if current_length > max_length:
                    # Keep the most recent tokens
                    start_idx = current_length - max_length
                    self.key_cache[i] = self.key_cache[i][..., start_idx:, :]
                    self.value_cache[i] = self.value_cache[i][..., start_idx:, :]
                    
    def get_cache_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_memory = 0
        layer_memories = []
        
        for i, (k, v) in enumerate(zip(self.key_cache, self.value_cache)):
            layer_memory = 0
            if k is not None:
                layer_memory += k.numel() * k.element_size()
            if v is not None:
                layer_memory += v.numel() * v.element_size()
            layer_memories.append(layer_memory)
            total_memory += layer_memory
            
        return {
            "total_memory_mb": total_memory / (1024 * 1024),
            "layer_memories_mb": [mem / (1024 * 1024) for mem in layer_memories],
            "num_layers": len(self.key_cache),
            "max_seq_length": self.get_max_length(),
            "seen_tokens": self.seen_tokens,
            "cache_type": "dynamic"
        }


class StaticCache(BaseCache):
    """Static KV cache with pre-allocated memory.
    
    More memory efficient for known sequence lengths.
    Used in production inference with fixed context windows.
    
    Best for:
    - Production inference with known max length
    - Batch processing
    - Memory-constrained environments
    - High-throughput scenarios
    """
    
    def __init__(
        self, 
        max_batch_size: int, 
        max_cache_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ):
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate cache tensors
        cache_shape = (max_batch_size, num_heads, max_cache_len, head_dim)
        
        self.key_cache = torch.zeros(
            num_layers, *cache_shape, dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            num_layers, *cache_shape, dtype=dtype, device=device
        )
        
        # Track current position in cache
        self.cache_position = 0
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor, 
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update static cache with new states."""
        cache_position = cache_kwargs.get("cache_position", self.cache_position) if cache_kwargs else self.cache_position
        
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Ensure we don't exceed cache limits
        if cache_position + seq_len > self.max_cache_len:
            warnings.warn(f"Cache overflow: position {cache_position} + seq_len {seq_len} > max_len {self.max_cache_len}")
            seq_len = max(0, self.max_cache_len - cache_position)
            if seq_len == 0:
                # Return existing cache if no room for new tokens
                return (
                    self.key_cache[layer_idx, :batch_size, :, :self.cache_position],
                    self.value_cache[layer_idx, :batch_size, :, :self.cache_position]
                )
            key_states = key_states[:, :, :seq_len, :]
            value_states = value_states[:, :, :seq_len, :]
        
        if batch_size > self.max_batch_size:
            warnings.warn(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
            batch_size = self.max_batch_size
            key_states = key_states[:batch_size]
            value_states = value_states[:batch_size]
        
        # Copy new states to cache
        if seq_len > 0:
            self.key_cache[layer_idx, :batch_size, :, cache_position:cache_position + seq_len] = key_states
            self.value_cache[layer_idx, :batch_size, :, cache_position:cache_position + seq_len] = value_states
            
            # Update cache position for next update
            self.cache_position = cache_position + seq_len
        
        # Return full cache up to current position
        return (
            self.key_cache[layer_idx, :batch_size, :, :self.cache_position],
            self.value_cache[layer_idx, :batch_size, :, :self.cache_position]
        )
        
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length."""
        return self.cache_position
        
    def reset(self):
        """Reset cache position."""
        self.cache_position = 0
        # Optionally zero out the cache for clean state
        # self.key_cache.zero_()
        # self.value_cache.zero_()
        
    def get_cache_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_memory = (
            self.key_cache.numel() * self.key_cache.element_size() +
            self.value_cache.numel() * self.value_cache.element_size()
        )
        
        return {
            "total_memory_mb": total_memory / (1024 * 1024),
            "cache_shape": list(self.key_cache.shape),
            "utilization": self.cache_position / self.max_cache_len,
            "max_batch_size": self.max_batch_size,
            "max_cache_len": self.max_cache_len,
            "current_position": self.cache_position,
            "cache_type": "static"
        }


class SlidingWindowCache(BaseCache):
    """Sliding window cache for long sequences.
    
    Maintains a fixed-size window of recent key-value pairs,
    automatically discarding older entries.
    
    Best for:
    - Very long sequences (100k+ tokens)
    - Streaming applications
    - Memory-constrained long-form generation
    """
    
    def __init__(self, window_size: int = 4096):
        self.window_size = window_size
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self.total_tokens_seen = 0
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with sliding window logic."""
        # Ensure cache lists are long enough
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            
        if self.key_cache[layer_idx] is None:
            # First time caching for this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate with existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            
            # Apply sliding window
            current_length = self.key_cache[layer_idx].shape[-2]
            if current_length > self.window_size:
                # Keep only the most recent tokens
                start_idx = current_length - self.window_size
                self.key_cache[layer_idx] = self.key_cache[layer_idx][..., start_idx:, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][..., start_idx:, :]
                
        self.total_tokens_seen += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length from cache."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-2]
        return 0
        
    def reset(self):
        """Reset cache to empty state."""
        self.key_cache = []
        self.value_cache = []
        self.total_tokens_seen = 0
        
    def get_cache_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_memory = 0
        for k, v in zip(self.key_cache, self.value_cache):
            if k is not None:
                total_memory += k.numel() * k.element_size()
            if v is not None:
                total_memory += v.numel() * v.element_size()
                
        return {
            "total_memory_mb": total_memory / (1024 * 1024),
            "window_size": self.window_size,
            "total_tokens_seen": self.total_tokens_seen,
            "num_layers": len(self.key_cache),
            "current_seq_length": self.get_seq_length(),
            "cache_type": "sliding_window"
        }


class QuantizedCache(BaseCache):
    """Quantized cache for memory efficiency.
    
    Stores key-value states in lower precision (int8/int4) to save memory.
    Useful for very large models or long sequences.
    
    Best for:
    - Memory-constrained environments
    - Very long sequences
    - Large batch sizes
    """
    
    def __init__(self, quantization_bits: int = 8):
        self.quantization_bits = quantization_bits
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self.key_scales: List[Optional[torch.Tensor]] = []
        self.value_scales: List[Optional[torch.Tensor]] = []
        self.seen_tokens = 0
        
    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to specified bit precision."""
        if self.quantization_bits == 8:
            # Quantize to int8
            scale = tensor.abs().max() / 127.0
            quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        elif self.quantization_bits == 4:
            # Quantize to int4 (stored in int8)
            scale = tensor.abs().max() / 7.0
            quantized = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)
        else:
            raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}")
            
        return quantized, scale
        
    def _dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor back to float."""
        return quantized.float() * scale
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with quantized states."""
        # Ensure cache lists are long enough
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.key_scales.append(None)
            self.value_scales.append(None)
            
        # Quantize new states
        key_quantized, key_scale = self._quantize(key_states)
        value_quantized, value_scale = self._quantize(value_states)
        
        if self.key_cache[layer_idx] is None:
            # First time caching for this layer
            self.key_cache[layer_idx] = key_quantized
            self.value_cache[layer_idx] = value_quantized
            self.key_scales[layer_idx] = key_scale
            self.value_scales[layer_idx] = value_scale
        else:
            # Dequantize existing cache, concatenate, and re-quantize
            existing_keys = self._dequantize(self.key_cache[layer_idx], self.key_scales[layer_idx])
            existing_values = self._dequantize(self.value_cache[layer_idx], self.value_scales[layer_idx])
            
            # Concatenate
            new_keys = torch.cat([existing_keys, key_states], dim=-2)
            new_values = torch.cat([existing_values, value_states], dim=-2)
            
            # Re-quantize
            self.key_cache[layer_idx], self.key_scales[layer_idx] = self._quantize(new_keys)
            self.value_cache[layer_idx], self.value_scales[layer_idx] = self._quantize(new_values)
            
        # Return dequantized states for computation
        final_keys = self._dequantize(self.key_cache[layer_idx], self.key_scales[layer_idx])
        final_values = self._dequantize(self.value_cache[layer_idx], self.value_scales[layer_idx])
        
        self.seen_tokens += key_states.shape[-2]
        return final_keys, final_values
        
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length from cache."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-2]
        return 0
        
    def reset(self):
        """Reset cache to empty state."""
        self.key_cache = []
        self.value_cache = []
        self.key_scales = []
        self.value_scales = []
        self.seen_tokens = 0
        
    def get_cache_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_memory = 0
        for k, v, k_s, v_s in zip(self.key_cache, self.value_cache, self.key_scales, self.value_scales):
            if k is not None:
                total_memory += k.numel() * k.element_size()
                total_memory += k_s.numel() * k_s.element_size()
            if v is not None:
                total_memory += v.numel() * v.element_size()
                total_memory += v_s.numel() * v_s.element_size()
                
        return {
            "total_memory_mb": total_memory / (1024 * 1024),
            "quantization_bits": self.quantization_bits,
            "compression_ratio": 32 / self.quantization_bits,  # Assuming float32 baseline
            "num_layers": len(self.key_cache),
            "seen_tokens": self.seen_tokens,
            "cache_type": "quantized"
        }


def create_cache(
    cache_type: str = "dynamic",
    **kwargs
) -> BaseCache:
    """Factory function to create different types of caches.
    
    Args:
        cache_type: Type of cache ("dynamic", "static", "sliding_window", "quantized")
        **kwargs: Additional arguments for cache creation
        
    Returns:
        Appropriate cache instance
    """
    if cache_type == "dynamic":
        return DynamicCache()
    elif cache_type == "static":
        return StaticCache(**kwargs)
    elif cache_type == "sliding_window":
        return SlidingWindowCache(**kwargs)
    elif cache_type == "quantized":
        return QuantizedCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


# Testing function
def test_cache_implementations():
    """Test different cache implementations."""
    print("🧪 Testing Cache Implementations...")
    
    # Test parameters
    batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64
    num_layers = 4
    
    # Create test tensors
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Test Dynamic Cache
    print("   Testing DynamicCache...")
    dynamic_cache = DynamicCache()
    for layer_idx in range(num_layers):
        k, v = dynamic_cache.update(key_states, value_states, layer_idx)
        assert k.shape == key_states.shape
        
    stats = dynamic_cache.get_cache_memory_usage()
    print(f"   Dynamic cache: {stats['total_memory_mb']:.2f} MB")
    
    # Test Static Cache
    print("   Testing StaticCache...")
    static_cache = StaticCache(
        max_batch_size=batch_size,
        max_cache_len=32,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=torch.device('cpu')
    )
    k, v = static_cache.update(key_states, value_states, 0)
    assert k.shape[:-2] == (batch_size, num_heads)
    
    stats = static_cache.get_cache_memory_usage()
    print(f"   Static cache: {stats['total_memory_mb']:.2f} MB")
    
    # Test Sliding Window Cache
    print("   Testing SlidingWindowCache...")
    sliding_cache = SlidingWindowCache(window_size=24)
    k, v = sliding_cache.update(key_states, value_states, 0)
    assert k.shape == key_states.shape
    
    # Test Quantized Cache
    print("   Testing QuantizedCache...")
    quantized_cache = QuantizedCache(quantization_bits=8)
    k, v = quantized_cache.update(key_states, value_states, 0)
    assert k.shape == key_states.shape
    
    stats = quantized_cache.get_cache_memory_usage()
    print(f"   Quantized cache: {stats['total_memory_mb']:.2f} MB, compression: {stats['compression_ratio']:.1f}x")
    
    print("✅ Cache implementation tests passed!")


if __name__ == "__main__":
    test_cache_implementations()