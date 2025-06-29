"""LLaMA attention components.

This module contains attention-related components including RoPE,
attention mechanisms, and optimized implementations.
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle imports for different execution contexts
try:
    from .config import LlamaMoEConfig
    from ...utils.cache import BaseCache, DynamicCache
except ImportError:
    # Direct execution - use absolute imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from training_infra.models.llama.config import LlamaMoEConfig
    from training_infra.utils.cache import BaseCache, DynamicCache


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with LLaMA 3/4 enhancements.
    
    Supports:
    - Original RoPE from LLaMA 1/2
    - Linear scaling for longer sequences (LLaMA 3)
    - Dynamic NTK scaling (LLaMA 3/4)
    - Yarn scaling for ultra-long contexts
    - Su scaling (Code Llama style)
    - Efficient caching for inference
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048, 
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "linear",
        original_max_position_embeddings: int = 2048
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        self.original_max_position_embeddings = original_max_position_embeddings
        
        # Calculate base frequencies
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
        
        # Build initial cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )
    
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies with scaling support."""
        # Base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        if self.scaling_type == "linear":
            # Linear scaling: divide frequencies by scaling factor
            inv_freq = inv_freq / self.scaling_factor
            
        elif self.scaling_type == "dynamic":
            # Dynamic NTK scaling: adjust base frequency
            if self.max_position_embeddings > self.original_max_position_embeddings:
                ratio = self.max_position_embeddings / self.original_max_position_embeddings
                alpha = ratio - 1
                base_adjusted = self.base * ((1 + alpha) ** (self.dim / (self.dim - 2)))
                inv_freq = 1.0 / (base_adjusted ** (torch.arange(0, self.dim, 2).float() / self.dim))
                
        elif self.scaling_type == "yarn":
            # YaRN scaling: frequency-dependent scaling
            if self.max_position_embeddings > self.original_max_position_embeddings:
                ratio = self.max_position_embeddings / self.original_max_position_embeddings
                freqs = torch.arange(0, self.dim, 2).float() / self.dim
                gamma = 0.1
                
                scale = torch.where(
                    freqs < gamma,
                    1.0,
                    torch.where(
                        freqs > 1 - gamma,
                        1.0 / ratio,
                        (1 - (freqs - gamma) / (1 - 2 * gamma)) + 
                        ((freqs - gamma) / (1 - 2 * gamma)) / ratio
                    )
                )
                inv_freq = inv_freq * scale
                
        elif self.scaling_type == "su":
            # Su scaling (Code Llama style)
            if self.max_position_embeddings > self.original_max_position_embeddings:
                ratio = self.max_position_embeddings / self.original_max_position_embeddings
                freqs = torch.arange(0, self.dim, 2).float() / self.dim
                smooth_factor = (1 - freqs) * 1.0 + freqs * ratio
                inv_freq = inv_freq / smooth_factor
                
        return inv_freq
        
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Precompute and cache cos/sin values for efficiency."""
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return  # Use existing cache
            
        self._cached_seq_len = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self._cached_cos = emb.cos().to(dtype)
        self._cached_sin = emb.sin().to(dtype)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
            
        # Extend cache if needed
        if seq_len > self._cached_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        return (
            self._cached_cos[:seq_len].to(dtype=x.dtype),
            self._cached_sin[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, 
                        position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    if position_ids is None:
        cos = cos[:q.shape[-2]]
        sin = sin[:q.shape[-2]]
    else:
        cos = cos[position_ids]
        sin = sin[position_ids]
        
    cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin.unsqueeze(1)  # [seq_len, 1, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LLaMAAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""
    
    def __init__(self, config: LlamaMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = 0.0  # LLaMA doesn't use attention dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        # Import parallel layers (will fallback to standard if not available)
        try:
            from .parallel import ColumnParallelLinear, RowParallelLinear
            LinearLayer = ColumnParallelLinear
            OutputLayer = RowParallelLinear
        except ImportError:
            # Fallback to standard PyTorch layers
            LinearLayer = nn.Linear
            OutputLayer = nn.Linear
            
        # Linear projections
        self.q_proj = LinearLayer(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = LinearLayer(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = LinearLayer(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = OutputLayer(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # RoPE with LLaMA 3/4 enhancements
        rope_scaling = getattr(config, 'rope_scaling', None)
        if rope_scaling:
            scaling_type = rope_scaling.get("type", "linear")
            scaling_factor = rope_scaling.get("factor", 1.0)
            original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 2048)
        else:
            scaling_type = "linear"
            scaling_factor = 1.0
            original_max_position_embeddings = self.max_position_embeddings
            
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=scaling_factor,
            scaling_type=scaling_type,
            original_max_position_embeddings=original_max_position_embeddings
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[BaseCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[BaseCache]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Handle past key values for generation
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin, 
                "cos": cos, 
                "cache_position": cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        elif use_cache:
            # Create new cache if needed
            if not hasattr(self, '_cache') or self._cache is None:
                self._cache = DynamicCache()
            cache_kwargs = {
                "sin": sin,
                "cos": cos, 
                "cache_position": cache_position
            }
            key_states, value_states = self._cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            past_key_value = self._cache
            
        # Repeat key and value states for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention - use Flash Attention if available and beneficial
        use_flash_attn = (
            hasattr(self.config, 'use_flash_attention') and 
            self.config.use_flash_attention and
            query_states.dtype in [torch.float16, torch.bfloat16]
        )
        
        if use_flash_attn:
            try:
                from .flash_attention import flash_attention_forward
                attn_output = flash_attention_forward(
                    query_states, key_states, value_states, attention_mask, 
                    self.attention_dropout, self.training
                )
                attn_weights = None
            except (ImportError, RuntimeError):
                # Fallback to standard attention
                use_flash_attn = False
        
        if not use_flash_attn:
            # Standard attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                # Get the correct slice of attention mask for current key length
                key_len = key_states.shape[-2]
                query_len = query_states.shape[-2]
                
                # Slice the mask to match query and key dimensions
                if attention_mask.shape[-1] >= key_len and attention_mask.shape[-2] >= query_len:
                    causal_mask = attention_mask[:, :, -query_len:, -key_len:]
                else:
                    # Create a new mask if existing one doesn't fit
                    causal_mask = self._create_causal_mask(
                        query_states.shape[0], query_len, key_len, 
                        query_states.dtype, query_states.device
                    )
                
                attn_weights = attn_weights + causal_mask
                
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value
        
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key and value states for grouped query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def _create_causal_mask(self, batch_size: int, query_len: int, key_len: int, dtype: torch.dtype, device: torch.device):
        """Create causal attention mask for specific query and key lengths."""
        mask = torch.full((query_len, key_len), torch.finfo(dtype).min, device=device)
        
        # For causal attention, each query position can only attend to previous positions
        for i in range(query_len):
            # Allow attention to all key positions up to the corresponding query position
            if key_len >= query_len:
                # Keys include history + current queries
                attend_until = key_len - query_len + i + 1
            else:
                # Keys are shorter than queries (unusual case)
                attend_until = min(i + 1, key_len)
            
            mask[i, :attend_until] = 0
        
        return mask[None, None, :, :].expand(batch_size, 1, query_len, key_len)


# Testing function
def test_attention_components():
    """Test attention components."""
    print("🧪 Testing Attention Components...")
    
    # Test RoPE
    rope = RotaryPositionalEmbedding(64, max_position_embeddings=2048)
    x = torch.randn(2, 8, 16, 64)
    cos, sin = rope(x)
    assert cos.shape == (16, 64)
    assert sin.shape == (16, 64)
    print("   ✅ RoPE working")
    
    # Test RoPE scaling
    rope_scaled = RotaryPositionalEmbedding(
        64, max_position_embeddings=8192, scaling_factor=4.0, scaling_type="dynamic"
    )
    cos_scaled, sin_scaled = rope_scaled(x)
    print("   ✅ RoPE scaling working")
    
    # Test attention layer
    from .config import LLaMAVariants
    config = LLaMAVariants.tiny_llama()
    attn = LLaMAAttention(config, layer_idx=0)
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output, weights, cache = attn(hidden_states, use_cache=True)
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    print("   ✅ Attention forward working")
    
    print("✅ Attention components tests passed!")


if __name__ == "__main__":
    test_attention_components()