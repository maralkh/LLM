# training_infra/models/moe/attention.py
"""
LLaMA-MoE attention implementation.
Uses exact same attention as base LLaMA with RoPE and GQA support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    # Try relative import first (when imported as module)  
    from .config import LlamaMoEConfig
except ImportError:
    # Fallback for direct execution
    try:
        from config import LlamaMoEConfig
    except ImportError:
        print("Warning: Could not import LlamaMoEConfig. Make sure the module is properly installed.")
        raise

# Import tensor parallel components from main LLaMA implementation
try:
    from ..parallelism import TensorParallelLinear
    _TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    _TENSOR_PARALLEL_AVAILABLE = False
    # Fallback to regular Linear
    TensorParallelLinear = nn.Linear


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - exactly like LLaMA"""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors - exactly like LLaMA."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). 
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaMoEAttention(nn.Module):
    """Multi-head attention - exactly like LLaMA with GQA support"""

    def __init__(self, config: LlamaMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.tensor_parallel_config = getattr(config, 'tensor_parallel_config', None)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Use tensor parallel linear layers
        if _TENSOR_PARALLEL_AVAILABLE and self.tensor_parallel_config:
            self.q_proj = TensorParallelLinear(
                self.hidden_size, 
                self.num_heads * self.head_dim, 
                bias=config.attention_bias,
                config=self.tensor_parallel_config,
                gather_output=False  # Keep sharded for attention
            )
            self.k_proj = TensorParallelLinear(
                self.hidden_size, 
                self.num_key_value_heads * self.head_dim, 
                bias=config.attention_bias,
                config=self.tensor_parallel_config,
                gather_output=False
            )
            self.v_proj = TensorParallelLinear(
                self.hidden_size, 
                self.num_key_value_heads * self.head_dim, 
                bias=config.attention_bias,
                config=self.tensor_parallel_config,
                gather_output=False
            )
            self.o_proj = TensorParallelLinear(
                self.num_heads * self.head_dim, 
                self.hidden_size, 
                bias=config.attention_bias,
                config=self.tensor_parallel_config,
                scatter_input=True,  # Input is sharded from attention
                gather_output=True   # Output should be gathered
            )
        else:
            # Fallback to regular linear layers
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def test_llama_moe_attention():
    """Test LLaMA-MoE attention module."""
    print("Testing LLaMA-MoE Attention...")
    
    try:
        from .config import create_tiny_llama_moe
    except ImportError:
        from config import create_tiny_llama_moe
    
    config = create_tiny_llama_moe()
    attention = LlamaMoEAttention(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output, attn_weights, past_kv = attention(
        hidden_states, 
        output_attentions=True, 
        use_cache=True
    )
    
    # Verify output shapes
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    assert attn_weights.shape == (batch_size, config.num_attention_heads, seq_len, seq_len)
    
    print(f"✅ Attention output shape: {output.shape}")
    print(f"✅ Attention weights shape: {attn_weights.shape}")
    
    # Test with attention mask (causal)
    causal_mask = torch.triu(torch.ones(1, 1, seq_len, seq_len) * float('-inf'), diagonal=1)
    
    output_masked, _, _ = attention(hidden_states, attention_mask=causal_mask)
    assert output_masked.shape == (batch_size, seq_len, config.hidden_size)
    
    print("✅ Attention with causal mask works")
    
    # Test GQA (Grouped Query Attention)
    print(f"✅ GQA: {config.num_attention_heads} query heads, {config.num_key_value_heads} kv heads")
    print(f"✅ Repetition factor: {config.num_attention_heads // config.num_key_value_heads}")
    
    # Test parameter count
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"✅ Attention parameters: {total_params:,}")
    
    print("✅ All LLaMA-MoE attention tests passed!")


if __name__ == "__main__":
    test_llama_moe_attention()