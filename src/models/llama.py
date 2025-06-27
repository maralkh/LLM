# training_infra/models/llama.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# Import our advanced parallelism modules
from ..parallelism import (
    TensorParallelLinear, 
    TensorParallelEmbedding, 
    TensorParallelConfig,
    FlashAttention,
    FlashMHA,
    enable_flash_attention,
    ActivationCheckpointing,
    GradientCompression
)

@dataclass
class LlamaConfig:
    """Configuration for LLaMA model with parallelism support"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # For GQA
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    
    # Advanced parallelism configurations
    tensor_parallel_config: Optional[TensorParallelConfig] = None
    use_flash_attention: bool = True
    use_activation_checkpointing: bool = False
    use_gradient_compression: bool = False
    checkpoint_every_n_layers: int = 4
    compression_ratio: float = 0.1
    mixed_precision: bool = True
    sequence_parallel: bool = False  # For very long sequences
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
            
        # Create default tensor parallel config if not provided
        if self.tensor_parallel_config is None:
            self.tensor_parallel_config = TensorParallelConfig()

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with optional tensor parallelism"""
    
    def __init__(self, hidden_size, eps=1e-6, tensor_parallel_config=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.tensor_parallel_config = tensor_parallel_config or TensorParallelConfig()

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with caching optimization"""
    
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
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
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

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaAttention(nn.Module):
    """Multi-headed attention with advanced parallelism support"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
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
        self.tensor_parallel_config = config.tensor_parallel_config

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Use tensor parallel linear layers
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
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        # Initialize FlashAttention if enabled
        self.use_flash_attention = config.use_flash_attention
        if self.use_flash_attention:
            try:
                self.flash_attn = FlashMHA(
                    embed_dim=self.hidden_size,
                    num_heads=self.num_heads,
                    bias=config.attention_bias,
                    dropout=config.attention_dropout,
                    causal=True,
                    device=None,
                    dtype=None
                )
                print(f"FlashAttention enabled for layer {layer_idx}")
            except Exception as e:
                warnings.warn(f"FlashAttention not available, falling back to standard attention: {e}")
                self.use_flash_attention = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        # Use Flash Attention if available and not outputting attention weights
        if self.use_flash_attention and not output_attentions and past_key_value is None:
            return self._flash_attention_forward(hidden_states, attention_mask, position_ids)

        # Standard attention computation with tensor parallelism
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Adjust for tensor parallelism - heads are sharded across devices
        local_num_heads = self.num_heads // self.tensor_parallel_config.tensor_parallel_size
        local_num_kv_heads = self.num_key_value_heads // self.tensor_parallel_config.tensor_parallel_size

        query_states = query_states.view(bsz, q_len, local_num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, local_num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, local_num_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads (for local heads)
        local_kv_groups = local_num_heads // local_num_kv_heads
        key_states = self._repeat_kv(key_states, local_kv_groups)
        value_states = self._repeat_kv(value_states, local_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)  # Local hidden size

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(self, hidden_states, attention_mask, position_ids):
        """Forward pass using FlashAttention"""
        try:
            # FlashAttention expects different input format
            output = self.flash_attn(
                hidden_states, 
                key_padding_mask=attention_mask,
                need_weights=False
            )
            return output, None, None
        except Exception as e:
            warnings.warn(f"FlashAttention failed, falling back to standard: {e}")
            # Fallback to standard attention
            return self.forward(hidden_states, attention_mask, position_ids, 
                               output_attentions=False, use_cache=False)

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation and tensor parallelism"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_config = config.tensor_parallel_config
        
        # Use tensor parallel linear layers for MLP
        self.gate_proj = TensorParallelLinear(
            self.hidden_size, 
            self.intermediate_size, 
            bias=False,
            config=self.tensor_parallel_config,
            gather_output=False  # Keep sharded for elementwise ops
        )
        self.up_proj = TensorParallelLinear(
            self.hidden_size, 
            self.intermediate_size, 
            bias=False,
            config=self.tensor_parallel_config,
            gather_output=False
        )
        self.down_proj = TensorParallelLinear(
            self.intermediate_size, 
            self.hidden_size, 
            bias=False,
            config=self.tensor_parallel_config,
            scatter_input=True,  # Input is sharded
            gather_output=True   # Gather final output
        )
        self.act_fn = self._get_activation_function(config.hidden_act)

    def _get_activation_function(self, hidden_act):
        if hidden_act == "silu":
            return F.silu
        elif hidden_act == "gelu":
            return F.gelu
        elif hidden_act == "relu":
            return F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x):
        # SwiGLU: gate_proj(x) * silu(up_proj(x))
        # Both projections are sharded, so elementwise ops work correctly
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate = self.act_fn(gate_output) * up_output
        return self.down_proj(intermediate)

class LlamaDecoderLayer(nn.Module):
    """LLaMA Decoder Layer with advanced optimizations"""
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, 
                                     tensor_parallel_config=config.tensor_parallel_config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                              tensor_parallel_config=config.tensor_parallel_config)
        
        # Activation checkpointing
        self.use_activation_checkpointing = config.use_activation_checkpointing
        if self.use_activation_checkpointing:
            self.activation_checkpoint = ActivationCheckpointing()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Apply activation checkpointing if enabled
        if self.use_activation_checkpointing and self.training:
            return self.activation_checkpoint.checkpoint(
                self._forward_impl,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position
            )
        else:
            return self._forward_impl(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position
            )
    
    def _forward_impl(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position
    ):
        """Actual forward implementation"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel(nn.Module):
    """LLaMA Model with advanced parallelism and optimization support"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Use tensor parallel embedding
        self.embed_tokens = TensorParallelEmbedding(
            config.vocab_size, 
            config.hidden_size, 
            self.padding_idx,
            config=config.tensor_parallel_config
        )
        
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                           tensor_parallel_config=config.tensor_parallel_config)

        self.gradient_checkpointing = config.use_activation_checkpointing
        
        # Initialize gradient compression if enabled
        if config.use_gradient_compression:
            self.gradient_compression = GradientCompression(
                compression_ratio=config.compression_ratio
            )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def enable_gradient_compression(self):
        """Enable gradient compression for memory efficiency"""
        if hasattr(self, 'gradient_compression'):
            for module in self.modules():
                if hasattr(module, 'weight') and module.weight.requires_grad:
                    self.gradient_compression.register_hooks(module.weight)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare attention mask (simplified for this implementation)
        if attention_mask is not None:
            # Convert attention mask to additive mask
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Apply layer-wise checkpointing
            if (self.config.use_activation_checkpointing and 
                self.training and 
                idx % self.config.checkpoint_every_n_layers == 0):
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    use_reentrant=False
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }

class LlamaForCausalLM(nn.Module):
    """LLaMA Model for Causal Language Modeling with advanced optimizations"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.config = config
        
        # Use tensor parallel linear for output
        self.lm_head = TensorParallelLinear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False,
            config=config.tensor_parallel_config,
            gather_output=True  # Need to gather for loss computation
        )

        # Initialize weights and apply final processing
        self.post_init()
        
        # Enable optimizations
        if config.use_gradient_compression:
            self.model.enable_gradient_compression()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs["last_hidden_state"]
        
        # Use mixed precision for forward pass if enabled
        if self.config.mixed_precision and self.training:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
            
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0

            # Keep only the unprocessed tokens:
            if past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def post_init(self):
        """Post initialization - tie weights if needed"""
        if hasattr(self.config, 'tie_word_embeddings') and self.config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        """Tie word embeddings with output layer"""
        if isinstance(self.lm_head, TensorParallelLinear) and isinstance(self.model.embed_tokens, TensorParallelEmbedding):
            # For tensor parallel, we need to ensure proper weight sharing
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head.weight = self.model.embed_tokens.weight

    def enable_flash_attention(self):
        """Enable FlashAttention for all layers"""
        for layer in self.model.layers:
            if hasattr(layer.self_attn, 'use_flash_attention'):
                layer.self_attn.use_flash_attention = True

    def disable_flash_attention(self):
        """Disable FlashAttention for all layers"""
        for layer in self.model.layers:
            if hasattr(layer.self_attn, 'use_flash_attention'):
                layer.self_attn.use_flash_attention = False

    def get_memory_usage(self):
        """Get detailed memory usage information"""
        memory_info = {
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            })
        
        return memory_info

# Enhanced model creation functions with parallelism support
def create_llama_7b_parallel(tensor_parallel_size=1, use_flash_attention=True, use_checkpointing=False):
    """Create LLaMA 7B model with advanced parallelism"""
    tensor_parallel_config = TensorParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=False,
        async_communication=True
    )
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tensor_parallel_config=tensor_parallel_config,
        use_flash_attention=use_flash_attention,
        use_activation_checkpointing=use_checkpointing,
        checkpoint_every_n_layers=4,
        mixed_precision=True
    )
    return LlamaForCausalLM(config)

def create_llama_13b_parallel(tensor_parallel_size=2, use_flash_attention=True, use_checkpointing=True):
    """Create LLaMA 13B model with advanced parallelism"""
    tensor_parallel_config = TensorParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=False,
        async_communication=True
    )
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tensor_parallel_config=tensor_parallel_config,
        use_flash_attention=use_flash_attention,
        use_activation_checkpointing=use_checkpointing,
        checkpoint_every_n_layers=4,
        mixed_precision=True,
        use_gradient_compression=True,
        compression_ratio=0.1
    )
    return LlamaForCausalLM(config)

def create_llama_30b_parallel(tensor_parallel_size=4, use_flash_attention=True, use_checkpointing=True):
    """Create LLaMA 30B model with advanced parallelism"""
    tensor_parallel_config = TensorParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=True,  # Enable sequence parallelism for larger models
        async_communication=True
    )
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=6656,
        intermediate_size=17920,
        num_hidden_layers=60,
        num_attention_heads=52,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tensor_parallel_config=tensor_parallel_config,
        use_flash_attention=use_flash_attention,
        use_activation_checkpointing=use_checkpointing,
        checkpoint_every_n_layers=2,  # More frequent checkpointing
        mixed_precision=True,
        use_gradient_compression=True,
        compression_ratio=0.05,  # Higher compression
        sequence_parallel=True
    )
    return LlamaForCausalLM(config)

def create_llama_65b_parallel(tensor_parallel_size=8, use_flash_attention=True, use_checkpointing=True):
    """Create LLaMA 65B model with maximum parallelism"""
    tensor_parallel_config = TensorParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=True,
        async_communication=True
    )
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=22016,
        num_hidden_layers=80,
        num_attention_heads=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tensor_parallel_config=tensor_parallel_config,
        use_flash_attention=use_flash_attention,
        use_activation_checkpointing=use_checkpointing,
        checkpoint_every_n_layers=1,  # Checkpoint every layer
        mixed_precision=True,
        use_gradient_compression=True,
        compression_ratio=0.02,  # Maximum compression
        sequence_parallel=True
    )
    return LlamaForCausalLM(config)

def create_llama2_7b_parallel(tensor_parallel_size=1, use_flash_attention=True, extended_context=True):
    """Create LLaMA 2 7B model with extended context and parallelism"""
    tensor_parallel_config = TensorParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=extended_context,  # Enable for long context
        async_communication=True
    )
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=4096 if extended_context else 2048,
        rms_norm_eps=1e-5,
        tensor_parallel_config=tensor_parallel_config,
        use_flash_attention=use_flash_attention,
        use_activation_checkpointing=extended_context,
        checkpoint_every_n_layers=4,
        mixed_precision=True,
        sequence_parallel=extended_context
    )
    return LlamaForCausalLM(config)

def create_code_llama_7b_parallel(tensor_parallel_size=1, use_flash_attention=True):
    """Create Code Llama 7B model with long context support"""
    tensor_parallel_config = TensorParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=True,  # Essential for 16k context
        async_communication=True
    )
    
    config = LlamaConfig(
        vocab_size=32016,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=16384,  # Long context for code
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,  # Different RoPE base
        tensor_parallel_config=tensor_parallel_config,
        use_flash_attention=use_flash_attention,
        use_activation_checkpointing=True,  # Necessary for long sequences
        checkpoint_every_n_layers=4,
        mixed_precision=True,
        sequence_parallel=True
    )
    return LlamaForCausalLM(config)

# Utility functions for model optimization
def optimize_model_for_training(model, optimization_level="medium"):
    """Apply various optimizations to the model"""
    
    if optimization_level == "basic":
        # Enable Flash Attention only
        model.enable_flash_attention()
        
    elif optimization_level == "medium":
        # Enable Flash Attention and some checkpointing
        model.enable_flash_attention()
        model.config.use_activation_checkpointing = True
        model.config.checkpoint_every_n_layers = 4
        
    elif optimization_level == "aggressive":
        # Enable all optimizations
        model.enable_flash_attention()
        model.config.use_activation_checkpointing = True
        model.config.checkpoint_every_n_layers = 2
        model.config.use_gradient_compression = True
        model.config.compression_ratio = 0.1
        model.model.enable_gradient_compression()
        
    return model

def estimate_model_memory(config: LlamaConfig, batch_size=1, sequence_length=2048):
    """Estimate memory requirements for the model"""
    
    # Parameter memory
    param_memory = (
        config.vocab_size * config.hidden_size +  # embeddings
        config.num_hidden_layers * (
            4 * config.hidden_size * config.hidden_size +  # attention projections
            3 * config.hidden_size * config.intermediate_size  # MLP
        ) +
        config.hidden_size * config.vocab_size  # output projection
    ) * 4  # 4 bytes per float32
    
    # Activation memory (approximate)
    activation_memory = (
        batch_size * sequence_length * config.hidden_size * 
        config.num_hidden_layers * 4  # rough estimate
    ) * 4
    
    # Reduce by tensor parallelism
    if config.tensor_parallel_config:
        tp_size = config.tensor_parallel_config.tensor_parallel_size
        param_memory = param_memory // tp_size
        activation_memory = activation_memory  # Activations still distributed
    
    # Reduce by activation checkpointing
    if config.use_activation_checkpointing:
        activation_memory = activation_memory // config.checkpoint_every_n_layers
    
    total_memory_mb = (param_memory + activation_memory) / 1024 / 1024
    
    return {
        "parameter_memory_mb": param_memory / 1024 / 1024,
        "activation_memory_mb": activation_memory / 1024 / 1024,
        "total_memory_mb": total_memory_mb,
        "recommended_gpu_memory_gb": total_memory_mb / 1024 * 1.5  # 50% overhead
    }

# Example usage and testing
if __name__ == "__main__":
    import torch.distributed as dist
    
    # Example: Create and optimize a 7B model
    print("Creating LLaMA 7B with parallelism...")
    model = create_llama_7b_parallel(
        tensor_parallel_size=2,
        use_flash_attention=True,
        use_checkpointing=True
    )
    
    # Optimize for training
    model = optimize_model_for_training(model, "medium")
    
    # Print memory estimates
    memory_info = estimate_model_memory(model.config)
    print(f"Estimated memory usage: {memory_info}")
    
    # Print model info
    print(f"Model memory usage: {model.get_memory_usage()}")
    
    # Example forward pass
    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 512)).cuda()
        
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"Output shape: {outputs['logits'].shape}")
            print("Forward pass successful!")
    
    print("Model creation and testing completed!")