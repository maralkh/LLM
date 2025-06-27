"""LLaMA model implementation.

This module provides the main LLaMA model using modular components
from separate files for better organization and maintainability.
"""


from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle imports for different execution contexts
try:
    from .config import LLaMAConfig
    from ..base import LanguageModel
    from ...utils.cache import BaseCache, DynamicCache
    from .attention import LLaMAAttention
    from .mlp import LLaMAMLP, RMSNorm
except ImportError:
    # Direct execution - use absolute imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from training_infra.models.llama.config import LLaMAConfig
    from training_infra.models.base import LanguageModel
    from training_infra.utils.cache import BaseCache, DynamicCache
    from training_infra.models.llama.attention import LLaMAAttention
    from training_infra.models.llama.mlp import LLaMAMLP, RMSNorm


class LLaMADecoderLayer(nn.Module):
    """LLaMA decoder layer with attention and MLP."""
    
    def __init__(self, config: LLaMAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        
        self.self_attn = LLaMAAttention(config=config, layer_idx=layer_idx)
        self.mlp = LLaMAMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[BaseCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # MLP
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


class LLaMAModel(LanguageModel):
    """LLaMA model implementation.
    
    Complete LLaMA model with embeddings, transformer layers, and output head.
    Supports various configurations from tiny (150M) to large (70B+) models.
    """
    
    def __init__(self, config: LLaMAConfig):
        super().__init__(
            vocab_size=config.vocab_size,
            model_name=f"llama_{config.model_type}"
        )
        
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Import parallel embedding if available
        try:
            from .parallel import VocabParallelEmbedding
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )
        except ImportError:
            # Fallback to standard embedding
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output projection
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            # Try to use parallel output layer
            try:
                from .parallel import ColumnParallelLinear
                self.lm_head = ColumnParallelLinear(
                    config.hidden_size, config.vocab_size, bias=False, gather_output=True
                )
            except ImportError:
                # Fallback to standard linear layer
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
        # Initialize weights
        self._initialize_weights()
        
        # Set dtype
        self.to(config.get_torch_dtype())
        
    def _initialize_weights(self):
        """Initialize model weights using LLaMA-specific initialization."""
        try:
            from ..utils.initialization import apply_initialization
        except ImportError:
            # Fallback to basic initialization
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            return
        
        apply_initialization(
            self,
            method='llama',
            width=self.config.hidden_size,
            std=self.config.initializer_range
        )
        
    def get_input_embeddings(self):
        return self.embed_tokens
        
    def set_input_embeddings(self, value):
        self.embed_tokens = value
        
    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.embed_tokens
        return self.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        if self.config.tie_word_embeddings:
            self.embed_tokens = new_embeddings
        else:
            self.lm_head = new_embeddings
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        
        # Handle inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        if use_cache is None:
            use_cache = self.config.use_cache
            
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        hidden_states = inputs_embeds
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=hidden_states.device)
            
        # Create causal mask
        causal_mask = self._create_causal_mask(batch_size, seq_length, hidden_states.dtype, hidden_states.device)
        
        # Combine masks
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask * causal_mask
        else:
            attention_mask = causal_mask
            
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0)
            
        # Forward through layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
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
                
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Output projection
        if self.config.tie_word_embeddings:
            # Use embedding weights for output projection
            if hasattr(self.embed_tokens, 'weight'):
                logits = F.linear(hidden_states, self.embed_tokens.weight)
            else:
                # For parallel embeddings, might need different handling
                logits = self.embed_tokens(hidden_states)  # This might not work for VocabParallelEmbedding
        else:
            logits = self.lm_head(hidden_states)
            
        return logits
        
    def _create_causal_mask(self, batch_size: int, seq_length: int, dtype: torch.dtype, device: torch.device):
        """Create causal attention mask."""
        mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        return mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Simple text generation."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop at EOS token
                if next_token.item() == self.config.eos_token_id:
                    break
                    
        return input_ids
        
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.to_dict()
        
    @classmethod
    def _from_config(cls, config: Dict[str, Any], device=None, model_name="llama_model"):
        """Create model from config dictionary."""
        llama_config = LLaMAConfig.from_dict(config)
        model = cls(llama_config)
        if device:
            model.to(device)
        model.model_name = model_name
        return model


# Testing function
def test_llama_model():
    """Test LLaMA model creation and forward pass."""
    print("🧪 Testing LLaMA Model...")
    
    try:
        from .config import LLaMAVariants
    except ImportError:
        from training_infra.models.llama.config import LLaMAVariants
    
    # Test tiny model for fast testing
    config = LLaMAVariants.tiny_llama("float32")
    model = LLaMAModel(config)
    
    print(f"   Created {config.model_type} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    # Test generation
    gen_input = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(gen_input, max_length=10, do_sample=False)
    assert generated.shape[1] > gen_input.shape[1], "Generation should extend input"
    
    # Test model info
    model.print_model_info()
    
    print("✅ LLaMA model tests passed!")


if __name__ == "__main__":
    test_llama_model()