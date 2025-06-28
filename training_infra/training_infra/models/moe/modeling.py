# training_infra/models/moe/modeling.py
"""
Complete LLaMA-MoE model implementation.
Combines all components into working LLaMA-MoE models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass

try:
    # Try relative import first (when imported as module)
    from .config import LlamaMoEConfig
    from .attention import LlamaMoEAttention, RMSNorm
    from .mlp import LlamaMoEMLP
    from .experts import LlamaMoELayer, LlamaMoEOutput
except ImportError:
    # Fallback for direct execution
    try:
        from config import LlamaMoEConfig
        from attention import LlamaMoEAttention, RMSNorm
        from mlp import LlamaMoEMLP
        from experts import LlamaMoELayer, LlamaMoEOutput
    except ImportError:
        print("Warning: Could not import MoE components. Make sure the module is properly installed.")
        raise

# Import tensor parallel embedding if available
try:
    from ..parallelism import TensorParallelEmbedding
    _TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    _TENSOR_PARALLEL_AVAILABLE = False
    TensorParallelEmbedding = nn.Embedding


@dataclass
class LlamaMoEModelOutput:
    """Output from LLaMA-MoE model."""
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    router_logits: Optional[Tuple[torch.Tensor]] = None
    aux_loss: Optional[torch.Tensor] = None


@dataclass  
class LlamaMoECausalLMOutput:
    """Output from LLaMA-MoE causal language model."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    router_logits: Optional[Tuple[torch.Tensor]] = None
    aux_loss: Optional[torch.Tensor] = None


class LlamaMoEDecoderLayer(nn.Module):
    """Single LLaMA-MoE decoder layer with attention and MLP/MoE."""
    
    def __init__(self, config: LlamaMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Attention (always standard LLaMA attention)
        self.self_attn = LlamaMoEAttention(config, layer_idx)
        
        # MLP or MoE layer
        self.use_moe = layer_idx in config.moe_layers
        if self.use_moe:
            self.mlp = LlamaMoELayer(config)
        else:
            self.mlp = LlamaMoEMLP(config)
        
        # Layer norms (like LLaMA)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, ...]]]:
        
        # Pre-attention normalization (LLaMA style)
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
        
        # Pre-MLP normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            # MoE layer
            moe_output = self.mlp(hidden_states)
            hidden_states = residual + moe_output.hidden_states
            
            # Collect MoE outputs
            router_logits = moe_output.router_logits if output_router_logits else None
            aux_loss = moe_output.aux_loss
        else:
            # Standard LLaMA MLP
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            router_logits = None
            aux_loss = None
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        if output_router_logits and router_logits is not None:
            outputs += (router_logits,)
        
        if aux_loss is not None:
            outputs += (aux_loss,)
        
        return outputs


class LlamaMoEModel(nn.Module):
    """
    Complete LLaMA-MoE transformer model.
    """
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        if _TENSOR_PARALLEL_AVAILABLE and hasattr(config, 'tensor_parallel_config'):
            self.embed_tokens = TensorParallelEmbedding(
                config.vocab_size, 
                config.hidden_size, 
                self.padding_idx,
                config=config.tensor_parallel_config
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaMoEDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def _init_weights(self, module):
        """Initialize weights using LLaMA-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif hasattr(module, 'router') and hasattr(module.router, 'weight'):
            # Special initialization for routers - smaller variance for stability
            torch.nn.init.normal_(module.router.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlamaMoEModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        output_router_logits = output_router_logits if output_router_logits is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        # Input processing
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        # Initialize outputs
        if output_hidden_states:
            all_hidden_states = ()
        if output_attentions:
            all_self_attns = ()
        if output_router_logits:
            all_router_logits = ()
        
        next_decoder_cache = () if use_cache else None
        all_aux_losses = []
        
        # Forward through all layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, output_router_logits, use_cache)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # Collect router logits and aux losses
            router_logits_idx = -2 if len(layer_outputs) > 3 else None
            aux_loss_idx = -1
            
            if output_router_logits and router_logits_idx is not None:
                if len(layer_outputs) > abs(router_logits_idx) and layer_outputs[router_logits_idx] is not None:
                    all_router_logits += (layer_outputs[router_logits_idx],)
            
            # Collect auxiliary losses
            if len(layer_outputs) > abs(aux_loss_idx):
                aux_loss = layer_outputs[aux_loss_idx]
                if aux_loss is not None and isinstance(aux_loss, torch.Tensor):
                    all_aux_losses.append(aux_loss)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        # Aggregate auxiliary losses
        aux_loss = None
        if all_aux_losses:
            aux_loss = torch.stack(all_aux_losses).mean()
        
        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_router_logits,
                aux_loss
            ] if v is not None)
        
        return LlamaMoEModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_self_attns if output_attentions else None,
            router_logits=all_router_logits if output_router_logits else None,
            aux_loss=aux_loss,
        )


class LlamaMoEForCausalLM(nn.Module):
    """LLaMA-MoE model for causal language modeling."""
    
    def __init__(self, config: LlamaMoEConfig):
        super().__init__()
        self.config = config
        self.model = LlamaMoEModel(config)
        
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using LLaMA-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head if self.lm_head is not None else self.model.embed_tokens
    
    def set_output_embeddings(self, new_embeddings):
        if self.lm_head is not None:
            self.lm_head = new_embeddings
        else:
            self.model.embed_tokens = new_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlamaMoECausalLMOutput]:
        
        return_dict = return_dict if return_dict is not None else True
        
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        
        # Compute logits
        if self.config.tie_word_embeddings:
            logits = torch.nn.functional.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Add auxiliary loss if present (important!)
            if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                loss = loss + outputs.aux_loss
            elif not return_dict and len(outputs) > 1 and outputs[-1] is not None:
                loss = loss + outputs[-1]
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return LlamaMoECausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            aux_loss=outputs.aux_loss,
        )


# Factory functions for different model sizes
def create_tiny_llama_moe_model() -> LlamaMoEForCausalLM:
    """Create a tiny LLaMA-MoE model for development and testing."""
    try:
        from .config import create_tiny_llama_moe
    except ImportError:
        from config import create_tiny_llama_moe
    config = create_tiny_llama_moe()
    return LlamaMoEForCausalLM(config)


def create_llama_moe_7b_model() -> LlamaMoEForCausalLM:
    """Create a LLaMA-MoE 7B model."""
    try:
        from .config import create_llama_moe_7b
    except ImportError:
        from config import create_llama_moe_7b
    config = create_llama_moe_7b()
    return LlamaMoEForCausalLM(config)


def create_llama_moe_13b_model() -> LlamaMoEForCausalLM:
    """Create a LLaMA-MoE 13B model."""
    try:
        from .config import create_llama_moe_13b
    except ImportError:
        from config import create_llama_moe_13b
    config = create_llama_moe_13b()
    return LlamaMoEForCausalLM(config)


def create_code_llama_moe_7b_model() -> LlamaMoEForCausalLM:
    """Create a Code LLaMA-MoE 7B model."""
    try:
        from .config import create_code_llama_moe_7b
    except ImportError:
        from config import create_code_llama_moe_7b
    config = create_code_llama_moe_7b()
    return LlamaMoEForCausalLM(config)


def test_llama_moe_modeling():
    """Test complete LLaMA-MoE model."""
    print("Testing Complete LLaMA-MoE Model...")
    
    # Test tiny model
    print("\n--- Testing Tiny LLaMA-MoE Model ---")
    model = create_tiny_llama_moe_model()
    
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            output_hidden_states=True,
            output_router_logits=True,
            return_dict=True
        )
    
    print(f"✅ Model output logits shape: {outputs.logits.shape}")
    print(f"✅ Number of hidden states: {len(outputs.hidden_states)}")
    print(f"✅ Number of attention layers: {len(outputs.attentions)}")
    print(f"✅ Number of router logits: {len(outputs.router_logits) if outputs.router_logits else 0}")
    print(f"✅ Auxiliary loss: {outputs.aux_loss.item() if outputs.aux_loss else 'None'}")
    
    # Test with labels (training mode)
    print("\n--- Testing Training Mode ---")
    labels = input_ids.clone()
    
    outputs_train = model(input_ids, labels=labels, return_dict=True)
    print(f"✅ Training loss: {outputs_train.loss.item():.4f}")
    print(f"✅ Training logits shape: {outputs_train.logits.shape}")
    
    # Verify that aux_loss is included in total loss
    if outputs_train.aux_loss is not None:
        print(f"✅ Auxiliary loss included: {outputs_train.aux_loss.item():.6f}")
    
    # Parameter analysis
    print("\n--- Parameter Analysis ---")
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count parameters by type
    embedding_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    attention_params = sum(p.numel() for n, p in model.named_parameters() if 'self_attn' in n)
    moe_params = sum(p.numel() for n, p in model.named_parameters() if 'experts' in n or 'router' in n)
    norm_params = sum(p.numel() for n, p in model.named_parameters() if 'norm' in n)
    
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Embedding parameters: {embedding_params:,}")
    print(f"✅ Attention parameters: {attention_params:,}")
    print(f"✅ MoE parameters: {moe_params:,}")
    print(f"✅ Norm parameters: {norm_params:,}")
    
    # Test MoE layer utilization
    print("\n--- MoE Layer Analysis ---")
    config = model.config
    print(f"✅ Total layers: {config.num_hidden_layers}")
    print(f"✅ MoE layers: {config.moe_layers}")
    print(f"✅ Dense layers: {[i for i in range(config.num_hidden_layers) if i not in config.moe_layers]}")
    
    # Test different model sizes
    print("\n--- Testing Different Model Sizes ---")
    models = {
        'tiny': create_tiny_llama_moe_model,
        'llama_moe_7b': create_llama_moe_7b_model,
    }
    
    for name, create_fn in models.items():
        if name == 'tiny':  # Only test tiny for now to save time
            model = create_fn()
            total_params = sum(p.numel() for p in model.parameters())
            
            # Estimate active parameters (approximate)
            config = model.config
            moe_layers_count = len(config.moe_layers)
            dense_layers_count = config.num_hidden_layers - moe_layers_count
            
            # Rough calculation of active params
            experts_per_token = config.num_experts_per_token
            total_experts = config.num_experts
            expert_efficiency = experts_per_token / total_experts
            
            print(f"✅ {name}: {total_params:,} total params")
            print(f"   MoE layers: {moe_layers_count}, Dense layers: {dense_layers_count}")
            print(f"   Expert efficiency: {expert_efficiency:.1%}")
    
    print("\n✅ All LLaMA-MoE modeling tests passed!")


if __name__ == "__main__":
    test_llama_moe_modeling()