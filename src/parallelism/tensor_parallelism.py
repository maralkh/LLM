# training_infra/parallelism/tensor_parallel.py
"""
Tensor Parallelism implementation for large model training
Uses centralized distributed initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Union, Tuple, Callable
from dataclasses import dataclass
import math
import logging

# Import centralized distributed functions
try:
    from .distributed_init import (
        get_tensor_parallel_group,
        get_tensor_parallel_size,
        get_tensor_parallel_rank,
        get_world_size,
        get_rank
    )
    _CENTRALIZED_INIT_AVAILABLE = True
except ImportError:
    logging.warning("Centralized initialization not available, using fallback")
    _CENTRALIZED_INIT_AVAILABLE = False
    
    # Fallback implementations
    def get_tensor_parallel_group():
        return None
    
    def get_tensor_parallel_size():
        return 1
    
    def get_tensor_parallel_rank():
        return 0
    
    def get_world_size():
        return dist.get_world_size() if dist.is_initialized() else 1
    
    def get_rank():
        return dist.get_rank() if dist.is_initialized() else 0

@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism"""
    tensor_parallel_size: int = 1
    sequence_parallel: bool = False
    gather_output: bool = True
    init_method_std: float = 0.02
    use_cpu_initialization: bool = False
    perform_initialization: bool = True
    gradient_accumulation_fusion: bool = False
    async_tensor_model_parallel_allreduce: bool = False
    use_sequence_parallel: bool = False
    overlap_grad_reduce: bool = True

class _AllGather(torch.autograd.Function):
    """All-gather the input tensor across model parallel group"""
    
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        if group is None:
            group = get_tensor_parallel_group()
        
        world_size = get_tensor_parallel_size()
        
        if world_size == 1 or group is None:
            return input_tensor
        
        # Store group for backward pass
        ctx.group = group
        ctx.world_size = world_size
        
        # Allocate output tensor
        output_shape = list(input_tensor.shape)
        output_shape[-1] *= world_size
        
        output_tensor = torch.empty(
            output_shape,
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )
        
        # Split output tensor for all_gather
        output_list = list(torch.chunk(output_tensor, world_size, dim=-1))
        
        # All-gather
        dist.all_gather(output_list, input_tensor.contiguous(), group=group)
        
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = ctx.world_size
        rank = get_tensor_parallel_rank()
        
        if world_size == 1 or group is None:
            return grad_output, None
        
        # Split the gradient along the last dimension
        dim_size = grad_output.shape[-1] // world_size
        grad_input = grad_output[..., rank * dim_size:(rank + 1) * dim_size].contiguous()
        
        return grad_input, None

class _ReduceScatter(torch.autograd.Function):
    """Reduce-scatter the input tensor across model parallel group"""
    
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        if group is None:
            group = get_tensor_parallel_group()
            
        world_size = get_tensor_parallel_size()
        rank = get_tensor_parallel_rank()
        
        if world_size == 1 or group is None:
            return input_tensor
        
        # Store for backward pass
        ctx.group = group
        ctx.world_size = world_size
        
        # Split input tensor along last dimension
        dim_size = input_tensor.shape[-1] // world_size
        input_list = [
            input_tensor[..., i * dim_size:(i + 1) * dim_size].contiguous()
            for i in range(world_size)
        ]
        
        # Reduce-scatter
        output_tensor = torch.empty_like(input_list[rank])
        dist.reduce_scatter(output_tensor, input_list, group=group)
        
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = ctx.world_size
        
        if world_size == 1 or group is None:
            return grad_output, None
        
        # All-gather gradients
        grad_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(grad_list, grad_output.contiguous(), group=group)
        
        # Concatenate along last dimension
        grad_input = torch.cat(grad_list, dim=-1)
        return grad_input, None

class _AllReduce(torch.autograd.Function):
    """All-reduce the input tensor across model parallel group"""
    
    @staticmethod
    def forward(ctx, input_tensor, group=None):
        if group is None:
            group = get_tensor_parallel_group()
            
        world_size = get_tensor_parallel_size()
        
        if world_size == 1 or group is None:
            return input_tensor
        
        # Store for backward pass
        ctx.group = group
        
        # All-reduce
        output = input_tensor.clone()
        dist.all_reduce(output, group=group)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        
        if group is None:
            return grad_output, None
        
        # All-reduce gradients
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, group=group)
        
        return grad_input, None

# Public API functions
def all_gather_tensor_parallel(input_tensor, group=None):
    """All-gather tensor across tensor parallel group"""
    return _AllGather.apply(input_tensor, group)

def reduce_scatter_tensor_parallel(input_tensor, group=None):
    """Reduce-scatter tensor across tensor parallel group"""
    return _ReduceScatter.apply(input_tensor, group)

def all_reduce_tensor_parallel(input_tensor, group=None):
    """All-reduce tensor across tensor parallel group"""
    return _AllReduce.apply(input_tensor, group)

class TensorParallelLinear(nn.Module):
    """Base class for tensor parallel linear layers"""
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 gather_output: bool = True,
                 init_method: Optional[Callable] = None,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 skip_bias_add: bool = False,
                 config: Optional[TensorParallelConfig] = None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.config = config or TensorParallelConfig()
        
        # Get tensor parallel info
        self.world_size = get_tensor_parallel_size()
        self.rank = get_tensor_parallel_rank()
        self.group = get_tensor_parallel_group()
        
        # This will be overridden in subclasses
        self.input_size_per_partition = input_size
        self.output_size_per_partition = output_size
        
        # Initialize in subclasses
        self.weight = None
        self.bias = None
    
    def _initialize_weight(self, weight_shape: Tuple[int, ...], init_method: Optional[Callable] = None):
        """Initialize weight tensor"""
        self.weight = nn.Parameter(torch.empty(
            weight_shape,
            dtype=torch.float32,
            device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        ))
        
        if init_method is None:
            # Default Kaiming initialization
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            init_method(self.weight)
    
    def _initialize_bias(self, bias_shape: Tuple[int, ...]):
        """Initialize bias tensor"""
        if bias_shape:
            self.bias = nn.Parameter(torch.empty(
                bias_shape,
                dtype=torch.float32,
                device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
            ))
            
            # Initialize bias to zero
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

class ColumnParallelLinear(TensorParallelLinear):
    """Column-wise parallel linear layer"""
    
    def __init__(self, input_size: int, output_size: int, bias: bool = True, **kwargs):
        super().__init__(input_size, output_size, bias, **kwargs)
        
        # Column parallel: split output dimension
        assert output_size % self.world_size == 0, f"Output size {output_size} not divisible by world size {self.world_size}"
        
        self.input_size_per_partition = input_size
        self.output_size_per_partition = output_size // self.world_size
        
        # Initialize weight and bias
        self._initialize_weight((self.output_size_per_partition, self.input_size_per_partition))
        if bias:
            self._initialize_bias((self.output_size_per_partition,))
        else:
            self._initialize_bias(())
    
    def forward(self, input_tensor):
        """Forward pass for column parallel linear layer"""
        # Linear transformation
        output_parallel = F.linear(input_tensor, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather outputs across tensor parallel group
            output = all_gather_tensor_parallel(output_parallel, self.group)
        else:
            output = output_parallel
        
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output

class RowParallelLinear(TensorParallelLinear):
    """Row-wise parallel linear layer"""
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = False,
                 **kwargs):
        super().__init__(input_size, output_size, bias, **kwargs)
        
        self.input_is_parallel = input_is_parallel
        
        # Row parallel: split input dimension
        assert input_size % self.world_size == 0, f"Input size {input_size} not divisible by world size {self.world_size}"
        
        self.input_size_per_partition = input_size // self.world_size
        self.output_size_per_partition = output_size
        
        # Initialize weight and bias
        self._initialize_weight((self.output_size_per_partition, self.input_size_per_partition))
        if bias:
            self._initialize_bias((self.output_size_per_partition,))
        else:
            self._initialize_bias(())
    
    def forward(self, input_tensor):
        """Forward pass for row parallel linear layer"""
        # Split input if not already parallel
        if not self.input_is_parallel:
            input_parallel = reduce_scatter_tensor_parallel(input_tensor, self.group)
        else:
            input_parallel = input_tensor
        
        # Linear transformation
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-reduce across tensor parallel group
        output = all_reduce_tensor_parallel(output_parallel, self.group)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output

class TensorParallelEmbedding(nn.Module):
    """Embedding layer with tensor parallelism"""
    
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 init_method: Optional[Callable] = None,
                 config: Optional[TensorParallelConfig] = None):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.config = config or TensorParallelConfig()
        
        # Get tensor parallel info
        self.world_size = get_tensor_parallel_size()
        self.rank = get_tensor_parallel_rank()
        self.group = get_tensor_parallel_group()
        
        # Decide parallelism strategy
        if self.config.sequence_parallel or embedding_dim >= num_embeddings:
            # Parallelize embedding dimension
            self._parallelize_embedding_dim()
        else:
            # Parallelize vocabulary dimension
            self._parallelize_vocab_dim()
    
    def _parallelize_embedding_dim(self):
        """Parallelize along embedding dimension"""
        assert self.embedding_dim % self.world_size == 0, \
            f"Embedding dim {self.embedding_dim} not divisible by world size {self.world_size}"
        
        self.num_embeddings_per_partition = self.num_embeddings
        self.embedding_dim_per_partition = self.embedding_dim // self.world_size
        self.vocab_start_index = 0
        self.vocab_end_index = self.num_embeddings
        self.parallel_embedding_dim = True
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition,
            self.embedding_dim_per_partition
        ))
    
    def _parallelize_vocab_dim(self):
        """Parallelize along vocabulary dimension"""
        assert self.num_embeddings % self.world_size == 0, \
            f"Vocab size {self.num_embeddings} not divisible by world size {self.world_size}"
        
        self.num_embeddings_per_partition = self.num_embeddings // self.world_size
        self.embedding_dim_per_partition = self.embedding_dim
        self.vocab_start_index = self.rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.parallel_embedding_dim = False
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition,
            self.embedding_dim_per_partition
        ))
    
    def forward(self, input_ids):
        """Forward pass"""
        if self.parallel_embedding_dim:
            # Embedding dimension is parallelized
            output_parallel = F.embedding(
                input_ids, self.weight, self.padding_idx,
                self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            # All-gather across embedding dimension
            output = all_gather_tensor_parallel(output_parallel, self.group)
        else:
            # Vocabulary is parallelized
            # Mask input IDs outside current partition
            mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
            masked_input = input_ids - self.vocab_start_index
            masked_input[mask] = 0
            
            # Embedding lookup
            output_parallel = F.embedding(
                masked_input, self.weight, self.padding_idx,
                self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            
            # Mask out embeddings for out-of-range tokens
            output_parallel[mask] = 0.0
            
            # All-reduce to sum contributions from all partitions
            output = all_reduce_tensor_parallel(output_parallel, self.group)
        
        return output

class TensorParallelMultiHeadAttention(nn.Module):
    """Multi-head attention with tensor parallelism"""
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout: float = 0.1,
                 bias: bool = True,
                 init_method: Optional[Callable] = None,
                 output_layer_init_method: Optional[Callable] = None,
                 config: Optional[TensorParallelConfig] = None):
        super().__init__()
        
        self.config = config or TensorParallelConfig()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        
        # Get tensor parallel info
        self.world_size = get_tensor_parallel_size()
        
        assert num_attention_heads % self.world_size == 0, \
            f"Number of attention heads {num_attention_heads} not divisible by world size {self.world_size}"
        
        self.num_attention_heads_per_partition = num_attention_heads // self.world_size
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        
        # Q, K, V projections (column parallel)
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            3 * self.hidden_size_per_partition,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            config=self.config
        )
        
        # Output projection (row parallel)
        self.dense = RowParallelLinear(
            self.hidden_size_per_partition,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            config=self.config
        )
        
        # Dropout
        self.attention_dropout_layer = nn.Dropout(attention_dropout)
        
        # Scale factor for attention scores
        self.norm_factor = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        """Forward pass"""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # QKV projection
        mixed_x_layer = self.query_key_value(hidden_states)
        
        # Reshape for multi-head attention
        new_tensor_shape = (
            batch_size,
            seq_length,
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        
        # Split into Q, K, V
        query_layer, key_layer, value_layer = torch.chunk(mixed_x_layer, 3, dim=-1)
        
        # Transpose for attention computation: [batch, heads, seq, head_dim]
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)
        
        # Handle past key-value for caching
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_layer = torch.cat([past_key, key_layer], dim=-2)
            value_layer = torch.cat([past_value, value_layer], dim=-2)
        
        # Cache current key-value if requested
        if use_cache:
            present_key_value = (key_layer, value_layer)
        else:
            present_key_value = None
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.norm_factor
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout_layer(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Transpose back: [batch, seq, heads, head_dim]
        context_layer = context_layer.transpose(1, 2).contiguous()
        
        # Reshape: [batch, seq, hidden_size_per_partition]
        new_context_layer_shape = (
            batch_size,
            seq_length,
            self.hidden_size_per_partition
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        output = self.dense(context_layer)
        
        if use_cache:
            return output, present_key_value
        else:
            return output

class TensorParallelMLP(nn.Module):
    """MLP with tensor parallelism"""
    
    def __init__(self,
                 hidden_size: int,
                 ffn_hidden_size: Optional[int] = None,
                 bias: bool = True,
                 activation_func: str = "gelu",
                 init_method: Optional[Callable] = None,
                 config: Optional[TensorParallelConfig] = None):
        super().__init__()
        
        self.config = config or TensorParallelConfig()
        
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size
        
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        
        # Up projection (column parallel)
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            config=self.config
        )
        
        # Down projection (row parallel)
        self.dense_4h_to_h = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            init_method=init_method,
            config=self.config
        )
        
        # Activation function
        if activation_func == "gelu":
            self.activation_func = F.gelu
        elif activation_func == "relu":
            self.activation_func = F.relu
        elif activation_func == "swish" or activation_func == "silu":
            self.activation_func = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {activation_func}")
    
    def forward(self, hidden_states):
        """Forward pass"""
        # Up projection
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        
        # Activation
        intermediate_parallel = self.activation_func(intermediate_parallel)
        
        # Down projection
        output = self.dense_4h_to_h(intermediate_parallel)
        
        return output

class TensorParallelTransformerLayer(nn.Module):
    """Complete transformer layer with tensor parallelism"""
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 ffn_hidden_size: Optional[int] = None,
                 attention_dropout: float = 0.1,
                 hidden_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5,
                 bias: bool = True,
                 activation_func: str = "gelu",
                 config: Optional[TensorParallelConfig] = None):
        super().__init__()
        
        self.config = config or TensorParallelConfig()
        self.hidden_size = hidden_size
        
        # Self-attention
        self.self_attention = TensorParallelMultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            config=self.config
        )
        
        # MLP
        self.mlp = TensorParallelMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=bias,
            activation_func=activation_func,
            config=self.config
        )
        
        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Dropout
        self.hidden_dropout = nn.Dropout(hidden_dropout)
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        """Forward pass"""
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if use_cache:
            attention_output, present_key_value = self.self_attention(
                hidden_states, attention_mask, past_key_value, use_cache=True
            )
        else:
            attention_output = self.self_attention(
                hidden_states, attention_mask, past_key_value, use_cache=False
            )
            present_key_value = None
        
        attention_output = self.hidden_dropout(attention_output)
        hidden_states = residual + attention_output
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.hidden_dropout(mlp_output)
        hidden_states = residual + mlp_output
        
        if use_cache:
            return hidden_states, present_key_value
        else:
            return hidden_states

def convert_layer_to_tensor_parallel(layer: nn.Module, 
                                   config: TensorParallelConfig) -> nn.Module:
    """Convert a single layer to tensor parallel version"""
    
    if get_tensor_parallel_size() == 1:
        return layer
    
    if isinstance(layer, nn.Linear):
        # Determine the type of linear layer based on its position/name
        # This is a heuristic and might need adjustment based on the specific model
        in_features = layer.in_features
        out_features = layer.out_features
        has_bias = layer.bias is not None
        
        # For most cases, use column parallel for projections that increase dimensionality
        # and row parallel for projections that decrease dimensionality
        if out_features > in_features:
            # Likely an up-projection (e.g., in MLP)
            new_layer = ColumnParallelLinear(
                in_features, out_features, bias=has_bias, 
                gather_output=False, config=config
            )
        else:
            # Likely a down-projection or attention output
            new_layer = RowParallelLinear(
                in_features, out_features, bias=has_bias,
                input_is_parallel=True, config=config
            )
        
        # Copy weights if possible (this is tricky with tensor parallelism)
        # In practice, you'd want to properly partition and copy the weights
        logging.warning("Weight copying not implemented for tensor parallel conversion")
        
        return new_layer
    
    elif isinstance(layer, nn.Embedding):
        new_layer = TensorParallelEmbedding(
            layer.num_embeddings,
            layer.embedding_dim,
            padding_idx=layer.padding_idx,
            config=config
        )
        # Similar weight copying issue
        logging.warning("Weight copying not implemented for tensor parallel embedding conversion")
        return new_layer
    
    else:
        # For other layer types, return as-is
        return layer

def convert_model_to_tensor_parallel(model: nn.Module, 
                                   config: TensorParallelConfig) -> nn.Module:
    """Convert entire model to tensor parallel version"""
    
    if get_tensor_parallel_size() == 1:
        logging.info("Tensor parallel size is 1, no conversion needed")
        return model
    
    logging.info(f"Converting model to tensor parallel with size {get_tensor_parallel_size()}")
    
    def replace_layers_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, nn.Embedding)):
                new_layer = convert_layer_to_tensor_parallel(child, config)
                setattr(module, name, new_layer)
            else:
                # Recursively process child modules
                replace_layers_recursive(child)
    
    # Make a copy to avoid modifying the original
    import copy
    model_copy = copy.deepcopy(model)
    replace_layers_recursive(model_copy)
    
    return model_copy

# Utility functions for tensor parallel operations
def sync_tensor_parallel_parameters(model: nn.Module):
    """Synchronize parameters across tensor parallel ranks"""
    group = get_tensor_parallel_group()
    if group is None or get_tensor_parallel_size() == 1:
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=group)

def get_tensor_parallel_vocab_range():
    """Get vocabulary range for current tensor parallel rank"""
    world_size = get_tensor_parallel_size()
    rank = get_tensor_parallel_rank()
    
    if world_size == 1:
        return None, None
    
    # This is a simplified version - in practice, you'd get this from the embedding layer
    vocab_size = 50000  # Example vocab size
    vocab_per_rank = vocab_size // world_size
    
    start_idx = rank * vocab_per_rank
    end_idx = start_idx + vocab_per_rank
    
    return start_idx, end_idx

def estimate_tensor_parallel_memory_usage(model_params: int, 
                                        config: TensorParallelConfig) -> dict:
    """Estimate memory usage with tensor parallelism"""
    
    world_size = get_tensor_parallel_size()
    
    # Model parameters are distributed across tensor parallel ranks
    params_per_rank = model_params / world_size
    
    # Memory estimation (in GB)
    memory_breakdown = {
        "model_weights_gb": (params_per_rank * 2) / (1024**3),  # FP16
        "gradients_gb": (params_per_rank * 2) / (1024**3),      # FP16
        "optimizer_states_gb": (params_per_rank * 8) / (1024**3),  # Adam FP32
        "activations_gb": 2.0,  # Rough estimate
        "communication_buffers_gb": 0.5,  # For all-gather/reduce-scatter
        "framework_overhead_gb": 1.0
    }
    
    memory_breakdown["total_per_gpu_gb"] = sum(memory_breakdown.values())
    
    # Communication overhead estimation
    memory_breakdown["communication_overhead_percent"] = min(10.0, world_size * 1.5)
    
    return memory_breakdown

# Export all public components
__all__ = [
    # Configuration
    'TensorParallelConfig',
    
    # Core layers
    'TensorParallelLinear',
    'ColumnParallelLinear', 
    'RowParallelLinear',
    'TensorParallelEmbedding',
    'TensorParallelMultiHeadAttention',
    'TensorParallelMLP',
    'TensorParallelTransformerLayer',
    
    # Communication primitives
    'all_gather_tensor_parallel',
    'reduce_scatter_tensor_parallel',
    'all_reduce_tensor_parallel',
    
    # Model conversion utilities
    'convert_layer_to_tensor_parallel',
    'convert_model_to_tensor_parallel',
    
    # Utility functions
    'sync_tensor_parallel_parameters',
    'get_tensor_parallel_vocab_range',
    'estimate_tensor_parallel_memory_usage',
    
    # Distributed state functions (re-exported for convenience)
    'get_tensor_parallel_group',
    'get_tensor_parallel_size',
    'get_tensor_parallel_rank'
]

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # This would typically be done after initializing distributed training
    # with the centralized initialization system
    
    print("Tensor Parallel Module Test")
    print(f"World size: {get_tensor_parallel_size()}")
    print(f"Rank: {get_tensor_parallel_rank()}")
    
    # Example model conversion
    if torch.cuda.is_available() and get_tensor_parallel_size() > 1:
        # Create a simple test model
        test_model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512)
        )
        
        config = TensorParallelConfig(tensor_parallel_size=get_tensor_parallel_size())
        
        # Convert to tensor parallel
        tp_model = convert_model_to_tensor_parallel(test_model, config)
        
        print("Model converted to tensor parallel")
        print(f"Original model parameters: {sum(p.numel() for p in test_model.parameters())}")
        print(f"TP model parameters per rank: {sum(p.numel() for p in tp_model.parameters())}")
    
    # Memory estimation example
    memory_usage = estimate_tensor_parallel_memory_usage(7_000_000_000, TensorParallelConfig())
    print(f"\nMemory usage estimation for 7B model:")
    for key, value in memory_usage.items():
        if key.endswith('_gb'):
            print(f"  {key}: {value:.2f} GB")
        else:
            print(f"  {key}: {value}")