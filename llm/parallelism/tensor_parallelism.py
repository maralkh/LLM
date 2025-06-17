# training_infra/parallelism/tensor_parallel.py
"""
Tensor Parallelism implementation for large model training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import math

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

def initialize_tensor_parallel(tensor_parallel_size: int = 1):
    """Initialize tensor parallel groups"""
    
    if tensor_parallel_size == 1:
        return
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size % tensor_parallel_size != 0:
        raise ValueError(f"World size {world_size} is not divisible by tensor parallel size {tensor_parallel_size}")
    
    num_tensor_parallel_groups = world_size // tensor_parallel_size
    
    # Create tensor parallel groups
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _TENSOR_MODEL_PARALLEL_RANK
    
    for i in range(num_tensor_parallel_groups):
        ranks = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
        group = dist.new_group(ranks)
        
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_WORLD_SIZE = tensor_parallel_size
            _TENSOR_MODEL_PARALLEL_RANK = ranks.index(rank)

# Global variables for tensor parallel groups
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
_TENSOR_MODEL_PARALLEL_RANK = 0

def get_tensor_model_parallel_group():
    """Get tensor model parallel group"""
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_world_size():
    """Get tensor model parallel world size"""
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE

def get_tensor_model_parallel_rank():
    """Get tensor model parallel rank"""
    return _TENSOR_MODEL_PARALLEL_RANK

class _AllGather(torch.autograd.Function):
    """All-gather the input tensor across model parallel group"""
    
    @staticmethod
    def forward(ctx, input_tensor):
        group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        
        if world_size == 1:
            return input_tensor
        
        # Allocate output tensor
        output_tensor = torch.empty(
            (world_size,) + input_tensor.shape,
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )
        
        # All-gather
        output_list = list(torch.chunk(output_tensor, world_size, dim=0))
        dist.all_gather(output_list, input_tensor.contiguous(), group=group)
        
        # Reshape and return
        output = torch.cat(output_list, dim=-1)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        
        if world_size == 1:
            return grad_output
        
        # Split the gradient
        dim_size = grad_output.shape[-1] // world_size
        grad_input = grad_output[..., rank * dim_size:(rank + 1) * dim_size].contiguous()
        
        return grad_input

class _ReduceScatter(torch.autograd.Function):
    """Reduce-scatter the input tensor across model parallel group"""
    
    @staticmethod
    def forward(ctx, input_tensor):
        group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        
        if world_size == 1:
            return input_tensor
        
        # Split input tensor
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
        group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        
        if world_size == 1:
            return grad_output
        
        # All-gather gradients
        grad_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(grad_list, grad_output.contiguous(), group=group)
        
        # Concatenate
        grad_input = torch.cat(grad_list, dim=-1)
        return grad_input

def all_gather_tensor_parallel(input_tensor):
    """All-gather tensor across tensor parallel group"""
    return _AllGather.apply(input_tensor)

def reduce_scatter_tensor_parallel(input_tensor):
    """Reduce-scatter tensor across tensor parallel group"""
    return _ReduceScatter.apply(input_tensor)

class TensorParallelLinear(nn.Module):
    """Linear layer with tensor parallelism"""
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 gather_output: bool = True,
                 init_method: Optional[callable] = None,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 skip_bias_add: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        
        # Get tensor parallel info
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size
        self.output_size_per_partition = output_size // world_size
        
        # Parameters
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size_per_partition,
            dtype=torch.float32
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                dtype=torch.float32
            ))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self._initialize_weights(init_method)
    
    def _initialize_weights(self, init_method):
        """Initialize weights"""
        if init_method is None:
            # Default initialization
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.input_size
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            init_method(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def forward(self, input_tensor):
        """Forward pass"""
        # Matrix multiplication
        output_parallel = F.linear(input_tensor, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather outputs across tensor parallel group
            output = all_gather_tensor_parallel(output_parallel)
        else:
            output = output_parallel
        
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output

class ColumnParallelLinear(TensorParallelLinear):
    """Column-wise parallel linear layer"""
    
    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__(input_size, output_size, **kwargs)

class RowParallelLinear(nn.Module):
    """Row-wise parallel linear layer"""
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = False,
                 init_method: Optional[callable] = None,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 skip_bias_add: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        
        # Get tensor parallel info
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size // world_size
        self.output_size_per_partition = output_size
        
        # Parameters
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size_per_partition,
            dtype=torch.float32
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                dtype=torch.float32
            ))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self._initialize_weights(init_method)
    
    def _initialize_weights(self, init_method):
        """Initialize weights"""
        if init_method is None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.input_size
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            init_method(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def forward(self, input_tensor):
        """Forward pass"""
        # If input is not parallel, need to scatter it first
        if not self.input_is_parallel:
            input_parallel = reduce_scatter_tensor_parallel(input_tensor)
        else:
            input_parallel = input_tensor
        
        # Matrix multiplication
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-reduce across tensor parallel group
        group = get_tensor_model_parallel_group()
        if group is not None:
            dist.all_reduce(output_parallel, group=group)
        
        # Add bias
        if self.bias is not None:
            output = output_parallel + self.bias
        else:
            output = output_parallel
        
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
                 init_method: Optional[callable] = None,
                 keep_master_weight_for_test: bool = False):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        # Get tensor parallel info
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        
        # Partition vocabulary
        self.vocab_start_index = rank * num_embeddings // world_size
        self.vocab_end_index = (rank + 1) * num_embeddings // world_size
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        
        # Partition embedding dimension
        self.embedding_dim_per_partition = embedding_dim // world_size
        
        # Parameters
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition,
            self.embedding_dim_per_partition
        ))
        
        # Initialize
        self._initialize_weights(init_method)
    
    def _initialize_weights(self, init_method):
        """Initialize weights"""
        if init_method is None:
            nn.init.normal_(self.weight)
        else:
            init_method(self.weight)
    
    def forward(self, input_ids):
        """Forward pass"""
        # Mask input IDs outside current partition
        mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
        masked_input = input_ids - self.vocab_start_index
        masked_input[mask] = 0
        
        # Embedding lookup
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )
        
        # Mask out embeddings for out-of-range tokens
        output_parallel[mask] = 0.0
        
        # All-gather across tensor parallel group
        output = all_gather_tensor_parallel(output_parallel)
        
        return output

class TensorParallelAttention(nn.Module):
    """Multi-head attention with tensor parallelism"""
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout: float = 0.1,
                 init_method: Optional[callable] = None,
                 output_layer_init_method: Optional[callable] = None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        
        # Get tensor parallel info
        world_size = get_tensor_model_parallel_world_size()
        self.num_attention_heads_per_partition = num_attention_heads // world_size
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        self.hidden_size_per_partition = hidden_size // world_size
        
        # Q, K, V projections
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,
            init_method=init_method
        )
        
        # Output projection
        self.dense = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            init_method=output_layer_init_method
        )
        
        # Dropout
        self.attention_dropout_layer = nn.Dropout(attention_dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        """Forward pass"""
        batch_size, seq_length, _ = hidden_states.shape
        
        # QKV projection
        mixed_x_layer = self.query_key_value(hidden_states)
        
        # Reshape and split Q, K, V
        new_tensor_shape = (
            batch_size,
            seq_length,
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        
        query_layer, key_layer, value_layer = torch.chunk(mixed_x_layer, 3, dim=-1)
        
        # Transpose for attention computation
        query_layer = query_layer.permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)
        
        # Attention computation
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores += attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout_layer(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Transpose back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape
        new_context_layer_shape = (
            batch_size,
            seq_length,
            self.hidden_size_per_partition
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        output = self.dense(context_layer)
        
        return output

def convert_to_tensor_parallel(model, config: TensorParallelConfig):
    """Convert regular model to tensor parallel version"""
    
    if config.tensor_parallel_size == 1:
        return model
    
    # Initialize tensor parallel groups
    initialize_tensor_parallel(config.tensor_parallel_size)
    
    # Replace layers with tensor parallel versions
    def replace_linear_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Determine if this should be column or row parallel
                # This is model-specific logic
                if 'attention' in name.lower() and ('q' in name or 'k' in name or 'v' in name):
                    # Query, Key, Value projections are column parallel
                    new_layer = ColumnParallelLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        gather_output=False
                    )
                elif 'attention' in name.lower() and 'out' in name:
                    # Output projection is row parallel
                    new_layer = RowParallelLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        input_is_parallel=True
                    )
                elif 'mlp' in name.lower() and ('gate' in name or 'up' in name):
                    # MLP gate and up projections are column parallel
                    new_layer = ColumnParallelLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        gather_output=False
                    )
                elif 'mlp' in name.lower() and 'down' in name:
                    # MLP down projection is row parallel
                    new_layer = RowParallelLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        input_is_parallel=True
                    )
                else:
                    # Default to column parallel
                    new_layer = ColumnParallelLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None
                    )
                
                setattr(module, name, new_layer)
            elif isinstance(child, nn.Embedding):
                # Replace embedding with tensor parallel version
                new_embedding = TensorParallelEmbedding(
                    child.num_embeddings,
                    child.embedding_dim,
                    padding_idx=child.padding_idx
                )
                setattr(module, name, new_embedding)
            else:
                # Recursively process child modules
                replace_linear_layers(child)
    
    replace_linear_layers(model)
    return model

# Utility functions
def get_tensor_parallel_src_rank():
    """Get the source rank for tensor parallel communication"""
    return get_tensor_model_parallel_rank()

def tensor_parallel_all_reduce(tensor):
    """All-reduce tensor across tensor parallel group"""
    group = get_tensor_model_parallel_group()
    if group is not None:
        dist.all_reduce(tensor, group=group)
    return tensor

def tensor_parallel_all_gather_object(obj):
    """All-gather object across tensor parallel group"""
    group = get_tensor_model_parallel_group()
    world_size = get_tensor_model_parallel_world_size()
    
    if world_size == 1:
        return [obj]
    
    # Gather objects
    output = [None] * world_size
    dist.all_gather_object(output, obj, group=group)
    return output

# Export tensor parallel components
__all__ = [
    'TensorParallelConfig',
    'TensorParallelLinear',
    'ColumnParallelLinear', 
    'RowParallelLinear',
    'TensorParallelEmbedding',
    'TensorParallelAttention',
    'initialize_tensor_parallel',
    'convert_to_tensor_parallel',
    'get_tensor_model_parallel_group',
    'get_tensor_model_parallel_world_size',
    'get_tensor_model_parallel_rank',
    'all_gather_tensor_parallel',
    'reduce_scatter_tensor_parallel',
    'tensor_parallel_all_reduce'
]