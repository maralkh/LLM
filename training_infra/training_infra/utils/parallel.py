"""Parallel linear layers for distributed training.

This module provides tensor parallel implementations that can be drop-in
replacements for standard PyTorch linear layers when distributed training
is enabled.
"""

from typing import Optional, Any, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# Global state for parallel configuration
_TENSOR_PARALLEL_SIZE = 1
_TENSOR_PARALLEL_RANK = 0
_TENSOR_PARALLEL_GROUP = None


def set_tensor_parallel_size(size: int):
    """Set tensor parallel size."""
    global _TENSOR_PARALLEL_SIZE
    _TENSOR_PARALLEL_SIZE = size


def set_tensor_parallel_rank(rank: int):
    """Set tensor parallel rank."""
    global _TENSOR_PARALLEL_RANK
    _TENSOR_PARALLEL_RANK = rank


def set_tensor_parallel_group(group):
    """Set tensor parallel process group."""
    global _TENSOR_PARALLEL_GROUP
    _TENSOR_PARALLEL_GROUP = group


def get_tensor_parallel_size() -> int:
    """Get current tensor parallel size."""
    return _TENSOR_PARALLEL_SIZE


def get_tensor_parallel_rank() -> int:
    """Get current tensor parallel rank."""
    return _TENSOR_PARALLEL_RANK


def get_tensor_parallel_group():
    """Get current tensor parallel group."""
    return _TENSOR_PARALLEL_GROUP


class VocabParallelEmbedding(nn.Module):
    """Vocabulary parallel embedding layer.
    
    Splits the vocabulary across tensor parallel ranks.
    Each rank handles a subset of the vocabulary.
    """
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        padding_idx: Optional[int] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Calculate vocabulary partition
        self.tensor_parallel_size = get_tensor_parallel_size()
        self.tensor_parallel_rank = get_tensor_parallel_rank()
        
        if self.tensor_parallel_size > 1:
            # Partition vocabulary
            vocab_per_rank = num_embeddings // self.tensor_parallel_size
            vocab_remainder = num_embeddings % self.tensor_parallel_size
            
            self.vocab_start_index = self.tensor_parallel_rank * vocab_per_rank
            if self.tensor_parallel_rank < vocab_remainder:
                self.vocab_start_index += self.tensor_parallel_rank
                self.vocab_end_index = self.vocab_start_index + vocab_per_rank + 1
            else:
                self.vocab_start_index += vocab_remainder
                self.vocab_end_index = self.vocab_start_index + vocab_per_rank
                
            self.num_embeddings_per_rank = self.vocab_end_index - self.vocab_start_index
        else:
            # No parallelism
            self.vocab_start_index = 0
            self.vocab_end_index = num_embeddings
            self.num_embeddings_per_rank = num_embeddings
            
        # Create embedding table for this rank's vocabulary
        self.weight = Parameter(torch.empty(
            self.num_embeddings_per_rank, embedding_dim, dtype=dtype
        ))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                # Only zero out padding if it's in this rank's vocabulary
                if self.vocab_start_index <= self.padding_idx < self.vocab_end_index:
                    local_padding_idx = self.padding_idx - self.vocab_start_index
                    self.weight[local_padding_idx].fill_(0)
                    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with vocabulary parallelism."""
        if self.tensor_parallel_size == 1:
            # No parallelism - standard embedding
            return F.embedding(input_ids, self.weight, self.padding_idx)
            
        # Create mask for tokens handled by this rank
        mask = (input_ids >= self.vocab_start_index) & (input_ids < self.vocab_end_index)
        
        # Convert global token IDs to local token IDs
        local_input_ids = input_ids - self.vocab_start_index
        local_input_ids = torch.where(mask, local_input_ids, 0)
        
        # Compute embeddings for local vocabulary
        embeddings = F.embedding(local_input_ids, self.weight, None)
        
        # Zero out embeddings for tokens not handled by this rank
        embeddings = torch.where(mask.unsqueeze(-1), embeddings, torch.zeros_like(embeddings))
        
        # All-reduce to get complete embeddings
        if self.tensor_parallel_size > 1 and get_tensor_parallel_group() is not None:
            torch.distributed.all_reduce(embeddings, group=get_tensor_parallel_group())
            
        return embeddings


class ColumnParallelLinear(nn.Module):
    """Column parallel linear layer.
    
    Splits the weight matrix along the column dimension.
    Input: [batch_size, input_size]
    Output: [batch_size, output_size // tensor_parallel_size]
    """
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        bias: bool = True,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        # Calculate output partition
        self.tensor_parallel_size = get_tensor_parallel_size()
        self.tensor_parallel_rank = get_tensor_parallel_rank()
        
        if self.tensor_parallel_size > 1:
            assert output_size % self.tensor_parallel_size == 0, \
                f"output_size ({output_size}) must be divisible by tensor_parallel_size ({self.tensor_parallel_size})"
            self.output_size_per_partition = output_size // self.tensor_parallel_size
        else:
            self.output_size_per_partition = output_size
            
        # Weight and bias
        self.weight = Parameter(torch.empty(
            self.output_size_per_partition, input_size, dtype=dtype
        ))
        
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and bias."""
        # Use the same initialization as standard PyTorch Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with column parallelism."""
        # Linear transformation
        output = F.linear(input_tensor, self.weight, self.bias)
        
        if self.gather_output and self.tensor_parallel_size > 1:
            # Gather outputs from all ranks
            if get_tensor_parallel_group() is not None:
                output_list = [torch.zeros_like(output) for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather(output_list, output, group=get_tensor_parallel_group())
                output = torch.cat(output_list, dim=-1)
                
        return output


class RowParallelLinear(nn.Module):
    """Row parallel linear layer.
    
    Splits the weight matrix along the row dimension.
    Input: [batch_size, input_size // tensor_parallel_size] (already partitioned)
    Output: [batch_size, output_size]
    """
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        bias: bool = True,
        input_is_parallel: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        
        # Calculate input partition
        self.tensor_parallel_size = get_tensor_parallel_size()
        self.tensor_parallel_rank = get_tensor_parallel_rank()
        
        if self.tensor_parallel_size > 1 and input_is_parallel:
            assert input_size % self.tensor_parallel_size == 0, \
                f"input_size ({input_size}) must be divisible by tensor_parallel_size ({self.tensor_parallel_size})"
            self.input_size_per_partition = input_size // self.tensor_parallel_size
        else:
            self.input_size_per_partition = input_size
            
        # Weight and bias
        self.weight = Parameter(torch.empty(
            output_size, self.input_size_per_partition, dtype=dtype
        ))
        
        if bias:
            # Only rank 0 has bias to avoid duplicate addition
            if self.tensor_parallel_rank == 0:
                self.bias = Parameter(torch.empty(output_size, dtype=dtype))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and bias."""
        # Use the same initialization as standard PyTorch Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with row parallelism."""
        # Linear transformation
        output = F.linear(input_tensor, self.weight)
        
        if self.tensor_parallel_size > 1:
            # All-reduce to sum outputs from all ranks
            if get_tensor_parallel_group() is not None:
                torch.distributed.all_reduce(output, group=get_tensor_parallel_group())
                
        # Add bias (only rank 0 has bias)
        if self.bias is not None:
            output = output + self.bias
            
        return output


class ParallelMLP(nn.Module):
    """MLP with tensor parallelism.
    
    Combines column and row parallel layers for efficient MLP computation.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        activation_fn,
        bias: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        # Column parallel for gate and up projections
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=bias, gather_output=False, dtype=dtype
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=bias, gather_output=False, dtype=dtype
        )
        
        # Row parallel for down projection
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=bias, input_is_parallel=True, dtype=dtype
        )
        
        self.activation_fn = activation_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU and tensor parallelism."""
        gate = self.activation_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# Utility functions for easy integration
def make_parallel_if_available(layer_type: str, *args, **kwargs):
    """Create parallel layer if tensor parallelism is enabled, otherwise standard layer."""
    if get_tensor_parallel_size() > 1:
        if layer_type == "column":
            return ColumnParallelLinear(*args, **kwargs)
        elif layer_type == "row":
            return RowParallelLinear(*args, **kwargs)
        elif layer_type == "embedding":
            return VocabParallelEmbedding(*args, **kwargs)
    
    # Fallback to standard layers
    if layer_type in ["column", "row"]:
        input_size, output_size = args[:2]
        bias = kwargs.get('bias', True)
        return nn.Linear(input_size, output_size, bias=bias)
    elif layer_type == "embedding":
        return nn.Embedding(*args, **kwargs)


# Testing function
def test_parallel_layers():
    """Test parallel layer implementations."""
    print("🧪 Testing Parallel Layers...")
    
    # Test without parallelism (should work like standard layers)
    set_tensor_parallel_size(1)
    
    # Test ColumnParallelLinear
    col_linear = ColumnParallelLinear(256, 512, bias=True)
    x = torch.randn(4, 16, 256)
    out = col_linear(x)
    assert out.shape == (4, 16, 512)
    print("   ✅ ColumnParallelLinear working")
    
    # Test RowParallelLinear
    row_linear = RowParallelLinear(512, 256, bias=True, input_is_parallel=False)
    out2 = row_linear(out)
    assert out2.shape == (4, 16, 256)
    print("   ✅ RowParallelLinear working")
    
    # Test VocabParallelEmbedding
    embedding = VocabParallelEmbedding(1000, 256, padding_idx=0)
    input_ids = torch.randint(0, 1000, (4, 16))
    emb_out = embedding(input_ids)
    assert emb_out.shape == (4, 16, 256)
    print("   ✅ VocabParallelEmbedding working")
    
    # Test ParallelMLP
    mlp = ParallelMLP(256, 512, torch.nn.functional.silu, bias=False)
    mlp_out = mlp(emb_out)
    assert mlp_out.shape == (4, 16, 256)
    print("   ✅ ParallelMLP working")
    
    print("✅ Parallel layers tests passed!")


if __name__ == "__main__":
    test_parallel_layers()