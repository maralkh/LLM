# training_infra/parallelism/pipeline_parallel.py
"""
Pipeline Parallelism implementation for large model training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipeline.sync import Pipe
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import threading
import queue
import time
from collections import defaultdict
import logging

@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism"""
    pipeline_parallel_size: int = 1
    micro_batch_size: int = 1
    chunks: int = 1
    checkpoint: str = "never"  # "always", "except_last", "never"
    deferred_batch_norm: bool = False
    balance: Optional[List[int]] = None
    devices: Optional[List[torch.device]] = None
    worker_map: Optional[Dict[int, str]] = None
    input_device: Optional[torch.device] = None

# Global variables for pipeline parallel groups
_PIPELINE_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 1
_PIPELINE_MODEL_PARALLEL_RANK = 0

def initialize_pipeline_parallel(pipeline_parallel_size: int = 1):
    """Initialize pipeline parallel groups"""
    
    if pipeline_parallel_size == 1:
        return
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size % pipeline_parallel_size != 0:
        raise ValueError(f"World size {world_size} is not divisible by pipeline parallel size {pipeline_parallel_size}")
    
    num_pipeline_parallel_groups = world_size // pipeline_parallel_size
    
    # Create pipeline parallel groups
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    global _PIPELINE_MODEL_PARALLEL_RANK
    
    for i in range(num_pipeline_parallel_groups):
        ranks = list(range(i * pipeline_parallel_size, (i + 1) * pipeline_parallel_size))
        group = dist.new_group(ranks)
        
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pipeline_parallel_size
            _PIPELINE_MODEL_PARALLEL_RANK = ranks.index(rank)

def get_pipeline_model_parallel_group():
    """Get pipeline model parallel group"""
    return _PIPELINE_MODEL_PARALLEL_GROUP

def get_pipeline_model_parallel_world_size():
    """Get pipeline model parallel world size"""
    return _PIPELINE_MODEL_PARALLEL_WORLD_SIZE

def get_pipeline_model_parallel_rank():
    """Get pipeline model parallel rank"""
    return _PIPELINE_MODEL_PARALLEL_RANK

def is_pipeline_first_stage():
    """Check if current rank is first pipeline stage"""
    return get_pipeline_model_parallel_rank() == 0

def is_pipeline_last_stage():
    """Check if current rank is last pipeline stage"""
    return get_pipeline_model_parallel_rank() == get_pipeline_model_parallel_world_size() - 1

class PipelineStage(nn.Module):
    """A single stage in pipeline parallelism"""
    
    def __init__(self, layers: nn.ModuleList, stage_id: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.is_first_stage = stage_id == 0
        self.is_last_stage = False  # Will be set by PipelineParallelModel
    
    def forward(self, x):
        """Forward pass through stage layers"""
        for layer in self.layers:
            x = layer(x)
        return x

class PipelineParallelModel(nn.Module):
    """Pipeline parallel model wrapper"""
    
    def __init__(self, 
                 stages: List[PipelineStage],
                 config: PipelineConfig):
        super().__init__()
        
        self.stages = nn.ModuleList(stages)
        self.config = config
        self.num_stages = len(stages)
        
        # Mark last stage
        if self.stages:
            self.stages[-1].is_last_stage = True
        
        # Current stage (will be set during distributed setup)
        self.current_stage_id = get_pipeline_model_parallel_rank()
        
        # Communication buffers
        self.input_buffers = []
        self.output_buffers = []
        
        # Initialize pipeline communication
        self._setup_communication()
    
    def _setup_communication(self):
        """Setup communication for pipeline parallelism"""
        
        world_size = get_pipeline_model_parallel_world_size()
        rank = get_pipeline_model_parallel_rank()
        
        if world_size == 1:
            return
        
        # Setup send/recv groups
        self.prev_rank = rank - 1 if rank > 0 else None
        self.next_rank = rank + 1 if rank < world_size - 1 else None
        
        # Create communication tags
        self.send_tag = rank
        self.recv_tag = rank - 1 if rank > 0 else None
    
    def forward(self, x):
        """Forward pass with pipeline parallelism"""
        
        if self.config.pipeline_parallel_size == 1:
            # No pipeline parallelism, run all stages
            for stage in self.stages:
                x = stage(x)
            return x
        
        # Get current stage
        current_stage = self.stages[self.current_stage_id]
        
        # Pipeline forward pass
        return self._pipeline_forward(x, current_stage)
    
    def _pipeline_forward(self, x, stage):
        """Execute pipeline forward pass"""
        
        rank = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        group = get_pipeline_model_parallel_group()
        
        if rank == 0:
            # First stage - process input
            output = stage(x)
            
            if world_size > 1:
                # Send to next stage
                dist.send(output, dst=self.next_rank, group=group, tag=self.send_tag)
                return None  # First stage doesn't return to caller
            else:
                return output
                
        elif rank == world_size - 1:
            # Last stage - receive from previous, process, and return
            if self.prev_rank is not None:
                # Receive from previous stage
                input_tensor = torch.empty_like(x)
                dist.recv(input_tensor, src=self.prev_rank, group=group, tag=self.recv_tag)
                x = input_tensor
            
            # Process through current stage
            output = stage(x)
            return output
            
        else:
            # Middle stage - receive, process, send
            if self.prev_rank is not None:
                # Receive from previous stage
                input_tensor = torch.empty_like(x)
                dist.recv(input_tensor, src=self.prev_rank, group=group, tag=self.recv_tag)
                x = input_tensor
            
            # Process through current stage
            output = stage(x)
            
            if self.next_rank is not None:
                # Send to next stage
                dist.send(output, dst=self.next_rank, group=group, tag=self.send_tag)
            
            return None  # Middle stages don't return to caller

class MicroBatchScheduler:
    """Scheduler for micro-batch pipeline execution"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.micro_batch_size = config.micro_batch_size
        self.chunks = config.chunks
        
    def split_batch(self, batch):
        """Split batch into micro-batches"""
        batch_size = batch.size(0)
        
        if batch_size % self.micro_batch_size != 0:
            raise ValueError(f"Batch size {batch_size} is not divisible by micro batch size {self.micro_batch_size}")
        
        num_micro_batches = batch_size // self.micro_batch_size
        micro_batches = []
        
        for i in range(num_micro_batches):
            start_idx = i * self.micro_batch_size
            end_idx = start_idx + self.micro_batch_size
            micro_batch = batch[start_idx:end_idx]
            micro_batches.append(micro_batch)
        
        return micro_batches
    
    def schedule_forward(self, micro_batches, model):
        """Schedule forward passes for micro-batches"""
        outputs = []
        
        for micro_batch in micro_batches:
            output = model(micro_batch)
            if output is not None:  # Only last stage returns output
                outputs.append(output)
        
        return outputs

class PipelineParallelTrainer:
    """Trainer with pipeline parallelism support"""
    
    def __init__(self, 
                 model: PipelineParallelModel,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 config: PipelineConfig):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.scheduler = MicroBatchScheduler(config)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 1
        self.accumulated_loss = 0.0
        
    def train_step(self, batch, targets=None):
        """Execute one training step with pipeline parallelism"""
        
        # Split batch into micro-batches
        micro_batches = self.scheduler.split_batch(batch)
        
        if targets is not None:
            micro_targets = self.scheduler.split_batch(targets)
        else:
            micro_targets = [None] * len(micro_batches)
        
        total_loss = 0.0
        num_outputs = 0
        
        # Process micro-batches
        for micro_batch, micro_target in zip(micro_batches, micro_targets):
            
            # Forward pass
            output = self.model(micro_batch)
            
            # Only last stage computes loss
            if output is not None and is_pipeline_last_stage():
                if micro_target is not None:
                    loss = self.loss_fn(output, micro_target)
                    total_loss += loss.item()
                    num_outputs += 1
                    
                    # Backward pass
                    loss.backward()
        
        # Update parameters
        if num_outputs > 0:
            # Average gradients across micro-batches
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.div_(num_outputs)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return total_loss / num_outputs
        
        return 0.0

class AsyncPipelineParallelModel(nn.Module):
    """Asynchronous pipeline parallel model for better throughput"""
    
    def __init__(self, 
                 stages: List[PipelineStage],
                 config: PipelineConfig):
        super().__init__()
        
        self.stages = nn.ModuleList(stages)
        self.config = config
        self.current_stage_id = get_pipeline_model_parallel_rank()
        
        # Async communication queues
        self.input_queue = queue.Queue(maxsize=config.chunks)
        self.output_queue = queue.Queue(maxsize=config.chunks)
        
        # Worker threads
        self.worker_thread = None
        self.running = False
        
        self._setup_async_communication()
    
    def _setup_async_communication(self):
        """Setup asynchronous communication"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._async_worker)
        self.worker_thread.start()
    
    def _async_worker(self):
        """Asynchronous worker for pipeline processing"""
        
        rank = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        group = get_pipeline_model_parallel_group()
        
        current_stage = self.stages[self.current_stage_id]
        
        while self.running:
            try:
                # Get input from queue
                if rank == 0:
                    # First stage gets input from input queue
                    input_data = self.input_queue.get(timeout=1.0)
                else:
                    # Other stages receive from previous stage
                    input_data = self._async_receive(group)
                
                if input_data is None:
                    continue
                
                # Process through current stage
                output = current_stage(input_data)
                
                if rank == world_size - 1:
                    # Last stage puts output to output queue
                    self.output_queue.put(output)
                else:
                    # Other stages send to next stage
                    self._async_send(output, group)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in async worker: {e}")
                break
    
    def _async_send(self, tensor, group):
        """Asynchronous send operation"""
        next_rank = get_pipeline_model_parallel_rank() + 1
        if next_rank < get_pipeline_model_parallel_world_size():
            dist.isend(tensor, dst=next_rank, group=group)
    
    def _async_receive(self, group):
        """Asynchronous receive operation"""
        prev_rank = get_pipeline_model_parallel_rank() - 1
        if prev_rank >= 0:
            # This is a simplified version - in practice you'd need proper tensor shapes
            tensor = torch.empty((self.config.micro_batch_size, 768))  # Example shape
            dist.irecv(tensor, src=prev_rank, group=group)
            return tensor
        return None
    
    def forward(self, x):
        """Forward pass with async pipeline"""
        
        if get_pipeline_model_parallel_rank() == 0:
            # First stage - put input to queue
            self.input_queue.put(x)
            return None
        
        elif is_pipeline_last_stage():
            # Last stage - get output from queue
            return self.output_queue.get()
        
        else:
            # Middle stages don't interact with forward call
            return None
    
    def shutdown(self):
        """Shutdown async pipeline"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()

def create_pipeline_stages(model: nn.Module, 
                          num_stages: int,
                          balance: Optional[List[int]] = None) -> List[PipelineStage]:
    """Create pipeline stages from a model"""
    
    if balance is None:
        # Auto-balance layers across stages
        layers = list(model.children())
        layers_per_stage = len(layers) // num_stages
        balance = [layers_per_stage] * num_stages
        
        # Distribute remaining layers
        remaining = len(layers) - sum(balance)
        for i in range(remaining):
            balance[i] += 1
    
    if sum(balance) != len(list(model.children())):
        raise ValueError(f"Balance {balance} doesn't match number of layers {len(list(model.children()))}")
    
    stages = []
    layers = list(model.children())
    layer_idx = 0
    
    for stage_id, num_layers in enumerate(balance):
        stage_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            if layer_idx < len(layers):
                stage_layers.append(layers[layer_idx])
                layer_idx += 1
        
        stage = PipelineStage(stage_layers, stage_id)
        stages.append(stage)
    
    return stages

def setup_pipeline_parallel_model(model: nn.Module,
                                config: PipelineConfig) -> PipelineParallelModel:
    """Setup pipeline parallel model from regular model"""
    
    # Initialize pipeline parallelism
    initialize_pipeline_parallel(config.pipeline_parallel_size)
    
    # Create pipeline stages
    stages = create_pipeline_stages(
        model, 
        config.pipeline_parallel_size,
        config.balance
    )
    
    # Create pipeline parallel model
    pipeline_model = PipelineParallelModel(stages, config)
    
    return pipeline_model

# Utility functions for pipeline parallel training
def get_pipeline_parallel_loss_scale():
    """Get loss scale for pipeline parallelism"""
    return 1.0 / get_pipeline_model_parallel_world_size()

def all_reduce_pipeline_parallel_gradients(model):
    """All-reduce gradients across pipeline parallel groups"""
    
    group = get_pipeline_model_parallel_group()
    if group is None:
        return
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, group=group)
            param.grad.data.div_(get_pipeline_model_parallel_world_size())

def checkpoint_pipeline_stage(stage_output, checkpoint_type="never"):
    """Apply checkpointing to pipeline stage output"""
    
    if checkpoint_type == "always":
        return torch.utils.checkpoint.checkpoint(lambda x: x, stage_output)
    elif checkpoint_type == "except_last" and not is_pipeline_last_stage():
        return torch.utils.checkpoint.checkpoint(lambda x: x, stage_output)
    else:
        return stage_output

# Example usage functions
def example_transformer_pipeline():
    """Example of setting up pipeline parallelism for transformer model"""
    
    # Example transformer model (simplified)
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=12):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.output(x)
    
    # Create model
    model = SimpleTransformer()
    
    # Pipeline config
    config = PipelineConfig(
        pipeline_parallel_size=4,
        micro_batch_size=2,
        chunks=4,
        checkpoint="except_last"
    )
    
    # Setup pipeline model
    pipeline_model = setup_pipeline_parallel_model(model, config)
    
    return pipeline_model

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize distributed training
    if torch.distributed.is_available():
        dist.init_process_group(backend='nccl')
        
        # Create pipeline model
        pipeline_model = example_transformer_pipeline()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(pipeline_model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        
        # Create trainer
        config = PipelineConfig(pipeline_parallel_size=4, micro_batch_size=2)
        trainer = PipelineParallelTrainer(pipeline_model, optimizer, loss_fn, config)
        
        # Example training step
        batch = torch.randint(0, 10000, (8, 512))  # Example batch
        targets = torch.randint(0, 10000, (8, 512)) # Example targets
        
        loss = trainer.train_step(batch, targets)
        print(f"Training loss: {loss}")
        
        # Cleanup
        if hasattr(pipeline_model, 'shutdown'):
            pipeline_model.shutdown()