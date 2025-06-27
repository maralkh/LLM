# training_infra/distributed/trainer.py
"""
Advanced distributed trainer for LLaMA models with full parallelism support.
"""

import os
import time
import warnings
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from ..trainer import Trainer
from ..config import TrainingConfig
from .config import DistributedConfig
from .microbatch_scheduler import MicrobatchScheduler, BatchSplitter
from ..models.llama import LlamaForCausalLM
from ..parallelism import (
    init_distributed, cleanup_distributed,
    get_tensor_parallel_rank, get_pipeline_parallel_rank,
    get_data_parallel_rank, all_reduce_grads
)


class DistributedTrainer(Trainer):
    """Base distributed trainer with common functionality"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        distributed_config: DistributedConfig,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        **kwargs
    ):
        self.distributed_config = distributed_config
        
        # Initialize distributed environment
        self._init_distributed()
        
        # Setup model with parallelism
        model = self._setup_distributed_model(model)
        
        # Initialize base trainer
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            **kwargs
        )
        
        # Setup distributed components
        self._setup_distributed_dataloaders()
        self._setup_distributed_optimizer()
        
        # Performance tracking
        self.performance_tracker = DistributedPerformanceTracker()
        
    def _init_distributed(self):
        """Initialize distributed training environment"""
        
        if not dist.is_initialized():
            # Get environment variables
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            rank = int(os.environ.get('RANK', 0))
            
            # Initialize distributed
            init_distributed(
                backend=self.distributed_config.communication.backend,
                rank=rank,
                world_size=world_size,
                timeout_minutes=self.distributed_config.communication.timeout_minutes
            )
        
        # Set device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')
        
        # Store distributed information
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Calculate parallel group information
        self.tensor_parallel_rank = get_tensor_parallel_rank()
        self.pipeline_parallel_rank = get_pipeline_parallel_rank()
        self.data_parallel_rank = get_data_parallel_rank()
        
        self.is_main_process = self.global_rank == 0
        
        # Set random seeds for reproducibility
        if self.distributed_config.deterministic:
            self._set_deterministic_seeds()
    
    def _set_deterministic_seeds(self):
        """Set deterministic seeds for reproducible training"""
        seed = self.distributed_config.seed + self.global_rank
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        
        if self.distributed_config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_distributed_model(self, model: nn.Module) -> nn.Module:
        """Setup model with appropriate parallelism"""
        
        # Move model to device
        model = model.to(self.device)
        
        # Apply tensor parallelism (handled by model architecture)
        if self.distributed_config.uses_tensor_parallelism:
            if hasattr(model, 'config') and hasattr(model.config, 'tensor_parallel_config'):
                model.config.tensor_parallel_config.tensor_parallel_size = (
                    self.distributed_config.tensor_parallel.tensor_parallel_size
                )
        
        # Apply pipeline parallelism
        if self.distributed_config.uses_pipeline_parallelism:
            model = self._setup_pipeline_parallelism(model)
        
        # Apply data parallelism (DDP)
        if self.distributed_config.uses_data_parallelism:
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.distributed_config.data_parallel.find_unused_parameters,
                bucket_cap_mb=self.distributed_config.data_parallel.bucket_cap_mb,
                gradient_as_bucket_view=self.distributed_config.data_parallel.gradient_as_bucket_view
            )
        
        # Apply ZeRO optimizer if requested
        if self.distributed_config.zero.stage > 0:
            model = self._setup_zero_optimizer(model)
        
        return model
    
    def _setup_pipeline_parallelism(self, model: nn.Module) -> nn.Module:
        """Setup pipeline parallelism"""
        try:
            from ..parallelism import PipelineParallelLlama
            
            pipeline_config = self.distributed_config.pipeline_parallel
            
            model = PipelineParallelLlama(model, pipeline_config)
            
            if self.is_main_process:
                print(f"✅ Pipeline parallelism enabled with {pipeline_config.pipeline_parallel_size} stages")
            
            return model
            
        except ImportError:
            warnings.warn("Pipeline parallelism not available, falling back to data parallelism")
            return model
    
    def _setup_zero_optimizer(self, model: nn.Module) -> nn.Module:
        """Setup ZeRO optimizer"""
        try:
            from deepspeed import zero
            
            zero_config = {
                "stage": self.distributed_config.zero.stage,
                "reduce_bucket_size": self.distributed_config.zero.reduce_bucket_size_mb * 1024 * 1024,
                "allgather_bucket_size": self.distributed_config.zero.allgather_bucket_size_mb * 1024 * 1024,
                "overlap_comm": self.distributed_config.zero.overlap_communication,
                "cpu_offload": self.distributed_config.zero.cpu_offload,
            }
            
            model = zero.Init(config_dict_or_path=zero_config)(model)
            
            if self.is_main_process:
                print(f"✅ ZeRO Stage {self.distributed_config.zero.stage} enabled")
            
            return model
            
        except ImportError:
            warnings.warn("DeepSpeed ZeRO not available")
            return model
    
    def _setup_distributed_dataloaders(self):
        """Setup dataloaders with distributed samplers"""
        
        if self.distributed_config.uses_data_parallelism:
            # Setup train dataloader
            if self.train_dataloader is not None:
                self.train_dataloader = self._create_distributed_dataloader(
                    self.train_dataloader, shuffle=True
                )
            
            # Setup validation dataloader
            if self.val_dataloader is not None:
                self.val_dataloader = self._create_distributed_dataloader(
                    self.val_dataloader, shuffle=False
                )
    
    def _create_distributed_dataloader(self, dataloader: DataLoader, shuffle: bool) -> DataLoader:
        """Create distributed version of dataloader"""
        
        if isinstance(dataloader.sampler, DistributedSampler):
            return dataloader  # Already distributed
        
        dataset = dataloader.dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.distributed_config.data_parallel.data_parallel_size,
            rank=self.data_parallel_rank,
            shuffle=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=getattr(dataloader, 'pin_memory', False),
            drop_last=True  # Important for distributed training
        )
    
    def _setup_distributed_optimizer(self):
        """Setup optimizer with distributed considerations"""
        
        # Scale learning rate by data parallel size
        if self.distributed_config.uses_data_parallelism:
            original_lr = self.config.optimizer.lr
            scaled_lr = original_lr * self.distributed_config.data_parallel.data_parallel_size
            self.config.optimizer.lr = scaled_lr
            
            if self.is_main_process:
                print(f"📈 Scaled learning rate from {original_lr} to {scaled_lr}")
        
        # Create optimizer (base class handles this)
        super()._setup_optimizer()
        
        # Setup gradient compression if requested
        if self.distributed_config.communication.use_gradient_compression:
            self._setup_gradient_compression()
    
    def _setup_gradient_compression(self):
        """Setup gradient compression for communication efficiency"""
        try:
            from ..parallelism import GradientCompression
            
            self.gradient_compressor = GradientCompression(
                compression_ratio=self.distributed_config.communication.compression_ratio
            )
            
            # Register compression hooks
            for param in self.model.parameters():
                if param.requires_grad:
                    self.gradient_compressor.register_hooks(param)
            
            if self.is_main_process:
                print(f"✅ Gradient compression enabled ({self.distributed_config.communication.compression_ratio:.1%})")
                    
        except ImportError:
            warnings.warn("Gradient compression not available")
    
    def train_step(self, batch):
        """Base distributed training step"""
        
        # Track performance
        self.performance_tracker.start_step()
        
        # Check if using pipeline parallelism
        if self.distributed_config.uses_pipeline_parallelism:
            loss = self._pipeline_train_step(batch)
        else:
            loss = self._standard_train_step(batch)
        
        # Track performance
        self.performance_tracker.end_step()
        
        # Logging
        if (self.global_step % self.config.logging.log_every == 0 and self.is_main_process):
            self._log_distributed_metrics(loss)
        
        return loss
    
    def _standard_train_step(self, batch):
        """Standard training step without pipeline parallelism"""
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with mixed precision
        if self.distributed_config.mixed_precision.enabled and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            self._optimizer_step()
        
        return loss * self.config.gradient_accumulation_steps
    
    def _optimizer_step(self):
        """Execute optimizer step with distributed synchronization"""
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.distributed_config.mixed_precision.enabled and self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
        
        # All-reduce gradients if using tensor parallelism
        if self.distributed_config.uses_tensor_parallelism:
            all_reduce_grads(self.model.parameters())
        
        # Optimizer step
        if self.distributed_config.mixed_precision.enabled and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        self.global_step += 1
    
    def _log_distributed_metrics(self, loss):
        """Log distributed training metrics"""
        
        metrics = {
            'train/loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'train/lr': self.get_current_lr(),
            'train/step': self.global_step,
        }
        
        # Add distributed metrics
        metrics.update(self._get_distributed_metrics())
        
        # Add performance metrics
        perf_metrics = self.performance_tracker.get_metrics()
        for key, value in perf_metrics.items():
            metrics[f'performance/{key}'] = value
        
        self.logger.log_metrics(metrics, self.global_step)
    
    def _get_distributed_metrics(self) -> Dict[str, Any]:
        """Get distributed training metrics"""
        
        metrics = {}
        
        # Memory metrics
        if torch.cuda.is_available():
            metrics.update({
                'memory/allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'memory/reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'memory/max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            })
        
        # Parallel group info
        metrics.update({
            'parallel/tensor_parallel_rank': self.tensor_parallel_rank,
            'parallel/pipeline_parallel_rank': self.pipeline_parallel_rank,
            'parallel/data_parallel_rank': self.data_parallel_rank,
            'parallel/world_size': self.world_size,
        })
        
        return metrics
    
    def cleanup(self):
        """Cleanup distributed resources"""
        
        if hasattr(self, 'performance_tracker'):
            self.performance_tracker.cleanup()
        
        cleanup_distributed()


class LlamaDistributedTrainer(DistributedTrainer):
    """Specialized distributed trainer for LLaMA models"""
    
    def __init__(
        self,
        model: LlamaForCausalLM,
        config: TrainingConfig,
        distributed_config: DistributedConfig,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        use_moe: bool = False,
        moe_config: Optional[Any] = None,
        **kwargs
    ):
        self.use_moe = use_moe
        self.moe_config = moe_config
        self.total_aux_loss = 0.0
        
        super().__init__(
            model=model,
            config=config,
            distributed_config=distributed_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            **kwargs
        )
    
    def compute_loss(self, batch):
        """Compute loss with MoE auxiliary losses"""
        
        # Extract inputs and labels
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, labels = batch
        else:
            input_ids = batch['input_ids']
            labels = batch.get('labels', input_ids)
        
        # Move to device
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass with mixed precision
        with torch.autocast(
            device_type='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=getattr(torch, self.distributed_config.mixed_precision.dtype),
            enabled=self.distributed_config.mixed_precision.enabled
        ):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        # Add MoE auxiliary losses
        if self.use_moe:
            aux_losses = self._collect_aux_losses()
            total_aux_loss = sum(aux_losses.values()) if aux_losses else 0
            if isinstance(total_aux_loss, torch.Tensor):
                loss = loss + 0.01 * total_aux_loss  # Small weight for aux loss
                self.total_aux_loss = total_aux_loss.item()
        
        return loss
    
    def _collect_aux_losses(self) -> Dict[str, torch.Tensor]:
        """Collect auxiliary losses from MoE layers"""
        
        aux_losses = {}
        
        # Get model (unwrap DDP if necessary)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(model, 'get_aux_losses'):
            aux_losses = model.get_aux_losses()
        else:
            # Manual collection from MoE layers
            for name, module in model.named_modules():
                if hasattr(module, 'aux_loss') and module.aux_loss is not None:
                    aux_losses[f"{name}_aux_loss"] = module.aux_loss
        
        return aux_losses


class PipelineDistributedTrainer(LlamaDistributedTrainer):
    """Distributed trainer with pipeline parallelism and microbatching"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Setup microbatch scheduler if using pipeline parallelism
        if self.distributed_config.uses_pipeline_parallelism:
            self._setup_microbatch_scheduler()
    
    def _setup_microbatch_scheduler(self):
        """Setup microbatch scheduler for pipeline parallelism"""
        
        # Calculate microbatch configuration
        batch_size = self.config.batch_size
        pipeline_config = self.distributed_config.pipeline_parallel
        
        if pipeline_config.microbatch_size:
            microbatch_size = pipeline_config.microbatch_size
            num_microbatches = batch_size // microbatch_size
        else:
            # Auto-configure: aim for 4-8 microbatches per pipeline stage
            target_microbatches = pipeline_config.pipeline_parallel_size * 6
            microbatch_size = max(1, batch_size // target_microbatches)
            num_microbatches = batch_size // microbatch_size
        
        # Update pipeline config
        pipeline_config.microbatch_size = microbatch_size
        pipeline_config.num_microbatches = num_microbatches
        
        # Create microbatch scheduler
        self.microbatch_scheduler = MicrobatchScheduler(
            num_microbatches=num_microbatches,
            microbatch_size=microbatch_size,
            num_pipeline_stages=pipeline_config.pipeline_parallel_size,
            current_stage=self.pipeline_parallel_rank,
            schedule_type=pipeline_config.schedule,
            virtual_stages=pipeline_config.virtual_pipeline_stages,
            enable_profiling=True
        )
        
        if self.is_main_process:
            print(f"🔄 Pipeline parallelism configured:")
            print(f"  Microbatch size: {microbatch_size}")
            print(f"  Number of microbatches: {num_microbatches}")
            print(f"  Schedule: {pipeline_config.schedule}")
    
    def _pipeline_train_step(self, batch):
        """Training step with pipeline parallelism and microbatching"""
        
        if not hasattr(self, 'microbatch_scheduler'):
            # Fallback to standard training if no scheduler
            return self._standard_train_step(batch)
        
        # Split batch into microbatches
        microbatches = BatchSplitter.split_batch(
            batch,
            self.microbatch_scheduler.microbatch_size,
            self.microbatch_scheduler.num_microbatches
        )
        
        # Define forward and backward functions
        def forward_fn(microbatch, microbatch_id):
            return self.compute_loss(microbatch)
        
        def backward_fn(microbatch_id):
            # Backward is handled automatically by autograd
            pass
        
        def optimizer_step_fn():
            self._optimizer_step()
        
        # Execute pipeline schedule
        results = self.microbatch_scheduler.execute_schedule(
            forward_fn=forward_fn,
            backward_fn=backward_fn,
            optimizer_step_fn=optimizer_step_fn,
            microbatch_data=microbatches
        )
        
        # Calculate average loss
        if results['losses']:
            avg_loss = sum(loss.item() for loss in results['losses']) / len(results['losses'])
        else:
            avg_loss = 0.0
        
        return avg_loss
    
    def _log_distributed_metrics(self, loss):
        """Log metrics including pipeline performance"""
        
        super()._log_distributed_metrics(loss)
        
        # Add pipeline-specific metrics
        if hasattr(self, 'microbatch_scheduler'):
            pipeline_metrics = self.microbatch_scheduler.get_metrics()
            
            additional_metrics = {}
            for key, value in pipeline_metrics.items():
                additional_metrics[f'pipeline/{key}'] = value
            
            self.logger.log_metrics(additional_metrics, self.global_step)


class AdaptiveDistributedTrainer(PipelineDistributedTrainer):
    """Adaptive trainer with dynamic optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive strategies
        self.memory_optimizer = MemoryOptimizer()
        self.batch_size_adapter = BatchSizeAdapter(self.config.batch_size)
        
        # Performance monitoring
        self.step_times = []
        self.memory_usage = []
        self.oom_count = 0
    
    def train_step(self, batch):
        """Adaptive training step with error handling"""
        
        step_start = time.time()
        
        try:
            # Try normal training step
            loss = super().train_step(batch)
            
            # Record successful step
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            self.batch_size_adapter.record_success()
            
            # Track memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                self.memory_usage.append(memory_gb)
                self.memory_optimizer.update_memory_usage(memory_gb)
            
            return loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return self._handle_oom(batch, e)
            else:
                raise e
    
    def _handle_oom(self, batch, error):
        """Handle out-of-memory error"""
        
        self.oom_count += 1
        self.logger.warning(f"OOM detected at step {self.global_step}: {error}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Adapt batch size
        new_batch_size = self.batch_size_adapter.handle_oom()
        
        # Adapt microbatch configuration if using pipeline
        if hasattr(self, 'microbatch_scheduler'):
            self._adapt_microbatch_config(new_batch_size)
        
        # Enable more aggressive memory optimizations
        self.memory_optimizer.enable_aggressive_mode()
        
        # Retry with smaller batch
        reduced_batch = self._reduce_batch_size(batch, new_batch_size)
        
        try:
            return super().train_step(reduced_batch)
        except RuntimeError as e:
            self.logger.error(f"Failed even with reduced batch size: {e}")
            raise e
    
    def _reduce_batch_size(self, batch, new_batch_size):
        """Reduce batch size for OOM recovery"""
        
        if isinstance(batch, dict):
            reduced_batch = {}
            for key, value in batch.items():
                reduced_batch[key] = value[:new_batch_size]
            return reduced_batch
        elif isinstance(batch, (list, tuple)):
            reduced_batch = []
            for tensor in batch:
                reduced_batch.append(tensor[:new_batch_size])
            return tuple(reduced_batch) if isinstance(batch, tuple) else reduced_batch
        else:
            return batch[:new_batch_size]
    
    def _adapt_microbatch_config(self, new_batch_size):
        """Adapt microbatch configuration for new batch size"""
        
        if hasattr(self, 'microbatch_scheduler'):
            old_microbatch_size = self.microbatch_scheduler.microbatch_size
            new_num_microbatches = max(1, new_batch_size // old_microbatch_size)
            
            self.microbatch_scheduler.num_microbatches = new_num_microbatches
            
            self.logger.info(f"Adapted microbatch config: {new_num_microbatches} microbatches")


class MemoryOptimizer:
    """Adaptive memory optimization"""
    
    def __init__(self, memory_threshold_gb: float = 40.0):
        self.memory_threshold_gb = memory_threshold_gb
        self.aggressive_mode = False
        self.memory_history = []
    
    def update_memory_usage(self, memory_gb: float):
        """Update memory usage tracking"""
        self.memory_history.append(memory_gb)
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)
    
    def enable_aggressive_mode(self):
        """Enable aggressive memory optimizations"""
        self.aggressive_mode = True
        
        # Enable additional optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def should_use_checkpointing(self) -> bool:
        """Decide if activation checkpointing should be used"""
        if self.aggressive_mode:
            return True
        
        if self.memory_history:
            avg_memory = sum(self.memory_history[-10:]) / min(10, len(self.memory_history))
            return avg_memory > self.memory_threshold_gb * 0.8
        
        return False


class BatchSizeAdapter:
    """Adaptive batch size management"""
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = initial_batch_size * 2
        
        self.oom_count = 0
        self.success_count = 0
    
    def handle_oom(self) -> int:
        """Handle OOM by reducing batch size"""
        self.oom_count += 1
        self.success_count = 0
        
        if self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        
        return self.current_batch_size
    
    def record_success(self):
        """Record successful training step"""
        self.success_count += 1
        
        # Gradually increase batch size if stable
        if (self.success_count >= 100 and 
            self.current_batch_size < self.max_batch_size and
            self.oom_count == 0):
            
            self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            self.success_count = 0


class DistributedPerformanceTracker:
    """Track performance metrics for distributed training"""
    
    def __init__(self):
        self.step_start_time = None
        self.total_steps = 0
        self.total_time = 0.0
        self.communication_time = 0.0
        self.compute_time = 0.0
    
    def start_step(self):
        """Start timing a training step"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_start_time = time.time()
    
    def end_step(self):
        """End timing a training step"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.total_time += step_time
            self.total_steps += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        if self.total_steps == 0:
            return {}
        
        return {
            'avg_step_time_ms': (self.total_time / self.total_steps) * 1000,
            'total_steps': self.total_steps,
            'steps_per_second': self.total_steps / max(self.total_time, 1e-8),
        }
    
    def cleanup(self):
        """Cleanup resources"""
        pass


# Factory function for creating trainers
def create_distributed_trainer(
    model: LlamaForCausalLM,
    config: TrainingConfig,
    distributed_config: DistributedConfig,
    trainer_type: str = "standard",
    **kwargs
) -> DistributedTrainer:
    """Factory function to create appropriate distributed trainer"""
    
    trainer_classes = {
        "standard": LlamaDistributedTrainer,
        "pipeline": PipelineDistributedTrainer,
        "adaptive": AdaptiveDistributedTrainer,
    }
    
    if trainer_type not in trainer_classes:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(trainer_classes.keys())}")
    
    trainer_class = trainer_classes[trainer_type]
    
    return trainer_class(
        model=model,
        config=config,
        distributed_config=distributed_config,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    from .config import ConfigurationFactory
    from ..models.llama import create_llama_7b_parallel
    
    # Create model and configurations
    model = create_llama_7b_parallel(tensor_parallel_size=2)
    
    config = TrainingConfig(
        model_name="llama_7b_distributed",
        batch_size=8,
        epochs=1
    )
    
    distributed_config = ConfigurationFactory.create_hybrid_parallel_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=1
    )
    
    # Create trainer
    trainer = create_distributed_trainer(
        model=model,
        config=config,
        distributed_config=distributed_config,
        trainer_type="adaptive"
    )
    
    print("✅ Distributed trainer created successfully!")
    distributed_config.print_configuration()