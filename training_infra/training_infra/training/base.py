# training_infra/training/base.py
"""
Clean base trainer implementation.

Focuses on the core training loop with setup utilities separated.
"""

import time
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import setup utilities
try:
    from ..utils.setup import (
        setup_device, extract_parallelism_config, setup_parallelism,
        setup_distributed, setup_optimizer, setup_scheduler, 
        setup_mixed_precision, setup_data_loaders, print_training_setup
    )
except ImportError:
    # Direct execution - use absolute imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from training_infra.utils.setup import setup_device, extract_parallelism_config, setup_parallelism, \
        setup_distributed, setup_optimizer, setup_scheduler, \
        setup_mixed_precision, setup_data_loaders, print_training_setup
    
@dataclass
class TrainingState:
    """Tracks the current state of training."""
    
    global_step: int = 0
    total_steps: int = 0
    
    # Loss tracking
    train_loss: float = 0.0
    eval_loss: float = 0.0
    best_eval_loss: float = float('inf')
    
    # Performance tracking
    examples_seen: int = 0
    tokens_seen: int = 0
    
    # Timing
    training_time: float = 0.0
    last_log_time: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Memory tracking
    peak_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'global_step': self.global_step,
            'total_steps': self.total_steps,
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'best_eval_loss': self.best_eval_loss,
            'examples_seen': self.examples_seen,
            'tokens_seen': self.tokens_seen,
            'training_time': self.training_time,
            'learning_rate': self.learning_rate,
            'peak_memory_mb': self.peak_memory_mb,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        return cls(**data)


class BaseTrainer:
    """
    Clean base trainer class for training language models.
    
    Features:
    - Step-based training loop
    - Gradient accumulation
    - Mixed precision training
    - Checkpointing and resuming
    - Distributed and parallel training ready
    - Memory optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        training_config,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        **kwargs
    ):
        self.model = model
        self.config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Validate config
        if not self.config.max_steps:
            raise ValueError("max_steps must be specified in TrainingConfig")
        
        # Initialize training state
        self.state = TrainingState()
        self.state.total_steps = self.config.max_steps
        
        # Setup components
        self._setup_all_components()
        
        # Print setup summary
        print_training_setup(
            self.model, self.device, self.config, self.state.total_steps,
            self.parallelism_config, self.is_distributed, self.world_size, self.rank
        )
    
    def _setup_all_components(self):
        """Setup all training components."""
        # Device setup
        self.device = setup_device(self.config)
        self.model.to(self.device)
        
        # Parallelism setup
        self.parallelism_config = extract_parallelism_config()
        self.model = setup_parallelism(self.model, self.parallelism_config)
        
        # Distributed setup
        self.is_distributed, self.world_size, self.rank = setup_distributed(self.parallelism_config)
        
        # Optimization setup
        self.optimizer = setup_optimizer(self.model, self.config)
        self.scheduler = setup_scheduler(self.optimizer, self.config, self.state.total_steps)
        self.scaler = setup_mixed_precision(self.config, self.device)
        
        # Data loaders
        self.train_dataloader, self.eval_dataloader = setup_data_loaders(
            self.train_dataset, self.eval_dataset, self.config
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self):
        """Main step-based training loop."""
        print(f"\n🚀 Starting Training")
        print("=" * 60)
        print(f"Total Steps: {self.state.total_steps:,}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Gradient Accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.config.get_effective_batch_size()}")
        print("=" * 60)
        
        if self.train_dataloader is None:
            raise ValueError("No training data available")
        
        self.model.train()
        start_time = time.time()
        
        try:
            # Step-based training loop
            while self.state.global_step < self.state.total_steps:
                step_loss = self._train_step_loop()
                
                # Update global step and state
                self.state.global_step += 1
                self.state.train_loss = step_loss
                
                # Logging
                if self.state.global_step % self.config.logging.log_every_n_steps == 0:
                    self._log_step(step_loss)
                
                # Evaluation
                if (self.config.do_eval and 
                    self.eval_dataloader is not None and
                    self.state.global_step % self.config.logging.eval_every_n_steps == 0):
                    eval_loss = self._evaluate()
                    self.state.eval_loss = eval_loss
                    
                    # Check if this is the best model
                    if eval_loss < self.state.best_eval_loss:
                        self.state.best_eval_loss = eval_loss
                        self._save_checkpoint(is_best=True)
                        print(f"🎉 New best model! Eval loss: {eval_loss:.4f}")
                    
                    self.model.train()  # Back to training mode
                
                # Save checkpoint
                if self.state.global_step % self.config.logging.save_every_n_steps == 0:
                    self._save_checkpoint()
            
            self._finish_training(start_time)
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
            self._save_checkpoint(prefix="interrupted")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            self._save_checkpoint(prefix="error")
            raise
    
    def _train_step_loop(self) -> float:
        """Perform gradient accumulation steps and return average loss."""
        accumulated_loss = 0.0
        
        for accum_step in range(self.config.gradient_accumulation_steps):
            # Get next batch
            batch = self._get_next_batch()
            
            # Perform training step
            is_last_accum_step = (accum_step == self.config.gradient_accumulation_steps - 1)
            loss = self._train_step(batch, should_update_optimizer=is_last_accum_step)
            accumulated_loss += loss
        
        # Return average loss over accumulation steps
        return accumulated_loss / self.config.gradient_accumulation_steps
    
    def _get_next_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch from data loader."""
        try:
            if not hasattr(self, '_data_iterator'):
                self._data_iterator = iter(self.train_dataloader)
            
            batch = next(self._data_iterator)
        except StopIteration:
            # Reset iterator when dataset is exhausted
            self._data_iterator = iter(self.train_dataloader)
            batch = next(self._data_iterator)
        
        return batch
    
    def _train_step(self, batch: Dict[str, torch.Tensor], should_update_optimizer: bool = True) -> float:
        """Perform one training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward and backward pass
        if self.config.use_mixed_precision and self.scaler is not None:
            loss = self._mixed_precision_step(batch, should_update_optimizer)
        else:
            loss = self._regular_precision_step(batch, should_update_optimizer)
        
        # Update counters
        self._update_counters(batch)
        
        return loss.item() * self.config.gradient_accumulation_steps  # Return unscaled loss
    
    def _mixed_precision_step(self, batch: Dict[str, torch.Tensor], should_update_optimizer: bool) -> torch.Tensor:
        """Mixed precision training step."""
        with autocast():
            loss = self._compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step only on last accumulation step
        if should_update_optimizer:
            self._update_optimizer_mixed_precision()
        
        return loss
    
    def _regular_precision_step(self, batch: Dict[str, torch.Tensor], should_update_optimizer: bool) -> torch.Tensor:
        """Regular precision training step."""
        loss = self._compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Optimizer step only on last accumulation step
        if should_update_optimizer:
            self._update_optimizer_regular()
        
        return loss
    
    def _update_optimizer_mixed_precision(self):
        """Update optimizer with mixed precision."""
        # Gradient clipping
        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        # Scheduler step
        self._update_scheduler()
    
    def _update_optimizer_regular(self):
        """Update optimizer with regular precision."""
        # Gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Scheduler step
        self._update_scheduler()
    
    def _update_scheduler(self):
        """Update learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()
            self.state.learning_rate = self.optimizer.param_groups[0]['lr']
    
    def _update_counters(self, batch: Dict[str, torch.Tensor]):
        """Update training counters."""
        batch_size = batch['input_ids'].size(0)
        seq_length = batch['input_ids'].size(1)
        self.state.examples_seen += batch_size
        self.state.tokens_seen += batch_size * seq_length
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        # Language modeling loss
        input_ids = batch['input_ids']
        target_ids = batch.get('target_ids', input_ids)
        
        # Forward pass
        outputs = self.model(input_ids)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Reshape for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def _evaluate(self) -> float:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                if self.config.use_mixed_precision and self.scaler is not None:
                    with autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"📊 Eval loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _log_step(self, loss: float):
        """Log training progress."""
        current_time = time.time()
        
        if self.state.last_log_time > 0:
            time_per_step = (current_time - self.state.last_log_time) / self.config.logging.log_every_n_steps
            steps_per_sec = 1.0 / time_per_step
            examples_per_sec = self.config.get_effective_batch_size() * steps_per_sec
        else:
            time_per_step = 0.0
            steps_per_sec = 0.0
            examples_per_sec = 0.0
        
        self.state.last_log_time = current_time
        
        # Memory usage
        if self.device.type == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            self.state.peak_memory_mb = max(self.state.peak_memory_mb, memory_used)
        else:
            memory_used = 0
            memory_reserved = 0
        
        print(f"Step {self.state.global_step:6d} | "
              f"Loss: {loss:.4f} | "
              f"LR: {self.state.learning_rate:.2e} | "
              f"Steps/s: {steps_per_sec:.2f} | "
              f"Ex/s: {examples_per_sec:.1f} | "
              f"Mem: {memory_used:.0f}MB")
    
    def _finish_training(self, start_time: float):
        """Finish training and print summary."""
        self.state.training_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"Total steps: {self.state.global_step:,}/{self.state.total_steps:,}")
        print(f"Total time: {self.state.training_time:.2f}s")
        print(f"Final train loss: {self.state.train_loss:.4f}")
        if self.state.eval_loss > 0:
            print(f"Final eval loss: {self.state.eval_loss:.4f}")
            print(f"Best eval loss: {self.state.best_eval_loss:.4f}")
    
    def _save_checkpoint(self, prefix: str = "checkpoint", is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.state.to_dict(),
            'config': self.config.to_dict(),
            'step': self.state.global_step,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"{prefix}_step_{self.state.global_step}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Also save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
    
    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Resume training from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        if 'training_state' in checkpoint:
            self.state = TrainingState.from_dict(checkpoint['training_state'])
        else:
            # Fallback for old checkpoint format
            self.state.global_step = checkpoint.get('step', 0)
        
        print(f"✅ Resumed from step {self.state.global_step}")
    
    def get_model(self) -> nn.Module:
        """Get the training model."""
        return self.model
    
    def get_training_state(self) -> TrainingState:
        """Get current training state."""
        return self.state


def test_base_trainer():
    """Test the clean base trainer implementation."""
    print("🧪 Testing Clean Base Trainer")
    
    try:
        # Mock model for testing
        class SimpleModel(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                return self.linear(x)
        
        # Create test model
        model = SimpleModel()
        
        # Mock training config
        class MockConfig:
            max_steps = 20
            batch_size = 4
            eval_batch_size = 4
            gradient_accumulation_steps = 2
            use_mixed_precision = False
            use_cpu = False
            device = "auto"
            gradient_clipping = 1.0
            dataloader_num_workers = 0
            pin_memory = False
            do_eval = False
            
            class optimizer:
                name = "adamw"
                lr = 1e-3
                @staticmethod
                def get_optimizer_kwargs():
                    return {"lr": 1e-3, "weight_decay": 0.01}
            
            class scheduler:
                name = "cosine"
                min_lr_ratio = 0.1
                @staticmethod
                def get_scheduler_kwargs():
                    return {}
            
            class logging:
                log_every_n_steps = 5
                eval_every_n_steps = 10
                save_every_n_steps = 10
                checkpoint_dir = "./test_checkpoints"
            
            @staticmethod
            def get_effective_batch_size():
                return 8
            
            @staticmethod
            def get_warmup_steps(total_steps):
                return 0
            
            @staticmethod
            def to_dict():
                return {"test": "config"}
        
        config = MockConfig()
        
        print("✅ Created test model and config")
        
        # Create trainer
        trainer = BaseTrainer(model=model, training_config=config)
        print("✅ Clean trainer initialized")
        
        # Test training state
        state = trainer.get_training_state()
        print(f"✅ Training state: step {state.global_step}, total {state.total_steps}")
        
        print("🎉 Clean base trainer test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Clean base trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_base_trainer()