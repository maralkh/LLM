# training_infra/trainer.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Callable
import os
from pathlib import Path
import time
import json

from .config import TrainingConfig
from .logger import TrainingLogger
from .callbacks import Callback, EarlyStopping, ModelCheckpoint, ProgressBar
from .utils import (
    set_seed, get_device, setup_distributed, cleanup_distributed,
    create_optimizer, create_scheduler, save_checkpoint, load_checkpoint,
    AverageMeter, Timer, validate_config, log_system_info, get_memory_usage
)

class Trainer:
    """Production-ready training infrastructure"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 callbacks: Optional[List[Callback]] = None):
        
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Validate configuration
        validate_config(config)
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device and distributed training
        self.device = get_device()
        self.is_distributed = setup_distributed()
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup distributed model
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                find_unused_parameters=config.distributed.find_unused_parameters,
                gradient_as_bucket_view=config.distributed.gradient_as_bucket_view
            )
        
        # Initialize training components
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        
        # Mixed precision
        self.scaler = None
        if config.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
            self.amp_dtype = getattr(torch, config.amp_dtype)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False
        self.best_metric = float('inf')
        
        # Setup logger (only on main process)
        self.logger = None
        if self.is_main_process:
            self.logger = TrainingLogger(config)
            log_system_info(self.logger)
        
        # Setup callbacks
        self.callbacks = callbacks or []
        if self.is_main_process:
            # Add default callbacks
            self.callbacks.extend([
                ProgressBar(),
                ModelCheckpoint(
                    filepath=os.path.join(config.checkpoint.save_dir, 'checkpoint-{epoch:03d}.pt'),
                    monitor=config.checkpoint.monitor,
                    mode=config.checkpoint.mode,
                    save_best_only=False,
                    period=1
                ),
                ModelCheckpoint(
                    filepath=os.path.join(config.checkpoint.save_dir, 'best-model.pt'),
                    monitor=config.checkpoint.monitor,
                    mode=config.checkpoint.mode,
                    save_best_only=True
                )
            ])
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self.resume_from_checkpoint(config.resume_from)
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        if self.is_main_process:
            self.logger.log_text(f"Resuming training from {checkpoint_path}")
        
        checkpoint = load_checkpoint(
            checkpoint_path, 
            self.model, 
            self.optimizer, 
            self.scheduler, 
            self.scaler,
            self.device
        )
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Metrics tracking
        train_loss = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        timer = Timer()
        timer.start()
        
        # Call callbacks
        self._call_callbacks('on_epoch_begin', self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            data_time.update(timer.elapsed())
            
            # Call callbacks
            self._call_callbacks('on_batch_begin', batch_idx)
            
            # Forward pass
            loss = self.train_step(batch)
            
            # Update metrics
            train_loss.update(loss.item(), self._get_batch_size(batch))
            
            # Log metrics periodically
            if self.global_step % self.config.logging.log_every == 0 and self.is_main_process:
                metrics = {
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.current_epoch,
                    'train/batch_time': batch_time.avg,
                    'train/data_time': data_time.avg,
                }
                
                # Add memory usage if CUDA
                if self.device.type == 'cuda':
                    memory_info = get_memory_usage()
                    metrics.update({f'memory/{k}': v for k, v in memory_info.items()})
                
                self.logger.log_metrics(metrics, self.global_step)
            
            # Call callbacks
            batch_logs = {'loss': loss.item()}
            self._call_callbacks('on_batch_end', batch_idx, batch_logs)
            
            # Update step counter
            self.global_step += 1
            
            # Check if should stop
            if self.should_stop:
                break
            
            # Check max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                self.should_stop = True
                break
            
            batch_time.update(timer.elapsed())
            timer.start()
        
        # Epoch metrics
        epoch_metrics = {
            'train/epoch_loss': train_loss.avg,
            'train/avg_batch_time': batch_time.avg,
            'train/avg_data_time': data_time.avg,
        }
        
        return epoch_metrics
    
    def train_step(self, batch) -> torch.Tensor:
        """Single training step"""
        # Move batch to device
        batch = self._move_to_device(batch)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.config.use_amp and self.scaler:
            with autocast(dtype=self.amp_dtype):
                loss = self.compute_loss(batch)
            
            # Scale loss and backward pass
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular forward pass
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        # Scheduler step (if step-based)
        if self.scheduler and hasattr(self.scheduler, 'step') and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return loss
    
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch - should be overridden"""
        # This is a placeholder - users should override this method
        # For example, for language modeling:
        # inputs, targets = batch
        # outputs = self.model(inputs)
        # return F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        raise NotImplementedError("compute_loss method must be implemented by subclass or passed as argument")
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        # Call callbacks
        self._call_callbacks('on_validation_begin')
        
        val_loss = AverageMeter()
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_to_device(batch)
                
                # Forward pass
                if self.config.use_amp and self.scaler:
                    with autocast(dtype=self.amp_dtype):
                        loss = self.compute_loss(batch)
                else:
                    loss = self.compute_loss(batch)
                
                val_loss.update(loss.item(), self._get_batch_size(batch))
                
                # Break early if specified
                if self.config.eval_steps and batch_idx >= self.config.eval_steps:
                    break
        
        val_metrics = {
            'val/loss': val_loss.avg,
        }
        
        # Call callbacks
        self._call_callbacks('on_validation_end', val_metrics)
        
        return val_metrics
    
    def fit(self):
        """Main training loop"""
        if self.is_main_process:
            self.logger.log_model_info(self.model, self.optimizer, self.scheduler)
            self.logger.log_hyperparameters(self.config.to_dict())
        
        # Call callbacks
        self._call_callbacks('on_train_begin')
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                
                # Set epoch for distributed sampler
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = {}
                if epoch % self.config.eval_every == 0 or epoch == self.config.epochs - 1:
                    val_metrics = self.validate()
                
                # Combine metrics
                epoch_logs = {**train_metrics, **val_metrics}
                
                # Log epoch metrics
                if self.is_main_process:
                    self.logger.log_metrics(epoch_logs, self.global_step)
                    self.logger.update_epoch(epoch)
                
                # Call callbacks
                self._call_callbacks('on_epoch_end', epoch, epoch_logs)
                
                # Check early stopping
                if self.should_stop:
                    if self.is_main_process:
                        self.logger.log_text(f"Training stopped early at epoch {epoch + 1}")
                    break
                
                # Save checkpoint periodically
                if self.is_main_process and (epoch + 1) % self.config.checkpoint.save_every == 0:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint.save_dir, 
                        f'checkpoint-epoch-{epoch + 1:03d}.pt'
                    )
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.global_step, epoch_logs.get('train/epoch_loss', 0),
                        checkpoint_path, self.config, self.scaler
                    )
        
        except KeyboardInterrupt:
            if self.is_main_process:
                self.logger.log_text("Training interrupted by user")
        
        finally:
            # Call callbacks
            self._call_callbacks('on_train_end')
            
            # Save final checkpoint
            if self.is_main_process:
                final_checkpoint = os.path.join(self.config.checkpoint.save_dir, 'final-checkpoint.pt')
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.current_epoch, self.global_step, 0,
                    final_checkpoint, self.config, self.scaler
                )
                
                self.logger.close()
            
            # Cleanup distributed
            if self.is_distributed:
                cleanup_distributed()
    
    def _call_callbacks(self, method_name: str, *args, **kwargs):
        """Call method on all callbacks"""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                getattr(callback, method_name)(self, *args, **kwargs)
    
    def _move_to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(item) for item in batch)
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        else:
            return batch
    
    def _get_batch_size(self, batch):
        """Get batch size from batch"""
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        elif isinstance(batch, (list, tuple)):
            return self._get_batch_size(batch[0])
        elif isinstance(batch, dict):
            return self._get_batch_size(next(iter(batch.values())))
        else:
            return 1
    
    def save_model(self, filepath: str):
        """Save model weights"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), filepath)
        
        if self.is_main_process:
            self.logger.log_text(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        state_dict = torch.load(filepath, map_location=self.device)
        
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(state_dict)
        
        if self.is_main_process:
            self.logger.log_text(f"Model loaded from {filepath}")

# Custom trainer for specific tasks
class LanguageModelTrainer(Trainer):
    """Trainer specifically for language modeling tasks"""
    
    def compute_loss(self, batch):
        """Compute language modeling loss"""
        input_ids, labels = batch
        outputs = self.model(input_ids, labels=labels)
        return outputs.loss

class ClassificationTrainer(Trainer):
    """Trainer for classification tasks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, batch):
        """Compute classification loss"""
        inputs, targets = batch
        outputs = self.model(inputs)
        return self.criterion(outputs, targets)
    
    def validate(self):
        """Validate with accuracy metrics"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        val_loss = AverageMeter()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                inputs, targets = batch
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss.update(loss.item(), targets.size(0))
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        
        return {
            'val/loss': val_loss.avg,
            'val/accuracy': accuracy
        }