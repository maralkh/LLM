# training_infra/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Callable, Union
import os
from pathlib import Path
import time
import json

try:
    from config import TrainingConfig
    from logger import TrainingLogger, SimpleLogger
    from callbacks import Callback, EarlyStopping, ModelCheckpoint, ProgressBar
    from utils import (
        set_seed, get_device, setup_distributed, cleanup_distributed,
        create_optimizer, create_scheduler, save_checkpoint, load_checkpoint,
        AverageMeter, Timer, validate_config, log_system_info, get_memory_usage
    )
except ImportError as e:
    print(f"Warning: Some modules not found: {e}")
    # Create minimal fallbacks
    class SimpleLogger:
        def __init__(self, config=None):
            pass
        def log_text(self, text): print(text)
        def log_metrics(self, metrics, step=None): print(f"Step {step}: {metrics}")
        def log_model_info(self, model, optimizer, scheduler=None): pass
        def log_hyperparameters(self, hparams): pass
        def update_epoch(self, epoch): pass
        def close(self): pass
    
    TrainingLogger = SimpleLogger
    
    def set_seed(seed): pass
    def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def setup_distributed(): return False
    def cleanup_distributed(): pass
    def create_optimizer(model, config): return torch.optim.AdamW(model.parameters(), lr=1e-4)
    def create_scheduler(optimizer, config): return None
    def save_checkpoint(*args, **kwargs): pass
    def load_checkpoint(*args, **kwargs): return {}
    def validate_config(config): pass
    def log_system_info(logger): pass
    def get_memory_usage(): return {}
    
    class AverageMeter:
        def __init__(self):
            self.reset()
        def reset(self):
            self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
        def update(self, val, n=1):
            self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    
    class Timer:
        def __init__(self):
            self.start_time = None
        def start(self):
            self.start_time = time.time()
        def elapsed(self):
            return time.time() - self.start_time if self.start_time else 0

class Trainer:
    """Production-ready training infrastructure with flexible loss function support"""
    
    def __init__(self, 
                 model: nn.Module,
                 config,  # TrainingConfig
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 callbacks: Optional[List] = None,
                 loss_fn: Optional[Callable] = None):
        
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()  # Default loss function
        
        # Validate configuration
        try:
            validate_config(config)
        except Exception as e:
            print(f"Warning: Config validation failed: {e}")
        
        # Set random seed
        try:
            set_seed(getattr(config, 'seed', 42))
        except:
            torch.manual_seed(42)
        
        # Setup device and distributed training
        self.device = get_device()
        try:
            self.is_distributed = setup_distributed()
            self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        except:
            self.is_distributed = False
            self.is_main_process = True
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup distributed model
        if self.is_distributed:
            try:
                self.model = DDP(
                    self.model,
                    device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                    find_unused_parameters=getattr(config.distributed, 'find_unused_parameters', False),
                    gradient_as_bucket_view=getattr(config.distributed, 'gradient_as_bucket_view', True)
                )
            except Exception as e:
                print(f"Warning: Failed to setup DDP: {e}")
                self.is_distributed = False
        
        # Initialize training components
        try:
            self.optimizer = create_optimizer(self.model, config)
            self.scheduler = create_scheduler(self.optimizer, config)
        except Exception as e:
            print(f"Warning: Using default optimizer: {e}")
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=getattr(config.optimizer, 'lr', 1e-4))
            self.scheduler = None
        
        # Mixed precision
        self.scaler = None
        try:
            if getattr(config, 'use_amp', False) and self.device.type == 'cuda':
                self.scaler = GradScaler()
                self.amp_dtype = getattr(torch, getattr(config, 'amp_dtype', 'float16'))
        except Exception as e:
            print(f"Warning: Mixed precision disabled: {e}")
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False
        self.best_metric = float('inf')
        
        # Setup logger (only on main process)
        self.logger = None
        if self.is_main_process:
            try:
                self.logger = TrainingLogger(config)
                log_system_info(self.logger)
            except Exception as e:
                print(f"Warning: Using simple logger: {e}")
                self.logger = SimpleLogger(config)
        
        # Setup callbacks
        self.callbacks = callbacks or []
        if self.is_main_process:
            # Add default callbacks if available
            try:
                checkpoint_dir = getattr(config.checkpoint, 'save_dir', './checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Only add if ModelCheckpoint is available
                if 'ModelCheckpoint' in globals():
                    self.callbacks.extend([
                        ModelCheckpoint(
                            filepath=os.path.join(checkpoint_dir, 'checkpoint-{epoch:03d}.pt'),
                            monitor=getattr(config.checkpoint, 'monitor', 'val_loss'),
                            mode=getattr(config.checkpoint, 'mode', 'min'),
                            save_best_only=False,
                            period=1
                        ),
                        ModelCheckpoint(
                            filepath=os.path.join(checkpoint_dir, 'best-model.pt'),
                            monitor=getattr(config.checkpoint, 'monitor', 'val_loss'),
                            mode=getattr(config.checkpoint, 'mode', 'min'),
                            save_best_only=True
                        )
                    ])
            except Exception as e:
                print(f"Warning: Could not setup default callbacks: {e}")
        
        # Resume from checkpoint if specified
        if hasattr(config, 'resume_from') and config.resume_from:
            self.resume_from_checkpoint(config.resume_from)
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        try:
            if self.is_main_process and self.logger:
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
        except Exception as e:
            print(f"Warning: Failed to resume from checkpoint: {e}")
    
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
            log_every = getattr(self.config.logging, 'log_every', 100) if hasattr(self.config, 'logging') else 100
            if self.global_step % log_every == 0 and self.is_main_process and self.logger:
                metrics = {
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.current_epoch,
                    'train/batch_time': batch_time.avg,
                    'train/data_time': data_time.avg,
                }
                
                # Add memory usage if CUDA
                if self.device.type == 'cuda':
                    try:
                        memory_info = get_memory_usage()
                        metrics.update({f'memory/{k}': v for k, v in memory_info.items()})
                    except:
                        pass
                
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
            max_steps = getattr(self.config, 'max_steps', None)
            if max_steps and self.global_step >= max_steps:
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
        use_amp = getattr(self.config, 'use_amp', False) if hasattr(self.config, 'use_amp') else False
        if use_amp and self.scaler:
            with autocast(dtype=self.amp_dtype):
                loss = self.compute_loss(batch)
            
            # Scale loss and backward pass
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # Gradient clipping
            max_grad_norm = getattr(self.config, 'max_grad_norm', 0)
            if max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_grad_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular forward pass
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = getattr(self.config, 'max_grad_norm', 0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        # Scheduler step (if step-based)
        if self.scheduler and hasattr(self.scheduler, 'step') and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return loss
    
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch using the provided loss function"""
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
                outputs = self.model(inputs)
                return self.loss_fn(outputs, targets)
            else:
                # For more complex batch formats, pass all to loss function
                return self.loss_fn(*batch)
        elif isinstance(batch, dict):
            # For dictionary batches, try common patterns
            if 'input_ids' in batch and 'labels' in batch:
                outputs = self.model(**batch)
                if hasattr(outputs, 'loss'):
                    return outputs.loss
                else:
                    return self.loss_fn(outputs.logits, batch['labels'])
            else:
                # Generic dict handling
                outputs = self.model(**batch)
                if hasattr(outputs, 'loss'):
                    return outputs.loss
                else:
                    raise ValueError("Cannot compute loss from dict batch without 'labels' key")
        else:
            # Single tensor batch
            outputs = self.model(batch)
            if hasattr(outputs, 'loss'):
                return outputs.loss
            else:
                raise ValueError("Cannot compute loss from single tensor batch without targets")
    
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
                use_amp = getattr(self.config, 'use_amp', False) if hasattr(self.config, 'use_amp') else False
                if use_amp and self.scaler:
                    with autocast(dtype=self.amp_dtype):
                        loss = self.compute_loss(batch)
                else:
                    loss = self.compute_loss(batch)
                
                val_loss.update(loss.item(), self._get_batch_size(batch))
                
                # Break early if specified
                eval_steps = getattr(self.config, 'eval_steps', None)
                if eval_steps and batch_idx >= eval_steps:
                    break
        
        val_metrics = {
            'val/loss': val_loss.avg,
        }
        
        # Call callbacks
        self._call_callbacks('on_validation_end', val_metrics)
        
        return val_metrics
    
    def fit(self):
        """Main training loop"""
        if self.is_main_process and self.logger:
            try:
                self.logger.log_model_info(self.model, self.optimizer, self.scheduler)
                config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
                self.logger.log_hyperparameters(config_dict)
            except Exception as e:
                print(f"Warning: Could not log model info: {e}")
        
        # Call callbacks
        self._call_callbacks('on_train_begin')
        
        try:
            epochs = getattr(self.config, 'epochs', 10)
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                
                # Set epoch for distributed sampler
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = {}
                eval_every = getattr(self.config, 'eval_every', 1)
                if epoch % eval_every == 0 or epoch == epochs - 1:
                    val_metrics = self.validate()
                
                # Combine metrics
                epoch_logs = {**train_metrics, **val_metrics}
                
                # Log epoch metrics
                if self.is_main_process and self.logger:
                    self.logger.log_metrics(epoch_logs, self.global_step)
                    self.logger.update_epoch(epoch)
                
                # Call callbacks
                self._call_callbacks('on_epoch_end', epoch, epoch_logs)
                
                # Check early stopping
                if self.should_stop:
                    if self.is_main_process and self.logger:
                        self.logger.log_text(f"Training stopped early at epoch {epoch + 1}")
                    break
                
                # Save checkpoint periodically
                if self.is_main_process and hasattr(self.config, 'checkpoint'):
                    save_every = getattr(self.config.checkpoint, 'save_every', 1000)
                    if (epoch + 1) % save_every == 0:
                        try:
                            checkpoint_path = os.path.join(
                                getattr(self.config.checkpoint, 'save_dir', './checkpoints'), 
                                f'checkpoint-epoch-{epoch + 1:03d}.pt'
                            )
                            save_checkpoint(
                                self.model, self.optimizer, self.scheduler,
                                epoch, self.global_step, epoch_logs.get('train/epoch_loss', 0),
                                checkpoint_path, self.config, self.scaler
                            )
                        except Exception as e:
                            print(f"Warning: Could not save checkpoint: {e}")
        
        except KeyboardInterrupt:
            if self.is_main_process and self.logger:
                self.logger.log_text("Training interrupted by user")
        
        finally:
            # Call callbacks
            self._call_callbacks('on_train_end')
            
            # Save final checkpoint
            if self.is_main_process:
                try:
                    checkpoint_dir = getattr(self.config.checkpoint, 'save_dir', './checkpoints') if hasattr(self.config, 'checkpoint') else './checkpoints'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    final_checkpoint = os.path.join(checkpoint_dir, 'final-checkpoint.pt')
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.current_epoch, self.global_step, 0,
                        final_checkpoint, self.config, self.scaler
                    )
                except Exception as e:
                    print(f"Warning: Could not save final checkpoint: {e}")
                
                if self.logger:
                    self.logger.close()
            
            # Cleanup distributed
            if self.is_distributed:
                cleanup_distributed()
    
    def _call_callbacks(self, method_name: str, *args, **kwargs):
        """Call method on all callbacks"""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    getattr(callback, method_name)(self, *args, **kwargs)
                except Exception as e:
                    print(f"Warning: Callback {callback.__class__.__name__}.{method_name} failed: {e}")
    
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
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), filepath)
            
            if self.is_main_process and self.logger:
                self.logger.log_text(f"Model saved to {filepath}")
            else:
                print(f"✅ Model saved to {filepath}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_state_dict(state_dict)
            
            if self.is_main_process and self.logger:
                self.logger.log_text(f"Model loaded from {filepath}")
            else:
                print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

# Custom trainer for specific tasks
class LanguageModelTrainer(Trainer):
    """Trainer specifically for language modeling tasks"""
    
    def __init__(self, *args, **kwargs):
        # Set default loss function for language modeling
        if 'loss_fn' not in kwargs:
            kwargs['loss_fn'] = nn.CrossEntropyLoss(ignore_index=-100)
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, batch):
        """Compute language modeling loss with proper label shifting"""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, labels = batch
            outputs = self.model(input_ids)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 'last_hidden_state'):
                # If model doesn't have LM head, we can't compute loss
                raise ValueError("Model doesn't have language modeling head")
            else:
                logits = outputs
            
            # Shift labels for next token prediction
            if logits.size(-1) > 1:  # Multi-class prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                return self.loss_fn(logits.view(-1), labels.view(-1))
        
        elif isinstance(batch, dict):
            # Handle transformers format
            if 'input_ids' in batch and 'labels' in batch:
                outputs = self.model(**batch)
                if hasattr(outputs, 'loss'):
                    return outputs.loss
                else:
                    return self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                      batch['labels'].view(-1))
        
        # Fallback to parent implementation
        return super().compute_loss(batch)

class ClassificationTrainer(Trainer):
    """Trainer for classification tasks with accuracy metrics"""
    
    def __init__(self, *args, **kwargs):
        # Set default loss function for classification
        if 'loss_fn' not in kwargs:
            kwargs['loss_fn'] = nn.CrossEntropyLoss()
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, batch):
        """Compute classification loss"""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            outputs = self.model(inputs)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            return self.loss_fn(logits, targets)
        
        # Fallback to parent implementation
        return super().compute_loss(batch)
    
    def validate(self):
        """Validate with accuracy metrics"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        # Call callbacks
        self._call_callbacks('on_validation_begin')
        
        val_loss = AverageMeter()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    
                    # Handle different output formats
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    loss = self.loss_fn(logits, targets)
                    val_loss.update(loss.item(), targets.size(0))
                    
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        val_metrics = {
            'val/loss': val_loss.avg,
            'val/accuracy': accuracy
        }
        
        # Call callbacks
        self._call_callbacks('on_validation_end', val_metrics)
        
        return val_metrics

# Flexible trainer with custom loss function
class FlexibleTrainer(Trainer):
    """Trainer that accepts any custom loss computation function"""
    
    def __init__(self, *args, compute_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        if compute_loss_fn:
            self.custom_compute_loss = compute_loss_fn
    
    def compute_loss(self, batch):
        """Use custom loss function if provided, otherwise use default"""
        if hasattr(self, 'custom_compute_loss'):
            return self.custom_compute_loss(self, batch)
        else:
            return super().compute_loss(batch)