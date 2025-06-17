# training_infra/callbacks.py
import torch
import numpy as np
from typing import Any, Dict, Optional, List
from pathlib import Path
import json
import time
from abc import ABC, abstractmethod

class Callback(ABC):
    """Base callback class"""
    
    def on_train_begin(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        pass
    
    def on_batch_begin(self, trainer, batch_idx):
        pass
    
    def on_batch_end(self, trainer, batch_idx, logs=None):
        pass
    
    def on_validation_begin(self, trainer):
        pass
    
    def on_validation_end(self, trainer, logs=None):
        pass

class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, monitor='val_loss', patience=5, mode='min', 
                 min_delta=0.0, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
        
        self.best = np.Inf if mode == 'min' else -np.Inf
    
    def on_train_begin(self, trainer):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.mode == 'min' else -np.Inf
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            trainer.logger.logger.warning(f'Early stopping conditioned on metric `{self.monitor}` which is not available.')
            return
        
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone().cpu() for k, v in trainer.model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True
                trainer.logger.logger.info(f'Early stopping at epoch {epoch + 1}')
                
                if self.restore_best_weights and self.best_weights:
                    trainer.logger.logger.info('Restoring model weights from best epoch')
                    trainer.model.load_state_dict({k: v.to(trainer.device) for k, v in self.best_weights.items()})

class ModelCheckpoint(Callback):
    """Model checkpointing callback"""
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=False,
                 save_weights_only=False, period=1):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        
        self.epochs_since_last_save = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        else:
            self.monitor_op = np.greater
            self.best = -np.Inf
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        self.epochs_since_last_save += 1
        
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            if self.save_best_only:
                if logs is None:
                    return
                
                current = logs.get(self.monitor)
                if current is None:
                    return
                
                if self.monitor_op(current, self.best):
                    self.best = current
                    self._save_checkpoint(trainer, epoch, logs)
            else:
                self._save_checkpoint(trainer, epoch, logs)
    
    def _save_checkpoint(self, trainer, epoch, logs):
        filepath = str(self.filepath).format(epoch=epoch + 1, **logs if logs else {})
        
        if self.save_weights_only:
            torch.save(trainer.model.state_dict(), filepath)
        else:
            checkpoint = {
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'step': trainer.global_step,
                'best_metric': self.best,
                'config': trainer.config.to_dict()
            }
            
            if trainer.scheduler:
                checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
            if trainer.scaler:
                checkpoint['scaler_state_dict'] = trainer.scaler.state_dict()
            
            torch.save(checkpoint, filepath)
        
        trainer.logger.logger.info(f'Checkpoint saved to {filepath}')

class LearningRateScheduler(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(self, scheduler, monitor=None):
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        if self.monitor:
            if logs and self.monitor in logs:
                self.scheduler.step(logs[self.monitor])
            else:
                trainer.logger.logger.warning(f'Metric {self.monitor} not found in logs')
        else:
            self.scheduler.step()

class ReduceLROnPlateau(Callback):
    """Reduce learning rate on plateau"""
    
    def __init__(self, monitor='val_loss', factor=0.5, patience=3, mode='min',
                 min_delta=1e-4, cooldown=0, min_lr=0, verbose=True):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait = 0
        self.cooldown_counter = 0
        
        if mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
    
    def on_train_begin(self, trainer):
        self.wait = 0
        self.cooldown_counter = 0
        self.best = np.Inf if self.mode == 'min' else -np.Inf
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.cooldown_counter > 0:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = trainer.optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                
                if old_lr - new_lr > np.finfo(np.float32).eps:
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    if self.verbose:
                        trainer.logger.logger.info(f'ReduceLROnPlateau reducing learning rate to {new_lr}')
                    
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

class ProgressBar(Callback):
    """Progress bar callback"""
    
    def __init__(self, update_freq=1):
        self.update_freq = update_freq
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, trainer):
        self.start_time = time.time()
        trainer.logger.logger.info(f'Starting training for {trainer.config.epochs} epochs')
    
    def on_epoch_begin(self, trainer, epoch):
        self.epoch_start_time = time.time()
        trainer.logger.logger.info(f'Epoch {epoch + 1}/{trainer.config.epochs}')
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        
        log_msg = f'Epoch {epoch + 1} completed in {epoch_time:.2f}s'
        if logs:
            metrics_str = ' | '.join([f'{k}: {v:.6f}' for k, v in logs.items()])
            log_msg += f' | {metrics_str}'
        
        trainer.logger.logger.info(log_msg)
        trainer.logger.logger.info(f'Total training time: {total_time:.2f}s')

class MetricsHistory(Callback):
    """Track metrics history"""
    
    def __init__(self):
        self.history = {}
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        if logs:
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
    
    def save_history(self, filepath):
        """Save metrics history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

class GradientClipping(Callback):
    """Gradient clipping callback"""
    
    def __init__(self, max_norm=1.0, norm_type=2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_batch_end(self, trainer, batch_idx, logs=None):
        if trainer.model.training:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), 
                self.max_norm, 
                norm_type=self.norm_type
            )
            
            if logs:
                logs['grad_norm'] = grad_norm.item()