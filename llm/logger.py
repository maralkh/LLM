# training_infra/logger.py
import os
import json
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class TrainingLogger:
    """Production-ready training logger with multiple backends"""
    
    def __init__(self, config):
        self.config = config
        self.log_dir = Path(config.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logger()
        
        # Initialize metrics storage
        self.metrics_history = defaultdict(list)
        self.current_epoch = 0
        self.current_step = 0
        
        # Initialize backends
        self.wandb_run = None
        self.tb_writer = None
        
        if config.logging.use_wandb and WANDB_AVAILABLE:
            self._setup_wandb()
        elif config.logging.use_wandb:
            self.logger.warning("wandb requested but not available")
            
        if config.logging.use_tensorboard and TENSORBOARD_AVAILABLE:
            self._setup_tensorboard()
        elif config.logging.use_tensorboard:
            self.logger.warning("tensorboard requested but not available")
    
    def _setup_file_logger(self):
        """Setup file logging"""
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = self.log_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            self.wandb_run = wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                config=self.config.to_dict(),
                dir=str(self.log_dir)
            )
            self.logger.info("WandB initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging"""
        try:
            tb_dir = self.log_dir / 'tensorboard'
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
            self.logger.info("TensorBoard initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.tb_writer = None
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to all backends"""
        if step is None:
            step = self.current_step
        
        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append({
                'step': step,
                'value': value,
                'timestamp': time.time()
            })
        
        # Log to WandB
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        
        # Log to file (if should log this step)
        if step % self.config.logging.log_every == 0:
            metrics_str = ' | '.join([f'{k}: {v:.6f}' for k, v in metrics.items()])
            self.logger.info(f'Step {step} | {metrics_str}')
    
    def log_text(self, text: str, step: Optional[int] = None):
        """Log text message"""
        if step is None:
            step = self.current_step
            
        self.logger.info(text)
        
        if self.wandb_run:
            self.wandb_run.log({"log": text}, step=step)
    
    def log_model_info(self, model, optimizer, scheduler=None):
        """Log model and optimizer information"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/non_trainable_parameters": total_params - trainable_params,
            "optimizer/name": optimizer.__class__.__name__,
            "optimizer/lr": optimizer.param_groups[0]['lr']
        }
        
        if scheduler:
            info["scheduler/name"] = scheduler.__class__.__name__
        
        self.log_metrics(info)
        
        # Log model architecture to file
        model_info_file = self.log_dir / 'model_info.txt'
        with open(model_info_file, 'w') as f:
            f.write(str(model))
            f.write(f"\n\nTotal Parameters: {total_params:,}")
            f.write(f"\nTrainable Parameters: {trainable_params:,}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        if self.wandb_run:
            self.wandb_run.config.update(hparams)
        
        if self.tb_writer:
            # TensorBoard expects specific format
            hparams_clean = {}
            for k, v in hparams.items():
                if isinstance(v, (int, float, str, bool)):
                    hparams_clean[k] = v
            
            if hparams_clean:
                self.tb_writer.add_hparams(hparams_clean, {})
        
        # Save to JSON
        hparams_file = self.log_dir / 'hyperparameters.json'
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
    
    def save_metrics_history(self):
        """Save complete metrics history to file"""
        history_file = self.log_dir / 'metrics_history.json'
        with open(history_file, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
    
    def update_step(self, step: int):
        """Update current step"""
        self.current_step = step
    
    def update_epoch(self, epoch: int):
        """Update current epoch"""
        self.current_epoch = epoch
    
    def close(self):
        """Close all logging backends"""
        self.save_metrics_history()
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        if self.tb_writer:
            self.tb_writer.close()
        
        self.logger.info("Training completed. All logs saved.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()