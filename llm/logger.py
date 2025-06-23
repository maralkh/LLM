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
    
    def _format_metric_value(self, value):
        """Format metric value for display"""
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                # Format floats with appropriate precision
                if abs(value) >= 1000:
                    return f"{value:.2f}"
                elif abs(value) >= 1:
                    return f"{value:.4f}"
                elif abs(value) >= 0.01:
                    return f"{value:.6f}"
                else:
                    return f"{value:.2e}"
            else:
                # Format integers
                return str(value)
        elif isinstance(value, str):
            return value
        elif hasattr(value, 'item'):  # Handle tensor scalars
            return self._format_metric_value(value.item())
        else:
            return str(value)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to all backends"""
        if step is None:
            step = self.current_step
        
        # Clean metrics for different backends
        clean_metrics = {}
        for key, value in metrics.items():
            # Convert tensor scalars to Python scalars
            if hasattr(value, 'item'):
                clean_value = value.item()
            else:
                clean_value = value
            
            # Only log numeric values to external backends
            if isinstance(clean_value, (int, float)) and not (
                isinstance(clean_value, float) and (
                    clean_value != clean_value or  # NaN check
                    clean_value == float('inf') or 
                    clean_value == float('-inf')
                )
            ):
                clean_metrics[key] = clean_value
        
        # Store in history
        for key, value in clean_metrics.items():
            self.metrics_history[key].append({
                'step': step,
                'value': value,
                'timestamp': time.time()
            })
        
        # Log to WandB
        if self.wandb_run:
            try:
                self.wandb_run.log(clean_metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")
        
        # Log to TensorBoard
        if self.tb_writer:
            try:
                for key, value in clean_metrics.items():
                    self.tb_writer.add_scalar(key, value, step)
            except Exception as e:
                self.logger.warning(f"Failed to log to TensorBoard: {e}")
        
        # Log to file (if should log this step)
        if step % self.config.logging.log_every == 0:
            try:
                # Format all original metrics (including strings) for display
                formatted_metrics = []
                for k, v in metrics.items():
                    formatted_value = self._format_metric_value(v)
                    formatted_metrics.append(f'{k}: {formatted_value}')
                
                metrics_str = ' | '.join(formatted_metrics)
                self.logger.info(f'Step {step} | {metrics_str}')
            except Exception as e:
                # Fallback to simple string representation
                self.logger.info(f'Step {step} | {str(metrics)}')
                self.logger.warning(f"Error formatting metrics: {e}")
    
    def log_text(self, text: str, step: Optional[int] = None):
        """Log text message"""
        if step is None:
            step = self.current_step
            
        self.logger.info(text)
        
        if self.wandb_run:
            try:
                self.wandb_run.log({"log": text}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log text to WandB: {e}")
    
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
        try:
            model_info_file = self.log_dir / 'model_info.txt'
            with open(model_info_file, 'w') as f:
                f.write(str(model))
                f.write(f"\n\nTotal Parameters: {total_params:,}")
                f.write(f"\nTrainable Parameters: {trainable_params:,}")
        except Exception as e:
            self.logger.warning(f"Failed to save model info: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        if self.wandb_run:
            try:
                self.wandb_run.config.update(hparams)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to WandB: {e}")
        
        if self.tb_writer:
            try:
                # TensorBoard expects specific format
                hparams_clean = {}
                for k, v in hparams.items():
                    if isinstance(v, (int, float, str, bool)):
                        hparams_clean[k] = v
                
                if hparams_clean:
                    self.tb_writer.add_hparams(hparams_clean, {})
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
        
        # Save to JSON
        try:
            hparams_file = self.log_dir / 'hyperparameters.json'
            with open(hparams_file, 'w') as f:
                json.dump(hparams, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save hyperparameters to file: {e}")
    
    def save_metrics_history(self):
        """Save complete metrics history to file"""
        try:
            history_file = self.log_dir / 'metrics_history.json'
            with open(history_file, 'w') as f:
                json.dump(dict(self.metrics_history), f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save metrics history: {e}")
    
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
            try:
                self.wandb_run.finish()
            except Exception as e:
                self.logger.warning(f"Failed to close WandB: {e}")
        
        if self.tb_writer:
            try:
                self.tb_writer.close()
            except Exception as e:
                self.logger.warning(f"Failed to close TensorBoard: {e}")
        
        self.logger.info("Training completed. All logs saved.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Fallback logger for when the main logger is not available
class SimpleLogger:
    """Simple fallback logger"""
    
    def __init__(self, config=None):
        self.logger = logging.getLogger('simple_training')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Simple metric logging"""
        try:
            formatted_metrics = []
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    if isinstance(v, float):
                        formatted_metrics.append(f'{k}: {v:.4f}')
                    else:
                        formatted_metrics.append(f'{k}: {v}')
                else:
                    formatted_metrics.append(f'{k}: {str(v)}')
            
            metrics_str = ' | '.join(formatted_metrics)
            step_str = f"Step {step}" if step is not None else "Metrics"
            self.logger.info(f'{step_str} | {metrics_str}')
        except Exception as e:
            self.logger.info(f'Metrics: {str(metrics)} (formatting error: {e})')
    
    def log_text(self, text: str, step: Optional[int] = None):
        """Simple text logging"""
        self.logger.info(text)
    
    def log_model_info(self, model, optimizer, scheduler=None):
        """Simple model info logging"""
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model parameters: {total_params:,}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Simple hyperparameter logging"""
        self.logger.info(f"Hyperparameters: {hparams}")
    
    def update_step(self, step: int):
        """Update step (no-op for simple logger)"""
        pass
    
    def update_epoch(self, epoch: int):
        """Update epoch (no-op for simple logger)"""
        pass
    
    def close(self):
        """Close logger"""
        self.logger.info("Training completed.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()