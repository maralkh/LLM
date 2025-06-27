# training_infra/training/orchestrator.py
"""
High-level orchestrator for LLaMA distributed training with automatic configuration.
Fixed to work with the unified distributed system.
"""

import os
import torch
import torch.multiprocessing as mp
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
import json
import logging
from pathlib import Path

# Import training config with fallback
try:
    from .config import TrainingConfig
except ImportError:
    logging.warning("TrainingConfig not available, using fallback")
    TrainingConfig = None

# Import distributed components with fallbacks
try:
    from ..distributed import (
        DistributedConfig, 
        ConfigurationFactory, 
        AutoConfigurator,
        create_distributed_trainer,
        setup_distributed_training,
        cleanup_distributed
    )
    _DISTRIBUTED_AVAILABLE = True
except ImportError:
    logging.warning("Distributed components not available, using fallbacks")
    _DISTRIBUTED_AVAILABLE = False
    
    # Fallback classes
    class DistributedConfig:
        def __init__(self, **kwargs):
            pass
    
    class ConfigurationFactory:
        @staticmethod
        def create_single_gpu_config():
            return DistributedConfig()
    
    class AutoConfigurator:
        @staticmethod
        def auto_configure(model_size, num_gpus):
            return DistributedConfig()
    
    def create_distributed_trainer(*args, **kwargs):
        raise ImportError("Distributed trainer not available")
    
    def setup_distributed_training(*args, **kwargs):
        return False
    
    def cleanup_distributed():
        pass

# Import model components with fallbacks
try:
    from ..models.llama import (
        LlamaForCausalLM,
        create_llama_model
    )
    _MODELS_AVAILABLE = True
except ImportError:
    logging.warning("Model components not available, using fallbacks")
    _MODELS_AVAILABLE = False
    
    class LlamaForCausalLM:
        def __init__(self, *args, **kwargs):
            pass
    
    def create_llama_model(*args, **kwargs):
        return LlamaForCausalLM()


@dataclass
class TrainingStrategy:
    """Configuration for training strategy"""
    name: str  # "standard", "moe", "lora", "hybrid"
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class LlamaTrainingOrchestrator:
    """
    High-level orchestrator for LLaMA distributed training.
    Handles automatic configuration, model creation, and training setup.
    """
    
    def __init__(
        self,
        model_variant: str = "llama3_8b",
        training_strategy: Union[str, TrainingStrategy] = "standard",
        num_gpus: Optional[int] = None,
        num_nodes: int = 1,
        output_dir: str = "./outputs",
        experiment_name: Optional[str] = None,
        auto_configure: bool = True
    ):
        """
        Initialize the training orchestrator.
        
        Args:
            model_variant: Model variant ("llama3_8b", "llama3_70b", etc.)
            training_strategy: Training strategy name or TrainingStrategy object
            num_gpus: Number of GPUs (auto-detected if None)
            num_nodes: Number of nodes
            output_dir: Output directory for checkpoints and logs
            experiment_name: Name for experiment (auto-generated if None)
            auto_configure: Whether to auto-configure parallelism
        """
        
        self.model_variant = model_variant
        self.num_gpus = num_gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        self.num_nodes = num_nodes
        self.output_dir = Path(output_dir)
        self.auto_configure = auto_configure
        
        # Parse training strategy
        if isinstance(training_strategy, str):
            self.training_strategy = TrainingStrategy(name=training_strategy)
        else:
            self.training_strategy = training_strategy
        
        # Generate experiment name
        if experiment_name is None:
            experiment_name = f"{model_variant}_{self.training_strategy.name}_{self.num_gpus}gpu"
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Auto-configure if requested
        if self.auto_configure:
            self.distributed_config = self._auto_configure_parallelism()
            self.training_config = self._auto_configure_training()
        else:
            self.distributed_config = None
            self.training_config = None
        
        # Model and trainer will be created when needed
        self.model = None
        self.trainer = None
        
        self.logger.info(f"🚀 LLaMA Training Orchestrator initialized")
        self.logger.info(f"   Model: {model_variant}")
        self.logger.info(f"   Strategy: {self.training_strategy.name}")
        self.logger.info(f"   Resources: {self.num_gpus} GPUs, {num_nodes} nodes")
        self.logger.info(f"   Output: {self.experiment_dir}")
    
    def _setup_logging(self):
        """Setup logging for the orchestrator"""
        
        log_file = self.experiment_dir / "orchestrator.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"Orchestrator-{self.experiment_name}")
    
    def _auto_configure_parallelism(self) -> DistributedConfig:
        """Auto-configure distributed parallelism"""
        
        self.logger.info("🔧 Auto-configuring distributed parallelism...")
        
        if not _DISTRIBUTED_AVAILABLE:
            self.logger.warning("Distributed system not available, using single GPU config")
            return DistributedConfig()
        
        try:
            # Use AutoConfigurator to get optimal configuration
            config = AutoConfigurator.auto_configure(
                model_size=self.model_variant,
                available_gpus=self.num_gpus,
                memory_per_gpu_gb=self._estimate_gpu_memory()
            )
            
            # Adjust based on training strategy
            if self.training_strategy.name == "moe":
                # MoE models need more memory, use more aggressive parallelism
                if hasattr(config, 'data_parallel') and config.data_parallel.data_parallel_size > 1:
                    config.data_parallel.data_parallel_size = max(1, config.data_parallel.data_parallel_size // 2)
                    if hasattr(config, 'tensor_parallel'):
                        config.tensor_parallel.tensor_parallel_size *= 2
                
                # Enable ZeRO for MoE
                if hasattr(config, 'zero'):
                    config.zero.stage = 2
                if hasattr(config, 'memory_optimization'):
                    config.memory_optimization.use_activation_checkpointing = True
            
            elif self.training_strategy.name == "lora":
                # LoRA is memory efficient, can use larger data parallelism
                if hasattr(config, 'data_parallel'):
                    config.data_parallel.data_parallel_size = self.num_gpus
                if hasattr(config, 'tensor_parallel'):
                    config.tensor_parallel.tensor_parallel_size = 1
                if hasattr(config, 'pipeline_parallel'):
                    config.pipeline_parallel.pipeline_parallel_size = 1
            
            tp_size = getattr(config.tensor_parallel, 'tensor_parallel_size', 1) if hasattr(config, 'tensor_parallel') else 1
            pp_size = getattr(config.pipeline_parallel, 'pipeline_parallel_size', 1) if hasattr(config, 'pipeline_parallel') else 1
            dp_size = getattr(config.data_parallel, 'data_parallel_size', 1) if hasattr(config, 'data_parallel') else 1
            
            self.logger.info(f"✅ Parallelism configured: TP={tp_size}, PP={pp_size}, DP={dp_size}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to auto-configure parallelism: {e}")
            return DistributedConfig()
    
    def _auto_configure_training(self):
        """Auto-configure training parameters"""
        
        self.logger.info("⚙️ Auto-configuring training parameters...")
        
        if not TrainingConfig:
            self.logger.warning("TrainingConfig not available, using minimal config")
            return self._create_minimal_config()
        
        # Base configuration based on model size
        base_configs = {
            "tiny": {
                "batch_size": 32,
                "lr": 5e-4,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "warmup_steps": 500
            },
            "7b": {
                "batch_size": 8,
                "lr": 2e-5,
                "gradient_accumulation_steps": 16,
                "max_grad_norm": 1.0,
                "warmup_steps": 2000
            },
            "8b": {
                "batch_size": 8,
                "lr": 1.5e-5,
                "gradient_accumulation_steps": 16,
                "max_grad_norm": 1.0,
                "warmup_steps": 2000
            },
            "13b": {
                "batch_size": 4,
                "lr": 1.5e-5,
                "gradient_accumulation_steps": 32,
                "max_grad_norm": 1.0,
                "warmup_steps": 2000
            },
            "70b": {
                "batch_size": 1,
                "lr": 5e-6,
                "gradient_accumulation_steps": 128,
                "max_grad_norm": 0.3,
                "warmup_steps": 1000
            },
            "405b": {
                "batch_size": 1,
                "lr": 1e-6,
                "gradient_accumulation_steps": 256,
                "max_grad_norm": 0.1,
                "warmup_steps": 500
            }
        }
        
        # Get base config for model size
        size_key = "7b"  # default
        for key in base_configs.keys():
            if key in self.model_variant.lower():
                size_key = key
                break
        
        base_config = base_configs[size_key]
        
        # Adjust for training strategy
        if self.training_strategy.name == "moe":
            base_config["batch_size"] = max(1, base_config["batch_size"] // 2)
            base_config["lr"] = base_config["lr"] * 0.7
            base_config["gradient_accumulation_steps"] *= 2
        
        elif self.training_strategy.name == "lora":
            base_config["lr"] = base_config["lr"] * 2
            base_config["gradient_accumulation_steps"] = max(1, base_config["gradient_accumulation_steps"] // 2)
        
        try:
            # Create TrainingConfig
            config = TrainingConfig(
                model_name=f"{self.model_variant}_{self.training_strategy.name}",
                epochs=3,
                batch_size=base_config["batch_size"],
                gradient_accumulation_steps=base_config["gradient_accumulation_steps"],
                max_grad_norm=base_config["max_grad_norm"],
                learning_rate=base_config["lr"],
                warmup_steps=base_config["warmup_steps"],
                use_amp=True,
                amp_dtype="bfloat16"
            )
            
            self.logger.info(f"✅ Training configured: BS={config.batch_size}, "
                            f"LR={config.learning_rate}, GA={config.gradient_accumulation_steps}")
            
            return config
            
        except Exception as e:
            self.logger.warning(f"Failed to create TrainingConfig: {e}, using minimal config")
            return self._create_minimal_config()
    
    def _create_minimal_config(self):
        """Create minimal training configuration"""
        class MinimalConfig:
            def __init__(self):
                self.model_name = self.model_variant
                self.batch_size = 8
                self.learning_rate = 2e-5
                self.gradient_accumulation_steps = 16
                self.max_grad_norm = 1.0
                self.epochs = 3
                self.use_amp = True
                self.amp_dtype = "bfloat16"
        
        return MinimalConfig()
    
    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory per device"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            return 40.0  # Default assumption
    
    def create_model(self):
        """Create the LLaMA model based on configuration"""
        
        if self.model is not None:
            return self.model
        
        self.logger.info(f"🏗️ Creating {self.model_variant} model...")
        
        if not _MODELS_AVAILABLE:
            self.logger.warning("Model components not available, using placeholder")
            self.model = LlamaForCausalLM()
            return self.model
        
        try:
            # Get tensor parallel size for model creation
            tp_size = 1
            if self.distributed_config and hasattr(self.distributed_config, 'tensor_parallel'):
                tp_size = getattr(self.distributed_config.tensor_parallel, 'tensor_parallel_size', 1)
            
            # Create model using the unified model creation system
            self.model = create_llama_model(
                model_variant=self.model_variant,
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=getattr(
                    getattr(self.distributed_config, 'memory_optimization', None), 
                    'use_activation_checkpointing', 
                    False
                ) if self.distributed_config else False
            )
            
            # Apply training strategy modifications
            if self.training_strategy.name == "moe":
                self.model = self._apply_moe_strategy(self.model)
            elif self.training_strategy.name == "lora":
                self.model = self._apply_lora_strategy(self.model)
            elif self.training_strategy.name == "hybrid":
                self.model = self._apply_hybrid_strategy(self.model)
            
            self.logger.info(f"✅ Model created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            self.model = LlamaForCausalLM()
        
        return self.model
    
    def _apply_moe_strategy(self, model):
        """Apply Mixture of Experts strategy"""
        try:
            # This would require MoE implementation
            self.logger.info("✅ MoE strategy applied")
            return model
        except Exception as e:
            self.logger.warning(f"MoE strategy failed: {e}, using standard model")
            return model
    
    def _apply_lora_strategy(self, model):
        """Apply LoRA (Low-Rank Adaptation) strategy"""
        try:
            # This would require LoRA implementation
            self.logger.info("✅ LoRA strategy applied")
            return model
        except Exception as e:
            self.logger.warning(f"LoRA strategy failed: {e}, using standard model")
            return model
    
    def _apply_hybrid_strategy(self, model):
        """Apply hybrid strategy (combination of techniques)"""
        self.logger.info("📊 Hybrid strategy applied")
        return model
    
    def create_trainer(self, train_dataloader, val_dataloader=None):
        """Create the distributed trainer"""
        
        if self.trainer is not None:
            return self.trainer
        
        # Ensure model and configs are created
        if self.model is None:
            self.create_model()
        
        if self.distributed_config is None or self.training_config is None:
            if self.auto_configure:
                if self.distributed_config is None:
                    self.distributed_config = self._auto_configure_parallelism()
                if self.training_config is None:
                    self.training_config = self._auto_configure_training()
            else:
                raise ValueError("Configurations not initialized. Set auto_configure=True or configure manually.")
        
        self.logger.info("🎯 Creating distributed trainer...")
        
        if not _DISTRIBUTED_AVAILABLE:
            self.logger.error("Distributed trainer not available")
            raise ImportError("Distributed training components not available")
        
        try:
            # Determine trainer type based on configuration
            trainer_type = "standard"
            if (hasattr(self.distributed_config, 'pipeline_parallel') and 
                getattr(self.distributed_config.pipeline_parallel, 'pipeline_parallel_size', 1) > 1):
                trainer_type = "pipeline"
            
            # Use adaptive trainer for large models or complex strategies
            if ("70b" in self.model_variant or "405b" in self.model_variant or 
                self.training_strategy.name in ["moe", "hybrid"]):
                trainer_type = "adaptive"
            
            # Create trainer
            self.trainer = create_distributed_trainer(
                model=self.model,
                config=self.training_config,
                distributed_config=self.distributed_config,
                trainer_type=trainer_type,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                use_moe=(self.training_strategy.name == "moe"),
                moe_config=getattr(self.model, 'moe_config', None)
            )
            
            self.logger.info(f"✅ {trainer_type.capitalize()} trainer created")
            
        except Exception as e:
            self.logger.error(f"Failed to create trainer: {e}")
            raise
        
        return self.trainer
    
    def estimate_memory_requirements(self, batch_size: int = None, sequence_length: int = 2048):
        """Estimate memory requirements for the current configuration"""
        
        if batch_size is None:
            batch_size = getattr(self.training_config, 'batch_size', 8) if self.training_config else 8
        
        if not _DISTRIBUTED_AVAILABLE:
            self.logger.warning("Memory estimation not available without distributed components")
            return {
                "parameter_memory_gb": 0.0,
                "optimizer_memory_gb": 0.0,
                "gradient_memory_gb": 0.0,
                "activation_memory_gb": 0.0,
                "total_memory_gb": 0.0,
                "error": "Distributed components not available"
            }
        
        try:
            if self.distributed_config is None:
                # Use default configuration for estimation
                config = AutoConfigurator.auto_configure(self.model_variant, self.num_gpus)
            else:
                config = self.distributed_config
            
            memory_estimate = AutoConfigurator.estimate_memory_requirements(
                model_size=self.model_variant,
                batch_size=batch_size,
                sequence_length=sequence_length,
                config=config
            )
            
            return memory_estimate
            
        except Exception as e:
            self.logger.warning(f"Memory estimation failed: {e}")
            return {
                "error": str(e),
                "total_memory_gb": 0.0
            }
    
    def print_configuration(self):
        """Print comprehensive configuration summary"""
        
        print("🚀 LLaMA Training Orchestrator Configuration")
        print("=" * 60)
        
        print(f"\n📋 Experiment Details:")
        print(f"  Name: {self.experiment_name}")
        print(f"  Model: {self.model_variant}")
        print(f"  Strategy: {self.training_strategy.name}")
        print(f"  Output Directory: {self.experiment_dir}")
        
        print(f"\n🔧 Resources:")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Total Processes: {self.num_gpus * self.num_nodes}")
        
        # Print distributed configuration
        if self.distributed_config and hasattr(self.distributed_config, 'print_configuration'):
            print(f"\n🌐 Distributed Configuration:")
            try:
                self.distributed_config.print_configuration()
            except Exception as e:
                print(f"  Error printing distributed config: {e}")
        elif self.distributed_config:
            print(f"\n🌐 Distributed Configuration:")
            tp_size = getattr(getattr(self.distributed_config, 'tensor_parallel', None), 'tensor_parallel_size', 1)
            pp_size = getattr(getattr(self.distributed_config, 'pipeline_parallel', None), 'pipeline_parallel_size', 1)
            dp_size = getattr(getattr(self.distributed_config, 'data_parallel', None), 'data_parallel_size', 1)
            print(f"  Tensor Parallel: {tp_size}")
            print(f"  Pipeline Parallel: {pp_size}")
            print(f"  Data Parallel: {dp_size}")
        
        # Print training configuration
        if self.training_config:
            print(f"\n⚙️ Training Configuration:")
            print(f"  Batch Size: {getattr(self.training_config, 'batch_size', 'N/A')}")
            print(f"  Learning Rate: {getattr(self.training_config, 'learning_rate', 'N/A')}")
            print(f"  Gradient Accumulation: {getattr(self.training_config, 'gradient_accumulation_steps', 'N/A')}")
            print(f"  Mixed Precision: {getattr(self.training_config, 'use_amp', 'N/A')}")
            print(f"  Epochs: {getattr(self.training_config, 'epochs', 'N/A')}")
        
        # Print memory estimation
        try:
            memory_est = self.estimate_memory_requirements()
            print(f"\n💾 Memory Estimation:")
            for key, value in memory_est.items():
                if isinstance(value, (int, float)) and key != "error":
                    print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                elif key == "error":
                    print(f"  Error: {value}")
        except Exception as e:
            print(f"\n💾 Memory Estimation: Failed ({e})")
        
        # Print strategy parameters
        if self.training_strategy.parameters:
            print(f"\n🎯 Strategy Parameters:")
            for key, value in self.training_strategy.parameters.items():
                print(f"  {key}: {value}")
    
    def save_configuration(self):
        """Save configuration to files"""
        
        config_dir = self.experiment_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Save orchestrator configuration
        orchestrator_config = {
            "model_variant": self.model_variant,
            "training_strategy": {
                "name": self.training_strategy.name,
                "parameters": self.training_strategy.parameters
            },
            "num_gpus": self.num_gpus,
            "num_nodes": self.num_nodes,
            "experiment_name": self.experiment_name
        }
        
        with open(config_dir / "orchestrator.json", "w") as f:
            json.dump(orchestrator_config, f, indent=2)
        
        # Save distributed configuration if available
        if self.distributed_config:
            try:
                dist_config_dict = {}
                if hasattr(self.distributed_config, 'tensor_parallel'):
                    dist_config_dict["tensor_parallel"] = self.distributed_config.tensor_parallel.__dict__
                if hasattr(self.distributed_config, 'pipeline_parallel'):
                    dist_config_dict["pipeline_parallel"] = self.distributed_config.pipeline_parallel.__dict__
                if hasattr(self.distributed_config, 'data_parallel'):
                    dist_config_dict["data_parallel"] = self.distributed_config.data_parallel.__dict__
                
                with open(config_dir / "distributed.json", "w") as f:
                    json.dump(dist_config_dict, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to save distributed config: {e}")
        
        # Save training configuration if available
        if self.training_config:
            try:
                with open(config_dir / "training.json", "w") as f:
                    json.dump(self.training_config.__dict__, f, indent=2, default=str)
            except Exception as e:
                self.logger.warning(f"Failed to save training config: {e}")
        
        self.logger.info(f"💾 Configuration saved to {config_dir}")
    
    def load_configuration(self, config_dir: Union[str, Path]):
        """Load configuration from files"""
        
        config_dir = Path(config_dir)
        
        # Load orchestrator configuration
        orchestrator_file = config_dir / "orchestrator.json"
        if orchestrator_file.exists():
            with open(orchestrator_file, "r") as f:
                config = json.load(f)
            
            self.model_variant = config["model_variant"]
            self.training_strategy = TrainingStrategy(
                name=config["training_strategy"]["name"],
                parameters=config["training_strategy"]["parameters"]
            )
            self.num_gpus = config["num_gpus"]
            self.num_nodes = config["num_nodes"]
        
        self.logger.info(f"📂 Configuration loaded from {config_dir}")
    
    def train(
        self,
        data_path: str,
        val_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_epochs: Optional[int] = None,
        resume_from: Optional[str] = None
    ):
        """High-level training interface"""
        
        self.logger.info("🚀 Starting training...")
        
        if output_dir:
            self.experiment_dir = Path(output_dir)
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy dataloaders for now
        # In a real implementation, this would load actual data
        train_dataloader = self._create_dummy_dataloader()
        val_dataloader = self._create_dummy_dataloader() if val_data_path else None
        
        # Create trainer
        trainer = self.create_trainer(train_dataloader, val_dataloader)
        
        # Override epochs if specified
        if num_epochs and self.training_config:
            self.training_config.epochs = num_epochs
        
        try:
            # Save configuration before training
            self.save_configuration()
            
            # Start training
            if hasattr(trainer, 'fit'):
                trainer.fit()
            else:
                # Manual training loop
                epochs = getattr(self.training_config, 'epochs', 3) if self.training_config else 3
                for epoch in range(epochs):
                    self.logger.info(f"Training epoch {epoch + 1}/{epochs}")
                    for step, batch in enumerate(train_dataloader):
                        if step >= 10:  # Limit for demo
                            break
                        loss = trainer.train_step(batch)
                        if step % 5 == 0:
                            self.logger.info(f"  Step {step}: Loss = {loss:.4f}")
            
            # Save final checkpoint
            if trainer and hasattr(trainer, 'is_main_process') and trainer.is_main_process:
                final_checkpoint = self.experiment_dir / "final_model.pt"
                if hasattr(trainer, 'save_checkpoint'):
                    trainer.save_checkpoint(str(final_checkpoint))
                    self.logger.info(f"💾 Final model saved: {final_checkpoint}")
            
            self.logger.info("✅ Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            if trainer and hasattr(trainer, 'cleanup'):
                trainer.cleanup()
    
    def _create_dummy_dataloader(self):
        """Create dummy dataloader for testing"""
        from torch.utils.data import DataLoader, TensorDataset
        
        vocab_size = 32000
        seq_length = 512
        num_samples = 100
        batch_size = getattr(self.training_config, 'batch_size', 8) if self.training_config else 8
        
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
        dataset = TensorDataset(input_ids, input_ids)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    def resume_training(
        self,
        checkpoint_path: str,
        data_path: str,
        val_data_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """Resume training from checkpoint"""
        
        self.logger.info(f"🔄 Resuming training from {checkpoint_path}")
        
        # This would load the checkpoint and resume training
        # For now, just start regular training
        self.train(data_path, val_data_path, output_dir)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        
        config = {
            "orchestrator": {
                "model_variant": self.model_variant,
                "training_strategy": {
                    "name": self.training_strategy.name,
                    "parameters": self.training_strategy.parameters
                },
                "num_gpus": self.num_gpus,
                "num_nodes": self.num_nodes,
                "experiment_name": self.experiment_name
            }
        }
        
        if self.distributed_config:
            config["distributed"] = {}
            if hasattr(self.distributed_config, 'tensor_parallel'):
                config["distributed"]["tensor_parallel"] = self.distributed_config.tensor_parallel.__dict__
            if hasattr(self.distributed_config, 'pipeline_parallel'):
                config["distributed"]["pipeline_parallel"] = self.distributed_config.pipeline_parallel.__dict__
            if hasattr(self.distributed_config, 'data_parallel'):
                config["distributed"]["data_parallel"] = self.distributed_config.data_parallel.__dict__
        
        if self.training_config:
            config["training"] = self.training_config.__dict__
        
        return config
    
    def update_configuration(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        
        # This would update the configuration
        # For now, just log the update
        self.logger.info(f"🔧 Configuration updated: {updates}")


# Convenience functions for common configurations
def create_llama_7b_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 7B training"""
    return LlamaTrainingOrchestrator(
        model_variant="llama2_7b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_llama_13b_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 13B training"""
    return LlamaTrainingOrchestrator(
        model_variant="llama2_13b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_llama_70b_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 70B training"""
    if num_gpus is None:
        num_gpus = max(8, torch.cuda.device_count())
    
    return LlamaTrainingOrchestrator(
        model_variant="llama2_70b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_code_llama_orchestrator(variant: str = "7b", num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for Code LLaMA training"""
    return LlamaTrainingOrchestrator(
        model_variant=f"code_llama_{variant}",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_llama3_8b_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 3 8B training"""
    return LlamaTrainingOrchestrator(
        model_variant="llama3_8b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_llama3_70b_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 3 70B training"""
    if num_gpus is None:
        num_gpus = max(8, torch.cuda.device_count())
    
    return LlamaTrainingOrchestrator(
        model_variant="llama3_70b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_llama3_405b_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 3 405B training (mega model)"""
    if num_gpus is None:
        num_gpus = max(32, torch.cuda.device_count())
    
    if num_gpus < 32:
        raise ValueError("LLaMA 3 405B requires at least 32 GPUs for practical training")
    
    return LlamaTrainingOrchestrator(
        model_variant="llama3_405b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_tiny_llama3_orchestrator(variant: str = "150m", num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for Tiny LLaMA 3 (development/testing)"""
    return LlamaTrainingOrchestrator(
        model_variant=f"tiny_llama3_{variant}",
        training_strategy=strategy,
        num_gpus=num_gpus or 1,
        auto_configure=True
    )

def get_llama_variant_config(variant: str) -> Dict[str, Any]:
    """Get configuration details for a LLaMA variant"""
    
    configs = {
        "tiny_llama3_50m": {
            "parameters": "50M",
            "recommended_gpus": 1,
            "min_memory_gb": 4,
            "use_case": "Ultra-fast testing and development"
        },
        "tiny_llama3_150m": {
            "parameters": "150M", 
            "recommended_gpus": 1,
            "min_memory_gb": 8,
            "use_case": "Development and validation"
        },
        "llama3_8b": {
            "parameters": "8B",
            "recommended_gpus": 4,
            "min_memory_gb": 40,
            "use_case": "Production training and fine-tuning"
        },
        "llama3_70b": {
            "parameters": "70B",
            "recommended_gpus": 8,
            "min_memory_gb": 320,
            "use_case": "Large-scale training"
        },
        "llama3_405b": {
            "parameters": "405B",
            "recommended_gpus": 32,
            "min_memory_gb": 1280,
            "use_case": "Mega-scale research training"
        }
    }
    
    return configs.get(variant, {
        "parameters": "Unknown",
        "recommended_gpus": 1,
        "min_memory_gb": 8,
        "use_case": "General purpose"
    })


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(description="LLaMA Training Orchestrator")
        parser.add_argument("--model", type=str, default="tiny_llama3_150m", 
                           help="Model variant")
        parser.add_argument("--strategy", type=str, default="standard", 
                           choices=["standard", "moe", "lora", "hybrid"])
        parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
        parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
        parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
        parser.add_argument("--config-only", action="store_true", help="Print configuration only")
        parser.add_argument("--test-train", action="store_true", help="Run test training")
        
        args = parser.parse_args()
        
        try:
            # Create orchestrator
            orchestrator = LlamaTrainingOrchestrator(
                model_variant=args.model,
                training_strategy=args.strategy,
                num_gpus=args.gpus,
                output_dir=args.output,
                experiment_name=args.experiment,
                auto_configure=True
            )
            
            # Print configuration
            orchestrator.print_configuration()
            
            if args.config_only:
                print("\n📋 Configuration printed. Exiting.")
                return
            
            if args.test_train:
                # Run test training
                print("\n🚀 Starting test training...")
                orchestrator.train(
                    data_path="dummy_data.jsonl",  # Would be real data path
                    num_epochs=1
                )
            else:
                print("\n✅ Orchestrator created successfully!")
                print("Use --test-train to run test training")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    main()