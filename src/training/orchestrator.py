# training_infra/orchestrator.py
"""
High-level orchestrator for LLaMA distributed training with automatic configuration.
"""

import os
import torch
import torch.multiprocessing as mp
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from .config import TrainingConfig
from ..distributed.config import DistributedConfig, ConfigurationFactory, AutoConfigurator
from ..distributed.trainer import create_distributed_trainer
from ..models.llama import (
    create_llama_7b_parallel, create_llama_13b_parallel,
    create_llama_30b_parallel, create_llama_65b_parallel,
    create_llama2_7b_parallel, create_code_llama_7b_parallel
)


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
        model_variant: str = "llama2_7b",
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
            model_variant: Model variant ("llama1_7b", "llama2_7b", "llama2_13b", etc.)
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
        
        # Use AutoConfigurator to get optimal configuration
        config = AutoConfigurator.auto_configure(
            model_size=self.model_variant,
            available_gpus=self.num_gpus,
            memory_per_gpu_gb=self._estimate_gpu_memory()
        )
        
        # Adjust based on training strategy
        if self.training_strategy.name == "moe":
            # MoE models need more memory, use more aggressive parallelism
            if config.data_parallel.data_parallel_size > 1:
                config.data_parallel.data_parallel_size = max(1, config.data_parallel.data_parallel_size // 2)
                config.tensor_parallel.tensor_parallel_size *= 2
            
            # Enable ZeRO for MoE
            config.zero.stage = 2
            config.memory_optimization.use_activation_checkpointing = True
        
        elif self.training_strategy.name == "lora":
            # LoRA is memory efficient, can use larger data parallelism
            config.data_parallel.data_parallel_size = self.num_gpus
            config.tensor_parallel.tensor_parallel_size = 1
            config.pipeline_parallel.pipeline_parallel_size = 1
        
        self.logger.info(f"✅ Parallelism configured: TP={config.tensor_parallel.tensor_parallel_size}, "
                        f"PP={config.pipeline_parallel.pipeline_parallel_size}, "
                        f"DP={config.data_parallel.data_parallel_size}")
        
        return config
    
    def _auto_configure_training(self) -> TrainingConfig:
        """Auto-configure training parameters"""
        
        self.logger.info("⚙️ Auto-configuring training parameters...")
        
        # Base configuration based on model size
        base_configs = {
            "7b": {
                "batch_size": 8,
                "lr": 2e-5,
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
            "30b": {
                "batch_size": 2,
                "lr": 1e-5,
                "gradient_accumulation_steps": 64,
                "max_grad_norm": 0.5,
                "warmup_steps": 1000
            },
            "70b": {
                "batch_size": 1,
                "lr": 5e-6,
                "gradient_accumulation_steps": 128,
                "max_grad_norm": 0.3,
                "warmup_steps": 1000
            }
        }
        
        # Get base config for model size
        size_key = next((k for k in base_configs.keys() if k in self.model_variant), "7b")
        base_config = base_configs[size_key]
        
        # Adjust for training strategy
        if self.training_strategy.name == "moe":
            base_config["batch_size"] = max(1, base_config["batch_size"] // 2)
            base_config["lr"] = base_config["lr"] * 0.7
            base_config["gradient_accumulation_steps"] *= 2
        
        elif self.training_strategy.name == "lora":
            base_config["lr"] = base_config["lr"] * 2
            base_config["gradient_accumulation_steps"] = max(1, base_config["gradient_accumulation_steps"] // 2)
        
        # Create TrainingConfig
        config = TrainingConfig(
            model_name=f"{self.model_variant}_{self.training_strategy.name}",
            epochs=3,
            batch_size=base_config["batch_size"],
            gradient_accumulation_steps=base_config["gradient_accumulation_steps"],
            max_grad_norm=base_config["max_grad_norm"],
            
            optimizer=TrainingConfig.OptimizerConfig(
                name="adamw",
                lr=base_config["lr"],
                weight_decay=0.01,
                betas=(0.9, 0.95)
            ),
            
            scheduler=TrainingConfig.SchedulerConfig(
                name="cosine",
                warmup_steps=base_config["warmup_steps"],
                min_lr=base_config["lr"] * 0.1
            ),
            
            use_amp=True,
            amp_dtype="bfloat16",
            
            logging=TrainingConfig.LoggingConfig(
                log_every=50,
                use_tensorboard=True,
                use_wandb=True,
                wandb_project=f"llama_{self.experiment_name}"
            ),
            
            checkpoint=TrainingConfig.CheckpointConfig(
                save_dir=str(self.experiment_dir / "checkpoints"),
                save_every=1000,
                monitor="val_loss",
                mode="min",
                keep_last=3
            )
        )
        
        self.logger.info(f"✅ Training configured: BS={config.batch_size}, "
                        f"LR={config.optimizer.lr}, GA={config.gradient_accumulation_steps}")
        
        return config
    
    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory per device"""
        if not torch.cuda.is_available():
            return 0.0
        
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    def create_model(self):
        """Create the LLaMA model based on configuration"""
        
        if self.model is not None:
            return self.model
        
        self.logger.info(f"🏗️ Creating {self.model_variant} model...")
        
        # Get tensor parallel size for model creation
        tp_size = 1
        if self.distributed_config:
            tp_size = self.distributed_config.tensor_parallel.tensor_parallel_size
        
        # Model creation functions
        model_creators = {
            "llama1_7b": lambda: create_llama_7b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=self.distributed_config.memory_optimization.use_activation_checkpointing
            ),
            "llama2_7b": lambda: create_llama2_7b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                extended_context=True
            ),
            "llama2_13b": lambda: create_llama_13b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=True
            ),
            "llama2_30b": lambda: create_llama_30b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=True
            ),
            "llama2_70b": lambda: create_llama_65b_parallel(  # Using 65b as proxy for 70b
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=True
            ),
            "code_llama_7b": lambda: create_code_llama_7b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True
            ),
            # LLaMA 3 variants
            "llama3_8b": lambda: create_llama3_8b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=self.distributed_config.memory_optimization.use_activation_checkpointing
            ),
            "llama3_8b_instruct": lambda: create_llama3_8b_instruct_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=True
            ),
            "llama3_70b": lambda: create_llama3_70b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=True
            ),
            "llama3_70b_instruct": lambda: create_llama3_70b_instruct_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=True
            ),
            "llama3_405b": lambda: create_llama3_405b_parallel(
                tensor_parallel_size=tp_size,
                use_flash_attention=True,
                use_checkpointing=True
            ),
            # Tiny LLaMA 3 variants for development and testing
            "tiny_llama3_150m": lambda: create_tiny_llama3_150m(
                tensor_parallel_size=1,  # Single GPU is enough
                use_flash_attention=True,
                use_checkpointing=False  # No need for small model
            ),
            "tiny_llama3_50m": lambda: create_tiny_llama3_50m(
                tensor_parallel_size=1,
                use_flash_attention=True,
                use_checkpointing=False
            )
        }
        
        if self.model_variant not in model_creators:
            raise ValueError(f"Unsupported model variant: {self.model_variant}")
        
        # Create base model
        self.model = model_creators[self.model_variant]()
        
        # Apply training strategy modifications
        if self.training_strategy.name == "moe":
            self.model = self._apply_moe_strategy(self.model)
        elif self.training_strategy.name == "lora":
            self.model = self._apply_lora_strategy(self.model)
        elif self.training_strategy.name == "hybrid":
            self.model = self._apply_hybrid_strategy(self.model)
        
        self.logger.info(f"✅ Model created successfully")
        
        return self.model
    
    def _apply_moe_strategy(self, model):
        """Apply Mixture of Experts strategy"""
        try:
            from .models.moe import MoEConfig
            from .advanced import LlamaMoEModel
            
            # Create MoE configuration
            moe_config = MoEConfig(
                hidden_size=model.config.hidden_size,
                num_experts=self.training_strategy.parameters.get("num_experts", 8),
                num_experts_per_tok=self.training_strategy.parameters.get("experts_per_token", 2),
                intermediate_size=model.config.intermediate_size,
                router_aux_loss_coef=0.01
            )
            
            # Use MoE in every 4th layer by default
            moe_layers = self.training_strategy.parameters.get(
                "moe_layers", 
                list(range(3, model.config.num_hidden_layers, 4))
            )
            
            moe_model = LlamaMoEModel(model.config, moe_config, moe_layers)
            
            self.logger.info(f"✅ MoE strategy applied: {len(moe_layers)} layers, "
                           f"{moe_config.num_experts} experts")
            
            return moe_model
            
        except ImportError:
            self.logger.warning("MoE modules not available, using standard model")
            return model
    
    def _apply_lora_strategy(self, model):
        """Apply LoRA (Low-Rank Adaptation) strategy"""
        try:
            from .advanced import LlamaWithLoRA
            
            lora_rank = self.training_strategy.parameters.get("lora_rank", 16)
            lora_alpha = self.training_strategy.parameters.get("lora_alpha", 32)
            
            lora_model = LlamaWithLoRA(model, lora_rank=lora_rank, lora_alpha=lora_alpha)
            
            self.logger.info(f"✅ LoRA strategy applied: rank={lora_rank}, alpha={lora_alpha}")
            
            return lora_model
            
        except ImportError:
            self.logger.warning("LoRA modules not available, using standard model")
            return model
    
    def _apply_hybrid_strategy(self, model):
        """Apply hybrid strategy (combination of techniques)"""
        # This could combine multiple strategies
        # For now, just return the standard model
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
            raise ValueError("Configurations not initialized. Set auto_configure=True or configure manually.")
        
        self.logger.info("🎯 Creating distributed trainer...")
        
        # Determine trainer type based on configuration
        trainer_type = "standard"
        if self.distributed_config.uses_pipeline_parallelism:
            trainer_type = "pipeline"
        
        # Use adaptive trainer for large models or complex strategies
        if ("30b" in self.model_variant or "70b" in self.model_variant or 
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
        
        return self.trainer
    
    def estimate_memory_requirements(self, batch_size: int = None, sequence_length: int = 2048):
        """Estimate memory requirements for the current configuration"""
        
        if batch_size is None:
            batch_size = self.training_config.batch_size if self.training_config else 8
        
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
        if self.distributed_config:
            print(f"\n🌐 Distributed Configuration:")
            self.distributed_config.print_configuration()
        
        # Print training configuration
        if self.training_config:
            print(f"\n⚙️ Training Configuration:")
            print(f"  Batch Size: {self.training_config.batch_size}")
            print(f"  Learning Rate: {self.training_config.optimizer.lr}")
            print(f"  Gradient Accumulation: {self.training_config.gradient_accumulation_steps}")
            print(f"  Mixed Precision: {self.training_config.use_amp}")
            print(f"  Epochs: {self.training_config.epochs}")
        
        # Print memory estimation
        try:
            memory_est = self.estimate_memory_requirements()
            print(f"\n💾 Memory Estimation:")
            for key, value in memory_est.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
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
        
        # Save distributed configuration
        if self.distributed_config:
            dist_config_dict = {
                "tensor_parallel": self.distributed_config.tensor_parallel.__dict__,
                "pipeline_parallel": self.distributed_config.pipeline_parallel.__dict__,
                "data_parallel": self.distributed_config.data_parallel.__dict__,
                "zero": self.distributed_config.zero.__dict__,
                "mixed_precision": self.distributed_config.mixed_precision.__dict__,
                "communication": self.distributed_config.communication.__dict__,
                "memory_optimization": self.distributed_config.memory_optimization.__dict__
            }
            
            with open(config_dir / "distributed.json", "w") as f:
                json.dump(dist_config_dict, f, indent=2)
        
        # Save training configuration
        if self.training_config:
            with open(config_dir / "training.json", "w") as f:
                json.dump(self.training_config.__dict__, f, indent=2, default=str)
        
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
    
    def launch_training(
        self,
        train_dataloader,
        val_dataloader=None,
        master_addr: str = "localhost",
        master_port: str = "12355"
    ):
        """Launch distributed training"""
        
        self.logger.info("🚀 Launching distributed training...")
        
        # Save configuration before training
        self.save_configuration()
        
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(self.num_gpus * self.num_nodes)
        
        def training_function(local_rank, num_gpus, num_nodes, node_rank):
            """Main training function for each process"""
            
            # Set local rank environment
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(node_rank * num_gpus + local_rank)
            
            # Create trainer
            trainer = self.create_trainer(train_dataloader, val_dataloader)
            
            try:
                # Start training
                trainer.fit()
                
                # Save final checkpoint
                if trainer.is_main_process:
                    final_checkpoint = self.experiment_dir / "final_model.pt"
                    trainer.save_checkpoint(str(final_checkpoint))
                    self.logger.info(f"💾 Final model saved: {final_checkpoint}")
                
            except Exception as e:
                self.logger.error(f"❌ Training failed: {e}")
                raise
            finally:
                trainer.cleanup()
        
        # Launch training processes
        if self.num_gpus > 1:
            mp.spawn(
                training_function,
                args=(self.num_gpus, self.num_nodes, 0),  # Single node for now
                nprocs=self.num_gpus,
                join=True
            )
        else:
            # Single GPU training
            os.environ['LOCAL_RANK'] = '0'
            os.environ['RANK'] = '0'
            training_function(0, 1, 1, 0)
        
        self.logger.info("✅ Training completed!")
    
    def benchmark_configuration(self):
        """Benchmark the current configuration"""
        
        self.logger.info("⏱️ Benchmarking configuration...")
        
        try:
            from .benchmarking import PerformanceBenchmark
            
            benchmark = PerformanceBenchmark(
                model_variant=self.model_variant,
                distributed_config=self.distributed_config
            )
            
            results = benchmark.run_quick_benchmark()
            
            # Save benchmark results
            benchmark_file = self.experiment_dir / "benchmark_results.json"
            with open(benchmark_file, "w") as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"📊 Benchmark completed, results saved to {benchmark_file}")
            
            return results
            
        except ImportError:
            self.logger.warning("Benchmarking module not available")
            return None


# Convenience functions for common configurations
def create_llama_7b_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 7B training"""
    return LlamaTrainingOrchestrator(
        model_variant="llama2_70b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_code_llama_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for Code LLaMA training"""
    return LlamaTrainingOrchestrator(
        model_variant="code_llama_7b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

# LLaMA 3 convenience functions
def create_llama3_8b_orchestrator(num_gpus: int = None, strategy: str = "standard", instruct: bool = False) -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 3 8B training"""
    model_variant = "llama3_8b_instruct" if instruct else "llama3_8b"
    
    return LlamaTrainingOrchestrator(
        model_variant=model_variant,
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_llama3_70b_orchestrator(num_gpus: int = None, strategy: str = "standard", instruct: bool = False) -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 3 70B training"""
    if num_gpus is None:
        num_gpus = max(8, torch.cuda.device_count())
    
    model_variant = "llama3_70b_instruct" if instruct else "llama3_70b"
    
    return LlamaTrainingOrchestrator(
        model_variant=model_variant,
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

def create_code_llama_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for Code LLaMA training"""
    return LlamaTrainingOrchestrator(
        model_variant="code_llama_7b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

# LLaMA 3 convenience functions
def create_llama3_8b_orchestrator(num_gpus: int = None, strategy: str = "standard", instruct: bool = False) -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 3 8B training"""
    model_variant = "llama3_8b_instruct" if instruct else "llama3_8b"
    
    return LlamaTrainingOrchestrator(
        model_variant=model_variant,
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )

def create_llama3_70b_orchestrator(num_gpus: int = None, strategy: str = "standard", instruct: bool = False) -> LlamaTrainingOrchestrator:
    """Create orchestrator for LLaMA 3 70B training"""
    if num_gpus is None:
        num_gpus = max(8, torch.cuda.device_count())
    
    model_variant = "llama3_70b_instruct" if instruct else "llama3_70b"
    
    return LlamaTrainingOrchestrator(
        model_variant=model_variant,
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

def create_code_llama_orchestrator(num_gpus: int = None, strategy: str = "standard") -> LlamaTrainingOrchestrator:
    """Create orchestrator for Code LLaMA training"""
    return LlamaTrainingOrchestrator(
        model_variant="code_llama_7b",
        training_strategy=strategy,
        num_gpus=num_gpus,
        auto_configure=True
    )


# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    
    def dummy_dataloader():
        """Create dummy dataloader for testing"""
        from torch.utils.data import DataLoader, TensorDataset
        
        vocab_size = 32000
        seq_length = 512
        num_samples = 1000
        
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
        dataset = TensorDataset(input_ids, input_ids)
        
        return DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    
    parser = argparse.ArgumentParser(description="LLaMA Training Orchestrator")
    parser.add_argument("--model", type=str, default="llama2_7b", 
                       choices=["llama1_7b", "llama2_7b", "llama2_13b", "llama2_70b", "code_llama_7b"])
    parser.add_argument("--strategy", type=str, default="standard", 
                       choices=["standard", "moe", "lora", "hybrid"])
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    parser.add_argument("--config-only", action="store_true", help="Print configuration only")
    
    args = parser.parse_args()
    
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
        exit(0)
    
    if args.benchmark:
        # Run benchmark
        orchestrator.benchmark_configuration()
    else:
        # Run training with dummy data
        print("\n🚀 Starting training with dummy data...")
        train_dataloader = dummy_dataloader()
        val_dataloader = dummy_dataloader()
        
        orchestrator.launch_training(train_dataloader, val_dataloader)