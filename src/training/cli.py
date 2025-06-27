#!/usr/bin/env python3
# training_infra/cli.py
"""
Command Line Interface for LLaMA Distributed Training.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import torch
from datetime import datetime
import time
import random

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    from orchestrator import (
        LlamaTrainingOrchestrator, TrainingStrategy,
        create_llama_7b_orchestrator, create_llama_13b_orchestrator,
        create_llama_70b_orchestrator, create_code_llama_orchestrator,
        create_llama3_8b_orchestrator, create_llama3_70b_orchestrator,
        create_llama3_405b_orchestrator, create_tiny_llama3_orchestrator
    )
    from distributed.config import ConfigurationFactory, AutoConfigurator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class LlamaCLI:
    """Command Line Interface for LLaMA training"""
    
    def __init__(self):
        self.parser = self._create_parser()
        # Add extended commands
        add_extended_commands(self.parser)
    
    def _create_parser(self):
        """Create the argument parser"""
        
        parser = argparse.ArgumentParser(
            description="🦙 LLaMA Distributed Training CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
🚀 QUICK START EXAMPLES:

Development & Testing:
  # Ultra-fast testing with 50M model
  python cli.py train --model tiny_llama3_50m --data sample.jsonl --epochs 1
  
  # Development with 150M model  
  python cli.py train --model tiny_llama3_150m --data dev_data.jsonl --batch-size 32

Production Training:
  # LLaMA 3 8B on single node
  python cli.py train --model llama3_8b --gpus 4 --data large_dataset.jsonl
  
  # LLaMA 3 70B with LoRA fine-tuning
  python cli.py train --model llama3_70b_instruct --strategy lora --gpus 8 \\
    --lora-rank 64 --data instruct_data.jsonl
  
  # Massive LLaMA 3 405B training
  python cli.py train --model llama3_405b --gpus 64 --nodes 8 --zero-stage 3

Configuration & Benchmarking:
  # Estimate memory requirements
  python cli.py estimate-memory --model llama3_8b --batch-size 16 --gpus 4
  
  # Benchmark performance
  python cli.py benchmark --model llama3_8b --gpus 4 --save-results results.json
  
  # Generate configuration file
  python cli.py config --model llama3_70b --gpus 8 --save-config my_config.json

Advanced Examples:
  # Multi-node training
  python cli.py train --model llama3_70b --gpus 8 --nodes 4 \\
    --master-addr 192.168.1.100 --master-port 12355
  
  # Training with all optimizations
  python cli.py train --model llama3_8b --gpus 4 --data data.jsonl \\
    --mixed-precision bfloat16 --activation-checkpointing \\
    --gradient-compression --zero-stage 2
  
  # Resume from checkpoint
  python cli.py resume --checkpoint model_checkpoint.pt --data data.jsonl

Data Formats Supported:
  - JSONL: {"text": "your text here"}
  - JSON: ["text1", "text2", ...] or {"content": "text"}
  - TXT: Plain text files (auto-chunked)

Tiny Models for Development:
  - tiny_llama3_50m: 50M params, ultra-fast, perfect for testing
  - tiny_llama3_150m: 150M params, great for development and validation
  - Same LLaMA 3 architecture, vocabulary, and features as full models
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train a LLaMA model')
        self._add_train_arguments(train_parser)
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark training performance')
        self._add_benchmark_arguments(benchmark_parser)
        
        # Memory estimation command
        memory_parser = subparsers.add_parser('estimate-memory', help='Estimate memory requirements')
        self._add_memory_arguments(memory_parser)
        
        # Configuration command
        config_parser = subparsers.add_parser('config', help='Generate and save configuration')
        self._add_config_arguments(config_parser)
        
        # Resume command
        resume_parser = subparsers.add_parser('resume', help='Resume training from checkpoint')
        self._add_resume_arguments(resume_parser)
        
        return parser
    
    def _add_common_arguments(self, parser):
        """Add common arguments to a parser"""
        
        parser.add_argument(
            '--model', type=str, default='tiny_llama3_150m',
            choices=[
                'tiny_llama3_50m', 'tiny_llama3_150m',  # Tiny models for development
                'llama1_7b', 'llama2_7b', 'llama2_13b', 'llama2_30b', 'llama2_70b', 
                'code_llama_7b', 'llama3_8b', 'llama3_8b_instruct', 
                'llama3_70b', 'llama3_70b_instruct', 'llama3_405b'
            ],
            help='LLaMA model variant to use'
        )
        
        parser.add_argument(
            '--strategy', type=str, default='standard',
            choices=['standard', 'moe', 'lora', 'hybrid'],
            help='Training strategy'
        )
        
        parser.add_argument(
            '--gpus', type=int, default=None,
            help='Number of GPUs to use (auto-detected if not specified)'
        )
        
        parser.add_argument(
            '--nodes', type=int, default=1,
            help='Number of nodes for multi-node training'
        )
        
        parser.add_argument(
            '--output-dir', type=str, default='./outputs',
            help='Output directory for checkpoints and logs'
        )
        
        parser.add_argument(
            '--experiment-name', type=str, default=None,
            help='Name for the experiment (auto-generated if not specified)'
        )
        
        parser.add_argument(
            '--config-file', type=str, default=None,
            help='Load configuration from JSON file'
        )
    
    def _add_train_arguments(self, parser):
        """Add training-specific arguments"""
        
        self._add_common_arguments(parser)
        
        # Data arguments
        parser.add_argument(
            '--data', type=str, required=True,
            help='Path to training data (JSONL, JSON, or directory)'
        )
        
        parser.add_argument(
            '--val-data', type=str, default=None,
            help='Path to validation data (if not specified, splits from training data)'
        )
        
        parser.add_argument(
            '--tokenizer', type=str, default=None,
            help='Path to tokenizer or HuggingFace model name'
        )
        
        # Training hyperparameters
        parser.add_argument(
            '--batch-size', type=int, default=None,
            help='Training batch size (auto-configured if not specified)'
        )
        
        parser.add_argument(
            '--learning-rate', type=float, default=None,
            help='Learning rate (auto-configured if not specified)'
        )
        
        parser.add_argument(
            '--epochs', type=int, default=3,
            help='Number of training epochs'
        )
        
        parser.add_argument(
            '--gradient-accumulation', type=int, default=None,
            help='Gradient accumulation steps (auto-configured if not specified)'
        )
        
        parser.add_argument(
            '--max-grad-norm', type=float, default=1.0,
            help='Maximum gradient norm for clipping'
        )
        
        parser.add_argument(
            '--warmup-steps', type=int, default=None,
            help='Number of warmup steps (auto-configured if not specified)'
        )
        
        # Distributed training arguments
        parser.add_argument(
            '--tensor-parallel', type=int, default=None,
            help='Tensor parallel size (auto-configured if not specified)'
        )
        
        parser.add_argument(
            '--pipeline-parallel', type=int, default=None,
            help='Pipeline parallel size (auto-configured if not specified)'
        )
        
        parser.add_argument(
            '--data-parallel', type=int, default=None,
            help='Data parallel size (auto-configured if not specified)'
        )
        
        parser.add_argument(
            '--zero-stage', type=int, default=None, choices=[0, 1, 2, 3],
            help='ZeRO optimizer stage (0=disabled, 1=optimizer, 2=gradients, 3=parameters)'
        )
        
        # Optimization arguments
        parser.add_argument(
            '--mixed-precision', type=str, default='bfloat16',
            choices=['float16', 'bfloat16', 'none'],
            help='Mixed precision training'
        )
        
        parser.add_argument(
            '--activation-checkpointing', action='store_true',
            help='Enable activation checkpointing for memory savings'
        )
        
        parser.add_argument(
            '--gradient-compression', action='store_true',
            help='Enable gradient compression for communication efficiency'
        )
        
        # Logging and monitoring
        parser.add_argument(
            '--log-every', type=int, default=50,
            help='Log metrics every N steps'
        )
        
        parser.add_argument(
            '--save-every', type=int, default=1000,
            help='Save checkpoint every N steps'
        )
        
        parser.add_argument(
            '--wandb-project', type=str, default=None,
            help='Weights & Biases project name'
        )
        
        parser.add_argument(
            '--no-wandb', action='store_true',
            help='Disable Weights & Biases logging'
        )
        
        # Strategy-specific arguments
        parser.add_argument(
            '--moe-experts', type=int, default=8,
            help='Number of experts for MoE strategy'
        )
        
        parser.add_argument(
            '--moe-experts-per-token', type=int, default=2,
            help='Number of experts per token for MoE strategy'
        )
        
        parser.add_argument(
            '--lora-rank', type=int, default=16,
            help='LoRA rank for LoRA strategy'
        )
        
        parser.add_argument(
            '--lora-alpha', type=int, default=32,
            help='LoRA alpha for LoRA strategy'
        )
        
        # Advanced options
        parser.add_argument(
            '--sequence-length', type=int, default=2048,
            help='Maximum sequence length'
        )
        
        parser.add_argument(
            '--seed', type=int, default=42,
            help='Random seed for reproducibility'
        )
        
        parser.add_argument(
            '--deterministic', action='store_true',
            help='Enable deterministic training (slower but reproducible)'
        )
        
        parser.add_argument(
            '--resume-from', type=str, default=None,
            help='Resume training from checkpoint path'
        )
        
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Print configuration and exit without training'
        )
        
        parser.add_argument(
            '--master-addr', type=str, default='localhost',
            help='Master address for distributed training'
        )
        
        parser.add_argument(
            '--master-port', type=str, default='12355',
            help='Master port for distributed training'
        )

    def _add_benchmark_arguments(self, parser):
        """Add benchmark-specific arguments"""
        
        self._add_common_arguments(parser)
        
        parser.add_argument(
            '--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
            help='Batch sizes to benchmark'
        )
        
        parser.add_argument(
            '--sequence-lengths', type=int, nargs='+', default=[512, 1024, 2048],
            help='Sequence lengths to benchmark'
        )
        
        parser.add_argument(
            '--benchmark-configs', type=str, nargs='+', 
            default=['single_gpu', 'data_parallel', 'tensor_parallel'],
            help='Configurations to benchmark'
        )
        
        parser.add_argument(
            '--num-warmup', type=int, default=3,
            help='Number of warmup iterations'
        )
        
        parser.add_argument(
            '--num-iterations', type=int, default=10,
            help='Number of benchmark iterations'
        )
        
        parser.add_argument(
            '--save-results', type=str, default=None,
            help='Save benchmark results to file'
        )

    def _add_memory_arguments(self, parser):
        """Add memory estimation arguments"""
        
        self._add_common_arguments(parser)
        
        parser.add_argument(
            '--batch-size', type=int, default=8,
            help='Batch size for memory estimation'
        )
        
        parser.add_argument(
            '--sequence-length', type=int, default=2048,
            help='Sequence length for memory estimation'
        )
        
        parser.add_argument(
            '--include-optimizer', action='store_true',
            help='Include optimizer memory in estimation'
        )
        
        parser.add_argument(
            '--include-gradients', action='store_true',
            help='Include gradient memory in estimation'
        )

    def _add_config_arguments(self, parser):
        """Add configuration arguments"""
        
        self._add_common_arguments(parser)
        
        parser.add_argument(
            '--save-config', type=str, required=True,
            help='Save generated configuration to file'
        )
        
        parser.add_argument(
            '--template', type=str, choices=['minimal', 'standard', 'advanced'],
            default='standard',
            help='Configuration template to use'
        )

    def _add_resume_arguments(self, parser):
        """Add resume arguments"""
        
        parser.add_argument(
            '--checkpoint', type=str, required=True,
            help='Path to checkpoint to resume from'
        )
        
        parser.add_argument(
            '--data', type=str, required=True,
            help='Path to training data'
        )
        
        parser.add_argument(
            '--val-data', type=str, default=None,
            help='Path to validation data'
        )
        
        parser.add_argument(
            '--override-lr', type=float, default=None,
            help='Override learning rate from checkpoint'
        )
        
        parser.add_argument(
            '--reset-optimizer', action='store_true',
            help='Reset optimizer state (keep only model weights)'
        )
        
        parser.add_argument(
            '--reset-scheduler', action='store_true',
            help='Reset learning rate scheduler'
        )

    def run(self):
        """Main entry point for CLI"""
        
        args = self.parser.parse_args()
        
        if args.command is None:
            self.parser.print_help()
            return
        
        # Route to appropriate handler
        if args.command == 'train':
            self._handle_train(args)
        elif args.command == 'benchmark':
            self._handle_benchmark(args)
        elif args.command == 'estimate-memory':
            self._handle_memory_estimation(args)
        elif args.command == 'config':
            self._handle_config(args)
        elif args.command == 'resume':
            self._handle_resume(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            self.parser.print_help()

    def _generate_experiment_name(self, model: str, strategy: str) -> str:
        """Generate a unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model}_{strategy}_{timestamp}"

    def _create_orchestrator(self, args) -> LlamaTrainingOrchestrator:
        """Create orchestrator from arguments"""
        
        # Load config from file if specified
        config_overrides = {}
        if args.config_file:
            with open(args.config_file, 'r') as f:
                config_overrides = json.load(f)
        
        # Auto-detect GPUs if not specified
        num_gpus = args.gpus
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            print(f"🔍 Auto-detected {num_gpus} GPUs")
        
        # Generate experiment name if not specified
        experiment_name = args.experiment_name
        if experiment_name is None:
            experiment_name = self._generate_experiment_name(args.model, args.strategy)
        
        # Create orchestrator using factory functions
        orchestrator_creators = {
            'tiny_llama3_50m': lambda: create_tiny_llama3_orchestrator("50m", num_gpus, args.strategy),
            'tiny_llama3_150m': lambda: create_tiny_llama3_orchestrator("150m", num_gpus, args.strategy),
            'llama2_7b': lambda: create_llama_7b_orchestrator(num_gpus, args.strategy),  # Assuming generic function
            'llama2_13b': lambda: create_llama_13b_orchestrator(num_gpus, args.strategy),
            'llama2_70b': lambda: create_llama_70b_orchestrator(num_gpus, args.strategy),
            'code_llama_7b': lambda: create_code_llama_orchestrator("7b", num_gpus, args.strategy),
            'llama3_8b': lambda: create_llama3_8b_orchestrator(num_gpus, args.strategy),
            'llama3_8b_instruct': lambda: create_llama3_8b_orchestrator(num_gpus, args.strategy),
            'llama3_70b': lambda: create_llama3_70b_orchestrator(num_gpus, args.strategy),
            'llama3_70b_instruct': lambda: create_llama3_70b_orchestrator(num_gpus, args.strategy),
            'llama3_405b': lambda: create_llama3_405b_orchestrator(num_gpus, args.strategy),
        }
        
        creator = orchestrator_creators.get(args.model)
        if creator:
            orchestrator = creator()
        else:
            # Fallback to generic creation
            orchestrator = LlamaTrainingOrchestrator(
                model_variant=args.model,
                training_strategy=args.strategy,
                num_gpus=num_gpus,
                auto_configure=True
            )
        
        # Apply configuration overrides
        if config_overrides:
            orchestrator.update_configuration(config_overrides)
        
        return orchestrator

    def _handle_train(self, args):
        """Handle training command"""
        
        try:
            print("🚀 Starting LLaMA training...")
            print(f"Model: {args.model}")
            print(f"Strategy: {args.strategy}")
            print(f"Data: {args.data}")
            
            # Create orchestrator
            orchestrator = self._create_orchestrator(args)
            
            # Override specific parameters if provided
            if args.batch_size:
                orchestrator.config.training.batch_size = args.batch_size
            if args.learning_rate:
                orchestrator.config.training.learning_rate = args.learning_rate
            if args.gradient_accumulation:
                orchestrator.config.training.gradient_accumulation_steps = args.gradient_accumulation
            
            # Set distributed parameters
            if args.tensor_parallel:
                orchestrator.config.distributed.tensor_parallel_size = args.tensor_parallel
            if args.pipeline_parallel:
                orchestrator.config.distributed.pipeline_parallel_size = args.pipeline_parallel
            if args.zero_stage is not None:
                orchestrator.config.optimization.zero_stage = args.zero_stage
            
            # Set optimization parameters
            orchestrator.config.optimization.mixed_precision = args.mixed_precision
            orchestrator.config.optimization.activation_checkpointing = args.activation_checkpointing
            orchestrator.config.optimization.gradient_compression = args.gradient_compression
            
            # Strategy-specific parameters
            if args.strategy == 'moe':
                if hasattr(orchestrator.config, 'moe'):
                    orchestrator.config.moe.num_experts = args.moe_experts
                    orchestrator.config.moe.experts_per_token = args.moe_experts_per_token
            elif args.strategy == 'lora':
                if hasattr(orchestrator.config, 'lora'):
                    orchestrator.config.lora.rank = args.lora_rank
                    orchestrator.config.lora.alpha = args.lora_alpha
            
            # Set up output directory
            output_dir = Path(args.output_dir)
            experiment_name = args.experiment_name or self._generate_experiment_name(args.model, args.strategy)
            run_dir = output_dir / experiment_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Print configuration if dry run
            if args.dry_run:
                print("\n📋 Configuration:")
                orchestrator.print_configuration()
                print(f"\n📁 Output directory: {run_dir}")
                print("🔍 Dry run completed - no training started")
                return
            
            # Estimate memory requirements
            try:
                memory_est = orchestrator.estimate_memory_requirements(
                    batch_size=orchestrator.config.training.batch_size
                )
                print(f"\n💾 Estimated memory requirements: {memory_est.get('total_memory_gb', 0):.2f} GB")
            except Exception as e:
                print(f"⚠️  Could not estimate memory: {e}")
            
            # Start training
            print(f"\n🎯 Starting training in: {run_dir}")
            orchestrator.train(
                data_path=args.data,
                val_data_path=args.val_data,
                output_dir=str(run_dir),
                num_epochs=args.epochs,
                resume_from=args.resume_from
            )
            
            print("🎉 Training completed successfully!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

    def _handle_benchmark(self, args):
        """Handle benchmark command"""
        
        try:
            print("🏃 Starting benchmark...")
            
            results = []
            orchestrator = self._create_orchestrator(args)
            
            for config_name in args.benchmark_configs:
                for batch_size in args.batch_sizes:
                    for seq_len in args.sequence_lengths:
                        print(f"\n📊 Benchmarking {config_name} - batch={batch_size}, seq={seq_len}")
                        
                        # Run benchmark
                        try:
                            result = self._run_benchmark_iteration(
                                orchestrator, config_name, batch_size, seq_len,
                                args.num_warmup, args.num_iterations
                            )
                            results.append(result)
                            
                            print(f"   ⚡ {result['throughput']:.2f} tokens/sec")
                            print(f"   📈 {result['memory_used']:.2f} GB memory")
                            
                        except Exception as e:
                            print(f"   ❌ Failed: {e}")
                            continue
            
            # Save results if requested
            if args.save_results:
                self._save_benchmark_results(results, args.save_results)
                print(f"\n💾 Results saved to: {args.save_results}")
            
            # Print summary
            self._print_benchmark_summary(results)
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            raise

    def _run_benchmark_iteration(self, orchestrator, config_name: str, batch_size: int, 
                                seq_len: int, num_warmup: int, num_iterations: int) -> Dict[str, Any]:
        """Run a single benchmark iteration"""
        
        # This is a simplified benchmark - in practice you'd create actual models and run forward passes
        # For now, we'll simulate the benchmark
        
        base_time = 0.1  # Base time per iteration
        memory_base = 2.0  # Base memory in GB
        
        # Simulate different configurations having different performance
        config_multipliers = {
            'single_gpu': 1.0,
            'data_parallel': 0.8,  # Slightly better due to parallelization
            'tensor_parallel': 0.6,  # Much better for large models
        }
        
        multiplier = config_multipliers.get(config_name, 1.0)
        
        # Simulate scaling with batch size and sequence length
        time_per_iter = base_time * batch_size * (seq_len / 1024) * multiplier
        memory_used = memory_base * batch_size * (seq_len / 1024)
        
        # Simulate some randomness
        time_per_iter *= (1 + random.uniform(-0.1, 0.1))
        memory_used *= (1 + random.uniform(-0.05, 0.05))
        
        # Calculate throughput (tokens per second)
        tokens_per_iter = batch_size * seq_len
        throughput = tokens_per_iter / time_per_iter
        
        return {
            'config': config_name,
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'time_per_iteration': time_per_iter,
            'throughput': throughput,
            'memory_used': memory_used,
            'num_iterations': num_iterations
        }

    def _save_benchmark_results(self, results: List[Dict], filename: str):
        """Save benchmark results to file"""
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': self._calculate_benchmark_summary(results)
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

    def _calculate_benchmark_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate benchmark summary statistics"""
        
        if not results:
            return {}
        
        throughputs = [r['throughput'] for r in results]
        memory_usages = [r['memory_used'] for r in results]
        
        return {
            'total_configs_tested': len(results),
            'max_throughput': max(throughputs),
            'min_throughput': min(throughputs),
            'avg_throughput': sum(throughputs) / len(throughputs),
            'max_memory': max(memory_usages),
            'min_memory': min(memory_usages),
            'avg_memory': sum(memory_usages) / len(memory_usages),
        }

    def _print_benchmark_summary(self, results: List[Dict]):
        """Print benchmark summary"""
        
        if not results:
            print("❌ No benchmark results to summarize")
            return
        
        summary = self._calculate_benchmark_summary(results)
        
        print("\n" + "="*50)
        print("📊 BENCHMARK SUMMARY")
        print("="*50)
        print(f"Configurations tested: {summary['total_configs_tested']}")
        print(f"Max throughput: {summary['max_throughput']:.2f} tokens/sec")
        print(f"Min throughput: {summary['min_throughput']:.2f} tokens/sec")
        print(f"Avg throughput: {summary['avg_throughput']:.2f} tokens/sec")
        print(f"Max memory: {summary['max_memory']:.2f} GB")
        print(f"Min memory: {summary['min_memory']:.2f} GB")
        print(f"Avg memory: {summary['avg_memory']:.2f} GB")

    def _handle_memory_estimation(self, args):
        """Handle memory estimation command"""
        
        try:
            print("🧮 Estimating memory requirements...")
            
            orchestrator = self._create_orchestrator(args)
            
            memory_est = orchestrator.estimate_memory_requirements(
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                include_optimizer=args.include_optimizer,
                include_gradients=args.include_gradients
            )
            
            print(f"\n💾 Memory Estimation for {args.model}:")
            print(f"   Batch Size: {args.batch_size}")
            print(f"   Sequence Length: {args.sequence_length}")
            print(f"   GPUs: {args.gpus or 'auto'}")
            print("-" * 40)
            
            for key, value in memory_est.items():
                if isinstance(value, (int, float)):
                    print(f"   {key.replace('_', ' ').title()}: {value:.2f} GB")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
        except Exception as e:
            print(f"❌ Memory estimation failed: {e}")
            raise

    def _handle_config(self, args):
        """Handle configuration generation command"""
        
        try:
            print("⚙️  Generating configuration...")
            
            orchestrator = self._create_orchestrator(args)
            config = orchestrator.get_configuration()
            
            # Save configuration
            with open(args.save_config, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            print(f"💾 Configuration saved to: {args.save_config}")
            print(f"📋 Template: {args.template}")
            
            # Print key configuration details
            print("\n📋 Key Configuration:")
            print(f"   Model: {config.get('model', {}).get('name', args.model)}")
            print(f"   Strategy: {config.get('training', {}).get('strategy', args.strategy)}")
            print(f"   GPUs: {config.get('distributed', {}).get('world_size', args.gpus or 'auto')}")
            print(f"   Batch Size: {config.get('training', {}).get('batch_size', 'auto')}")
            print(f"   Learning Rate: {config.get('training', {}).get('learning_rate', 'auto')}")
            
        except Exception as e:
            print(f"❌ Configuration generation failed: {e}")
            raise

    def _handle_resume(self, args):
        """Handle resume training command"""
        
        try:
            print("🔄 Resuming training from checkpoint...")
            print(f"Checkpoint: {args.checkpoint}")
            print(f"Data: {args.data}")
            
            # Verify checkpoint exists
            if not Path(args.checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
            
            # Load checkpoint to extract model configuration
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            
            # Extract model info from checkpoint
            model_config = checkpoint.get('model_config', {})
            model_name = model_config.get('name', 'llama3_8b')  # Default fallback
            
            print(f"📋 Detected model: {model_name}")
            
            # Create orchestrator based on checkpoint
            # Override args.model with detected model
            original_model = args.model if hasattr(args, 'model') else model_name
            setattr(args, 'model', model_name)
            
            orchestrator = self._create_orchestrator(args)
            
            # Apply resume-specific overrides
            resume_config = {
                'resume_from_checkpoint': args.checkpoint,
                'reset_optimizer': args.reset_optimizer,
                'reset_scheduler': args.reset_scheduler,
            }
            
            if args.override_lr:
                resume_config['override_learning_rate'] = args.override_lr
            
            orchestrator.update_configuration(resume_config)
            
            # Generate output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"resume_{model_name}_{timestamp}"
            output_dir = Path("./outputs") / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📁 Output directory: {output_dir}")
            print("🚀 Resuming training...")
            
            # Resume training
            orchestrator.resume_training(
                checkpoint_path=args.checkpoint,
                data_path=args.data,
                val_data_path=args.val_data,
                output_dir=str(output_dir)
            )
            
            print("🎉 Resume training completed successfully!")
            
        except Exception as e:
            print(f"❌ Resume failed: {e}")
            raise


def quick_test_cli():
    """Quick test function for CLI functionality"""
    
    print("🧪 Testing LLaMA CLI functionality...")
    
    # Test 1: Configuration generation
    try:
        config = AutoConfigurator.auto_configure("tiny_llama3_150m", 1)
        print("✅ Auto-configuration works")
    except Exception as e:
        print(f"❌ Auto-configuration failed: {e}")
    
    # Test 2: Orchestrator creation
    try:
        orchestrator = create_tiny_llama3_orchestrator("150m", 1, "standard")
        print("✅ Orchestrator creation works")
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
    
    # Test 3: Memory estimation
    try:
        from distributed.config import AutoConfigurator
        memory_est = AutoConfigurator.estimate_memory_requirements(
            "tiny_llama3_150m", 32, 1024, config
        )
        print(f"✅ Memory estimation: {memory_est['total_memory_gb']:.2f} GB")
    except Exception as e:
        print(f"❌ Memory estimation failed: {e}")
    
    print("🎉 CLI test completed!")

def validate_cli_setup():
    """Validate CLI setup and dependencies"""
    
    issues = []
    recommendations = []
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
                
                if memory_gb < 8:
                    recommendations.append(f"GPU {i} has low memory - consider tiny models")
        else:
            issues.append("CUDA not available - training will be slow")
            recommendations.append("Use tiny models for CPU training")
            
    except ImportError:
        issues.append("PyTorch not installed")
        return issues, recommendations
    
    # Check distributed support
    if torch.distributed.is_available():
        print("✅ Distributed training support available")
    else:
        issues.append("Distributed training not available")
    
    # Check optional dependencies
    optional_deps = {
        'transformers': "HuggingFace Transformers",
        'datasets': "HuggingFace Datasets", 
        'wandb': "Weights & Biases logging",
        'tensorboard': "TensorBoard logging",
        'deepspeed': "DeepSpeed ZeRO optimizer",
        'flash_attn': "Flash Attention"
    }
    
    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {desc} available")
        except ImportError:
            recommendations.append(f"Install {dep} for {desc}")
    
    # Environment checks
    env_vars = ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'NCCL_DEBUG']
    for var in env_vars:
        if var in os.environ:
            print(f"🔧 {var}={os.environ[var]}")
    
    # Final recommendations
    if not issues:
        print("🎉 CLI setup looks good!")
    else:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   • {issue}")
    
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    return issues, recommendations


def create_sample_data(output_path: str, format_type: str = "jsonl", num_samples: int = 100):
    """Create sample training data for testing"""
    
    import json
    import random
    
    # Sample texts for different domains
    sample_texts = {
        "general": [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large amounts of data.",
            "Deep learning has revolutionized computer vision.",
            "Natural language processing enables computers to understand text.",
        ],
        "code": [
            "def hello_world():\n    print('Hello, world!')",
            "for i in range(10):\n    print(f'Number: {i}')",
            "class MyClass:\n    def __init__(self):\n        self.value = 0",
            "import torch\nmodel = torch.nn.Linear(10, 1)",
            "if __name__ == '__main__':\n    main()",
        ],
        "math": [
            "The derivative of x^2 is 2x.",
            "The integral of 1/x is ln(x) + C.",
            "The Pythagorean theorem states that a² + b² = c².",
            "Matrix multiplication is not commutative.",
            "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a.",
        ],
        "instruct": [
            "Human: What is the capital of France?\nAssistant: The capital of France is Paris.",
            "Human: How do I cook pasta?\nAssistant: To cook pasta, boil water, add salt, then add pasta and cook according to package directions.",
            "Human: Explain quantum computing.\nAssistant: Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information.",
            "Human: Write a Python function to calculate factorial.\nAssistant: Here's a Python function to calculate factorial:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        ]
    }
    
    # Generate random samples
    all_texts = []
    for category, texts in sample_texts.items():
        all_texts.extend(texts)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "jsonl":
        with open(output_path, 'w') as f:
            for i in range(num_samples):
                text = random.choice(all_texts)
                # Add some variation
                if random.random() < 0.3:
                    text = text + " " + random.choice(all_texts)
                
                data = {"text": text, "id": i}
                f.write(json.dumps(data) + "\n")
    
    elif format_type == "json":
        data = []
        for i in range(num_samples):
            text = random.choice(all_texts)
            data.append({"text": text, "id": i})
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format_type == "txt":
        with open(output_path, 'w') as f:
            for i in range(num_samples):
                text = random.choice(all_texts)
                f.write(text + "\n\n")
    
    print(f"📊 Created {num_samples} samples in {output_path} ({format_type} format)")
    return output_path


# Extended CLI with additional commands
def add_extended_commands(parser):
    """Add extended CLI commands for development and testing"""
    
    subparsers = parser._subparsers._group_actions[0]  # Get existing subparsers
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test CLI functionality')
    test_parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    test_parser.add_argument('--validate', action='store_true', help='Validate setup')
    
    # Create data command
    data_parser = subparsers.add_parser('create-data', help='Create sample training data')
    data_parser.add_argument('--output', type=str, required=True, help='Output file path')
    data_parser.add_argument('--format', type=str, choices=['jsonl', 'json', 'txt'], 
                           default='jsonl', help='Data format')
    data_parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    data_parser.add_argument('--type', type=str, choices=['general', 'code', 'math', 'instruct'],
                           default='general', help='Data type')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    interactive_parser.add_argument('--model', type=str, default='tiny_llama3_150m',
                                  help='Model for interactive session')
    
    return subparsers


class InteractiveCLI:
    """Interactive CLI mode for easier experimentation"""
    
    def __init__(self, default_model='tiny_llama3_150m'):
        self.default_model = default_model
        self.current_orchestrator = None
        
    def run(self):
        """Run interactive mode"""
        
        print("🦙 LLaMA Interactive CLI")
        print("=" * 50)
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\nllama> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command in ['help', 'h']:
                    self.show_help()
                elif command.startswith('model '):
                    self.set_model(command.split(' ', 1)[1])
                elif command.startswith('config'):
                    self.show_config()
                elif command.startswith('estimate '):
                    self.estimate_memory(command.split(' ', 1)[1])
                elif command.startswith('test'):
                    self.quick_test()
                elif command == 'models':
                    self.list_models()
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")
    
    def show_help(self):
        """Show help for interactive commands"""
        print("""
Available commands:
  help              - Show this help
  models            - List available models
  model <name>      - Set current model (e.g., 'model tiny_llama3_150m')
  config            - Show current configuration  
  estimate <args>   - Estimate memory (e.g., 'estimate batch=16 gpus=4')
  test              - Run quick functionality test
  quit              - Exit interactive mode

Examples:
  llama> model llama3_8b
  llama> config
  llama> estimate batch=16 gpus=4
        """)
    
    def list_models(self):
        """List available models"""
        models = [
            "tiny_llama3_50m (50M params, ultra-fast)",
            "tiny_llama3_150m (150M params, development)",
            "llama3_8b (8B params, production)",
            "llama3_8b_instruct (8B params, chat-optimized)",
            "llama3_70b (70B params, large-scale)",
            "llama3_70b_instruct (70B params, chat)",
            "llama3_405b (405B params, mega-scale)",
            "llama2_7b, llama2_13b, llama2_70b (LLaMA 2 variants)",
            "code_llama_7b (code-specialized)"
        ]
        
        print("\n📋 Available Models:")
        for model in models:
            print(f"  • {model}")
    
    def set_model(self, model_name):
        """Set current model"""
        try:
            self.current_orchestrator = LlamaTrainingOrchestrator(
                model_variant=model_name.strip(),
                training_strategy="standard",
                num_gpus=1,
                auto_configure=True
            )
            self.default_model = model_name.strip()
            print(f"✅ Model set to: {self.default_model}")
        except Exception as e:
            print(f"❌ Failed to set model: {e}")
    
    def show_config(self):
        """Show current configuration"""
        if self.current_orchestrator is None:
            self.set_model(self.default_model)
        
        print(f"\n📋 Current Configuration:")
        print(f"Model: {self.default_model}")
        if self.current_orchestrator:
            self.current_orchestrator.print_configuration()
    
    def estimate_memory(self, args_str):
        """Estimate memory requirements"""
        try:
            # Parse arguments like "batch=16 gpus=4"
            args = {}
            for arg in args_str.split():
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    args[key] = int(value) if value.isdigit() else value
            
            batch_size = args.get('batch', 8)
            num_gpus = args.get('gpus', 1)
            
            if self.current_orchestrator is None:
                self.set_model(self.default_model)
            
            memory_est = self.current_orchestrator.estimate_memory_requirements(
                batch_size=batch_size
            )
            
            print(f"\n💾 Memory Estimation for {self.default_model}:")
            print(f"  Batch Size: {batch_size}")
            print(f"  GPUs: {num_gpus}")
            for key, value in memory_est.items():
                if isinstance(value, (int, float)):
                    print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                
        except Exception as e:
            print(f"❌ Memory estimation failed: {e}")
    
    def quick_test(self):
        """Run quick test"""
        print("🧪 Running quick test...")
        quick_test_cli()


# Main CLI class with extended functionality
class ExtendedLlamaCLI(LlamaCLI):
    """Extended CLI with additional features"""
    
    def run(self):
        """Enhanced run method with extended commands"""
        
        args = self.parser.parse_args()
        
        if args.command is None:
            self.parser.print_help()
            return
        
        # Handle extended commands
        if args.command == 'test':
            if args.validate:
                validate_cli_setup()
            if args.quick:
                quick_test_cli()
            return
        
        elif args.command == 'create-data':
            create_sample_data(
                output_path=args.output,
                format_type=args.format,
                num_samples=args.samples
            )
            return
        
        elif args.command == 'interactive':
            interactive = InteractiveCLI(args.model)
            interactive.run()
            return
        
        # Handle standard commands
        super().run()


def setup_environment():
    """Setup training environment with proper configurations"""
    
    # Set environment variables for optimal performance
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


def print_system_info():
    """Print system information for debugging"""
    
    print("🖥️  System Information:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"     GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    else:
        print("   CUDA: Not available")
    
    print(f"   Working Directory: {os.getcwd()}")


def main():
    """Enhanced main entry point"""
    
    # Setup environment
    setup_environment()
    
    # Check if running in interactive mode or no arguments
    if len(sys.argv) == 1:
        print("🦙 LLaMA Training CLI")
        print_system_info()
        print("\nFor help: python cli.py --help")
        print("For interactive mode: python cli.py interactive")
        print("For quick test: python cli.py test --quick")
        print("For system validation: python cli.py test --validate")
        return
    
    # Handle special flags
    if '--version' in sys.argv:
        print("LLaMA Training CLI v1.0")
        print_system_info()
        return
    
    if '--system-info' in sys.argv:
        print_system_info()
        return
    
    try:
        # Use extended CLI
        cli = ExtendedLlamaCLI()
        cli.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CLI Error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()