### Python API

```python
from training_infra import quick_start

# 🚀 Tiny LLaMA 3 for development (super fast!)
orchestrator = quick_start(
    model_variant="tiny_llama3_150m",
    strategy="standard", 
    num_gpus=1  # Single GPU or even CPU!
)

# LLaMA 3 8B training - auto-configures everything
orchestrator = quick_start(
    model_variant="llama3_8b",
    strategy="standard", 
    num_gpus=4
)

# LLaMA 3 70B with instruct tuning
orchestrator = quick_start(
    model_variant="llama3_70b_instruct",
    strategy="lora",
    num_gpus=8
)

# Launch training (you provide the data)
orchestrator.launch_training(train_dataloader, val_dataloader)
```

### CLI Usage

```bash
# 🚀 Super fast development with Tiny LLaMA 3
python -m training_infra.cli train \
    --model tiny_llama3_150m \
    --gpus 1 \
    --data /path/to/small_dataset.jsonl \
    --batch-size 32

# Train LLaMA 3 8B with 4 GPUs
python -m training_infra.cli train \
    --model llama3_8b \
    --gpus 4 \
    --data /path/to/training_data.jsonl \
    --strategy standard

# Train LLaMA 3 70B with instruct tuning
python -m training_infra.cli train \
    --model llama3_70b_instruct \
    --strategy lora \
    --gpus 8 \
    --lora-rank 64

# Train massive LLaMA 3 405B (requires 32+ GPUs)
python -m training_infra.cli train \
    --model llama3_405b \
    --gpus 64 \
    --strategy standard \
    --zero-stage 3
```# 🦙 LLaMA Distributed Training Infrastructure

A comprehensive, production-ready framework for training LLaMA models with advanced distributed parallelism, designed for scalability, efficiency, and ease of use.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🚀 Key Features

### **🔥 Advanced Parallelism**
- **3D Parallelism**: Tensor + Pipeline + Data parallelism
- **Microbatching**: 1F1B, GPipe, and Chimera scheduling
- **ZeRO Optimizer**: Stages 1, 2, and 3 support
- **Sequence Parallelism**: For very long sequences

### **🧠 Training Strategies**
- **Standard**: Classic transformer training
- **MoE (Mixture of Experts)**: Sparse expert models
- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **Hybrid**: Combination of multiple techniques

### **⚡ Memory Optimizations**
- **Activation Checkpointing**: Adaptive gradient checkpointing
- **Flash Attention**: Memory-efficient attention computation
- **Gradient Compression**: Reduce communication overhead
- **Mixed Precision**: bfloat16/float16 training

### **🎯 Production Features**
- **Auto-Configuration**: Intelligent resource management
- **CLI Interface**: Easy command-line usage
- **Memory Estimation**: Predict resource requirements
- **Performance Benchmarking**: Compare configurations
- **Comprehensive Logging**: Weights & Biases, TensorBoard

## 📦 Installation

### Prerequisites
```bash
# CUDA-capable GPUs recommended
# Python 3.8+ required
# PyTorch 2.0+ required
```

### Basic Installation
```bash
pip install torch>=2.0.0 transformers>=4.20.0
git clone https://github.com/your-org/llama-training-infra.git
cd llama-training-infra
pip install -e .
```

### With Optional Dependencies
```bash
# For advanced features
pip install -e .[deepspeed,flash_attn,moe,visualization,benchmarking]
```

## 🏃‍♂️ Quick Start

### Python API

```python
from training_infra import quick_start

# LLaMA 3 8B training - auto-configures everything
orchestrator = quick_start(
    model_variant="llama3_8b",
    strategy="standard", 
    num_gpus=4
)

# LLaMA 3 70B with instruct tuning
orchestrator = quick_start(
    model_variant="llama3_70b_instruct",
    strategy="lora",
    num_gpus=8
)

# Launch training (you provide the data)
orchestrator.launch_training(train_dataloader, val_dataloader)
```

### CLI Usage

```bash
# Train LLaMA 3 8B with 4 GPUs
python -m training_infra.cli train \
    --model llama3_8b \
    --gpus 4 \
    --data /path/to/training_data.jsonl \
    --strategy standard

# Train LLaMA 3 70B with instruct tuning
python -m training_infra.cli train \
    --model llama3_70b_instruct \
    --strategy lora \
    --gpus 8 \
    --lora-rank 64

# Train massive LLaMA 3 405B (requires 32+ GPUs)
python -m training_infra.cli train \
    --model llama3_405b \
    --gpus 64 \
    --strategy standard \
    --zero-stage 3
```

## 🚀 LLaMA 3 Quick Examples

### Tiny LLaMA 3 - Development & Testing
```python
from training_infra import create_tiny_llama3_orchestrator

# Ultra-fast 50M model for rapid experimentation
tiny_orchestrator = create_tiny_llama3_orchestrator(
    size="50m",      # 50M parameters
    num_gpus=1,      # Single GPU or CPU
    strategy="standard"
)

# Perfect for:
# - Testing new training code
# - Validating data pipelines  
# - Architecture experiments
# - CI/CD testing

tiny_orchestrator.launch_training(small_dataloader)
```

### LLaMA 3 8B Training
```python
from training_infra import create_llama3_8b_orchestrator

# Create orchestrator for LLaMA 3 8B
orchestrator = create_llama3_8b_orchestrator(
    num_gpus=4,
    strategy="standard"
)

# Auto-configured for optimal performance
orchestrator.print_configuration()
orchestrator.launch_training(train_dataloader, val_dataloader)
```

### LLaMA 3 70B with LoRA Fine-tuning
```python
from training_infra import create_llama3_70b_orchestrator, TrainingStrategy

# Create LoRA strategy for efficient fine-tuning
lora_strategy = TrainingStrategy(
    name="lora",
    parameters={
        "lora_rank": 64,
        "lora_alpha": 128,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
)

# Create orchestrator
orchestrator = create_llama3_70b_orchestrator(
    num_gpus=8,
    strategy=lora_strategy,
    instruct=True  # Use instruct variant
)

orchestrator.launch_training(finetune_dataloader)
```

### LLaMA 3 405B Mega Model Training
```python
from training_infra import create_llama3_405b_orchestrator

# Requires 32+ GPUs
orchestrator = create_llama3_405b_orchestrator(
    num_gpus=64,  # Use 64 GPUs for optimal performance
    strategy="standard"
)

# Automatically configures:
# - 8-way tensor parallelism
# - 8-way pipeline parallelism  
# - ZeRO stage 3
# - Aggressive memory optimizations

orchestrator.launch_training(massive_dataloader)
```

### CLI Examples for LLaMA 3
```bash
# Ultra-fast development with Tiny LLaMA 3 50M
python -m training_infra.cli train \
    --model tiny_llama3_50m \
    --gpus 1 \
    --data /path/to/test_data.jsonl \
    --batch-size 64 \
    --epochs 1

# Development with Tiny LLaMA 3 150M
python -m training_infra.cli train \
    --model tiny_llama3_150m \
    --gpus 1 \
    --data /path/to/dev_data.jsonl \
    --batch-size 32

# LLaMA 3 8B training
python -m training_infra.cli train \
    --model llama3_8b \
    --gpus 4 \
    --data /path/to/data.jsonl \
    --batch-size 8

# LLaMA 3 70B instruct fine-tuning with LoRA
python -m training_infra.cli train \
    --model llama3_70b_instruct \
    --strategy lora \
    --gpus 8 \
    --lora-rank 64 \
    --data /path/to/instruct_data.jsonl

# LLaMA 3 405B training (mega scale)
python -m training_infra.cli train \
    --model llama3_405b \
    --gpus 64 \
    --nodes 8 \
    --zero-stage 3 \
    --activation-checkpointing \
    --gradient-compression
```

### Development Workflow with Tiny LLaMA 3
```bash
# Step 1: Test with ultra-fast 50M model
python -m training_infra.cli train \
    --model tiny_llama3_50m \
    --gpus 1 \
    --data sample_data.jsonl \
    --epochs 1 \
    --dry-run  # Just test configuration

# Step 2: Validate with 150M model  
python -m training_infra.cli train \
    --model tiny_llama3_150m \
    --gpus 1 \
    --data dev_data.jsonl \
    --epochs 2

# Step 3: Scale to production model
python -m training_infra.cli train \
    --model llama3_8b \
    --gpus 4 \
    --data full_dataset.jsonl \
    --epochs 3
```

### Custom Configuration

```python
from training_infra import LlamaTrainingOrchestrator, TrainingStrategy
from training_infra.distributed import ConfigurationFactory

# Create custom training strategy
strategy = TrainingStrategy(
    name="moe",
    parameters={
        "num_experts": 16,
        "experts_per_token": 2,
        "moe_layers": [4, 8, 12, 16, 20, 24, 28]
    }
)

# Create orchestrator with custom config
orchestrator = LlamaTrainingOrchestrator(
    model_variant="llama2_13b",
    training_strategy=strategy,
    num_gpus=8,
    auto_configure=True
)

# Override distributed configuration
orchestrator.distributed_config = ConfigurationFactory.create_hybrid_parallel_config(
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    data_parallel_size=1,
    use_zero=True,
    zero_stage=3
)

# Print detailed configuration
orchestrator.print_configuration()

# Launch training
orchestrator.launch_training(train_dataloader, val_dataloader)
```

### Manual Trainer Creation

```python
from training_infra.distributed import create_distributed_trainer
from training_infra.models.llama import create_llama_7b_parallel
from training_infra import TrainingConfig

# Create model
model = create_llama_7b_parallel(
    tensor_parallel_size=2,
    use_flash_attention=True,
    use_checkpointing=True
)

# Create training config
training_config = TrainingConfig(
    model_name="custom_llama_7b",
    batch_size=4,
    learning_rate=1e-5,
    epochs=3
)

# Create distributed config
distributed_config = ConfigurationFactory.create_tensor_parallel_config(2)

# Create trainer
trainer = create_distributed_trainer(
    model=model,
    config=training_config,
    distributed_config=distributed_config,
    trainer_type="adaptive"
)

# Train
trainer.fit()
```

## 🔧 Configuration Options

### Model Variants

| Model | Parameters | Context Length | Vocab Size | Min GPUs | Architecture Highlights |
|-------|------------|----------------|------------|----------|-------------------------|
| `tiny_llama3_50m` | **50M** | 2K | 128K | **1 (CPU OK)** | **🚀 Ultra-fast development model** |
| `tiny_llama3_150m` | **150M** | 8K | 128K | **1** | **⚡ Development & testing optimized** |
| `llama1_7b` | 7B | 2K | 32K | 1 | Original LLaMA architecture |
| `llama2_7b` | 7B | 4K | 32K | 1 | Extended context, improved training |
| `llama2_13b` | 13B | 4K | 32K | 2 | Larger model, better performance |
| `llama2_70b` | 70B | 4K | 32K | 8 | Grouped Query Attention (GQA) |
| `llama3_8b` | 8B | 8K | 128K | 1 | **Enhanced architecture, larger vocab** |
| `llama3_8b_instruct` | 8B | 8K | 128K | 1 | **Chat-optimized, multi-turn conversations** |
| `llama3_70b` | 70B | 8K | 128K | 8 | **Improved GQA, better reasoning** |
| `llama3_70b_instruct` | 70B | 8K | 128K | 8 | **Advanced instruction following** |
| `llama3_405b` | 405B | 8K | 128K | 32+ | **🔥 Mega model, state-of-the-art** |
| `code_llama_7b` | 7B | 16K | 32K | 1 | Code-specialized, long context |

#### 🚀 Tiny LLaMA 3 - Development & Testing Models:
- **⚡ 50M Model**: Ultra-fast training, perfect for code testing and experimentation
- **🔧 150M Model**: Ideal for development, architecture validation, and rapid prototyping  
- **🧪 Same Architecture**: Uses LLaMA 3 architecture (GQA, large vocab, RoPE scaling)
- **💻 CPU/Single GPU**: Can run on modest hardware for development

#### LLaMA 3 Key Improvements:
- **🚀 4x Larger Vocabulary** (128K vs 32K tokens)
- **📏 Extended Context** (8K vs 4K tokens)  
- **🧠 Enhanced Architecture** (improved FFN, better GQA)
- **💬 Superior Instruction Following** (instruct variants)
- **⚡ Better Training Stability** (improved RoPE scaling)

### Training Strategies

#### Standard
```python
# Classic transformer training
strategy = TrainingStrategy(name="standard")
```

#### MoE (Mixture of Experts)
```python
strategy = TrainingStrategy(
    name="moe",
    parameters={
        "num_experts": 8,           # Number of expert networks
        "experts_per_token": 2,     # Experts activated per token
        "moe_layers": [4, 8, 12]    # Which layers to make sparse
    }
)
```

#### LoRA (Low-Rank Adaptation)
```python
strategy = TrainingStrategy(
    name="lora",
    parameters={
        "lora_rank": 16,      # Rank of adaptation matrices
        "lora_alpha": 32,     # Scaling parameter
        "target_modules": ["q_proj", "v_proj"]  # Modules to adapt
    }
)
```

### Parallelism Strategies

#### Data Parallelism
```python
# Replicate model across GPUs
config = ConfigurationFactory.create_data_parallel_config(num_gpus=4)
```

#### Tensor Parallelism
```python
# Split model layers across GPUs
config = ConfigurationFactory.create_tensor_parallel_config(num_gpus=4)
```

#### Pipeline Parallelism
```python
# Split model into pipeline stages
config = ConfigurationFactory.create_pipeline_parallel_config(
    pipeline_stages=4,
    microbatch_size=2,
    schedule="1f1b"  # or "gpipe", "chimera"
)
```

#### Hybrid 3D Parallelism
```python
# Combine all parallelism types
config = ConfigurationFactory.create_hybrid_parallel_config(
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    data_parallel_size=2,
    use_zero=True,
    zero_stage=2
)
```

## 📊 Memory Estimation

Estimate memory requirements before training:

```python
from training_infra import estimate_requirements

# Estimate memory for LLaMA 7B
reqs = estimate_requirements(
    model_variant="llama2_7b",
    batch_size=16,
    sequence_length=2048,
    num_gpus=4
)

print(f"Memory per GPU: {reqs['memory_per_gpu_gb']:.1f} GB")
print(f"Recommendations: {reqs['recommendations']}")
```

CLI version:
```bash
python -m training_infra.cli estimate-memory \
    --model llama2_7b \
    --batch-size 16 \
    --gpus 4
```

## ⚡ Performance Benchmarking

Benchmark different configurations:

```python
from training_infra.benchmarking import PerformanceBenchmark

benchmark = PerformanceBenchmark("llama2_7b")
results = benchmark.run_comprehensive_benchmark()
benchmark.print_summary()
```

CLI version:
```bash
python -m training_infra.cli benchmark \
    --model llama2_7b \
    --gpus 4 \
    --batch-sizes 1 2 4 8 \
    --save-results benchmark_results.json
```

## 🔄 Pipeline Parallelism Deep Dive

### Microbatch Scheduling

Our pipeline parallelism implementation supports multiple scheduling strategies:

```python
from training_infra.distributed.microbatch_scheduler import MicrobatchScheduler

scheduler = MicrobatchScheduler(
    num_microbatches=8,
    microbatch_size=2,
    num_pipeline_stages=4,
    current_stage=1,
    schedule_type="1f1b"  # or "gpipe", "chimera", "interleaved"
)

# Execute pipeline schedule
schedule = scheduler.get_execution_schedule()
results = scheduler.execute_schedule(forward_fn, backward_fn, optimizer_fn, microbatches)

# View performance metrics
scheduler.print_metrics()
```

### Schedule Types

- **1F1B (1 Forward 1 Backward)**: Interleaved execution for memory efficiency
- **GPipe**: All forwards then all backwards for simplicity
- **Chimera**: Adaptive scheduling based on memory pressure
- **Interleaved**: Virtual pipeline stages for better GPU utilization

## 🗂️ Project Structure

```
training_infra/
├── __init__.py                 # Main package interface
├── cli.py                      # Command-line interface
├── orchestrator.py             # High-level training orchestrator
├── config.py                   # Training configuration
├── trainer.py                  # Base trainer class
├── distributed/
│   ├── config.py              # Distributed configurations
│   ├── trainer.py             # Distributed trainers
│   └── microbatch_scheduler.py # Pipeline parallelism
├── models/
│   ├── llama.py               # LLaMA model implementations
│   └── moe.py                 # Mixture of Experts
└── parallelism/               # Low-level parallelism primitives
```

## 🎯 Best Practices

### Memory Optimization
1. **Use activation checkpointing** for large models
2. **Enable ZeRO optimizer** for models > 13B parameters
3. **Use mixed precision** (bfloat16 preferred)
4. **Gradient compression** for slow interconnects

### Parallelism Strategy
- **Tiny models (50M-150M)**: Single GPU, data parallelism for speed
- **8B models**: Data parallelism or 2-way tensor parallelism
- **70B models**: 8-way tensor parallelism or hybrid 3D
- **405B models**: Hybrid 3D parallelism with ZeRO stage 3 (32+ GPUs required)

### Memory Requirements Summary

| Model | Single GPU | Recommended Setup | Training Memory |
|-------|------------|-------------------|-----------------|
| Tiny LLaMA 3 50M | **0.5GB** | 1x GPU (even old ones!) | **~2GB** |
| Tiny LLaMA 3 150M | **1GB** | 1x GPU | **~4GB** |
| LLaMA 3 8B | **20GB** | 1x RTX 4090 or A100 | **~25GB** |
| LLaMA 3 70B | **200GB** | 8x A100 (tensor parallel) | **~25GB per GPU** |
| LLaMA 3 405B | **1TB+** | 32+ H100 (3D parallel) | **~30GB per GPU** |

### LLaMA 3 Specific Optimizations
- **Larger vocabulary** requires more memory for embeddings (~512MB extra)
- **Extended context** benefits from sequence parallelism
- **GQA optimization** reduces KV cache memory usage significantly
- **RoPE scaling** enables longer sequence training without memory explosion

### Pipeline Parallelism
- **Use 1F1B schedule** for best memory efficiency
- **4-8 microbatches per stage** for optimal throughput
- **Even model splitting** across pipeline stages

## 🐛 Troubleshooting

### ⚠️ Common Issues

#### Out of Memory (OOM)
```bash
# Start with tiny model for development
--model tiny_llama3_150m

# For larger models, try smaller batch size
--batch-size 2

# Enable checkpointing
--activation-checkpointing

# Use ZeRO
--zero-stage 2

# More parallelism
--tensor-parallel 4
```

#### Slow Training
```bash
# Use tiny model for rapid iteration
--model tiny_llama3_50m

# Enable Flash Attention
# (automatically enabled)

# Use mixed precision
--mixed-precision bfloat16

# Optimize communication
--gradient-compression
```

#### Development Workflow
```bash
# 1. Test logic with ultra-fast model
python -m training_infra.cli train \
    --model tiny_llama3_50m \
    --dry-run

# 2. Validate with small dataset
python -m training_infra.cli train \
    --model tiny_llama3_150m \
    --epochs 1

# 3. Scale to production
python -m training_infra.cli train \
    --model llama3_8b \
    --gpus 4
```

#### Poor GPU Utilization
```bash
# More microbatches for pipeline
--pipeline-parallel 4

# Async communication
# (enabled by default)

# Check data loading
# Use sufficient num_workers
```

### Environment Validation

```python
from training_infra import validate_environment

validation = validate_environment()
if validation["errors"]:
    print("Issues found:", validation["errors"])
else:
    print("Environment ready for training!")
```

## 📚 Examples

See the `examples/` directory for complete training scripts:

- `examples/train_llama_7b.py` - Basic LLaMA 7B training
- `examples/train_with_moe.py` - MoE training example
- `examples/finetune_with_lora.py` - LoRA fine-tuning
- `examples/multi_node_training.py` - Multi-node setup
- `examples/custom_data_loading.py` - Custom data pipeline

## 🤝 Contributing

We welcome contributions! Please see:

- [Contributing Guidelines](CONTRIBUTING.md)
- [Development Setup](docs/development.md)
- [Code Style Guide](docs/style.md)

### Development Setup
```bash
git clone https://github.com/your-org/llama-training-infra.git
cd llama-training-infra
pip install -e .[dev]
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Claude for writing the code
- Meta AI for the LLaMA architecture
- Microsoft DeepSpeed team for ZeRO optimizer
- HuggingFace for transformers library
- NVIDIA for Flash Attention and FasterTransformer optimizations

