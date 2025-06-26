# training_infra/pipeline/synthetic_distillation.py
"""
Complete pipeline combining synthetic data generation and knowledge distillation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import copy
from pathlib import Path
import numpy as np
import time

from ..data.synthetic import (
    SyntheticDataGenerator, SyntheticDataConfig,
    MathDataGenerator, CodeDataGenerator,
    create_synthetic_data_generator
)
from ..data.distillation import (
    DistillationTrainer, DistillationConfig,
    ProgressiveDistillationTrainer, SelfDistillationTrainer,
    create_distillation_trainer
)
from ..models.llama import create_llama_7b
from ..config import TrainingConfig

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import json

@dataclass
class ProductionCompressionConfig:
    """Configuration for production compression pipeline"""
    # Compression settings
    target_compression: float = 0.3  # Target model size ratio
    compression_method: str = "distillation"  # "distillation", "pruning", "quantization"
    
    # Synthetic data generation
    num_synthetic_samples: int = 50000
    synthetic_domains: List[str] = None
    data_quality_threshold: float = 0.8
    
    # Distillation settings
    distillation_epochs: int = 10
    distillation_lr: float = 1e-4
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    
    # Student model architecture
    student_layers: Optional[int] = None
    student_hidden_size: Optional[int] = None
    student_attention_heads: Optional[int] = None
    
    # Evaluation settings
    eval_datasets: List[str] = None
    performance_threshold: float = 0.85  # Minimum performance to maintain
    
    # Production settings
    batch_size: int = 16
    max_sequence_length: int = 512
    device: str = "auto"
    
    def __post_init__(self):
        if self.synthetic_domains is None:
            self.synthetic_domains = ["general", "reasoning", "creative"]
        if self.eval_datasets is None:
            self.eval_datasets = ["validation"]

class ProductionCompressionPipeline:
    """Complete pipeline for production model compression"""
    
    def __init__(self, teacher_model, tokenizer, config: Optional[ProductionCompressionConfig] = None):
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.config = config or ProductionCompressionConfig()
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Pipeline state
        self.synthetic_data = []
        self.student_model = None
        self.compression_metrics = {}
        
    def run_complete_pipeline(self, synthetic_config=None, distillation_config=None) -> Dict[str, Any]:
        """Run end-to-end compression pipeline"""
        print(f"🚀 Starting Production Compression Pipeline")
        print(f"Target compression: {self.config.target_compression:.1%}")
        print(f"Device: {self.device}")
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Analyze teacher model
            teacher_stats = self._analyze_teacher_model()
            print(f"✅ Teacher model analysis complete")
            
            # Step 2: Generate synthetic training data
            self.synthetic_data = self._generate_synthetic_data()
            print(f"✅ Generated {len(self.synthetic_data)} synthetic samples")
            
            # Step 3: Design student architecture
            self.student_model = self._create_student_model(teacher_stats)
            print(f"✅ Student model created")
            
            # Step 4: Run distillation training
            distillation_results = self._run_distillation_training()
            print(f"✅ Distillation training complete")
            
            # Step 5: Evaluate compressed model
            evaluation_results = self._evaluate_compressed_model()
            print(f"✅ Evaluation complete")
            
            # Step 6: Optimize for production
            production_model = self._optimize_for_production()
            print(f"✅ Production optimization complete")
            
            # Step 7: Generate deployment artifacts
            deployment_artifacts = self._generate_deployment_artifacts()
            print(f"✅ Deployment artifacts generated")
            
            pipeline_time = time.time() - pipeline_start
            
            # Compile final results
            results = {
                'pipeline_status': 'success',
                'pipeline_time': pipeline_time,
                'teacher_stats': teacher_stats,
                'student_model': production_model,
                'compression_ratio': self._calculate_compression_ratio(),
                'distillation_results': distillation_results,
                'evaluation_results': evaluation_results,
                'deployment_artifacts': deployment_artifacts,
                'synthetic_data_stats': {
                    'total_samples': len(self.synthetic_data),
                    'domains': self.config.synthetic_domains,
                    'avg_quality_score': self._calculate_avg_quality_score()
                }
            }
            
            print(f"🎉 Pipeline completed successfully in {pipeline_time:.2f}s")
            print(f"📊 Compression ratio: {results['compression_ratio']:.1%}")
            print(f"📈 Performance retention: {evaluation_results.get('performance_retention', 'N/A')}")
            
            return results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            return {
                'pipeline_status': 'failed',
                'error': str(e),
                'pipeline_time': time.time() - pipeline_start
            }
    
    def _analyze_teacher_model(self) -> Dict[str, Any]:
        """Analyze teacher model architecture and performance"""
        print("🔍 Analyzing teacher model...")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.teacher_model.parameters())
        trainable_params = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
        
        # Get model architecture info
        if hasattr(self.teacher_model, 'config'):
            config = self.teacher_model.config
            architecture_info = {
                'num_layers': getattr(config, 'num_hidden_layers', 'unknown'),
                'hidden_size': getattr(config, 'hidden_size', 'unknown'),
                'num_attention_heads': getattr(config, 'num_attention_heads', 'unknown'),
                'vocab_size': getattr(config, 'vocab_size', 'unknown')
            }
        else:
            architecture_info = {'type': 'custom'}
        
        # Estimate memory usage
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'architecture': architecture_info,
            'target_parameters': int(total_params * self.config.target_compression),
            'target_size_mb': model_size_mb * self.config.target_compression
        }
        
        return stats
    
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate high-quality synthetic training data"""
        print("🎲 Generating synthetic training data...")
        
        synthetic_data = []
        samples_per_domain = self.config.num_synthetic_samples // len(self.config.synthetic_domains)
        
        for domain in self.config.synthetic_domains:
            print(f"  Generating {samples_per_domain} samples for domain: {domain}")
            
            domain_samples = self._generate_domain_samples(domain, samples_per_domain)
            synthetic_data.extend(domain_samples)
        
        # Filter by quality
        high_quality_data = [
            sample for sample in synthetic_data 
            if sample.get('quality_score', 0) >= self.config.data_quality_threshold
        ]
        
        print(f"  Filtered to {len(high_quality_data)} high-quality samples")
        return high_quality_data
    
    def _generate_domain_samples(self, domain: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples for a specific domain"""
        samples = []
        
        # Domain-specific prompts
        domain_prompts = self._get_domain_prompts(domain)
        
        for i in range(num_samples):
            prompt = domain_prompts[i % len(domain_prompts)]
            
            try:
                # Generate with teacher model
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = self.teacher_model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 128,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Calculate quality score (simplified)
                quality_score = self._calculate_quality_score(prompt, generated_text)
                
                sample = {
                    'domain': domain,
                    'prompt': prompt,
                    'response': generated_text,
                    'quality_score': quality_score,
                    'input_ids': inputs[0].cpu(),
                    'target_ids': outputs[0].cpu()
                }
                
                samples.append(sample)
                
            except Exception as e:
                print(f"    Warning: Failed to generate sample {i}: {e}")
                continue
        
        return samples
    
    def _get_domain_prompts(self, domain: str) -> List[str]:
        """Get prompts for specific domain"""
        prompts = {
            'general': [
                "Explain the concept of",
                "What is the relationship between",
                "Describe the process of",
                "Compare and contrast",
                "Analyze the importance of"
            ],
            'reasoning': [
                "If A implies B and B implies C, then",
                "Given the following premises, what can we conclude:",
                "Solve this step by step:",
                "What is the logical flaw in this argument:",
                "Use deductive reasoning to determine:"
            ],
            'creative': [
                "Write a short story about",
                "Create a poem that describes",
                "Imagine a world where",
                "Design a creative solution for",
                "Compose a dialogue between"
            ]
        }
        
        base_prompts = prompts.get(domain, prompts['general'])
        
        # Expand prompts with variations
        expanded_prompts = []
        topics = ["artificial intelligence", "climate change", "space exploration", 
                 "quantum physics", "human psychology", "technology ethics"]
        
        for prompt in base_prompts:
            for topic in topics:
                expanded_prompts.append(f"{prompt} {topic}")
        
        return expanded_prompts
    
    def _calculate_quality_score(self, prompt: str, response: str) -> float:
        """Calculate quality score for generated text"""
        # Simplified quality metrics
        score = 0.0
        
        # Length check
        if 10 <= len(response) <= 500:
            score += 0.3
        
        # Coherence check (simplified)
        if response and not response.startswith(prompt):
            score += 0.2
        
        # Repetition check
        words = response.split()
        if len(set(words)) / max(len(words), 1) > 0.7:
            score += 0.3
        
        # Completion check
        if response.endswith('.') or response.endswith('!') or response.endswith('?'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _create_student_model(self, teacher_stats: Dict[str, Any]) -> nn.Module:
        """Create student model architecture"""
        print("🏗️ Creating student model architecture...")
        
        # Calculate student architecture dimensions
        if self.config.student_layers is None:
            teacher_layers = teacher_stats['architecture'].get('num_layers', 12)
            if isinstance(teacher_layers, int):
                self.config.student_layers = max(1, int(teacher_layers * self.config.target_compression))
            else:
                self.config.student_layers = 6  # Default
        
        if self.config.student_hidden_size is None:
            teacher_hidden = teacher_stats['architecture'].get('hidden_size', 768)
            if isinstance(teacher_hidden, int):
                self.config.student_hidden_size = max(128, int(teacher_hidden * 0.8))
            else:
                self.config.student_hidden_size = 512  # Default
        
        if self.config.student_attention_heads is None:
            teacher_heads = teacher_stats['architecture'].get('num_attention_heads', 12)
            if isinstance(teacher_heads, int):
                self.config.student_attention_heads = max(1, int(teacher_heads * 0.75))
            else:
                self.config.student_attention_heads = 8  # Default
        
        # Create student model (simplified implementation)
        try:
            if hasattr(self.teacher_model, 'config'):
                # Clone and modify teacher config
                student_config = type(self.teacher_model.config)(**self.teacher_model.config.__dict__)
                student_config.num_hidden_layers = self.config.student_layers
                student_config.hidden_size = self.config.student_hidden_size
                student_config.num_attention_heads = self.config.student_attention_heads
                
                # Create student model with same architecture type
                student_model = type(self.teacher_model)(student_config)
            else:
                # Fallback: create a simple transformer
                student_model = self._create_simple_transformer()
            
            student_model.to(self.device)
            
            # Initialize with smaller weights
            self._initialize_student_weights(student_model)
            
            print(f"  Student layers: {self.config.student_layers}")
            print(f"  Student hidden size: {self.config.student_hidden_size}")
            print(f"  Student attention heads: {self.config.student_attention_heads}")
            
            return student_model
            
        except Exception as e:
            print(f"  Warning: Could not create optimized student model: {e}")
            print("  Using teacher model as fallback")
            return self.teacher_model
    
    def _create_simple_transformer(self) -> nn.Module:
        """Create a simple transformer model as fallback"""
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size=50257, hidden_size=512, num_layers=6):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(hidden_size, 8) 
                    for _ in range(num_layers)
                ])
                self.lm_head = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_ids, **kwargs):
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                logits = self.lm_head(x)
                return type('Output', (), {'logits': logits})()
            
            def generate(self, input_ids, **kwargs):
                # Simplified generation
                return input_ids
        
        return SimpleTransformer(
            hidden_size=self.config.student_hidden_size,
            num_layers=self.config.student_layers
        )
    
    def _initialize_student_weights(self, student_model: nn.Module):
        """Initialize student model weights"""
        for module in student_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _run_distillation_training(self) -> Dict[str, Any]:
        """Run knowledge distillation training"""
        print("🎓 Running distillation training...")
        
        if not self.synthetic_data:
            raise ValueError("No synthetic data available for training")
        
        # Setup training
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(), 
            lr=self.config.distillation_lr
        )
        
        loss_history = []
        
        # Training loop
        for epoch in range(self.config.distillation_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(self.synthetic_data), self.config.batch_size):
                batch_data = self.synthetic_data[i:i + self.config.batch_size]
                
                # Prepare batch
                input_ids = []
                target_ids = []
                
                for sample in batch_data:
                    if 'input_ids' in sample and 'target_ids' in sample:
                        input_ids.append(sample['input_ids'])
                        target_ids.append(sample['target_ids'])
                
                if not input_ids:
                    continue
                
                # Pad sequences
                max_len = min(max(len(seq) for seq in input_ids), self.config.max_sequence_length)
                
                padded_inputs = torch.zeros(len(input_ids), max_len, dtype=torch.long)
                padded_targets = torch.zeros(len(target_ids), max_len, dtype=torch.long)
                
                for j, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
                    seq_len = min(len(inp), max_len)
                    padded_inputs[j, :seq_len] = inp[:seq_len]
                    tgt_len = min(len(tgt), max_len)
                    padded_targets[j, :tgt_len] = tgt[:tgt_len]
                
                padded_inputs = padded_inputs.to(self.device)
                padded_targets = padded_targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                try:
                    # Student forward
                    student_outputs = self.student_model(padded_inputs)
                    
                    # Teacher forward (for distillation)
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(padded_inputs)
                    
                    # Compute distillation loss
                    loss = self._compute_distillation_loss(
                        student_outputs.logits,
                        teacher_outputs.logits,
                        padded_targets
                    )
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"    Warning: Batch failed: {e}")
                    continue
            
            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_loss)
            
            print(f"  Epoch {epoch + 1}/{self.config.distillation_epochs}, Loss: {avg_loss:.4f}")
        
        return {
            'loss_history': loss_history,
            'final_loss': loss_history[-1] if loss_history else float('inf'),
            'num_epochs': self.config.distillation_epochs,
            'num_samples_trained': len(self.synthetic_data)
        }
    
    def _compute_distillation_loss(self, student_logits, teacher_logits, target_ids):
        """Compute knowledge distillation loss"""
        # Distillation loss (KL divergence)
        student_probs = torch.softmax(student_logits / self.config.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.config.temperature, dim=-1)
        
        distill_loss = torch.nn.functional.kl_div(
            torch.log(student_probs + 1e-8),
            teacher_probs,
            reduction='batchmean'
        ) * (self.config.temperature ** 2)
        
        # Task-specific loss (cross-entropy)
        task_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            target_ids.view(-1),
            ignore_index=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
        )
        
        # Combined loss
        total_loss = self.config.alpha * distill_loss + (1 - self.config.alpha) * task_loss
        
        return total_loss
    
    def _evaluate_compressed_model(self) -> Dict[str, Any]:
        """Evaluate compressed model performance"""
        print("📊 Evaluating compressed model...")
        
        # Create evaluation dataset
        eval_data = self.synthetic_data[-100:]  # Use last 100 samples for eval
        
        teacher_perplexity = self._calculate_perplexity(self.teacher_model, eval_data)
        student_perplexity = self._calculate_perplexity(self.student_model, eval_data)
        
        # Calculate performance metrics
        performance_retention = teacher_perplexity / max(student_perplexity, 1e-8)
        
        # Speed benchmark
        speed_metrics = self._benchmark_inference_speed()
        
        results = {
            'teacher_perplexity': teacher_perplexity,
            'student_perplexity': student_perplexity,
            'performance_retention': performance_retention,
            'meets_threshold': performance_retention >= self.config.performance_threshold,
            'speed_improvement': speed_metrics['speedup'],
            'memory_reduction': self._calculate_memory_reduction(),
            'inference_latency_ms': speed_metrics['student_latency_ms']
        }
        
        print(f"  Performance retention: {performance_retention:.2%}")
        print(f"  Speed improvement: {speed_metrics['speedup']:.1f}x")
        print(f"  Memory reduction: {results['memory_reduction']:.1%}")
        
        return results
    
    def _calculate_perplexity(self, model, eval_data) -> float:
        """Calculate model perplexity on evaluation data"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for sample in eval_data[:20]:  # Limit for speed
                if 'input_ids' not in sample:
                    continue
                
                try:
                    input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                    
                    if input_ids.shape[1] > 1:
                        outputs = model(input_ids)
                        
                        # Calculate loss
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction='sum'
                        )
                        
                        total_loss += loss.item()
                        total_tokens += shift_labels.numel()
                        
                except Exception:
                    continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def _benchmark_inference_speed(self) -> Dict[str, float]:
        """Benchmark inference speed of both models"""
        test_input = torch.randint(0, 1000, (1, 50)).to(self.device)
        
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                _ = self.teacher_model(test_input)
                _ = self.student_model(test_input)
        
        # Benchmark teacher
        teacher_times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = self.teacher_model(test_input)
            teacher_times.append(time.time() - start_time)
        
        # Benchmark student
        student_times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = self.student_model(test_input)
            student_times.append(time.time() - start_time)
        
        teacher_avg = sum(teacher_times) / len(teacher_times)
        student_avg = sum(student_times) / len(student_times)
        
        return {
            'teacher_latency_ms': teacher_avg * 1000,
            'student_latency_ms': student_avg * 1000,
            'speedup': teacher_avg / max(student_avg, 1e-8)
        }
    
    def _calculate_memory_reduction(self) -> float:
        """Calculate memory reduction percentage"""
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        reduction = (teacher_params - student_params) / teacher_params
        return reduction * 100
    
    def _optimize_for_production(self) -> nn.Module:
        """Optimize model for production deployment"""
        print("🔧 Optimizing for production...")
        
        # Set to eval mode
        self.student_model.eval()
        
        # Optionally apply quantization
        try:
            # Dynamic quantization for CPU deployment
            quantized_model = torch.quantization.quantize_dynamic(
                self.student_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("  Applied dynamic quantization")
            return quantized_model
        except Exception as e:
            print(f"  Quantization failed: {e}")
            return self.student_model
    
    def _generate_deployment_artifacts(self) -> Dict[str, Any]:
        """Generate artifacts needed for deployment"""
        print("📦 Generating deployment artifacts...")
        
        artifacts = {
            'model_config': {
                'model_type': 'compressed_llm',
                'compression_method': self.config.compression_method,
                'compression_ratio': self.config.target_compression,
                'architecture': {
                    'num_layers': self.config.student_layers,
                    'hidden_size': self.config.student_hidden_size,
                    'num_attention_heads': self.config.student_attention_heads
                }
            },
            'inference_config': {
                'max_sequence_length': self.config.max_sequence_length,
                'batch_size': self.config.batch_size,
                'temperature': 0.7,
                'top_p': 0.9
            },
            'performance_benchmarks': self.compression_metrics,
            'deployment_instructions': {
                'framework': 'pytorch',
                'python_version': '>=3.8',
                'required_packages': ['torch', 'transformers'],
                'memory_requirements_mb': self._estimate_deployment_memory(),
                'recommended_hardware': self._recommend_hardware()
            }
        }
        
        return artifacts
    
    def _estimate_deployment_memory(self) -> float:
        """Estimate memory requirements for deployment"""
        model_params = sum(p.numel() for p in self.student_model.parameters())
        # Estimate: 4 bytes per parameter + overhead
        memory_mb = (model_params * 4) / (1024 * 1024) * 1.5
        return memory_mb
    
    def _recommend_hardware(self) -> str:
        """Recommend hardware for deployment"""
        memory_mb = self._estimate_deployment_memory()
        
        if memory_mb < 500:
            return "CPU (2+ cores, 2GB+ RAM)"
        elif memory_mb < 2000:
            return "CPU (4+ cores, 4GB+ RAM) or GPU (4GB+ VRAM)"
        else:
            return "GPU (8GB+ VRAM) recommended"
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate actual compression ratio achieved"""
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        return student_params / teacher_params
    
    def _calculate_avg_quality_score(self) -> float:
        """Calculate average quality score of synthetic data"""
        if not self.synthetic_data:
            return 0.0
        
        scores = [sample.get('quality_score', 0.0) for sample in self.synthetic_data]
        return sum(scores) / len(scores)
    
    def save_compressed_model(self, save_path: str):
        """Save the compressed model and artifacts"""
        print(f"💾 Saving compressed model to {save_path}")
        
        # Save model
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'config': self.config,
            'compression_metrics': self.compression_metrics
        }, f"{save_path}/compressed_model.pt")
        
        # Save deployment artifacts
        artifacts = self._generate_deployment_artifacts()
        with open(f"{save_path}/deployment_config.json", 'w') as f:
            json.dump(artifacts, f, indent=2)
        
        print(f"✅ Model saved successfully")
    
    def load_compressed_model(self, load_path: str):
        """Load a previously compressed model"""
        print(f"📂 Loading compressed model from {load_path}")
        
        checkpoint = torch.load(f"{load_path}/compressed_model.pt", map_location=self.device)
        
        # Recreate student model architecture
        self.config = checkpoint['config']
        teacher_stats = self._analyze_teacher_model()
        self.student_model = self._create_student_model(teacher_stats)
        
        # Load weights
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.compression_metrics = checkpoint.get('compression_metrics', {})
        
        print(f"✅ Model loaded successfully")
        
class SyntheticDistillationPipeline:
    """Complete pipeline for synthetic data generation and model distillation"""
    
    def __init__(self, 
                 teacher_model: nn.Module,
                 tokenizer,
                 target_compression: float = 0.5):
        
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.target_compression = target_compression
        
        # Pipeline components
        self.synthetic_generator = None
        self.student_model = None
        self.distillation_trainer = None
        
        # Generated data
        self.synthetic_datasets = {}
        
        # Statistics
        self.pipeline_stats = {
            'data_generation_time': 0.0,
            'distillation_time': 0.0,
            'total_synthetic_samples': 0,
            'final_model_size': 0,
            'compression_achieved': 0.0
        }
    
    def run_complete_pipeline(self,
                            synthetic_config: SyntheticDataConfig = None,
                            distillation_config: DistillationConfig = None,
                            training_config: TrainingConfig = None) -> Dict[str, Any]:
        """
        Run the complete synthetic data generation + distillation pipeline
        
        Args:
            synthetic_config: Configuration for synthetic data generation
            distillation_config: Configuration for knowledge distillation
            training_config: Configuration for training
            
        Returns:
            Dictionary with pipeline results
        """
        
        print("🌟 Starting Complete Synthetic Data + Distillation Pipeline")
        print("=" * 70)
        
        # Step 1: Generate synthetic data
        print("\n📊 Step 1: Synthetic Data Generation")
        synthetic_data = self.generate_synthetic_training_data(synthetic_config)
        
        # Step 2: Create student model
        print("\n🏗️ Step 2: Student Model Creation")
        self.create_student_model()
        
        # Step 3: Knowledge distillation
        print("\n🎓 Step 3: Knowledge Distillation")
        distilled_model = self.distill_knowledge(
            synthetic_data, 
            distillation_config, 
            training_config
        )
        
        # Step 4: Evaluation
        print("\n📈 Step 4: Model Evaluation")
        evaluation_results = self.evaluate_pipeline()
        
        # Step 5: Save results
        print("\n💾 Step 5: Save Results")
        self.save_pipeline_results()
        
        results = {
            'synthetic_data': synthetic_data,
            'student_model': distilled_model,
            'evaluation_results': evaluation_results,
            'pipeline_stats': self.pipeline_stats
        }
        
        print("\n🎉 Pipeline completed successfully!")
        self._print_pipeline_summary()
        
        return results
    
    def generate_synthetic_training_data(self, config: SyntheticDataConfig = None) -> List[Dict[str, Any]]:
        """Generate comprehensive synthetic training data"""
        
        start_time = time.time()
        
        if config is None:
            config = SyntheticDataConfig(
                num_samples=2000,
                temperature=0.8,
                use_quality_filter=True,
                use_diversity_filter=True,
                include_instructions=True,
                include_conversations=True,
                include_reasoning=True,
                include_creative=True
            )
        
        # Create synthetic data generator
        self.synthetic_generator = create_synthetic_data_generator(
            teacher_model=self.teacher_model,
            tokenizer=self.tokenizer,
            config=config
        )
        
        print(f"📋 Generating {config.num_samples} synthetic samples...")
        
        # Generate general synthetic data
        general_data = self.synthetic_generator.generate_synthetic_dataset()
        self.synthetic_datasets['general'] = general_data
        
        # Generate specialized data
        specialized_data = self._generate_specialized_data(config)
        
        # Combine all synthetic data
        all_synthetic_data = general_data + specialized_data
        
        generation_time = time.time() - start_time
        self.pipeline_stats['data_generation_time'] = generation_time
        self.pipeline_stats['total_synthetic_samples'] = len(all_synthetic_data)
        
        print(f"✅ Generated {len(all_synthetic_data)} synthetic samples in {generation_time:.2f}s")
        
        return all_synthetic_data
    
    def _generate_specialized_data(self, config: SyntheticDataConfig) -> List[Dict[str, Any]]:
        """Generate specialized synthetic data for specific domains"""
        
        specialized_data = []
        
        # Math data generation
        if config.include_reasoning:
            print("🧮 Generating mathematical reasoning data...")
            math_generator = MathDataGenerator(self.teacher_model, self.tokenizer)
            math_data = math_generator.generate_math_problems(num_problems=200)
            
            # Convert to standard format
            for item in math_data:
                specialized_data.append({
                    'prompt': item['problem'],
                    'response': item['solution'],
                    'data_type': 'mathematical_reasoning',
                    'difficulty': item['difficulty'],
                    'problem_type': item['problem_type']
                })
            
            self.synthetic_datasets['math'] = math_data
        
        # Code data generation
        if config.include_instructions:
            print("💻 Generating code instruction data...")
            code_generator = CodeDataGenerator(self.teacher_model, self.tokenizer)
            code_data = code_generator.generate_coding_problems(num_problems=150)
            
            # Convert to standard format
            for item in code_data:
                specialized_data.append({
                    'prompt': item['problem'],
                    'response': item['solution'],
                    'data_type': 'code_instruction',
                    'language': item['language'],
                    'problem_type': item['problem_type']
                })
            
            self.synthetic_datasets['code'] = code_data
        
        # Domain-specific conversation data
        conversation_data = self._generate_domain_conversations()
        specialized_data.extend(conversation_data)
        self.synthetic_datasets['domain_conversations'] = conversation_data
        
        print(f"📚 Generated {len(specialized_data)} specialized samples")
        
        return specialized_data
    
    def _generate_domain_conversations(self) -> List[Dict[str, Any]]:
        """Generate domain-specific conversation data"""
        
        domains = {
            'science': [
                "Explain the process of photosynthesis",
                "What causes climate change?",
                "How do vaccines work?",
                "Describe the structure of an atom"
            ],
            'technology': [
                "What is artificial intelligence?",
                "How does blockchain technology work?",
                "Explain quantum computing",
                "What are the benefits of cloud computing?"
            ],
            'health': [
                "What are the benefits of regular exercise?",
                "How can I maintain a healthy diet?",
                "What is the importance of sleep?",
                "How do I manage stress effectively?"
            ]
        }
        
        conversation_data = []
        
        for domain, prompts in domains.items():
            for prompt in prompts:
                try:
                    # Generate response using teacher model
                    from ..inference.engine import GenerationConfig
                    from ..inference.sampling import SamplingConfig
                    
                    gen_config = GenerationConfig(
                        sampling=SamplingConfig(temperature=0.7, top_p=0.9),
                        max_new_tokens=200
                    )
                    
                    input_ids = self.tokenizer.encode(f"Human: {prompt}\nAssistant:", return_tensors='pt')
                    
                    # Use teacher model to generate
                    with torch.no_grad():
                        outputs = self.teacher_model.generate(
                            input_ids,
                            max_new_tokens=200,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][input_ids.size(1):], 
                        skip_special_tokens=True
                    ).strip()
                    
                    conversation_data.append({
                        'prompt': f"Human: {prompt}\nAssistant:",
                        'response': response,
                        'data_type': f'{domain}_conversation',
                        'domain': domain
                    })
                    
                except Exception as e:
                    print(f"⚠️ Failed to generate conversation for '{prompt}': {e}")
                    continue
        
        return conversation_data
    
    def create_student_model(self):
        """Create smaller student model based on teacher"""
        
        print(f"🏗️ Creating student model with {self.target_compression}x compression...")
        
        # Get teacher model parameters
        teacher_config = getattr(self.teacher_model, 'config', None)
        
        if teacher_config:
            # Create smaller config for student
            student_config = copy.deepcopy(teacher_config)
            
            # Reduce model size based on compression ratio
            student_config.hidden_size = int(teacher_config.hidden_size * self.target_compression)
            student_config.intermediate_size = int(teacher_config.intermediate_size * self.target_compression)
            student_config.num_hidden_layers = int(teacher_config.num_hidden_layers * self.target_compression)
            student_config.num_attention_heads = max(1, int(teacher_config.num_attention_heads * self.target_compression))
            
            # Ensure dimensions are compatible
            student_config.hidden_size = (student_config.hidden_size // student_config.num_attention_heads) * student_config.num_attention_heads
            
            print(f"Student model: {student_config.num_hidden_layers} layers, {student_config.hidden_size} hidden size")
            
            # Create student model with smaller config
            from ..models.llama import LlamaForCausalLM
            self.student_model = LlamaForCausalLM(student_config)
        else:
            # Fallback: create a smaller version of teacher
            self.student_model = copy.deepcopy(self.teacher_model)
            # In practice, you'd implement actual model compression here
        
        # Calculate actual compression achieved
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        actual_compression = student_params / teacher_params
        self.pipeline_stats['compression_achieved'] = actual_compression
        self.pipeline_stats['final_model_size'] = student_params
        
        print(f"✅ Student model created: {student_params:,} parameters ({actual_compression:.2f}x of teacher)")
    
    def distill_knowledge(self,
                         synthetic_data: List[Dict[str, Any]],
                         distillation_config: DistillationConfig = None,
                         training_config: TrainingConfig = None) -> nn.Module:
        """Perform knowledge distillation using synthetic data"""
        
        start_time = time.time()
        
        if distillation_config is None:
            distillation_config = DistillationConfig(
                temperature=4.0,
                alpha=0.7,
                beta=0.3,
                use_feature_distillation=True,
                use_progressive=True,
                progressive_epochs=[0, 3, 6],
                progressive_temperatures=[8.0, 4.0, 2.0]
            )
        
        if training_config is None:
            training_config = TrainingConfig(
                epochs=10,
                batch_size=8,
                optimizer=TrainingConfig.OptimizerConfig(lr=5e-5),
                logging=TrainingConfig.LoggingConfig(log_every=50)
            )
        
        # Create dataset from synthetic data
        train_dataset = SyntheticDataset(synthetic_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
        
        # Create distillation trainer
        if distillation_config.use_progressive:
            self.distillation_trainer = ProgressiveDistillationTrainer(
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                config=training_config,
                distillation_config=distillation_config,
                train_dataloader=train_loader
            )
        else:
            self.distillation_trainer = create_distillation_trainer(
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                train_dataloader=train_loader,
                config=training_config,
                distillation_config=distillation_config
            )
        
        print(f"🎓 Starting knowledge distillation for {training_config.epochs} epochs...")
        
        # Train student model
        self.distillation_trainer.fit()
        
        distillation_time = time.time() - start_time
        self.pipeline_stats['distillation_time'] = distillation_time
        
        print(f"✅ Knowledge distillation completed in {distillation_time:.2f}s")
        
        return self.student_model
    
    def evaluate_pipeline(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of the pipeline"""
        
        print("📊 Evaluating pipeline effectiveness...")
        
        # Create test prompts
        test_prompts = [
            "Explain the concept of machine learning",
            "Solve: 3x + 7 = 22",
            "Write a Python function to reverse a string",
            "What are the causes of global warming?",
            "How do you maintain a healthy lifestyle?"
        ]
        
        evaluation_results = {
            'test_prompts': test_prompts,
            'teacher_responses': [],
            'student_responses': [],
            'response_similarities': [],
            'generation_times': {'teacher': [], 'student': []},
            'model_sizes': {
                'teacher_params': sum(p.numel() for p in self.teacher_model.parameters()),
                'student_params': sum(p.numel() for p in self.student_model.parameters())
            }
        }
        
        # Generate responses from both models
        for prompt in test_prompts:
            # Teacher response
            teacher_response, teacher_time = self._generate_timed_response(self.teacher_model, prompt)
            evaluation_results['teacher_responses'].append(teacher_response)
            evaluation_results['generation_times']['teacher'].append(teacher_time)
            
            # Student response
            student_response, student_time = self._generate_timed_response(self.student_model, prompt)
            evaluation_results['student_responses'].append(student_response)
            evaluation_results['generation_times']['student'].append(student_time)
            
            # Calculate similarity
            similarity = self._calculate_response_similarity(teacher_response, student_response)
            evaluation_results['response_similarities'].append(similarity)
        
        # Calculate summary metrics
        evaluation_results['summary'] = {
            'avg_similarity': np.mean(evaluation_results['response_similarities']),
            'avg_teacher_time': np.mean(evaluation_results['generation_times']['teacher']),
            'avg_student_time': np.mean(evaluation_results['generation_times']['student']),
            'speedup_factor': np.mean(evaluation_results['generation_times']['teacher']) / np.mean(evaluation_results['generation_times']['student']),
            'compression_ratio': evaluation_results['model_sizes']['student_params'] / evaluation_results['model_sizes']['teacher_params']
        }
        
        print(f"📈 Evaluation Results:")
        print(f"  Average Similarity: {evaluation_results['summary']['avg_similarity']:.3f}")
        print(f"  Speedup Factor: {evaluation_results['summary']['speedup_factor']:.2f}x")
        print(f"  Compression Ratio: {evaluation_results['summary']['compression_ratio']:.3f}")
        
        return evaluation_results
    
    def _generate_timed_response(self, model: nn.Module, prompt: str) -> Tuple[str, float]:
        """Generate response and measure time"""
        
        start_time = time.time()
        
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][input_ids.size(1):], 
                skip_special_tokens=True
            ).strip()
            
        except Exception as e:
            response = f"[Generation failed: {e}]"
        
        generation_time = time.time() - start_time
        
        return response, generation_time
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses"""
        
        # Simple token-based similarity
        tokens1 = set(response1.lower().split())
        tokens2 = set(response2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def save_pipeline_results(self, save_dir: str = "./synthetic_distillation_results"):
        """Save all pipeline results"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save synthetic datasets
        for dataset_name, dataset in self.synthetic_datasets.items():
            dataset_file = save_path / f"synthetic_{dataset_name}_data.json"
            with open(dataset_file, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
        
        # Save student model
        if self.student_model:
            model_path = save_path / "distilled_student_model.pt"
            torch.save(self.student_model.state_dict(), model_path)
        
        # Save pipeline statistics
        stats_file = save_path / "pipeline_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.pipeline_stats, f, indent=2, default=str)
        
        # Save configurations
        config_file = save_path / "pipeline_config.json"
        pipeline_config = {
            'target_compression': self.target_compression,
            'teacher_model_info': {
                'parameters': sum(p.numel() for p in self.teacher_model.parameters()),
                'architecture': self.teacher_model.__class__.__name__
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(pipeline_config, f, indent=2, default=str)
        
        print(f"💾 Pipeline results saved to: {save_dir}")
    
    def _print_pipeline_summary(self):
        """Print comprehensive pipeline summary"""
        
        print("\n📋 Pipeline Summary")
        print("=" * 50)
        print(f"🕒 Data Generation Time: {self.pipeline_stats['data_generation_time']:.2f}s")
        print(f"🕒 Distillation Time: {self.pipeline_stats['distillation_time']:.2f}s")
        print(f"📊 Total Synthetic Samples: {self.pipeline_stats['total_synthetic_samples']:,}")
        print(f"🗜️ Compression Achieved: {self.pipeline_stats['compression_achieved']:.3f}")
        print(f"📦 Final Model Size: {self.pipeline_stats['final_model_size']:,} parameters")
        
        print(f"\n📈 Performance Gains:")
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = self.pipeline_stats['final_model_size']
        size_reduction = (1 - student_params / teacher_params) * 100
        
        print(f"  Model Size Reduction: {size_reduction:.1f}%")
        print(f"  Memory Savings: ~{size_reduction:.1f}%")
        print(f"  Potential Speedup: ~{1/self.pipeline_stats['compression_achieved']:.1f}x")

class SyntheticDataset(Dataset):
    """Dataset wrapper for synthetic data"""
    
    def __init__(self, synthetic_data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = synthetic_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine prompt and response
        full_text = f"{item['prompt']} {item['response']}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For language modeling
        }

# Advanced pipeline variants
class MultiTeacherDistillationPipeline(SyntheticDistillationPipeline):
    """Pipeline with multiple teacher models for ensemble distillation"""
    
    def __init__(self, teacher_models: List[nn.Module], tokenizer, target_compression: float = 0.5):
        # Use first teacher as primary
        super().__init__(teacher_models[0], tokenizer, target_compression)
        self.teacher_models = teacher_models
        self.ensemble_size = len(teacher_models)
    
    def generate_synthetic_training_data(self, config: SyntheticDataConfig = None) -> List[Dict[str, Any]]:
        """Generate synthetic data using ensemble of teachers"""
        
        all_synthetic_data = []
        
        # Generate data from each teacher
        for i, teacher in enumerate(self.teacher_models):
            print(f"📚 Generating data from teacher {i+1}/{self.ensemble_size}")
            
            # Temporarily set current teacher
            original_teacher = self.teacher_model
            self.teacher_model = teacher
            
            # Generate data
            teacher_data = super().generate_synthetic_training_data(config)
            
            # Tag data with teacher source
            for item in teacher_data:
                item['teacher_source'] = i
            
            all_synthetic_data.extend(teacher_data)
            
            # Restore original teacher
            self.teacher_model = original_teacher
        
        return all_synthetic_data

class DomainAdaptiveDistillationPipeline(SyntheticDistillationPipeline):
    """Pipeline with domain-specific distillation"""
    
    def __init__(self, *args, domains: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.domains = domains or ['general', 'math', 'code', 'science']
        self.domain_weights = {domain: 1.0 for domain in self.domains}
    
    def distill_knowledge(self, synthetic_data: List[Dict[str, Any]], *args, **kwargs) -> nn.Module:
        """Domain-adaptive knowledge distillation"""
        
        # Group data by domain
        domain_data = {}
        for item in synthetic_data:
            domain = item.get('data_type', 'general')
            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append(item)
        
        # Progressive distillation by domain
        for domain, data in domain_data.items():
            if domain in self.domains:
                print(f"🎯 Distilling knowledge for domain: {domain}")
                
                # Create domain-specific dataset
                domain_dataset = SyntheticDataset(data, self.tokenizer)
                domain_loader = DataLoader(domain_dataset, batch_size=8, shuffle=True)
                
                # Domain-specific distillation (simplified)
                # In practice, you'd adjust distillation parameters per domain
                
        return super().distill_knowledge(synthetic_data, *args, **kwargs)

# Complete example function
def run_synthetic_distillation_example():
    """Run complete synthetic data + distillation example"""
    
    print("🌟 Running Complete Synthetic Data + Distillation Example")
    print("=" * 70)
    
    # Create dummy teacher model
    teacher_model = create_llama_7b()
    
    # Dummy tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            self.pad_token_id = 0
        
        def encode(self, text, return_tensors=None):
            tokens = [hash(word) % 32000 for word in text.split()[:100]]
            if return_tensors == 'pt':
                return torch.tensor(tokens).unsqueeze(0)
            return tokens
        
        def decode(self, tokens, skip_special_tokens=True):
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            return f"[Generated response from {len(tokens)} tokens]"
        
        def __call__(self, text, **kwargs):
            tokens = self.encode(text)
            max_length = kwargs.get('max_length', 512)
            
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
            
            return {
                'input_ids': torch.tensor(tokens).unsqueeze(0),
                'attention_mask': torch.ones(1, len(tokens))
            }
    
    tokenizer = DummyTokenizer()
    
    # Create pipeline
    pipeline = SyntheticDistillationPipeline(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        target_compression=0.3  # 30% of original size
    )
    
    # Configure synthetic data generation
    synthetic_config = SyntheticDataConfig(
        num_samples=500,  # Small for demo
        temperature=0.8,
        use_quality_filter=True,
        include_instructions=True,
        include_conversations=True,
        include_reasoning=True
    )
    
    # Configure distillation
    distillation_config = DistillationConfig(
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
        use_feature_distillation=True,
        use_progressive=True
    )
    
    # Configure training
    training_config = TrainingConfig(
        epochs=3,  # Short for demo
        batch_size=4,
        optimizer=TrainingConfig.OptimizerConfig(lr=5e-5)
    )
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            synthetic_config=synthetic_config,
            distillation_config=distillation_config,
            training_config=training_config
        )
        
        print("\n🎉 Example completed successfully!")
        
        # Print results summary
        print(f"\n📊 Results Summary:")
        print(f"  Synthetic samples generated: {len(results['synthetic_data'])}")
        print(f"  Student model parameters: {results['pipeline_stats']['final_model_size']:,}")
        print(f"  Compression achieved: {results['pipeline_stats']['compression_achieved']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Export components
__all__ = [
    'SyntheticDistillationPipeline',
    'MultiTeacherDistillationPipeline', 
    'DomainAdaptiveDistillationPipeline',
    'SyntheticDataset',
    'run_synthetic_distillation_example'
]