# training_infra/advanced.py
"""Integration of advanced features: LLaMA, MoE, and Inference"""

from .models.llama import LlamaForCausalLM, LlamaConfig, create_llama_7b, create_llama2_7b
from .models.moe import MoEConfig, MoELayer, SparseMoEBlock, MoETransformerLayer, MoELlamaLayer
from .inference.engine import InferenceEngine, GenerationConfig, generate_text, chat_generate
from .inference.sampling import SamplingConfig, create_sampler
from .trainer import Trainer
from .config import TrainingConfig
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

class AdvancedLlamaTrainer(Trainer):
    """Specialized trainer for LLaMA models with advanced features"""
    
    def __init__(self, *args, use_moe: bool = False, moe_config: Optional[MoEConfig] = None, **kwargs):
        self.use_moe = use_moe
        self.moe_config = moe_config
        self.total_aux_loss = 0.0
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, batch):
        """Compute loss including MoE auxiliary losses"""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, labels = batch
        else:
            input_ids = batch['input_ids']
            labels = batch.get('labels', input_ids)
        
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        # Add MoE auxiliary losses if using MoE
        if self.use_moe and hasattr(self.model, 'get_aux_losses'):
            aux_losses = self.model.get_aux_losses()
            total_aux_loss = sum(aux_losses.values())
            loss = loss + total_aux_loss
            self.total_aux_loss = total_aux_loss.item()
        
        return loss
    
    def train_step(self, batch):
        """Training step with MoE loss tracking"""
        loss = super().train_step(batch)
        
        # Log MoE specific metrics
        if self.use_moe and self.global_step % self.config.logging.log_every == 0 and self.is_main_process:
            moe_metrics = {
                'train/aux_loss': self.total_aux_loss,
                'train/main_loss': loss.item() - self.total_aux_loss,
            }
            self.logger.log_metrics(moe_metrics, self.global_step)
        
        return loss

class LlamaMoEModel(nn.Module):
    """LLaMA model with Mixture of Experts layers"""
    
    def __init__(self, llama_config: LlamaConfig, moe_config: MoEConfig, moe_layers: List[int]):
        super().__init__()
        self.config = llama_config
        self.moe_config = moe_config
        self.moe_layers = set(moe_layers)
        
        # Create base LLaMA model
        from .models.llama import LlamaModel
        self.llama = LlamaModel(llama_config)
        
        # Replace specified layers with MoE layers
        self._replace_layers_with_moe()
        
        # Language modeling head
        self.lm_head = nn.Linear(llama_config.hidden_size, llama_config.vocab_size, bias=False)
        
        # Track auxiliary losses
        self.aux_losses = {}
    
    def _replace_layers_with_moe(self):
        """Replace specified layers with MoE layers"""
        new_layers = nn.ModuleList()
        
        for i, layer in enumerate(self.llama.layers):
            if i in self.moe_layers:
                # Create MoE layer
                moe_layer = MoELlamaLayer(layer.self_attn, self.moe_config)
                new_layers.append(moe_layer)
            else:
                new_layers.append(layer)
        
        self.llama.layers = new_layers
    
    def forward(self, input_ids, labels=None, **kwargs):
        """Forward pass with MoE loss collection"""
        self.aux_losses = {}
        
        # LLaMA forward pass
        outputs = self.llama(input_ids, **kwargs)
        hidden_states = outputs['last_hidden_state']
        
        # Collect auxiliary losses from MoE layers
        for i, layer in enumerate(self.llama.layers):
            if hasattr(layer, 'moe_block'):
                layer_aux_losses = getattr(layer, '_last_aux_losses', {})
                for loss_name, loss_value in layer_aux_losses.items():
                    full_loss_name = f"layer_{i}_{loss_name}"
                    self.aux_losses[full_loss_name] = loss_value
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'aux_losses': self.aux_losses
        }
    
    def get_aux_losses(self):
        """Get auxiliary losses for training"""
        return self.aux_losses

def create_llama_moe_7b(moe_layers: List[int] = None, num_experts: int = 8) -> LlamaMoEModel:
    """Create LLaMA 7B with MoE layers"""
    if moe_layers is None:
        # Use MoE in every 4th layer by default
        moe_layers = list(range(3, 32, 4))
    
    llama_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )
    
    moe_config = MoEConfig(
        hidden_size=4096,
        num_experts=num_experts,
        num_experts_per_tok=2,
        intermediate_size=11008,
        router_aux_loss_coef=0.01
    )
    
    return LlamaMoEModel(llama_config, moe_config, moe_layers)

class AdvancedInferenceEngine(InferenceEngine):
    """Extended inference engine with LLaMA and MoE support"""
    
    def __init__(self, model, tokenizer=None, device=None):
        super().__init__(model, tokenizer, device)
        self.is_moe_model = self._check_if_moe_model()
    
    def _check_if_moe_model(self) -> bool:
        """Check if model has MoE layers"""
        for module in self.model.modules():
            if hasattr(module, 'moe_block') or 'moe' in module.__class__.__name__.lower():
                return True
        return False
    
    def generate_with_expert_analysis(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        analyze_experts: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate with expert utilization analysis for MoE models"""
        if not self.is_moe_model or not analyze_experts:
            return super().generate(input_ids, config, **kwargs)
        
        # Enable expert tracking
        expert_stats = {'expert_usage': [], 'load_balancing_losses': []}
        
        # Custom generation with expert tracking
        # This would require modifications to the generation loop
        # to track expert usage patterns
        
        result = super().generate(input_ids, config, **kwargs)
        result['expert_analysis'] = expert_stats
        
        return result

# Training configurations for advanced models
def create_llama_training_config() -> TrainingConfig:
    """Create optimized training config for LLaMA models"""
    return TrainingConfig(
        model_name="llama_7b",
        epochs=3,
        batch_size=8,
        gradient_accumulation_steps=16,  # Effective batch size: 128
        max_grad_norm=1.0,
        
        optimizer=TrainingConfig.OptimizerConfig(
            name="adamw",
            lr=2e-5,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        ),
        
        scheduler=TrainingConfig.SchedulerConfig(
            name="cosine",
            warmup_steps=2000,
            min_lr=2e-6
        ),
        
        use_amp=True,
        amp_dtype="bfloat16",
        
        logging=TrainingConfig.LoggingConfig(
            log_every=50,
            use_tensorboard=True,
            use_wandb=True,
            wandb_project="llama_training"
        ),
        
        checkpoint=TrainingConfig.CheckpointConfig(
            save_every=1000,
            monitor="val_loss",
            mode="min",
            keep_last=3
        )
    )

def create_moe_training_config() -> TrainingConfig:
    """Create optimized training config for MoE models"""
    config = create_llama_training_config()
    
    # MoE specific adjustments
    config.batch_size = 4  # Smaller batch size for MoE
    config.gradient_accumulation_steps = 32  # Maintain effective batch size
    config.optimizer.lr = 1e-5  # Lower learning rate for MoE
    config.max_grad_norm = 0.5  # Tighter gradient clipping
    
    return config

# Inference configurations
def create_llama_inference_config() -> GenerationConfig:
    """Create optimized inference config for LLaMA models"""
    return GenerationConfig(
        sampling=SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            frequency_penalty=0.1
        ),
        max_new_tokens=512,
        use_cache=True
    )

def create_moe_inference_config() -> GenerationConfig:
    """Create optimized inference config for MoE models"""
    config = create_llama_inference_config()
    # MoE models might benefit from different sampling parameters
    config.sampling.temperature = 0.8
    config.sampling.top_k = 50
    return config

# Example usage functions
def train_llama_model(train_dataloader, val_dataloader, model_size="7b", use_moe=False):
    """Complete example of training a LLaMA model"""
    
    # Create model
    if use_moe:
        if model_size == "7b":
            model = create_llama_moe_7b()
        else:
            raise ValueError(f"MoE not implemented for {model_size}")
        config = create_moe_training_config()
        trainer_class = AdvancedLlamaTrainer
        trainer_kwargs = {'use_moe': True, 'moe_config': model.moe_config}
    else:
        if model_size == "7b":
            model = create_llama_7b()
        elif model_size == "7b_v2":
            model = create_llama2_7b()
        else:
            raise ValueError(f"Model size {model_size} not supported")
        config = create_llama_training_config()
        trainer_class = AdvancedLlamaTrainer
        trainer_kwargs = {'use_moe': False}
    
    # Create trainer
    trainer = trainer_class(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **trainer_kwargs
    )
    
    # Train
    trainer.fit()
    
    return trainer

def setup_llama_inference(model_path: str, tokenizer_path: str = None, use_moe: bool = False):
    """Setup LLaMA model for inference"""
    
    # Load model
    if use_moe:
        # Load MoE model
        model = torch.load(model_path, map_location='cpu')
        config = create_moe_inference_config()
    else:
        # Load regular LLaMA model
        model = torch.load(model_path, map_location='cpu')
        config = create_llama_inference_config()
    
    # Load tokenizer if available
    tokenizer = None
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except ImportError:
            print("Transformers not available, tokenizer not loaded")
    
    # Create inference engine
    if use_moe:
        engine = AdvancedInferenceEngine(model, tokenizer)
    else:
        engine = InferenceEngine(model, tokenizer)
    
    return engine, config

def interactive_chat(engine: InferenceEngine, config: GenerationConfig, tokenizer):
    """Interactive chat with LLaMA model"""
    
    print("LLaMA Chat Interface (type 'quit' to exit)")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if not user_input:
            continue
        
        # Add to conversation
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        try:
            response = chat_generate(
                engine.model,
                tokenizer,
                conversation_history,
                max_length=config.max_length,
                temperature=config.sampling.temperature,
                top_p=config.sampling.top_p
            )
            
            print(f"Assistant: {response}")
            conversation_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"Error generating response: {e}")

# Utility functions
def analyze_moe_model(model: LlamaMoEModel, dataloader, num_batches: int = 10):
    """Analyze expert utilization in MoE model"""
    model.eval()
    expert_usage = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # Forward pass
            outputs = model(batch['input_ids'])
            aux_losses = outputs.get('aux_losses', {})
            
            # Collect expert usage statistics
            for loss_name, loss_value in aux_losses.items():
                if loss_name not in expert_usage:
                    expert_usage[loss_name] = []
                expert_usage[loss_name].append(loss_value.item())
    
    # Calculate statistics
    stats = {}
    for loss_name, values in expert_usage.items():
        stats[loss_name] = {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
    
    return stats

def benchmark_inference_methods(model, tokenizer, test_prompts: List[str]):
    """Benchmark different inference methods"""
    
    engine = AdvancedInferenceEngine(model, tokenizer)
    
    # Different sampling configurations
    configs = {
        'greedy': GenerationConfig(sampling=SamplingConfig(do_sample=False)),
        'temperature': GenerationConfig(sampling=SamplingConfig(temperature=0.8)),
        'nucleus': GenerationConfig(sampling=SamplingConfig(temperature=0.8, top_p=0.9)),
        'top_k': GenerationConfig(sampling=SamplingConfig(temperature=0.8, top_k=50)),
        'mirostat': GenerationConfig(sampling=SamplingConfig(mirostat_mode=2, mirostat_tau=5.0))
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"Benchmarking {config_name}...")
        
        # Benchmark performance
        input_ids = tokenizer.encode(test_prompts[0], return_tensors='pt')
        benchmark_results = engine.benchmark(input_ids, config, num_runs=5)
        
        # Generate samples
        samples = []
        for prompt in test_prompts[:3]:  # Test first 3 prompts
            response = generate_text(
                model, tokenizer, prompt,
                max_length=config.max_length,
                **config.sampling.__dict__
            )
            samples.append(response)
        
        results[config_name] = {
            'performance': benchmark_results,
            'samples': samples
        }
    
    return results

# Advanced model architectures
class LlamaWithLoRA(nn.Module):
    """LLaMA model with LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
    
    def __init__(self, base_model: LlamaForCausalLM, lora_rank: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Add LoRA adapters to attention layers
        self._add_lora_adapters()
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _add_lora_adapters(self):
        """Add LoRA adapters to attention layers"""
        for layer in self.base_model.model.layers:
            attn = layer.self_attn
            
            # Add LoRA to query and value projections
            self._add_lora_to_linear(attn, 'q_proj')
            self._add_lora_to_linear(attn, 'v_proj')
    
    def _add_lora_to_linear(self, module, linear_name):
        """Add LoRA adapter to a linear layer"""
        linear_layer = getattr(module, linear_name)
        
        # Create LoRA components
        lora_a = nn.Linear(linear_layer.in_features, self.lora_rank, bias=False)
        lora_b = nn.Linear(self.lora_rank, linear_layer.out_features, bias=False)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_b.weight)
        
        # Register as modules
        setattr(module, f'{linear_name}_lora_a', lora_a)
        setattr(module, f'{linear_name}_lora_b', lora_b)
        
        # Replace forward method
        original_forward = linear_layer.forward
        
        def lora_forward(x):
            base_output = original_forward(x)
            lora_output = lora_b(lora_a(x))
            scaling = self.lora_alpha / self.lora_rank
            return base_output + scaling * lora_output
        
        linear_layer.forward = lora_forward
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

class HybridMoEModel(nn.Module):
    """Hybrid model combining dense and sparse (MoE) layers"""
    
    def __init__(self, llama_config: LlamaConfig, moe_config: MoEConfig, 
                 dense_layers: List[int], moe_layers: List[int]):
        super().__init__()
        self.config = llama_config
        self.moe_config = moe_config
        self.dense_layers = set(dense_layers)
        self.moe_layers = set(moe_layers)
        
        # Create base model structure
        from .models.llama import LlamaModel, LlamaMLP
        self.llama = LlamaModel(llama_config)
        
        # Replace layers based on configuration
        new_layers = nn.ModuleList()
        for i, layer in enumerate(self.llama.layers):
            if i in self.moe_layers:
                # Replace with MoE layer
                moe_layer = MoELlamaLayer(layer.self_attn, self.moe_config)
                new_layers.append(moe_layer)
            elif i in self.dense_layers:
                # Keep as dense layer but potentially modify
                new_layers.append(layer)
            else:
                # Default behavior
                new_layers.append(layer)
        
        self.llama.layers = new_layers
        self.lm_head = nn.Linear(llama_config.hidden_size, llama_config.vocab_size, bias=False)
    
    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.llama(input_ids, **kwargs)
        hidden_states = outputs['last_hidden_state']
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'loss': loss, 'logits': logits, 'hidden_states': hidden_states}

# Advanced training strategies
class CurriculumLearningTrainer(AdvancedLlamaTrainer):
    """Trainer with curriculum learning support"""
    
    def __init__(self, *args, curriculum_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_schedule = curriculum_schedule or self._default_curriculum()
        self.current_difficulty = 0.1
    
    def _default_curriculum(self):
        """Default curriculum schedule"""
        return {
            0: 0.1,      # Start with 10% difficulty
            1000: 0.3,   # Increase to 30% at step 1000
            5000: 0.6,   # 60% at step 5000
            10000: 1.0   # Full difficulty at step 10000
        }
    
    def train_step(self, batch):
        """Training step with curriculum learning"""
        # Update difficulty based on schedule
        for step_threshold in sorted(self.curriculum_schedule.keys(), reverse=True):
            if self.global_step >= step_threshold:
                self.current_difficulty = self.curriculum_schedule[step_threshold]
                break
        
        # Apply curriculum to batch (implementation depends on your data)
        # This is a placeholder - you'd implement actual curriculum logic here
        filtered_batch = self._apply_curriculum(batch, self.current_difficulty)
        
        return super().train_step(filtered_batch)
    
    def _apply_curriculum(self, batch, difficulty):
        """Apply curriculum learning to batch"""
        # Placeholder implementation
        # In practice, you might:
        # - Filter examples by length
        # - Select examples by complexity score
        # - Gradually introduce harder concepts
        return batch

class GradientAccumulationTrainer(AdvancedLlamaTrainer):
    """Enhanced trainer with smart gradient accumulation"""
    
    def __init__(self, *args, adaptive_accumulation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_accumulation = adaptive_accumulation
        self.loss_history = []
    
    def train_step(self, batch):
        """Training step with adaptive gradient accumulation"""
        loss = self.compute_loss(batch)
        
        # Track loss for adaptive accumulation
        if self.adaptive_accumulation:
            self.loss_history.append(loss.item())
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)
        
        # Scale loss by gradient accumulation steps
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Adaptive gradient accumulation
        if self.adaptive_accumulation and len(self.loss_history) > 10:
            # Adjust accumulation based on loss stability
            loss_std = torch.std(torch.tensor(self.loss_history[-10:]))
            if loss_std > 0.1:  # High variance
                self.config.gradient_accumulation_steps = min(32, self.config.gradient_accumulation_steps + 1)
            elif loss_std < 0.01:  # Low variance
                self.config.gradient_accumulation_steps = max(1, self.config.gradient_accumulation_steps - 1)
        
        return loss * self.config.gradient_accumulation_steps  # Return unscaled loss for logging

# Export all advanced features
__all__ = [
    # Models
    'LlamaMoEModel',
    'LlamaWithLoRA', 
    'HybridMoEModel',
    'create_llama_moe_7b',
    
    # Training
    'AdvancedLlamaTrainer',
    'CurriculumLearningTrainer',
    'GradientAccumulationTrainer',
    
    # Inference
    'AdvancedInferenceEngine',
    
    # Configurations
    'create_llama_training_config',
    'create_moe_training_config',
    'create_llama_inference_config',
    'create_moe_inference_config',
    
    # Utilities
    'train_llama_model',
    'setup_llama_inference',
    'interactive_chat',
    'analyze_moe_model',
    'benchmark_inference_methods',
]