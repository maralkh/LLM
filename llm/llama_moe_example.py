# examples/complete_llama_moe_example.py
"""
Complete example showing how to use all advanced features:
- LLaMA model training
- Mixture of Experts (MoE)  
- Advanced sampling methods
- Inference engine
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path

# Import our training infrastructure
from training_infra.advanced import (
    train_llama_model, 
    create_llama_moe_7b,
    create_llama_training_config,
    create_moe_training_config,
    AdvancedLlamaTrainer,
    AdvancedInferenceEngine,
    interactive_chat
)

from training_infra.inference.sampling import SamplingConfig
from training_infra.inference.engine import GenerationConfig, generate_text
from training_infra.models.llama import create_llama_7b

class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration"""
    
    def __init__(self, texts, tokenizer=None, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.tokenizer:
            # If tokenizer available, tokenize properly
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze()
            return {"input_ids": input_ids, "labels": input_ids.clone()}
        else:
            # Simple token simulation for demo
            tokens = [hash(word) % 32000 for word in text.split()[:self.max_length]]
            tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad
            input_ids = torch.tensor(tokens[:self.max_length], dtype=torch.long)
            return {"input_ids": input_ids, "labels": input_ids.clone()}

def create_demo_data():
    """Create demonstration dataset"""
    
    # Sample texts for training
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        "Python is a powerful programming language for AI development.",
        "Large language models have revolutionized natural language processing.",
        "Training neural networks requires careful consideration of hyperparameters.",
        "Transformer architectures have become the foundation of modern NLP.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Fine-tuning pre-trained models is an effective transfer learning strategy.",
    ] * 100  # Repeat for more training data
    
    return texts

def demonstrate_basic_llama_training():
    """Demonstrate basic LLaMA model training"""
    
    print("🦙 Demonstrating Basic LLaMA Training")
    print("=" * 50)
    
    # Create dataset
    texts = create_demo_data()
    train_texts = texts[:600]
    val_texts = texts[600:800]
    
    train_dataset = SimpleTextDataset(train_texts, max_length=128)
    val_dataset = SimpleTextDataset(val_texts, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Create model
    model = create_llama_7b()
    
    # Create training config
    config = create_llama_training_config()
    config.epochs = 2  # Short training for demo
    config.logging.log_every = 10
    config.eval_every = 50
    
    # Train
    trainer = AdvancedLlamaTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        use_moe=False
    )
    
    print("Starting basic LLaMA training...")
    trainer.fit()
    print("✅ Basic training completed!")
    
    return trainer.model

def demonstrate_moe_training():
    """Demonstrate Mixture of Experts training"""
    
    print("\n🧠 Demonstrating MoE Training")
    print("=" * 50)
    
    # Create dataset
    texts = create_demo_data()
    train_texts = texts[:400]  # Smaller dataset for MoE demo
    val_texts = texts[400:500]
    
    train_dataset = SimpleTextDataset(train_texts, max_length=128)
    val_dataset = SimpleTextDataset(val_texts, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Smaller batch for MoE
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # Create MoE model
    moe_layers = [4, 8, 12, 16]  # Use MoE in these layers
    model = create_llama_moe_7b(moe_layers=moe_layers, num_experts=4)
    
    # Create MoE training config
    config = create_moe_training_config()
    config.epochs = 1  # Very short for demo
    config.logging.log_every = 5
    config.eval_every = 25
    
    # Train with MoE
    trainer = AdvancedLlamaTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        use_moe=True,
        moe_config=model.moe_config
    )
    
    print("Starting MoE training...")
    trainer.fit()
    print("✅ MoE training completed!")
    
    return trainer.model

def demonstrate_advanced_sampling():
    """Demonstrate various sampling methods"""
    
    print("\n🎯 Demonstrating Advanced Sampling Methods")
    print("=" * 50)
    
    # Create a simple model for demo
    model = create_llama_7b()
    
    # Create inference engine
    engine = AdvancedInferenceEngine(model)
    
    # Test prompt
    test_prompt = "The future of artificial intelligence"
    input_ids = torch.randint(0, 32000, (1, 10))  # Simulate tokenized input
    
    # Different sampling configurations
    sampling_configs = {
        "Greedy": SamplingConfig(do_sample=False),
        
        "Temperature": SamplingConfig(
            temperature=0.8,
            do_sample=True
        ),
        
        "Nucleus (Top-p)": SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        ),
        
        "Top-K": SamplingConfig(
            temperature=0.8,
            top_k=50,
            do_sample=True
        ),
        
        "Typical": SamplingConfig(
            temperature=0.8,
            typical_p=0.95,
            do_sample=True
        ),
        
        "Min-P": SamplingConfig(
            temperature=0.8,
            min_p=0.1,
            do_sample=True
        ),
        
        "Mirostat": SamplingConfig(
            mirostat_mode=2,
            mirostat_tau=5.0,
            mirostat_eta=0.1,
            do_sample=True
        ),
        
        "DRY Sampling": SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            dry_multiplier=0.8,
            dry_base=1.75,
            do_sample=True
        ),
        
        "Combined": SamplingConfig(
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            frequency_penalty=0.1,
            do_sample=True
        )
    }
    
    print("Testing different sampling methods:")
    print("-" * 40)
    
    for method_name, sampling_config in sampling_configs.items():
        try:
            # Create generation config
            gen_config = GenerationConfig(
                sampling=sampling_config,
                max_new_tokens=20
            )
            
            # Generate
            result = engine.generate(input_ids, gen_config)
            generated_tokens = result['generated_tokens']
            
            print(f"📍 {method_name:15}: Generated {generated_tokens.shape[1]} tokens")
            print(f"   Tokens/sec: {result['tokens_per_second']:.2f}")
            print(f"   Time: {result['generation_time']:.3f}s")
            
        except Exception as e:
            print(f"❌ {method_name:15}: Error - {str(e)[:50]}...")
        
        print()

def demonstrate_inference_features():
    """Demonstrate advanced inference features"""
    
    print("\n🚀 Demonstrating Advanced Inference Features")
    print("=" * 50)
    
    # Create model and engine
    model = create_llama_7b()
    engine = AdvancedInferenceEngine(model)
    
    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms:",
        "Write a short story about a robot:",
        "List the benefits of renewable energy:"
    ]
    
    # Benchmark different configurations
    configs = {
        "Creative Writing": GenerationConfig(
            sampling=SamplingConfig(
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.05
            ),
            max_new_tokens=50
        ),
        
        "Factual Response": GenerationConfig(
            sampling=SamplingConfig(
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1
            ),
            max_new_tokens=50
        ),
        
        "Code Generation": GenerationConfig(
            sampling=SamplingConfig(
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.1
            ),
            max_new_tokens=50
        )
    }
    
    print("Benchmarking inference configurations:")
    print("-" * 40)
    
    for config_name, config in configs.items():
        print(f"\n📊 Testing: {config_name}")
        
        # Benchmark performance
        input_ids = torch.randint(0, 32000, (1, 20))
        benchmark_result = engine.benchmark(input_ids, config, num_runs=3)
        
        print(f"   Average time: {benchmark_result['avg_time']:.3f}s")
        print(f"   Tokens/sec: {benchmark_result['avg_tokens_per_second']:.2f}")
        print(f"   Max tokens/sec: {benchmark_result['max_tokens_per_second']:.2f}")

def demonstrate_streaming_generation():
    """Demonstrate streaming text generation"""
    
    print("\n📡 Demonstrating Streaming Generation")
    print("=" * 50)
    
    model = create_llama_7b()
    engine = AdvancedInferenceEngine(model)
    
    # Create streaming config
    config = GenerationConfig(
        sampling=SamplingConfig(
            temperature=0.8,
            top_p=0.9
        ),
        max_new_tokens=30,
        stream=True
    )
    
    input_ids = torch.randint(0, 32000, (1, 10))
    
    print("Streaming generation (simulated):")
    print("Prompt: [simulated tokens]")
    print("Response: ", end="", flush=True)
    
    try:
        # Simulate streaming
        total_tokens = 0
        for tokens in engine.generate_stream(input_ids, config):
            # In real scenario, you'd decode tokens to text
            print("█", end="", flush=True)
            total_tokens += 1
            
            # Simulate processing time
            import time
            time.sleep(0.1)
        
        print(f"\n✅ Generated {total_tokens} tokens via streaming")
        
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")

def save_and_load_models():
    """Demonstrate model saving and loading"""
    
    print("\n💾 Demonstrating Model Save/Load")
    print("=" * 50)
    
    # Create and train a small model
    model = create_llama_7b()
    
    # Save model
    save_path = Path("./demo_models")
    save_path.mkdir(exist_ok=True)
    
    model_path = save_path / "demo_llama.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save configuration
    config_path = save_path / "model_config.json"
    config = {
        "model_type": "llama_7b",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_layers": 32
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to: {config_path}")
    
    # Load model
    loaded_model = create_llama_7b()
    loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"✅ Model loaded from: {model_path}")
    
    # Verify models are the same
    original_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(p.numel() for p in loaded_model.parameters())
    
    print(f"Original model parameters: {original_params:,}")
    print(f"Loaded model parameters: {loaded_params:,}")
    print(f"Models match: {original_params == loaded_params}")

def main():
    """Main demonstration function"""
    
    print("🌟 Complete LLaMA + MoE + Advanced Sampling Demo")
    print("=" * 60)
    print("This demo showcases all advanced features of the training infrastructure:")
    print("- LLaMA model architecture")
    print("- Mixture of Experts (MoE)")
    print("- Advanced sampling methods")
    print("- Inference engine features")
    print("- Model saving/loading")
    print("=" * 60)
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        # Run demonstrations
        demonstrate_basic_llama_training()
        demonstrate_moe_training()
        demonstrate_advanced_sampling()
        demonstrate_inference_features()
        demonstrate_streaming_generation()
        save_and_load_models()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("- Modify configurations for your specific use case")
        print("- Add your own datasets and tokenizers")
        print("- Experiment with different MoE configurations")
        print("- Try different sampling methods for your tasks")
        print("- Scale up to larger models and datasets")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
