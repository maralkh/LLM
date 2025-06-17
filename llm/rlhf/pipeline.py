# training_infra/rlhf/pipeline.py
"""
Complete RLHF pipeline integration
Combines PPO, DPO, and GRPO for comprehensive human feedback training
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
import json
import copy
from pathlib import Path
import numpy as np

from .reward_model import RewardModel, RewardModelTrainer, train_reward_model
from .ppo import PPOTrainer, PPOConfig, create_ppo_trainer
from .dpo import DPOTrainer, DPOConfig, create_dpo_trainer 
from .grpo import GRPOTrainer, GRPOConfig, create_grpo_trainer
from ..models.llama import create_llama_7b
from ..config import TrainingConfig

class RLHFPipeline:
    """Complete RLHF training pipeline"""
    
    def __init__(self, base_model, tokenizer, config: Dict[str, Any] = None):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = next(base_model.parameters()).device
        
        # Components
        self.reward_model = None
        self.policy_model = None
        self.ref_model = None
        
        # Training history
        self.training_history = {
            'sft': {},
            'reward_model': {},
            'rlhf': {}
        }
    
    def step1_supervised_finetuning(self, sft_data: List[Dict], num_epochs: int = 3):
        """Step 1: Supervised Fine-Tuning on high-quality demonstrations"""
        
        print("🚀 Step 1: Supervised Fine-Tuning")
        print("=" * 50)
        
        # Create SFT dataset
        class SFTDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer, max_length=512):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                text = item['prompt'] + item['response']
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0)
                return {
                    'input_ids': input_ids,
                    'labels': input_ids.clone()
                }
        
        # Create data loader
        sft_dataset = SFTDataset(sft_data, self.tokenizer)
        sft_loader = torch.utils.data.DataLoader(
            sft_dataset, 
            batch_size=8, 
            shuffle=True
        )
        
        # Create training config
        sft_config = TrainingConfig(
            epochs=num_epochs,
            batch_size=8,
            optimizer=TrainingConfig.OptimizerConfig(lr=2e-5),
            logging=TrainingConfig.LoggingConfig(log_every=50),
            checkpoint=TrainingConfig.CheckpointConfig(save_every=1000)
        )
        
        # Train using base trainer
        from ..trainer import Trainer
        
        class SFTTrainer(Trainer):
            def compute_loss(self, batch):
                input_ids = batch['input_ids']
                labels = batch['labels']
                outputs = self.model(input_ids, labels=labels)
                return outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        trainer = SFTTrainer(
            model=self.base_model,
            config=sft_config,
            train_dataloader=sft_loader
        )
        
        trainer.fit()
        
        # Save SFT model as policy model
        self.policy_model = copy.deepcopy(self.base_model)
        self.ref_model = copy.deepcopy(self.base_model)  # Frozen reference
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.training_history['sft'] = {
            'num_epochs': num_epochs,
            'num_examples': len(sft_data),
            'final_loss': trainer.stats if hasattr(trainer, 'stats') else None
        }
        
        print("✅ SFT completed!")
        return trainer
    
    def step2_reward_model_training(self, preference_data: List[Dict], num_epochs: int = 3):
        """Step 2: Train reward model on human preferences"""
        
        print("\n🎯 Step 2: Reward Model Training")
        print("=" * 50)
        
        # Train reward model
        self.reward_model = train_reward_model(
            base_model=copy.deepcopy(self.base_model),
            train_data=preference_data,
            tokenizer=self.tokenizer
        )
        
        self.training_history['reward_model'] = {
            'num_epochs': num_epochs,
            'num_preferences': len(preference_data),
        }
        
        print("✅ Reward Model training completed!")
        return self.reward_model
    
    def step3a_ppo_training(self, prompts: List[str], num_epochs: int = 20):
        """Step 3a: PPO training with reward model"""
        
        print("\n🔄 Step 3a: PPO Training")
        print("=" * 50)
        
        if self.reward_model is None:
            raise ValueError("Reward model must be trained first!")
        
        # Create PPO config
        ppo_config = PPOConfig(
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            ppo_epochs=4,
            rollout_batch_size=16,
            max_new_tokens=128
        )
        
        # Create PPO trainer
        ppo_trainer = create_ppo_trainer(
            base_model=copy.deepcopy(self.policy_model),
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            config=ppo_config,
            ref_model=self.ref_model
        )
        
        # Train
        ppo_trainer.train(prompts, num_epochs=num_epochs, prompts_per_epoch=50)
        
        # Update policy model
        self.policy_model = ppo_trainer.actor.base_model
        
        self.training_history['rlhf']['ppo'] = {
            'num_epochs': num_epochs,
            'num_prompts': len(prompts),
            'final_stats': ppo_trainer.stats
        }
        
        print("✅ PPO training completed!")
        return ppo_trainer
    
    def step3b_dpo_training(self, preference_data: List[Dict], num_epochs: int = 3):
        """Step 3b: DPO training (alternative to PPO)"""
        
        print("\n📊 Step 3b: DPO Training")
        print("=" * 50)
        
        # Create DPO config
        dpo_config = DPOConfig(
            beta=0.1,
            learning_rate=5e-7,
            max_length=512
        )
        
        # Create DPO trainer
        dpo_trainer = create_dpo_trainer(
            policy_model=copy.deepcopy(self.policy_model),
            ref_model=self.ref_model,
            train_data=preference_data,
            tokenizer=self.tokenizer,
            config=dpo_config
        )
        
        # Update training config
        dpo_trainer.config.epochs = num_epochs
        
        # Train
        dpo_trainer.fit()
        
        # Update policy model
        self.policy_model = dpo_trainer.policy_model
        
        self.training_history['rlhf']['dpo'] = {
            'num_epochs': num_epochs,
            'num_preferences': len(preference_data),
            'final_stats': dpo_trainer.get_dpo_stats()
        }
        
        print("✅ DPO training completed!")
        return dpo_trainer
    
    def step3c_grpo_training(self, prompts: List[str], num_epochs: int = 15):
        """Step 3c: GRPO training (alternative to PPO/DPO)"""
        
        print("\n🎲 Step 3c: GRPO Training")
        print("=" * 50)
        
        if self.reward_model is None:
            raise ValueError("Reward model must be trained first!")
        
        # Create GRPO config
        grpo_config = GRPOConfig(
            group_size=8,
            temperature=0.7,
            learning_rate=1e-6,
            ranking_loss_type="listwise",
            sampling_strategy="diverse"
        )
        
        # Create GRPO trainer
        grpo_trainer = create_grpo_trainer(
            policy_model=copy.deepcopy(self.policy_model),
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            config=grpo_config,
            ref_model=self.ref_model
        )
        
        # Train
        grpo_trainer.train(prompts, num_epochs=num_epochs, prompts_per_epoch=40)
        
        # Update policy model
        self.policy_model = grpo_trainer.policy_model
        
        self.training_history['rlhf']['grpo'] = {
            'num_epochs': num_epochs,
            'num_prompts': len(prompts),
            'final_stats': grpo_trainer.stats
        }
        
        print("✅ GRPO training completed!")
        return grpo_trainer
    
    def evaluate_model(self, test_prompts: List[str], test_preferences: List[Dict] = None):
        """Evaluate the trained model"""
        
        print("\n📈 Model Evaluation")
        print("=" * 50)
        
        results = {}
        
        # Generate responses
        self.policy_model.eval()
        generated_responses = []
        
        with torch.no_grad():
            for prompt in test_prompts:
                prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Generate response
                output_ids = self.policy_model.generate(
                    prompt_ids,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(
                    output_ids[0][len(prompt_ids[0]):], 
                    skip_special_tokens=True
                )
                
                generated_responses.append({
                    'prompt': prompt,
                    'response': response
                })
        
        results['generated_responses'] = generated_responses
        
        # Evaluate with reward model if available
        if self.reward_model:
            rewards = []
            
            with torch.no_grad():
                for item in generated_responses:
                    full_text = item['prompt'] + item['response']
                    text_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
                    reward = self.reward_model.get_reward(text_ids)
                    rewards.append(reward.item())
            
            results['reward_stats'] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards)
            }
        
        # Evaluate on preferences if provided
        if test_preferences and self.reward_model:
            from .reward_model import evaluate_reward_model_correlation
            correlation_results = evaluate_reward_model_correlation(
                self.reward_model, test_preferences, self.tokenizer
            )
            results['preference_accuracy'] = correlation_results
        
        return results
    
    def save_pipeline(self, save_dir: str):
        """Save the complete pipeline"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        if self.policy_model:
            torch.save(self.policy_model.state_dict(), save_path / 'policy_model.pt')
        
        if self.reward_model:
            torch.save(self.reward_model.state_dict(), save_path / 'reward_model.pt')
        
        if self.ref_model:
            torch.save(self.ref_model.state_dict(), save_path / 'ref_model.pt')
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save config
        with open(save_path / 'pipeline_config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        print(f"✅ Pipeline saved to {save_dir}")
    
    def load_pipeline(self, save_dir: str):
        """Load a saved pipeline"""
        
        save_path = Path(save_dir)
        
        # Load models
        if (save_path / 'policy_model.pt').exists():
            self.policy_model = copy.deepcopy(self.base_model)
            self.policy_model.load_state_dict(torch.load(save_path / 'policy_model.pt'))
        
        if (save_path / 'reward_model.pt').exists():
            from .reward_model import RewardModel, RewardModelConfig
            config = RewardModelConfig()
            self.reward_model = RewardModel(copy.deepcopy(self.base_model), config)
            self.reward_model.load_state_dict(torch.load(save_path / 'reward_model.pt'))
        
        if (save_path / 'ref_model.pt').exists():
            self.ref_model = copy.deepcopy(self.base_model)
            self.ref_model.load_state_dict(torch.load(save_path / 'ref_model.pt'))
        
        # Load training history
        if (save_path / 'training_history.json').exists():
            with open(save_path / 'training_history.json', 'r') as f:
                self.training_history = json.load(f)
        
        print(f"✅ Pipeline loaded from {save_dir}")

# Complete RLHF training functions
def train_full_rlhf_pipeline(
    base_model,
    tokenizer,
    sft_data: List[Dict],
    preference_data: List[Dict],
    prompts: List[str],
    method: str = "ppo",  # "ppo", "dpo", "grpo"
    save_dir: str = None
):
    """
    Complete RLHF training pipeline
    
    Args:
        base_model: Pre-trained base model
        tokenizer: Tokenizer
        sft_data: SFT training data [{'prompt': str, 'response': str}]
        preference_data: Preference data [{'prompt': str, 'chosen': str, 'rejected': str}]
        prompts: Prompts for RL training
        method: RLHF method to use
        save_dir: Directory to save results
    """
    
    print("🌟 Starting Complete RLHF Pipeline")
    print("=" * 60)
    
    # Create pipeline
    pipeline = RLHFPipeline(base_model, tokenizer)
    
    # Step 1: SFT
    pipeline.step1_supervised_finetuning(sft_data, num_epochs=3)
    
    # Step 2: Reward Model
    pipeline.step2_reward_model_training(preference_data, num_epochs=3)
    
    # Step 3: RLHF
    if method == "ppo":
        trainer = pipeline.step3a_ppo_training(prompts, num_epochs=20)
    elif method == "dpo":
        trainer = pipeline.step3b_dpo_training(preference_data, num_epochs=3)
    elif method == "grpo":
        trainer = pipeline.step3c_grpo_training(prompts, num_epochs=15)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Evaluate
    test_prompts = prompts[:10]  # Use first 10 prompts for evaluation
    results = pipeline.evaluate_model(test_prompts, preference_data[:20])
    
    print("\n📊 Final Results:")
    print("-" * 30)
    if 'reward_stats' in results:
        print(f"Average Reward: {results['reward_stats']['mean']:.4f}")
        print(f"Reward Std: {results['reward_stats']['std']:.4f}")
    
    if 'preference_accuracy' in results:
        print(f"Preference Accuracy: {results['preference_accuracy']['accuracy']:.4f}")
    
    # Save pipeline
    if save_dir:
        pipeline.save_pipeline(save_dir)
    
    return pipeline, trainer, results

# Example data creation functions
def create_sample_sft_data(num_samples: int = 100) -> List[Dict]:
    """Create sample SFT data"""
    
    sample_data = []
    
    topics = [
        "Explain quantum computing",
        "Write a recipe for chocolate cake", 
        "Describe the water cycle",
        "Explain machine learning",
        "Write a short story about space",
        "Describe how photosynthesis works",
        "Explain the theory of relativity",
        "Write about renewable energy"
    ]
    
    for i in range(num_samples):
        topic = topics[i % len(topics)]
        prompt = f"{topic}:"
        response = f"This is a high-quality response about {topic.lower()}. [Sample response {i+1}]"
        
        sample_data.append({
            'prompt': prompt,
            'response': response
        })
    
    return sample_data

def create_sample_preference_data(num_samples: int = 200) -> List[Dict]:
    """Create sample preference data"""
    
    preference_data = []
    
    prompts = [
        "Explain artificial intelligence:",
        "Write about climate change:",
        "Describe healthy eating:",
        "Explain cryptocurrency:", 
        "Write about space exploration:",
        "Describe renewable energy:",
        "Explain cybersecurity:",
        "Write about mental health:"
    ]
    
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        
        chosen = f"This is a well-structured, informative response about {prompt[:-1].lower()}. [High quality response {i+1}]"
        rejected = f"This is a poor quality response. [Low quality response {i+1}]"
        
        preference_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    return preference_data

def create_sample_prompts(num_prompts: int = 50) -> List[str]:
    """Create sample prompts for RL training"""
    
    prompt_templates = [
        "Explain the benefits of",
        "Write a guide about",
        "Describe how to",
        "What are the main features of",
        "Compare and contrast",
        "Analyze the impact of",
        "Provide tips for",
        "Discuss the importance of"
    ]
    
    topics = [
        "renewable energy",
        "healthy lifestyle",
        "artificial intelligence", 
        "sustainable development",
        "digital privacy",
        "climate change",
        "space exploration",
        "mental wellness"
    ]
    
    prompts = []
    for i in range(num_prompts):
        template = prompt_templates[i % len(prompt_templates)]
        topic = topics[i % len(topics)]
        prompts.append(f"{template} {topic}:")
    
    return prompts

# Complete example usage
def run_complete_rlhf_example():
    """Run a complete RLHF training example"""
    
    print("🚀 Running Complete RLHF Example")
    print("=" * 50)
    
    # Create base model (you can replace with your own)
    base_model = create_llama_7b()
    
    # Assume you have a tokenizer (you'd load your own)
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            self.pad_token_id = 0
        
        def encode(self, text, return_tensors=None):
            # Dummy implementation
            tokens = [hash(word) % 32000 for word in text.split()[:100]]
            if return_tensors == 'pt':
                return torch.tensor(tokens).unsqueeze(0)
            return tokens
        
        def decode(self, tokens, skip_special_tokens=True):
            return f"[Generated text from tokens: {len(tokens)} tokens]"
        
        def __call__(self, text, **kwargs):
            tokens = self.encode(text)
            return {
                'input_ids': torch.tensor(tokens).unsqueeze(0),
                'attention_mask': torch.ones(1, len(tokens))
            }
    
    tokenizer = DummyTokenizer()
    
    # Create sample data
    sft_data = create_sample_sft_data(50)
    preference_data = create_sample_preference_data(100)
    prompts = create_sample_prompts(30)
    
    print(f"Created {len(sft_data)} SFT examples")
    print(f"Created {len(preference_data)} preference pairs")
    print(f"Created {len(prompts)} training prompts")
    
    # Run different RLHF methods
    methods = ["dpo", "ppo", "grpo"]
    
    for method in methods:
        print(f"\n🔥 Training with {method.upper()}")
        
        try:
            pipeline, trainer, results = train_full_rlhf_pipeline(
                base_model=copy.deepcopy(base_model),
                tokenizer=tokenizer,
                sft_data=sft_data,
                preference_data=preference_data,
                prompts=prompts,
                method=method,
                save_dir=f"./rlhf_results_{method}"
            )
            
            print(f"✅ {method.upper()} training completed successfully!")
            
        except Exception as e:
            print(f"❌ {method.upper()} training failed: {e}")

if __name__ == "__main__":
    run_complete_rlhf_example()