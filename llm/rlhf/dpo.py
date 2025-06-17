# training_infra/rlhf/dpo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
import copy
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt

from ..trainer import Trainer
from ..config import TrainingConfig

@dataclass
class DPOConfig:
    """Configuration for DPO training"""
    # DPO hyperparameters
    beta: float = 0.1  # Temperature parameter for DPO loss
    reference_free: bool = False  # Whether to use reference-free DPO
    loss_type: str = "sigmoid"  # "sigmoid" or "hinge" or "ipo"
    
    # Training parameters
    learning_rate: float = 5e-7
    max_length: int = 512
    max_prompt_length: int = 256
    max_target_length: int = 256
    
    # Data processing
    truncation_mode: str = "keep_end"  # "keep_end" or "keep_start"
    
    # Regularization
    sft_weight: float = 0.0  # Weight for SFT loss
    label_smoothing: float = 0.0
    
    # Advanced features
    use_peft: bool = False  # Use parameter-efficient fine-tuning
    peft_config: Optional[Dict] = None

class DPOTrainer(Trainer):
    """Direct Preference Optimization Trainer"""
    
    def __init__(self, 
                 policy_model,
                 ref_model,
                 config: DPOConfig,
                 train_dataloader,
                 val_dataloader=None,
                 tokenizer=None,
                 **trainer_kwargs):
        
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.dpo_config = config
        self.tokenizer = tokenizer
        
        # Freeze reference model
        if self.ref_model:
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        
        # Create training config
        training_config = TrainingConfig(
            optimizer=TrainingConfig.OptimizerConfig(
                name="adamw", 
                lr=config.learning_rate,
                weight_decay=0.01
            ),
            **trainer_kwargs
        )
        
        # Initialize base trainer
        super().__init__(
            model=policy_model,
            config=training_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        # DPO-specific metrics
        self.dpo_stats = {
            'chosen_rewards': [],
            'rejected_rewards': [],
            'reward_margin': [],
            'accuracy': [],
            'dpo_loss': [],
            'sft_loss': []
        }
    
    def compute_loss(self, batch):
        """Compute DPO loss"""
        # Extract batch components
        chosen_input_ids = batch['chosen_input_ids']
        rejected_input_ids = batch['rejected_input_ids']
        chosen_attention_mask = batch.get('chosen_attention_mask', None)
        rejected_attention_mask = batch.get('rejected_attention_mask', None)
        
        # Get policy model logits
        policy_chosen_logits = self.get_batch_logits(
            chosen_input_ids, chosen_attention_mask, self.policy_model
        )
        policy_rejected_logits = self.get_batch_logits(
            rejected_input_ids, rejected_attention_mask, self.policy_model
        )
        
        # Get reference model logits (if not reference-free)
        if self.dpo_config.reference_free:
            ref_chosen_logits = 0
            ref_rejected_logits = 0
        else:
            with torch.no_grad():
                ref_chosen_logits = self.get_batch_logits(
                    chosen_input_ids, chosen_attention_mask, self.ref_model
                )
                ref_rejected_logits = self.get_batch_logits(
                    rejected_input_ids, rejected_attention_mask, self.ref_model
                )
        
        # Compute DPO loss
        dpo_loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logits, policy_rejected_logits,
            ref_chosen_logits, ref_rejected_logits
        )
        
        # Add SFT loss if specified
        total_loss = dpo_loss
        sft_loss = 0
        
        if self.dpo_config.sft_weight > 0:
            # Compute SFT loss on chosen responses
            sft_loss = self.compute_sft_loss(chosen_input_ids, chosen_attention_mask)
            total_loss = dpo_loss + self.dpo_config.sft_weight * sft_loss
        
        # Update statistics
        self.dpo_stats['dpo_loss'].append(dpo_loss.item())
        self.dpo_stats['sft_loss'].append(sft_loss.item() if isinstance(sft_loss, torch.Tensor) else 0)
        self.dpo_stats['chosen_rewards'].extend(chosen_rewards.cpu().tolist())
        self.dpo_stats['rejected_rewards'].extend(rejected_rewards.cpu().tolist())
        self.dpo_stats['reward_margin'].extend((chosen_rewards - rejected_rewards).cpu().tolist())
        
        # Compute accuracy (chosen > rejected)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        self.dpo_stats['accuracy'].append(accuracy.item())
        
        return total_loss
    
    def get_batch_logits(self, input_ids, attention_mask, model):
        """Get logits for a batch from model"""
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        return logits
    
    def dpo_loss(self, policy_chosen_logits, policy_rejected_logits,
                ref_chosen_logits, ref_rejected_logits):
        """Compute DPO loss"""
        
        # Compute log probabilities for sequences
        policy_chosen_logps = self.get_batch_loss_metrics(
            policy_chosen_logits, self.current_batch['chosen_input_ids'], 
            average_log_prob=True
        )
        policy_rejected_logps = self.get_batch_loss_metrics(
            policy_rejected_logits, self.current_batch['rejected_input_ids'],
            average_log_prob=True
        )
        
        if self.dpo_config.reference_free:
            ref_chosen_logps = 0
            ref_rejected_logps = 0
        else:
            ref_chosen_logps = self.get_batch_loss_metrics(
                ref_chosen_logits, self.current_batch['chosen_input_ids'],
                average_log_prob=True
            )
            ref_rejected_logps = self.get_batch_loss_metrics(
                ref_rejected_logits, self.current_batch['rejected_input_ids'],
                average_log_prob=True
            )
        
        # Compute rewards (log probability ratios)
        policy_chosen_rewards = policy_chosen_logps - ref_chosen_logps
        policy_rejected_rewards = policy_rejected_logps - ref_rejected_logps
        
        # Compute DPO loss
        logits = self.dpo_config.beta * (policy_chosen_rewards - policy_rejected_rewards)
        
        if self.dpo_config.loss_type == "sigmoid":
            # Standard DPO loss
            loss = -F.logsigmoid(logits).mean()
        elif self.dpo_config.loss_type == "hinge":
            # Hinge loss variant
            loss = F.relu(1 - logits).mean()
        elif self.dpo_config.loss_type == "ipo":
            # IPO (Identity Preference Optimization) loss
            loss = (logits - 1/(2 * self.dpo_config.beta)).pow(2).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.dpo_config.loss_type}")
        
        # Apply label smoothing if specified
        if self.dpo_config.label_smoothing > 0:
            loss = loss * (1 - self.dpo_config.label_smoothing) + \
                   self.dpo_config.label_smoothing * 0.5
        
        return loss, policy_chosen_rewards, policy_rejected_rewards
    
    def get_batch_loss_metrics(self, logits, labels, average_log_prob=False):
        """Compute loss metrics for a batch"""
        # Shift labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for true tokens
        gathered_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding tokens
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            mask = (shift_labels != self.tokenizer.pad_token_id).float()
            gathered_log_probs = gathered_log_probs * mask
            
            if average_log_prob:
                # Average over non-padding tokens
                return gathered_log_probs.sum(-1) / mask.sum(-1).clamp(min=1)
            else:
                return gathered_log_probs.sum(-1)
        else:
            if average_log_prob:
                return gathered_log_probs.mean(-1)
            else:
                return gathered_log_probs.sum(-1)
    
    def compute_sft_loss(self, input_ids, attention_mask):
        """Compute supervised fine-tuning loss"""
        outputs = self.policy_model(input_ids, attention_mask=attention_mask, labels=input_ids)
        return outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
    
    def train_step(self, batch):
        """Override train step to store current batch"""
        self.current_batch = batch
        return super().train_step(batch)
    
    def validate(self):
        """Validate DPO model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_metrics = {
            'val_dpo_loss': [],
            'val_accuracy': [],
            'val_reward_margin': []
        }
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                
                # Compute validation loss
                loss = self.compute_loss(batch)
                val_metrics['val_dpo_loss'].append(loss.item())
        
        # Average metrics
        avg_metrics = {}
        for key, values in val_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        # Add recent training metrics
        if self.dpo_stats['accuracy']:
            avg_metrics['val_accuracy'] = np.mean(self.dpo_stats['accuracy'][-10:])
        if self.dpo_stats['reward_margin']:
            avg_metrics['val_reward_margin'] = np.mean(self.dpo_stats['reward_margin'][-10:])
        
        return avg_metrics
    
    def get_dpo_stats(self):
        """Get DPO training statistics"""
        if not self.dpo_stats['chosen_rewards']:
            return {}
        
        return {
            'avg_chosen_reward': np.mean(self.dpo_stats['chosen_rewards']),
            'avg_rejected_reward': np.mean(self.dpo_stats['rejected_rewards']),
            'avg_reward_margin': np.mean(self.dpo_stats['reward_margin']),
            'accuracy': np.mean(self.dpo_stats['accuracy']),
            'avg_dpo_loss': np.mean(self.dpo_stats['dpo_loss']),
            'reward_std': np.std(self.dpo_stats['chosen_rewards'] + self.dpo_stats['rejected_rewards'])
        }

class DPODataset(torch.utils.data.Dataset):
    """Dataset for DPO training"""
    
    def __init__(self, data: List[Dict], tokenizer, config: DPOConfig):
        """
        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            tokenizer: Tokenizer
            config: DPO configuration
        """
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Combine prompt with responses
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        # Tokenize chosen
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Tokenize rejected
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        }

# Advanced DPO variants
class ReferenceFreeTrainer(DPOTrainer):
    """Reference-free DPO trainer"""
    
    def __init__(self, policy_model, config: DPOConfig, **kwargs):
        # Set reference-free mode
        config.reference_free = True
        
        # No reference model needed
        super().__init__(
            policy_model=policy_model,
            ref_model=None,
            config=config,
            **kwargs
        )

class IPOTrainer(DPOTrainer):
    """Identity Preference Optimization trainer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set IPO loss type
        self.dpo_config.loss_type = "ipo"

class CPOTrainer(DPOTrainer):
    """Conservative Preference Optimization trainer"""
    
    def __init__(self, *args, conservative_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.conservative_weight = conservative_weight
    
    def dpo_loss(self, policy_chosen_logits, policy_rejected_logits,
                ref_chosen_logits, ref_rejected_logits):
        """Compute CPO loss with conservative regularization"""
        
        # Standard DPO loss
        dpo_loss, chosen_rewards, rejected_rewards = super().dpo_loss(
            policy_chosen_logits, policy_rejected_logits,
            ref_chosen_logits, ref_rejected_logits
        )
        
        # Conservative regularization: penalize deviation from reference
        if not self.dpo_config.reference_free:
            conservative_loss = (chosen_rewards.pow(2) + rejected_rewards.pow(2)).mean()
            total_loss = dpo_loss + self.conservative_weight * conservative_loss
        else:
            total_loss = dpo_loss
        
        return total_loss, chosen_rewards, rejected_rewards

class IterativeDPOTrainer(DPOTrainer):
    """Iterative DPO trainer with online preference collection"""
    
    def __init__(self, *args, iteration_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_steps = iteration_steps
        self.iteration_count = 0
        
    def should_collect_preferences(self):
        """Check if it's time to collect new preferences"""
        return self.global_step % self.iteration_steps == 0 and self.global_step > 0
    
    def collect_online_preferences(self, prompts: List[str], num_samples=2):
        """Collect preferences by sampling from current policy"""
        self.policy_model.eval()
        new_data = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Generate multiple responses
                responses = []
                for _ in range(num_samples):
                    inputs = self.tokenizer(prompt, return_tensors='pt')
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_length=self.dpo_config.max_length,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                    responses.append(response)
                
                # For demo, randomly assign preferences
                # In practice, you'd use a reward model or human annotation
                chosen_idx = np.random.randint(0, len(responses))
                rejected_idx = (chosen_idx + 1) % len(responses)
                
                new_data.append({
                    'prompt': prompt,
                    'chosen': responses[chosen_idx],
                    'rejected': responses[rejected_idx]
                })
        
        return new_data

# Utility functions
def create_dpo_trainer(policy_model, ref_model, train_data: List[Dict], 
                      val_data: List[Dict] = None, tokenizer=None, 
                      config: DPOConfig = None):
    """Factory function to create DPO trainer"""
    
    if config is None:
        config = DPOConfig()
    
    # Create datasets
    train_dataset = DPODataset(train_data, tokenizer, config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=8,  # Small batch size for DPO
        shuffle=True
    )
    
    val_loader = None
    if val_data:
        val_dataset = DPODataset(val_data, tokenizer, config)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False
        )
    
    return DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer
    )

def train_dpo_pipeline(policy_model, ref_model, train_data: List[Dict],
                      val_data: List[Dict] = None, tokenizer=None,
                      config: DPOConfig = None, num_epochs: int = 3):
    """Complete DPO training pipeline"""
    
    # Create trainer
    trainer = create_dpo_trainer(
        policy_model=policy_model,
        ref_model=ref_model,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        config=config
    )
    
    # Update training config for DPO
    trainer.config.epochs = num_epochs
    trainer.config.logging.log_every = 20
    trainer.config.eval_every = 200
    
    # Train
    trainer.fit()
    
    return trainer

def evaluate_dpo_model(dpo_trainer: DPOTrainer, test_data: List[Dict]) -> Dict:
    """Evaluate DPO model"""
    
    test_dataset = DPODataset(test_data, dpo_trainer.tokenizer, dpo_trainer.dpo_config)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    dpo_trainer.model.eval()
    metrics = {
        'accuracy': [],
        'reward_margin': [],
        'chosen_rewards': [],
        'rejected_rewards': []
    }
    
    with torch.no_grad():
        for batch in test_loader:
            batch = dpo_trainer._move_to_device(batch)
            
            # Compute rewards
            chosen_input_ids = batch['chosen_input_ids']
            rejected_input_ids = batch['rejected_input_ids']
            chosen_attention_mask = batch.get('chosen_attention_mask', None)
            rejected_attention_mask = batch.get('rejected_attention_mask', None)
            
            # Get logits
            policy_chosen_logits = dpo_trainer.get_batch_logits(
                chosen_input_ids, chosen_attention_mask, dpo_trainer.policy_model
            )
            policy_rejected_logits = dpo_trainer.get_batch_logits(
                rejected_input_ids, rejected_attention_mask, dpo_trainer.policy_model
            )
            
            if dpo_trainer.ref_model is not None:
                ref_chosen_logits = dpo_trainer.get_batch_logits(
                    chosen_input_ids, chosen_attention_mask, dpo_trainer.ref_model
                )
                ref_rejected_logits = dpo_trainer.get_batch_logits(
                    rejected_input_ids, rejected_attention_mask, dpo_trainer.ref_model
                )
            else:
                ref_chosen_logits = 0
                ref_rejected_logits = 0
            
            # Store current batch for loss computation
            dpo_trainer.current_batch = batch
            
            # Compute rewards
            _, chosen_rewards, rejected_rewards = dpo_trainer.dpo_loss(
                policy_chosen_logits, policy_rejected_logits,
                ref_chosen_logits, ref_rejected_logits
            )
            
            # Calculate metrics
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            
            metrics['accuracy'].append(accuracy.item())
            metrics['reward_margin'].append(reward_margin.item())
            metrics['chosen_rewards'].extend(chosen_rewards.cpu().tolist())
            metrics['rejected_rewards'].extend(rejected_rewards.cpu().tolist())
    
    # Aggregate metrics
    return {
        'accuracy': np.mean(metrics['accuracy']),
        'reward_margin': np.mean(metrics['reward_margin']),
        'avg_chosen_reward': np.mean(metrics['chosen_rewards']),
        'avg_rejected_reward': np.mean(metrics['rejected_rewards']),
        'reward_std': np.std(metrics['chosen_rewards'] + metrics['rejected_rewards'])
    }

def compare_dpo_variants(policy_model, ref_model, train_data: List[Dict],
                        test_data: List[Dict], tokenizer, num_epochs=2):
    """Compare different DPO variants"""
    
    print("🔬 Comparing DPO Variants")
    print("=" * 50)
    
    variants = {
        'Standard DPO': {
            'trainer_class': DPOTrainer,
            'config': DPOConfig(beta=0.1, loss_type='sigmoid')
        },
        'IPO': {
            'trainer_class': IPOTrainer,
            'config': DPOConfig(beta=0.1, loss_type='ipo')
        },
        'Reference-Free DPO': {
            'trainer_class': ReferenceFreeTrainer,
            'config': DPOConfig(beta=0.1, reference_free=True)
        },
        'Conservative DPO': {
            'trainer_class': CPOTrainer,
            'config': DPOConfig(beta=0.1),
            'kwargs': {'conservative_weight': 0.1}
        }
    }
    
    results = {}
    
    for variant_name, variant_config in variants.items():
        print(f"\n🧪 Testing {variant_name}")
        print("-" * 30)
        
        try:
            # Create trainer
            trainer_class = variant_config['trainer_class']
            config = variant_config['config']
            kwargs = variant_config.get('kwargs', {})
            
            if variant_name == 'Reference-Free DPO':
                trainer = trainer_class(
                    policy_model=copy.deepcopy(policy_model),
                    config=config,
                    train_dataloader=torch.utils.data.DataLoader(
                        DPODataset(train_data[:100], tokenizer, config),
                        batch_size=4, shuffle=True
                    ),
                    tokenizer=tokenizer,
                    **kwargs
                )
            else:
                trainer = trainer_class(
                    policy_model=copy.deepcopy(policy_model),
                    ref_model=ref_model,
                    config=config,
                    train_dataloader=torch.utils.data.DataLoader(
                        DPODataset(train_data[:100], tokenizer, config),
                        batch_size=4, shuffle=True
                    ),
                    tokenizer=tokenizer,
                    **kwargs
                )
            
            # Train for a few steps (demo)
            trainer.config.epochs = 1
            trainer.config.max_steps = 50
            trainer.fit()
            
            # Evaluate
            eval_metrics = evaluate_dpo_model(trainer, test_data[:50])
            results[variant_name] = eval_metrics
            
            print(f"✅ {variant_name} completed")
            print(f"   Accuracy: {eval_metrics['accuracy']:.3f}")
            print(f"   Reward Margin: {eval_metrics['reward_margin']:.3f}")
            
        except Exception as e:
            print(f"❌ {variant_name} failed: {e}")
            results[variant_name] = None
    
    return results

def visualize_dpo_training(dpo_trainer: DPOTrainer, save_path: str = None):
    """Visualize DPO training progress"""
    
    stats = dpo_trainer.dpo_stats
    
    if not stats['chosen_rewards']:
        print("No training data to visualize")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DPO Training Progress', fontsize=16)
    
    # Plot 1: Loss over time
    if stats['dpo_loss']:
        ax1.plot(stats['dpo_loss'], label='DPO Loss', color='blue')
        if stats['sft_loss'] and any(x > 0 for x in stats['sft_loss']):
            ax1.plot(stats['sft_loss'], label='SFT Loss', color='orange')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
    
    # Plot 2: Accuracy over time
    if stats['accuracy']:
        ax2.plot(stats['accuracy'], color='green')
        ax2.set_title('Preference Accuracy')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
    
    # Plot 3: Reward distribution
    if stats['chosen_rewards'] and stats['rejected_rewards']:
        ax3.hist(stats['chosen_rewards'], alpha=0.7, label='Chosen', bins=30, color='green')
        ax3.hist(stats['rejected_rewards'], alpha=0.7, label='Rejected', bins=30, color='red')
        ax3.set_title('Reward Distribution')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)
    
    # Plot 4: Reward margin over time
    if stats['reward_margin']:
        ax4.plot(stats['reward_margin'], color='purple')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Reward Margin (Chosen - Rejected)')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Reward Margin')
        ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {save_path}")
    
    return fig

def create_preference_data_from_completions(prompts: List[str], 
                                          completions_per_prompt: List[List[str]],
                                          preference_function: callable = None) -> List[Dict]:
    """Create preference data from completions"""
    
    if preference_function is None:
        # Default: prefer longer responses (simple heuristic)
        def preference_function(prompt, comp1, comp2):
            return len(comp1) > len(comp2)
    
    preference_data = []
    
    for prompt, completions in zip(prompts, completions_per_prompt):
        if len(completions) < 2:
            continue
        
        # Compare all pairs
        for i in range(len(completions)):
            for j in range(i + 1, len(completions)):
                comp1, comp2 = completions[i], completions[j]
                
                if preference_function(prompt, comp1, comp2):
                    chosen, rejected = comp1, comp2
                else:
                    chosen, rejected = comp2, comp1
                
                preference_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                })
    
    return preference_data

def save_dpo_results(trainer: DPOTrainer, save_dir: str):
    """Save DPO training results and model"""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_path / "dpo_model.pt"
    torch.save(trainer.policy_model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Save training stats
    stats_path = save_path / "training_stats.json"
    stats_to_save = trainer.get_dpo_stats()
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    print(f"📊 Stats saved to {stats_path}")
    
    # Save config
    config_path = save_path / "dpo_config.json"
    config_dict = {
        'beta': trainer.dpo_config.beta,
        'reference_free': trainer.dpo_config.reference_free,
        'loss_type': trainer.dpo_config.loss_type,
        'learning_rate': trainer.dpo_config.learning_rate,
        'max_length': trainer.dpo_config.max_length,
        'sft_weight': trainer.dpo_config.sft_weight,
        'label_smoothing': trainer.dpo_config.label_smoothing
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"⚙️ Config saved to {config_path}")
    
    # Create visualization
    viz_path = save_path / "training_progress.png"
    visualize_dpo_training(trainer, str(viz_path))
    
    return save_path

def load_dpo_model(model_class, model_path: str, config_path: str = None):
    """Load trained DPO model"""
    
    # Load model state
    model = model_class()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Load config if provided
    config = None
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            config = DPOConfig(**config_dict)
    
    return model, config

# Example usage and testing functions
def create_sample_preference_data(num_samples: int = 100) -> List[Dict]:
    """Create sample preference data for testing"""
    
    import random
    
    # Sample prompts
    prompts = [
        "Write a short story about",
        "Explain the concept of",
        "What are the benefits of",
        "How do you solve",
        "Describe the process of",
        "What is the difference between",
        "Give me advice on",
        "Compare and contrast"
    ]
    
    topics = [
        "artificial intelligence", "climate change", "space exploration",
        "quantum computing", "renewable energy", "genetic engineering",
        "virtual reality", "blockchain technology", "machine learning",
        "neural networks", "data science", "cybersecurity"
    ]
    
    preference_data = []
    
    for _ in range(num_samples):
        prompt = f"{random.choice(prompts)} {random.choice(topics)}"
        
        # Generate mock responses (in practice, these would be real model outputs)
        chosen_response = f"This is a comprehensive and well-structured response about {random.choice(topics)}. " \
                         f"It provides detailed information and clear explanations. " \
                         f"The response addresses the question thoroughly and offers valuable insights."
        
        rejected_response = f"Brief response about {random.choice(topics)}. Limited details."
        
        preference_data.append({
            'prompt': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response
        })
    
    return preference_data

def demonstrate_dpo_pipeline():
    """Demonstrate complete DPO training pipeline"""
    
    print("🚀 DPO Training Pipeline Demonstration")
    print("=" * 60)
    
    # Mock tokenizer for demo
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
            self.vocab_size = 1000
        
        def __call__(self, text, **kwargs):
            # Simple tokenization for demo
            tokens = [hash(word) % self.vocab_size for word in text.split()[:100]]
            max_length = kwargs.get('max_length', 512)
            
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
            
            return {
                'input_ids': torch.tensor(tokens).unsqueeze(0),
                'attention_mask': torch.ones(1, len(tokens))
            }
        
        def decode(self, tokens, **kwargs):
            return f"[Decoded text from {len(tokens)} tokens]"
    
    # Mock model for demo
    class MockModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, nhead=8), 
                num_layers=2
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embedding(input_ids)
            x = x.transpose(0, 1)  # seq_len, batch, hidden
            x = self.transformer(x)
            x = x.transpose(0, 1)  # batch, seq_len, hidden
            logits = self.lm_head(x)
            
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            return type('Output', (), {'logits': logits, 'loss': loss})()
    
    try:
        # Create models and tokenizer
        print("📋 Creating models and data...")
        policy_model = MockModel()
        ref_model = MockModel()
        tokenizer = MockTokenizer()
        
        # Create preference data
        train_data = create_sample_preference_data(200)
        val_data = create_sample_preference_data(50)
        test_data = create_sample_preference_data(30)
        
        print(f"✅ Created {len(train_data)} training examples")
        print(f"✅ Created {len(val_data)} validation examples")
        print(f"✅ Created {len(test_data)} test examples")
        
        # Configure DPO
        dpo_config = DPOConfig(
            beta=0.1,
            learning_rate=1e-4,
            max_length=256,
            sft_weight=0.1,
            loss_type='sigmoid'
        )
        
        print(f"\n⚙️ DPO Configuration:")
        print(f"   Beta: {dpo_config.beta}")
        print(f"   Learning Rate: {dpo_config.learning_rate}")
        print(f"   Max Length: {dpo_config.max_length}")
        print(f"   SFT Weight: {dpo_config.sft_weight}")
        
        # Train DPO model
        print("\n🔄 Training DPO model...")
        trainer = train_dpo_pipeline(
            policy_model=policy_model,
            ref_model=ref_model,
            train_data=train_data[:50],  # Small subset for demo
            val_data=val_data[:20],
            tokenizer=tokenizer,
            config=dpo_config,
            num_epochs=1  # Short training for demo
        )
        
        print("✅ DPO training completed!")
        
        # Get training statistics
        stats = trainer.get_dpo_stats()
        print(f"\n📊 Training Statistics:")
        print(f"   Average Chosen Reward: {stats.get('avg_chosen_reward', 0):.3f}")
        print(f"   Average Rejected Reward: {stats.get('avg_rejected_reward', 0):.3f}")
        print(f"   Reward Margin: {stats.get('avg_reward_margin', 0):.3f}")
        print(f"   Preference Accuracy: {stats.get('accuracy', 0):.3f}")
        
        # Evaluate model
        print("\n🔍 Evaluating model...")
        eval_results = evaluate_dpo_model(trainer, test_data[:20])
        
        print(f"📈 Evaluation Results:")
        print(f"   Test Accuracy: {eval_results['accuracy']:.3f}")
        print(f"   Test Reward Margin: {eval_results['reward_margin']:.3f}")
        print(f"   Average Chosen Reward: {eval_results['avg_chosen_reward']:.3f}")
        print(f"   Average Rejected Reward: {eval_results['avg_rejected_reward']:.3f}")
        
        # Compare variants
        print("\n🧪 Comparing DPO variants...")
        variant_results = compare_dpo_variants(
            policy_model=MockModel(),
            ref_model=MockModel(),
            train_data=train_data[:100],
            test_data=test_data[:30],
            tokenizer=tokenizer,
            num_epochs=1
        )
        
        print(f"\n📊 Variant Comparison:")
        for variant, results in variant_results.items():
            if results:
                print(f"   {variant}:")
                print(f"     Accuracy: {results['accuracy']:.3f}")
                print(f"     Reward Margin: {results['reward_margin']:.3f}")
            else:
                print(f"   {variant}: Failed")
        
        # Save results
        print("\n💾 Saving results...")
        save_dir = "dpo_experiment_results"
        save_path = save_dpo_results(trainer, save_dir)
        print(f"✅ Results saved to {save_path}")
        
        return trainer, eval_results, variant_results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def benchmark_dpo_hyperparameters():
    """Benchmark different DPO hyperparameters"""
    
    print("🔧 DPO Hyperparameter Benchmarking")
    print("=" * 50)
    
    # Test different beta values
    beta_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    loss_types = ['sigmoid', 'hinge', 'ipo']
    
    results = {}
    
    for beta in beta_values:
        for loss_type in loss_types:
            config_name = f"beta_{beta}_loss_{loss_type}"
            print(f"\n🧪 Testing {config_name}")
            
            try:
                config = DPOConfig(beta=beta, loss_type=loss_type, learning_rate=1e-4)
                
                # Mock training (simplified)
                # In practice, you would run full training here
                mock_result = {
                    'accuracy': np.random.uniform(0.5, 0.9),
                    'reward_margin': np.random.uniform(0.1, 1.0),
                    'convergence_steps': np.random.randint(100, 1000)
                }
                
                results[config_name] = mock_result
                print(f"   Accuracy: {mock_result['accuracy']:.3f}")
                print(f"   Reward Margin: {mock_result['reward_margin']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[config_name] = None
    
    # Find best configuration
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_config = max(valid_results.keys(), 
                         key=lambda k: valid_results[k]['accuracy'])
        print(f"\n🏆 Best Configuration: {best_config}")
        print(f"   Accuracy: {valid_results[best_config]['accuracy']:.3f}")
        print(f"   Reward Margin: {valid_results[best_config]['reward_margin']:.3f}")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstration
    print("Starting DPO demonstration...")
    trainer, eval_results, variant_results = demonstrate_dpo_pipeline()
    
    if trainer is not None:
        print("\n🎉 Demonstration completed successfully!")
        
        # Run hyperparameter benchmark
        print("\n" + "="*60)
        benchmark_results = benchmark_dpo_hyperparameters()
        
        print(f"\n✅ Benchmarked {len(benchmark_results)} configurations")
    else:
        print("\n❌ Demonstration failed")
    
    print("\nDPO training infrastructure is ready for use!")