# training_infra/rlhf/grpo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
import copy

from ..trainer import Trainer
from ..config import TrainingConfig
from .reward_model import RewardModel

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    # GRPO hyperparameters
    group_size: int = 8  # Number of responses per prompt for comparison
    temperature: float = 0.7  # Sampling temperature
    top_p: float = 0.9  # Nucleus sampling threshold
    
    # Learning parameters
    learning_rate: float = 1e-6
    beta: float = 0.01  # KL regularization coefficient
    clip_epsilon: float = 0.2  # PPO-style clipping
    
    # Training parameters
    max_length: int = 512
    max_new_tokens: int = 128
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Ranking loss
    ranking_loss_type: str = "listwise"  # "pairwise", "listwise", "contrastive"
    margin: float = 0.1  # Margin for pairwise ranking
    
    # Group sampling strategies
    sampling_strategy: str = "diverse"  # "diverse", "beam", "random", "nucleus"
    diversity_penalty: float = 0.1  # For diverse sampling
    
    # Baseline estimation
    use_baseline: bool = True
    baseline_momentum: float = 0.9
    
    # Advanced features
    use_advantage_normalization: bool = True
    entropy_regularization: float = 0.01
    length_normalization: bool = True

class GRPOTrainer:
    """Group Relative Policy Optimization Trainer"""
    
    def __init__(self, 
                 policy_model,
                 reward_model: RewardModel,
                 config: GRPOConfig,
                 tokenizer=None,
                 ref_model=None):
        
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.config = config
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        
        # Freeze reference model if provided
        if self.ref_model:
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Baseline for advantage estimation
        self.baseline = 0.0
        
        # Device
        self.device = next(self.policy_model.parameters()).device
        
        # Statistics tracking
        self.stats = {
            'total_episodes': 0,
            'avg_reward': 0.0,
            'avg_advantage': 0.0,
            'ranking_loss': 0.0,
            'kl_penalty': 0.0,
            'entropy': 0.0,
            'baseline': 0.0
        }
    
    def generate_group_responses(self, prompt: str) -> List[Dict]:
        """Generate multiple responses for a single prompt"""
        
        self.policy_model.eval()
        responses = []
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = prompt_tokens.shape[1]
        
        with torch.no_grad():
            for i in range(self.config.group_size):
                if self.config.sampling_strategy == "diverse":
                    # Diverse sampling with penalty for similar responses
                    response = self._diverse_sampling(prompt_tokens, responses, i)
                elif self.config.sampling_strategy == "beam":
                    # Beam search with different beam indices
                    response = self._beam_sampling(prompt_tokens, beam_idx=i)
                elif self.config.sampling_strategy == "nucleus":
                    # Nucleus sampling with varied parameters
                    response = self._nucleus_sampling(prompt_tokens, top_p=self.config.top_p)
                else:  # random
                    response = self._random_sampling(prompt_tokens)
                
                responses.append({
                    'prompt': prompt,
                    'response_text': response['text'],
                    'response_tokens': response['tokens'],
                    'full_tokens': response['full_tokens'],
                    'log_probs': response['log_probs'],
                    'response_id': i
                })
        
        return responses
    
    def _diverse_sampling(self, prompt_tokens: torch.Tensor, existing_responses: List[Dict], 
                         response_id: int) -> Dict:
        """Sample diverse responses"""
        
        current_tokens = prompt_tokens.clone()
        generated_tokens = []
        log_probs = []
        
        for step in range(self.config.max_new_tokens):
            # Get logits
            outputs = self.policy_model(current_tokens)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Apply diversity penalty
            if existing_responses and self.config.diversity_penalty > 0:
                logits = self._apply_diversity_penalty(logits, existing_responses, step)
            
            # Apply temperature
            logits = logits / self.config.temperature
            
            # Nucleus sampling
            probs = F.softmax(logits, dim=-1)
            
            # Top-p filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            # Find cutoff
            cutoff_index = torch.where(cumulative_probs > self.config.top_p)[0]
            if len(cutoff_index) > 0:
                cutoff_index = cutoff_index[0].item()
                sorted_probs[cutoff_index:] = 0
                probs = torch.zeros_like(probs)
                probs[sorted_indices[:cutoff_index]] = sorted_probs[:cutoff_index]
                probs = probs / probs.sum()
            
            # Sample token
            next_token = torch.multinomial(probs, 1)
            log_prob = torch.log(probs[next_token] + 1e-10)
            
            generated_tokens.append(next_token.item())
            log_probs.append(log_prob.item())
            
            # Update current tokens
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode response
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'text': response_text,
            'tokens': torch.tensor(generated_tokens),
            'full_tokens': current_tokens,
            'log_probs': torch.tensor(log_probs)
        }
    
    def _apply_diversity_penalty(self, logits: torch.Tensor, existing_responses: List[Dict], 
                                step: int) -> torch.Tensor:
        """Apply diversity penalty to encourage different responses"""
        
        penalty_logits = logits.clone()
        
        for existing in existing_responses:
            existing_tokens = existing['response_tokens']
            if step < len(existing_tokens):
                # Penalize tokens that were used in existing responses
                token_id = existing_tokens[step].item()
                penalty_logits[token_id] -= self.config.diversity_penalty
        
        return penalty_logits
    
    def _nucleus_sampling(self, prompt_tokens: torch.Tensor, top_p: float) -> Dict:
        """Standard nucleus sampling"""
        
        current_tokens = prompt_tokens.clone()
        generated_tokens = []
        log_probs = []
        
        for _ in range(self.config.max_new_tokens):
            outputs = self.policy_model(current_tokens)
            logits = outputs.logits[0, -1, :] / self.config.temperature
            
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            cutoff_index = torch.where(cumulative_probs > top_p)[0]
            if len(cutoff_index) > 0:
                cutoff_index = cutoff_index[0].item()
                sorted_probs[cutoff_index:] = 0
                probs = torch.zeros_like(probs)
                probs[sorted_indices[:cutoff_index]] = sorted_probs[:cutoff_index]
                probs = probs / probs.sum()
            
            next_token = torch.multinomial(probs, 1)
            log_prob = torch.log(probs[next_token] + 1e-10)
            
            generated_tokens.append(next_token.item())
            log_probs.append(log_prob.item())
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'text': response_text,
            'tokens': torch.tensor(generated_tokens),
            'full_tokens': current_tokens,
            'log_probs': torch.tensor(log_probs)
        }
    
    def _beam_sampling(self, prompt_tokens: torch.Tensor, beam_idx: int) -> Dict:
        """Beam search sampling"""
        # Simplified beam search - just vary temperature
        temp = self.config.temperature * (1 + 0.1 * beam_idx)
        return self._nucleus_sampling(prompt_tokens, self.config.top_p)
    
    def _random_sampling(self, prompt_tokens: torch.Tensor) -> Dict:
        """Random sampling"""
        # High temperature sampling
        temp = self.config.temperature * 1.5
        return self._nucleus_sampling(prompt_tokens, 1.0)  # No top-p filtering
    
    def compute_rewards(self, responses: List[Dict]) -> torch.Tensor:
        """Compute rewards for a group of responses"""
        
        rewards = []
        
        with torch.no_grad():
            for response in responses:
                full_tokens = response['full_tokens']
                reward = self.reward_model.get_reward(full_tokens)
                
                # Length normalization if enabled
                if self.config.length_normalization:
                    response_length = len(response['response_tokens'])
                    reward = reward / (response_length ** 0.5) if response_length > 0 else reward
                
                rewards.append(reward)
        
        return torch.stack(rewards)
    
    def compute_ranking_loss(self, responses: List[Dict], rewards: torch.Tensor) -> torch.Tensor:
        """Compute ranking-based loss"""
        
        if self.config.ranking_loss_type == "pairwise":
            return self._pairwise_ranking_loss(responses, rewards)
        elif self.config.ranking_loss_type == "listwise":
            return self._listwise_ranking_loss(responses, rewards)
        elif self.config.ranking_loss_type == "contrastive":
            return self._contrastive_ranking_loss(responses, rewards)
        else:
            raise ValueError(f"Unknown ranking loss type: {self.config.ranking_loss_type}")
    
    def _pairwise_ranking_loss(self, responses: List[Dict], rewards: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss"""
        
        total_loss = 0
        num_pairs = 0
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # Get log probabilities for both responses
                log_probs_i = responses[i]['log_probs'].sum()
                log_probs_j = responses[j]['log_probs'].sum()
                
                # Reward difference
                reward_diff = rewards[i] - rewards[j]
                
                # Probability ratio
                prob_ratio = log_probs_i - log_probs_j
                
                # Ranking loss (higher reward should have higher probability)
                if reward_diff > 0:
                    # i should be preferred over j
                    loss = F.margin_ranking_loss(
                        prob_ratio.unsqueeze(0), 
                        torch.zeros(1, device=self.device),
                        torch.ones(1, device=self.device),
                        margin=self.config.margin
                    )
                else:
                    # j should be preferred over i
                    loss = F.margin_ranking_loss(
                        torch.zeros(1, device=self.device),
                        prob_ratio.unsqueeze(0),
                        torch.ones(1, device=self.device), 
                        margin=self.config.margin
                    )
                
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.device)
    
    def _listwise_ranking_loss(self, responses: List[Dict], rewards: torch.Tensor) -> torch.Tensor:
        """Listwise ranking loss using ListNet"""
        
        # Get log probabilities for all responses
        log_probs = torch.stack([r['log_probs'].sum() for r in responses])
        
        # Softmax over log probabilities (policy distribution)
        policy_probs = F.softmax(log_probs, dim=0)
        
        # Softmax over rewards (target distribution)
        target_probs = F.softmax(rewards, dim=0)
        
        # KL divergence loss
        loss = F.kl_div(torch.log(policy_probs + 1e-10), target_probs, reduction='batchmean')
        
        return loss
    
    def _contrastive_ranking_loss(self, responses: List[Dict], rewards: torch.Tensor) -> torch.Tensor:
        """Contrastive ranking loss"""
        
        # Find best and worst responses
        best_idx = torch.argmax(rewards)
        worst_idx = torch.argmin(rewards)
        
        # Get log probabilities
        best_log_prob = responses[best_idx]['log_probs'].sum()
        worst_log_prob = responses[worst_idx]['log_probs'].sum()
        
        # Contrastive loss: maximize gap between best and worst
        margin = self.config.margin
        loss = F.relu(margin - (best_log_prob - worst_log_prob))
        
        return loss
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages using baseline"""
        
        # Update baseline
        if self.config.use_baseline:
            current_avg_reward = rewards.mean().item()
            self.baseline = (self.config.baseline_momentum * self.baseline + 
                           (1 - self.config.baseline_momentum) * current_avg_reward)
        
        # Compute advantages
        advantages = rewards - self.baseline
        
        # Normalize advantages
        if self.config.use_advantage_normalization and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def compute_kl_penalty(self, responses: List[Dict]) -> torch.Tensor:
        """Compute KL penalty against reference model"""
        
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.device)
        
        total_kl = 0
        
        with torch.no_grad():
            for response in responses:
                full_tokens = response['full_tokens']
                
                # Current policy probabilities
                current_outputs = self.policy_model(full_tokens)
                current_logits = current_outputs.logits
                
                # Reference policy probabilities  
                ref_outputs = self.ref_model(full_tokens)
                ref_logits = ref_outputs.logits
                
                # KL divergence
                kl = F.kl_div(
                    F.log_softmax(current_logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction='batchmean'
                )
                
                total_kl += kl
        
        return total_kl / len(responses)
    
    def train_step(self, prompts: List[str]) -> Dict:
        """Single GRPO training step"""
        
        self.policy_model.train()
        total_loss = 0
        
        for prompt in prompts:
            # Generate group of responses
            responses = self.generate_group_responses(prompt)
            
            # Compute rewards
            rewards = self.compute_rewards(responses)
            
            # Compute advantages
            advantages = self.compute_advantages(rewards)
            
            # Compute ranking loss
            ranking_loss = self.compute_ranking_loss(responses, rewards)
            
            # Compute KL penalty
            kl_penalty = self.compute_kl_penalty(responses)
            
            # Compute entropy regularization
            entropy_loss = 0
            for response in responses:
                log_probs = response['log_probs']
                entropy_loss -= (log_probs * torch.exp(log_probs)).sum()
            entropy_loss = entropy_loss / len(responses)
            
            # Total loss
            loss = (ranking_loss + 
                   self.config.beta * kl_penalty + 
                   self.config.entropy_regularization * entropy_loss)
            
            total_loss += loss
        
        # Backward pass
        total_loss = total_loss / len(prompts)
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update statistics
        self.stats.update({
            'total_episodes': self.stats['total_episodes'] + len(prompts) * self.config.group_size,
            'avg_reward': rewards.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'ranking_loss': ranking_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'entropy': -entropy_loss.item(),
            'baseline': self.baseline
        })
        
        return self.stats
    
    def train(self, prompts_dataset: List[str], num_epochs: int = 10, 
              prompts_per_epoch: int = 50):
        """Full GRPO training loop"""
        
        print(f"Starting GRPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Sample prompts for this epoch
            sampled_prompts = np.random.choice(
                prompts_dataset,
                size=min(prompts_per_epoch, len(prompts_dataset)),
                replace=False
            ).tolist()
            
            # Split into batches
            for i in range(0, len(sampled_prompts), self.config.batch_size):
                batch_prompts = sampled_prompts[i:i + self.config.batch_size]
                stats = self.train_step(batch_prompts)
            
            # Logging
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Avg Reward: {stats['avg_reward']:.4f}")
            print(f"  Avg Advantage: {stats['avg_advantage']:.4f}")
            print(f"  Ranking Loss: {stats['ranking_loss']:.4f}")
            print(f"  KL Penalty: {stats['kl_penalty']:.4f}")
            print(f"  Entropy: {stats['entropy']:.4f}")
            print(f"  Baseline: {stats['baseline']:.4f}")
            print("-" * 50)

# Utility functions
def create_grpo_trainer(policy_model, reward_model: RewardModel, tokenizer,
                       config: GRPOConfig = None, ref_model=None):
    """Factory function to create GRPO trainer"""
    
    if config is None:
        config = GRPOConfig()
    
    return GRPOTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        config=config,
        tokenizer=tokenizer,
        ref_model=ref_model
    )

def train_grpo_pipeline(policy_model, reward_model: RewardModel, tokenizer,
                       prompts_dataset: List[str], config: GRPOConfig = None,
                       ref_model=None):
    """Complete GRPO training pipeline"""
    
    # Create GRPO trainer
    grpo_trainer = create_grpo_trainer(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
        ref_model=ref_model
    )
    
    # Train
    grpo_trainer.train(prompts_dataset, num_epochs=15, prompts_per_epoch=40)
    
    return grpo_trainer

def evaluate_grpo_model(grpo_trainer: GRPOTrainer, test_prompts: List[str]) -> Dict:
    """Evaluate GRPO-trained model"""
    
    all_responses = []
    all_rewards = []
    
    for prompt in test_prompts:
        # Generate group responses
        responses = grpo_trainer.generate_group_responses(prompt)
        
        # Compute rewards
        rewards = grpo_trainer.compute_rewards(responses)
        
        all_responses.extend(responses)
        all_rewards.extend(rewards.tolist())
    
    # Calculate metrics
    avg_reward = np.mean(all_rewards)
    reward_std = np.std(all_rewards)
    
    # Group-level metrics (best response per prompt)
    group_best_rewards = []
    for i in range(0, len(all_rewards), grpo_trainer.config.group_size):
        group_rewards = all_rewards[i:i + grpo_trainer.config.group_size]
        group_best_rewards.append(max(group_rewards))
    
    return {
        'avg_reward': avg_reward,
        'reward_std': reward_std,
        'avg_best_reward': np.mean(group_best_rewards),
        'best_reward_std': np.std(group_best_rewards),
        'total_responses': len(all_responses),
        'total_prompts': len(test_prompts),
        'responses_per_prompt': grpo_trainer.config.group_size,
        'reward_distribution': {
            'min': np.min(all_rewards),
            'max': np.max(all_rewards),
            'median': np.median(all_rewards),
            'p25': np.percentile(all_rewards, 25),
            'p75': np.percentile(all_rewards, 75),
            'p90': np.percentile(all_rewards, 90)
        }
    }

# Export GRPO components
__all__ = [
    'GRPOConfig', 'GRPOTrainer',
    'create_grpo_trainer', 'train_grpo_pipeline', 'evaluate_grpo_model'
]