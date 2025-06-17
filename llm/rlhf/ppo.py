# training_infra/rlhf/ppo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import copy

from ..trainer import Trainer
from ..config import TrainingConfig
from .reward_model import RewardModel

@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    
    # Training parameters
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    rollout_batch_size: int = 16
    max_length: int = 512
    max_new_tokens: int = 128
    
    # KL divergence control
    target_kl: float = 0.01
    kl_coef: float = 0.1
    adaptive_kl: bool = True
    kl_coef_decay: float = 0.999
    
    # Reward shaping
    reward_baseline: float = 0.0
    reward_scale: float = 1.0
    use_whitening: bool = True
    
    # Experience replay
    buffer_size: int = 1000
    min_buffer_size: int = 100

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.clear()
    
    def clear(self):
        """Clear the buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, reward, value, log_prob, done):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        self.advantages = []
        self.returns = []
        
        rewards = self.rewards + [next_value]
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = rewards[step] + self.config.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - self.dones[step]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[step])
        
        # Normalize advantages
        if self.config.use_whitening and len(self.advantages) > 1:
            advantages = torch.tensor(self.advantages)
            self.advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).tolist()
    
    def get_batch(self):
        """Get all experiences as batch"""
        return {
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'rewards': torch.tensor(self.rewards),
            'values': torch.tensor(self.values),
            'log_probs': torch.tensor(self.log_probs),
            'advantages': torch.tensor(self.advantages),
            'returns': torch.tensor(self.returns)
        }
    
    def __len__(self):
        return len(self.states)

class PPOActor(nn.Module):
    """Actor model for PPO (policy network)"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through actor"""
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def get_action_probs(self, input_ids, attention_mask=None):
        """Get action probabilities"""
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        return F.softmax(logits, dim=-1)
    
    def get_log_probs(self, input_ids, action_ids, attention_mask=None):
        """Get log probabilities for given actions"""
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for taken actions
        action_log_probs = torch.gather(log_probs, -1, action_ids.unsqueeze(-1)).squeeze(-1)
        
        return action_log_probs

class PPOCritic(nn.Module):
    """Critic model for PPO (value network)"""
    
    def __init__(self, base_model, hidden_size: int = 4096):
        super().__init__()
        self.base_model = base_model
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through critic"""
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs['last_hidden_state']
        
        # Mean pooling
        if attention_mask is not None:
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Get value estimate
        values = self.value_head(pooled)
        return values.squeeze(-1)

class PPOTrainer:
    """PPO Trainer for RLHF"""
    
    def __init__(self, 
                 actor_model,
                 critic_model, 
                 reward_model: RewardModel,
                 config: PPOConfig,
                 tokenizer=None,
                 ref_model=None):
        
        self.config = config
        self.tokenizer = tokenizer
        
        # Models
        self.actor = PPOActor(actor_model)
        self.critic = PPOCritic(critic_model)
        self.reward_model = reward_model
        self.ref_model = PPOActor(ref_model) if ref_model else None
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # Freeze reference model
        if self.ref_model:
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-4)
        
        # Experience buffer
        self.buffer = PPOBuffer(config)
        
        # Metrics tracking
        self.stats = {
            'total_episodes': 0,
            'avg_reward': 0.0,
            'avg_kl': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
        
        # Device
        self.device = next(self.actor.parameters()).device
    
    def generate_response(self, prompts: List[str], max_new_tokens: int = None) -> List[Dict]:
        """Generate responses using current policy"""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        self.actor.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                prompt_length = prompt_ids.shape[1]
                
                # Generate response
                generated_ids = []
                current_ids = prompt_ids
                
                for _ in range(max_new_tokens):
                    # Get action probabilities
                    outputs = self.actor.forward(current_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                    next_token_logits = logits[0, -1, :]
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    generated_ids.append(next_token.item())
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                    
                    # Stop if EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # Decode response
                response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                responses.append({
                    'prompt': prompt,
                    'response': response_text,
                    'prompt_ids': prompt_ids,
                    'response_ids': torch.tensor(generated_ids),
                    'full_ids': current_ids
                })
        
        return responses
    
    def compute_rewards(self, responses: List[Dict]) -> List[float]:
        """Compute rewards for responses using reward model"""
        rewards = []
        
        self.reward_model.eval()
        with torch.no_grad():
            for response_data in responses:
                full_ids = response_data['full_ids']
                reward = self.reward_model.get_reward(full_ids)
                rewards.append(reward.item())
        
        return rewards
    
    def compute_kl_penalty(self, responses: List[Dict]) -> List[float]:
        """Compute KL divergence penalty against reference model"""
        if self.ref_model is None:
            return [0.0] * len(responses)
        
        kl_penalties = []
        
        with torch.no_grad():
            for response_data in responses:
                full_ids = response_data['full_ids']
                
                # Current policy probabilities
                current_probs = self.actor.get_action_probs(full_ids)
                
                # Reference policy probabilities
                ref_probs = self.ref_model.get_action_probs(full_ids)
                
                # KL divergence
                kl_div = F.kl_div(
                    F.log_softmax(current_probs, dim=-1),
                    F.softmax(ref_probs, dim=-1),
                    reduction='batchmean'
                )
                
                kl_penalties.append(kl_div.item())
        
        return kl_penalties
    
    def collect_experiences(self, prompts: List[str]) -> Dict:
        """Collect experiences for PPO update"""
        # Generate responses
        responses = self.generate_response(prompts)
        
        # Compute rewards
        rewards = self.compute_rewards(responses)
        
        # Compute KL penalties
        kl_penalties = self.compute_kl_penalty(responses)
        
        # Compute final rewards (reward - kl_penalty)
        final_rewards = [
            r - self.config.kl_coef * kl 
            for r, kl in zip(rewards, kl_penalties)
        ]
        
        # Add experiences to buffer
        for i, (response_data, reward) in enumerate(zip(responses, final_rewards)):
            full_ids = response_data['full_ids']
            
            # Get value estimate
            value = self.critic.forward(full_ids).item()
            
            # Get log probability (simplified - would need actual action sequence)
            log_prob = 0.0  # Placeholder - implement proper log prob calculation
            
            # Add to buffer
            self.buffer.add(
                state=full_ids,
                action=response_data['response_ids'],
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=True  # Each response is complete episode
            )
        
        return {
            'responses': responses,
            'rewards': rewards,
            'kl_penalties': kl_penalties,
            'final_rewards': final_rewards
        }
    
    def ppo_update(self):
        """Perform PPO update"""
        if len(self.buffer) < self.config.min_buffer_size:
            return
        
        # Compute advantages
        self.buffer.compute_gae()
        
        # Get batch data
        batch = self.buffer.get_batch()
        
        # PPO epochs
        for _ in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(batch['states']))
            
            # Mini-batch updates
            for start in range(0, len(indices), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_indices = indices[start:end]
                
                if len(mb_indices) < self.config.mini_batch_size:
                    continue
                
                # Get mini-batch
                mb_states = batch['states'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_advantages = batch['advantages'][mb_indices]
                mb_returns = batch['returns'][mb_indices]
                mb_old_log_probs = batch['log_probs'][mb_indices]
                
                # Actor update
                self.actor_optimizer.zero_grad()
                
                # Get current log probabilities
                current_log_probs = self.actor.get_log_probs(mb_states, mb_actions)
                
                # Compute ratio
                ratio = torch.exp(current_log_probs - mb_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_advantages
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy = -current_log_probs.mean()
                entropy_loss = -self.config.entropy_coef * entropy
                
                # Total actor loss
                actor_loss = policy_loss + entropy_loss
                actor_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic update
                self.critic_optimizer.zero_grad()
                
                # Get current values
                current_values = self.critic.forward(mb_states)
                
                # Value loss
                value_loss = F.mse_loss(current_values, mb_returns)
                value_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                # Update stats
                self.stats['policy_loss'] = policy_loss.item()
                self.stats['value_loss'] = value_loss.item()
                self.stats['entropy'] = entropy.item()
        
        # Clear buffer
        self.buffer.clear()
    
    def train_step(self, prompts: List[str]):
        """Single training step"""
        # Collect experiences
        experiences = self.collect_experiences(prompts)
        
        # Update models
        self.ppo_update()
        
        # Update stats
        self.stats['total_episodes'] += len(prompts)
        self.stats['avg_reward'] = np.mean(experiences['final_rewards'])
        self.stats['avg_kl'] = np.mean(experiences['kl_penalties'])
        
        # Adaptive KL coefficient
        if self.config.adaptive_kl:
            avg_kl = self.stats['avg_kl']
            if avg_kl > 2 * self.config.target_kl:
                self.config.kl_coef *= 1.5
            elif avg_kl < self.config.target_kl / 2:
                self.config.kl_coef *= 0.5
            
            # Decay KL coefficient
            self.config.kl_coef *= self.config.kl_coef_decay
        
        return self.stats
    
    def train(self, prompts_dataset: List[str], num_epochs: int = 10, 
              prompts_per_epoch: int = 100):
        """Full training loop"""
        
        print(f"Starting PPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Sample prompts for this epoch
            sampled_prompts = np.random.choice(
                prompts_dataset, 
                size=min(prompts_per_epoch, len(prompts_dataset)),
                replace=False
            ).tolist()
            
            # Training step
            stats = self.train_step(sampled_prompts)
            
            # Logging
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Avg Reward: {stats['avg_reward']:.4f}")
            print(f"  Avg KL: {stats['avg_kl']:.4f}")
            print(f"  Policy Loss: {stats['policy_loss']:.4f}")
            print(f"  Value Loss: {stats['value_loss']:.4f}")
            print(f"  Entropy: {stats['entropy']:.4f}")
            print(f"  KL Coef: {self.config.kl_coef:.6f}")
            print("-" * 50)
    
    def save_checkpoint(self, filepath: str):
        """Save PPO checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'stats': self.stats
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load PPO checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.stats = checkpoint['stats']

# Utility functions
def create_ppo_trainer(base_model, reward_model: RewardModel, tokenizer, 
                      config: PPOConfig = None, ref_model=None):
    """Factory function to create PPO trainer"""
    if config is None:
        config = PPOConfig()
    
    # Create separate models for actor and critic
    actor_model = copy.deepcopy(base_model)
    critic_model = copy.deepcopy(base_model)
    
    return PPOTrainer(
        actor_model=actor_model,
        critic_model=critic_model, 
        reward_model=reward_model,
        config=config,
        tokenizer=tokenizer,
        ref_model=ref_model
    )

def load_prompts_dataset(filepath: str) -> List[str]:
    """Load prompts dataset"""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'prompts' in data:
        return data['prompts']
    else:
        raise ValueError("Unsupported dataset format")

# Example training pipeline
def train_ppo_pipeline(base_model, reward_model: RewardModel, tokenizer,
                      prompts_dataset: List[str], config: PPOConfig = None):
    """Complete PPO training pipeline"""
    
    # Create reference model (frozen copy of base model)
    ref_model = copy.deepcopy(base_model)
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Create PPO trainer
    ppo_trainer = create_ppo_trainer(
        base_model=base_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
        ref_model=ref_model
    )
    
    # Train
    ppo_trainer.train(prompts_dataset, num_epochs=20, prompts_per_epoch=50)
    
    return ppo_trainer

def evaluate_ppo_model(ppo_trainer: PPOTrainer, test_prompts: List[str]) -> Dict:
    """Evaluate PPO-trained model"""
    
    # Generate responses
    responses = ppo_trainer.generate_response(test_prompts)
    
    # Compute rewards
    rewards = ppo_trainer.compute_rewards(responses)
    
    # Compute KL divergences
    kl_divs = ppo_trainer.compute_kl_penalty(responses)
    
    return {
        'responses': responses,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_kl': np.mean(kl_divs),
        'reward_distribution': {
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards),
            'p25': np.percentile(rewards, 25),
            'p75': np.percentile(rewards, 75)
        }
    }

# Advanced PPO variants
class PPOWithClipping(PPOTrainer):
    """PPO with advanced clipping strategies"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_schedule = lambda epoch: max(0.01, self.config.clip_epsilon * (0.99 ** epoch))
    
    def ppo_update(self, epoch=0):
        """PPO update with scheduled clipping"""
        # Update clipping coefficient
        self.config.clip_epsilon = self.clip_schedule(epoch)
        
        # Call parent update
        super().ppo_update()

class PPOWithRewardShaping(PPOTrainer):
    """PPO with reward shaping"""
    
    def __init__(self, *args, shaping_functions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shaping_functions = shaping_functions or []
    
    def compute_rewards(self, responses: List[Dict]) -> List[float]:
        """Compute rewards with shaping"""
        # Base rewards
        base_rewards = super().compute_rewards(responses)
        
        # Apply shaping functions
        shaped_rewards = []
        for i, (base_reward, response_data) in enumerate(zip(base_rewards, responses)):
            shaped_reward = base_reward
            
            for shaping_fn in self.shaping_functions:
                shaped_reward += shaping_fn(response_data, base_reward)
            
            shaped_rewards.append(shaped_reward)
        
        return shaped_rewards

# Reward shaping functions
def length_penalty(response_data: Dict, base_reward: float, 
                  target_length: int = 100, penalty_scale: float = 0.1) -> float:
    """Penalty for responses that are too long or too short"""
    response_length = len(response_data['response_ids'])
    length_diff = abs(response_length - target_length)
    return -penalty_scale * length_diff / target_length

def repetition_penalty(response_data: Dict, base_reward: float,
                      penalty_scale: float = 0.2) -> float:
    """Penalty for repetitive responses"""
    response_ids = response_data['response_ids']
    unique_tokens = len(set(response_ids.tolist()))
    total_tokens = len(response_ids)
    
    if total_tokens == 0:
        return 0
    
    repetition_ratio = 1 - (unique_tokens / total_tokens)
    return -penalty_scale * repetition_ratio

def coherence_reward(response_data: Dict, base_reward: float,
                    reward_scale: float = 0.1) -> float:
    """Reward for coherent responses (placeholder implementation)"""
    # This would require a coherence scoring model
    # For now, return a small positive reward
    return reward_scale * 0.1

# Export all PPO components
__all__ = [
    'PPOConfig', 'PPOTrainer', 'PPOBuffer', 'PPOActor', 'PPOCritic',
    'create_ppo_trainer', 'train_ppo_pipeline', 'evaluate_ppo_model',
    'PPOWithClipping', 'PPOWithRewardShaping',
    'length_penalty', 'repetition_penalty', 'coherence_reward'
]