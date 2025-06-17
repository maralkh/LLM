# training_infra/rlhf/reward_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from ..trainer import Trainer
from ..config import TrainingConfig

@dataclass
class RewardModelConfig:
    """Configuration for reward model training"""
    # Model architecture
    base_model_name: str = "llama_7b"
    hidden_size: int = 4096
    dropout: float = 0.1
    
    # Training specific
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_length: int = 512
    
    # Ranking loss
    loss_type: str = "ranking"  # "ranking", "regression", "classification"
    margin: float = 0.1
    
    # Data augmentation
    use_data_augmentation: bool = True
    noise_std: float = 0.01

class RewardModel(nn.Module):
    """Reward model for RLHF"""
    
    def __init__(self, base_model, config: RewardModelConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Freeze base model if needed
        for param in self.base_model.parameters():
            param.requires_grad = True  # Keep trainable for now
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.loss_type = config.loss_type
        self.margin = config.margin
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to get reward scores
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            rewards: [batch_size, 1]
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Use last hidden state, take last token or mean pooling
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs['last_hidden_state']
        
        # Mean pooling over sequence length (masked)
        if attention_mask is not None:
            # Mask out padding tokens
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Get reward score
        rewards = self.reward_head(pooled)
        return rewards
    
    def compute_loss(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor,
                    chosen_mask: Optional[torch.Tensor] = None, 
                    rejected_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute ranking loss between chosen and rejected responses
        
        Args:
            chosen_ids: [batch_size, seq_len] - preferred responses
            rejected_ids: [batch_size, seq_len] - rejected responses
            
        Returns:
            loss: scalar tensor
        """
        # Get reward scores
        chosen_rewards = self.forward(chosen_ids, chosen_mask)
        rejected_rewards = self.forward(rejected_ids, rejected_mask)
        
        if self.loss_type == "ranking":
            # Ranking loss: chosen should have higher reward than rejected
            loss = F.margin_ranking_loss(
                chosen_rewards.squeeze(-1), 
                rejected_rewards.squeeze(-1),
                torch.ones_like(chosen_rewards.squeeze(-1)),
                margin=self.margin
            )
        elif self.loss_type == "regression":
            # MSE loss (assuming labels are provided)
            # This would require additional labels in the input
            raise NotImplementedError("Regression loss requires reward labels")
        elif self.loss_type == "classification":
            # Binary classification loss
            logits = torch.cat([chosen_rewards, rejected_rewards], dim=0)
            targets = torch.cat([
                torch.ones(chosen_rewards.size(0), device=chosen_rewards.device),
                torch.zeros(rejected_rewards.size(0), device=rejected_rewards.device)
            ])
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def get_reward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get reward score for generation"""
        self.eval()
        with torch.no_grad():
            rewards = self.forward(input_ids, attention_mask)
        return rewards.squeeze(-1)

class RewardModelTrainer(Trainer):
    """Specialized trainer for reward models"""
    
    def __init__(self, reward_model: RewardModel, config: TrainingConfig, 
                 train_dataloader, val_dataloader=None, **kwargs):
        super().__init__(reward_model, config, train_dataloader, val_dataloader, **kwargs)
        self.reward_model = reward_model
    
    def compute_loss(self, batch):
        """Compute reward model loss"""
        # Expect batch to contain chosen and rejected responses
        chosen_ids = batch['chosen_input_ids']
        rejected_ids = batch['rejected_input_ids']
        chosen_mask = batch.get('chosen_attention_mask', None)
        rejected_mask = batch.get('rejected_attention_mask', None)
        
        loss = self.reward_model.compute_loss(chosen_ids, rejected_ids, chosen_mask, rejected_mask)
        return loss
    
    def validate(self):
        """Validate reward model"""
        if self.val_dataloader is None:
            return {}
        
        self.reward_model.eval()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                
                # Compute loss
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                
                # Compute accuracy (chosen > rejected)
                chosen_rewards = self.reward_model.forward(
                    batch['chosen_input_ids'], 
                    batch.get('chosen_attention_mask', None)
                )
                rejected_rewards = self.reward_model.forward(
                    batch['rejected_input_ids'], 
                    batch.get('rejected_attention_mask', None)
                )
                
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                total_accuracy += accuracy.item()
                total_samples += 1
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy,
            'val_reward_margin': (chosen_rewards - rejected_rewards).mean().item()
        }

class RewardDataset(torch.utils.data.Dataset):
    """Dataset for reward model training"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """
        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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
        
        # Tokenize
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length', 
            max_length=self.max_length,
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

def create_reward_model(base_model, config: RewardModelConfig = None):
    """Factory function to create reward model"""
    if config is None:
        config = RewardModelConfig()
    
    return RewardModel(base_model, config)

def train_reward_model(base_model, train_data: List[Dict], val_data: List[Dict] = None,
                      tokenizer=None, config: RewardModelConfig = None):
    """
    Complete reward model training pipeline
    
    Args:
        base_model: Pre-trained language model
        train_data: Training data with prompt/chosen/rejected
        val_data: Validation data
        tokenizer: Tokenizer
        config: Reward model config
        
    Returns:
        Trained reward model
    """
    if config is None:
        config = RewardModelConfig()
    
    # Create reward model
    reward_model = create_reward_model(base_model, config)
    
    # Create datasets
    train_dataset = RewardDataset(train_data, tokenizer, config.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    val_loader = None
    if val_data:
        val_dataset = RewardDataset(val_data, tokenizer, config.max_length)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
    
    # Create training config
    training_config = TrainingConfig(
        epochs=3,
        batch_size=config.batch_size,
        optimizer=TrainingConfig.OptimizerConfig(
            name="adamw",
            lr=config.learning_rate,
            weight_decay=0.01
        ),
        logging=TrainingConfig.LoggingConfig(
            log_every=50,
            use_tensorboard=True
        )
    )
    
    # Train
    trainer = RewardModelTrainer(
        reward_model=reward_model,
        config=training_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    trainer.fit()
    
    return reward_model

# Utility functions for reward model evaluation
def evaluate_reward_model_correlation(reward_model, test_data: List[Dict], tokenizer):
    """Evaluate correlation between reward model and human preferences"""
    reward_model.eval()
    
    human_preferences = []  # 1 if chosen > rejected, 0 otherwise
    model_preferences = []
    
    with torch.no_grad():
        for item in test_data:
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']
            
            # Get model predictions
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            chosen_ids = tokenizer(chosen_text, return_tensors='pt', truncation=True, max_length=512)
            rejected_ids = tokenizer(rejected_text, return_tensors='pt', truncation=True, max_length=512)
            
            chosen_reward = reward_model.get_reward(chosen_ids['input_ids'], chosen_ids['attention_mask'])
            rejected_reward = reward_model.get_reward(rejected_ids['input_ids'], rejected_ids['attention_mask'])
            
            # Human preference (assuming chosen is always preferred in training data)
            human_preferences.append(1)
            
            # Model preference
            model_preferences.append(1 if chosen_reward > rejected_reward else 0)
    
    # Calculate accuracy
    accuracy = sum(h == m for h, m in zip(human_preferences, model_preferences)) / len(human_preferences)
    
    return {
        'accuracy': accuracy,
        'total_examples': len(human_preferences),
        'model_correct': sum(model_preferences),
        'human_correct': sum(human_preferences)
    }

def analyze_reward_distribution(reward_model, texts: List[str], tokenizer):
    """Analyze reward score distribution"""
    reward_model.eval()
    rewards = []
    
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            reward = reward_model.get_reward(ids['input_ids'], ids['attention_mask'])
            rewards.append(reward.item())
    
    rewards = np.array(rewards)
    
    return {
        'mean': rewards.mean(),
        'std': rewards.std(),
        'min': rewards.min(),
        'max': rewards.max(),
        'percentiles': {
            '25': np.percentile(rewards, 25),
            '50': np.percentile(rewards, 50),
            '75': np.percentile(rewards, 75),
            '90': np.percentile(rewards, 90),
            '95': np.percentile(rewards, 95),
        }
    }