import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RewardModelConfig:
    """Configuration for reward model training"""
    model_name: str = "reward_model"
    hidden_size: int = 768
    num_labels: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_length: int = 512
    dropout: float = 0.1

class ProcessRewardModel(nn.Module):
    """Process Reward Model for step-by-step evaluation"""
    
    def __init__(self, base_model, config: RewardModelConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Reward head for each step
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask=None, step_positions=None):
        """
        Forward pass for PRM
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            step_positions: Positions of reasoning steps
        """
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Extract step representations
        if step_positions is not None:
            step_rewards = []
            for i, positions in enumerate(step_positions):
                batch_rewards = []
                for pos in positions:
                    if pos < hidden_states.shape[1]:
                        step_hidden = hidden_states[i, pos, :]
                        reward = self.reward_head(step_hidden)
                        batch_rewards.append(reward)
                step_rewards.append(torch.stack(batch_rewards) if batch_rewards else torch.tensor([]))
            return step_rewards
        else:
            # Return rewards for all positions
            rewards = self.reward_head(hidden_states)
            return rewards

class OutcomeRewardModel(nn.Module):
    """Outcome Reward Model for final result evaluation"""
    
    def __init__(self, base_model, config: RewardModelConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Single reward head for final outcome
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass for ORM"""
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token or last token for classification
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Use last token
            hidden_states = outputs.last_hidden_state
            pooled_output = hidden_states[:, -1, :]
        
        # Get reward score
        reward = self.reward_head(pooled_output)
        return reward

def train_process_reward_model(
    base_model,
    train_data: List[Dict[str, Any]],
    tokenizer,
    config: Optional[RewardModelConfig] = None,
    **kwargs
):
    """
    Train Process Reward Model
    
    Args:
        base_model: Base language model
        train_data: Training data with step-by-step annotations
        tokenizer: Tokenizer
        config: Training configuration
    
    Expected data format:
    [
        {
            'prompt': 'Solve: 2x + 5 = 11',
            'steps': [
                {'step_text': 'Subtract 5 from both sides: 2x = 6', 'reward': 0.9},
                {'step_text': 'Divide by 2: x = 3', 'reward': 0.95}
            ]
        }
    ]
    """
    if config is None:
        config = RewardModelConfig()
    
    # Initialize PRM
    prm = ProcessRewardModel(base_model, config)
    optimizer = torch.optim.AdamW(prm.parameters(), lr=config.learning_rate)
    
    device = next(base_model.parameters()).device
    prm.to(device)
    
    print(f"Training Process Reward Model with {len(train_data)} examples...")
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        
        for batch_idx, example in enumerate(train_data):
            optimizer.zero_grad()
            
            # Prepare input
            prompt = example['prompt']
            steps = example['steps']
            
            # Create full text with steps
            full_text = prompt + "\n"
            step_positions = []
            target_rewards = []
            
            current_pos = len(tokenizer.encode(full_text))
            
            for step in steps:
                step_text = step['step_text']
                step_reward = step['reward']
                
                full_text += step_text + "\n"
                
                # Track position and reward
                step_positions.append(current_pos)
                target_rewards.append(step_reward)
                
                # Update position
                current_pos = len(tokenizer.encode(full_text))
            
            # Tokenize
            encoding = tokenizer(
                full_text,
                truncation=True,
                padding=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass
            step_rewards = prm(input_ids, attention_mask, [step_positions])
            
            # Compute loss
            if step_rewards and len(step_rewards[0]) > 0:
                predicted_rewards = step_rewards[0].squeeze()
                target_rewards_tensor = torch.tensor(target_rewards, device=device, dtype=torch.float)
                
                # Ensure same length
                min_len = min(len(predicted_rewards), len(target_rewards_tensor))
                if min_len > 0:
                    loss = F.mse_loss(predicted_rewards[:min_len], target_rewards_tensor[:min_len])
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return prm

def train_outcome_reward_model(
    base_model,
    train_data: List[Dict[str, Any]],
    tokenizer,
    config: Optional[RewardModelConfig] = None,
    **kwargs
):
    """
    Train Outcome Reward Model
    
    Args:
        base_model: Base language model
        train_data: Training data with outcome ratings
        tokenizer: Tokenizer
        config: Training configuration
    
    Expected data format:
    [
        {
            'prompt': 'What is 2+2?',
            'response': '2+2 = 4',
            'reward': 1.0
        },
        {
            'prompt': 'What is 2+2?', 
            'response': '2+2 = 5',
            'reward': 0.0
        }
    ]
    """
    if config is None:
        config = RewardModelConfig()
    
    # Initialize ORM
    orm = OutcomeRewardModel(base_model, config)
    optimizer = torch.optim.AdamW(orm.parameters(), lr=config.learning_rate)
    
    device = next(base_model.parameters()).device
    orm.to(device)
    
    print(f"Training Outcome Reward Model with {len(train_data)} examples...")
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        
        for batch_idx, example in enumerate(train_data):
            optimizer.zero_grad()
            
            # Prepare input
            prompt = example['prompt']
            response = example['response']
            target_reward = example['reward']
            
            # Combine prompt and response
            full_text = f"{prompt}\n{response}"
            
            # Tokenize
            encoding = tokenizer(
                full_text,
                truncation=True,
                padding=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass
            predicted_reward = orm(input_ids, attention_mask)
            
            # Compute loss
            target_reward_tensor = torch.tensor([[target_reward]], device=device, dtype=torch.float)
            loss = F.mse_loss(predicted_reward, target_reward_tensor)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return orm

def create_process_reward_dataset(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create process reward dataset from reasoning examples
    
    Args:
        examples: List of reasoning examples with steps
    """
    dataset = []
    
    for example in examples:
        if 'steps' in example and 'prompt' in example:
            dataset.append({
                'prompt': example['prompt'],
                'steps': example['steps']
            })
    
    return dataset

def create_outcome_reward_dataset(
    prompts: List[str],
    responses: List[str], 
    rewards: List[float]
) -> List[Dict[str, Any]]:
    """
    Create outcome reward dataset from prompt-response pairs
    
    Args:
        prompts: List of prompts
        responses: List of responses
        rewards: List of reward scores
    """
    dataset = []
    
    for prompt, response, reward in zip(prompts, responses, rewards):
        dataset.append({
            'prompt': prompt,
            'response': response,
            'reward': reward
        })
    
    return dataset

def evaluate_reward_model(
    model,
    test_data: List[Dict[str, Any]],
    tokenizer,
    model_type: str = "outcome"
):
    """
    Evaluate trained reward model
    
    Args:
        model: Trained reward model (PRM or ORM)
        test_data: Test dataset
        tokenizer: Tokenizer
        model_type: "process" or "outcome"
    """
    model.eval()
    device = next(model.parameters()).device
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for example in test_data:
            if model_type == "outcome":
                # ORM evaluation
                prompt = example['prompt']
                response = example['response']
                target = example['reward']
                
                full_text = f"{prompt}\n{response}"
                
                encoding = tokenizer(
                    full_text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                pred_reward = model(input_ids, attention_mask)
                predictions.append(pred_reward.item())
                targets.append(target)
                
            elif model_type == "process":
                # PRM evaluation
                prompt = example['prompt']
                steps = example['steps']
                
                # Simplified evaluation for PRM
                for step in steps:
                    target = step['reward']
                    targets.append(target)
                    predictions.append(0.5)  # Placeholder
    
    # Calculate metrics
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'predictions': predictions,
        'targets': targets
    }