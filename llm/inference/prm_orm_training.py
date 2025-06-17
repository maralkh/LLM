# training_infra/rlhf/prm_orm_training.py
"""
Training Process Reward Models (PRM) and Outcome Reward Models (ORM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import numpy as np

from ..trainer import Trainer
from ..config import TrainingConfig
from .reward_model import RewardModel, RewardModelConfig
from ..inference.reward_guided import ProcessRewardModel, OutcomeRewardModel

class ProcessRewardDataset(Dataset):
    """Dataset for training Process Reward Models"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """
        Args:
            data: List of dicts with step-by-step reasoning
                Format: {
                    'prompt': str,
                    'steps': [
                        {'step_text': str, 'reward': float, 'is_correct': bool},
                        ...
                    ],
                    'final_answer': str
                }
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process data into step-level examples
        self.step_examples = []
        self._process_data()
    
    def _process_data(self):
        """Process reasoning chains into step-level examples"""
        
        for item in self.data:
            prompt = item['prompt']
            steps = item['steps']
            
            # Build reasoning chain progressively
            current_text = prompt
            
            for i, step in enumerate(steps):
                step_text = step['step_text']
                reward = step.get('reward', 1.0 if step.get('is_correct', True) else -1.0)
                
                # Add step to current text
                current_text += f" {step_text}"
                
                # Create training example
                self.step_examples.append({
                    'text': current_text,
                    'step_position': len(self.tokenizer.encode(current_text)) - 1,
                    'step_reward': reward,
                    'step_index': i,
                    'is_final': i == len(steps) - 1
                })
    
    def __len__(self):
        return len(self.step_examples)
    
    def __getitem__(self, idx):
        example = self.step_examples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            example['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'step_position': min(example['step_position'], self.max_length - 1),
            'step_reward': torch.tensor(example['step_reward'], dtype=torch.float),
            'step_index': torch.tensor(example['step_index'], dtype=torch.long),
            'is_final': torch.tensor(example['is_final'], dtype=torch.bool)
        }

class OutcomeRewardDataset(Dataset):
    """Dataset for training Outcome Reward Models"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """
        Args:
            data: List of dicts with complete reasoning
                Format: {
                    'prompt': str,
                    'reasoning': str,
                    'final_answer': str,
                    'correctness': float,  # 0-1 score
                    'helpfulness': float,  # -1 to 1 score
                    'overall_quality': float  # Overall reward
                }
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine prompt, reasoning, and answer
        full_text = f"{item['prompt']} {item['reasoning']} {item['final_answer']}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'correctness': torch.tensor(item['correctness'], dtype=torch.float),
            'helpfulness': torch.tensor(item['helpfulness'], dtype=torch.float),
            'overall_quality': torch.tensor(item['overall_quality'], dtype=torch.float)
        }

class PRMTrainer(Trainer):
    """Trainer for Process Reward Models"""
    
    def __init__(self, prm_model: ProcessRewardModel, config: TrainingConfig,
                 train_dataloader, val_dataloader=None, **kwargs):
        super().__init__(prm_model, config, train_dataloader, val_dataloader, **kwargs)
        self.prm_model = prm_model
    
    def compute_loss(self, batch):
        """Compute PRM training loss"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        step_positions = batch['step_position']
        target_rewards = batch['step_reward']
        
        # Get PRM outputs
        outputs = self.prm_model.forward(input_ids)
        
        # Extract step rewards at specified positions
        batch_size = input_ids.size(0)
        predicted_rewards = []
        predicted_confidences = []
        
        for i in range(batch_size):
            pos = step_positions[i].item()
            if pos < outputs['step_rewards'].size(1):
                predicted_rewards.append(outputs['step_rewards'][i, pos])
                predicted_confidences.append(outputs['step_confidences'][i, pos])
            else:
                # Fallback to last position
                predicted_rewards.append(outputs['step_rewards'][i, -1])
                predicted_confidences.append(outputs['step_confidences'][i, -1])
        
        predicted_rewards = torch.stack(predicted_rewards)
        predicted_confidences = torch.stack(predicted_confidences)
        
        # Reward prediction loss (MSE)
        reward_loss = F.mse_loss(predicted_rewards, target_rewards)
        
        # Confidence loss (encourage high confidence for correct predictions)
        reward_errors = torch.abs(predicted_rewards - target_rewards)
        confidence_targets = torch.exp(-reward_errors)  # High confidence for low error
        confidence_loss = F.binary_cross_entropy(predicted_confidences, confidence_targets)
        
        # Combined loss
        total_loss = reward_loss + 0.1 * confidence_loss
        
        return total_loss
    
    def validate(self):
        """Validate PRM model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_metrics = {
            'reward_loss': [],
            'confidence_loss': [],
            'reward_mae': [],
            'confidence_accuracy': []
        }
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                
                # Compute metrics
                loss = self.compute_loss(batch)
                val_metrics['reward_loss'].append(loss.item())
                
                # Additional metrics
                input_ids = batch['input_ids']
                step_positions = batch['step_position']
                target_rewards = batch['step_reward']
                
                outputs = self.prm_model.forward(input_ids)
                
                batch_size = input_ids.size(0)
                predicted_rewards = []
                predicted_confidences = []
                
                for i in range(batch_size):
                    pos = step_positions[i].item()
                    if pos < outputs['step_rewards'].size(1):
                        predicted_rewards.append(outputs['step_rewards'][i, pos])
                        predicted_confidences.append(outputs['step_confidences'][i, pos])
                    else:
                        predicted_rewards.append(outputs['step_rewards'][i, -1])
                        predicted_confidences.append(outputs['step_confidences'][i, -1])
                
                predicted_rewards = torch.stack(predicted_rewards)
                predicted_confidences = torch.stack(predicted_confidences)
                
                # MAE for rewards
                mae = torch.mean(torch.abs(predicted_rewards - target_rewards))
                val_metrics['reward_mae'].append(mae.item())
                
                # Confidence accuracy (high confidence should correlate with low error)
                reward_errors = torch.abs(predicted_rewards - target_rewards)
                confidence_accuracy = torch.mean((predicted_confidences > 0.5) == (reward_errors < 0.5))
                val_metrics['confidence_accuracy'].append(confidence_accuracy.item())
        
        # Average metrics
        return {key: np.mean(values) for key, values in val_metrics.items()}

class ORMTrainer(Trainer):
    """Trainer for Outcome Reward Models"""
    
    def __init__(self, orm_model: OutcomeRewardModel, config: TrainingConfig,
                 train_dataloader, val_dataloader=None, **kwargs):
        super().__init__(orm_model, config, train_dataloader, val_dataloader, **kwargs)
        self.orm_model = orm_model
    
    def compute_loss(self, batch):
        """Compute ORM training loss"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_correctness = batch['correctness']
        target_helpfulness = batch['helpfulness']
        target_overall = batch['overall_quality']
        
        # Get ORM outputs
        outputs = self.orm_model.forward(input_ids, attention_mask)
        
        # Individual losses
        correctness_loss = F.binary_cross_entropy(
            outputs['correctness'].squeeze(-1), 
            target_correctness
        )
        
        helpfulness_loss = F.mse_loss(
            outputs['helpfulness'].squeeze(-1), 
            target_helpfulness
        )
        
        overall_loss = F.mse_loss(
            outputs['overall_reward'].squeeze(-1), 
            target_overall
        )
        
        # Combined loss
        total_loss = overall_loss + 0.3 * correctness_loss + 0.2 * helpfulness_loss
        
        return total_loss
    
    def validate(self):
        """Validate ORM model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_metrics = {
            'overall_loss': [],
            'correctness_loss': [],
            'helpfulness_loss': [],
            'correctness_accuracy': [],
            'overall_correlation': []
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                
                # Compute loss
                loss = self.compute_loss(batch)
                val_metrics['overall_loss'].append(loss.item())
                
                # Get predictions
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = self.orm_model.forward(input_ids, attention_mask)
                
                # Correctness accuracy
                correctness_pred = (outputs['correctness'] > 0.5).float()
                correctness_target = batch['correctness']
                correctness_acc = torch.mean((correctness_pred.squeeze(-1) == correctness_target).float())
                val_metrics['correctness_accuracy'].append(correctness_acc.item())
                
                # Store for correlation
                all_predictions.extend(outputs['overall_reward'].squeeze(-1).cpu().tolist())
                all_targets.extend(batch['overall_quality'].cpu().tolist())
        
        # Calculate correlation
        if all_predictions and all_targets:
            correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
            val_metrics['overall_correlation'] = [correlation]
        
        return {key: np.mean(values) for key, values in val_metrics.items()}

# Training pipeline functions
def train_process_reward_model(
    base_model,
    training_data: List[Dict],
    validation_data: List[Dict] = None,
    tokenizer=None,
    config: TrainingConfig = None,
    num_epochs: int = 5
) -> ProcessRewardModel:
    """
    Train a Process Reward Model
    
    Args:
        base_model: Pre-trained base model
        training_data: Step-by-step reasoning data
        validation_data: Validation data
        tokenizer: Tokenizer
        config: Training configuration
        num_epochs: Number of training epochs
    
    Returns:
        Trained PRM model
    """
    
    # Create PRM model
    prm_model = ProcessRewardModel(base_model)
    
    # Create datasets
    train_dataset = ProcessRewardDataset(training_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_loader = None
    if validation_data:
        val_dataset = ProcessRewardDataset(validation_data, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Default config
    if config is None:
        config = TrainingConfig(
            epochs=num_epochs,
            batch_size=8,
            optimizer=TrainingConfig.OptimizerConfig(lr=1e-5),
            logging=TrainingConfig.LoggingConfig(log_every=50),
            checkpoint=TrainingConfig.CheckpointConfig(save_every=1000)
        )
    
    # Create trainer
    trainer = PRMTrainer(
        prm_model=prm_model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    # Train
    trainer.fit()
    
    return prm_model

def train_outcome_reward_model(
    base_model,
    training_data: List[Dict],
    validation_data: List[Dict] = None,
    tokenizer=None,
    config: TrainingConfig = None,
    num_epochs: int = 3
) -> OutcomeRewardModel:
    """
    Train an Outcome Reward Model
    
    Args:
        base_model: Pre-trained base model
        training_data: Complete reasoning examples with quality scores
        validation_data: Validation data
        tokenizer: Tokenizer
        config: Training configuration
        num_epochs: Number of training epochs
    
    Returns:
        Trained ORM model
    """
    
    # Create ORM model
    from ..rlhf.reward_model import RewardModelConfig
    reward_config = RewardModelConfig(hidden_size=base_model.config.hidden_size)
    orm_model = OutcomeRewardModel(base_model, reward_config)
    
    # Create datasets
    train_dataset = OutcomeRewardDataset(training_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_loader = None
    if validation_data:
        val_dataset = OutcomeRewardDataset(validation_data, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Default config
    if config is None:
        config = TrainingConfig(
            epochs=num_epochs,
            batch_size=8,
            optimizer=TrainingConfig.OptimizerConfig(lr=1e-5),
            logging=TrainingConfig.LoggingConfig(log_every=50)
        )
    
    # Create trainer
    trainer = ORMTrainer(
        orm_model=orm_model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    # Train
    trainer.fit()
    
    return orm_model

# Data creation utilities
def create_step_reward_data_from_math_problems(problems: List[Dict]) -> List[Dict]:
    """
    Create PRM training data from math problems with step-by-step solutions
    
    Args:
        problems: List of math problems with solutions
            Format: {
                'problem': str,
                'solution_steps': [str, ...],
                'final_answer': str,
                'is_correct': bool
            }
    
    Returns:
        Formatted data for PRM training
    """
    
    prm_data = []
    
    for problem in problems:
        prompt = f"Problem: {problem['problem']}\nSolution:"
        
        steps = []
        for i, step_text in enumerate(problem['solution_steps']):
            # Assign rewards based on correctness and step quality
            if problem['is_correct']:
                # Correct solution: higher rewards for later steps
                step_reward = 0.5 + 0.5 * (i / len(problem['solution_steps']))
            else:
                # Incorrect solution: decreasing rewards
                step_reward = 0.5 - 0.5 * (i / len(problem['solution_steps']))
            
            steps.append({
                'step_text': step_text,
                'reward': step_reward,
                'is_correct': problem['is_correct']
            })
        
        prm_data.append({
            'prompt': prompt,
            'steps': steps,
            'final_answer': problem['final_answer']
        })
    
    return prm_data

def create_outcome_reward_data_from_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Create ORM training data from Q&A pairs with quality annotations
    
    Args:
        qa_pairs: List of Q&A pairs with quality scores
            Format: {
                'question': str,
                'answer': str,
                'reasoning': str,
                'factual_correctness': float,  # 0-1
                'helpfulness_score': float,    # 0-1
                'clarity_score': float         # 0-1
            }
    
    Returns:
        Formatted data for ORM training
    """
    
    orm_data = []
    
    for qa in qa_pairs:
        # Calculate overall quality score
        overall_quality = (
            0.5 * qa['factual_correctness'] +
            0.3 * qa['helpfulness_score'] +
            0.2 * qa['clarity_score']
        )
        
        # Convert helpfulness to -1 to 1 scale
        helpfulness = 2 * qa['helpfulness_score'] - 1
        
        orm_data.append({
            'prompt': f"Question: {qa['question']}\nAnswer:",
            'reasoning': qa['reasoning'],
            'final_answer': qa['answer'],
            'correctness': qa['factual_correctness'],
            'helpfulness': helpfulness,
            'overall_quality': overall_quality
        })
    
    return orm_data

def evaluate_reward_models(
    prm_model: ProcessRewardModel,
    orm_model: OutcomeRewardModel,
    test_data: List[Dict],
    tokenizer
) -> Dict[str, Any]:
    """
    Evaluate trained reward models on test data
    
    Args:
        prm_model: Trained Process Reward Model
        orm_model: Trained Outcome Reward Model  
        test_data: Test examples
        tokenizer: Tokenizer
    
    Returns:
        Evaluation metrics
    """
    
    prm_model.eval()
    orm_model.eval()
    
    prm_predictions = []
    orm_predictions = []
    
    with torch.no_grad():
        for example in test_data:
            # Create full text
            full_text = f"{example['prompt']} {example['reasoning']} {example['final_answer']}"
            
            # Tokenize
            tokens = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
            
            # PRM evaluation (evaluate each step)
            prm_outputs = prm_model.forward(tokens['input_ids'])
            step_rewards = prm_outputs['step_rewards'][0].cpu().tolist()
            step_confidences = prm_outputs['step_confidences'][0].cpu().tolist()
            
            # ORM evaluation (evaluate final outcome)
            orm_outputs = orm_model.forward(tokens['input_ids'], tokens['attention_mask'])
            final_score = orm_outputs['final_score'][0].item()
            correctness = orm_outputs['correctness'][0].item()
            helpfulness = orm_outputs['helpfulness'][0].item()
            
            prm_predictions.append({
                'step_rewards': step_rewards,
                'avg_step_reward': np.mean(step_rewards) if step_rewards else 0,
                'step_confidences': step_confidences
            })
            
            orm_predictions.append({
                'final_score': final_score,
                'correctness': correctness,
                'helpfulness': helpfulness
            })
    
    # Calculate metrics
    prm_avg_rewards = [pred['avg_step_reward'] for pred in prm_predictions]
    orm_final_scores = [pred['final_score'] for pred in orm_predictions]
    
    evaluation_results = {
        'prm_metrics': {
            'avg_step_reward_mean': np.mean(prm_avg_rewards),
            'avg_step_reward_std': np.std(prm_avg_rewards),
            'step_reward_distribution': {
                'min': np.min(prm_avg_rewards),
                'max': np.max(prm_avg_rewards),
                'median': np.median(prm_avg_rewards)
            }
        },
        'orm_metrics': {
            'final_score_mean': np.mean(orm_final_scores),
            'final_score_std': np.std(orm_final_scores),
            'correctness_mean': np.mean([pred['correctness'] for pred in orm_predictions]),
            'helpfulness_mean': np.mean([pred['helpfulness'] for pred in orm_predictions])
        },
        'correlation': {
            'prm_orm_correlation': np.corrcoef(prm_avg_rewards, orm_final_scores)[0, 1] if len(prm_avg_rewards) > 1 else 0.0
        }
    }
    
    return evaluation_results

# Export training components
__all__ = [
    'ProcessRewardDataset',
    'OutcomeRewardDataset', 
    'PRMTrainer',
    'ORMTrainer',
    'train_process_reward_model',
    'train_outcome_reward_model',
    'create_step_reward_data_from_math_problems',
    'create_outcome_reward_data_from_qa_pairs',
    'evaluate_reward_models'
]