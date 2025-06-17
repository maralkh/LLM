# training_infra/distillation/distillation.py
"""
Knowledge Distillation for creating smaller, efficient models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import copy
from abc import ABC, abstractmethod

from ..trainer import Trainer
from ..config import TrainingConfig
from ..inference.engine import InferenceEngine

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    # Temperature and weights
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for hard target loss
    
    # Distillation types
    use_response_distillation: bool = True
    use_feature_distillation: bool = True
    use_attention_distillation: bool = False
    use_hidden_state_distillation: bool = False
    
    # Feature matching
    feature_layers: List[int] = None  # Teacher layers to match
    student_layers: List[int] = None  # Student layers to use
    feature_loss_weight: float = 0.1
    
    # Progressive distillation
    use_progressive: bool = False
    progressive_epochs: List[int] = None
    progressive_temperatures: List[float] = None
    
    # Self-distillation
    use_self_distillation: bool = False
    ensemble_size: int = 3
    
    # Data augmentation during distillation
    use_augmentation: bool = True
    noise_std: float = 0.01
    
    # Adaptive temperature
    use_adaptive_temperature: bool = False
    temp_schedule: str = "cosine"  # "cosine", "linear", "exponential"
    min_temperature: float = 1.0
    max_temperature: float = 8.0
    
    # Online distillation
    use_online_distillation: bool = False
    teacher_update_frequency: int = 100
    
    # Loss functions
    distillation_loss_fn: str = "kl_div"  # "kl_div", "mse", "cosine"
    feature_loss_fn: str = "mse"  # "mse", "cosine", "attention_transfer"

class DistillationLoss(nn.Module):
    """Different loss functions for knowledge distillation"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor = None, temperature: float = None) -> Dict[str, torch.Tensor]:
        """
        Compute distillation losses
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len] - hard targets (optional)
            temperature: Temperature for distillation (optional)
            
        Returns:
            Dictionary of losses
        """
        if temperature is None:
            temperature = self.config.temperature
        
        losses = {}
        
        # Distillation loss (soft targets)
        if self.config.use_response_distillation:
            if self.config.distillation_loss_fn == "kl_div":
                distill_loss = self._kl_divergence_loss(student_logits, teacher_logits, temperature)
            elif self.config.distillation_loss_fn == "mse":
                distill_loss = self._mse_loss(student_logits, teacher_logits, temperature)
            elif self.config.distillation_loss_fn == "cosine":
                distill_loss = self._cosine_loss(student_logits, teacher_logits, temperature)
            else:
                raise ValueError(f"Unknown distillation loss: {self.config.distillation_loss_fn}")
            
            losses['distillation_loss'] = distill_loss
        
        # Hard target loss
        if targets is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            losses['hard_target_loss'] = hard_loss
        
        # Combined loss
        total_loss = 0
        if 'distillation_loss' in losses:
            total_loss += self.config.alpha * losses['distillation_loss']
        if 'hard_target_loss' in losses:
            total_loss += self.config.beta * losses['hard_target_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _kl_divergence_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """KL divergence loss between student and teacher"""
        
        # Apply temperature
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        
        # Scale by temperature squared (as in original distillation paper)
        return kl_loss * (temperature ** 2)
    
    def _mse_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  temperature: float) -> torch.Tensor:
        """MSE loss between softmax outputs"""
        
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        return F.mse_loss(student_probs, teacher_probs)
    
    def _cosine_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     temperature: float) -> torch.Tensor:
        """Cosine similarity loss"""
        
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(student_probs, teacher_probs, dim=-1)
        
        # Convert to loss (1 - similarity)
        return (1 - cos_sim).mean()

class FeatureDistillationLoss(nn.Module):
    """Loss for feature-level distillation"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        
        # Projection layers for feature matching
        self.projections = nn.ModuleDict()
    
    def add_projection(self, name: str, student_dim: int, teacher_dim: int):
        """Add projection layer for dimension matching"""
        self.projections[name] = nn.Linear(student_dim, teacher_dim)
    
    def forward(self, student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute feature distillation losses
        
        Args:
            student_features: Dictionary of student feature tensors
            teacher_features: Dictionary of teacher feature tensors
            
        Returns:
            Dictionary of feature losses
        """
        
        feature_losses = {}
        
        for feature_name in student_features:
            if feature_name in teacher_features:
                student_feat = student_features[feature_name]
                teacher_feat = teacher_features[feature_name]
                
                # Project student features if needed
                if feature_name in self.projections:
                    student_feat = self.projections[feature_name](student_feat)
                
                # Compute feature loss
                if self.config.feature_loss_fn == "mse":
                    loss = F.mse_loss(student_feat, teacher_feat)
                elif self.config.feature_loss_fn == "cosine":
                    loss = 1 - F.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()
                elif self.config.feature_loss_fn == "attention_transfer":
                    loss = self._attention_transfer_loss(student_feat, teacher_feat)
                else:
                    loss = F.mse_loss(student_feat, teacher_feat)
                
                feature_losses[f"{feature_name}_loss"] = loss
        
        # Total feature loss
        if feature_losses:
            total_feature_loss = sum(feature_losses.values())
            feature_losses['total_feature_loss'] = total_feature_loss
        
        return feature_losses
    
    def _attention_transfer_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Attention transfer loss (Zagoruyko & Komodakis, 2016)"""
        
        # Compute attention maps
        student_attention = self._compute_attention_map(student_feat)
        teacher_attention = self._compute_attention_map(teacher_feat)
        
        return F.mse_loss(student_attention, teacher_attention)
    
    def _compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention map from features"""
        
        # Sum over channel dimension and normalize
        attention = torch.sum(features ** 2, dim=-1)
        attention = F.normalize(attention.view(attention.size(0), -1), p=2, dim=1)
        
        return attention

class DistillationTrainer(Trainer):
    """Trainer for knowledge distillation"""
    
    def __init__(self, 
                 student_model: nn.Module,
                 teacher_model: nn.Module,
                 config: TrainingConfig,
                 distillation_config: DistillationConfig,
                 train_dataloader,
                 val_dataloader=None,
                 **kwargs):
        
        # Initialize base trainer with student model
        super().__init__(student_model, config, train_dataloader, val_dataloader, **kwargs)
        
        self.teacher_model = teacher_model
        self.distillation_config = distillation_config
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Initialize distillation losses
        self.distillation_loss = DistillationLoss(distillation_config)
        self.feature_loss = FeatureDistillationLoss(distillation_config)
        
        # Setup feature matching if needed
        self._setup_feature_matching()
        
        # Temperature scheduling
        self.current_temperature = distillation_config.temperature
        
        # Distillation statistics
        self.distillation_stats = {
            'distillation_loss': [],
            'feature_loss': [],
            'hard_target_loss': [],
            'temperature_schedule': []
        }
    
    def _setup_feature_matching(self):
        """Setup feature matching between teacher and student"""
        
        if not self.distillation_config.use_feature_distillation:
            return
        
        # Get feature dimensions (this would need to be customized for specific models)
        # For now, assume both models have similar architectures
        
        # Example: match hidden states from specific layers
        if hasattr(self.model, 'layers') and hasattr(self.teacher_model, 'layers'):
            student_dim = self.model.config.hidden_size if hasattr(self.model, 'config') else 4096
            teacher_dim = self.teacher_model.config.hidden_size if hasattr(self.teacher_model, 'config') else 4096
            
            if student_dim != teacher_dim:
                self.feature_loss.add_projection("hidden_states", student_dim, teacher_dim)
    
    def compute_loss(self, batch):
        """Compute distillation loss"""
        
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        attention_mask = batch.get('attention_mask', None)
        
        # Student forward pass
        student_outputs = self.model(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=self.distillation_config.use_hidden_state_distillation,
            output_attentions=self.distillation_config.use_attention_distillation
        )
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.distillation_config.use_hidden_state_distillation,
                output_attentions=self.distillation_config.use_attention_distillation
            )
        
        # Update temperature if adaptive
        if self.distillation_config.use_adaptive_temperature:
            self.current_temperature = self._get_adaptive_temperature()
        
        # Response-level distillation
        response_losses = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            labels,
            self.current_temperature
        )
        
        total_loss = response_losses['total_loss']
        
        # Feature-level distillation
        if self.distillation_config.use_feature_distillation:
            student_features = self._extract_features(student_outputs)
            teacher_features = self._extract_features(teacher_outputs)
            
            feature_losses = self.feature_loss(student_features, teacher_features)
            
            if 'total_feature_loss' in feature_losses:
                total_loss += self.distillation_config.feature_loss_weight * feature_losses['total_feature_loss']
                self.distillation_stats['feature_loss'].append(feature_losses['total_feature_loss'].item())
        
        # Data augmentation
        if self.distillation_config.use_augmentation and self.training:
            total_loss += self._augmentation_loss(student_outputs, teacher_outputs, input_ids)
        
        # Update statistics
        if 'distillation_loss' in response_losses:
            self.distillation_stats['distillation_loss'].append(response_losses['distillation_loss'].item())
        if 'hard_target_loss' in response_losses:
            self.distillation_stats['hard_target_loss'].append(response_losses['hard_target_loss'].item())
        
        self.distillation_stats['temperature_schedule'].append(self.current_temperature)
        
        return total_loss
    
    def _extract_features(self, model_outputs) -> Dict[str, torch.Tensor]:
        """Extract features for distillation"""
        
        features = {}
        
        # Hidden states
        if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states:
            # Use last hidden state or specific layers
            if self.distillation_config.feature_layers:
                for i, layer_idx in enumerate(self.distillation_config.feature_layers):
                    if layer_idx < len(model_outputs.hidden_states):
                        features[f"hidden_layer_{layer_idx}"] = model_outputs.hidden_states[layer_idx]
            else:
                features["hidden_states"] = model_outputs.hidden_states[-1]  # Last layer
        
        # Attention weights
        if hasattr(model_outputs, 'attentions') and model_outputs.attentions:
            for i, attention in enumerate(model_outputs.attentions):
                features[f"attention_layer_{i}"] = attention
        
        return features
    
    def _get_adaptive_temperature(self) -> float:
        """Get adaptive temperature based on training progress"""
        
        progress = self.global_step / (len(self.train_dataloader) * self.config.epochs)
        progress = min(progress, 1.0)
        
        if self.distillation_config.temp_schedule == "cosine":
            temp = (self.distillation_config.min_temperature + 
                   (self.distillation_config.max_temperature - self.distillation_config.min_temperature) * 
                   (1 + np.cos(np.pi * progress)) / 2)
        elif self.distillation_config.temp_schedule == "linear":
            temp = (self.distillation_config.max_temperature - 
                   progress * (self.distillation_config.max_temperature - self.distillation_config.min_temperature))
        elif self.distillation_config.temp_schedule == "exponential":
            temp = (self.distillation_config.max_temperature * 
                   (self.distillation_config.min_temperature / self.distillation_config.max_temperature) ** progress)
        else:
            temp = self.distillation_config.temperature
        
        return temp
    
    def _augmentation_loss(self, student_outputs, teacher_outputs, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute augmentation loss with noisy inputs"""
        
        # Add noise to input embeddings
        if hasattr(self.model, 'get_input_embeddings'):
            embedding_layer = self.model.get_input_embeddings()
            clean_embeddings = embedding_layer(input_ids)
            
            # Add Gaussian noise
            noise = torch.randn_like(clean_embeddings) * self.distillation_config.noise_std
            noisy_embeddings = clean_embeddings + noise
            
            # Forward pass with noisy embeddings
            noisy_student_outputs = self.model(inputs_embeds=noisy_embeddings)
            
            # Consistency loss between clean and noisy outputs
            consistency_loss = F.kl_div(
                F.log_softmax(noisy_student_outputs.logits / self.current_temperature, dim=-1),
                F.softmax(student_outputs.logits / self.current_temperature, dim=-1),
                reduction='batchmean'
            )
            
            return 0.1 * consistency_loss
        
        return torch.tensor(0.0, device=input_ids.device)
    
    def validate(self):
        """Validate student model"""
        
        base_metrics = super().validate()
        
        # Add distillation-specific metrics
        if self.distillation_stats['distillation_loss']:
            base_metrics.update({
                'avg_distillation_loss': np.mean(self.distillation_stats['distillation_loss'][-100:]),
                'avg_feature_loss': np.mean(self.distillation_stats['feature_loss'][-100:]) if self.distillation_stats['feature_loss'] else 0,
                'current_temperature': self.current_temperature
            })
        
        return base_metrics

class ProgressiveDistillationTrainer(DistillationTrainer):
    """Progressive distillation with curriculum"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not self.distillation_config.use_progressive:
            return
        
        # Setup progressive schedule
        self.progressive_epochs = self.distillation_config.progressive_epochs or [0, 5, 10]
        self.progressive_temperatures = self.distillation_config.progressive_temperatures or [8.0, 4.0, 2.0]
        
        assert len(self.progressive_epochs) == len(self.progressive_temperatures)
    
    def on_epoch_begin(self, trainer, epoch):
        """Update temperature for progressive distillation"""
        
        if not self.distillation_config.use_progressive:
            return
        
        # Find appropriate temperature for current epoch
        for i, epoch_threshold in enumerate(reversed(self.progressive_epochs)):
            if epoch >= epoch_threshold:
                temp_idx = len(self.progressive_epochs) - 1 - i
                self.current_temperature = self.progressive_temperatures[temp_idx]
                break
        
        print(f"Progressive distillation: Epoch {epoch}, Temperature: {self.current_temperature}")

class SelfDistillationTrainer(DistillationTrainer):
    """Self-distillation using ensemble of student models"""
    
    def __init__(self, student_models: List[nn.Module], *args, **kwargs):
        # Use first model as primary student
        super().__init__(student_models[0], *args, **kwargs)
        
        self.student_ensemble = student_models
        self.ensemble_size = len(student_models)
        
        # Initialize all models
        for model in self.student_ensemble[1:]:
            model.to(self.device)
    
    def compute_loss(self, batch):
        """Compute self-distillation loss using ensemble"""
        
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        
        # Get outputs from all student models
        student_outputs = []
        for model in self.student_ensemble:
            output = model(input_ids)
            student_outputs.append(output.logits)
        
        # Create ensemble teacher (average of all models except primary)
        ensemble_logits = torch.stack(student_outputs[1:]).mean(dim=0)
        
        # Distill primary student from ensemble
        distillation_losses = self.distillation_loss(
            student_outputs[0],
            ensemble_logits,
            labels,
            self.current_temperature
        )
        
        return distillation_losses['total_loss']

class OnlineDistillationTrainer(DistillationTrainer):
    """Online distillation with teacher model updates"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enable teacher model updates
        for param in self.teacher_model.parameters():
            param.requires_grad = True
        
        # Separate optimizer for teacher
        self.teacher_optimizer = torch.optim.AdamW(
            self.teacher_model.parameters(),
            lr=self.config.optimizer.lr * 0.1  # Lower LR for teacher
        )
    
    def train_step(self, batch):
        """Training step with online teacher updates"""
        
        # Regular distillation step
        loss = super().train_step(batch)
        
        # Update teacher periodically
        if self.global_step % self.distillation_config.teacher_update_frequency == 0:
            self._update_teacher(batch)
        
        return loss
    
    def _update_teacher(self, batch):
        """Update teacher model"""
        
        self.teacher_optimizer.zero_grad()
        
        # Teacher forward pass
        teacher_outputs = self.teacher_model(batch['input_ids'], labels=batch.get('labels'))
        teacher_loss = teacher_outputs.loss if hasattr(teacher_outputs, 'loss') else teacher_outputs['loss']
        
        teacher_loss.backward()
        self.teacher_optimizer.step()

# Specialized distillation methods
class AttentionDistillation(nn.Module):
    """Attention-based knowledge distillation"""
    
    def __init__(self, student_layers: int, teacher_layers: int):
        super().__init__()
        self.student_layers = student_layers
        self.teacher_layers = teacher_layers
        
        # Attention transfer matrices
        self.attention_transfers = nn.ModuleList([
            nn.Linear(student_layers, teacher_layers)
            for _ in range(12)  # Assume 12 attention heads
        ])
    
    def forward(self, student_attentions: List[torch.Tensor], 
                teacher_attentions: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention distillation loss"""
        
        total_loss = 0
        
        for s_att, t_att, transfer in zip(student_attentions, teacher_attentions, self.attention_transfers):
            # Transfer student attention to teacher space
            transferred_att = transfer(s_att)
            
            # MSE loss between attention maps
            loss = F.mse_loss(transferred_att, t_att)
            total_loss += loss
        
        return total_loss / len(student_attentions)

class HiddenStateDistillation(nn.Module):
    """Hidden state knowledge distillation"""
    
    def __init__(self, student_dim: int, teacher_dim: int, num_layers: int):
        super().__init__()
        
        # Layer-wise projections
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.ReLU(),
                nn.Linear(teacher_dim, teacher_dim)
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, student_hidden_states: List[torch.Tensor],
                teacher_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Compute hidden state distillation loss"""
        
        total_loss = 0
        
        for s_hidden, t_hidden, projection in zip(
            student_hidden_states, teacher_hidden_states, self.projections
        ):
            # Project student hidden states
            projected_hidden = projection(s_hidden)
            
            # MSE loss
            loss = F.mse_loss(projected_hidden, t_hidden)
            total_loss += loss
        
        return total_loss / len(student_hidden_states)

# Utility functions
def create_distillation_trainer(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_dataloader,
    val_dataloader=None,
    config: TrainingConfig = None,
    distillation_config: DistillationConfig = None
) -> DistillationTrainer:
    """Factory function to create distillation trainer"""
    
    if config is None:
        config = TrainingConfig(
            epochs=10,
            batch_size=16,
            optimizer=TrainingConfig.OptimizerConfig(lr=5e-5)
        )
    
    if distillation_config is None:
        distillation_config = DistillationConfig()
    
    return DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        config=config,
        distillation_config=distillation_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )

def compress_model_with_distillation(
    teacher_model: nn.Module,
    student_architecture: str,
    train_data,
    compression_ratio: float = 0.5
) -> nn.Module:
    """
    Compress a model using knowledge distillation
    
    Args:
        teacher_model: Large teacher model
        student_architecture: Architecture for student model
        train_data: Training data
        compression_ratio: Target compression ratio
        
    Returns:
        Compressed student model
    """
    
    # This would create a smaller student model based on the teacher
    # Implementation depends on specific model architectures
    
    print(f"Compressing model by {compression_ratio}x using knowledge distillation")
    
    # Placeholder implementation
    student_model = copy.deepcopy(teacher_model)
    
    return student_model

def evaluate_distillation_quality(
    teacher_model: nn.Module,
    student_model: nn.Module,
    test_data,
    tokenizer
) -> Dict[str, float]:
    """
    Evaluate the quality of knowledge distillation
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        test_data: Test dataset
        tokenizer: Tokenizer
        
    Returns:
        Evaluation metrics
    """
    
    teacher_engine = InferenceEngine(teacher_model, tokenizer)
    student_engine = InferenceEngine(student_model, tokenizer)
    
    metrics = {
        'teacher_perplexity': 0.0,
        'student_perplexity': 0.0,
        'knowledge_retention': 0.0,
        'compression_ratio': 0.0,
        'speedup': 0.0
    }
    
    # Placeholder implementation
    # In practice, you'd compute actual metrics
    
    return metrics

# Export distillation components
__all__ = [
    'DistillationConfig',
    'DistillationLoss',
    'FeatureDistillationLoss',
    'DistillationTrainer',
    'ProgressiveDistillationTrainer',
    'SelfDistillationTrainer',
    'OnlineDistillationTrainer',
    'AttentionDistillation',
    'HiddenStateDistillation',
    'create_distillation_trainer',
    'compress_model_with_distillation',
    'evaluate_distillation_quality'
]