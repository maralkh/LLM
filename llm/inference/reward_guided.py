# training_infra/inference/reward_guided.py
"""
Reward-guided inference with Process Reward Models (PRM) and Outcome Reward Models (ORM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
import copy
from abc import ABC, abstractmethod

from .engine import InferenceEngine, GenerationConfig
from .sampling import SamplingConfig, create_sampler
from ..rlhf.reward_model import RewardModel

@dataclass
class RewardGuidedConfig:
    """Configuration for reward-guided inference"""
    # Reward guidance
    use_prm: bool = True  # Use Process Reward Model
    use_orm: bool = True  # Use Outcome Reward Model
    prm_weight: float = 0.3  # Weight for PRM guidance
    orm_weight: float = 0.7  # Weight for ORM guidance
    
    # Search strategies
    search_strategy: str = "beam_search"  # "beam_search", "mcts", "best_of_n", "guided_sampling"
    num_beams: int = 4
    num_candidates: int = 8
    max_search_depth: int = 5
    
    # Reward thresholds
    min_step_reward: float = -1.0  # Minimum acceptable step reward
    min_final_reward: float = 0.0  # Minimum acceptable final reward
    reward_alpha: float = 0.1  # Reward guidance strength
    
    # Tree search parameters (for MCTS)
    mcts_simulations: int = 100
    mcts_exploration: float = 1.4  # UCB exploration parameter
    mcts_temperature: float = 0.7
    
    # Step-by-step guidance
    step_penalty: float = 0.1  # Penalty for each step (encourage efficiency)
    length_normalization: bool = True
    early_stopping: bool = True
    
    # Verification
    verify_steps: bool = True  # Verify each reasoning step
    verification_threshold: float = 0.5
    max_retries: int = 3

class ProcessRewardModel(nn.Module):
    """Process Reward Model - evaluates reasoning steps"""
    
    def __init__(self, base_model, hidden_size: int = 4096):
        super().__init__()
        self.base_model = base_model
        
        # Step reward head
        self.step_reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Reward between -1 and 1
        )
        
        # Step confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Confidence between 0 and 1
        )
    
    def forward(self, input_ids: torch.Tensor, step_positions: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate reasoning steps
        
        Args:
            input_ids: [batch_size, seq_len]
            step_positions: Positions where reasoning steps occur
            
        Returns:
            Dictionary with step rewards and confidences
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        
        if step_positions is None:
            # Use all positions
            step_positions = list(range(hidden_states.size(1)))
        
        # Extract step representations
        step_rewards = []
        step_confidences = []
        
        for pos in step_positions:
            if pos < hidden_states.size(1):
                step_hidden = hidden_states[:, pos, :]
                
                # Compute step reward
                reward = self.step_reward_head(step_hidden)
                confidence = self.confidence_head(step_hidden)
                
                step_rewards.append(reward)
                step_confidences.append(confidence)
        
        return {
            'step_rewards': torch.stack(step_rewards, dim=1) if step_rewards else torch.empty(0),
            'step_confidences': torch.stack(step_confidences, dim=1) if step_confidences else torch.empty(0),
            'hidden_states': hidden_states
        }
    
    def get_step_reward(self, input_ids: torch.Tensor, step_position: int) -> Tuple[float, float]:
        """Get reward and confidence for a specific step"""
        with torch.no_grad():
            outputs = self.forward(input_ids, [step_position])
            
            if len(outputs['step_rewards']) > 0:
                reward = outputs['step_rewards'][0, 0].item()
                confidence = outputs['step_confidences'][0, 0].item()
                return reward, confidence
            
            return 0.0, 0.0

class OutcomeRewardModel(RewardModel):
    """Outcome Reward Model - evaluates final results"""
    
    def __init__(self, base_model, config):
        super().__init__(base_model, config)
        
        # Add outcome-specific heads
        self.correctness_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()  # Correctness probability
        )
        
        self.helpfulness_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Tanh()  # Helpfulness score
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate final outcome
        
        Returns:
            Dictionary with overall reward, correctness, and helpfulness
        """
        # Get base reward
        overall_reward = super().forward(input_ids, attention_mask)
        
        # Get additional outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling
        if attention_mask is not None:
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Compute specific metrics
        correctness = self.correctness_head(pooled)
        helpfulness = self.helpfulness_head(pooled)
        
        return {
            'overall_reward': overall_reward,
            'correctness': correctness,
            'helpfulness': helpfulness,
            'final_score': overall_reward + 0.5 * correctness.squeeze(-1) + 0.3 * helpfulness.squeeze(-1)
        }

class RewardGuidedInferenceEngine(InferenceEngine):
    """Inference engine with reward guidance"""
    
    def __init__(self, 
                 model: nn.Module,
                 prm: Optional[ProcessRewardModel] = None,
                 orm: Optional[OutcomeRewardModel] = None,
                 tokenizer=None,
                 device: Optional[torch.device] = None):
        
        super().__init__(model, tokenizer, device)
        self.prm = prm
        self.orm = orm
        
        # Move reward models to device
        if self.prm:
            self.prm.to(self.device)
            self.prm.eval()
        
        if self.orm:
            self.orm.to(self.device)
            self.orm.eval()
    
    def generate_with_reward_guidance(
        self,
        input_ids: torch.Tensor,
        config: RewardGuidedConfig,
        generation_config: GenerationConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with reward guidance
        
        Args:
            input_ids: Input token IDs
            config: Reward guidance configuration
            generation_config: Standard generation configuration
            
        Returns:
            Dictionary with generated sequences and reward information
        """
        
        if config.search_strategy == "beam_search":
            return self._reward_guided_beam_search(input_ids, config, generation_config)
        elif config.search_strategy == "mcts":
            return self._reward_guided_mcts(input_ids, config, generation_config)
        elif config.search_strategy == "best_of_n":
            return self._best_of_n_with_rewards(input_ids, config, generation_config)
        elif config.search_strategy == "guided_sampling":
            return self._guided_sampling(input_ids, config, generation_config)
        else:
            raise ValueError(f"Unknown search strategy: {config.search_strategy}")
    
    def _reward_guided_beam_search(
        self,
        input_ids: torch.Tensor,
        config: RewardGuidedConfig,
        generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """Beam search with reward guidance"""
        
        batch_size = input_ids.size(0)
        beam_size = config.num_beams
        max_length = generation_config.max_length
        
        # Initialize beams: [batch_size * beam_size, seq_len]
        beams = input_ids.repeat_interleave(beam_size, dim=0)
        beam_scores = torch.zeros(batch_size * beam_size, device=self.device)
        beam_step_rewards = [[] for _ in range(batch_size * beam_size)]
        
        finished_beams = []
        
        for step in range(max_length - input_ids.size(1)):
            # Get next token logits
            with torch.no_grad():
                outputs = self.model(beams)
                logits = outputs.logits[:, -1, :]  # [batch_size * beam_size, vocab_size]
            
            # Get top-k candidates for each beam
            top_k_logits, top_k_tokens = torch.topk(logits, k=beam_size, dim=-1)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Expand beams with candidates
            new_beams = []
            new_scores = []
            new_step_rewards = []
            
            for beam_idx in range(batch_size * beam_size):
                current_beam = beams[beam_idx]
                current_score = beam_scores[beam_idx]
                
                for k in range(beam_size):
                    # Create new candidate
                    new_token = top_k_tokens[beam_idx, k]
                    new_beam = torch.cat([current_beam, new_token.unsqueeze(0)])
                    
                    # Calculate token probability score
                    token_score = torch.log(top_k_probs[beam_idx, k])
                    
                    # Calculate reward guidance
                    reward_bonus = 0.0
                    
                    # Process reward (if PRM available)
                    if config.use_prm and self.prm:
                        step_reward, confidence = self.prm.get_step_reward(
                            new_beam.unsqueeze(0), 
                            new_beam.size(0) - 1
                        )
                        
                        if step_reward >= config.min_step_reward:
                            reward_bonus += config.prm_weight * step_reward * confidence
                        else:
                            # Penalize bad steps
                            reward_bonus -= config.prm_weight * abs(step_reward)
                    
                    # Combine scores
                    total_score = current_score + token_score + config.reward_alpha * reward_bonus
                    
                    new_beams.append(new_beam)
                    new_scores.append(total_score)
                    new_step_rewards.append(beam_step_rewards[beam_idx] + [reward_bonus])
            
            # Select top beams
            new_scores = torch.stack(new_scores)
            top_beam_indices = torch.topk(new_scores, k=beam_size, dim=0)[1]
            
            # Update beams
            beams = torch.stack([new_beams[idx] for idx in top_beam_indices])
            beam_scores = new_scores[top_beam_indices]
            beam_step_rewards = [new_step_rewards[idx] for idx in top_beam_indices]
            
            # Check for finished sequences
            if generation_config.eos_token_id is not None:
                for beam_idx, beam in enumerate(beams):
                    if beam[-1].item() == generation_config.eos_token_id:
                        finished_beams.append({
                            'sequence': beam,
                            'score': beam_scores[beam_idx].item(),
                            'step_rewards': beam_step_rewards[beam_idx]
                        })
        
        # Evaluate final sequences with ORM
        final_results = []
        
        sequences_to_evaluate = finished_beams if finished_beams else [
            {
                'sequence': beam,
                'score': score.item(),
                'step_rewards': step_rewards
            }
            for beam, score, step_rewards in zip(beams, beam_scores, beam_step_rewards)
        ]
        
        for result in sequences_to_evaluate:
            sequence = result['sequence']
            
            # Outcome reward (if ORM available)
            final_reward = 0.0
            if config.use_orm and self.orm:
                with torch.no_grad():
                    orm_outputs = self.orm.forward(sequence.unsqueeze(0))
                    final_reward = orm_outputs['final_score'].item()
            
            # Combine all scores
            total_score = (result['score'] + 
                          config.orm_weight * final_reward - 
                          config.step_penalty * len(result['step_rewards']))
            
            final_results.append({
                'sequence': sequence,
                'total_score': total_score,
                'beam_score': result['score'],
                'final_reward': final_reward,
                'step_rewards': result['step_rewards'],
                'avg_step_reward': np.mean(result['step_rewards']) if result['step_rewards'] else 0.0
            })
        
        # Sort by total score
        final_results.sort(key=lambda x: x['total_score'], reverse=True)
        
        return {
            'best_sequence': final_results[0]['sequence'] if final_results else beams[0],
            'all_results': final_results,
            'num_candidates': len(final_results),
            'best_score': final_results[0]['total_score'] if final_results else 0.0
        }
    
    def _reward_guided_mcts(
        self,
        input_ids: torch.Tensor,
        config: RewardGuidedConfig,
        generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """Monte Carlo Tree Search with reward guidance"""
        
        class MCTSNode:
            def __init__(self, sequence: torch.Tensor, parent=None, action=None):
                self.sequence = sequence
                self.parent = parent
                self.action = action  # Token that led to this node
                self.children = {}
                self.visits = 0
                self.total_reward = 0.0
                self.step_reward = 0.0
                self.is_terminal = False
            
            def is_fully_expanded(self, vocab_size: int, max_children: int = 10):
                return len(self.children) >= min(max_children, vocab_size)
            
            def best_child(self, exploration_weight: float = 1.4):
                choices_weights = [
                    (child.total_reward / child.visits) + 
                    exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
                    for child in self.children.values()
                ]
                return list(self.children.values())[np.argmax(choices_weights)]
            
            def add_child(self, action: int, sequence: torch.Tensor):
                child = MCTSNode(sequence, parent=self, action=action)
                self.children[action] = child
                return child
        
        # Initialize root
        root = MCTSNode(input_ids[0])
        
        # MCTS simulation loop
        for simulation in range(config.mcts_simulations):
            # Selection
            node = root
            path = [node]
            
            while not node.is_terminal and node.children:
                if not node.is_fully_expanded(self.model.config.vocab_size):
                    break
                node = node.best_child(config.mcts_exploration)
                path.append(node)
            
            # Expansion
            if not node.is_terminal and node.sequence.size(0) < generation_config.max_length:
                # Get possible actions
                with torch.no_grad():
                    outputs = self.model(node.sequence.unsqueeze(0))
                    logits = outputs.logits[0, -1, :]
                    
                # Sample top-k actions
                top_k_logits, top_k_tokens = torch.topk(logits, k=10)
                probs = F.softmax(top_k_logits / config.mcts_temperature, dim=0)
                
                # Add children
                for i, token in enumerate(top_k_tokens):
                    if token.item() not in node.children:
                        new_sequence = torch.cat([node.sequence, token.unsqueeze(0)])
                        child = node.add_child(token.item(), new_sequence)
                        
                        # Evaluate step with PRM
                        if config.use_prm and self.prm:
                            step_reward, _ = self.prm.get_step_reward(
                                new_sequence.unsqueeze(0),
                                new_sequence.size(0) - 1
                            )
                            child.step_reward = step_reward
                        
                        path.append(child)
                        node = child
                        break
            
            # Simulation (rollout)
            final_reward = 0.0
            if config.use_orm and self.orm and node.sequence.size(0) >= input_ids.size(1) + 3:
                with torch.no_grad():
                    orm_outputs = self.orm.forward(node.sequence.unsqueeze(0))
                    final_reward = orm_outputs['final_score'].item()
            
            # Backpropagation
            for node in reversed(path):
                node.visits += 1
                node.total_reward += final_reward + node.step_reward
        
        # Select best path
        best_child = root.best_child(exploration_weight=0)  # Pure exploitation
        best_sequence = best_child.sequence
        
        return {
            'best_sequence': best_sequence,
            'mcts_stats': {
                'root_visits': root.visits,
                'best_child_visits': best_child.visits,
                'best_child_reward': best_child.total_reward / best_child.visits,
                'num_children': len(root.children)
            }
        }
    
    def _best_of_n_with_rewards(
        self,
        input_ids: torch.Tensor,
        config: RewardGuidedConfig,
        generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """Generate N candidates and select best based on rewards"""
        
        candidates = []
        
        # Generate multiple candidates
        for i in range(config.num_candidates):
            # Adjust sampling parameters for diversity
            temp_config = copy.deepcopy(generation_config)
            temp_config.sampling.temperature = 0.7 + 0.3 * (i / config.num_candidates)
            
            # Generate candidate
            result = super().generate(input_ids, temp_config)
            candidate_sequence = result['sequences'][0]
            
            # Evaluate candidate
            total_reward = 0.0
            step_rewards = []
            
            # Process rewards (PRM)
            if config.use_prm and self.prm:
                step_positions = list(range(input_ids.size(1), candidate_sequence.size(0), 2))
                for pos in step_positions:
                    if pos < candidate_sequence.size(0):
                        step_reward, confidence = self.prm.get_step_reward(
                            candidate_sequence.unsqueeze(0), pos
                        )
                        step_rewards.append(step_reward * confidence)
                
                avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
                total_reward += config.prm_weight * avg_step_reward
            
            # Outcome reward (ORM)
            if config.use_orm and self.orm:
                with torch.no_grad():
                    orm_outputs = self.orm.forward(candidate_sequence.unsqueeze(0))
                    final_reward = orm_outputs['final_score'].item()
                    total_reward += config.orm_weight * final_reward
            
            candidates.append({
                'sequence': candidate_sequence,
                'total_reward': total_reward,
                'step_rewards': step_rewards,
                'generation_result': result
            })
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda x: x['total_reward'])
        
        return {
            'best_sequence': best_candidate['sequence'],
            'best_reward': best_candidate['total_reward'],
            'all_candidates': candidates,
            'reward_distribution': {
                'mean': np.mean([c['total_reward'] for c in candidates]),
                'std': np.std([c['total_reward'] for c in candidates]),
                'min': min(c['total_reward'] for c in candidates),
                'max': max(c['total_reward'] for c in candidates)
            }
        }
    
    def _guided_sampling(
        self,
        input_ids: torch.Tensor,
        config: RewardGuidedConfig,
        generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """Token-by-token generation with reward guidance"""
        
        current_sequence = input_ids[0].clone()
        step_rewards = []
        generation_log = []
        
        for step in range(generation_config.max_new_tokens or 50):
            # Get next token logits
            with torch.no_grad():
                outputs = self.model(current_sequence.unsqueeze(0))
                logits = outputs.logits[0, -1, :]
            
            # Get top candidates
            top_k = min(20, logits.size(0))
            top_logits, top_tokens = torch.topk(logits, k=top_k)
            
            # Evaluate each candidate with PRM
            candidate_scores = []
            
            for i, token in enumerate(top_tokens):
                candidate_sequence = torch.cat([current_sequence, token.unsqueeze(0)])
                
                # Base probability
                base_score = F.softmax(top_logits, dim=0)[i].item()
                
                # Process reward
                step_reward = 0.0
                if config.use_prm and self.prm:
                    reward, confidence = self.prm.get_step_reward(
                        candidate_sequence.unsqueeze(0),
                        candidate_sequence.size(0) - 1
                    )
                    step_reward = reward * confidence
                
                # Combined score
                total_score = base_score + config.reward_alpha * step_reward
                candidate_scores.append(total_score)
            
            # Select token based on guided probabilities
            candidate_scores = torch.tensor(candidate_scores)
            guided_probs = F.softmax(candidate_scores / 0.7, dim=0)  # Temperature 0.7
            
            selected_idx = torch.multinomial(guided_probs, 1).item()
            selected_token = top_tokens[selected_idx]
            selected_reward = candidate_scores[selected_idx].item()
            
            # Update sequence
            current_sequence = torch.cat([current_sequence, selected_token.unsqueeze(0)])
            step_rewards.append(selected_reward)
            
            generation_log.append({
                'step': step,
                'selected_token': selected_token.item(),
                'selected_reward': selected_reward,
                'candidate_scores': candidate_scores.tolist(),
                'guided_probs': guided_probs.tolist()
            })
            
            # Early stopping
            if (config.early_stopping and 
                selected_token.item() == generation_config.eos_token_id):
                break
            
            # Check minimum reward threshold
            if selected_reward < config.min_step_reward:
                break
        
        # Final evaluation with ORM
        final_reward = 0.0
        if config.use_orm and self.orm:
            with torch.no_grad():
                orm_outputs = self.orm.forward(current_sequence.unsqueeze(0))
                final_reward = orm_outputs['final_score'].item()
        
        return {
            'best_sequence': current_sequence,
            'step_rewards': step_rewards,
            'final_reward': final_reward,
            'avg_step_reward': np.mean(step_rewards) if step_rewards else 0.0,
            'generation_log': generation_log
        }

# Utility functions
def create_process_reward_model(base_model, hidden_size: int = 4096) -> ProcessRewardModel:
    """Create a Process Reward Model"""
    return ProcessRewardModel(base_model, hidden_size)

def create_outcome_reward_model(base_model, config) -> OutcomeRewardModel:
    """Create an Outcome Reward Model"""
    return OutcomeRewardModel(base_model, config)

def create_reward_guided_engine(
    model: nn.Module,
    prm: Optional[ProcessRewardModel] = None,
    orm: Optional[OutcomeRewardModel] = None,
    tokenizer=None
) -> RewardGuidedInferenceEngine:
    """Create reward-guided inference engine"""
    return RewardGuidedInferenceEngine(model, prm, orm, tokenizer)

# Export components
__all__ = [
    'RewardGuidedConfig',
    'ProcessRewardModel',
    'OutcomeRewardModel', 
    'RewardGuidedInferenceEngine',
    'create_process_reward_model',
    'create_outcome_reward_model',
    'create_reward_guided_engine'
]