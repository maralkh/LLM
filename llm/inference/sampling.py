# training_infra/inference/sampling.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
import math
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SamplingConfig:
    """Configuration for sampling methods"""
    # Basic parameters
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    
    # Advanced sampling
    eta_cutoff: Optional[float] = None  # Eta sampling
    epsilon_cutoff: Optional[float] = None  # Epsilon sampling
    
    # Repetition control
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False
    
    # Diverse beam search
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    
    # Contrastive search
    penalty_alpha: Optional[float] = None
    
    # Mirostat sampling
    mirostat_mode: int = 0  # 0: disabled, 1: Mirostat 1.0, 2: Mirostat 2.0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    
    # DRY (Don't Repeat Yourself) sampling
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    
    # Speculative decoding
    use_speculative_decoding: bool = False
    draft_model: Optional[torch.nn.Module] = None
    num_speculative_tokens: int = 4
    
    # Classifier-free guidance
    guidance_scale: Optional[float] = None
    negative_prompt_ids: Optional[torch.Tensor] = None
    
    # Stopping criteria
    max_length: int = 100
    max_new_tokens: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    stop_strings: Optional[List[str]] = None
    
    # Special modes
    do_sample: bool = True
    use_cache: bool = True
    
    def __post_init__(self):
        if self.max_new_tokens is None:
            self.max_new_tokens = self.max_length

class BaseSampler(ABC):
    """Base class for sampling methods"""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
    
    @abstractmethod
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample next token from logits"""
        pass

class GreedySampler(BaseSampler):
    """Greedy sampling (deterministic)"""
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

class TemperatureSampler(BaseSampler):
    """Temperature-based sampling"""
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.config.temperature <= 0:
            return torch.argmax(logits, dim=-1)
        
        logits = logits / self.config.temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

class TopKSampler(BaseSampler):
    """Top-K sampling"""
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.config.top_k is None or self.config.top_k <= 0:
            return TemperatureSampler(self.config).sample(logits, **kwargs)
        
        # Get top-k values and indices
        top_k = min(self.config.top_k, logits.size(-1))
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        # Create mask for non-top-k tokens
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_values)
        
        # Apply temperature and sample
        if self.config.temperature > 0:
            mask = mask / self.config.temperature
        
        probs = F.softmax(mask, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

class TopPSampler(BaseSampler):
    """Top-P (nucleus) sampling"""
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.config.top_p is None or self.config.top_p >= 1.0:
            return TemperatureSampler(self.config).sample(logits, **kwargs)
        
        # Apply temperature first
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff point
        sorted_indices_to_remove = cumulative_probs > self.config.top_p
        
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Set logits to -inf for tokens to remove
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        # Unsort the logits
        logits_to_use = torch.full_like(logits, float('-inf'))
        logits_to_use.scatter_(-1, sorted_indices, sorted_logits)
        
        probs = F.softmax(logits_to_use, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

class MinPSampler(BaseSampler):
    """Min-P sampling"""
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.config.min_p is None or self.config.min_p <= 0:
            return TemperatureSampler(self.config).sample(logits, **kwargs)
        
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        
        probs = F.softmax(logits, dim=-1)
        
        # Get maximum probability
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        
        # Calculate min_p threshold
        threshold = self.config.min_p * max_prob
        
        # Mask tokens below threshold
        mask = probs >= threshold
        filtered_logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

class TypicalSampler(BaseSampler):
    """Typical sampling (locally typical sampling)"""
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.config.typical_p is None or self.config.typical_p >= 1.0:
            return TemperatureSampler(self.config).sample(logits, **kwargs)
        
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        
        probs = F.softmax(logits, dim=-1)
        
        # Calculate entropy and conditional entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        
        # Calculate surprisal (negative log probability)
        surprisal = -log_probs
        
        # Calculate absolute difference from entropy (typicality)
        abs_diff = torch.abs(surprisal - entropy)
        
        # Sort by typicality (smaller difference = more typical)
        sorted_diff, sorted_indices = torch.sort(abs_diff, dim=-1)
        sorted_probs = torch.gather(probs, -1, sorted_indices)
        
        # Find cumulative probability mass
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff point
        cutoff_mask = cumulative_probs < self.config.typical_p
        cutoff_mask = torch.cat([
            torch.ones_like(cutoff_mask[..., :1]),  # Always keep most typical token
            cutoff_mask[..., :-1]
        ], dim=-1)
        
        # Create filtered logits
        filtered_logits = torch.full_like(logits, float('-inf'))
        selected_indices = sorted_indices[cutoff_mask]
        
        # This is a bit complex due to batching, so we'll use a loop
        for batch_idx in range(logits.size(0)):
            batch_mask = cutoff_mask[batch_idx]
            batch_indices = sorted_indices[batch_idx][batch_mask]
            filtered_logits[batch_idx, batch_indices] = logits[batch_idx, batch_indices]
        
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

class MirostatSampler(BaseSampler):
    """Mirostat sampling for controlling perplexity"""
    
    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        self.tau = config.mirostat_tau
        self.eta = config.mirostat_eta
        self.mode = config.mirostat_mode
        self.surprise_value = 0.0  # Initialize surprise value
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.mode == 0:
            return TemperatureSampler(self.config).sample(logits, **kwargs)
        
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        
        probs = F.softmax(logits, dim=-1)
        
        if self.mode == 1:
            # Mirostat 1.0
            return self._mirostat_v1(logits, probs)
        elif self.mode == 2:
            # Mirostat 2.0
            return self._mirostat_v2(logits, probs)
    
    def _mirostat_v1(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Mirostat 1.0 implementation"""
        # Calculate entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Adjust surprise value based on entropy
        surprise_error = entropy - self.tau
        self.surprise_value = max(0, self.surprise_value - self.eta * surprise_error.item())
        
        # Filter tokens based on surprise value
        surprisal = -log_probs
        mask = surprisal <= self.surprise_value
        
        if not mask.any():
            # Fallback to top token if no tokens meet criteria
            return torch.argmax(logits, dim=-1)
        
        filtered_logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        
        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    
    def _mirostat_v2(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Mirostat 2.0 implementation"""
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Calculate derivative approximation
        derivative = sorted_probs[..., :-1] - sorted_probs[..., 1:]
        
        # Find cutoff based on tau and derivative
        k = 1
        for i in range(len(derivative[0])):
            if derivative[0, i] < self.tau:
                k = i + 1
                break
        
        # Adjust k based on surprise value
        k = max(1, int(k + self.surprise_value))
        k = min(k, sorted_probs.size(-1))
        
        # Select top-k tokens
        selected_indices = sorted_indices[..., :k]
        filtered_logits = torch.full_like(logits, float('-inf'))
        
        for batch_idx in range(logits.size(0)):
            filtered_logits[batch_idx, selected_indices[batch_idx]] = logits[batch_idx, selected_indices[batch_idx]]
        
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        sampled_token = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        # Update surprise value
        if sampled_token.numel() == 1:
            token_prob = probs.gather(-1, sampled_token.unsqueeze(-1)).squeeze(-1)
            surprise = -torch.log(token_prob).item()
            self.surprise_value = self.surprise_value - self.eta * (surprise - self.tau)
        
        return sampled_token

class ContrastiveSearchSampler(BaseSampler):
    """Contrastive search sampling"""
    
    def sample(self, logits: torch.Tensor, hidden_states: torch.Tensor = None, 
               past_hidden_states: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if self.config.penalty_alpha is None or hidden_states is None:
            return TopKSampler(self.config).sample(logits, **kwargs)
        
        # Get top-k candidates
        top_k = self.config.top_k or 4
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        top_k_probs = F.softmax(top_k_values, dim=-1)
        
        if past_hidden_states is None:
            # First token, just use probability
            return torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
        
        # Calculate similarity penalty
        # This is a simplified version - in practice you'd compute cosine similarity
        # between current hidden state and past hidden states
        batch_size = hidden_states.size(0)
        
        # Placeholder for similarity calculation
        # In real implementation, you'd compute:
        # similarity = cosine_similarity(hidden_states, past_hidden_states)
        similarity_penalty = torch.zeros(batch_size, top_k, device=logits.device)
        
        # Combine probability and penalty
        contrastive_scores = (1 - self.config.penalty_alpha) * top_k_probs - self.config.penalty_alpha * similarity_penalty
        
        # Sample from contrastive scores
        best_idx = torch.argmax(contrastive_scores, dim=-1)
        return top_k_indices.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

class DRYSampler(BaseSampler):
    """DRY (Don't Repeat Yourself) sampling"""
    
    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        self.sequence_breakers = set()  # Tokens that break sequences
    
    def sample(self, logits: torch.Tensor, input_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if self.config.dry_multiplier <= 0 or input_ids is None:
            return TemperatureSampler(self.config).sample(logits, **kwargs)
        
        # Apply DRY penalty
        penalized_logits = self._apply_dry_penalty(logits, input_ids)
        
        # Use temperature sampling on penalized logits
        if self.config.temperature > 0:
            penalized_logits = penalized_logits / self.config.temperature
        
        probs = F.softmax(penalized_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _apply_dry_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply DRY penalty to logits"""
        batch_size, vocab_size = logits.shape
        penalized_logits = logits.clone()
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()
            
            # Find repeating subsequences
            penalties = self._find_repetition_penalties(sequence)
            
            # Apply penalties
            for token_id, penalty in penalties.items():
                if token_id < vocab_size:
                    penalized_logits[batch_idx, token_id] -= penalty
        
        return penalized_logits
    
    def _find_repetition_penalties(self, sequence: List[int]) -> Dict[int, float]:
        """Find tokens that would create repetitions"""
        penalties = {}
        seq_len = len(sequence)
        
        # Look for potential repetitions
        for i in range(max(0, seq_len - 50), seq_len):  # Only check recent history
            for length in range(self.config.dry_allowed_length, min(20, seq_len - i)):
                if i + length >= seq_len:
                    break
                
                subsequence = sequence[i:i+length]
                
                # Check if this subsequence appears earlier
                for j in range(max(0, i - 100), i):
                    if j + length > i:
                        break
                    
                    if sequence[j:j+length] == subsequence:
                        # Found repetition, penalize the next token
                        if i + length < seq_len:
                            next_token = sequence[i + length]
                            penalty_strength = self.config.dry_multiplier * (self.config.dry_base ** (length - self.config.dry_allowed_length))
                            penalties[next_token] = penalties.get(next_token, 0) + penalty_strength
        
        return penalties

class SpeculativeDecoder:
    """Speculative decoding for faster inference"""
    
    def __init__(self, target_model: torch.nn.Module, draft_model: torch.nn.Module, 
                 num_speculative_tokens: int = 4):
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
    
    def generate_speculative(self, input_ids: torch.Tensor, sampler: BaseSampler, 
                            max_new_tokens: int = 100) -> torch.Tensor:
        """Generate tokens using speculative decoding"""
        generated_tokens = []
        current_ids = input_ids
        
        while len(generated_tokens) < max_new_tokens:
            # Draft phase: generate multiple tokens with draft model
            draft_tokens = []
            draft_input = current_ids
            
            for _ in range(self.num_speculative_tokens):
                with torch.no_grad():
                    draft_logits = self.draft_model(draft_input).logits[:, -1, :]
                    draft_token = sampler.sample(draft_logits)
                    draft_tokens.append(draft_token)
                    draft_input = torch.cat([draft_input, draft_token.unsqueeze(-1)], dim=-1)
            
            # Verification phase: check with target model
            extended_input = torch.cat([current_ids] + [t.unsqueeze(-1) for t in draft_tokens], dim=-1)
            
            with torch.no_grad():
                target_logits = self.target_model(extended_input).logits
            
            # Accept/reject tokens
            accepted_tokens = 0
            for i, draft_token in enumerate(draft_tokens):
                target_logit = target_logits[:, current_ids.size(1) + i - 1, :]
                
                # Calculate acceptance probability
                target_prob = F.softmax(target_logit, dim=-1)
                draft_logit = self.draft_model(extended_input[:, :current_ids.size(1) + i]).logits[:, -1, :]
                draft_prob = F.softmax(draft_logit, dim=-1)
                
                acceptance_prob = torch.min(
                    torch.ones_like(target_prob),
                    target_prob / (draft_prob + 1e-10)
                ).gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                
                if torch.rand(1).item() < acceptance_prob.item():
                    generated_tokens.append(draft_token)
                    accepted_tokens += 1
                else:
                    # Reject and sample from corrected distribution
                    corrected_prob = torch.clamp(target_prob - draft_prob, min=0)
                    corrected_prob = corrected_prob / (corrected_prob.sum(dim=-1, keepdim=True) + 1e-10)
                    corrected_token = torch.multinomial(corrected_prob, num_samples=1).squeeze(-1)
                    generated_tokens.append(corrected_token)
                    break
            
            # Update current sequence
            current_ids = torch.cat([
                current_ids,
                torch.stack(generated_tokens[-accepted_tokens-1:]).unsqueeze(0).transpose(0, 1)
            ], dim=-1)
            
            if len(generated_tokens) >= max_new_tokens:
                break
        
        return torch.stack(generated_tokens[:max_new_tokens])

class CompositeSampler(BaseSampler):
    """Combines multiple sampling methods"""
    
    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        self.samplers = self._create_sampler_pipeline()
    
    def _create_sampler_pipeline(self) -> List[BaseSampler]:
        """Create sampling pipeline based on config"""
        pipeline = []
        
        # Order matters - more restrictive samplers first
        if self.config.top_k is not None and self.config.top_k > 0:
            pipeline.append(TopKSampler(self.config))
        
        if self.config.top_p is not None and self.config.top_p < 1.0:
            pipeline.append(TopPSampler(self.config))
        
        if self.config.min_p is not None and self.config.min_p > 0:
            pipeline.append(MinPSampler(self.config))
        
        if self.config.typical_p is not None and self.config.typical_p < 1.0:
            pipeline.append(TypicalSampler(self.config))
        
        if self.config.mirostat_mode > 0:
            pipeline.append(MirostatSampler(self.config))
        
        # Default to temperature sampling
        if not pipeline or self.config.temperature != 1.0:
            pipeline.append(TemperatureSampler(self.config))
        
        return pipeline
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply sampling pipeline"""
        if not self.config.do_sample:
            return GreedySampler(self.config).sample(logits, **kwargs)
        
        # Apply repetition penalties first
        if hasattr(self, '_apply_repetition_penalties'):
            logits = self._apply_repetition_penalties(logits, **kwargs)
        
        # Apply DRY sampling if enabled
        if self.config.dry_multiplier > 0:
            return DRYSampler(self.config).sample(logits, **kwargs)
        
        # Use the last sampler in pipeline (most general)
        if self.samplers:
            return self.samplers[-1].sample(logits, **kwargs)
        
        return GreedySampler(self.config).sample(logits, **kwargs)

class AdaptiveSampler(BaseSampler):
    """Adaptive sampling that switches methods based on context"""
    
    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        self.confidence_threshold = 0.9
        self.entropy_threshold = 2.0
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Adaptively choose sampling method based on logits"""
        probs = F.softmax(logits, dim=-1)
        
        # Calculate confidence (max probability)
        max_prob = torch.max(probs, dim=-1)[0]
        
        # Calculate entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Choose sampling method based on confidence and entropy
        if max_prob > self.confidence_threshold:
            # High confidence - use greedy or low temperature
            return GreedySampler(self.config).sample(logits, **kwargs)
        elif entropy > self.entropy_threshold:
            # High entropy - use nucleus sampling
            return TopPSampler(self.config).sample(logits, **kwargs)
        else:
            # Medium confidence - use temperature sampling
            return TemperatureSampler(self.config).sample(logits, **kwargs)

# Utility functions
def create_sampler(config: SamplingConfig) -> BaseSampler:
    """Factory function to create appropriate sampler"""
    if not config.do_sample:
        return GreedySampler(config)
    
    # Check for special sampling modes
    if config.penalty_alpha is not None:
        return ContrastiveSearchSampler(config)
    elif config.dry_multiplier > 0:
        return DRYSampler(config)
    elif config.mirostat_mode > 0:
        return MirostatSampler(config)
    else:
        return CompositeSampler(config)

def sample_with_logits_processor(
    logits: torch.Tensor,
    sampler: BaseSampler,
    logits_processors: Optional[List[Callable]] = None,
    **kwargs
) -> torch.Tensor:
    """Sample with custom logits processors"""
    
    # Apply logits processors
    if logits_processors:
        for processor in logits_processors:
            logits = processor(logits, **kwargs)
    
    return sampler.sample(logits, **kwargs)

def batch_sample(
    logits: torch.Tensor,
    configs: List[SamplingConfig],
    **kwargs
) -> torch.Tensor:
    """Sample different sequences with different configs"""
    batch_size = logits.size(0)
    assert len(configs) == batch_size, "Number of configs must match batch size"
    
    results = []
    for i, config in enumerate(configs):
        sampler = create_sampler(config)
        token = sampler.sample(logits[i:i+1], **kwargs)
        results.append(token)
    
    return torch.cat(results, dim=0)