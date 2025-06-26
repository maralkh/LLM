"""
Inference engine with advanced sampling strategies and reward-guided generation.
"""
from typing import List, Dict, Any, Optional

from .engine import (
    InferenceEngine,
    BatchInferenceEngine,
    StreamingInferenceEngine
)

from .sampling import (
    BaseSampler,
    GreedySampler,
    TemperatureSampler,
    TopKSampler,
    TopPSampler,
    MinPSampler,
    TypicalSampler,
    MirostatSampler,
    DRYSampler,
    SpeculativeDecoder,
    ContrastiveSearchSampler,
    AdaptiveSampler,
    create_sampler,
    SamplingConfig
)

from .reward_guided import (
    RewardGuidedConfig,
    ProcessRewardModel,
    OutcomeRewardModel, 
    RewardGuidedInferenceEngine,
    create_process_reward_model,
    create_outcome_reward_model,
    create_reward_guided_engine
)

__all__ = [
    # Engines
    "InferenceEngine",
    "BatchInferenceEngine", 
    "StreamingInferenceEngine",
    
    # Sampling
    "BaseSampler",
    "GreedySampler",
    "TemperatureSampler",
    "TopKSampler", 
    "TopPSampler",
    "MinPSampler",
    "TypicalSampler",
    "MirostatSampler",
    #"MirostatV2Sampler",
    "DRYSampler",
    "SpeculativeDecoder",
    "ContrastiveSearchSampler",
    "AdaptiveSampler",
    "create_sampler",
    "SamplingConfig",
    
    # Reward-guided inference
    "RewardGuidedEngine",
    "RewardGuidedConfig",
    "BeamSearchStrategy",
    "MCTSStrategy", 
    "BestOfNStrategy",
    "GuidedSamplingStrategy",
    "create_reward_guided_engine",
]

# Sampling strategy registry
SAMPLING_STRATEGIES = {
    "greedy": GreedySampler,
    "temperature": TemperatureSampler,
    "top_k": TopKSampler,
    "top_p": TopPSampler,
    "nucleus": TopPSampler,  # Alias
    "min_p": MinPSampler,
    "typical": TypicalSampler,
    "mirostat": MirostatSampler,
    #"mirostat_v2": MirostatV2Sampler,
    "dry": DRYSampler,
    "speculative": SpeculativeDecoder,
    "contrastive": ContrastiveSearchSampler,
    "adaptive": AdaptiveSampler,
}

def get_sampling_strategy(strategy_name: str, **kwargs):
    """Get a sampling strategy by name."""
    if strategy_name.lower() not in SAMPLING_STRATEGIES:
        available = ", ".join(SAMPLING_STRATEGIES.keys())
        raise ValueError(f"Unknown sampling strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = SAMPLING_STRATEGIES[strategy_name.lower()]
    return strategy_class(**kwargs)