"""
Reinforcement Learning from Human Feedback implementations.
"""
from typing import List, Dict, Any, Optional

from .reward_model import (
    RewardModel,
    #ProcessRewardModel,
    #OutcomeRewardModel,
    train_reward_model,
    evaluate_reward_model_correlation,
    analyze_reward_distribution

)

from .ppo import (
    PPOTrainer,
    PPOConfig,
    PPOBuffer,
    PPOActor,
    PPOCritic,
    create_ppo_trainer,
    load_prompts_dataset,
    train_ppo_pipeline,
    evaluate_ppo_model,
    PPOWithClipping,
    PPOWithRewardShaping,
    length_penalty,
    repetition_penalty,
    coherence_reward,

)

from .dpo import (
    DPOTrainer,
    DPOConfig,
    DPODataset,
    ReferenceFreeTrainer,
    ReferenceFreeConfig,
    IPOTrainer,
    CPOTrainer,
    IterativeDPOTrainer,
    create_dpo_trainer,
    train_dpo_pipeline,
    evaluate_dpo_model,
    compare_dpo_variants,
    visualize_dpo_training,
    create_preference_data_from_completions,
    save_dpo_results,
    load_dpo_model,
    create_sample_preference_data,
    demonstrate_dpo_pipeline,
    benchmark_dpo_hyperparameters,
)

from .grpo import (
    GRPOConfig, 
    GRPOTrainer,
    create_grpo_trainer, 
    train_grpo_pipeline, 
    evaluate_grpo_model
)

from .prm_orm_training import (
    train_process_reward_model,
    train_outcome_reward_model,
    create_process_reward_dataset,
    create_outcome_reward_dataset,
    ProcessRewardModel,
    OutcomeRewardModel,
    RewardModelConfig,
    evaluate_reward_model
)

__all__ = [
    # Reward models
    "RewardModel",
    "ProcessRewardModel",
    "OutcomeRewardModel", 
    "train_reward_model",
    "evaluate_reward_models",
    
    # PPO
    "PPOTrainer",
    "PPOConfig",
    "ActorCriticModel",
    "train_ppo",
    
    # DPO
    "DPOTrainer", 
    "DPOConfig",
    "ReferenceFreeConfig",
    "train_dpo",
    
    # GRPO
    "GRPOTrainer",
    "GRPOConfig", 
    "train_grpo",
    
    # PRM/ORM training
    # "train_process_reward_model",
    # "train_outcome_reward_model",
    # "create_process_reward_dataset",
    # "create_outcome_reward_dataset",
]

def train_full_rlhf_pipeline(
    base_model,
    tokenizer,
    sft_data,
    preference_data,
    prompts,
    method="dpo",
    **kwargs
):
    """
    Train a complete RLHF pipeline.
    
    Args:
        base_model: Base language model
        tokenizer: Tokenizer
        sft_data: Supervised fine-tuning data
        preference_data: Human preference data
        prompts: Training prompts
        method: RLHF method ("ppo", "dpo", "grpo")
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (pipeline, trainer, results)
    """
    if method.lower() == "ppo":
        trainer = PPOTrainer(base_model, tokenizer, **kwargs)
    elif method.lower() == "dpo":
        trainer = DPOTrainer(base_model, tokenizer, **kwargs)
    elif method.lower() == "grpo":
        trainer = GRPOTrainer(base_model, tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown RLHF method: {method}")
    
    # Training pipeline implementation would go here
    results = trainer.train(sft_data, preference_data, prompts)
    
    return trainer, trainer, results

# RLHF method registry
RLHF_METHODS = {
    "ppo": PPOTrainer,
    "dpo": DPOTrainer,
    "grpo": GRPOTrainer,
}

def get_rlhf_trainer(method: str, **kwargs):
    """Get an RLHF trainer by method name."""
    if method.lower() not in RLHF_METHODS:
        available = ", ".join(RLHF_METHODS.keys())
        raise ValueError(f"Unknown RLHF method '{method}'. Available: {available}")
    
    trainer_class = RLHF_METHODS[method.lower()]
    return trainer_class(**kwargs)