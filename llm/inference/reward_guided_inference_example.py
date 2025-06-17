# examples/reward_guided_inference_example.py
"""
Complete example of reward-guided inference with PRM and ORM
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import copy
import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np

# Import our modules
from training_infra.models.llama import create_llama_7b
from training_infra.inference.reward_guided import (
    RewardGuidedConfig, 
    ProcessRewardModel, 
    OutcomeRewardModel,
    RewardGuidedInferenceEngine,
    create_reward_guided_engine
)
from training_infra.inference.engine import GenerationConfig
from training_infra.inference.sampling import SamplingConfig
from training_infra.rlhf.prm_orm_training import (
    train_process_reward_model,
    train_outcome_reward_model,
    create_step_reward_data_from_math_problems,
    create_outcome_reward_data_from_qa_pairs,
    evaluate_reward_models
)

class DummyTokenizer:
    """Dummy tokenizer for demonstration"""
    
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = 32000
    
    def encode(self, text: str, return_tensors=None):
        # Simple hash-based tokenization for demo
        tokens = [hash(word) % self.vocab_size for word in text.split()[:100]]
        if return_tensors == 'pt':
            return torch.tensor(tokens).unsqueeze(0)
        return tokens
    
    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return f"[Generated text from {len(tokens)} tokens]"
    
    def __call__(self, text, **kwargs):
        tokens = self.encode(text)
        max_length = kwargs.get('max_length', 512)
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return {
            'input_ids': torch.tensor(tokens).unsqueeze(0),
            'attention_mask': torch.ones(1, len(tokens))
        }

def create_sample_math_problems() -> List[Dict]:
    """Create sample math problems for PRM training"""
    
    problems = [
        {
            'problem': "Solve: 2x + 5 = 11",
            'solution_steps': [
                "Start with the equation: 2x + 5 = 11",
                "Subtract 5 from both sides: 2x = 6", 
                "Divide both sides by 2: x = 3",
                "Check: 2(3) + 5 = 6 + 5 = 11 ✓"
            ],
            'final_answer': "x = 3",
            'is_correct': True
        },
        {
            'problem': "Find the area of a circle with radius 4",
            'solution_steps': [
                "Use the formula: A = πr²",
                "Substitute r = 4: A = π(4)²",
                "Calculate: A = π × 16",
                "Final answer: A = 16π ≈ 50.27"
            ],
            'final_answer': "16π square units",
            'is_correct': True
        },
        {
            'problem': "Solve: x² - 5x + 6 = 0",
            'solution_steps': [
                "Factor the quadratic equation",
                "Look for two numbers that multiply to 6 and add to -5",
                "Those numbers are -2 and -3",
                "So: (x - 2)(x - 3) = 0",
                "Therefore: x = 2 or x = 3"
            ],
            'final_answer': "x = 2 or x = 3",
            'is_correct': True
        },
        # Add some incorrect examples
        {
            'problem': "Solve: 3x + 2 = 14",
            'solution_steps': [
                "Start with: 3x + 2 = 14",
                "Add 2 to both sides: 3x = 16",  # Error: should subtract
                "Divide by 3: x = 16/3"
            ],
            'final_answer': "x = 16/3",
            'is_correct': False
        }
    ]
    
    return problems

def create_sample_qa_pairs() -> List[Dict]:
    """Create sample Q&A pairs for ORM training"""
    
    qa_pairs = [
        {
            'question': "Explain photosynthesis",
            'reasoning': "Photosynthesis is the process by which plants convert sunlight into energy. It occurs in chloroplasts and involves two main stages: light reactions and the Calvin cycle.",
            'answer': "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            'factual_correctness': 0.9,
            'helpfulness_score': 0.8,
            'clarity_score': 0.9
        },
        {
            'question': "What causes seasons?",
            'reasoning': "Seasons are caused by Earth's axial tilt of 23.5 degrees. As Earth orbits the sun, different hemispheres receive varying amounts of sunlight throughout the year.",
            'answer': "Seasons are caused by Earth's tilted axis, which affects how much sunlight different parts of Earth receive during its orbit around the sun.",
            'factual_correctness': 0.95,
            'helpfulness_score': 0.9,
            'clarity_score': 0.85
        },
        {
            'question': "How do vaccines work?",
            'reasoning': "Vaccines contain weakened or inactive parts of organisms that cause disease. They stimulate the immune system to recognize and fight the disease if encountered later.",
            'answer': "Vaccines train your immune system to recognize and fight diseases by exposing it to safe versions of disease-causing organisms.",
            'factual_correctness': 0.88,
            'helpfulness_score': 0.85,
            'clarity_score': 0.9
        },
        # Lower quality examples
        {
            'question': "Explain gravity",
            'reasoning': "Gravity is a force that pulls things down.",
            'answer': "Gravity makes things fall.",
            'factual_correctness': 0.4,
            'helpfulness_score': 0.3,
            'clarity_score': 0.5
        }
    ]
    
    return qa_pairs

def demonstrate_reward_model_training():
    """Demonstrate training PRM and ORM models"""
    
    print("🧠 Training Reward Models")
    print("=" * 50)
    
    # Create base model
    base_model = create_llama_7b()
    tokenizer = DummyTokenizer()
    
    # Create training data
    math_problems = create_sample_math_problems()
    qa_pairs = create_sample_qa_pairs()
    
    print(f"Created {len(math_problems)} math problems for PRM training")
    print(f"Created {len(qa_pairs)} Q&A pairs for ORM training")
    
    # Prepare PRM data
    prm_data = create_step_reward_data_from_math_problems(math_problems)
    print(f"Generated {len(prm_data)} step-by-step examples")
    
    # Prepare ORM data  
    orm_data = create_outcome_reward_data_from_qa_pairs(qa_pairs)
    print(f"Generated {len(orm_data)} outcome examples")
    
    # Train PRM
    print("\n🔄 Training Process Reward Model...")
    try:
        prm_model = train_process_reward_model(
            base_model=copy.deepcopy(base_model),
            training_data=prm_data[:3],  # Small sample for demo
            tokenizer=tokenizer,
            num_epochs=1  # Short training for demo
        )
        print("✅ PRM training completed!")
    except Exception as e:
        print(f"❌ PRM training failed: {e}")
        prm_model = ProcessRewardModel(copy.deepcopy(base_model))
    
    # Train ORM
    print("\n📊 Training Outcome Reward Model...")
    try:
        orm_model = train_outcome_reward_model(
            base_model=copy.deepcopy(base_model),
            training_data=orm_data[:3],  # Small sample for demo
            tokenizer=tokenizer,
            num_epochs=1  # Short training for demo
        )
        print("✅ ORM training completed!")
    except Exception as e:
        print(f"❌ ORM training failed: {e}")
        from training_infra.rlhf.reward_model import RewardModelConfig
        config = RewardModelConfig(hidden_size=4096)
        orm_model = OutcomeRewardModel(copy.deepcopy(base_model), config)
    
    return prm_model, orm_model

def demonstrate_reward_guided_generation():
    """Demonstrate reward-guided text generation"""
    
    print("\n🎯 Reward-Guided Generation")
    print("=" * 50)
    
    # Get trained models
    prm_model, orm_model = demonstrate_reward_model_training()
    
    # Create base model for generation
    generation_model = create_llama_7b()
    tokenizer = DummyTokenizer()
    
    # Create reward-guided engine
    engine = create_reward_guided_engine(
        model=generation_model,
        prm=prm_model,
        orm=orm_model,
        tokenizer=tokenizer
    )
    
    # Test prompts
    test_prompts = [
        "Solve the math problem: 5x - 3 = 17",
        "Explain how plants make oxygen",
        "Calculate the perimeter of a rectangle with length 8 and width 5"
    ]
    
    # Different reward guidance strategies
    strategies = {
        "Beam Search": RewardGuidedConfig(
            search_strategy="beam_search",
            num_beams=4,
            use_prm=True,
            use_orm=True,
            prm_weight=0.3,
            orm_weight=0.7
        ),
        
        "Best of N": RewardGuidedConfig(
            search_strategy="best_of_n",
            num_candidates=6,
            use_prm=True,
            use_orm=True,
            prm_weight=0.4,
            orm_weight=0.6
        ),
        
        "Guided Sampling": RewardGuidedConfig(
            search_strategy="guided_sampling",
            use_prm=True,
            use_orm=False,  # Only use PRM for step-by-step
            prm_weight=0.5,
            reward_alpha=0.2,
            early_stopping=True
        ),
        
        "Tree Search": RewardGuidedConfig(
            search_strategy="tree_search",
            use_prm=True,
            use_orm=True,
            prm_weight=0.3,
            orm_weight=0.4,
            tree_depth=3,
            branching_factor=3
        )
    }
    
    results = {}
    
    # Test each strategy
    for strategy_name, config in strategies.items():
        print(f"\n🔍 Testing {strategy_name} Strategy")
        print("-" * 30)
        
        strategy_results = []
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            
            start_time = time.time()
            
            try:
                # Generate with reward guidance
                result = engine.generate(
                    prompt=prompt,
                    config=config,
                    max_length=200,
                    temperature=0.7
                )
                
                generation_time = time.time() - start_time
                
                print(f"Generated: {result['text'][:100]}...")
                print(f"PRM Score: {result.get('prm_score', 'N/A'):.3f}")
                print(f"ORM Score: {result.get('orm_score', 'N/A'):.3f}")
                print(f"Combined Score: {result.get('combined_score', 'N/A'):.3f}")
                print(f"Time: {generation_time:.2f}s")
                
                strategy_results.append({
                    'prompt': prompt,
                    'result': result,
                    'time': generation_time,
                    'success': True
                })
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                strategy_results.append({
                    'prompt': prompt,
                    'result': None,
                    'time': 0,
                    'success': False,
                    'error': str(e)
                })
        
        results[strategy_name] = strategy_results
    
    return results

def compare_generation_strategies(results: Dict):
    """Compare different generation strategies"""
    
    print("\n📊 Strategy Comparison")
    print("=" * 50)
    
    # Calculate metrics
    metrics = {}
    
    for strategy_name, strategy_results in results.items():
        successful_results = [r for r in strategy_results if r['success']]
        
        if successful_results:
            avg_time = np.mean([r['time'] for r in successful_results])
            success_rate = len(successful_results) / len(strategy_results)
            
            # Get scores if available
            prm_scores = []
            orm_scores = []
            combined_scores = []
            
            for r in successful_results:
                if r['result']:
                    prm_scores.append(r['result'].get('prm_score', 0))
                    orm_scores.append(r['result'].get('orm_score', 0))
                    combined_scores.append(r['result'].get('combined_score', 0))
            
            metrics[strategy_name] = {
                'avg_time': avg_time,
                'success_rate': success_rate,
                'avg_prm_score': np.mean(prm_scores) if prm_scores else 0,
                'avg_orm_score': np.mean(orm_scores) if orm_scores else 0,
                'avg_combined_score': np.mean(combined_scores) if combined_scores else 0
            }
        else:
            metrics[strategy_name] = {
                'avg_time': 0,
                'success_rate': 0,
                'avg_prm_score': 0,
                'avg_orm_score': 0,
                'avg_combined_score': 0
            }
    
    # Print comparison table
    print(f"{'Strategy':<15} {'Success Rate':<12} {'Avg Time':<10} {'PRM Score':<10} {'ORM Score':<10} {'Combined':<10}")
    print("-" * 80)
    
    for strategy_name, metric in metrics.items():
        print(f"{strategy_name:<15} "
              f"{metric['success_rate']:<12.2%} "
              f"{metric['avg_time']:<10.2f} "
              f"{metric['avg_prm_score']:<10.3f} "
              f"{metric['avg_orm_score']:<10.3f} "
              f"{metric['avg_combined_score']:<10.3f}")
    
    return metrics

def visualize_results(metrics: Dict):
    """Create visualizations of the results"""
    
    print("\n📈 Creating Visualizations")
    print("=" * 30)
    
    strategies = list(metrics.keys())
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward-Guided Generation Strategy Comparison', fontsize=16)
    
    # Success Rate
    success_rates = [metrics[s]['success_rate'] for s in strategies]
    ax1.bar(strategies, success_rates, color='skyblue')
    ax1.set_title('Success Rate by Strategy')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 0.01, f'{v:.2%}', ha='center')
    
    # Average Time
    avg_times = [metrics[s]['avg_time'] for s in strategies]
    ax2.bar(strategies, avg_times, color='lightcoral')
    ax2.set_title('Average Generation Time')
    ax2.set_ylabel('Time (seconds)')
    for i, v in enumerate(avg_times):
        ax2.text(i, v + 0.01, f'{v:.2f}s', ha='center')
    
    # Reward Scores Comparison
    prm_scores = [metrics[s]['avg_prm_score'] for s in strategies]
    orm_scores = [metrics[s]['avg_orm_score'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax3.bar(x - width/2, prm_scores, width, label='PRM Score', color='lightgreen')
    ax3.bar(x + width/2, orm_scores, width, label='ORM Score', color='orange')
    ax3.set_title('Average Reward Scores')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.legend()
    
    # Combined Score
    combined_scores = [metrics[s]['avg_combined_score'] for s in strategies]
    ax4.bar(strategies, combined_scores, color='gold')
    ax4.set_title('Combined Reward Score')
    ax4.set_ylabel('Combined Score')
    for i, v in enumerate(combined_scores):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('reward_guided_generation_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'reward_guided_generation_comparison.png'")
    
    return fig

def demonstrate_adaptive_guidance():
    """Demonstrate adaptive reward guidance based on problem type"""
    
    print("\n🧬 Adaptive Reward Guidance")
    print("=" * 50)
    
    # Create models
    prm_model, orm_model = demonstrate_reward_model_training()
    generation_model = create_llama_7b()
    tokenizer = DummyTokenizer()
    
    engine = create_reward_guided_engine(
        model=generation_model,
        prm=prm_model,
        orm=orm_model,
        tokenizer=tokenizer
    )
    
    # Different problem types with adaptive configs
    problem_configs = {
        "Mathematical Problem": {
            "prompt": "Solve step by step: If a train travels 120 miles in 2 hours, what is its speed?",
            "config": RewardGuidedConfig(
                search_strategy="guided_sampling",
                use_prm=True,
                use_orm=False,  # Focus on step-by-step reasoning
                prm_weight=0.8,
                reward_alpha=0.3,
                early_stopping=True
            )
        },
        
        "Creative Writing": {
            "prompt": "Write a short story about a robot learning to paint",
            "config": RewardGuidedConfig(
                search_strategy="best_of_n",
                num_candidates=8,
                use_prm=False,  # Don't need step-by-step for creativity
                use_orm=True,
                orm_weight=0.9,
                diversity_penalty=0.2
            )
        },
        
        "Factual Explanation": {
            "prompt": "Explain how antibiotics work in the human body",
            "config": RewardGuidedConfig(
                search_strategy="beam_search",
                num_beams=5,
                use_prm=True,
                use_orm=True,
                prm_weight=0.4,  # Balanced approach
                orm_weight=0.6,
                early_stopping=True
            )
        },
        
        "Complex Reasoning": {
            "prompt": "Analyze the pros and cons of renewable energy adoption",
            "config": RewardGuidedConfig(
                search_strategy="tree_search",
                use_prm=True,
                use_orm=True,
                prm_weight=0.5,
                orm_weight=0.5,
                tree_depth=4,
                branching_factor=3
            )
        }
    }
    
    adaptive_results = {}
    
    for problem_type, problem_data in problem_configs.items():
        print(f"\n🎯 Testing {problem_type}")
        print("-" * 40)
        print(f"Prompt: {problem_data['prompt']}")
        
        try:
            result = engine.generate(
                prompt=problem_data['prompt'],
                config=problem_data['config'],
                max_length=250,
                temperature=0.7
            )
            
            print(f"Generated: {result['text'][:150]}...")
            print(f"Strategy: {problem_data['config'].search_strategy}")
            print(f"PRM Focus: {problem_data['config'].use_prm}")
            print(f"ORM Focus: {problem_data['config'].use_orm}")
            
            adaptive_results[problem_type] = result
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            adaptive_results[problem_type] = None
    
    return adaptive_results

def save_experiment_results(results: Dict, metrics: Dict, adaptive_results: Dict):
    """Save all experiment results to JSON file"""
    
    print("\n💾 Saving Results")
    print("=" * 30)
    
    # Prepare data for JSON serialization
    serializable_results = {}
    
    for strategy_name, strategy_results in results.items():
        serializable_results[strategy_name] = []
        for result in strategy_results:
            serializable_result = {
                'prompt': result['prompt'],
                'success': result['success'],
                'time': result['time']
            }
            
            if result['success'] and result['result']:
                serializable_result.update({
                    'generated_text': result['result'].get('text', ''),
                    'prm_score': result['result'].get('prm_score', 0),
                    'orm_score': result['result'].get('orm_score', 0),
                    'combined_score': result['result'].get('combined_score', 0)
                })
            
            if not result['success']:
                serializable_result['error'] = result.get('error', '')
            
            serializable_results[strategy_name].append(serializable_result)
    
    # Prepare adaptive results
    serializable_adaptive = {}
    for problem_type, result in adaptive_results.items():
        if result:
            serializable_adaptive[problem_type] = {
                'generated_text': result.get('text', ''),
                'prm_score': result.get('prm_score', 0),
                'orm_score': result.get('orm_score', 0),
                'combined_score': result.get('combined_score', 0)
            }
        else:
            serializable_adaptive[problem_type] = None
    
    # Complete experiment data
    experiment_data = {
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
        'strategy_results': serializable_results,
        'metrics': metrics,
        'adaptive_results': serializable_adaptive,
        'experiment_info': {
            'model_type': 'LLaMA-7B',
            'num_strategies_tested': len(results),
            'num_prompts_per_strategy': len(test_prompts),
            'adaptive_problem_types': len(adaptive_results)
        }
    }
    
    # Save to file
    output_file = Path(f"reward_guided_experiment_{experiment_data['timestamp']}.json")
    with open(output_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"📁 Results saved to {output_file}")
    
    return output_file

def main():
    """Main demonstration function"""
    
    print("🚀 Reward-Guided Inference Demonstration")
    print("=" * 60)
    print("This example demonstrates:")
    print("• Training Process Reward Models (PRM)")
    print("• Training Outcome Reward Models (ORM)")
    print("• Different reward-guided generation strategies")
    print("• Adaptive guidance based on problem type")
    print("• Performance comparison and visualization")
    print("=" * 60)
    
    try:
        # Step 1: Demonstrate basic reward-guided generation
        results = demonstrate_reward_guided_generation()
        
        # Step 2: Compare strategies
        metrics = compare_generation_strategies(results)
        
        # Step 3: Create visualizations
        fig = visualize_results(metrics)
        
        # Step 4: Demonstrate adaptive guidance
        adaptive_results = demonstrate_adaptive_guidance()
        
        # Step 5: Save all results
        output_file = save_experiment_results(results, metrics, adaptive_results)
        
        # Final summary
        print("\n🎉 Experiment Complete!")
        print("=" * 40)
        print(f"✅ Tested {len(results)} generation strategies")
        print(f"✅ Evaluated {len(adaptive_results)} adaptive configurations")
        print(f"✅ Generated comparison visualization")
        print(f"✅ Saved results to {output_file}")
        
        # Best strategy recommendation
        best_strategy = max(metrics.keys(), 
                           key=lambda k: metrics[k]['avg_combined_score'])
        print(f"\n🏆 Best performing strategy: {best_strategy}")
        print(f"   Combined Score: {metrics[best_strategy]['avg_combined_score']:.3f}")
        print(f"   Success Rate: {metrics[best_strategy]['success_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()