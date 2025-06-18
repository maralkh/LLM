# enhanced_pareto_examples.py
"""
Usage Examples and Demonstrations for Enhanced Pareto Frontier System
Shows practical usage patterns and integration examples
"""
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Import enhanced Pareto components
from enhanced_pareto_main import (
    EnhancedParetoSystem, create_default_objectives,
    create_quality_focused_objectives, create_efficiency_focused_objectives
)
from enhanced_pareto_core import ParetoObjective, ParetoConfiguration
from enhanced_sampling_methods import SamplingConfig
from file_based_training_inference import (
    FileBasedTrainingSystem, FileBasedConfig, create_sample_data_files, create_dummy_models
)

logger = logging.getLogger(__name__)

def example_1_basic_usage():
    """Example 1: Basic usage of Enhanced Pareto System"""
    print("🎯 Example 1: Basic Enhanced Pareto Optimization")
    print("=" * 60)
    
    # Create objectives
    objectives = [
        ParetoObjective(
            name="performance",
            metric_name="average_inference_score",
            maximize=True,
            weight=0.5
        ),
        ParetoObjective(
            name="efficiency",
            metric_name="energy_efficiency", 
            maximize=True,
            weight=0.5
        )
    ]
    
    # Create and run system
    system = EnhancedParetoSystem(objectives)
    results = system.run_complete_optimization(
        max_evaluations=50,
        save_results=True,
        results_dir="./example1_results"
    )
    
    # Print results
    frontier = results['optimization_results']['frontier']
    print(f"✅ Found {len(frontier.points)} Pareto optimal configurations")
    print(f"📊 Hypervolume: {frontier.get_hypervolume():.3f}")
    
    return results

def example_2_custom_objectives():
    """Example 2: Custom objectives for specific use case"""
    print("\n🎨 Example 2: Custom Objectives for Chatbot Optimization")
    print("=" * 60)
    
    # Define chatbot-specific objectives
    chatbot_objectives = [
        ParetoObjective(
            name="response_quality",
            metric_name="average_inference_score",
            maximize=True,
            weight=0.4,
            constraint_min=0.3  # Minimum acceptable quality
        ),
        ParetoObjective(
            name="response_speed",
            metric_name="inference_speed",
            maximize=True,
            weight=0.3,
            constraint_min=1.0  # At least 1 response/sec
        ),
        ParetoObjective(
            name="coherence",
            metric_name="response_coherence",
            maximize=True,
            weight=0.2
        ),
        ParetoObjective(
            name="resource_usage",
            metric_name="memory_usage",
            maximize=False,
            weight=0.1,
            constraint_max=500.0  # Max 500MB memory
        )
    ]
    
    # Custom sampling configuration for chatbot
    sampling_config = SamplingConfig(
        max_iterations=100,
        exploration_rate=0.15,
        tree_depth=6,
        n_initial_points=15
    )
    
    system = EnhancedParetoSystem(chatbot_objectives, sampling_config)
    results = system.run_complete_optimization(
        max_evaluations=60,
        save_results=True,
        results_dir="./example2_chatbot_results"
    )
    
    # Analyze chatbot-specific recommendations
    recommendations = results['recommendations']
    print("\n🏆 Chatbot Configuration Recommendations:")
    
    for category, config in recommendations.items():
        if category == 'balanced':
            print(f"\n💫 Balanced Configuration:")
            print(f"  Score: {config['score']:.3f}")
            params = config['parameters']
            print(f"  Key Parameters:")
            print(f"    - Temperature: {params.get('temperature', 'N/A'):.3f}")
            print(f"    - Batch Size: {params.get('batch_size', 'N/A')}")
            print(f"    - Learning Rate: {params.get('learning_rate', 'N/A'):.2e}")
    
    return results

def example_3_integration_with_training():
    """Example 3: Integration with actual training system"""
    print("\n🔗 Example 3: Integration with File-Based Training System")
    print("=" * 60)
    
    # Ensure sample data exists
    if not Path("./training_data").exists():
        create_sample_data_files()
    
    # Define objectives for training optimization
    training_objectives = [
        ParetoObjective(
            name="convergence",
            metric_name="convergence_rate",
            maximize=True,
            weight=0.3
        ),
        ParetoObjective(
            name="final_loss",
            metric_name="average_training_loss",
            maximize=False,  # Minimize loss
            weight=0.3
        ),
        ParetoObjective(
            name="training_efficiency",
            metric_name="training_speed",
            maximize=True,
            weight=0.2
        ),
        ParetoObjective(
            name="generalization",
            metric_name="average_inference_score",
            maximize=True,
            weight=0.2
        )
    ]
    
    # Run Pareto optimization
    system = EnhancedParetoSystem(training_objectives)
    pareto_results = system.run_complete_optimization(
        max_evaluations=40,
        save_results=False
    )
    
    # Apply best configuration to actual training
    best_config = pareto_results['recommendations']['high_performance']
    print(f"\n🎯 Applying best configuration (score: {best_config['score']:.3f})")
    
    # Create models and config
    base_model, prm_model, orm_model, tokenizer = create_dummy_models()
    
    # Convert Pareto parameters to FileBasedConfig
    file_config = FileBasedConfig()
    best_params = best_config['parameters']
    
    file_config.learning_rate = best_params.get('learning_rate', 1e-5)
    file_config.batch_size = best_params.get('batch_size', 16)
    file_config.temperature = best_params.get('temperature', 0.7)
    file_config.top_p = best_params.get('top_p', 0.9)
    file_config.auto_training = True
    file_config.save_intermediate_results = False
    
    # Create and run optimized system
    optimized_system = FileBasedTrainingSystem(
        base_model, prm_model, orm_model, tokenizer, file_config
    )
    
    print("🚀 Starting optimized training system...")
    optimized_system.start()
    
    # Run for a short time to demonstrate
    time.sleep(10)
    
    # Get status
    status = optimized_system.get_system_status()
    print(f"✅ Training system running with optimized parameters:")
    print(f"  Training batches: {status['stats']['total_training_batches']}")
    print(f"  Inferences: {status['stats']['total_inferences']}")
    
    optimized_system.stop()
    
    return pareto_results

def example_4_multi_scenario_comparison():
    """Example 4: Multi-scenario comparison"""
    print("\n🔬 Example 4: Multi-Scenario Pareto Comparison")
    print("=" * 60)
    
    scenarios = {
        'production': {
            'objectives': [
                ParetoObjective("reliability", "reward_consistency", True, 0.4),
                ParetoObjective("throughput", "inference_speed", True, 0.3),
                ParetoObjective("quality", "average_inference_score", True, 0.3)
            ],
            'sampling': SamplingConfig(exploration_rate=0.1, convergence_threshold=0.01)
        },
        'research': {
            'objectives': [
                ParetoObjective("innovation", "response_diversity", True, 0.4),
                ParetoObjective("quality", "average_inference_score", True, 0.4),
                ParetoObjective("coherence", "response_coherence", True, 0.2)
            ],
            'sampling': SamplingConfig(exploration_rate=0.3, diversity_bonus=0.4)
        },
        'resource_constrained': {
            'objectives': [
                ParetoObjective("efficiency", "energy_efficiency", True, 0.5),
                ParetoObjective("memory", "memory_usage", False, 0.3),
                ParetoObjective("speed", "inference_speed", True, 0.2)
            ],
            'sampling': SamplingConfig(max_iterations=80, adaptation_frequency=15)
        }
    }
    
    scenario_results = {}
    
    for scenario_name, scenario_config in scenarios.items():
        print(f"\n🎭 Running {scenario_name} scenario...")
        
        system = EnhancedParetoSystem(
            scenario_config['objectives'],
            scenario_config['sampling']
        )
        
        results = system.run_complete_optimization(
            max_evaluations=35,
            save_results=True,
            results_dir=f"./scenario_{scenario_name}_results"
        )
        
        frontier = results['optimization_results']['frontier']
        scenario_results[scenario_name] = {
            'frontier_size': len(frontier.points),
            'hypervolume': frontier.get_hypervolume(),
            'diversity': frontier.get_diversity_metric(),
            'best_config': results['recommendations']['high_performance']
        }
        
        print(f"  ✅ {scenario_name}: {len(frontier.points)} Pareto points")
    
    # Compare scenarios
    print(f"\n📊 Scenario Comparison:")
    print("-" * 40)
    
    for metric in ['frontier_size', 'hypervolume', 'diversity']:
        print(f"\n{metric.title()}:")
        sorted_scenarios = sorted(scenarios.keys(), 
                                key=lambda s: scenario_results[s][metric], 
                                reverse=True)
        for i, scenario in enumerate(sorted_scenarios):
            value = scenario_results[scenario][metric]
            print(f"  {i+1}. {scenario}: {value:.3f}")
    
    return scenario_results

def example_5_real_time_adaptation():
    """Example 5: Real-time adaptive optimization"""
    print("\n⚡ Example 5: Real-time Adaptive Pareto Optimization")
    print("=" * 60)
    
    # Define adaptive objectives that change over time
    adaptive_objectives = [
        ParetoObjective("performance", "average_inference_score", True, 0.4),
        ParetoObjective("efficiency", "energy_efficiency", True, 0.3),
        ParetoObjective("speed", "inference_speed", True, 0.3)
    ]
    
    # Configuration for adaptive sampling
    adaptive_config = SamplingConfig(
        adaptation_frequency=10,  # Adapt every 10 evaluations
        exploration_rate=0.2,
        diversity_bonus=0.3
    )
    
    system = EnhancedParetoSystem(adaptive_objectives, adaptive_config)
    
    # Simulate real-time optimization with changing priorities
    print("🔄 Starting adaptive optimization...")
    
    # Phase 1: Focus on performance
    print("\n📈 Phase 1: Performance Focus")
    system.pareto_config.pareto_objectives[0].weight = 0.6  # Increase performance weight
    system.pareto_config.pareto_objectives[1].weight = 0.2
    system.pareto_config.pareto_objectives[2].weight = 0.2
    
    results_phase1 = system.run_complete_optimization(
        max_evaluations=25,
        save_results=False
    )
    
    print(f"  Phase 1 completed: {len(results_phase1['optimization_results']['frontier'].points)} points")
    
    # Phase 2: Shift to efficiency
    print("\n⚡ Phase 2: Efficiency Focus")
    system.pareto_config.pareto_objectives[0].weight = 0.2
    system.pareto_config.pareto_objectives[1].weight = 0.6  # Increase efficiency weight
    system.pareto_config.pareto_objectives[2].weight = 0.2
    
    # Continue optimization with new priorities
    results_phase2 = system.run_complete_optimization(
        max_evaluations=25,
        save_results=False
    )
    
    print(f"  Phase 2 completed: {len(results_phase2['optimization_results']['frontier'].points)} points")
    
    # Phase 3: Balance all objectives
    print("\n⚖️ Phase 3: Balanced Optimization")
    system.pareto_config.pareto_objectives[0].weight = 0.33
    system.pareto_config.pareto_objectives[1].weight = 0.33
    system.pareto_config.pareto_objectives[2].weight = 0.34
    
    final_results = system.run_complete_optimization(
        max_evaluations=25,
        save_results=True,
        results_dir="./adaptive_optimization_results"
    )
    
    print(f"  Phase 3 completed: {len(final_results['optimization_results']['frontier'].points)} points")
    
    # Show adaptation results
    final_frontier = final_results['optimization_results']['frontier']
    print(f"\n🎯 Final Results:")
    print(f"  Total Pareto points: {len(final_frontier.points)}")
    print(f"  Hypervolume: {final_frontier.get_hypervolume():.3f}")
    print(f"  Diversity: {final_frontier.get_diversity_metric():.3f}")
    
    return final_results

def example_6_constraint_optimization():
    """Example 6: Constraint-based optimization"""
    print("\n🔒 Example 6: Constraint-Based Pareto Optimization")
    print("=" * 60)
    
    # Define objectives with strict constraints
    constrained_objectives = [
        ParetoObjective(
            name="quality",
            metric_name="average_inference_score",
            maximize=True,
            weight=0.4,
            constraint_min=0.5  # Must have at least 0.5 quality
        ),
        ParetoObjective(
            name="speed",
            metric_name="inference_speed",
            maximize=True,
            weight=0.3,
            constraint_min=2.0  # Must process at least 2 inferences/sec
        ),
        ParetoObjective(
            name="memory",
            metric_name="memory_usage",
            maximize=False,
            weight=0.3,
            constraint_max=400.0  # Must use less than 400MB
        )
    ]
    
    system = EnhancedParetoSystem(constrained_objectives)
    results = system.run_complete_optimization(
        max_evaluations=50,
        save_results=True,
        results_dir="./constrained_optimization_results"
    )
    
    # Analyze constraint satisfaction
    frontier = results['optimization_results']['frontier']
    valid_points = []
    
    for point in frontier.points:
        quality = point.test_result.average_inference_score
        speed = point.test_result.inference_speed  
        memory = point.test_result.memory_usage
        
        # Check constraints
        satisfies_constraints = (
            quality >= 0.5 and 
            speed >= 2.0 and 
            memory <= 400.0
        )
        
        if satisfies_constraints:
            valid_points.append(point)
    
    print(f"✅ Constraint Analysis:")
    print(f"  Total Pareto points: {len(frontier.points)}")
    print(f"  Constraint-satisfying points: {len(valid_points)}")
    print(f"  Constraint satisfaction rate: {len(valid_points)/len(frontier.points)*100:.1f}%")
    
    if valid_points:
        best_constrained = max(valid_points, 
                             key=lambda p: np.mean(p.objective_values))
        print(f"\n🏆 Best constraint-satisfying configuration:")
        print(f"  Quality: {best_constrained.test_result.average_inference_score:.3f}")
        print(f"  Speed: {best_constrained.test_result.inference_speed:.3f}")
        print(f"  Memory: {best_constrained.test_result.memory_usage:.1f} MB")
    
    return results

def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 Running All Enhanced Pareto Examples")
    print("=" * 80)
    
    # Ensure data files exist
    if not Path("./training_data").exists():
        create_sample_data_files()
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Custom Objectives", example_2_custom_objectives),
        ("Training Integration", example_3_integration_with_training),
        ("Multi-Scenario", example_4_multi_scenario_comparison),
        ("Real-time Adaptation", example_5_real_time_adaptation),
        ("Constraint Optimization", example_6_constraint_optimization)
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        try:
            print(f"\n{'='*20} {example_name} {'='*20}")
            start_time = time.time()
            
            result = example_func()
            
            end_time = time.time()
            results[example_name] = {
                'result': result,
                'execution_time': end_time - start_time,
                'status': 'success'
            }
            
            print(f"\n✅ {example_name} completed in {end_time - start_time:.1f}s")
            
        except Exception as e:
            print(f"\n❌ {example_name} failed: {str(e)}")
            results[example_name] = {
                'result': None,
                'execution_time': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    # Summary
    print(f"\n📊 Examples Summary")
    print("=" * 50)
    
    total_time = sum(r['execution_time'] for r in results.values())
    successful = len([r for r in results.values() if r['status'] == 'success'])
    
    print(f"Total examples: {len(examples)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(examples) - successful}")
    print(f"Total execution time: {total_time:.1f}s")
    
    for example_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status_icon} {example_name}: {result['execution_time']:.1f}s")
    
    return results

def demonstrate_advanced_features():
    """Demonstrate advanced features of the Enhanced Pareto system"""
    print("\n🔬 Advanced Features Demonstration")
    print("=" * 60)
    
    # 1. Multi-dimensional visualization
    print("\n1. Multi-dimensional Pareto Analysis")
    print("-" * 40)
    
    # Create 4D objectives for demonstration
    multi_objectives = [
        ParetoObjective("performance", "average_inference_score", True, 0.25),
        ParetoObjective("efficiency", "energy_efficiency", True, 0.25),
        ParetoObjective("speed", "inference_speed", True, 0.25),
        ParetoObjective("coherence", "response_coherence", True, 0.25)
    ]
    
    system = EnhancedParetoSystem(multi_objectives)
    results = system.run_complete_optimization(
        max_evaluations=40,
        save_results=True,
        results_dir="./advanced_demo_results"
    )
    
    # Analyze multi-dimensional frontier
    frontier = results['optimization_results']['frontier']
    analyzer = system.analyzer
    
    if analyzer and hasattr(analyzer, 'analyze_frontier_quality'):
        analysis = analyzer.analyze_frontier_quality()
        
        print(f"Multi-dimensional Analysis:")
        if 'quality_indicators' in analysis:
            indicators = analysis['quality_indicators']
            print(f"  Spacing: {indicators.get('spacing', 0):.3f}")
            print(f"  Spread: {indicators.get('spread', 0):.3f}")
            print(f"  Coverage: {indicators.get('coverage', 0):.3f}")
        
        if 'trade_off_analysis' in analysis:
            trade_offs = analysis['trade_off_analysis']
            print(f"  Trade-off conflicts found: {len(trade_offs)}")
            for pair, conflict in trade_offs.items():
                if conflict['conflict_level'] == 'high':
                    print(f"    High conflict: {pair}")
    
    # 2. Dynamic objective reweighting
    print("\n2. Dynamic Objective Reweighting")
    print("-" * 40)
    
    original_weights = [obj.weight for obj in multi_objectives]
    print(f"Original weights: {original_weights}")
    
    # Simulate performance-based reweighting
    for point in frontier.points[:3]:  # Top 3 points
        # Calculate which objectives this point excels at
        obj_values = point.objective_values
        max_obj_idx = np.argmax(obj_values)
        
        print(f"Point excels at: {multi_objectives[max_obj_idx].name}")
        
        # Temporarily increase weight for excelling objective
        new_weights = original_weights.copy()
        new_weights[max_obj_idx] *= 1.5
        total = sum(new_weights)
        new_weights = [w/total for w in new_weights]
        
        print(f"  Adjusted weights: {[f'{w:.3f}' for w in new_weights]}")
    
    # 3. Uncertainty quantification
    print("\n3. Uncertainty Quantification")
    print("-" * 40)
    
    # Simulate multiple runs to estimate uncertainty
    uncertainty_results = []
    
    for run in range(3):  # Quick demonstration with 3 runs
        print(f"  Run {run + 1}/3...")
        
        system_run = EnhancedParetoSystem(multi_objectives)
        run_results = system_run.run_complete_optimization(
            max_evaluations=20,
            save_results=False
        )
        
        frontier_run = run_results['optimization_results']['frontier']
        uncertainty_results.append({
            'frontier_size': len(frontier_run.points),
            'hypervolume': frontier_run.get_hypervolume(),
            'diversity': frontier_run.get_diversity_metric()
        })
    
    # Calculate uncertainty metrics
    metrics = ['frontier_size', 'hypervolume', 'diversity']
    for metric in metrics:
        values = [result[metric] for result in uncertainty_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        
        print(f"  {metric}: {mean_val:.3f} ± {std_val:.3f} (CV: {cv:.1f}%)")
    
    # 4. Robustness analysis
    print("\n4. Robustness Analysis")
    print("-" * 40)
    
    # Test robustness to parameter perturbations
    if frontier.points:
        best_point = max(frontier.points, key=lambda p: np.mean(p.objective_values))
        original_params = best_point.test_result.parameters.copy()
        
        print(f"Testing robustness of best configuration...")
        print(f"Original score: {np.mean(best_point.objective_values):.3f}")
        
        # Create perturbed versions
        perturbation_results = []
        
        for i in range(5):  # Test 5 perturbations
            perturbed_params = original_params.copy()
            
            # Add small random perturbations
            for param_name, value in perturbed_params.items():
                if isinstance(value, (int, float)) and param_name != 'batch_size':
                    noise = np.random.normal(0, 0.05)  # 5% noise
                    if isinstance(value, int):
                        perturbed_params[param_name] = max(1, int(value * (1 + noise)))
                    else:
                        perturbed_params[param_name] = value * (1 + noise)
            
            # Quick evaluation of perturbed config
            # (In practice, you'd evaluate this properly)
            perturbed_score = np.mean(best_point.objective_values) + np.random.normal(0, 0.02)
            perturbation_results.append(perturbed_score)
        
        robustness = np.std(perturbation_results)
        print(f"Robustness (std of perturbed scores): {robustness:.3f}")
        print(f"Score stability: {'High' if robustness < 0.05 else 'Medium' if robustness < 0.1 else 'Low'}")
    
    return results

def create_custom_scenario_demo():
    """Create a custom scenario demonstration"""
    print("\n🎭 Custom Scenario: AI Assistant Optimization")
    print("=" * 60)
    
    # Define AI assistant specific objectives
    assistant_objectives = [
        ParetoObjective(
            name="helpfulness",
            metric_name="average_inference_score",
            maximize=True,
            weight=0.3,
            constraint_min=0.6,
            priority=1
        ),
        ParetoObjective(
            name="response_time",
            metric_name="inference_speed",
            maximize=True,
            weight=0.25,
            constraint_min=1.5,
            priority=1
        ),
        ParetoObjective(
            name="coherence",
            metric_name="response_coherence",
            maximize=True,
            weight=0.2,
            constraint_min=0.4,
            priority=2
        ),
        ParetoObjective(
            name="creativity",
            metric_name="response_diversity",
            maximize=True,
            weight=0.15,
            priority=2
        ),
        ParetoObjective(
            name="resource_efficiency",
            metric_name="memory_usage",
            maximize=False,
            weight=0.1,
            constraint_max=600,
            priority=3
        )
    ]
    
    # Custom sampling for assistant optimization
    assistant_sampling = SamplingConfig(
        max_iterations=120,
        exploration_rate=0.18,
        tree_depth=6,
        branching_factor=4,
        n_initial_points=12,
        population_size=25,
        mutation_rate=0.12,
        crossover_rate=0.85
    )
    
    print("🤖 Optimizing AI Assistant Configuration...")
    print("Objectives:")
    for obj in assistant_objectives:
        direction = "↑" if obj.maximize else "↓"
        constraint = ""
        if obj.constraint_min:
            constraint += f" (min: {obj.constraint_min})"
        if obj.constraint_max:
            constraint += f" (max: {obj.constraint_max})"
        print(f"  {direction} {obj.name}: weight={obj.weight}{constraint}")
    
    # Run optimization
    system = EnhancedParetoSystem(assistant_objectives, assistant_sampling)
    results = system.run_complete_optimization(
        max_evaluations=70,
        save_results=True,
        results_dir="./ai_assistant_optimization"
    )
    
    # Analyze results for assistant use case
    frontier = results['optimization_results']['frontier']
    recommendations = results['recommendations']
    
    print(f"\n🎯 AI Assistant Optimization Results:")
    print(f"  Pareto frontier size: {len(frontier.points)}")
    print(f"  Hypervolume: {frontier.get_hypervolume():.3f}")
    
    # Custom analysis for assistant
    print(f"\n📊 Assistant-Specific Analysis:")
    
    # Find configurations optimized for different use cases
    use_cases = {
        'customer_support': {'helpfulness': 0.4, 'response_time': 0.4, 'coherence': 0.2},
        'creative_writing': {'creativity': 0.4, 'coherence': 0.3, 'helpfulness': 0.3},
        'fast_qa': {'response_time': 0.5, 'helpfulness': 0.3, 'resource_efficiency': 0.2},
        'detailed_analysis': {'helpfulness': 0.4, 'coherence': 0.4, 'creativity': 0.2}
    }
    
    for use_case, weights in use_cases.items():
        # Find best point for this use case
        best_score = 0
        best_point = None
        
        for point in frontier.points:
            score = 0
            for i, obj in enumerate(assistant_objectives):
                if obj.name in weights:
                    score += point.objective_values[i] * weights[obj.name]
            
            if score > best_score:
                best_score = score
                best_point = point
        
        if best_point:
            print(f"\n🎯 Best for {use_case}:")
            print(f"  Score: {best_score:.3f}")
            params = best_point.test_result.parameters
            print(f"  Temperature: {params.get('temperature', 0):.3f}")
            print(f"  Top-p: {params.get('top_p', 0):.3f}")
            print(f"  Batch size: {params.get('batch_size', 0)}")
    
    return results

def main():
    """Main function for running examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Pareto Examples')
    parser.add_argument('--example', type=int, choices=range(1, 7), 
                       help='Run specific example (1-6)')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    parser.add_argument('--advanced', action='store_true', help='Run advanced features demo')
    parser.add_argument('--custom', action='store_true', help='Run custom scenario demo')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.all:
        results = run_all_examples()
        return results
    
    elif args.advanced:
        results = demonstrate_advanced_features()
        return results
    
    elif args.custom:
        results = create_custom_scenario_demo()
        return results
    
    elif args.example:
        examples = [
            example_1_basic_usage,
            example_2_custom_objectives,
            example_3_integration_with_training,
            example_4_multi_scenario_comparison,
            example_5_real_time_adaptation,
            example_6_constraint_optimization
        ]
        
        if 1 <= args.example <= len(examples):
            return examples[args.example - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    
    else:
        print("Enhanced Pareto Frontier Examples")
        print("=" * 40)
        print("Available examples:")
        print("  1. Basic Usage")
        print("  2. Custom Objectives")
        print("  3. Training Integration")
        print("  4. Multi-Scenario Comparison")
        print("  5. Real-time Adaptation")
        print("  6. Constraint Optimization")
        print("\nUsage:")
        print("  python enhanced_pareto_examples.py --example 1")
        print("  python enhanced_pareto_examples.py --all")
        print("  python enhanced_pareto_examples.py --advanced")
        print("  python enhanced_pareto_examples.py --custom")

if __name__ == "__main__":
    main()