# enhanced_pareto_main.py
"""
Main execution and integration for Enhanced Pareto Frontier System
Combines advanced sampling with comprehensive Pareto optimization
"""
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse

# Import core components
from enhanced_pareto_core import (
    ParetoObjective, ParetoConfiguration, EnhancedParetoSampler,
    ParetoFrontier, ParetoPoint
)
from enhanced_pareto_analysis import ParetoAnalyzer, ParetoVisualizer
from enhanced_sampling_methods import SamplingConfig, ParameterSpace
from file_based_training_inference import create_sample_data_files

logger = logging.getLogger(__name__)

class EnhancedParetoSystem:
    """Complete Enhanced Pareto Frontier Optimization System"""
    
    def __init__(self, objectives: List[ParetoObjective], 
                 sampling_config: Optional[SamplingConfig] = None):
        """Initialize the enhanced Pareto system"""
        self.objectives = objectives
        self.sampling_config = sampling_config or SamplingConfig()
        self.pareto_config = ParetoConfiguration(
            sampling_config=self.sampling_config,
            pareto_objectives=objectives
        )
        
        # Initialize components
        self.parameter_space = ParameterSpace()
        self.sampler = None
        self.analyzer = None
        self.visualizer = None
        
        # Results storage
        self.optimization_results = None
        self.analysis_results = None
        
    def run_complete_optimization(self, max_evaluations: int = 200,
                                 save_results: bool = True,
                                 results_dir: str = "./enhanced_pareto_results") -> Dict[str, Any]:
        """Run complete Pareto optimization with analysis"""
        logger.info("🚀 Starting Complete Enhanced Pareto Optimization")
        
        # Create results directory
        if save_results:
            Path(results_dir).mkdir(exist_ok=True)
        
        # Ensure data files exist
        if not Path("./training_data").exists():
            create_sample_data_files()
        
        # Phase 1: Multi-objective optimization
        print("Phase 1: Multi-objective Pareto Optimization")
        print("=" * 50)
        
        self.sampler = EnhancedParetoSampler(self.parameter_space, self.pareto_config)
        self.optimization_results = self.sampler.run_pareto_optimization(max_evaluations)
        
        print(f"✅ Optimization completed:")
        print(f"  • Total evaluations: {self.optimization_results['total_evaluations']}")
        print(f"  • Pareto frontier size: {len(self.optimization_results['frontier'].points)}")
        
        # Phase 2: Advanced analysis
        print("\nPhase 2: Advanced Pareto Analysis")
        print("=" * 50)
        
        self.analyzer = ParetoAnalyzer(self.optimization_results['frontier'])
        self.analysis_results = self.analyzer.analyze_frontier_quality()
        
        print("✅ Analysis completed:")
        if 'basic_metrics' in self.analysis_results:
            metrics = self.analysis_results['basic_metrics']
            print(f"  • Hypervolume: {metrics.get('hypervolume', 0):.3f}")
            print(f"  • Diversity: {metrics.get('diversity', 0):.3f}")
            print(f"  • Convergence: {metrics.get('convergence', 0):.3f}")
        
        # Phase 3: Visualization
        print("\nPhase 3: Visualization Generation")
        print("=" * 50)
        
        self.visualizer = ParetoVisualizer(self.optimization_results['frontier'])
        
        if save_results:
            # Create interactive dashboard
            self.visualizer.create_interactive_dashboard(
                f"{results_dir}/pareto_dashboard.html"
            )
            
            # Create static plots
            self.visualizer.plot_convergence_history(
                self.optimization_results['convergence_history'],
                self.optimization_results['diversity_history'],
                self.optimization_results['hypervolume_history'],
                f"{results_dir}/convergence_history.png"
            )
            
            self.visualizer.plot_objective_relationships(
                f"{results_dir}/objective_relationships.png"
            )
            
            self.visualizer.plot_parameter_sensitivity(
                f"{results_dir}/parameter_sensitivity.png"
            )
            
            # Save results
            self._save_results(results_dir)
        
        print("✅ Visualization completed")
        
        # Phase 4: Generate recommendations
        print("\nPhase 4: Recommendations Generation")
        print("=" * 50)
        
        recommendations = self._generate_recommendations()
        
        print("✅ Recommendations generated:")
        for category, config in recommendations.items():
            if 'score' in config:
                print(f"  • {category}: Score = {config['score']:.3f}")
        
        # Compile final results
        complete_results = {
            'optimization_results': self.optimization_results,
            'analysis_results': self.analysis_results,
            'recommendations': recommendations,
            'configuration': {
                'objectives': [asdict(obj) for obj in self.objectives],
                'sampling_config': asdict(self.sampling_config),
                'pareto_config': asdict(self.pareto_config)
            }
        }
        
        if save_results:
            # Save complete results
            with open(f"{results_dir}/complete_results.json", 'w') as f:
                json.dump(self._serialize_results(complete_results), f, indent=2)
        
        return complete_results
    
    def _save_results(self, results_dir: str):
        """Save detailed results to files"""
        timestamp = int(time.time())
        
        # Save Pareto frontier points
        frontier_data = []
        for point in self.optimization_results['frontier'].points:
            point_data = {
                'objective_values': point.objective_values,
                'crowding_distance': point.crowding_distance,
                'parameters': point.test_result.parameters,
                'metrics': {
                    'inference_score': point.test_result.average_inference_score,
                    'training_loss': point.test_result.average_training_loss,
                    'energy_efficiency': point.test_result.energy_efficiency,
                    'memory_usage': point.test_result.memory_usage,
                    'total_time': point.test_result.total_time
                }
            }
            frontier_data.append(point_data)
        
        with open(f"{results_dir}/pareto_frontier_{timestamp}.json", 'w') as f:
            json.dump(frontier_data, f, indent=2)
        
        # Save analysis results
        with open(f"{results_dir}/analysis_results_{timestamp}.json", 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save CSV for easy analysis
        import pandas as pd
        
        df_data = []
        for point in self.optimization_results['frontier'].points:
            row = {'crowding_distance': point.crowding_distance}
            
            # Add objective values
            for i, obj in enumerate(self.objectives):
                row[f"obj_{obj.name}"] = point.objective_values[i]
            
            # Add parameters
            for param, value in point.test_result.parameters.items():
                row[f"param_{param}"] = value
            
            # Add metrics
            row.update({
                'inference_score': point.test_result.average_inference_score,
                'training_loss': point.test_result.average_training_loss,
                'energy_efficiency': point.test_result.energy_efficiency,
                'memory_usage': point.test_result.memory_usage
            })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(f"{results_dir}/pareto_frontier_{timestamp}.csv", index=False)
        
        logger.info(f"Results saved to {results_dir}")
    
    def _generate_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Generate configuration recommendations based on results"""
        points = self.optimization_results['frontier'].points
        
        if not points:
            return {}
        
        recommendations = {}
        
        # Best overall balance (highest crowding distance)
        best_balanced = max(points, key=lambda p: p.crowding_distance)
        recommendations['balanced'] = {
            'description': 'Best overall balance across all objectives',
            'parameters': best_balanced.test_result.parameters,
            'objective_values': best_balanced.objective_values,
            'crowding_distance': best_balanced.crowding_distance,
            'score': np.mean(best_balanced.objective_values)
        }
        
        # Best for each individual objective
        for i, objective in enumerate(self.objectives):
            best_for_obj = max(points, key=lambda p: p.objective_values[i])
            recommendations[f'best_{objective.name}'] = {
                'description': f'Best configuration for {objective.name}',
                'parameters': best_for_obj.test_result.parameters,
                'objective_values': best_for_obj.objective_values,
                'target_objective_value': best_for_obj.objective_values[i],
                'score': best_for_obj.objective_values[i]
            }
        
        # Conservative choice (good performance, low risk)
        if len(points) >= 3:
            # Find point with lowest variance in objective values
            variances = []
            for point in points:
                variance = np.var(point.objective_values)
                variances.append((point, variance))
            
            conservative_point = min(variances, key=lambda x: x[1])[0]
            recommendations['conservative'] = {
                'description': 'Conservative choice with consistent performance',
                'parameters': conservative_point.test_result.parameters,
                'objective_values': conservative_point.objective_values,
                'variance': min(variances, key=lambda x: x[1])[1],
                'score': np.mean(conservative_point.objective_values)
            }
        
        # High-performance choice (best combined score)
        combined_scores = []
        for point in points:
            # Weighted combination of objectives
            weights = [obj.weight for obj in self.objectives]
            weighted_score = sum(val * weight for val, weight in zip(point.objective_values, weights))
            combined_scores.append((point, weighted_score))
        
        high_performance_point = max(combined_scores, key=lambda x: x[1])[0]
        recommendations['high_performance'] = {
            'description': 'Maximum weighted performance across objectives',
            'parameters': high_performance_point.test_result.parameters,
            'objective_values': high_performance_point.objective_values,
            'weighted_score': max(combined_scores, key=lambda x: x[1])[1],
            'score': max(combined_scores, key=lambda x: x[1])[1]
        }
        
        return recommendations
    
    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize results for JSON storage"""
        serialized = {}
        
        for key, value in results.items():
            if key == 'optimization_results':
                serialized[key] = {
                    'total_evaluations': value['total_evaluations'],
                    'frontier_size': len(value['frontier'].points),
                    'convergence_history': value['convergence_history'],
                    'diversity_history': value['diversity_history'],
                    'hypervolume_history': value['hypervolume_history']
                }
            elif isinstance(value, dict):
                serialized[key] = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v 
                                 for k, v in value.items()}
            else:
                serialized[key] = str(value) if not isinstance(value, (int, float, str, bool, list, dict)) else value
        
        return serialized

def create_default_objectives() -> List[ParetoObjective]:
    """Create default multi-objective configuration"""
    objectives = [
        ParetoObjective(
            name="performance",
            metric_name="average_inference_score",
            maximize=True,
            weight=0.4,
            priority=1
        ),
        ParetoObjective(
            name="efficiency",
            metric_name="energy_efficiency",
            maximize=True,
            weight=0.3,
            priority=1
        ),
        ParetoObjective(
            name="speed",
            metric_name="inference_speed",
            maximize=True,
            weight=0.2,
            priority=2
        ),
        ParetoObjective(
            name="memory",
            metric_name="memory_usage",
            maximize=False,
            weight=0.1,
            priority=2,
            constraint_max=1000.0  # Max 1GB memory
        )
    ]
    return objectives

def create_quality_focused_objectives() -> List[ParetoObjective]:
    """Create quality-focused objectives"""
    objectives = [
        ParetoObjective(
            name="inference_quality",
            metric_name="average_inference_score",
            maximize=True,
            weight=0.5,
            priority=1
        ),
        ParetoObjective(
            name="coherence",
            metric_name="response_coherence",
            maximize=True,
            weight=0.3,
            priority=1
        ),
        ParetoObjective(
            name="consistency",
            metric_name="reward_consistency",
            maximize=True,
            weight=0.2,
            priority=2
        )
    ]
    return objectives

def create_efficiency_focused_objectives() -> List[ParetoObjective]:
    """Create efficiency-focused objectives"""
    objectives = [
        ParetoObjective(
            name="energy_efficiency",
            metric_name="energy_efficiency",
            maximize=True,
            weight=0.4,
            priority=1
        ),
        ParetoObjective(
            name="training_speed",
            metric_name="training_speed",
            maximize=True,
            weight=0.3,
            priority=1
        ),
        ParetoObjective(
            name="memory_efficiency",
            metric_name="memory_usage",
            maximize=False,
            weight=0.2,
            priority=1
        ),
        ParetoObjective(
            name="convergence",
            metric_name="convergence_rate",
            maximize=True,
            weight=0.1,
            priority=2
        )
    ]
    return objectives

def run_comparative_study():
    """Run comparative study of different objective configurations"""
    logger.info("🔬 Running Comparative Pareto Study")
    
    objective_configs = {
        'default': create_default_objectives(),
        'quality_focused': create_quality_focused_objectives(),
        'efficiency_focused': create_efficiency_focused_objectives()
    }
    
    results = {}
    
    for config_name, objectives in objective_configs.items():
        print(f"\n🎯 Testing {config_name} configuration")
        print("-" * 50)
        
        system = EnhancedParetoSystem(objectives)
        config_results = system.run_complete_optimization(
            max_evaluations=80,
            save_results=True,
            results_dir=f"./comparative_results_{config_name}"
        )
        
        results[config_name] = {
            'frontier_size': len(config_results['optimization_results']['frontier'].points),
            'hypervolume': config_results['optimization_results']['frontier'].get_hypervolume(),
            'diversity': config_results['optimization_results']['frontier'].get_diversity_metric(),
            'best_recommendations': config_results['recommendations']
        }
        
        print(f"✅ {config_name} completed:")
        print(f"  • Frontier size: {results[config_name]['frontier_size']}")
        print(f"  • Hypervolume: {results[config_name]['hypervolume']:.3f}")
        print(f"  • Diversity: {results[config_name]['diversity']:.3f}")
    
    # Compare results
    print("\n📊 Comparative Analysis")
    print("=" * 50)
    
    for metric in ['frontier_size', 'hypervolume', 'diversity']:
        print(f"\n{metric.title()}:")
        sorted_configs = sorted(objective_configs.keys(), 
                               key=lambda k: results[k][metric], reverse=True)
        for i, config in enumerate(sorted_configs):
            print(f"  {i+1}. {config}: {results[config][metric]:.3f}")
    
    return results

def run_sensitivity_analysis():
    """Run sensitivity analysis on sampling parameters"""
    logger.info("🧪 Running Sensitivity Analysis")
    
    base_objectives = create_default_objectives()
    base_config = SamplingConfig()
    
    # Test different sampling configurations
    configs_to_test = [
        ("default", SamplingConfig()),
        ("high_exploration", SamplingConfig(exploration_rate=0.3, max_iterations=150)),
        ("fast_convergence", SamplingConfig(convergence_threshold=0.005, adaptation_frequency=15)),
        ("diverse_search", SamplingConfig(tree_depth=7, branching_factor=4, diversity_bonus=0.4))
    ]
    
    results = {}
    
    for config_name, sampling_config in configs_to_test:
        print(f"\n🔧 Testing {config_name} sampling")
        
        system = EnhancedParetoSystem(base_objectives, sampling_config)
        config_results = system.run_complete_optimization(
            max_evaluations=60,
            save_results=False
        )
        
        frontier = config_results['optimization_results']['frontier']
        results[config_name] = {
            'frontier_size': len(frontier.points),
            'hypervolume': frontier.get_hypervolume(),
            'diversity': frontier.get_diversity_metric(),
            'convergence': frontier.get_convergence_metric(),
            'total_evaluations': config_results['optimization_results']['total_evaluations']
        }
        
        print(f"  • Frontier size: {results[config_name]['frontier_size']}")
        print(f"  • Hypervolume: {results[config_name]['hypervolume']:.3f}")
    
    print("\n📈 Sensitivity Analysis Results")
    print("=" * 50)
    
    for metric in ['hypervolume', 'diversity', 'frontier_size']:
        print(f"\nBest {metric}:")
        best_config = max(results.keys(), key=lambda k: results[k][metric])
        print(f"  {best_config}: {results[best_config][metric]:.3f}")
    
    return results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Enhanced Pareto Frontier Optimization')
    parser.add_argument('--mode', choices=['default', 'quality', 'efficiency', 'comparative', 'sensitivity'],
                       default='default', help='Optimization mode')
    parser.add_argument('--evaluations', type=int, default=150, help='Number of evaluations')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results')
    parser.add_argument('--results-dir', default='./enhanced_pareto_results', help='Results directory')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🎯 Enhanced Pareto Frontier Optimization System")
    print("=" * 60)
    
    if args.mode == 'comparative':
        results = run_comparative_study()
        print("\n✅ Comparative study completed!")
        
    elif args.mode == 'sensitivity':
        results = run_sensitivity_analysis()
        print("\n✅ Sensitivity analysis completed!")
        
    else:
        # Single configuration run
        if args.mode == 'quality':
            objectives = create_quality_focused_objectives()
            print("🎨 Running Quality-Focused Optimization")
        elif args.mode == 'efficiency':
            objectives = create_efficiency_focused_objectives()
            print("⚡ Running Efficiency-Focused Optimization")
        else:
            objectives = create_default_objectives()
            print("🚀 Running Default Multi-Objective Optimization")
        
        system = EnhancedParetoSystem(objectives)
        results = system.run_complete_optimization(
            max_evaluations=args.evaluations,
            save_results=args.save_results,
            results_dir=args.results_dir
        )
        
        print("\n🎉 Optimization completed successfully!")
        print(f"📁 Results saved to: {args.results_dir}")
        
        # Print summary
        frontier = results['optimization_results']['frontier']
        print(f"\n📊 Summary:")
        print(f"  • Pareto frontier size: {len(frontier.points)}")
        print(f"  • Hypervolume: {frontier.get_hypervolume():.3f}")
        print(f"  • Diversity: {frontier.get_diversity_metric():.3f}")
        print(f"  • Total evaluations: {results['optimization_results']['total_evaluations']}")
        
        # Print best recommendations
        print(f"\n🏆 Top Recommendations:")
        for category, config in results['recommendations'].items():
            if 'score' in config:
                print(f"  • {category}: Score = {config['score']:.3f}")

if __name__ == "__main__":
    main()