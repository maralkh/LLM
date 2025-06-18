# complete_integration_demo.py
"""
Complete Integration Demo and Usage Examples
Shows how to use the enhanced sampling system with the original file-based training
"""
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import all components
from file_based_training_inference import (
    FileBasedTrainingSystem, FileBasedConfig, 
    create_dummy_models, create_sample_data_files
)
from enhanced_sampling_methods import SamplingConfig, ParameterSpace
from sampling_orchestrator_utils import (
    AdaptiveSamplingOrchestrator, EnhancedParameterTester
)
from enhanced_pareto_main import (
    EnhancedParetoSystem, create_default_objectives
)

logger = logging.getLogger(__name__)

class CompleteOptimizationSystem:
    """Complete system integrating all optimization approaches"""
    
    def __init__(self):
        self.parameter_space = ParameterSpace()
        self.optimization_results = {}
        self.best_configurations = {}
        
    def run_complete_optimization_pipeline(self, max_evaluations: int = 150) -> Dict[str, Any]:
        """Run complete optimization pipeline with all methods"""
        print("🚀 Complete Enhanced Optimization Pipeline")
        print("=" * 60)
        
        # Ensure data files exist
        if not Path("./training_data").exists():
            print("📁 Creating sample training data...")
            create_sample_data_files()
        
        # Phase 1: Enhanced Sampling Comparison
        print("\n📊 Phase 1: Enhanced Sampling Method Comparison")
        print("-" * 50)
        
        sampling_results = self._run_sampling_comparison(max_evaluations // 3)
        
        # Phase 2: Adaptive Multi-Method Optimization
        print("\n🧠 Phase 2: Adaptive Multi-Method Optimization")
        print("-" * 50)
        
        adaptive_results = self._run_adaptive_optimization(max_evaluations // 3)
        
        # Phase 3: Pareto Frontier Optimization
        print("\n🎯 Phase 3: Multi-Objective Pareto Optimization")
        print("-" * 50)
        
        pareto_results = self._run_pareto_optimization(max_evaluations // 3)
        
        # Phase 4: Integration and Validation
        print("\n✅ Phase 4: Integration and Validation")
        print("-" * 50)
        
        validation_results = self._validate_best_configurations()
        
        # Compile final results
        complete_results = {
            'sampling_comparison': sampling_results,
            'adaptive_optimization': adaptive_results,
            'pareto_optimization': pareto_results,
            'validation_results': validation_results,
            'best_overall_config': self._select_best_overall_config(),
            'recommendations': self._generate_final_recommendations()
        }
        
        # Save results
        self._save_complete_results(complete_results)
        
        return complete_results
    
    def _run_sampling_comparison(self, max_evals: int) -> Dict[str, Any]:
        """Run sampling method comparison"""
        print("🔬 Comparing sampling methods...")
        
        # Custom sampling config for comparison
        config = SamplingConfig(
            max_iterations=max_evals,
            exploration_rate=0.15,
            tree_depth=5,
            population_size=20,
            n_initial_points=8
        )
        
        tester = EnhancedParameterTester(config)
        results = tester.run_sampling_comparison(max_evals)
        
        # Extract best configuration from each method
        for method, analysis in results['analysis'].items():
            if 'best_configuration' in analysis:
                self.best_configurations[f'sampling_{method}'] = {
                    'parameters': analysis['best_configuration'],
                    'score': analysis['best_score'],
                    'method': f'Enhanced Sampling - {method.title()}'
                }
        
        print(f"✅ Sampling comparison completed")
        print(f"   Best method: {max(results['analysis'].items(), key=lambda x: x[1]['best_score'])[0]}")
        
        return results
    
    def _run_adaptive_optimization(self, max_evals: int) -> Dict[str, Any]:
        """Run adaptive multi-method optimization"""
        print("🧠 Running adaptive optimization...")
        
        config = SamplingConfig(
            max_iterations=max_evals,
            adaptation_frequency=15,
            exploration_rate=0.2,
            diversity_bonus=0.3
        )
        
        orchestrator = AdaptiveSamplingOrchestrator(config)
        results = orchestrator.run_adaptive_sampling(max_evals)
        
        # Store best configuration
        if results['best_result']:
            score = (
                0.4 * results['best_result'].average_inference_score +
                0.3 * results['best_result'].energy_efficiency +
                0.2 * (1.0 / (1.0 + results['best_result'].average_training_loss)) +
                0.1 * results['best_result'].response_coherence
            )
            
            self.best_configurations['adaptive_multi_method'] = {
                'parameters': results['best_result'].parameters,
                'score': score,
                'method': 'Adaptive Multi-Method'
            }
        
        print(f"✅ Adaptive optimization completed")
        if 'final_analysis' in results and 'recommendations' in results['final_analysis']:
            recs = results['final_analysis']['recommendations']
            print(f"   Best sampler: {recs.get('best_performing_sampler', 'Unknown')}")
            print(f"   Suggested ensemble: {', '.join(recs.get('suggested_ensemble', []))}")
        
        return results
    
    def _run_pareto_optimization(self, max_evals: int) -> Dict[str, Any]:
        """Run Pareto frontier optimization"""
        print("🎯 Running Pareto optimization...")
        
        # Create objectives for comprehensive optimization
        objectives = create_default_objectives()
        
        pareto_system = EnhancedParetoSystem(objectives)
        results = pareto_system.run_complete_optimization(
            max_evaluations=max_evals,
            save_results=False
        )
        
        # Store best configurations from Pareto frontier
        if 'recommendations' in results:
            for category, config in results['recommendations'].items():
                if 'parameters' in config and 'score' in config:
                    self.best_configurations[f'pareto_{category}'] = {
                        'parameters': config['parameters'],
                        'score': config['score'],
                        'method': f'Pareto Optimization - {category.title()}'
                    }
        
        frontier_size = len(results['optimization_results']['frontier'].points)
        print(f"✅ Pareto optimization completed")
        print(f"   Frontier size: {frontier_size}")
        print(f"   Hypervolume: {results['optimization_results']['frontier'].get_hypervolume():.3f}")
        
        return results
    
    def _validate_best_configurations(self) -> Dict[str, Any]:
        """Validate best configurations with actual training"""
        print("🔍 Validating best configurations...")
        
        validation_results = {}
        
        # Select top 3 configurations for validation
        sorted_configs = sorted(
            self.best_configurations.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:3]
        
        for config_name, config_data in sorted_configs:
            print(f"   Testing {config_name}...")
            
            try:
                # Create models and system
                base_model, prm_model, orm_model, tokenizer = create_dummy_models()
                
                # Convert parameters to FileBasedConfig
                file_config = self._params_to_file_config(config_data['parameters'])
                
                # Create and run system
                system = FileBasedTrainingSystem(
                    base_model, prm_model, orm_model, tokenizer, file_config
                )
                
                start_time = time.time()
                system.start()
                
                # Run for validation period
                time.sleep(15)  # Quick validation
                
                # Get final status
                status = system.get_system_status()
                end_time = time.time()
                
                system.stop()
                
                # Calculate validation score
                validation_score = self._calculate_validation_score(status, end_time - start_time)
                
                validation_results[config_name] = {
                    'original_score': config_data['score'],
                    'validation_score': validation_score,
                    'training_batches': status['stats']['total_training_batches'],
                    'inferences': status['stats']['total_inferences'],
                    'method': config_data['method'],
                    'validation_time': end_time - start_time
                }
                
                print(f"     Original: {config_data['score']:.3f}, Validated: {validation_score:.3f}")
                
            except Exception as e:
                logger.error(f"Validation failed for {config_name}: {e}")
                validation_results[config_name] = {
                    'error': str(e),
                    'validation_score': 0.0
                }
        
        return validation_results
    
    def _params_to_file_config(self, params: Dict[str, Any]) -> FileBasedConfig:
        """Convert parameters to FileBasedConfig"""
        config = FileBasedConfig()
        
        # Map parameters
        mapping = {
            'learning_rate': 'learning_rate',
            'batch_size': 'batch_size',
            'temperature': 'temperature',
            'top_p': 'top_p',
            'num_candidates': 'num_candidates',
            'prm_weight': 'prm_weight',
            'orm_weight': 'orm_weight',
            'max_sequence_length': 'max_sequence_length',
            'training_epochs_per_batch': 'training_epochs_per_batch',
            'data_refresh_interval': 'data_refresh_interval'
        }
        
        for param_name, config_attr in mapping.items():
            if param_name in params:
                setattr(config, config_attr, params[param_name])
        
        # Set optimal settings for validation
        config.auto_training = True
        config.save_intermediate_results = False
        config.min_data_size_for_training = 5
        
        return config
    
    def _calculate_validation_score(self, status: Dict[str, Any], runtime: float) -> float:
        """Calculate validation score from system status"""
        stats = status.get('stats', {})
        
        # Normalize metrics
        training_rate = stats.get('total_training_batches', 0) / max(runtime, 1)
        inference_rate = stats.get('total_inferences', 0) / max(runtime, 1)
        
        # Calculate composite score
        score = (
            0.4 * min(training_rate / 5.0, 1.0) +  # Normalize to reasonable training rate
            0.4 * min(inference_rate / 10.0, 1.0) + # Normalize to reasonable inference rate
            0.2 * (1.0 if stats.get('total_training_batches', 0) > 0 else 0.0)  # Basic functionality
        )
        
        return score
    
    def _select_best_overall_config(self) -> Dict[str, Any]:
        """Select best overall configuration"""
        if not self.best_configurations:
            return {}
        
        # Find configuration with highest score
        best_config_name, best_config_data = max(
            self.best_configurations.items(),
            key=lambda x: x[1]['score']
        )
        
        return {
            'name': best_config_name,
            'parameters': best_config_data['parameters'],
            'score': best_config_data['score'],
            'method': best_config_data['method']
        }
    
    def _generate_final_recommendations(self) -> Dict[str, Any]:
        """Generate final recommendations"""
        recommendations = {
            'best_overall': self._select_best_overall_config(),
            'use_case_recommendations': {},
            'optimization_insights': [],
            'next_steps': []
        }
        
        # Use case specific recommendations
        if self.best_configurations:
            # High performance use case
            high_perf_configs = {k: v for k, v in self.best_configurations.items() 
                               if 'pareto_high_performance' in k or 'adaptive' in k}
            if high_perf_configs:
                best_high_perf = max(high_perf_configs.items(), key=lambda x: x[1]['score'])
                recommendations['use_case_recommendations']['high_performance'] = {
                    'config': best_high_perf[1]['parameters'],
                    'description': 'Best for maximum performance regardless of resources'
                }
            
            # Balanced use case
            balanced_configs = {k: v for k, v in self.best_configurations.items() 
                              if 'pareto_balanced' in k or 'sampling' in k}
            if balanced_configs:
                best_balanced = max(balanced_configs.items(), key=lambda x: x[1]['score'])
                recommendations['use_case_recommendations']['balanced'] = {
                    'config': best_balanced[1]['parameters'],
                    'description': 'Good balance of performance, speed, and resources'
                }
        
        # Optimization insights
        if len(self.best_configurations) > 1:
            scores = [config['score'] for config in self.best_configurations.values()]
            recommendations['optimization_insights'] = [
                f"Best score achieved: {max(scores):.3f}",
                f"Score range: {min(scores):.3f} to {max(scores):.3f}",
                f"Total configurations tested: {len(self.best_configurations)}",
                f"Most effective method: {self._select_best_overall_config().get('method', 'Unknown')}"
            ]
        
        # Next steps
        recommendations['next_steps'] = [
            "Apply best configuration to production training",
            "Monitor performance metrics in real environment",
            "Consider fine-tuning based on specific data characteristics",
            "Set up automated reoptimization schedule"
        ]
        
        return recommendations
    
    def _save_complete_results(self, results: Dict[str, Any]):
        """Save complete optimization results"""
        timestamp = int(time.time())
        results_dir = Path(f"./complete_optimization_results_{timestamp}")
        results_dir.mkdir(exist_ok=True)
        
        # Save main results
        with open(results_dir / "complete_results.json", 'w') as f:
            json.dump(self._serialize_for_json(results), f, indent=2)
        
        # Save best configurations
        with open(results_dir / "best_configurations.json", 'w') as f:
            json.dump(self.best_configurations, f, indent=2)
        
        # Save recommendations
        if 'recommendations' in results:
            with open(results_dir / "recommendations.json", 'w') as f:
                json.dump(results['recommendations'], f, indent=2)
        
        print(f"📁 Complete results saved to: {results_dir}")
    
    def _serialize_for_json(self, obj):
        """Serialize object for JSON storage"""
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

def run_quick_demo():
    """Run quick demonstration of the complete system"""
    print("🎭 Quick Demo: Complete Enhanced Optimization System")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create system
    system = CompleteOptimizationSystem()
    
    # Run with limited evaluations for demo
    results = system.run_complete_optimization_pipeline(max_evaluations=60)
    
    # Print summary
    print(f"\n🎉 Quick Demo Summary")
    print("=" * 40)
    
    best_config = results.get('best_overall_config', {})
    if best_config:
        print(f"🏆 Best Configuration:")
        print(f"   Method: {best_config.get('method', 'Unknown')}")
        print(f"   Score: {best_config.get('score', 0):.3f}")
        
        params = best_config.get('parameters', {})
        print(f"   Key Parameters:")
        print(f"     Learning Rate: {params.get('learning_rate', 'N/A'):.2e}")
        print(f"     Temperature: {params.get('temperature', 'N/A'):.3f}")
        print(f"     Batch Size: {params.get('batch_size', 'N/A')}")
    
    # Print recommendations
    recommendations = results.get('recommendations', {})
    if 'optimization_insights' in recommendations:
        print(f"\n💡 Key Insights:")
        for insight in recommendations['optimization_insights']:
            print(f"   • {insight}")
    
    if 'next_steps' in recommendations:
        print(f"\n📋 Next Steps:")
        for step in recommendations['next_steps']:
            print(f"   • {step}")
    
    return results

def run_production_optimization():
    """Run production-ready optimization"""
    print("🏭 Production Enhanced Optimization System")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create system
    system = CompleteOptimizationSystem()
    
    # Run full optimization
    results = system.run_complete_optimization_pipeline(max_evaluations=200)
    
    # Detailed analysis
    print(f"\n📊 Production Optimization Results")
    print("=" * 50)
    
    # Best configurations summary
    if system.best_configurations:
        print(f"🏆 Top Configurations Found:")
        sorted_configs = sorted(
            system.best_configurations.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        for i, (name, config) in enumerate(sorted_configs[:5], 1):
            print(f"   {i}. {name}: {config['score']:.3f} ({config['method']})")
    
    # Validation results
    validation = results.get('validation_results', {})
    if validation:
        print(f"\n✅ Validation Results:")
        for config_name, val_data in validation.items():
            if 'validation_score' in val_data:
                print(f"   {config_name}: {val_data['validation_score']:.3f}")
    
    # Use case recommendations
    use_cases = results.get('recommendations', {}).get('use_case_recommendations', {})
    if use_cases:
        print(f"\n🎯 Use Case Recommendations:")
        for use_case, rec in use_cases.items():
            print(f"   {use_case.title()}: {rec['description']}")
    
    return results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Enhanced Optimization System')
    parser.add_argument('--mode', choices=['demo', 'production', 'custom'], 
                       default='demo', help='Execution mode')
    parser.add_argument('--evaluations', type=int, default=150, 
                       help='Total evaluations across all phases')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        results = run_quick_demo()
    elif args.mode == 'production':
        results = run_production_optimization()
    elif args.mode == 'custom':
        system = CompleteOptimizationSystem()
        results = system.run_complete_optimization_pipeline(args.evaluations)
        print(f"✅ Custom optimization completed with {args.evaluations} evaluations")
    
    print(f"\n🎊 Optimization completed successfully!")

if __name__ == "__main__":
    main()