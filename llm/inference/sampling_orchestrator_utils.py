# sampling_orchestrator_utils.py
"""
Adaptive Sampling Orchestrator and Utility Functions
Coordinates multiple sampling methods and provides utility functions
"""
import numpy as np
import time
import threading
import queue
import random
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

from enhanced_sampling_methods import ParameterSpace, SamplingConfig
from sampling_serial_backtrack import SerialSampler, BacktrackingSampler
from sampling_bayesian_genetic import BayesianOptimizationSampler, GeneticAlgorithmSampler
from parameter_testing_and_pareto import TestResult

logger = logging.getLogger(__name__)

class AdaptiveSamplingOrchestrator:
    """Orchestrates multiple sampling methods adaptively"""
    
    def __init__(self, sampling_config: SamplingConfig = None):
        self.config = sampling_config or SamplingConfig()
        self.parameter_space = ParameterSpace()
        self.active_samplers = {}
        self.sampler_performance = {}
        self.global_history = []
        self.best_global_result = None
        self.running = False
        
    def run_adaptive_sampling(self, max_evaluations: int = 200) -> Dict[str, Any]:
        """Run adaptive sampling using multiple methods"""
        logger.info("🧠 Starting Adaptive Multi-Method Sampling")
        
        # Ensure data files exist
        from file_based_training_inference import create_sample_data_files
        from pathlib import Path
        if not Path("./training_data").exists():
            create_sample_data_files()
        
        # Initialize all samplers
        self._initialize_samplers()
        
        # Adaptive sampling loop
        evaluation_count = 0
        adaptation_interval = 20
        
        try:
            while evaluation_count < max_evaluations:
                # Select sampling method
                sampler_name = self._select_sampler()
                sampler = self.active_samplers[sampler_name]
                
                # Sample and evaluate
                params = sampler.sample_next()
                result = sampler.evaluate_params(params)
                
                # Update sampler and global state
                sampler.update(params, result)
                self._update_global_state(sampler_name, result)
                
                evaluation_count += 1
                
                # Adapt sampler weights
                if evaluation_count % adaptation_interval == 0:
                    self._adapt_sampler_weights()
                    logger.info(f"Adaptation at evaluation {evaluation_count}")
                    self._log_performance_summary()
                
                if evaluation_count % 50 == 0:
                    logger.info(f"Progress: {evaluation_count}/{max_evaluations} evaluations")
        
        except KeyboardInterrupt:
            logger.info(f"Sampling interrupted at evaluation {evaluation_count}")
        
        # Final analysis
        final_analysis = self._generate_final_analysis()
        
        return {
            'global_history': self.global_history,
            'best_result': self.best_global_result,
            'sampler_performance': self.sampler_performance,
            'final_analysis': final_analysis,
            'total_evaluations': evaluation_count
        }
    
    def _initialize_samplers(self):
        """Initialize all sampling methods"""
        self.active_samplers = {
            'serial': SerialSampler(self.parameter_space, self.config),
            'backtracking': BacktrackingSampler(self.parameter_space, self.config),
            'bayesian': BayesianOptimizationSampler(self.parameter_space, self.config),
            'genetic': GeneticAlgorithmSampler(self.parameter_space, self.config)
        }
        
        # Initialize performance tracking
        for name in self.active_samplers.keys():
            self.sampler_performance[name] = {
                'weight': 1.0 / len(self.active_samplers),
                'recent_scores': deque(maxlen=10),
                'total_evaluations': 0,
                'successful_evaluations': 0,
                'best_score': 0.0,
                'convergence_rate': 0.0,
                'diversity_score': 0.0
            }
        
        logger.info(f"Initialized {len(self.active_samplers)} adaptive samplers")
    
    def _select_sampler(self) -> str:
        """Select sampling method based on performance weights"""
        # Softmax selection based on weights
        names = list(self.sampler_performance.keys())
        weights = [self.sampler_performance[name]['weight'] for name in names]
        
        # Add exploration bonus for underused samplers
        total_evals = sum(self.sampler_performance[name]['total_evaluations'] for name in names)
        if total_evals > 0:
            for i, name in enumerate(names):
                usage_ratio = self.sampler_performance[name]['total_evaluations'] / total_evals
                if usage_ratio < 1.0 / len(names):  # Underused
                    weights[i] *= 1.2  # Boost weight
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Select based on weights
        return np.random.choice(names, p=weights)
    
    def _update_global_state(self, sampler_name: str, result: TestResult):
        """Update global state with new result"""
        score = self._result_to_score(result)
        
        # Update global history
        self.global_history.append((sampler_name, result))
        
        # Update best result
        if self.best_global_result is None or score > self._result_to_score(self.best_global_result):
            self.best_global_result = result
            logger.info(f"New best result from {sampler_name}: score={score:.3f}")
        
        # Update sampler performance
        perf = self.sampler_performance[sampler_name]
        perf['total_evaluations'] += 1
        perf['recent_scores'].append(score)
        if score > 0.1:  # Threshold for "successful"
            perf['successful_evaluations'] += 1
        if score > perf['best_score']:
            perf['best_score'] = score
    
    def _adapt_sampler_weights(self):
        """Adapt sampler weights based on recent performance"""
        for name, perf in self.sampler_performance.items():
            if len(perf['recent_scores']) > 0:
                # Calculate recent performance metrics
                recent_avg = np.mean(perf['recent_scores'])
                recent_improvement = (
                    perf['recent_scores'][-1] - perf['recent_scores'][0] 
                    if len(perf['recent_scores']) > 1 else 0
                )
                success_rate = (
                    perf['successful_evaluations'] / max(perf['total_evaluations'], 1)
                )
                
                # Calculate convergence rate
                if len(perf['recent_scores']) >= 3:
                    scores = list(perf['recent_scores'])
                    convergence = max(0, (scores[-1] - scores[0]) / len(scores))
                    perf['convergence_rate'] = convergence
                
                # Calculate diversity (how varied the recent scores are)
                if len(perf['recent_scores']) >= 2:
                    diversity = np.std(perf['recent_scores'])
                    perf['diversity_score'] = diversity
                
                # Update weight based on multiple factors
                performance_score = (
                    0.4 * recent_avg +
                    0.2 * max(0, recent_improvement) +
                    0.2 * success_rate +
                    0.1 * perf['convergence_rate'] +
                    0.1 * min(perf['diversity_score'], 0.5)  # Cap diversity bonus
                )
                
                # Exponential moving average for weight updates
                alpha = 0.3
                perf['weight'] = alpha * performance_score + (1 - alpha) * perf['weight']
        
        # Normalize weights
        total_weight = sum(perf['weight'] for perf in self.sampler_performance.values())
        if total_weight > 0:
            for perf in self.sampler_performance.values():
                perf['weight'] /= total_weight
    
    def _log_performance_summary(self):
        """Log current performance summary"""
        logger.info("📊 Sampler Performance Summary:")
        for name, perf in self.sampler_performance.items():
            recent_avg = np.mean(perf['recent_scores']) if perf['recent_scores'] else 0
            logger.info(f"  {name}: weight={perf['weight']:.3f}, "
                       f"recent_avg={recent_avg:.3f}, "
                       f"evals={perf['total_evaluations']}, "
                       f"success_rate={perf['successful_evaluations']/max(perf['total_evaluations'],1):.2f}")
    
    def _generate_final_analysis(self) -> Dict[str, Any]:
        """Generate final analysis of adaptive sampling"""
        analysis = {
            'sampler_statistics': {},
            'convergence_analysis': {},
            'best_configurations': {},
            'recommendations': {}
        }
        
        # Sampler statistics
        for name, perf in self.sampler_performance.items():
            analysis['sampler_statistics'][name] = {
                'total_evaluations': perf['total_evaluations'],
                'success_rate': perf['successful_evaluations'] / max(perf['total_evaluations'], 1),
                'best_score': perf['best_score'],
                'final_weight': perf['weight'],
                'convergence_rate': perf['convergence_rate'],
                'diversity_score': perf['diversity_score']
            }
        
        # Convergence analysis
        if len(self.global_history) > 20:
            scores = [self._result_to_score(result) for _, result in self.global_history]
            analysis['convergence_analysis'] = {
                'initial_avg': np.mean(scores[:10]),
                'final_avg': np.mean(scores[-10:]),
                'best_score': max(scores),
                'convergence_rate': self._calculate_convergence_rate(scores),
                'score_variance': np.var(scores)
            }
        
        # Best configurations by sampler
        sampler_results = defaultdict(list)
        for sampler_name, result in self.global_history:
            sampler_results[sampler_name].append(result)
        
        for name, results in sampler_results.items():
            if results:
                best_result = max(results, key=self._result_to_score)
                analysis['best_configurations'][name] = {
                    'parameters': best_result.parameters,
                    'score': self._result_to_score(best_result)
                }
        
        # Recommendations
        best_sampler = max(
            self.sampler_performance.items(),
            key=lambda x: x[1]['best_score']
        )[0]
        
        most_efficient = max(
            self.sampler_performance.items(),
            key=lambda x: x[1]['successful_evaluations'] / max(x[1]['total_evaluations'], 1)
        )[0]
        
        analysis['recommendations'] = {
            'best_performing_sampler': best_sampler,
            'most_efficient_sampler': most_efficient,
            'suggested_ensemble': [
                name for name, perf in self.sampler_performance.items()
                if perf['final_weight'] > 0.15
            ],
            'sampling_strategy': self._recommend_strategy()
        }
        
        return analysis
    
    def _recommend_strategy(self) -> str:
        """Recommend optimal sampling strategy based on results"""
        # Analyze which samplers performed best
        performances = [(name, perf['best_score']) for name, perf in self.sampler_performance.items()]
        performances.sort(key=lambda x: x[1], reverse=True)
        
        best_sampler = performances[0][0]
        
        if best_sampler == 'bayesian':
            return "Use Bayesian optimization for efficient exploration with GP modeling"
        elif best_sampler == 'genetic':
            return "Use genetic algorithm for population-based global optimization"
        elif best_sampler == 'backtracking':
            return "Use backtracking for systematic exploration with recovery"
        elif best_sampler == 'serial':
            return "Use serial adaptive sampling for balanced exploration/exploitation"
        else:
            return "Use ensemble approach combining multiple methods"
    
    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """Calculate convergence rate"""
        if len(scores) < 10:
            return 0.0
        
        early_scores = scores[:len(scores)//3]
        late_scores = scores[-len(scores)//3:]
        
        early_avg = np.mean(early_scores)
        late_avg = np.mean(late_scores)
        
        return max(0, (late_avg - early_avg) / max(early_avg, 1e-10))
    
    def _result_to_score(self, result: TestResult) -> float:
        """Convert result to score"""
        return (
            0.4 * result.average_inference_score +
            0.3 * result.energy_efficiency +
            0.2 * (1.0 / (1.0 + result.average_training_loss)) +
            0.1 * result.response_coherence
        )

class EnhancedParameterTester:
    """Enhanced parameter tester with multiple sampling strategies"""
    
    def __init__(self, sampling_config: SamplingConfig = None):
        self.config = sampling_config or SamplingConfig()
        self.parameter_space = ParameterSpace()
        self.results = defaultdict(list)
        
    def run_sampling_comparison(self, max_evaluations: int = 100) -> Dict[str, Any]:
        """Compare different sampling methods"""
        logger.info("🔬 Starting Sampling Method Comparison")
        
        # Ensure data files exist
        from file_based_training_inference import create_sample_data_files
        from pathlib import Path
        if not Path("./training_data").exists():
            create_sample_data_files()
        
        # Initialize samplers
        samplers = {
            'serial': SerialSampler(self.parameter_space, self.config),
            'backtracking': BacktrackingSampler(self.parameter_space, self.config),
            'bayesian': BayesianOptimizationSampler(self.parameter_space, self.config),
            'genetic': GeneticAlgorithmSampler(self.parameter_space, self.config)
        }
        
        # Run each sampler
        for method_name, sampler in samplers.items():
            logger.info(f"🎯 Testing {method_name} sampling")
            
            evaluations_per_method = max_evaluations // len(samplers)
            
            for i in range(evaluations_per_method):
                # Sample next configuration
                params = sampler.sample_next()
                
                # Evaluate
                result = sampler.evaluate_params(params)
                
                # Update sampler
                sampler.update(params, result)
                
                # Store result
                self.results[method_name].append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  {method_name}: {i+1}/{evaluations_per_method} evaluations")
            
            logger.info(f"✅ {method_name} sampling completed")
        
        # Analyze results
        analysis = self._analyze_sampling_results()
        
        return {
            'results': dict(self.results),
            'analysis': analysis,
            'samplers': samplers
        }
    
    def _analyze_sampling_results(self) -> Dict[str, Any]:
        """Analyze and compare sampling results"""
        analysis = {}
        
        for method, results in self.results.items():
            if not results:
                continue
            
            scores = [self._result_to_score(r) for r in results]
            
            analysis[method] = {
                'best_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'convergence_rate': self._calculate_convergence_rate(scores),
                'exploration_diversity': self._calculate_diversity(results),
                'efficiency': len(results) / max(1, len([r for r in results if self._result_to_score(r) > 0.5])),
                'best_configuration': max(results, key=self._result_to_score).parameters
            }
        
        return analysis
    
    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """Calculate convergence rate"""
        if len(scores) < 10:
            return 0.0
        
        # Calculate improvement over time
        early_scores = scores[:len(scores)//3]
        late_scores = scores[-len(scores)//3:]
        
        early_avg = np.mean(early_scores)
        late_avg = np.mean(late_scores)
        
        return max(0, (late_avg - early_avg) / max(early_avg, 1e-10))
    
    def _calculate_diversity(self, results: List[TestResult]) -> float:
        """Calculate exploration diversity"""
        if len(results) < 2:
            return 0.0
        
        # Calculate pairwise distances between configurations
        distances = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                dist = self.parameter_space.distance(
                    results[i].parameters, 
                    results[j].parameters
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _result_to_score(self, result: TestResult) -> float:
        """Convert result to score"""
        return (
            0.4 * result.average_inference_score +
            0.3 * result.energy_efficiency +
            0.2 * (1.0 / (1.0 + result.average_training_loss)) +
            0.1 * result.response_coherence
        )

# Utility functions for creating demo scenarios
def create_demo_scenarios():
    """Create demonstration scenarios for sampling methods"""
    scenarios = {
        'quick_test': {
            'config': SamplingConfig(
                max_iterations=50,
                exploration_rate=0.2,
                n_initial_points=5
            ),
            'description': 'Quick test with limited evaluations'
        },
        'thorough_exploration': {
            'config': SamplingConfig(
                max_iterations=150,
                exploration_rate=0.3,
                tree_depth=7,
                population_size=30
            ),
            'description': 'Thorough exploration with high diversity'
        },
        'fast_convergence': {
            'config': SamplingConfig(
                max_iterations=100,
                convergence_threshold=0.005,
                exploitation_rate=0.8,
                elite_ratio=0.3
            ),
            'description': 'Fast convergence to good solutions'
        },
        'robust_search': {
            'config': SamplingConfig(
                max_iterations=200,
                max_backtrack_depth=15,
                memory_size=2000,
                diversity_bonus=0.4
            ),
            'description': 'Robust search with backtracking and memory'
        }
    }
    
    return scenarios

def run_demo_comparison():
    """Run demonstration comparison of sampling methods"""
    print("🎭 Enhanced Sampling Methods Demo")
    print("=" * 50)
    
    scenarios = create_demo_scenarios()
    results = {}
    
    for scenario_name, scenario_info in scenarios.items():
        print(f"\n🔬 Testing scenario: {scenario_name}")
        print(f"Description: {scenario_info['description']}")
        
        # Run orchestrator with scenario config
        orchestrator = AdaptiveSamplingOrchestrator(scenario_info['config'])
        scenario_results = orchestrator.run_adaptive_sampling(max_evaluations=40)
        
        results[scenario_name] = {
            'best_score': scenario_results['final_analysis']['convergence_analysis'].get('best_score', 0),
            'total_evaluations': scenario_results['total_evaluations'],
            'best_sampler': scenario_results['final_analysis']['recommendations']['best_performing_sampler'],
            'ensemble': scenario_results['final_analysis']['recommendations']['suggested_ensemble']
        }
        
        print(f"✅ Best score: {results[scenario_name]['best_score']:.3f}")
        print(f"   Best sampler: {results[scenario_name]['best_sampler']}")
    
    # Summary comparison
    print(f"\n📊 Scenario Comparison Summary:")
    print("-" * 40)
    
    for scenario, result in results.items():
        print(f"{scenario:20s}: {result['best_score']:.3f} ({result['best_sampler']})")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Sampling Orchestrator')
    parser.add_argument('--mode', choices=['demo', 'comparison', 'adaptive'], 
                       default='demo', help='Execution mode')
    parser.add_argument('--evaluations', type=int, default=100, help='Number of evaluations')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.mode == 'demo':
        results = run_demo_comparison()
    elif args.mode == 'comparison':
        tester = EnhancedParameterTester()
        results = tester.run_sampling_comparison(args.evaluations)
        print(f"Comparison completed with {len(results['results'])} methods")
    elif args.mode == 'adaptive':
        orchestrator = AdaptiveSamplingOrchestrator()
        results = orchestrator.run_adaptive_sampling(args.evaluations)
        print(f"Adaptive sampling completed: {results['total_evaluations']} evaluations")
    
    print("✅ Enhanced sampling system demo completed!")