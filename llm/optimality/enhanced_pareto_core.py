# enhanced_pareto_core.py
"""
Enhanced Pareto Frontier Analysis with Advanced Sampling Methods
Core system for multi-objective optimization with intelligent sampling
"""
import torch
import numpy as np
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import copy
import itertools
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import pandas as pd

# Import enhanced sampling components
from enhanced_sampling_methods import (
    ParameterSpace, SamplingConfig, 
    TreeBasedSampler, SerialSampler, BacktrackingSampler,
    BayesianOptimizationSampler, GeneticAlgorithmSampler,
    AdaptiveSamplingOrchestrator
)

from parameter_testing_and_pareto import TestResult, PerformanceEvaluator

logger = logging.getLogger(__name__)

@dataclass
class ParetoObjective:
    """Definition of a Pareto optimization objective"""
    name: str
    metric_name: str
    maximize: bool
    weight: float = 1.0
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass 
class ParetoConfiguration:
    """Configuration for Pareto frontier analysis"""
    # Sampling configuration
    sampling_config: SamplingConfig
    
    # Pareto-specific settings
    pareto_objectives: List[ParetoObjective]
    dominance_epsilon: float = 0.01  # Epsilon for epsilon-dominance
    crowding_distance_weight: float = 0.1
    convergence_tolerance: float = 0.001
    max_frontier_size: int = 100
    
    # Multi-objective settings
    reference_point: Optional[List[float]] = None
    ideal_point: Optional[List[float]] = None
    nadir_point: Optional[List[float]] = None
    
    # Adaptive sampling for Pareto
    adaptation_frequency: int = 25
    diversity_bonus: float = 0.2
    convergence_bonus: float = 0.3

class ParetoPoint:
    """Represents a point on the Pareto frontier"""
    
    def __init__(self, test_result: TestResult, objectives: List[ParetoObjective]):
        self.test_result = test_result
        self.objectives = objectives
        self.objective_values = self._extract_objective_values()
        self.dominates_count = 0
        self.dominated_by = []
        self.crowding_distance = 0.0
        self.rank = 0
        
    def _extract_objective_values(self) -> List[float]:
        """Extract objective values from test result"""
        values = []
        for obj in self.objectives:
            value = getattr(self.test_result, obj.metric_name, 0.0)
            
            # Apply constraints
            if obj.constraint_min is not None:
                value = max(value, obj.constraint_min)
            if obj.constraint_max is not None:
                value = min(value, obj.constraint_max)
            
            # Negate if minimizing (for easier dominance checking)
            if not obj.maximize:
                value = -value
                
            values.append(value * obj.weight)
        
        return values
    
    def dominates(self, other: 'ParetoPoint', epsilon: float = 0.0) -> bool:
        """Check if this point dominates another (epsilon-dominance)"""
        at_least_one_better = False
        
        for i, (val1, val2) in enumerate(zip(self.objective_values, other.objective_values)):
            if val1 < val2 - epsilon:  # This point is worse in objective i
                return False
            elif val1 > val2 + epsilon:  # This point is better in objective i
                at_least_one_better = True
        
        return at_least_one_better
    
    def distance_to(self, other: 'ParetoPoint') -> float:
        """Calculate distance to another point in objective space"""
        return np.sqrt(sum((v1 - v2) ** 2 for v1, v2 in 
                          zip(self.objective_values, other.objective_values)))

class ParetoFrontier:
    """Manages Pareto frontier with advanced operations"""
    
    def __init__(self, config: ParetoConfiguration):
        self.config = config
        self.points = []
        self.dominated_points = []
        self.history = []
        self.generation = 0
        
    def add_point(self, test_result: TestResult) -> bool:
        """Add point to frontier, return True if it's non-dominated"""
        new_point = ParetoPoint(test_result, self.config.pareto_objectives)
        
        # Check dominance relationships
        is_dominated = False
        points_to_remove = []
        
        for existing_point in self.points:
            if existing_point.dominates(new_point, self.config.dominance_epsilon):
                is_dominated = True
                break
            elif new_point.dominates(existing_point, self.config.dominance_epsilon):
                points_to_remove.append(existing_point)
        
        if not is_dominated:
            # Remove dominated points
            for point in points_to_remove:
                self.points.remove(point)
                self.dominated_points.append(point)
            
            # Add new point
            self.points.append(new_point)
            self._update_crowding_distances()
            self._maintain_frontier_size()
            
            logger.debug(f"Added point to Pareto frontier (size: {len(self.points)})")
            return True
        else:
            self.dominated_points.append(new_point)
            return False
    
    def _update_crowding_distances(self):
        """Update crowding distances for all points"""
        if len(self.points) <= 2:
            for point in self.points:
                point.crowding_distance = float('inf')
            return
        
        # Initialize crowding distances
        for point in self.points:
            point.crowding_distance = 0.0
        
        # Calculate crowding distance for each objective
        for obj_idx in range(len(self.config.pareto_objectives)):
            # Sort points by this objective
            sorted_points = sorted(self.points, 
                                 key=lambda p: p.objective_values[obj_idx])
            
            # Boundary points get infinite distance
            sorted_points[0].crowding_distance = float('inf')
            sorted_points[-1].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_range = (sorted_points[-1].objective_values[obj_idx] - 
                        sorted_points[0].objective_values[obj_idx])
            
            if obj_range > 0:
                # Add crowding distance for middle points
                for i in range(1, len(sorted_points) - 1):
                    distance = (sorted_points[i+1].objective_values[obj_idx] - 
                               sorted_points[i-1].objective_values[obj_idx]) / obj_range
                    sorted_points[i].crowding_distance += distance
    
    def _maintain_frontier_size(self):
        """Maintain frontier size by removing crowded points"""
        if len(self.points) <= self.config.max_frontier_size:
            return
        
        # Sort by crowding distance (ascending)
        sorted_points = sorted(self.points, key=lambda p: p.crowding_distance)
        
        # Remove most crowded points
        points_to_remove = sorted_points[:len(self.points) - self.config.max_frontier_size]
        
        for point in points_to_remove:
            self.points.remove(point)
            self.dominated_points.append(point)
        
        logger.info(f"Trimmed frontier from {len(self.points) + len(points_to_remove)} to {len(self.points)} points")
    
    def get_hypervolume(self, reference_point: Optional[List[float]] = None) -> float:
        """Calculate hypervolume indicator"""
        if not self.points:
            return 0.0
        
        if reference_point is None:
            # Use nadir point as reference
            reference_point = self.config.nadir_point or [0.0] * len(self.config.pareto_objectives)
        
        # Simple hypervolume calculation for 2D case
        if len(self.config.pareto_objectives) == 2:
            # Sort points by first objective
            sorted_points = sorted(self.points, 
                                 key=lambda p: p.objective_values[0], reverse=True)
            
            hypervolume = 0.0
            prev_y = reference_point[1]
            
            for point in sorted_points:
                x, y = point.objective_values[0], point.objective_values[1]
                if x > reference_point[0] and y > prev_y:
                    hypervolume += (x - reference_point[0]) * (y - prev_y)
                    prev_y = y
            
            return hypervolume
        
        # For higher dimensions, use approximation
        return len(self.points)  # Simplified metric
    
    def get_diversity_metric(self) -> float:
        """Calculate diversity metric of the frontier"""
        if len(self.points) < 2:
            return 0.0
        
        distances = []
        for i, point1 in enumerate(self.points):
            for point2 in self.points[i+1:]:
                distances.append(point1.distance_to(point2))
        
        return np.mean(distances) if distances else 0.0
    
    def get_convergence_metric(self, ideal_point: Optional[List[float]] = None) -> float:
        """Calculate convergence metric to ideal point"""
        if not self.points:
            return float('inf')
        
        if ideal_point is None:
            ideal_point = self.config.ideal_point or [float('inf')] * len(self.config.pareto_objectives)
        
        distances = []
        for point in self.points:
            distance = np.sqrt(sum((v - ideal) ** 2 for v, ideal in 
                                 zip(point.objective_values, ideal_point)))
            distances.append(distance)
        
        return np.mean(distances)

class EnhancedParetoSampler:
    """Enhanced sampler for Pareto frontier optimization"""
    
    def __init__(self, parameter_space: ParameterSpace, 
                 config: ParetoConfiguration):
        self.parameter_space = parameter_space
        self.config = config
        self.frontier = ParetoFrontier(config)
        self.evaluator = PerformanceEvaluator()
        
        # Initialize adaptive sampling orchestrator
        self.orchestrator = AdaptiveSamplingOrchestrator(config.sampling_config)
        
        # Pareto-specific state
        self.generation = 0
        self.convergence_history = deque(maxlen=20)
        self.diversity_history = deque(maxlen=20)
        self.hypervolume_history = deque(maxlen=20)
        
        # Adaptive weights for objective balancing
        self.objective_weights = [1.0] * len(config.pareto_objectives)
        self.objective_success_rates = [0.5] * len(config.pareto_objectives)
    
    def run_pareto_optimization(self, max_evaluations: int = 200) -> Dict[str, Any]:
        """Run Pareto frontier optimization"""
        logger.info("🎯 Starting Enhanced Pareto Frontier Optimization")
        
        # Ensure data files exist
        if not Path("./training_data").exists():
            from file_based_training_inference import create_sample_data_files
            create_sample_data_files()
        
        evaluation_count = 0
        stagnation_count = 0
        last_frontier_size = 0
        
        try:
            while evaluation_count < max_evaluations:
                # Sample next configuration
                params = self._sample_next_configuration()
                
                # Evaluate configuration
                result = self._evaluate_configuration(params)
                
                # Add to frontier
                is_non_dominated = self.frontier.add_point(result)
                
                # Update metrics
                self._update_pareto_metrics()
                
                # Adaptive strategy updates
                if evaluation_count % self.config.adaptation_frequency == 0:
                    self._adapt_sampling_strategy()
                
                evaluation_count += 1
                
                # Check for stagnation
                if len(self.frontier.points) == last_frontier_size:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                    last_frontier_size = len(self.frontier.points)
                
                # Early stopping if stagnated
                if stagnation_count > 50:
                    logger.info(f"Early stopping due to stagnation at evaluation {evaluation_count}")
                    break
                
                if evaluation_count % 25 == 0:
                    self._log_progress(evaluation_count, max_evaluations)
        
        except KeyboardInterrupt:
            logger.info(f"Optimization interrupted at evaluation {evaluation_count}")
        
        # Generate final analysis
        final_analysis = self._generate_final_analysis(evaluation_count)
        
        return {
            'frontier': self.frontier,
            'total_evaluations': evaluation_count,
            'final_analysis': final_analysis,
            'convergence_history': list(self.convergence_history),
            'diversity_history': list(self.diversity_history),
            'hypervolume_history': list(self.hypervolume_history)
        }
    
    def _sample_next_configuration(self) -> Dict[str, Any]:
        """Sample next configuration using adaptive strategy"""
        # Use orchestrator for base sampling
        base_params = self._get_base_sample()
        
        # Apply Pareto-specific adaptations
        if len(self.frontier.points) > 5 and random.random() < 0.3:
            # 30% chance to sample around existing Pareto points
            return self._sample_around_pareto_point(base_params)
        elif random.random() < 0.2:
            # 20% chance for diversity-promoting sampling
            return self._sample_for_diversity(base_params)
        else:
            # Standard sampling
            return base_params
    
    def _get_base_sample(self) -> Dict[str, Any]:
        """Get base sample from orchestrator"""
        # Use random sampling as placeholder (orchestrator integration needed)
        return self.parameter_space.sample_random()
    
    def _sample_around_pareto_point(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample around existing Pareto points"""
        if not self.frontier.points:
            return base_params
        
        # Select Pareto point based on crowding distance (prefer less crowded)
        weights = [1.0 / (point.crowding_distance + 1e-6) for point in self.frontier.points]
        total_weight = sum(weights)
        
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
            selected_point = np.random.choice(self.frontier.points, p=probabilities)
            
            # Mutate around selected point
            return self._mutate_params(selected_point.test_result.parameters)
        
        return base_params
    
    def _sample_for_diversity(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample to increase diversity"""
        if len(self.frontier.points) < 2:
            return base_params
        
        # Find region with lowest density
        best_params = base_params
        max_min_distance = 0
        
        # Try multiple candidates
        for _ in range(10):
            candidate = self.parameter_space.sample_random()
            
            # Calculate minimum distance to existing Pareto points
            min_distance = float('inf')
            for point in self.frontier.points:
                distance = self.parameter_space.distance(
                    candidate, point.test_result.parameters
                )
                min_distance = min(min_distance, distance)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_params = candidate
        
        return best_params
    
    def _mutate_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate parameters for local exploration"""
        mutated = copy.deepcopy(base_params)
        
        # Mutate 1-2 parameters
        params_to_mutate = random.sample(
            list(mutated.keys()), 
            min(2, len(mutated))
        )
        
        for param_name in params_to_mutate:
            if param_name in self.parameter_space.continuous_params:
                low, high, scale = self.parameter_space.continuous_params[param_name]
                current_value = mutated[param_name]
                
                if scale == 'log':
                    log_value = np.log(current_value)
                    noise = np.random.normal(0, 0.1)
                    new_log_value = log_value + noise
                    mutated[param_name] = np.exp(np.clip(new_log_value, np.log(low), np.log(high)))
                else:
                    noise = np.random.normal(0, (high - low) * 0.1)
                    mutated[param_name] = np.clip(current_value + noise, low, high)
            
            elif param_name in self.parameter_space.discrete_params:
                mutated[param_name] = np.random.choice(
                    self.parameter_space.discrete_params[param_name]
                )
        
        return self.parameter_space._apply_constraints(mutated)
    
    def _evaluate_configuration(self, params: Dict[str, Any]) -> TestResult:
        """Evaluate parameter configuration"""
        try:
            from file_based_training_inference import (
                create_dummy_models, FileBasedTrainingSystem, FileBasedConfig
            )
            
            # Create models and config
            base_model, prm_model, orm_model, tokenizer = create_dummy_models()
            config = self._params_to_config(params)
            
            # Create system
            system = FileBasedTrainingSystem(base_model, prm_model, orm_model, tokenizer, config)
            
            start_time = time.time()
            system.start()
            
            # Quick evaluation for Pareto optimization
            time.sleep(20)
            
            # Evaluate performance
            training_metrics = self.evaluator.evaluate_training_performance(system, duration=15)
            inference_metrics = self.evaluator.evaluate_inference_performance(
                system, ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
            )
            resource_metrics = self.evaluator.evaluate_resource_usage()
            
            end_time = time.time()
            system.stop()
            
            # Create result
            result = TestResult(
                config_hash=str(hash(str(params))),
                parameters=params,
                average_training_loss=training_metrics.get('average_training_loss', float('inf')),
                training_speed=training_metrics.get('training_speed', 0),
                average_inference_score=inference_metrics.get('average_inference_score', 0),
                inference_speed=inference_metrics.get('inference_speed', 0),
                response_coherence=inference_metrics.get('response_coherence', 0),
                response_diversity=inference_metrics.get('response_diversity', 0),
                reward_consistency=1.0 - inference_metrics.get('score_variance', 1.0),
                memory_usage=resource_metrics.get('memory_usage', 0),
                computational_cost=resource_metrics.get('memory_usage', 0) * (end_time - start_time),
                convergence_rate=training_metrics.get('convergence_rate', 0),
                loss_variance=training_metrics.get('loss_variance', float('inf')),
                score_variance=inference_metrics.get('score_variance', float('inf')),
                total_time=end_time - start_time,
                energy_efficiency=inference_metrics.get('average_inference_score', 0) / 
                                max(end_time - start_time, 1e-10)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return failed result
            return TestResult(
                config_hash=str(hash(str(params))),
                parameters=params,
                average_training_loss=float('inf'),
                training_speed=0,
                average_inference_score=0,
                inference_speed=0,
                response_coherence=0,
                response_diversity=0,
                reward_consistency=0,
                memory_usage=0,
                computational_cost=float('inf'),
                convergence_rate=0,
                loss_variance=float('inf'),
                score_variance=float('inf'),
                total_time=0,
                energy_efficiency=0
            )
    
    def _params_to_config(self, params: Dict[str, Any]):
        """Convert parameters to FileBasedConfig"""
        from file_based_training_inference import FileBasedConfig
        
        config = FileBasedConfig()
        
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
        
        for param_name, config_name in mapping.items():
            if param_name in params:
                setattr(config, config_name, params[param_name])
        
        config.auto_training = True
        config.save_intermediate_results = False
        config.min_data_size_for_training = 5
        
        return config