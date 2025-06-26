# enhanced_sampling_methods.py
"""
Enhanced Sampling Methods for Parameter Testing
Includes tree-based, parallel, serial, and backtracking sampling strategies
"""
import torch
import numpy as np
import json
import time
import random
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from abc import ABC, abstractmethod
import copy
import heapq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm
import itertools

# Import your original classes
from file_based_training_inference import (
    FileBasedConfig, FileBasedTrainingSystem, 
    create_dummy_models, create_sample_data_files
)
from parameter_testing_and_pareto import TestResult, PerformanceEvaluator

logger = logging.getLogger(__name__)

@dataclass
class SamplingConfig:
    """Configuration for sampling methods"""
    # General sampling parameters
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    exploration_rate: float = 0.1
    
    # Tree-based sampling
    tree_depth: int = 5
    branching_factor: int = 3
    pruning_threshold: float = 0.1
    
    # Parallel sampling
    num_workers: int = 4
    batch_size: int = 8
    async_mode: bool = True
    
    # Backtracking
    max_backtrack_depth: int = 10
    improvement_threshold: float = 0.05
    memory_size: int = 1000
    
    # Bayesian optimization
    acquisition_function: str = "ei"  # ei, ucb, pi
    n_initial_points: int = 10
    
    # Genetic algorithm
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2

class ParameterSpace:
    """Represents the parameter space for sampling"""
    
    def __init__(self):
        self.continuous_params = {
            'learning_rate': (1e-6, 1e-3, 'log'),
            'temperature': (0.1, 2.0, 'linear'),
            'top_p': (0.5, 1.0, 'linear'),
            'prm_weight': (0.0, 1.0, 'linear'),
            'orm_weight': (0.0, 1.0, 'linear')
        }
        
        self.discrete_params = {
            'batch_size': [4, 8, 16, 32, 64],
            'num_candidates': [1, 3, 5, 8, 10],
            'max_sequence_length': [256, 512, 1024, 2048],
            'training_epochs_per_batch': [1, 2, 3, 4, 5],
            'data_refresh_interval': [10, 20, 30, 60, 120]
        }
        
        self.constraints = [
            lambda params: params.get('prm_weight', 0) + params.get('orm_weight', 0) <= 1.0,
            lambda params: params.get('batch_size', 1) <= params.get('max_sequence_length', 512) // 4
        ]
    
    def sample_random(self) -> Dict[str, Any]:
        """Sample random point from parameter space"""
        params = {}
        
        # Sample continuous parameters
        for name, (low, high, scale) in self.continuous_params.items():
            if scale == 'log':
                params[name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                params[name] = np.random.uniform(low, high)
        
        # Sample discrete parameters
        for name, values in self.discrete_params.items():
            params[name] = np.random.choice(values)
        
        # Apply constraints
        params = self._apply_constraints(params)
        return params
    
    def sample_grid(self, resolution: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """Sample grid points from parameter space"""
        if resolution is None:
            resolution = {name: 5 for name in self.continuous_params.keys()}
        
        grid_points = []
        
        # Generate grid for continuous parameters
        continuous_grids = {}
        for name, (low, high, scale) in self.continuous_params.items():
            n_points = resolution.get(name, 5)
            if scale == 'log':
                grid = np.logspace(np.log10(low), np.log10(high), n_points)
            else:
                grid = np.linspace(low, high, n_points)
            continuous_grids[name] = grid
        
        # Generate all combinations
        param_names = list(continuous_grids.keys()) + list(self.discrete_params.keys())
        param_values = (list(continuous_grids.values()) + 
                       list(self.discrete_params.values()))
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            params = self._apply_constraints(params)
            grid_points.append(params)
        
        return grid_points
    
    def _apply_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply constraints to parameter set"""
        # Normalize PRM and ORM weights
        if 'prm_weight' in params and 'orm_weight' in params:
            total = params['prm_weight'] + params['orm_weight']
            if total > 0:
                params['prm_weight'] /= total
                params['orm_weight'] /= total
        
        # Apply other constraints
        for constraint in self.constraints:
            try:
                if not constraint(params):
                    # Simple constraint repair
                    if params.get('batch_size', 1) > params.get('max_sequence_length', 512) // 4:
                        params['batch_size'] = min(self.discrete_params['batch_size'])
            except:
                pass
        
        return params
    
    def distance(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate distance between two parameter sets"""
        dist = 0.0
        count = 0
        
        for name in set(params1.keys()) | set(params2.keys()):
            if name in params1 and name in params2:
                if name in self.continuous_params:
                    # Normalize continuous parameters
                    low, high, _ = self.continuous_params[name]
                    v1 = (params1[name] - low) / (high - low)
                    v2 = (params2[name] - low) / (high - low)
                    dist += (v1 - v2) ** 2
                else:
                    # Discrete parameters
                    dist += 0 if params1[name] == params2[name] else 1
                count += 1
        
        return np.sqrt(dist / count) if count > 0 else 0.0

class BaseSampler(ABC):
    """Base class for all sampling methods"""
    
    def __init__(self, parameter_space: ParameterSpace, config: SamplingConfig):
        self.parameter_space = parameter_space
        self.config = config
        self.evaluator = PerformanceEvaluator()
        self.history = []
        self.best_result = None
    
    @abstractmethod
    def sample_next(self) -> Dict[str, Any]:
        """Sample next parameter configuration"""
        pass
    
    @abstractmethod
    def update(self, params: Dict[str, Any], result: TestResult):
        """Update sampler with evaluation result"""
        pass
    
    def evaluate_params(self, params: Dict[str, Any]) -> TestResult:
        """Evaluate parameter configuration"""
        try:
            # Create models and config
            base_model, prm_model, orm_model, tokenizer = create_dummy_models()
            config = self._params_to_config(params)
            
            # Create system
            system = FileBasedTrainingSystem(base_model, prm_model, orm_model, tokenizer, config)
            
            start_time = time.time()
            system.start()
            
            # Evaluate
            time.sleep(30)  # Quick evaluation
            
            training_metrics = self.evaluator.evaluate_training_performance(system, duration=20)
            inference_metrics = self.evaluator.evaluate_inference_performance(
                system, ["Test prompt 1", "Test prompt 2"]
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
    
    def _params_to_config(self, params: Dict[str, Any]) -> FileBasedConfig:
        """Convert parameters to FileBasedConfig"""
        config = FileBasedConfig()
        
        # Map parameters to config
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
        
        # Set test-specific settings
        config.auto_training = True
        config.save_intermediate_results = False
        config.min_data_size_for_training = 5
        
        return config

class TreeBasedSampler(BaseSampler):
    """Tree-based sampling with hierarchical exploration"""
    
    def __init__(self, parameter_space: ParameterSpace, config: SamplingConfig):
        super().__init__(parameter_space, config)
        self.tree = TreeNode(None, {})
        self.current_node = self.tree
        self.exploration_path = []
    
    def sample_next(self) -> Dict[str, Any]:
        """Sample next configuration using tree-based strategy"""
        if self.current_node.is_leaf():
            # Expand current node
            self._expand_node(self.current_node)
        
        # Select best child using UCB1
        if self.current_node.children:
            best_child = self._select_best_child(self.current_node)
            self.current_node = best_child
            self.exploration_path.append(best_child)
            
            # Generate parameters from path
            params = self._path_to_params()
            return params
        else:
            # Fallback to random sampling
            return self.parameter_space.sample_random()
    
    def update(self, params: Dict[str, Any], result: TestResult):
        """Update tree with evaluation result"""
        self.history.append((params, result))
        
        # Update current node
        score = self._result_to_score(result)
        self.current_node.update(score)
        
        # Backpropagate
        for node in reversed(self.exploration_path):
            node.update(score)
        
        # Prune poor branches
        if len(self.history) % 10 == 0:
            self._prune_tree()
        
        # Update best result
        if self.best_result is None or score > self._result_to_score(self.best_result):
            self.best_result = result
    
    def _expand_node(self, node: 'TreeNode'):
        """Expand tree node with new children"""
        for _ in range(self.config.branching_factor):
            # Sample random parameter modification
            child_params = copy.deepcopy(node.params)
            
            # Randomly modify one parameter
            param_name = random.choice(list(self.parameter_space.continuous_params.keys()) + 
                                     list(self.parameter_space.discrete_params.keys()))
            
            if param_name in self.parameter_space.continuous_params:
                low, high, scale = self.parameter_space.continuous_params[param_name]
                if scale == 'log':
                    child_params[param_name] = np.exp(
                        np.random.uniform(np.log(low), np.log(high))
                    )
                else:
                    child_params[param_name] = np.random.uniform(low, high)
            else:
                child_params[param_name] = np.random.choice(
                    self.parameter_space.discrete_params[param_name]
                )
            
            child_params = self.parameter_space._apply_constraints(child_params)
            child_node = TreeNode(node, child_params)
            node.add_child(child_node)
    
    def _select_best_child(self, node: 'TreeNode') -> 'TreeNode':
        """Select best child using UCB1"""
        if not node.children:
            return node
        
        best_score = float('-inf')
        best_child = None
        
        for child in node.children:
            if child.visits == 0:
                return child  # Prefer unvisited nodes
            
            # UCB1 formula
            exploitation = child.total_score / child.visits
            exploration = np.sqrt(2 * np.log(node.visits) / child.visits)
            ucb_score = exploitation + self.config.exploration_rate * exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child or node.children[0]
    
    def _path_to_params(self) -> Dict[str, Any]:
        """Convert exploration path to parameters"""
        params = {}
        for node in self.exploration_path:
            params.update(node.params)
        
        # Fill missing parameters
        for name in self.parameter_space.continuous_params.keys():
            if name not in params:
                low, high, scale = self.parameter_space.continuous_params[name]
                if scale == 'log':
                    params[name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    params[name] = np.random.uniform(low, high)
        
        for name in self.parameter_space.discrete_params.keys():
            if name not in params:
                params[name] = np.random.choice(self.parameter_space.discrete_params[name])
        
        return self.parameter_space._apply_constraints(params)
    
    def _prune_tree(self):
        """Prune poorly performing branches"""
        self._prune_node(self.tree)
    
    def _prune_node(self, node: 'TreeNode'):
        """Recursively prune node and children"""
        if not node.children:
            return
        
        # Calculate average performance
        if node.visits > 0:
            avg_score = node.total_score / node.visits
            
            # Remove children performing below threshold
            children_to_remove = []
            for child in node.children:
                if child.visits > 0:
                    child_avg = child.total_score / child.visits
                    if child_avg < avg_score - self.config.pruning_threshold:
                        children_to_remove.append(child)
            
            for child in children_to_remove:
                node.children.remove(child)
        
        # Recursively prune remaining children
        for child in node.children:
            self._prune_node(child)
    
    def _result_to_score(self, result: TestResult) -> float:
        """Convert test result to score for tree evaluation"""
        # Weighted combination of metrics
        score = (
            0.4 * result.average_inference_score +
            0.2 * result.energy_efficiency +
            0.2 * (1.0 / (1.0 + result.average_training_loss)) +
            0.1 * result.response_coherence +
            0.1 * result.response_diversity
        )
        return max(0, score)

class TreeNode:
    """Node in the parameter exploration tree"""
    
    def __init__(self, parent: Optional['TreeNode'], params: Dict[str, Any]):
        self.parent = parent
        self.params = params
        self.children = []
        self.visits = 0
        self.total_score = 0.0
    
    def add_child(self, child: 'TreeNode'):
        """Add child node"""
        self.children.append(child)
    
    def update(self, score: float):
        """Update node statistics"""
        self.visits += 1
        self.total_score += score
    
    def is_leaf(self) -> bool:
        """Check if node is leaf"""
        return len(self.children) == 0
    
    def average_score(self) -> float:
        """Get average score"""
        return self.total_score / self.visits if self.visits > 0 else 0.0

class ParallelSampler(BaseSampler):
    """Parallel sampling with worker coordination"""
    
    def __init__(self, parameter_space: ParameterSpace, config: SamplingConfig):
        super().__init__(parameter_space, config)
        self.sample_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.coordinator = None
        self.running = False
        
        # Initialize sample queue
        for _ in range(config.batch_size * 2):
            self.sample_queue.put(self.parameter_space.sample_random())
    
    def start_parallel_sampling(self):
        """Start parallel sampling workers"""
        self.running = True
        
        # Start coordinator
        self.coordinator = threading.Thread(target=self._coordinator_loop, daemon=True)
        self.coordinator.start()
        
        # Start workers
        for i in range(self.config.num_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                args=(i,), 
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.config.num_workers} parallel sampling workers")
    
    def stop_parallel_sampling(self):
        """Stop parallel sampling"""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        if self.coordinator:
            self.coordinator.join(timeout=5)
        
        logger.info("Stopped parallel sampling")
    
    def sample_next(self) -> Dict[str, Any]:
        """Get next sample from queue"""
        try:
            return self.sample_queue.get_nowait()
        except queue.Empty:
            return self.parameter_space.sample_random()
    
    def update(self, params: Dict[str, Any], result: TestResult):
        """Update with evaluation result"""
        self.result_queue.put((params, result))
        self.history.append((params, result))
        
        # Update best result
        score = self._result_to_score(result)
        if self.best_result is None or score > self._result_to_score(self.best_result):
            self.best_result = result
    
    def _coordinator_loop(self):
        """Coordinator thread for managing sampling strategy"""
        adaptation_interval = 50
        iteration = 0
        
        while self.running:
            time.sleep(1)
            iteration += 1
            
            # Process results
            while not self.result_queue.empty():
                try:
                    params, result = self.result_queue.get_nowait()
                    self._update_sampling_strategy(params, result)
                except queue.Empty:
                    break
            
            # Refill sample queue
            while self.sample_queue.qsize() < self.config.batch_size:
                new_sample = self._generate_adaptive_sample()
                self.sample_queue.put(new_sample)
            
            # Adapt strategy
            if iteration % adaptation_interval == 0:
                self._adapt_sampling_strategy()
    
    def _worker_loop(self, worker_id: int):
        """Worker thread for evaluating samples"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get sample
                params = self.sample_queue.get(timeout=1)
                
                # Evaluate
                result = self.evaluate_params(params)
                
                # Report result
                self.update(params, result)
                
                logger.debug(f"Worker {worker_id} completed evaluation")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _generate_adaptive_sample(self) -> Dict[str, Any]:
        """Generate adaptive sample based on history"""
        if len(self.history) < 10:
            return self.parameter_space.sample_random()
        
        # Analyze best performing configurations
        sorted_history = sorted(
            self.history, 
            key=lambda x: self._result_to_score(x[1]), 
            reverse=True
        )
        
        # Use top 25% for guidance
        top_configs = sorted_history[:len(sorted_history)//4]
        
        if random.random() < 0.7:  # 70% exploitation
            # Sample around best configurations
            base_params, _ = random.choice(top_configs)
            return self._mutate_params(base_params)
        else:  # 30% exploration
            return self.parameter_space.sample_random()
    
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
                
                # Gaussian mutation around current value
                if scale == 'log':
                    log_value = np.log(current_value)
                    noise = np.random.normal(0, 0.1)
                    new_log_value = log_value + noise
                    mutated[param_name] = np.exp(np.clip(new_log_value, np.log(low), np.log(high)))
                else:
                    noise = np.random.normal(0, (high - low) * 0.1)
                    mutated[param_name] = np.clip(current_value + noise, low, high)
            
            elif param_name in self.parameter_space.discrete_params:
                # Random choice from discrete options
                mutated[param_name] = np.random.choice(
                    self.parameter_space.discrete_params[param_name]
                )
        
        return self.parameter_space._apply_constraints(mutated)
    
    def _update_sampling_strategy(self, params: Dict[str, Any], result: TestResult):
        """Update sampling strategy based on result"""
        # Implementation for adaptive strategy updates
        pass
    
    def _adapt_sampling_strategy(self):
        """Adapt sampling strategy based on accumulated results"""
        # Implementation for strategy adaptation
        pass
    
    def _result_to_score(self, result: TestResult) -> float:
        """Convert result to score"""
        return (
            0.4 * result.average_inference_score +
            0.3 * result.energy_efficiency +
            0.2 * (1.0 / (1.0 + result.average_training_loss)) +
            0.1 * result.response_coherence
        )

class SerialSampler(BaseSampler):
    """Serial sampling with sophisticated exploration strategies"""
    
    def __init__(self, parameter_space: ParameterSpace, config: SamplingConfig):
        super().__init__(parameter_space, config)
        self.sampling_strategy = 'random'
        self.iteration = 0
        self.convergence_history = deque(maxlen=20)
        
        # Strategy-specific state
        self.grid_iterator = None
        self.current_region = None
        self.exploitation_center = None
    
    def sample_next(self) -> Dict[str, Any]:
        """Sample next configuration using current strategy"""
        self.iteration += 1
        
        # Adapt strategy based on progress
        if self.iteration % 20 == 0:
            self._adapt_strategy()
        
        if self.sampling_strategy == 'random':
            return self.parameter_space.sample_random()
        
        elif self.sampling_strategy == 'grid':
            return self._sample_grid()
        
        elif self.sampling_strategy == 'exploitation':
            return self._sample_exploitation()
        
        elif self.sampling_strategy == 'exploration':
            return self._sample_exploration()
        
        elif self.sampling_strategy == 'hybrid':
            return self._sample_hybrid()
        
        else:
            return self.parameter_space.sample_random()
    
    def update(self, params: Dict[str, Any], result: TestResult):
        """Update sampler with result"""
        self.history.append((params, result))
        
        # Track convergence
        score = self._result_to_score(result)
        self.convergence_history.append(score)
        
        # Update best result
        if self.best_result is None or score > self._result_to_score(self.best_result):
            self.best_result = result
            self.exploitation_center = params
        
        logger.debug(f"Serial sampler iteration {self.iteration}, score: {score:.3f}")
    
    def _adapt_strategy(self):
        """Adapt sampling strategy based on progress"""
        if len(self.convergence_history) < 10:
            return
        
        # Calculate convergence metrics
        recent_scores = list(self.convergence_history)[-10:]
        improvement = max(recent_scores) - min(recent_scores)
        variance = np.var(recent_scores)
        
        # Choose strategy based on convergence
        if improvement < self.config.convergence_threshold:
            if self.sampling_strategy != 'exploration':
                self.sampling_strategy = 'exploration'
                logger.info("Switched to exploration strategy")
        elif variance < 0.01:
            if self.sampling_strategy != 'exploitation':
                self.sampling_strategy = 'exploitation'
                logger.info("Switched to exploitation strategy")
        else:
            if self.sampling_strategy != 'hybrid':
                self.sampling_strategy = 'hybrid'
                logger.info("Switched to hybrid strategy")
    
    def _sample_grid(self) -> Dict[str, Any]:
        """Sample from grid"""
        if self.grid_iterator is None:
            grid_points = self.parameter_space.sample_grid({'learning_rate': 3, 'temperature': 3})
            self.grid_iterator = iter(grid_points)
        
        try:
            return next(self.grid_iterator)
        except StopIteration:
            self.grid_iterator = None
            return self.parameter_space.sample_random()
    
    def _sample_exploitation(self) -> Dict[str, Any]:
        """Sample around best known configuration"""
        if self.exploitation_center is None:
            return self.parameter_space.sample_random()
        
        # Local search around best configuration
        params = copy.deepcopy(self.exploitation_center)
        
        # Mutate parameters with small noise
        for name in params:
            if name in self.parameter_space.continuous_params:
                low, high, scale = self.parameter_space.continuous_params[name]
                current = params[name]
                
                if scale == 'log':
                    log_current = np.log(current)
                    noise = np.random.normal(0, 0.05)  # Small noise
                    new_log = log_current + noise
                    params[name] = np.exp(np.clip(new_log, np.log(low), np.log(high)))
                else:
                    noise = np.random.normal(0, (high - low) * 0.05)
                    params[name] = np.clip(current + noise, low, high)