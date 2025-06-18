# sampling_serial_backtrack.py
"""
Serial and Backtracking Sampling Methods
Advanced single-threaded and backtracking sampling strategies
"""
import numpy as np
import random
import copy
import itertools
import logging
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass

from enhanced_sampling_methods import BaseSampler, ParameterSpace, SamplingConfig
from parameter_testing_and_pareto import TestResult

logger = logging.getLogger(__name__)

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
        
        return self.parameter_space._apply_constraints(params)
    
    def _sample_exploration(self) -> Dict[str, Any]:
        """Sample for exploration in unexplored regions"""
        # Find unexplored regions by analyzing distance from previous samples
        if len(self.history) < 5:
            return self.parameter_space.sample_random()
        
        # Try multiple candidates and pick the one farthest from existing samples
        best_params = None
        max_min_distance = 0
        
        for _ in range(20):  # Try 20 candidates
            candidate = self.parameter_space.sample_random()
            
            # Calculate minimum distance to all previous samples
            min_distance = float('inf')
            for prev_params, _ in self.history:
                dist = self.parameter_space.distance(candidate, prev_params)
                min_distance = min(min_distance, dist)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_params = candidate
        
        return best_params or self.parameter_space.sample_random()
    
    def _sample_hybrid(self) -> Dict[str, Any]:
        """Hybrid sampling combining exploitation and exploration"""
        if random.random() < 0.6:  # 60% exploitation
            return self._sample_exploitation()
        else:  # 40% exploration
            return self._sample_exploration()
    
    def _result_to_score(self, result: TestResult) -> float:
        """Convert result to score"""
        return (
            0.4 * result.average_inference_score +
            0.3 * result.energy_efficiency +
            0.2 * (1.0 / (1.0 + result.average_training_loss)) +
            0.1 * result.response_coherence
        )

class BacktrackingSampler(BaseSampler):
    """Backtracking sampler that can undo poor decisions"""
    
    def __init__(self, parameter_space: ParameterSpace, config: SamplingConfig):
        super().__init__(parameter_space, config)
        self.search_tree = BacktrackNode(None, {}, 0)
        self.current_path = [self.search_tree]
        self.backtrack_stack = []
        self.decision_points = []
        self.last_improvement = 0
        
    def sample_next(self) -> Dict[str, Any]:
        """Sample next configuration with backtracking capability"""
        current_node = self.current_path[-1]
        
        # Check if we need to backtrack
        if self._should_backtrack():
            self._perform_backtrack()
            current_node = self.current_path[-1]
        
        # Generate new candidate from current node
        if len(current_node.unexplored_directions) > 0:
            # Explore a new direction
            direction = current_node.unexplored_directions.pop()
            new_params = self._apply_direction(current_node.params, direction)
        else:
            # No more directions, try random perturbation
            new_params = self._random_perturbation(current_node.params)
        
        # Create new node
        new_node = BacktrackNode(
            current_node, 
            new_params, 
            len(self.current_path)
        )
        
        # Add to path
        self.current_path.append(new_node)
        
        # Mark as decision point if we have multiple choices
        if len(current_node.unexplored_directions) > 0:
            self.decision_points.append(len(self.current_path) - 1)
        
        return new_params
    
    def update(self, params: Dict[str, Any], result: TestResult):
        """Update with evaluation result"""
        self.history.append((params, result))
        score = self._result_to_score(result)
        
        # Update current node
        if self.current_path:
            current_node = self.current_path[-1]
            current_node.score = score
            current_node.evaluated = True
            
            # Check for improvement
            if self.best_result is None or score > self._result_to_score(self.best_result):
                self.best_result = result
                self.last_improvement = len(self.history)
                
                # Mark this path as promising
                for node in self.current_path:
                    node.promising = True
            
            # Update parent's children info
            if current_node.parent:
                current_node.parent.children.append(current_node)
        
        logger.debug(f"Backtracking sampler: score={score:.3f}, path_length={len(self.current_path)}")
    
    def _should_backtrack(self) -> bool:
        """Determine if we should backtrack"""
        # Backtrack if no improvement for too long
        no_improvement_steps = len(self.history) - self.last_improvement
        if no_improvement_steps > self.config.max_backtrack_depth:
            return True
        
        # Backtrack if current path seems unpromising
        if len(self.current_path) > 3:
            recent_scores = [node.score for node in self.current_path[-3:] if node.evaluated]
            if len(recent_scores) >= 2 and all(s < self.config.improvement_threshold for s in recent_scores):
                return True
        
        return False
    
    def _perform_backtrack(self):
        """Perform backtracking to a previous decision point"""
        if not self.decision_points:
            # No decision points, restart from root
            self.current_path = [self.search_tree]
            return
        
        # Find a good backtrack point
        backtrack_point = None
        
        # Try to find a promising decision point
        for point_idx in reversed(self.decision_points):
            if point_idx < len(self.current_path):
                node = self.current_path[point_idx]
                if node.promising and len(node.unexplored_directions) > 0:
                    backtrack_point = point_idx
                    break
        
        # If no promising point, backtrack to earliest decision point
        if backtrack_point is None and self.decision_points:
            backtrack_point = self.decision_points[0]
        
        if backtrack_point is not None:
            # Backtrack to the decision point
            self.current_path = self.current_path[:backtrack_point + 1]
            
            # Remove decision points after backtrack point
            self.decision_points = [dp for dp in self.decision_points if dp <= backtrack_point]
            
            logger.info(f"Backtracked to depth {backtrack_point}")
        else:
            # No valid backtrack point, restart
            self.current_path = [self.search_tree]
            self.decision_points = []
            logger.info("Restarted from root")
    
    def _apply_direction(self, base_params: Dict[str, Any], direction: Dict[str, str]) -> Dict[str, Any]:
        """Apply a search direction to base parameters"""
        new_params = copy.deepcopy(base_params)
        
        for param_name, change_type in direction.items():
            if param_name in self.parameter_space.continuous_params:
                low, high, scale = self.parameter_space.continuous_params[param_name]
                current = new_params.get(param_name, (low + high) / 2)
                
                if change_type == 'increase':
                    if scale == 'log':
                        new_params[param_name] = min(high, current * 1.2)
                    else:
                        new_params[param_name] = min(high, current + (high - low) * 0.1)
                elif change_type == 'decrease':
                    if scale == 'log':
                        new_params[param_name] = max(low, current * 0.8)
                    else:
                        new_params[param_name] = max(low, current - (high - low) * 0.1)
            
            elif param_name in self.parameter_space.discrete_params:
                options = self.parameter_space.discrete_params[param_name]
                current = new_params.get(param_name, options[0])
                current_idx = options.index(current) if current in options else 0
                
                if change_type == 'increase' and current_idx < len(options) - 1:
                    new_params[param_name] = options[current_idx + 1]
                elif change_type == 'decrease' and current_idx > 0:
                    new_params[param_name] = options[current_idx - 1]
        
        return self.parameter_space._apply_constraints(new_params)
    
    def _random_perturbation(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random perturbation to parameters"""
        new_params = copy.deepcopy(base_params)
        
        # Randomly select 1-2 parameters to perturb
        param_names = list(self.parameter_space.continuous_params.keys()) + \
                     list(self.parameter_space.discrete_params.keys())
        
        to_perturb = random.sample(param_names, min(2, len(param_names)))
        
        for param_name in to_perturb:
            if param_name in self.parameter_space.continuous_params:
                low, high, scale = self.parameter_space.continuous_params[param_name]
                if scale == 'log':
                    new_params[param_name] = np.exp(
                        np.random.uniform(np.log(low), np.log(high))
                    )
                else:
                    new_params[param_name] = np.random.uniform(low, high)
            else:
                new_params[param_name] = np.random.choice(
                    self.parameter_space.discrete_params[param_name]
                )
        
        return self.parameter_space._apply_constraints(new_params)
    
    def _generate_search_directions(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate possible search directions from current parameters"""
        directions = []
        
        # For each parameter, create increase/decrease directions
        for param_name in params:
            if param_name in self.parameter_space.continuous_params or \
               param_name in self.parameter_space.discrete_params:
                directions.append({param_name: 'increase'})
                directions.append({param_name: 'decrease'})
        
        # Multi-parameter directions
        param_names = list(params.keys())
        if len(param_names) >= 2:
            # Coordinate increases/decreases
            for param1, param2 in itertools.combinations(param_names[:4], 2):
                directions.append({param1: 'increase', param2: 'increase'})
                directions.append({param1: 'decrease', param2: 'decrease'})
                directions.append({param1: 'increase', param2: 'decrease'})
                directions.append({param1: 'decrease', param2: 'increase'})
        
        return directions
    
    def _result_to_score(self, result: TestResult) -> float:
        """Convert result to score"""
        return (
            0.4 * result.average_inference_score +
            0.3 * result.energy_efficiency +
            0.2 * (1.0 / (1.0 + result.average_training_loss)) +
            0.1 * result.response_coherence
        )

class BacktrackNode:
    """Node in backtracking search tree"""
    
    def __init__(self, parent: Optional['BacktrackNode'], params: Dict[str, Any], depth: int):
        self.parent = parent
        self.params = params
        self.depth = depth
        self.children = []
        self.score = 0.0
        self.evaluated = False
        self.promising = False
        
        # Generate possible search directions
        self.unexplored_directions = self._generate_directions()
    
    def _generate_directions(self) -> List[Dict[str, str]]:
        """Generate search directions from this node"""
        directions = []
        
        for param_name in self.params:
            directions.append({param_name: 'increase'})
            directions.append({param_name: 'decrease'})
        
        random.shuffle(directions)
        return directions