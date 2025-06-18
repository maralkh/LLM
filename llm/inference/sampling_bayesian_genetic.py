# sampling_bayesian_genetic.py
"""
Bayesian Optimization and Genetic Algorithm Sampling Methods
Advanced optimization techniques using Gaussian Process and evolutionary algorithms
"""
import numpy as np
import random
import copy
import logging
from typing import Dict, Any, List, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm

from enhanced_sampling_methods import BaseSampler, ParameterSpace, SamplingConfig
from parameter_testing_and_pareto import TestResult

logger = logging.getLogger(__name__)

class BayesianOptimizationSampler(BaseSampler):
    """Bayesian optimization using Gaussian Process"""
    
    def __init__(self, parameter_space: ParameterSpace, config: SamplingConfig):
        super().__init__(parameter_space, config)
        self.gp = None
        self.scaler = StandardScaler()
        self.X_train = []
        self.y_train = []
        self.acquisition_func = config.acquisition_function
        
        # Initialize with random samples
        self.initial_samples = []
        for _ in range(config.n_initial_points):
            self.initial_samples.append(self.parameter_space.sample_random())
        self.initial_iter = 0
    
    def sample_next(self) -> Dict[str, Any]:
        """Sample next point using Bayesian optimization"""
        # Use initial random samples first
        if self.initial_iter < len(self.initial_samples):
            params = self.initial_samples[self.initial_iter]
            self.initial_iter += 1
            return params
        
        # Build/update GP model
        if len(self.X_train) >= 3:
            self._update_gp_model()
            
            # Optimize acquisition function
            best_params = self._optimize_acquisition()
            return best_params
        else:
            # Not enough data for GP, use random sampling
            return self.parameter_space.sample_random()
    
    def update(self, params: Dict[str, Any], result: TestResult):
        """Update GP model with new observation"""
        self.history.append((params, result))
        
        # Convert params to feature vector
        X = self._params_to_vector(params)
        y = self._result_to_score(result)
        
        self.X_train.append(X)
        self.y_train.append(y)
        
        # Update best result
        if self.best_result is None or y > self._result_to_score(self.best_result):
            self.best_result = result
        
        logger.debug(f"Bayesian optimization: added observation with score {y:.3f}")
    
    def _params_to_vector(self, params: Dict[str, Any]) -> List[float]:
        """Convert parameters to feature vector"""
        vector = []
        
        # Add continuous parameters
        for name in sorted(self.parameter_space.continuous_params.keys()):
            if name in params:
                low, high, scale = self.parameter_space.continuous_params[name]
                value = params[name]
                if scale == 'log':
                    # Log transform
                    normalized = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))
                else:
                    # Linear normalization
                    normalized = (value - low) / (high - low)
                vector.append(normalized)
            else:
                vector.append(0.5)  # Default middle value
        
        # Add discrete parameters (one-hot encoding)
        for name in sorted(self.parameter_space.discrete_params.keys()):
            options = self.parameter_space.discrete_params[name]
            if name in params:
                # One-hot encode
                one_hot = [0.0] * len(options)
                if params[name] in options:
                    idx = options.index(params[name])
                    one_hot[idx] = 1.0
                vector.extend(one_hot)
            else:
                # Default to first option
                one_hot = [1.0] + [0.0] * (len(options) - 1)
                vector.extend(one_hot)
        
        return vector
    
    def _vector_to_params(self, vector: List[float]) -> Dict[str, Any]:
        """Convert feature vector back to parameters"""
        params = {}
        idx = 0
        
        # Extract continuous parameters
        for name in sorted(self.parameter_space.continuous_params.keys()):
            low, high, scale = self.parameter_space.continuous_params[name]
            normalized = np.clip(vector[idx], 0, 1)  # Ensure bounds
            if scale == 'log':
                # Inverse log transform
                log_value = normalized * (np.log(high) - np.log(low)) + np.log(low)
                params[name] = np.exp(log_value)
            else:
                # Inverse linear normalization
                params[name] = normalized * (high - low) + low
            idx += 1
        
        # Extract discrete parameters
        for name in sorted(self.parameter_space.discrete_params.keys()):
            options = self.parameter_space.discrete_params[name]
            one_hot = vector[idx:idx + len(options)]
            # Select option with highest probability
            best_idx = np.argmax(one_hot)
            params[name] = options[best_idx]
            idx += len(options)
        
        return self.parameter_space._apply_constraints(params)
    
    def _update_gp_model(self):
        """Update Gaussian Process model"""
        if len(self.X_train) < 3:
            return
        
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit GP
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        self.gp.fit(X_scaled, y)
        logger.debug(f"Updated GP model with {len(X)} observations")
    
    def _optimize_acquisition(self) -> Dict[str, Any]:
        """Optimize acquisition function to find next point"""
        if self.gp is None:
            return self.parameter_space.sample_random()
        
        # Define acquisition function
        def acquisition(x):
            x = x.reshape(1, -1)
            x_scaled = self.scaler.transform(x)
            
            mu, sigma = self.gp.predict(x_scaled, return_std=True)
            
            if self.acquisition_func == 'ei':
                # Expected Improvement
                best_y = max(self.y_train)
                with np.errstate(divide='warn'):
                    imp = mu - best_y
                    Z = imp / (sigma + 1e-9)
                    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                    return -ei[0]  # Minimize negative EI
            
            elif self.acquisition_func == 'ucb':
                # Upper Confidence Bound
                kappa = 2.0
                return -(mu[0] + kappa * sigma[0])
            
            else:  # 'pi' - Probability of Improvement
                best_y = max(self.y_train)
                Z = (mu - best_y) / (sigma + 1e-9)
                return -norm.cdf(Z)[0]
        
        # Optimize acquisition function
        best_x = None
        best_acq = float('inf')
        
        # Try multiple random starts
        for _ in range(20):
            # Random starting point
            x0 = np.random.random(len(self.X_train[0]))
            
            try:
                result = minimize(
                    acquisition, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * len(x0)
                )
                
                if result.success and result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            except:
                continue
        
        # Convert back to parameters
        if best_x is not None:
            return self._vector_to_params(best_x.tolist())
        else:
            return self.parameter_space.sample_random()
    
    def _result_to_score(self, result: TestResult) -> float:
        """Convert result to score"""
        return (
            0.4 * result.average_inference_score +
            0.3 * result.energy_efficiency +
            0.2 * (1.0 / (1.0 + result.average_training_loss)) +
            0.1 * result.response_coherence
        )

class GeneticAlgorithmSampler(BaseSampler):
    """Genetic Algorithm for parameter optimization"""
    
    def __init__(self, parameter_space: ParameterSpace, config: SamplingConfig):
        super().__init__(parameter_space, config)
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.elite_size = int(config.population_size * config.elite_ratio)
        self.current_individual = 0
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population"""
        for _ in range(self.config.population_size):
            individual = self.parameter_space.sample_random()
            self.population.append(individual)
        
        self.fitness_scores = [0.0] * len(self.population)
        logger.info(f"Initialized GA population of size {len(self.population)}")
    
    def sample_next(self) -> Dict[str, Any]:
        """Sample next individual from current generation"""
        if self.current_individual < len(self.population):
            # Return next individual in current generation
            individual = self.population[self.current_individual]
            return individual
        else:
            # All individuals evaluated, evolve to next generation
            self._evolve_population()
            self.current_individual = 0
            if self.population:
                return self.population[self.current_individual]
            else:
                return self.parameter_space.sample_random()
    
    def update(self, params: Dict[str, Any], result: TestResult):
        """Update fitness score for evaluated individual"""
        self.history.append((params, result))
        score = self._result_to_score(result)
        
        # Update fitness for current individual
        if self.current_individual < len(self.fitness_scores):
            self.fitness_scores[self.current_individual] = score
        
        self.current_individual += 1
        
        # Update best result
        if self.best_result is None or score > self._result_to_score(self.best_result):
            self.best_result = result
        
        logger.debug(f"GA: individual {self.current_individual}/{len(self.population)}, score: {score:.3f}")
    
    def _evolve_population(self):
        """Evolve population to next generation"""
        if not self.fitness_scores or all(f == 0 for f in self.fitness_scores):
            # Reinitialize if no valid fitness scores
            self._initialize_population()
            return
        
        self.generation += 1
        logger.info(f"Evolving to generation {self.generation}")
        
        # Selection: keep elite
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        elite = [self.population[i] for i in elite_indices]
        elite_fitness = [self.fitness_scores[i] for i in elite_indices]
        
        # Generate new population
        new_population = elite.copy()  # Keep elite
        
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                # Mutation of elite
                parent = random.choice(elite) if elite else self.parameter_space.sample_random()
                child = copy.deepcopy(parent)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        # Update population and reset fitness scores
        self.population = new_population
        self.fitness_scores = [0.0] * len(self.population)
        
        logger.info(f"Generation {self.generation}: evolved {len(new_population)} individuals")
    
    def _tournament_selection(self) -> Dict[str, Any]:
        """Tournament selection"""
        tournament_size = 3
        if len(self.population) < tournament_size:
            return random.choice(self.population)
        
        tournament_indices = random.sample(range(len(self.population)), tournament_size)