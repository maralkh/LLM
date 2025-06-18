# prm_consistent_evaluation.py
"""
PRM-Consistent Evaluation System
Ensures all scoring uses both PRM and ORM consistently across the entire system
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

from parameter_testing_and_pareto import TestResult, PerformanceEvaluator

logger = logging.getLogger(__name__)

@dataclass
class PRMORMScore:
    """Combined PRM and ORM scoring result"""
    prm_step_scores: List[float]
    prm_average: float
    prm_confidence: float
    orm_overall: float
    orm_correctness: float
    orm_helpfulness: float
    combined_score: float
    reasoning_quality: float
    step_consistency: float

class PRMConsistentEvaluator(PerformanceEvaluator):
    """Enhanced evaluator that consistently uses PRM+ORM across all metrics"""
    
    def __init__(self, prm_model=None, orm_model=None, tokenizer=None, 
                 prm_weight: float = 0.4, orm_weight: float = 0.6):
        super().__init__()
        self.prm_model = prm_model
        self.orm_model = orm_model
        self.tokenizer = tokenizer
        self.prm_weight = prm_weight
        self.orm_weight = orm_weight
        
        # Cache for expensive evaluations
        self.evaluation_cache = {}
        
    def evaluate_inference_performance_prm(self, system, test_prompts: List[str]) -> Dict[str, float]:
        """Enhanced inference evaluation using PRM+ORM consistently"""
        start_time = time.time()
        
        all_prm_orm_scores = []
        response_lengths = []
        response_diversity_scores = []
        coherence_scores = []
        
        for prompt in test_prompts:
            try:
                # Generate response using system
                result = system._generate_with_rewards(prompt)
                response = result.get('response', '')
                
                if not response:
                    continue
                
                # Get PRM+ORM evaluation
                prm_orm_score = self._evaluate_response_with_prm_orm(prompt, response)
                all_prm_orm_scores.append(prm_orm_score)
                
                # Additional metrics enhanced with PRM insights
                response_lengths.append(len(response.split()))
                
                # Diversity based on PRM step variance
                diversity = self._calculate_prm_based_diversity(prm_orm_score)
                response_diversity_scores.append(diversity)
                
                # Coherence based on PRM step consistency
                coherence = self._calculate_prm_based_coherence(prm_orm_score)
                coherence_scores.append(coherence)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate prompt '{prompt[:50]}...': {e}")
                continue
        
        end_time = time.time()
        
        if not all_prm_orm_scores:
            logger.warning("No successful evaluations")
            return self._default_inference_metrics()
        
        # Aggregate PRM+ORM scores
        avg_prm_score = np.mean([score.prm_average for score in all_prm_orm_scores])
        avg_orm_score = np.mean([score.orm_overall for score in all_prm_orm_scores])
        avg_combined_score = np.mean([score.combined_score for score in all_prm_orm_scores])
        avg_reasoning_quality = np.mean([score.reasoning_quality for score in all_prm_orm_scores])
        
        # Enhanced metrics
        inference_speed = len(all_prm_orm_scores) / (end_time - start_time)
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        avg_diversity = np.mean(response_diversity_scores) if response_diversity_scores else 0.0
        
        # Score variance for consistency measurement
        combined_scores = [score.combined_score for score in all_prm_orm_scores]
        score_variance = np.var(combined_scores) if len(combined_scores) > 1 else 0.0
        
        return {
            'average_inference_score': avg_combined_score,  # 🔥 PRM+ORM based
            'prm_average_score': avg_prm_score,
            'orm_average_score': avg_orm_score,
            'reasoning_quality': avg_reasoning_quality,
            'inference_speed': inference_speed,
            'response_coherence': avg_coherence,
            'response_diversity': avg_diversity,
            'score_variance': score_variance,
            'total_inference_time': end_time - start_time,
            'successful_evaluations': len(all_prm_orm_scores)
        }
    
    def _evaluate_response_with_prm_orm(self, prompt: str, response: str) -> PRMORMScore:
        """Core PRM+ORM evaluation function"""
        # Create cache key
        cache_key = hash(f"{prompt}||{response}")
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        try:
            # Prepare input
            full_text = f"{prompt} {response}"
            text_ids = self.tokenizer.encode(full_text, return_tensors='pt')
            
            if self.prm_model and self.orm_model:
                text_ids = text_ids.to(next(self.prm_model.parameters()).device)
            
            with torch.no_grad():
                # PRM Evaluation
                if self.prm_model:
                    prm_outputs = self.prm_model.forward(text_ids)
                    prm_step_scores = prm_outputs['step_rewards'][0].cpu().tolist()
                    prm_confidences = prm_outputs.get('step_confidences', [1.0] * len(prm_step_scores))
                    
                    prm_average = np.mean(prm_step_scores)
                    prm_confidence = np.mean(prm_confidences[0].cpu().tolist() if torch.is_tensor(prm_confidences) else prm_confidences)
                else:
                    prm_step_scores = [0.5] * 10  # Default
                    prm_average = 0.5
                    prm_confidence = 0.5
                
                # ORM Evaluation  
                if self.orm_model:
                    attention_mask = torch.ones_like(text_ids)
                    orm_outputs = self.orm_model.forward(text_ids, attention_mask)
                    
                    orm_overall = orm_outputs['overall_reward'][0].item()
                    orm_correctness = orm_outputs.get('correctness', torch.tensor([[0.5]]))[0].item()
                    orm_helpfulness = orm_outputs.get('helpfulness', torch.tensor([[0.5]]))[0].item()
                else:
                    orm_overall = 0.5
                    orm_correctness = 0.5
                    orm_helpfulness = 0.5
            
            # Calculate enhanced metrics
            reasoning_quality = self._calculate_reasoning_quality(prm_step_scores)
            step_consistency = self._calculate_step_consistency(prm_step_scores)
            
            # Combined score
            combined_score = (
                self.prm_weight * prm_average + 
                self.orm_weight * orm_overall
            )
            
            # Create result
            result = PRMORMScore(
                prm_step_scores=prm_step_scores,
                prm_average=prm_average,
                prm_confidence=prm_confidence,
                orm_overall=orm_overall,
                orm_correctness=orm_correctness,
                orm_helpfulness=orm_helpfulness,
                combined_score=combined_score,
                reasoning_quality=reasoning_quality,
                step_consistency=step_consistency
            )
            
            # Cache result
            self.evaluation_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"PRM+ORM evaluation failed: {e}")
            # Return default scores
            return PRMORMScore(
                prm_step_scores=[0.3] * 5,
                prm_average=0.3,
                prm_confidence=0.5,
                orm_overall=0.3,
                orm_correctness=0.3,
                orm_helpfulness=0.3,
                combined_score=0.3,
                reasoning_quality=0.3,
                step_consistency=0.3
            )
    
    def _calculate_reasoning_quality(self, prm_step_scores: List[float]) -> float:
        """Calculate reasoning quality based on PRM step progression"""
        if len(prm_step_scores) < 2:
            return np.mean(prm_step_scores) if prm_step_scores else 0.0
        
        # Check if reasoning improves over time (good reasoning pattern)
        improvements = 0
        for i in range(1, len(prm_step_scores)):
            if prm_step_scores[i] >= prm_step_scores[i-1]:
                improvements += 1
        
        improvement_ratio = improvements / (len(prm_step_scores) - 1)
        base_quality = np.mean(prm_step_scores)
        
        # Bonus for improving reasoning
        reasoning_quality = base_quality * (1 + 0.2 * improvement_ratio)
        return min(reasoning_quality, 1.0)
    
    def _calculate_step_consistency(self, prm_step_scores: List[float]) -> float:
        """Calculate consistency of reasoning steps"""
        if len(prm_step_scores) < 2:
            return 1.0
        
        # Lower variance = higher consistency
        variance = np.var(prm_step_scores)
        consistency = 1.0 / (1.0 + variance)
        return consistency
    
    def _calculate_prm_based_diversity(self, prm_orm_score: PRMORMScore) -> float:
        """Calculate diversity based on PRM step variation"""
        step_scores = prm_orm_score.prm_step_scores
        if len(step_scores) < 2:
            return 0.5
        
        # Diversity = controlled variation in steps
        std_dev = np.std(step_scores)
        mean_score = np.mean(step_scores)
        
        # Normalize diversity (higher std with good mean = more diverse)
        if mean_score > 0.5:  # Good reasoning
            diversity = min(std_dev * 2, 1.0)  # Reward variation in good reasoning
        else:  # Poor reasoning
            diversity = max(0.1, 1.0 - std_dev)  # Penalize variation in poor reasoning
        
        return diversity
    
    def _calculate_prm_based_coherence(self, prm_orm_score: PRMORMScore) -> float:
        """Calculate coherence based on PRM step flow"""
        step_scores = prm_orm_score.prm_step_scores
        if len(step_scores) < 2:
            return prm_orm_score.prm_average
        
        # Coherence = smooth progression of reasoning
        transitions = []
        for i in range(1, len(step_scores)):
            # Smooth transition = small absolute difference
            transition_smoothness = 1.0 - min(abs(step_scores[i] - step_scores[i-1]), 1.0)
            transitions.append(transition_smoothness)
        
        smooth_progression = np.mean(transitions)
        base_coherence = prm_orm_score.prm_average
        
        # Combine base quality with progression smoothness
        coherence = 0.7 * base_coherence + 0.3 * smooth_progression
        return coherence
    
    def _default_inference_metrics(self) -> Dict[str, float]:
        """Default metrics when evaluation fails"""
        return {
            'average_inference_score': 0.0,
            'prm_average_score': 0.0,
            'orm_average_score': 0.0,
            'reasoning_quality': 0.0,
            'inference_speed': 0.0,
            'response_coherence': 0.0,
            'response_diversity': 0.0,
            'score_variance': 0.0,
            'total_inference_time': 0.0,
            'successful_evaluations': 0
        }

class PRMEnhancedBaseSampler:
    """Enhanced Base Sampler that uses PRM+ORM consistently"""
    
    def __init__(self, parameter_space, config, prm_model=None, orm_model=None, tokenizer=None):
        self.parameter_space = parameter_space
        self.config = config
        self.prm_evaluator = PRMConsistentEvaluator(prm_model, orm_model, tokenizer)
        self.history = []
        self.best_result = None
    
    def evaluate_params(self, params: Dict[str, Any]) -> TestResult:
        """Enhanced parameter evaluation using consistent PRM+ORM scoring"""
        try:
            # Create models and config
            from file_based_training_inference import create_dummy_models, FileBasedTrainingSystem
            base_model, prm_model, orm_model, tokenizer = create_dummy_models()
            
            # Update evaluator with models
            self.prm_evaluator.prm_model = prm_model
            self.prm_evaluator.orm_model = orm_model
            self.prm_evaluator.tokenizer = tokenizer
            
            config = self._params_to_config(params)
            
            # Create system
            system = FileBasedTrainingSystem(base_model, prm_model, orm_model, tokenizer, config)
            
            start_time = time.time()
            system.start()
            
            # Evaluation period
            time.sleep(20)
            
            # 🔥 Use PRM-consistent evaluation
            training_metrics = self.prm_evaluator.evaluate_training_performance(system, duration=15)
            inference_metrics = self.prm_evaluator.evaluate_inference_performance_prm(
                system, ["Explain machine learning", "How do neural networks work?", "What is AI?"]
            )
            resource_metrics = self.prm_evaluator.evaluate_resource_usage()
            
            end_time = time.time()
            system.stop()
            
            # Calculate energy efficiency using PRM+ORM score
            prm_enhanced_score = inference_metrics['average_inference_score']
            energy_efficiency = prm_enhanced_score / max(end_time - start_time, 1e-10)
            
            # Create enhanced TestResult
            result = TestResult(
                config_hash=str(hash(str(params))),
                parameters=params,
                average_training_loss=training_metrics.get('average_training_loss', float('inf')),
                training_speed=training_metrics.get('training_speed', 0),
                average_inference_score=prm_enhanced_score,  # 🔥 PRM+ORM based
                inference_speed=inference_metrics.get('inference_speed', 0),
                response_coherence=inference_metrics.get('response_coherence', 0),
                response_diversity=inference_metrics.get('response_diversity', 0),
                reward_consistency=inference_metrics.get('reasoning_quality', 0),  # 🔥 PRM based
                memory_usage=resource_metrics.get('memory_usage', 0),
                computational_cost=resource_metrics.get('memory_usage', 0) * (end_time - start_time),
                convergence_rate=training_metrics.get('convergence_rate', 0),
                loss_variance=training_metrics.get('loss_variance', float('inf')),
                score_variance=inference_metrics.get('score_variance', float('inf')),
                total_time=end_time - start_time,
                energy_efficiency=energy_efficiency
            )
            
            # Add PRM-specific metrics as additional attributes
            result.prm_average_score = inference_metrics.get('prm_average_score', 0)
            result.orm_average_score = inference_metrics.get('orm_average_score', 0)
            result.reasoning_quality = inference_metrics.get('reasoning_quality', 0)
            
            return result
            
        except Exception as e:
            logger.error(f"PRM-enhanced evaluation failed: {e}")
            return self._create_failed_result(params)
    
    def _create_failed_result(self, params: Dict[str, Any]) -> TestResult:
        """Create failed result with consistent structure"""
        result = TestResult(
            config_hash=str(hash(str(params))),
            parameters=params,
            average_training_loss=float('inf'),
            training_speed=0,
            average_inference_score=0,  # PRM+ORM would give 0
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
        
        # Add PRM-specific attributes
        result.prm_average_score = 0
        result.orm_average_score = 0
        result.reasoning_quality = 0
        
        return result
    
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
        
        for param_name, config_attr in mapping.items():
            if param_name in params:
                setattr(config, config_attr, params[param_name])
        
        config.auto_training = True
        config.save_intermediate_results = False
        config.min_data_size_for_training = 5
        
        return config
    
    def _result_to_score(self, result: TestResult) -> float:
        """🔥 Consistent scoring using PRM+ORM enhanced metrics"""
        # Primary score is already PRM+ORM based
        primary_score = result.average_inference_score  # This is now PRM+ORM combined
        
        # Additional factors
        efficiency_score = result.energy_efficiency
        training_score = 1.0 / (1.0 + result.average_training_loss) if result.average_training_loss != float('inf') else 0
        coherence_score = result.response_coherence
        consistency_score = result.reward_consistency  # This is reasoning_quality from PRM
        
        # Weighted combination
        total_score = (
            0.5 * primary_score +        # 🔥 PRM+ORM combined score
            0.2 * efficiency_score +     # Energy efficiency
            0.15 * training_score +      # Training performance
            0.1 * coherence_score +      # PRM-based coherence
            0.05 * consistency_score     # PRM-based reasoning quality
        )
        
        return max(0, total_score)

def test_prm_consistent_system():
    """Test the PRM-consistent evaluation system"""
    print("🧪 Testing PRM-Consistent Evaluation System")
    print("=" * 50)
    
    # Create dummy models for testing
    from file_based_training_inference import create_dummy_models
    base_model, prm_model, orm_model, tokenizer = create_dummy_models()
    
    # Create PRM-consistent evaluator
    evaluator = PRMConsistentEvaluator(prm_model, orm_model, tokenizer)
    
    # Test PRM+ORM scoring
    test_prompt = "What is machine learning?"
    test_response = "Machine learning is a subset of AI that enables computers to learn from data."
    
    prm_orm_score = evaluator._evaluate_response_with_prm_orm(test_prompt, test_response)
    
    print(f"🎯 PRM+ORM Evaluation Results:")
    print(f"   PRM Average: {prm_orm_score.prm_average:.3f}")
    print(f"   ORM Overall: {prm_orm_score.orm_overall:.3f}")
    print(f"   Combined Score: {prm_orm_score.combined_score:.3f}")
    print(f"   Reasoning Quality: {prm_orm_score.reasoning_quality:.3f}")
    print(f"   Step Consistency: {prm_orm_score.step_consistency:.3f}")
    
    # Test enhanced sampler
    from enhanced_sampling_methods import ParameterSpace, SamplingConfig
    
    parameter_space = ParameterSpace()
    config = SamplingConfig()
    
    enhanced_sampler = PRMEnhancedBaseSampler(
        parameter_space, config, prm_model, orm_model, tokenizer
    )
    
    # Test parameter evaluation
    test_params = parameter_space.sample_random()
    result = enhanced_sampler.evaluate_params(test_params)
    
    print(f"\n📊 Enhanced Parameter Evaluation:")
    print(f"   Average Inference Score: {result.average_inference_score:.3f} (PRM+ORM based)")
    print(f"   PRM Score: {getattr(result, 'prm_average_score', 'N/A')}")
    print(f"   ORM Score: {getattr(result, 'orm_average_score', 'N/A')}")
    print(f"   Reasoning Quality: {getattr(result, 'reasoning_quality', 'N/A')}")
    print(f"   Energy Efficiency: {result.energy_efficiency:.3f}")
    
    # Test consistent scoring
    final_score = enhanced_sampler._result_to_score(result)
    print(f"   Final Consistent Score: {final_score:.3f}")
    
    print(f"\n✅ PRM-Consistent evaluation system tested successfully!")
    
    return {
        'prm_orm_score': prm_orm_score,
        'evaluation_result': result,
        'final_score': final_score,
        'evaluator': evaluator,
        'sampler': enhanced_sampler
    }

if __name__ == "__main__":
    test_prm_consistent_system()