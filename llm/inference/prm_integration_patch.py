# prm_integration_patch.py
"""
Integration patch to make all existing samplers use PRM+ORM consistently
This patches the existing sampler classes to use the new PRM-consistent evaluation
"""
import logging
from typing import Dict, Any
from prm_consistent_evaluation import PRMEnhancedBaseSampler, PRMConsistentEvaluator

logger = logging.getLogger(__name__)

def patch_sampler_with_prm(sampler_class):
    """Decorator to patch existing samplers with PRM+ORM evaluation"""
    
    class PRMPatchedSampler(sampler_class):
        def __init__(self, parameter_space, config, **kwargs):
            super().__init__(parameter_space, config)
            
            # Add PRM enhancement
            self.prm_enhanced_base = PRMEnhancedBaseSampler(
                parameter_space, config, **kwargs
            )
            
            logger.info(f"🔧 Patched {sampler_class.__name__} with PRM+ORM consistency")
        
        def evaluate_params(self, params: Dict[str, Any]):
            """Use PRM-consistent evaluation"""
            return self.prm_enhanced_base.evaluate_params(params)
        
        def _result_to_score(self, result):
            """Use PRM-consistent scoring"""
            return self.prm_enhanced_base._result_to_score(result)
    
    PRMPatchedSampler.__name__ = f"PRM{sampler_class.__name__}"
    return PRMPatchedSampler

# Patch existing samplers
def apply_prm_patches():
    """Apply PRM patches to all existing samplers"""
    
    # Import existing samplers
    from sampling_serial_backtrack import SerialSampler, BacktrackingSampler
    from sampling_bayesian_genetic import BayesianOptimizationSampler, GeneticAlgorithmSampler
    
    # Create patched versions
    global PRMSerialSampler, PRMBacktrackingSampler
    global PRMBayesianOptimizationSampler, PRMGeneticAlgorithmSampler
    
    PRMSerialSampler = patch_sampler_with_prm(SerialSampler)
    PRMBacktrackingSampler = patch_sampler_with_prm(BacktrackingSampler)
    PRMBayesianOptimizationSampler = patch_sampler_with_prm(BayesianOptimizationSampler)
    PRMGeneticAlgorithmSampler = patch_sampler_with_prm(GeneticAlgorithmSampler)
    
    logger.info("✅ All samplers patched with PRM+ORM consistency")
    
    return {
        'PRMSerialSampler': PRMSerialSampler,
        'PRMBacktrackingSampler': PRMBacktrackingSampler,
        'PRMBayesianOptimizationSampler': PRMBayesianOptimizationSampler,
        'PRMGeneticAlgorithmSampler': PRMGeneticAlgorithmSampler
    }

class PRMConsistentOrchestrator:
    """Updated orchestrator that uses PRM-consistent samplers"""
    
    def __init__(self, sampling_config=None):
        from enhanced_sampling_methods import SamplingConfig, ParameterSpace
        
        self.config = sampling_config or SamplingConfig()
        self.parameter_space = ParameterSpace()
        
        # Apply patches
        patched_samplers = apply_prm_patches()
        
        # Create PRM-consistent models
        from file_based_training_inference import create_dummy_models
        base_model, prm_model, orm_model, tokenizer = create_dummy_models()
        
        # Initialize PRM-consistent samplers
        self.active_samplers = {
            'prm_serial': patched_samplers['PRMSerialSampler'](
                self.parameter_space, self.config, 
                prm_model=prm_model, orm_model=orm_model, tokenizer=tokenizer
            ),
            'prm_backtracking': patched_samplers['PRMBacktrackingSampler'](
                self.parameter_space, self.config,
                prm_model=prm_model, orm_model=orm_model, tokenizer=tokenizer
            ),
            'prm_bayesian': patched_samplers['PRMBayesianOptimizationSampler'](
                self.parameter_space, self.config,
                prm_model=prm_model, orm_model=orm_model, tokenizer=tokenizer
            ),
            'prm_genetic': patched_samplers['PRMGeneticAlgorithmSampler'](
                self.parameter_space, self.config,
                prm_model=prm_model, orm_model=orm_model, tokenizer=tokenizer
            )
        }
        
        # Performance tracking for PRM-enhanced samplers
        self.sampler_performance = {}
        for name in self.active_samplers.keys():
            self.sampler_performance[name] = {
                'weight': 1.0 / len(self.active_samplers),
                'prm_scores': [],
                'orm_scores': [],
                'combined_scores': [],
                'reasoning_quality_scores': [],
                'total_evaluations': 0,
                'successful_evaluations': 0
            }
        
        self.global_history = []
        self.best_global_result = None
        
        logger.info("🧠 Initialized PRM-Consistent Orchestrator")
    
    def run_prm_adaptive_sampling(self, max_evaluations: int = 200) -> Dict[str, Any]:
        """Run adaptive sampling with PRM+ORM consistency"""
        logger.info("🚀 Starting PRM-Consistent Adaptive Sampling")
        
        evaluation_count = 0
        adaptation_interval = 25
        
        while evaluation_count < max_evaluations:
            # Select sampler based on PRM+ORM performance
            sampler_name = self._select_prm_sampler()
            sampler = self.active_samplers[sampler_name]
            
            # Sample and evaluate with PRM+ORM
            params = sampler.sample_next()
            result = sampler.evaluate_params(params)
            
            # Update with PRM+ORM specific metrics
            sampler.update(params, result)
            self._update_prm_global_state(sampler_name, result)
            
            evaluation_count += 1
            
            # Adapt based on PRM+ORM performance
            if evaluation_count % adaptation_interval == 0:
                self._adapt_prm_sampler_weights()
                self._log_prm_performance_summary()
            
            if evaluation_count % 50 == 0:
                logger.info(f"PRM Progress: {evaluation_count}/{max_evaluations}")
        
        # Generate PRM-enhanced analysis
        final_analysis = self._generate_prm_analysis()
        
        return {
            'global_history': self.global_history,
            'best_result': self.best_global_result,
            'sampler_performance': self.sampler_performance,
            'final_analysis': final_analysis,
            'total_evaluations': evaluation_count,
            'consistency_type': 'PRM+ORM Enhanced'
        }
    
    def _select_prm_sampler(self) -> str:
        """Select sampler based on PRM+ORM performance"""
        import numpy as np
        
        names = list(self.sampler_performance.keys())
        weights = []
        
        for name in names:
            perf = self.sampler_performance[name]
            
            # Weight based on multiple PRM+ORM factors
            if perf['combined_scores']:
                combined_avg = np.mean(perf['combined_scores'])
                reasoning_avg = np.mean(perf['reasoning_quality_scores']) if perf['reasoning_quality_scores'] else 0
                success_rate = perf['successful_evaluations'] / max(perf['total_evaluations'], 1)
                
                # Enhanced weighting with PRM reasoning quality
                weight = (
                    0.4 * combined_avg +           # PRM+ORM combined performance
                    0.3 * reasoning_avg +          # PRM reasoning quality
                    0.2 * success_rate +           # Success rate
                    0.1 * perf['weight']           # Previous weight
                )
            else:
                weight = perf['weight']
            
            weights.append(weight)
        
        # Normalize and select
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            return np.random.choice(names, p=weights)
        else:
            return np.random.choice(names)
    
    def _update_prm_global_state(self, sampler_name: str, result):
        """Update global state with PRM+ORM enhanced metrics"""
        # Calculate PRM+ORM score
        prm_enhanced_score = result.average_inference_score  # Already PRM+ORM combined
        
        # Update global history
        self.global_history.append((sampler_name, result))
        
        # Update best result based on PRM+ORM score
        if (self.best_global_result is None or 
            prm_enhanced_score > self.best_global_result.average_inference_score):
            self.best_global_result = result
            logger.info(f"🔥 New best PRM+ORM result from {sampler_name}: {prm_enhanced_score:.3f}")
        
        # Update sampler-specific PRM performance
        perf = self.sampler_performance[sampler_name]
        perf['total_evaluations'] += 1
        perf['combined_scores'].append(prm_enhanced_score)
        
        # Extract PRM-specific metrics if available
        if hasattr(result, 'prm_average_score'):
            perf['prm_scores'].append(result.prm_average_score)
        if hasattr(result, 'orm_average_score'):
            perf['orm_scores'].append(result.orm_average_score)
        if hasattr(result, 'reasoning_quality'):
            perf['reasoning_quality_scores'].append(result.reasoning_quality)
        
        if prm_enhanced_score > 0.3:  # PRM+ORM success threshold
            perf['successful_evaluations'] += 1
    
    def _adapt_prm_sampler_weights(self):
        """Adapt sampler weights based on PRM+ORM performance"""
        import numpy as np
        
        for name, perf in self.sampler_performance.items():
            if perf['combined_scores']:
                # Calculate PRM+ORM specific metrics
                combined_avg = np.mean(perf['combined_scores'])
                combined_improvement = (
                    perf['combined_scores'][-1] - perf['combined_scores'][0]
                    if len(perf['combined_scores']) > 1 else 0
                )
                
                reasoning_quality = (
                    np.mean(perf['reasoning_quality_scores'][-5:])
                    if perf['reasoning_quality_scores'] else 0
                )
                
                success_rate = perf['successful_evaluations'] / max(perf['total_evaluations'], 1)
                
                # Enhanced performance score considering PRM reasoning
                performance_score = (
                    0.3 * combined_avg +
                    0.25 * max(0, combined_improvement) +
                    0.25 * reasoning_quality +           # 🔥 PRM reasoning bonus
                    0.2 * success_rate
                )
                
                # Exponential moving average
                alpha = 0.4
                perf['weight'] = alpha * performance_score + (1 - alpha) * perf['weight']
        
        # Normalize weights
        total_weight = sum(perf['weight'] for perf in self.sampler_performance.values())
        if total_weight > 0:
            for perf in self.sampler_performance.values():
                perf['weight'] /= total_weight
    
    def _log_prm_performance_summary(self):
        """Log PRM+ORM enhanced performance summary"""
        import numpy as np
        
        logger.info("📊 PRM+ORM Enhanced Performance Summary:")
        for name, perf in self.sampler_performance.items():
            combined_avg = np.mean(perf['combined_scores']) if perf['combined_scores'] else 0
            reasoning_avg = np.mean(perf['reasoning_quality_scores']) if perf['reasoning_quality_scores'] else 0
            prm_avg = np.mean(perf['prm_scores']) if perf['prm_scores'] else 0
            orm_avg = np.mean(perf['orm_scores']) if perf['orm_scores'] else 0
            
            logger.info(f"  {name}: weight={perf['weight']:.3f}, "
                       f"combined={combined_avg:.3f}, reasoning={reasoning_avg:.3f}, "
                       f"prm={prm_avg:.3f}, orm={orm_avg:.3f}")
    
    def _generate_prm_analysis(self) -> Dict[str, Any]:
        """Generate PRM+ORM enhanced final analysis"""
        import numpy as np
        from collections import defaultdict
        
        analysis = {
            'prm_orm_statistics': {},
            'reasoning_analysis': {},
            'sampler_comparison': {},
            'prm_insights': []
        }
        
        # PRM+ORM statistics by sampler
        for name, perf in self.sampler_performance.items():
            if perf['combined_scores']:
                analysis['prm_orm_statistics'][name] = {
                    'avg_combined_score': np.mean(perf['combined_scores']),
                    'avg_prm_score': np.mean(perf['prm_scores']) if perf['prm_scores'] else 0,
                    'avg_orm_score': np.mean(perf['orm_scores']) if perf['orm_scores'] else 0,
                    'avg_reasoning_quality': np.mean(perf['reasoning_quality_scores']) if perf['reasoning_quality_scores'] else 0,
                    'prm_orm_consistency': self._calculate_prm_orm_consistency(perf),
                    'total_evaluations': perf['total_evaluations']
                }
        
        # Reasoning analysis
        if self.global_history:
            all_reasoning_scores = []
            for _, result in self.global_history:
                if hasattr(result, 'reasoning_quality'):
                    all_reasoning_scores.append(result.reasoning_quality)
            
            if all_reasoning_scores:
                analysis['reasoning_analysis'] = {
                    'avg_reasoning_quality': np.mean(all_reasoning_scores),
                    'reasoning_improvement': all_reasoning_scores[-1] - all_reasoning_scores[0] if len(all_reasoning_scores) > 1 else 0,
                    'best_reasoning_score': max(all_reasoning_scores),
                    'reasoning_consistency': 1.0 - np.var(all_reasoning_scores)
                }
        
        # Sampler comparison
        best_prm_sampler = max(
            self.sampler_performance.items(),
            key=lambda x: np.mean(x[1]['prm_scores']) if x[1]['prm_scores'] else 0
        )[0]
        
        best_orm_sampler = max(
            self.sampler_performance.items(),
            key=lambda x: np.mean(x[1]['orm_scores']) if x[1]['orm_scores'] else 0
        )[0]
        
        best_reasoning_sampler = max(
            self.sampler_performance.items(),
            key=lambda x: np.mean(x[1]['reasoning_quality_scores']) if x[1]['reasoning_quality_scores'] else 0
        )[0]
        
        analysis['sampler_comparison'] = {
            'best_prm_sampler': best_prm_sampler,
            'best_orm_sampler': best_orm_sampler,
            'best_reasoning_sampler': best_reasoning_sampler,
            'most_consistent': self._find_most_consistent_sampler()
        }
        
        # PRM insights
        analysis['prm_insights'] = self._generate_prm_insights()
        
        return analysis
    
    def _calculate_prm_orm_consistency(self, perf: Dict) -> float:
        """Calculate consistency between PRM and ORM scores"""
        if not perf['prm_scores'] or not perf['orm_scores']:
            return 0.0
        
        # Use correlation between PRM and ORM scores
        import numpy as np
        if len(perf['prm_scores']) > 1 and len(perf['orm_scores']) > 1:
            min_len = min(len(perf['prm_scores']), len(perf['orm_scores']))
            prm_subset = perf['prm_scores'][:min_len]
            orm_subset = perf['orm_scores'][:min_len]
            
            correlation = np.corrcoef(prm_subset, orm_subset)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _find_most_consistent_sampler(self) -> str:
        """Find sampler with most consistent PRM+ORM performance"""
        import numpy as np
        
        best_consistency = 0
        most_consistent = list(self.sampler_performance.keys())[0]
        
        for name, perf in self.sampler_performance.items():
            if perf['combined_scores'] and len(perf['combined_scores']) > 1:
                # Low variance = high consistency
                consistency = 1.0 / (1.0 + np.var(perf['combined_scores']))
                if consistency > best_consistency:
                    best_consistency = consistency
                    most_consistent = name
        
        return most_consistent
    
    def _generate_prm_insights(self) -> List[str]:
        """Generate insights based on PRM+ORM analysis"""
        import numpy as np
        
        insights = []
        
        # Analyze PRM vs ORM performance
        all_prm_scores = []
        all_orm_scores = []
        
        for perf in self.sampler_performance.values():
            all_prm_scores.extend(perf['prm_scores'])
            all_orm_scores.extend(perf['orm_scores'])
        
        if all_prm_scores and all_orm_scores:
            avg_prm = np.mean(all_prm_scores)
            avg_orm = np.mean(all_orm_scores)
            
            if avg_prm > avg_orm + 0.1:
                insights.append("Process reasoning (PRM) consistently outperforms outcome evaluation (ORM)")
            elif avg_orm > avg_prm + 0.1:
                insights.append("Outcome evaluation (ORM) consistently outperforms process reasoning (PRM)")
            else:
                insights.append("PRM and ORM show balanced performance - good reasoning and outcomes")
        
        # Reasoning quality insights
        all_reasoning = []
        for perf in self.sampler_performance.values():
            all_reasoning.extend(perf['reasoning_quality_scores'])
        
        if all_reasoning:
            if np.mean(all_reasoning) > 0.7:
                insights.append("High reasoning quality achieved - step-by-step thinking is strong")
            elif np.mean(all_reasoning) < 0.4:
                insights.append("Low reasoning quality - consider improving step-by-step evaluation")
        
        # Sampler-specific insights
        best_sampler_name = max(
            self.sampler_performance.items(),
            key=lambda x: np.mean(x[1]['combined_scores']) if x[1]['combined_scores'] else 0
        )[0]
        
        insights.append(f"Best performing method: {best_sampler_name.replace('prm_', '').title()}")
        
        return insights

def test_prm_integration():
    """Test the complete PRM integration"""
    print("🔧 Testing Complete PRM Integration")
    print("=" * 50)
    
    # Test patching
    patched_samplers = apply_prm_patches()
    print(f"✅ Patched {len(patched_samplers)} sampler classes")
    
    # Test PRM-consistent orchestrator
    orchestrator = PRMConsistentOrchestrator()
    print(f"✅ Created PRM-consistent orchestrator with {len(orchestrator.active_samplers)} samplers")
    
    # Run short test
    results = orchestrator.run_prm_adaptive_sampling(max_evaluations=20)
    
    print(f"\n📊 PRM Integration Test Results:")
    print(f"   Total evaluations: {results['total_evaluations']}")
    print(f"   Consistency type: {results['consistency_type']}")
    
    if results['best_result']:
        print(f"   Best PRM+ORM score: {results['best_result'].average_inference_score:.3f}")
        if hasattr(results['best_result'], 'prm_average_score'):
            print(f"   Best PRM score: {results['best_result'].prm_average_score:.3f}")
        if hasattr(results['best_result'], 'orm_average_score'):
            print(f"   Best ORM score: {results['best_result'].orm_average_score:.3f}")
    
    # Print PRM insights
    if 'prm_insights' in results['final_analysis']:
        print(f"\n💡 PRM Insights:")
        for insight in results['final_analysis']['prm_insights']:
            print(f"   • {insight}")
    
    print(f"\n🎉 PRM integration test completed successfully!")
    
    return results

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_prm_integration()