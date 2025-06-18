# enhanced_online_training.py
"""
Enhanced File-Based Online Training System with Advanced Sampling
Integrates PRM-consistent evaluation and adaptive sampling into online training
"""
import torch
import torch.nn as nn
import json
import threading
import time
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import logging
from dataclasses import dataclass, asdict
from collections import deque
import random

# Import enhanced sampling components
from prm_consistent_evaluation import PRMConsistentEvaluator, PRMORMScore
from prm_integration_patch import PRMConsistentOrchestrator, apply_prm_patches
from enhanced_sampling_methods import SamplingConfig, ParameterSpace
from sampling_orchestrator_utils import AdaptiveSamplingOrchestrator

# Import original components
from file_based_training_inference import (
    FileBasedConfig, DataFileProcessor, FileWatcher,
    create_dummy_models, create_sample_data_files
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedOnlineConfig(FileBasedConfig):
    """Enhanced configuration with adaptive sampling support"""
    
    # Enhanced sampling settings
    enable_adaptive_sampling: bool = True
    sampling_adaptation_interval: int = 50  # evaluations
    parameter_exploration_rate: float = 0.15
    
    # PRM+ORM enhanced settings
    prm_orm_evaluation_frequency: int = 10  # every N inferences
    prm_weight_adaptation: bool = True
    orm_weight_adaptation: bool = True
    
    # Online optimization settings
    online_parameter_optimization: bool = True
    optimization_trigger_threshold: float = 0.1  # performance drop threshold
    optimization_cooldown: int = 100  # minimum evaluations between optimizations
    
    # Advanced training settings
    dynamic_batch_sizing: bool = True
    adaptive_learning_rate: bool = True
    quality_based_sampling: bool = True
    
    # Memory and performance
    evaluation_cache_size: int = 1000
    performance_history_size: int = 200

class EnhancedDataProcessor(DataFileProcessor):
    """Enhanced data processor with PRM+ORM quality assessment"""
    
    def __init__(self, config: EnhancedOnlineConfig, 
                 prm_model=None, orm_model=None, tokenizer=None):
        super().__init__(config)
        self.enhanced_config = config
        self.prm_evaluator = PRMConsistentEvaluator(prm_model, orm_model, tokenizer)
        
    def _enhance_training_data(self, data: List[Dict]) -> List[Dict]:
        """Enhanced data processing with PRM+ORM evaluation"""
        enhanced_data = []
        
        for item in data:
            # Original enhancement
            if 'reward' not in item and 'response' in item:
                if self.prm_evaluator.prm_model and self.prm_evaluator.orm_model:
                    # Use PRM+ORM for reward estimation
                    prm_orm_score = self.prm_evaluator._evaluate_response_with_prm_orm(
                        item['prompt'], item['response']
                    )
                    item['reward'] = prm_orm_score.combined_score
                    item['prm_score'] = prm_orm_score.prm_average
                    item['orm_score'] = prm_orm_score.orm_overall
                    item['reasoning_quality'] = prm_orm_score.reasoning_quality
                else:
                    item['reward'] = 0.5  # Default
            
            # Enhanced quality assessment
            item['quality_score'] = self._assess_enhanced_quality(item)
            enhanced_data.append(item)
        
        # Sort by enhanced quality
        enhanced_data.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Apply cache limits
        if len(enhanced_data) > self.enhanced_config.max_cache_size:
            enhanced_data = enhanced_data[:self.enhanced_config.max_cache_size]
        
        return enhanced_data
    
    def _assess_enhanced_quality(self, item: Dict) -> float:
        """Enhanced quality assessment using PRM+ORM insights"""
        quality_score = 0.0
        
        # Base quality (length-based)
        if 'prompt' in item:
            prompt_len = len(item['prompt'].split())
            quality_score += min(prompt_len / 50, 1.0) * 0.15
        
        if 'response' in item:
            response_len = len(item['response'].split())
            quality_score += min(response_len / 100, 1.0) * 0.15
        
        # Enhanced reward-based quality
        if 'reward' in item:
            quality_score += max(0, item['reward']) * 0.4
        
        # PRM-specific quality factors
        if 'reasoning_quality' in item:
            quality_score += item['reasoning_quality'] * 0.2
        
        # ORM-specific quality factors  
        if 'orm_score' in item:
            quality_score += max(0, item['orm_score']) * 0.1
        
        return min(quality_score, 1.0)

class AdaptiveParameterManager:
    """Manages adaptive parameter optimization during online training"""
    
    def __init__(self, config: EnhancedOnlineConfig):
        self.config = config
        self.parameter_space = ParameterSpace()
        self.orchestrator = None
        self.current_params = self._get_initial_params()
        self.performance_history = deque(maxlen=config.performance_history_size)
        self.last_optimization = 0
        self.optimization_results = {}
        
    def _get_initial_params(self) -> Dict[str, Any]:
        """Get initial parameters from config"""
        return {
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'num_candidates': self.config.num_candidates,
            'prm_weight': self.config.prm_weight,
            'orm_weight': self.config.orm_weight,
            'max_sequence_length': self.config.max_sequence_length,
            'training_epochs_per_batch': self.config.training_epochs_per_batch
        }
    
    def should_optimize_parameters(self, current_performance: float, 
                                 evaluation_count: int) -> bool:
        """Determine if parameter optimization should be triggered"""
        if not self.config.online_parameter_optimization:
            return False
        
        # Check cooldown
        if evaluation_count - self.last_optimization < self.config.optimization_cooldown:
            return False
        
        # Check if we have enough performance history
        if len(self.performance_history) < 20:
            return False
        
        # Check for performance drop
        recent_avg = np.mean(list(self.performance_history)[-10:])
        older_avg = np.mean(list(self.performance_history)[-20:-10])
        
        performance_drop = older_avg - recent_avg
        
        if performance_drop > self.config.optimization_trigger_threshold:
            logger.info(f"🔄 Performance drop detected: {performance_drop:.3f}, triggering optimization")
            return True
        
        return False
    
    def optimize_parameters(self, system, evaluation_count: int) -> Dict[str, Any]:
        """Run parameter optimization"""
        logger.info("🎯 Starting adaptive parameter optimization")
        
        # Initialize orchestrator if needed
        if self.orchestrator is None:
            sampling_config = SamplingConfig(
                max_iterations=30,
                exploration_rate=self.config.parameter_exploration_rate,
                adaptation_frequency=10
            )
            self.orchestrator = PRMConsistentOrchestrator(sampling_config)
        
        # Run optimization with limited evaluations (online context)
        optimization_results = self.orchestrator.run_prm_adaptive_sampling(max_evaluations=25)
        
        # Extract best parameters
        if optimization_results['best_result']:
            new_params = optimization_results['best_result'].parameters
            old_performance = np.mean(list(self.performance_history)[-5:])
            
            # Apply new parameters to system
            self._apply_parameters_to_system(system, new_params)
            
            # Update tracking
            self.current_params = new_params
            self.last_optimization = evaluation_count
            self.optimization_results = optimization_results
            
            logger.info(f"✅ Parameters optimized. Previous performance: {old_performance:.3f}")
            
            return {
                'success': True,
                'new_parameters': new_params,
                'optimization_results': optimization_results,
                'previous_performance': old_performance
            }
        else:
            logger.warning("❌ Parameter optimization failed")
            return {'success': False}
    
    def _apply_parameters_to_system(self, system, new_params: Dict[str, Any]):
        """Apply optimized parameters to the training system"""
        # Update config
        if 'learning_rate' in new_params:
            system.config.learning_rate = new_params['learning_rate']
            # Update optimizer
            for param_group in system.optimizer.param_groups:
                param_group['lr'] = new_params['learning_rate']
        
        if 'batch_size' in new_params:
            system.config.batch_size = new_params['batch_size']
        
        if 'temperature' in new_params:
            system.config.temperature = new_params['temperature']
        
        if 'top_p' in new_params:
            system.config.top_p = new_params['top_p']
        
        if 'prm_weight' in new_params:
            system.config.prm_weight = new_params['prm_weight']
        
        if 'orm_weight' in new_params:
            system.config.orm_weight = new_params['orm_weight']
        
        logger.info(f"🔧 Applied new parameters to system")
    
    def update_performance(self, performance: float):
        """Update performance history"""
        self.performance_history.append(performance)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'current_parameters': self.current_params,
            'performance_history_size': len(self.performance_history),
            'last_optimization': self.last_optimization,
            'recent_performance': np.mean(list(self.performance_history)[-5:]) if self.performance_history else 0,
            'optimization_results': self.optimization_results
        }

class EnhancedFileBasedTrainingSystem:
    """Enhanced file-based training system with adaptive sampling and PRM+ORM"""
    
    def __init__(self, base_model, prm_model, orm_model, tokenizer, 
                 config: EnhancedOnlineConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        # Models
        self.model = base_model
        self.prm_model = prm_model
        self.orm_model = orm_model
        self.device = next(base_model.parameters()).device
        
        # Enhanced components
        self.data_processor = EnhancedDataProcessor(config, prm_model, orm_model, tokenizer)
        self.prm_evaluator = PRMConsistentEvaluator(prm_model, orm_model, tokenizer)
        self.param_manager = AdaptiveParameterManager(config)
        
        # Optimizer (will be updated by param_manager)
        self.optimizer = torch.optim.AdamW(base_model.parameters(), lr=config.learning_rate)
        
        # Data storage
        self.training_data = []
        self.pending_inference_requests = queue.Queue()
        self.processed_results = []
        
        # Control flags
        self.running = False
        self.training_thread = None
        self.inference_thread = None
        self.file_watcher_thread = None
        self.optimization_thread = None
        
        # Enhanced statistics
        self.stats = {
            'start_time': time.time(),
            'total_training_batches': 0,
            'total_inferences': 0,
            'total_prm_orm_evaluations': 0,
            'total_parameter_optimizations': 0,
            'data_reload_count': 0,
            'last_training_time': 0,
            'average_training_loss': deque(maxlen=100),
            'average_inference_score': deque(maxlen=100),
            'prm_scores': deque(maxlen=100),
            'orm_scores': deque(maxlen=100),
            'reasoning_quality_scores': deque(maxlen=100),
            'parameter_optimization_history': []
        }
        
        # Create output directories
        Path(config.results_output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start the enhanced system"""
        self.running = True
        
        # Initial data load
        self._load_training_data()
        
        # Start file watcher
        self._start_file_watcher()
        
        # Start enhanced training thread
        if self.config.auto_training:
            self.training_thread = threading.Thread(target=self._enhanced_training_loop, daemon=True)
            self.training_thread.start()
        
        # Start enhanced inference thread
        self.inference_thread = threading.Thread(target=self._enhanced_inference_loop, daemon=True)
        self.inference_thread.start()
        
        # Start parameter optimization thread
        if self.config.online_parameter_optimization:
            self.optimization_thread = threading.Thread(target=self._optimization_monitoring_loop, daemon=True)
            self.optimization_thread.start()
        
        logger.info("🚀 Enhanced File-Based Training System Started")
        logger.info(f"🧠 Adaptive sampling: {'Enabled' if self.config.enable_adaptive_sampling else 'Disabled'}")
        logger.info(f"🎯 Parameter optimization: {'Enabled' if self.config.online_parameter_optimization else 'Disabled'}")
        logger.info(f"📁 Watching directory: {self.config.training_data_dir}")
    
    def stop(self):
        """Stop the enhanced system"""
        self.running = False
        
        # Stop all threads
        threads = [self.training_thread, self.inference_thread, 
                  self.file_watcher_thread, self.optimization_thread]
        
        for thread in threads:
            if thread:
                thread.join(timeout=5)
        
        logger.info("🛑 Enhanced File-Based System Stopped")
    
    def _enhanced_training_loop(self):
        """Enhanced training loop with adaptive capabilities"""
        logger.info("🎓 Enhanced training loop started")
        
        while self.running:
            try:
                if len(self.training_data) >= self.config.min_data_size_for_training:
                    self._enhanced_train_on_data()
                    self.stats['last_training_time'] = time.time()
                
                # Dynamic wait time based on performance
                if self.config.dynamic_batch_sizing:
                    wait_time = self._calculate_adaptive_wait_time()
                else:
                    wait_time = self.config.data_refresh_interval
                
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Enhanced training loop error: {e}")
                time.sleep(5)
    
    def _enhanced_train_on_data(self):
        """Enhanced training with adaptive batch sizing and PRM+ORM guidance"""
        # Dynamic batch size calculation
        if self.config.dynamic_batch_sizing:
            batch_size = self._calculate_adaptive_batch_size()
        else:
            batch_size = min(self.config.batch_size, len(self.training_data))
        
        # Enhanced quality-based sampling
        if self.config.quality_based_sampling:
            batch_data = self._sample_high_quality_batch(batch_size)
        else:
            # Standard weighted sampling
            weights = [item['quality_score'] for item in self.training_data]
            total_weight = sum(weights)
            
            if total_weight == 0:
                batch_data = random.sample(self.training_data, batch_size)
            else:
                probabilities = [w / total_weight for w in weights]
                batch_indices = np.random.choice(
                    len(self.training_data), 
                    size=batch_size, 
                    p=probabilities, 
                    replace=False
                )
                batch_data = [self.training_data[i] for i in batch_indices]
        
        # Enhanced training with PRM+ORM insights
        loss = self._train_batch_with_prm_orm(batch_data, batch_size)
        
        # Update statistics
        self.stats['average_training_loss'].append(loss)
        self.stats['total_training_batches'] += 1
        
        # Update parameter manager
        current_performance = self._calculate_current_performance()
        self.param_manager.update_performance(current_performance)
        
        logger.info(f"🎓 Enhanced training batch: Loss={loss:.4f}, Batch size={len(batch_data)}, Performance={current_performance:.3f}")
        
        # Adaptive checkpoint saving
        if self._should_save_checkpoint():
            self._save_enhanced_checkpoint()
    
    def _sample_high_quality_batch(self, batch_size: int) -> List[Dict]:
        """Sample batch with focus on high-quality examples"""
        # Separate data by quality tiers
        high_quality = [item for item in self.training_data if item['quality_score'] > 0.7]
        medium_quality = [item for item in self.training_data if 0.4 <= item['quality_score'] <= 0.7]
        low_quality = [item for item in self.training_data if item['quality_score'] < 0.4]
        
        # Adaptive sampling strategy
        high_ratio = 0.6 if len(high_quality) > batch_size * 0.3 else 0.8
        medium_ratio = 0.3
        low_ratio = 0.1
        
        # Calculate actual counts
        high_count = min(int(batch_size * high_ratio), len(high_quality))
        medium_count = min(int(batch_size * medium_ratio), len(medium_quality))
        low_count = min(batch_size - high_count - medium_count, len(low_quality))
        
        # Sample from each tier
        batch_data = []
        if high_count > 0:
            batch_data.extend(random.sample(high_quality, high_count))
        if medium_count > 0:
            batch_data.extend(random.sample(medium_quality, medium_count))
        if low_count > 0:
            batch_data.extend(random.sample(low_quality, low_count))
        
        # Fill remaining with random samples if needed
        while len(batch_data) < batch_size and len(batch_data) < len(self.training_data):
            remaining = [item for item in self.training_data if item not in batch_data]
            if remaining:
                batch_data.append(random.choice(remaining))
            else:
                break
        
        return batch_data
    
    def _train_batch_with_prm_orm(self, batch_data: List[Dict], batch_size: int) -> float:
        """Train batch with PRM+ORM enhanced loss"""
        # Create enhanced dataset
        class EnhancedBatchDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                
                if 'response' in item:
                    text = f"{item['prompt']} {item['response']}"
                else:
                    text = item['prompt']
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'reward': torch.tensor(item.get('reward', 0.0), dtype=torch.float),
                    'prm_score': torch.tensor(item.get('prm_score', 0.0), dtype=torch.float),
                    'orm_score': torch.tensor(item.get('orm_score', 0.0), dtype=torch.float),
                    'quality_score': torch.tensor(item.get('quality_score', 0.0), dtype=torch.float)
                }
        
        dataset = EnhancedBatchDataset(batch_data, self.tokenizer, self.config.max_sequence_length)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(self.config.training_epochs_per_batch):
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                
                # Enhanced loss calculation
                base_loss = outputs.loss
                
                # Multi-factor weighting
                reward_weights = torch.sigmoid(batch['reward'])
                quality_weights = batch['quality_score']
                prm_weights = torch.sigmoid(batch['prm_score']) 
                orm_weights = torch.sigmoid(batch['orm_score'])
                
                # Combined weighting
                combined_weights = (
                    0.4 * reward_weights +
                    0.3 * quality_weights + 
                    0.2 * prm_weights +
                    0.1 * orm_weights
                )
                
                weighted_loss = base_loss * combined_weights.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += weighted_loss.item()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        return avg_loss
    
    def _enhanced_inference_loop(self):
        """Enhanced inference loop with PRM+ORM evaluation"""
        logger.info("🔮 Enhanced inference loop started")
        
        inference_count = 0
        
        while self.running:
            try:
                # Load new requests
                self._load_inference_requests()
                
                # Process pending requests
                while not self.pending_inference_requests.empty():
                    request = self.pending_inference_requests.get()
                    result = self._enhanced_process_inference_request(request)
                    
                    if result:
                        self.processed_results.append(result)
                        self._save_inference_result(result)
                        
                        # Enhanced evaluation every N inferences
                        inference_count += 1
                        if (inference_count % self.config.prm_orm_evaluation_frequency == 0):
                            self._perform_enhanced_evaluation(result)
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Enhanced inference loop error: {e}")
                time.sleep(5)
    
    def _enhanced_process_inference_request(self, request: Dict) -> Optional[Dict]:
        """Enhanced inference processing with PRM+ORM evaluation"""
        try:
            prompt = request['prompt']
            
            # Generate with enhanced reward guidance
            result = self._generate_with_enhanced_rewards(prompt)
            
            # Add enhanced metadata
            result.update({
                'request_id': request.get('id', f"req_{int(time.time())}"),
                'timestamp': time.time(),
                'metadata': request.get('metadata', {}),
                'evaluation_method': 'PRM+ORM Enhanced'
            })
            
            # Update enhanced statistics
            self.stats['total_inferences'] += 1
            self.stats['average_inference_score'].append(result.get('combined_score', 0.0))
            
            if 'prm_score' in result:
                self.stats['prm_scores'].append(result['prm_score'])
            if 'orm_score' in result:
                self.stats['orm_scores'].append(result['orm_score'])
            if 'reasoning_quality' in result:
                self.stats['reasoning_quality_scores'].append(result['reasoning_quality'])
            
            logger.info(f"🔮 Enhanced inference: {prompt[:50]}... "
                       f"Combined: {result.get('combined_score', 0):.3f}, "
                       f"PRM: {result.get('prm_score', 0):.3f}, "
                       f"ORM: {result.get('orm_score', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced inference processing failed: {e}")
            return None
    
    def _generate_with_enhanced_rewards(self, prompt: str) -> Dict[str, Any]:
        """Enhanced generation with comprehensive PRM+ORM evaluation"""
        self.model.eval()
        
        with torch.no_grad():
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate multiple candidates
            candidates = []
            for _ in range(self.config.num_candidates):
                output_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.config.max_inference_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(
                    output_ids[0][len(prompt_ids[0]):],
                    skip_special_tokens=True
                )
                candidates.append(response)
            
            # Enhanced evaluation of candidates
            best_response = None
            best_score = float('-inf')
            best_details = {}
            
            for response in candidates:
                # Use PRM+ORM evaluator
                prm_orm_score = self.prm_evaluator._evaluate_response_with_prm_orm(prompt, response)
                
                if prm_orm_score.combined_score > best_score:
                    best_score = prm_orm_score.combined_score
                    best_response = response
                    best_details = {
                        'prm_scores': prm_orm_score.prm_step_scores,
                        'prm_score': prm_orm_score.prm_average,
                        'prm_confidence': prm_orm_score.prm_confidence,
                        'orm_score': prm_orm_score.orm_overall,
                        'orm_correctness': prm_orm_score.orm_correctness,
                        'orm_helpfulness': prm_orm_score.orm_helpfulness,
                        'combined_score': prm_orm_score.combined_score,
                        'reasoning_quality': prm_orm_score.reasoning_quality,
                        'step_consistency': prm_orm_score.step_consistency
                    }
        
        return {
            'prompt': prompt,
            'response': best_response,
            'candidates_count': len(candidates),
            **best_details
        }
    
    def _optimization_monitoring_loop(self):
        """Monitor system performance and trigger optimizations"""
        logger.info("🎯 Parameter optimization monitoring started")
        
        evaluation_count = 0
        
        while self.running:
            try:
                time.sleep(self.config.sampling_adaptation_interval)
                evaluation_count += self.config.sampling_adaptation_interval
                
                # Check if optimization should be triggered
                current_performance = self._calculate_current_performance()
                
                if self.param_manager.should_optimize_parameters(current_performance, evaluation_count):
                    # Run parameter optimization
                    optimization_result = self.param_manager.optimize_parameters(self, evaluation_count)
                    
                    if optimization_result['success']:
                        self.stats['total_parameter_optimizations'] += 1
                        self.stats['parameter_optimization_history'].append({
                            'timestamp': time.time(),
                            'evaluation_count': evaluation_count,
                            'previous_performance': optimization_result['previous_performance'],
                            'new_parameters': optimization_result['new_parameters']
                        })
                        
                        logger.info(f"🎉 Parameter optimization #{self.stats['total_parameter_optimizations']} completed")
                
            except Exception as e:
                logger.error(f"Optimization monitoring error: {e}")
                time.sleep(30)
    
    def _calculate_current_performance(self) -> float:
        """Calculate current system performance"""
        if not self.stats['average_inference_score']:
            return 0.0
        
        # Weighted performance calculation
        recent_inference_score = np.mean(list(self.stats['average_inference_score'])[-10:])
        recent_prm_score = np.mean(list(self.stats['prm_scores'])[-10:]) if self.stats['prm_scores'] else 0
        recent_orm_score = np.mean(list(self.stats['orm_scores'])[-10:]) if self.stats['orm_scores'] else 0
        recent_reasoning = np.mean(list(self.stats['reasoning_quality_scores'])[-10:]) if self.stats['reasoning_quality_scores'] else 0
        
        # Combined performance metric
        performance = (
            0.4 * recent_inference_score +
            0.25 * recent_prm_score +
            0.25 * recent_orm_score +
            0.1 * recent_reasoning
        )
        
        return performance
    
    def _calculate_adaptive_batch_size(self) -> int:
        """Calculate adaptive batch size based on performance"""
        base_size = self.config.batch_size
        
        # Adjust based on recent performance
        if self.stats['average_training_loss']:
            recent_losses = list(self.stats['average_training_loss'])[-5:]
            if len(recent_losses) >= 2:
                loss_trend = recent_losses[-1] - recent_losses[0]
                if loss_trend < 0:  # Loss decreasing (good)
                    multiplier = 1.2
                else:  # Loss increasing (bad)
                    multiplier = 0.8
                
                adaptive_size = int(base_size * multiplier)
                # Constrain within reasonable bounds
                adaptive_size = max(4, min(adaptive_size, len(self.training_data), 64))
                return adaptive_size
        
        return min(base_size, len(self.training_data))
    
    def _calculate_adaptive_wait_time(self) -> int:
        """Calculate adaptive wait time between training cycles"""
        base_time = self.config.data_refresh_interval
        
        # Adjust based on system load and performance
        current_performance = self._calculate_current_performance()
        
        if current_performance > 0.7:  # High performance
            return int(base_time * 0.8)  # Train more frequently
        elif current_performance < 0.4:  # Low performance
            return int(base_time * 1.5)  # Train less frequently, give time to improve
        else:
            return base_time
    
    def _should_save_checkpoint(self) -> bool:
        """Determine if checkpoint should be saved"""
        # Save every 10 batches or after parameter optimization
        return (self.stats['total_training_batches'] % 10 == 0 or 
                self.stats['total_parameter_optimizations'] > 0)
    
    def _save_enhanced_checkpoint(self):
        """Save enhanced checkpoint with optimization history"""
        try:
            timestamp = int(time.time())
            checkpoint_path = Path(self.config.checkpoints_dir) / f"enhanced_checkpoint_{timestamp}.pt"
            
            # Enhanced checkpoint data
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'stats': dict(self.stats),
                'config': asdict(self.config),
                'current_parameters': self.param_manager.current_params,
                'optimization_status': self.param_manager.get_optimization_status(),
                'prm_orm_weights': {
                    'prm_weight': self.config.prm_weight,
                    'orm_weight': self.config.orm_weight
                }
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Also save as latest
            latest_path = Path(self.config.checkpoints_dir) / "latest_enhanced_checkpoint.pt"
            torch.save(checkpoint_data, latest_path)
            
            logger.info(f"💾 Enhanced checkpoint saved: {checkpoint_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced checkpoint: {e}")
    
    def load_enhanced_checkpoint(self, checkpoint_path: str):
        """Load enhanced checkpoint with optimization state"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model and optimizer
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load enhanced state
            if 'stats' in checkpoint:
                self.stats.update(checkpoint['stats'])
            
            if 'current_parameters' in checkpoint:
                self.param_manager.current_params = checkpoint['current_parameters']
            
            if 'optimization_status' in checkpoint:
                opt_status = checkpoint['optimization_status']
                if 'performance_history_size' in opt_status:
                    # Restore some performance history context
                    pass
            
            if 'prm_orm_weights' in checkpoint:
                weights = checkpoint['prm_orm_weights']
                self.config.prm_weight = weights.get('prm_weight', self.config.prm_weight)
                self.config.orm_weight = weights.get('orm_weight', self.config.orm_weight)
            
            logger.info(f"📂 Enhanced checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load enhanced checkpoint: {e}")
    
    def _perform_enhanced_evaluation(self, result: Dict):
        """Perform comprehensive system evaluation"""
        self.stats['total_prm_orm_evaluations'] += 1
        
        # Log enhanced metrics periodically
        if self.stats['total_prm_orm_evaluations'] % 50 == 0:
            self._log_enhanced_status()
    
    def _log_enhanced_status(self):
        """Log comprehensive system status"""
        logger.info("📊 Enhanced System Status:")
        logger.info(f"  Training Batches: {self.stats['total_training_batches']}")
        logger.info(f"  Total Inferences: {self.stats['total_inferences']}")
        logger.info(f"  PRM+ORM Evaluations: {self.stats['total_prm_orm_evaluations']}")
        logger.info(f"  Parameter Optimizations: {self.stats['total_parameter_optimizations']}")
        
        if self.stats['average_inference_score']:
            avg_score = np.mean(list(self.stats['average_inference_score'])[-10:])
            logger.info(f"  Recent Performance: {avg_score:.3f}")
        
        if self.stats['prm_scores'] and self.stats['orm_scores']:
            avg_prm = np.mean(list(self.stats['prm_scores'])[-10:])
            avg_orm = np.mean(list(self.stats['orm_scores'])[-10:])
            logger.info(f"  Recent PRM: {avg_prm:.3f}, ORM: {avg_orm:.3f}")
        
        # Parameter status
        opt_status = self.param_manager.get_optimization_status()
        logger.info(f"  Current LR: {opt_status['current_parameters'].get('learning_rate', 'N/A'):.2e}")
        logger.info(f"  Current Batch Size: {opt_status['current_parameters'].get('batch_size', 'N/A')}")
    
    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system status"""
        base_status = {
            'stats': dict(self.stats),
            'training_data_size': len(self.training_data),
            'pending_requests': self.pending_inference_requests.qsize(),
            'processed_results': len(self.processed_results),
            'running': self.running,
            'uptime': time.time() - self.stats['start_time'],
            'config': asdict(self.config)
        }
        
        # Add enhanced status
        enhanced_status = {
            'optimization_status': self.param_manager.get_optimization_status(),
            'current_performance': self._calculate_current_performance(),
            'prm_orm_statistics': {
                'total_prm_orm_evaluations': self.stats['total_prm_orm_evaluations'],
                'avg_prm_score': np.mean(list(self.stats['prm_scores'])) if self.stats['prm_scores'] else 0,
                'avg_orm_score': np.mean(list(self.stats['orm_scores'])) if self.stats['orm_scores'] else 0,
                'avg_reasoning_quality': np.mean(list(self.stats['reasoning_quality_scores'])) if self.stats['reasoning_quality_scores'] else 0
            },
            'adaptive_features': {
                'adaptive_sampling_enabled': self.config.enable_adaptive_sampling,
                'parameter_optimization_enabled': self.config.online_parameter_optimization,
                'dynamic_batch_sizing': self.config.dynamic_batch_sizing,
                'quality_based_sampling': self.config.quality_based_sampling
            }
        }
        
        base_status.update(enhanced_status)
        return base_status
    
    # Override parent methods for enhanced functionality
    def _load_training_data(self):
        """Load training data with enhanced processing"""
        try:
            new_data = self.data_processor.load_training_data(self.config.training_data_dir)
            
            if new_data:
                # Enhanced data processing
                processed_data = self.data_processor._enhance_training_data(new_data)
                self.training_data = processed_data
                self.stats['data_reload_count'] += 1
                
                logger.info(f"📊 Loaded {len(self.training_data)} enhanced training examples")
                
                # Log quality distribution
                quality_scores = [item['quality_score'] for item in self.training_data]
                high_quality = len([q for q in quality_scores if q > 0.7])
                medium_quality = len([q for q in quality_scores if 0.4 <= q <= 0.7])
                low_quality = len([q for q in quality_scores if q < 0.4])
                
                logger.info(f"  Quality distribution: High={high_quality}, Medium={medium_quality}, Low={low_quality}")
            else:
                logger.warning("⚠️ No training data found")
                
        except Exception as e:
            logger.error(f"Failed to load enhanced training data: {e}")
    
    def _start_file_watcher(self):
        """Start enhanced file watcher"""
        def watch_files():
            try:
                from watchdog.observers import Observer
                
                observer = Observer()
                handler = FileWatcher(self._on_file_change)
                
                # Watch training data directory
                if Path(self.config.training_data_dir).exists():
                    observer.schedule(handler, self.config.training_data_dir, recursive=True)
                
                # Watch inference requests file
                requests_dir = Path(self.config.inference_requests_file).parent
                if requests_dir.exists():
                    observer.schedule(handler, str(requests_dir), recursive=False)
                
                observer.start()
                logger.info("📁 Enhanced file watcher started")
                
                while self.running:
                    time.sleep(1)
                
                observer.stop()
                observer.join()
                
            except Exception as e:
                logger.error(f"Enhanced file watcher error: {e}")
        
        self.file_watcher_thread = threading.Thread(target=watch_files, daemon=True)
        self.file_watcher_thread.start()
    
    def _on_file_change(self):
        """Handle file change events with enhanced logging"""
        logger.info("📁 Files changed, reloading enhanced data...")
        self._load_training_data()
        self._load_inference_requests()
        
        # Trigger immediate evaluation if significant data change
        if len(self.training_data) > 0:
            current_performance = self._calculate_current_performance()
            self.param_manager.update_performance(current_performance)
    
    def _load_inference_requests(self):
        """Load inference requests (unchanged from parent)"""
        try:
            requests = self.data_processor.load_inference_requests(
                self.config.inference_requests_file
            )
            
            for request in requests:
                if 'prompt' in request:
                    self.pending_inference_requests.put(request)
            
            if requests:
                logger.info(f"📥 Loaded {len(requests)} inference requests")
                
        except Exception as e:
            logger.error(f"Failed to load inference requests: {e}")
    
    def _save_inference_result(self, result: Dict):
        """Save enhanced inference result"""
        if not self.config.save_intermediate_results:
            return
        
        try:
            output_file = Path(self.config.results_output_dir) / f"enhanced_inference_results_{int(time.time() / 3600)}.jsonl"
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save enhanced inference result: {e}")

def test_enhanced_online_system():
    """Test the enhanced online training system"""
    print("🧪 Testing Enhanced Online Training System")
    print("=" * 60)
    
    # Create sample data
    if not Path("./training_data").exists():
        create_sample_data_files()
    
    # Create models
    base_model, prm_model, orm_model, tokenizer = create_dummy_models()
    
    # Create enhanced config
    config = EnhancedOnlineConfig(
        training_data_dir="./training_data",
        inference_requests_file="./inference_requests.jsonl",
        results_output_dir="./enhanced_results",
        checkpoints_dir="./enhanced_checkpoints",
        batch_size=8,
        data_refresh_interval=15,
        
        # Enhanced features
        enable_adaptive_sampling=True,
        online_parameter_optimization=True,
        dynamic_batch_sizing=True,
        quality_based_sampling=True,
        prm_orm_evaluation_frequency=5,
        optimization_trigger_threshold=0.05
    )
    
    # Create enhanced system
    system = EnhancedFileBasedTrainingSystem(
        base_model, prm_model, orm_model, tokenizer, config
    )
    
    print("🚀 Starting enhanced system...")
    system.start()
    
    # Let it run and observe enhanced features
    for i in range(6):
        time.sleep(10)
        status = system.get_enhanced_system_status()
        
        print(f"\n📊 Enhanced Status Update #{i+1}:")
        print(f"  Training Batches: {status['stats']['total_training_batches']}")
        print(f"  Inferences: {status['stats']['total_inferences']}")
        print(f"  PRM+ORM Evaluations: {status['stats']['total_prm_orm_evaluations']}")
        print(f"  Parameter Optimizations: {status['stats']['total_parameter_optimizations']}")
        print(f"  Current Performance: {status['current_performance']:.3f}")
        
        # Show enhanced statistics
        prm_orm_stats = status['prm_orm_statistics']
        print(f"  PRM Score: {prm_orm_stats['avg_prm_score']:.3f}")
        print(f"  ORM Score: {prm_orm_stats['avg_orm_score']:.3f}")
        print(f"  Reasoning Quality: {prm_orm_stats['avg_reasoning_quality']:.3f}")
        
        # Show current parameters
        opt_status = status['optimization_status']
        current_params = opt_status['current_parameters']
        print(f"  Current LR: {current_params.get('learning_rate', 'N/A'):.2e}")
        print(f"  Current Batch Size: {current_params.get('batch_size', 'N/A')}")
        
        # Add some dynamic data to test adaptation
        if i == 2:
            print("📚 Adding high-quality training data...")
            new_data = [
                {"prompt": "Advanced ML concepts", "response": "Detailed explanation of advanced machine learning", "reward": 0.95},
                {"prompt": "Deep learning architecture", "response": "Comprehensive overview of neural network architectures", "reward": 0.9}
            ]
            with open("./training_data/high_quality_data.jsonl", "w") as f:
                for item in new_data:
                    f.write(json.dumps(item) + "\n")
        
        if i == 4:
            print("📥 Adding more inference requests...")
            more_requests = [
                {"id": "enhanced_001", "prompt": "Explain quantum machine learning"},
                {"id": "enhanced_002", "prompt": "What are transformer architectures?"}
            ]
            with open("./inference_requests.jsonl", "a") as f:
                for req in more_requests:
                    f.write(json.dumps(req) + "\n")
    
    # Final enhanced status
    final_status = system.get_enhanced_system_status()
    print(f"\n🎉 Final Enhanced System Status:")
    print(f"  Total Runtime: {final_status['uptime']:.1f}s")
    print(f"  Training Batches: {final_status['stats']['total_training_batches']}")
    print(f"  Total Inferences: {final_status['stats']['total_inferences']}")
    print(f"  PRM+ORM Evaluations: {final_status['stats']['total_prm_orm_evaluations']}")
    print(f"  Parameter Optimizations: {final_status['stats']['total_parameter_optimizations']}")
    print(f"  Final Performance: {final_status['current_performance']:.3f}")
    
    # Show optimization history
    if final_status['stats']['parameter_optimization_history']:
        print(f"\n🎯 Parameter Optimization History:")
        for i, opt in enumerate(final_status['stats']['parameter_optimization_history'], 1):
            print(f"  Optimization #{i}: Performance {opt['previous_performance']:.3f} → Improved")
    
    # Save enhanced checkpoint
    system._save_enhanced_checkpoint()
    
    # Stop system
    system.stop()
    
    print(f"\n✅ Enhanced online training system test completed!")
    print(f"📁 Check enhanced results in:")
    print(f"  ./enhanced_results/ - Enhanced inference results")
    print(f"  ./enhanced_checkpoints/ - Enhanced model checkpoints")
    
    return system

def run_production_enhanced_system():
    """Run production enhanced online training system"""
    print("🏭 Starting Production Enhanced Online Training System")
    print("=" * 60)
    
    # Check data
    if not Path("./training_data").exists():
        print("📁 Creating sample data...")
        create_sample_data_files()
    
    # Create models
    base_model, prm_model, orm_model, tokenizer = create_dummy_models()
    
    # Production enhanced config
    config = EnhancedOnlineConfig(
        training_data_dir="./training_data",
        inference_requests_file="./inference_requests.jsonl",
        results_output_dir="./production_enhanced_results",
        checkpoints_dir="./production_enhanced_checkpoints",
        batch_size=16,
        data_refresh_interval=30,
        
        # Production enhanced settings
        enable_adaptive_sampling=True,
        online_parameter_optimization=True,
        dynamic_batch_sizing=True,
        quality_based_sampling=True,
        prm_orm_evaluation_frequency=10,
        sampling_adaptation_interval=100,
        optimization_trigger_threshold=0.08,
        optimization_cooldown=200,
        
        # Performance settings
        evaluation_cache_size=2000,
        performance_history_size=500,
        max_cache_size=100000
    )
    
    # Create and start enhanced system
    system = EnhancedFileBasedTrainingSystem(
        base_model, prm_model, orm_model, tokenizer, config
    )
    
    # Try to load enhanced checkpoint
    latest_checkpoint = Path(config.checkpoints_dir) / "latest_enhanced_checkpoint.pt"
    if latest_checkpoint.exists():
        print(f"📂 Loading enhanced checkpoint...")
        system.load_enhanced_checkpoint(str(latest_checkpoint))
    
    system.start()
    
    print("🚀 Production enhanced system started")
    print("🧠 Enhanced features enabled:")
    print("  ✅ Adaptive Sampling")
    print("  ✅ PRM+ORM Evaluation")
    print("  ✅ Online Parameter Optimization")
    print("  ✅ Dynamic Batch Sizing")
    print("  ✅ Quality-Based Sampling")
    print("\nMonitoring enhanced system performance...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(60)  # Status update every minute
            status = system.get_enhanced_system_status()
            
            print(f"[{time.strftime('%H:%M:%S')}] Enhanced Status:")
            print(f"  Data: {status['training_data_size']}, "
                  f"Batches: {status['stats']['total_training_batches']}, "
                  f"Inferences: {status['stats']['total_inferences']}, "
                  f"Performance: {status['current_performance']:.3f}")
            print(f"  PRM+ORM: {status['stats']['total_prm_orm_evaluations']}, "
                  f"Optimizations: {status['stats']['total_parameter_optimizations']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down production enhanced system...")
        system.stop()
        
        # Final comprehensive report
        final_status = system.get_enhanced_system_status()
        timestamp = int(time.time())
        
        # Export enhanced results
        system.export_results(f"./production_enhanced_results_{timestamp}.json")
        
        print(f"\n📈 Production Enhanced System Summary:")
        print(f"Uptime: {final_status['uptime']:.1f}s")
        print(f"Training Batches: {final_status['stats']['total_training_batches']}")
        print(f"Total Inferences: {final_status['stats']['total_inferences']}")
        print(f"PRM+ORM Evaluations: {final_status['stats']['total_prm_orm_evaluations']}")
        print(f"Parameter Optimizations: {final_status['stats']['total_parameter_optimizations']}")
        print(f"Final Performance: {final_status['current_performance']:.3f}")
        
        # Enhanced statistics
        prm_orm_stats = final_status['prm_orm_statistics']
        print(f"Average PRM Score: {prm_orm_stats['avg_prm_score']:.3f}")
        print(f"Average ORM Score: {prm_orm_stats['avg_orm_score']:.3f}")
        print(f"Average Reasoning Quality: {prm_orm_stats['avg_reasoning_quality']:.3f}")
        
        print("✅ Production enhanced system stopped successfully.")

if __name__ == "__main__":
    import sys
    import logging
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "production":
            run_production_enhanced_system()
        elif sys.argv[1] == "test":
            test_enhanced_online_system()
        else:
            print("Usage: python enhanced_online_training.py [production|test]")
    else:
        test_enhanced_online_system()