# examples/multi_model_reward_guided_inference.py
"""
Enhanced reward-guided inference with multiple models and automatic model selection
based on input distribution analysis
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import copy
import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import our modules
from training_infra.models.llama import create_llama_7b
from training_infra.inference.reward_guided import (
    RewardGuidedConfig, 
    ProcessRewardModel, 
    OutcomeRewardModel,
    RewardGuidedInferenceEngine,
    create_reward_guided_engine
)
from training_infra.inference.engine import GenerationConfig
from training_infra.inference.sampling import SamplingConfig
from training_infra.rlhf.prm_orm_training import (
    train_process_reward_model,
    train_outcome_reward_model,
    create_step_reward_data_from_math_problems,
    create_outcome_reward_data_from_qa_pairs,
    evaluate_reward_models
)

class TaskType(Enum):
    """Different types of tasks for model specialization"""
    MATHEMATICAL = "mathematical"
    CREATIVE_WRITING = "creative_writing"
    FACTUAL_QA = "factual_qa"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    SCIENTIFIC = "scientific"
    CONVERSATIONAL = "conversational"

@dataclass
class ModelSpec:
    """Specification for a specialized model"""
    model_id: str
    model: nn.Module
    prm: Optional[ProcessRewardModel]
    orm: Optional[OutcomeRewardModel]
    task_types: List[TaskType]
    specialized_domains: List[str]
    performance_metrics: Dict[str, float]
    description: str

@dataclass
class InputAnalysis:
    """Analysis results for input text"""
    task_type: TaskType
    confidence: float
    features: Dict[str, Any]
    keywords: List[str]
    complexity_score: float
    domain_indicators: List[str]

class InputClassifier:
    """Classifier to analyze input and determine appropriate model"""
    
    def __init__(self):
        # Task-specific keywords and patterns
        self.task_patterns = {
            TaskType.MATHEMATICAL: {
                'keywords': ['solve', 'calculate', 'equation', 'formula', 'algebra', 'geometry', 
                           'derivative', 'integral', 'probability', 'statistics', 'theorem'],
                'patterns': [r'\d+[x-z]\s*[+\-*/=]', r'[+\-*/=]\s*\d+', r'\b\d+\.\d+\b', 
                           r'[∫∑∏√∞≠≤≥∈∉⊂⊃∩∪]'],
                'complexity_indicators': ['differential', 'matrix', 'vector', 'proof', 'limit']
            },
            TaskType.CREATIVE_WRITING: {
                'keywords': ['story', 'write', 'creative', 'character', 'plot', 'narrative',
                           'poem', 'fiction', 'dialogue', 'scene', 'chapter'],
                'patterns': [r'write\s+a\s+story', r'create\s+a\s+character', r'once\s+upon'],
                'complexity_indicators': ['literary', 'metaphor', 'symbolism', 'genre']
            },
            TaskType.FACTUAL_QA: {
                'keywords': ['what', 'who', 'when', 'where', 'why', 'how', 'explain', 'define',
                           'fact', 'information', 'details', 'describe'],
                'patterns': [r'^(what|who|when|where|why|how)\s+', r'explain\s+', r'define\s+'],
                'complexity_indicators': ['comprehensive', 'detailed', 'analysis', 'research']
            },
            TaskType.REASONING: {
                'keywords': ['analyze', 'compare', 'evaluate', 'assess', 'argue', 'reason',
                           'logic', 'because', 'therefore', 'conclude', 'infer'],
                'patterns': [r'pros\s+and\s+cons', r'advantages\s+and\s+disadvantages', 
                           r'compare\s+.*\s+with', r'analyze\s+'],
                'complexity_indicators': ['critical', 'philosophical', 'ethical', 'complex']
            },
            TaskType.CODE_GENERATION: {
                'keywords': ['code', 'program', 'function', 'algorithm', 'python', 'javascript',
                           'class', 'method', 'variable', 'loop', 'debug'],
                'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'#.*code',
                           r'write\s+.*\s+code'],
                'complexity_indicators': ['optimization', 'architecture', 'framework', 'api']
            },
            TaskType.SCIENTIFIC: {
                'keywords': ['research', 'hypothesis', 'experiment', 'data', 'method',
                           'analysis', 'conclusion', 'theory', 'evidence', 'study'],
                'patterns': [r'research\s+', r'study\s+shows', r'according\s+to'],
                'complexity_indicators': ['peer-reviewed', 'methodology', 'statistical', 'empirical']
            },
            TaskType.CONVERSATIONAL: {
                'keywords': ['hello', 'hi', 'thanks', 'please', 'help', 'chat', 'talk',
                           'discuss', 'opinion', 'think'],
                'patterns': [r'^(hi|hello|hey)', r'what\s+do\s+you\s+think', r'can\s+you\s+help'],
                'complexity_indicators': ['personal', 'emotional', 'social', 'casual']
            }
        }
        
        # Initialize TF-IDF for semantic analysis
        self.tfidf_vectorizer = None
        self.domain_embeddings = {}
        
    def analyze_input(self, text: str) -> InputAnalysis:
        """Analyze input text to determine task type and characteristics"""
        
        text_lower = text.lower()
        
        # Calculate scores for each task type
        task_scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in text_lower)
            score += keyword_matches * 2
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in patterns['patterns']
                                if re.search(pattern, text_lower))
            score += pattern_matches * 3
            
            # Complexity indicators
            complexity_matches = sum(1 for indicator in patterns['complexity_indicators']
                                   if indicator in text_lower)
            score += complexity_matches * 1.5
            
            task_scores[task_type] = score
        
        # Determine best task type
        best_task = max(task_scores.keys(), key=lambda k: task_scores[k])
        confidence = task_scores[best_task] / max(sum(task_scores.values()), 1)
        
        # Extract features
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'question_marks': text.count('?'),
            'mathematical_symbols': len(re.findall(r'[+\-*/=<>≤≥∫∑]', text)),
            'code_indicators': len(re.findall(r'[{}();]', text)),
            'task_scores': {t.value: s for t, s in task_scores.items()}
        }
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(text, features)
        
        # Extract keywords and domain indicators
        keywords = self._extract_keywords(text)
        domain_indicators = self._identify_domains(text)
        
        return InputAnalysis(
            task_type=best_task,
            confidence=confidence,
            features=features,
            keywords=keywords,
            complexity_score=complexity_score,
            domain_indicators=domain_indicators
        )
    
    def _calculate_complexity(self, text: str, features: Dict) -> float:
        """Calculate complexity score based on various factors"""
        
        complexity = 0.0
        
        # Length-based complexity
        complexity += min(features['length'] / 1000, 1.0) * 0.3
        
        # Technical term density
        technical_terms = ['algorithm', 'optimization', 'analysis', 'methodology', 
                         'implementation', 'architecture', 'framework']
        tech_density = sum(1 for term in technical_terms if term in text.lower())
        complexity += min(tech_density / 5, 1.0) * 0.4
        
        # Sentence structure complexity
        avg_sentence_length = features['word_count'] / max(features['sentence_count'], 1)
        complexity += min(avg_sentence_length / 20, 1.0) * 0.3
        
        return min(complexity, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        
        # Simple keyword extraction (in practice, use more sophisticated NLP)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = Counter(words)
        
        # Filter out common words
        stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been',
                    'have', 'were', 'said', 'each', 'which', 'their', 'time', 'would'}
        
        keywords = [word for word, freq in word_freq.most_common(10) 
                   if word not in stopwords and len(word) > 3]
        
        return keywords[:5]
    
    def _identify_domains(self, text: str) -> List[str]:
        """Identify domain-specific indicators"""
        
        domain_keywords = {
            'mathematics': ['equation', 'formula', 'theorem', 'proof', 'calculation'],
            'science': ['hypothesis', 'experiment', 'theory', 'research', 'data'],
            'technology': ['algorithm', 'code', 'software', 'system', 'programming'],
            'business': ['market', 'strategy', 'analysis', 'revenue', 'customer'],
            'education': ['learn', 'teach', 'student', 'curriculum', 'knowledge'],
            'healthcare': ['patient', 'treatment', 'diagnosis', 'medical', 'health']
        }
        
        identified_domains = []
        text_lower = text.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_domains.append(domain)
        
        return identified_domains

class MultiModelInferenceEngine:
    """Multi-model inference engine with automatic model selection"""
    
    def __init__(self, models: List[ModelSpec], tokenizer, default_model_id: str = None):
        self.models = {model.model_id: model for model in models}
        self.tokenizer = tokenizer
        self.default_model_id = default_model_id or models[0].model_id
        self.classifier = InputClassifier()
        self.inference_engines = {}
        
        # Create inference engines for each model
        for model_spec in models:
            self.inference_engines[model_spec.model_id] = create_reward_guided_engine(
                model=model_spec.model,
                prm=model_spec.prm,
                orm=model_spec.orm,
                tokenizer=tokenizer
            )
        
        # Track usage statistics
        self.usage_stats = {model_id: 0 for model_id in self.models.keys()}
        self.performance_history = []
    
    def select_model(self, input_text: str) -> Tuple[str, InputAnalysis, float]:
        """Select the best model for the given input"""
        
        # Analyze input
        analysis = self.classifier.analyze_input(input_text)
        
        # Calculate model scores
        model_scores = {}
        
        for model_id, model_spec in self.models.items():
            score = 0.0
            
            # Task type compatibility
            if analysis.task_type in model_spec.task_types:
                score += 0.4
            
            # Domain specialization
            domain_matches = len(set(analysis.domain_indicators) & 
                               set(model_spec.specialized_domains))
            score += domain_matches * 0.2
            
            # Performance metrics
            task_performance = model_spec.performance_metrics.get(
                analysis.task_type.value, 0.5)
            score += task_performance * 0.3
            
            # Complexity handling
            if analysis.complexity_score > 0.7:
                # Prefer models with better reasoning capabilities
                reasoning_score = model_spec.performance_metrics.get('reasoning', 0.5)
                score += reasoning_score * 0.1
            
            model_scores[model_id] = score
        
        # Select best model
        best_model_id = max(model_scores.keys(), key=lambda k: model_scores[k])
        confidence = model_scores[best_model_id] / max(sum(model_scores.values()), 1)
        
        # Fallback to default if confidence is too low
        if confidence < 0.3:
            best_model_id = self.default_model_id
            confidence = 0.3
        
        return best_model_id, analysis, confidence
    
    def generate(self, prompt: str, config: RewardGuidedConfig = None, 
                max_length: int = 200, temperature: float = 0.7, 
                force_model: str = None) -> Dict[str, Any]:
        """Generate text using the most appropriate model"""
        
        start_time = time.time()
        
        # Model selection
        if force_model and force_model in self.models:
            selected_model_id = force_model
            analysis = self.classifier.analyze_input(prompt)
            selection_confidence = 1.0
        else:
            selected_model_id, analysis, selection_confidence = self.select_model(prompt)
        
        # Update usage statistics
        self.usage_stats[selected_model_id] += 1
        
        # Adaptive configuration based on task type
        if config is None:
            config = self._get_adaptive_config(analysis)
        
        # Generate using selected model
        try:
            engine = self.inference_engines[selected_model_id]
            result = engine.generate(
                prompt=prompt,
                config=config,
                max_length=max_length,
                temperature=temperature
            )
            
            generation_time = time.time() - start_time
            
            # Add metadata
            result.update({
                'selected_model': selected_model_id,
                'model_selection_confidence': selection_confidence,
                'input_analysis': {
                    'task_type': analysis.task_type.value,
                    'confidence': analysis.confidence,
                    'complexity_score': analysis.complexity_score,
                    'keywords': analysis.keywords,
                    'domains': analysis.domain_indicators
                },
                'generation_time': generation_time,
                'config_used': config.__dict__
            })
            
            # Record performance
            self.performance_history.append({
                'timestamp': time.time(),
                'model_id': selected_model_id,
                'task_type': analysis.task_type.value,
                'complexity': analysis.complexity_score,
                'generation_time': generation_time,
                'success': True
            })
            
            return result
            
        except Exception as e:
            # Fallback to default model
            if selected_model_id != self.default_model_id:
                try:
                    fallback_engine = self.inference_engines[self.default_model_id]
                    result = fallback_engine.generate(
                        prompt=prompt,
                        config=config,
                        max_length=max_length,
                        temperature=temperature
                    )
                    
                    result.update({
                        'selected_model': self.default_model_id,
                        'fallback_used': True,
                        'original_selection': selected_model_id,
                        'error': str(e)
                    })
                    
                    return result
                    
                except Exception as fallback_error:
                    raise Exception(f"Both primary ({e}) and fallback ({fallback_error}) failed")
            else:
                raise e
    
    def _get_adaptive_config(self, analysis: InputAnalysis) -> RewardGuidedConfig:
        """Get adaptive configuration based on input analysis"""
        
        task_configs = {
            TaskType.MATHEMATICAL: RewardGuidedConfig(
                search_strategy="guided_sampling",
                use_prm=True,
                use_orm=False,
                prm_weight=0.8,
                reward_alpha=0.3,
                early_stopping=True
            ),
            
            TaskType.CREATIVE_WRITING: RewardGuidedConfig(
                search_strategy="best_of_n",
                num_candidates=8,
                use_prm=False,
                use_orm=True,
                orm_weight=0.9,
                diversity_penalty=0.2
            ),
            
            TaskType.FACTUAL_QA: RewardGuidedConfig(
                search_strategy="beam_search",
                num_beams=5,
                use_prm=True,
                use_orm=True,
                prm_weight=0.4,
                orm_weight=0.6,
                early_stopping=True
            ),
            
            TaskType.REASONING: RewardGuidedConfig(
                search_strategy="tree_search",
                use_prm=True,
                use_orm=True,
                prm_weight=0.5,
                orm_weight=0.5,
                tree_depth=4,
                branching_factor=3
            ),
            
            TaskType.CODE_GENERATION: RewardGuidedConfig(
                search_strategy="beam_search",
                num_beams=4,
                use_prm=True,
                use_orm=True,
                prm_weight=0.6,
                orm_weight=0.4,
                early_stopping=True
            )
        }
        
        base_config = task_configs.get(analysis.task_type, task_configs[TaskType.FACTUAL_QA])
        
        # Adjust based on complexity
        if analysis.complexity_score > 0.8:
            # More thorough search for complex problems
            if hasattr(base_config, 'num_beams'):
                base_config.num_beams = min(base_config.num_beams + 2, 8)
            if hasattr(base_config, 'tree_depth'):
                base_config.tree_depth = min(base_config.tree_depth + 1, 6)
        
        return base_config
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get usage and performance statistics"""
        
        total_usage = sum(self.usage_stats.values())
        
        stats = {
            'usage_distribution': {
                model_id: (count / max(total_usage, 1)) * 100 
                for model_id, count in self.usage_stats.items()
            },
            'total_generations': total_usage,
            'model_performance': {}
        }
        
        # Calculate performance metrics per model
        for model_id in self.models.keys():
            model_history = [h for h in self.performance_history if h['model_id'] == model_id]
            
            if model_history:
                avg_time = np.mean([h['generation_time'] for h in model_history])
                success_rate = sum(1 for h in model_history if h['success']) / len(model_history)
                avg_complexity = np.mean([h['complexity'] for h in model_history])
                
                stats['model_performance'][model_id] = {
                    'average_generation_time': avg_time,
                    'success_rate': success_rate,
                    'average_complexity_handled': avg_complexity,
                    'total_requests': len(model_history)
                }
        
        return stats

# Enhanced dummy classes for demonstration
class DummyTokenizer:
    """Enhanced dummy tokenizer"""
    
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = 32000
    
    def encode(self, text: str, return_tensors=None):
        tokens = [hash(word) % self.vocab_size for word in text.split()[:100]]
        if return_tensors == 'pt':
            return torch.tensor(tokens).unsqueeze(0)
        return tokens
    
    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return f"[Generated text from {len(tokens)} tokens]"
    
    def __call__(self, text, **kwargs):
        tokens = self.encode(text)
        max_length = kwargs.get('max_length', 512)
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return {
            'input_ids': torch.tensor(tokens).unsqueeze(0),
            'attention_mask': torch.ones(1, len(tokens))
        }

def create_specialized_models() -> List[ModelSpec]:
    """Create specialized models for different task types"""
    
    # Create base models (in practice, these would be different architectures/weights)
    math_model = create_llama_7b()
    creative_model = create_llama_7b() 
    reasoning_model = create_llama_7b()
    code_model = create_llama_7b()
    general_model = create_llama_7b()
    
    # Create dummy reward models (simplified for demo)
    def create_dummy_prm(base_model):
        return ProcessRewardModel(copy.deepcopy(base_model))
    
    def create_dummy_orm(base_model):
        from training_infra.rlhf.reward_model import RewardModelConfig
        config = RewardModelConfig(hidden_size=4096)
        return OutcomeRewardModel(copy.deepcopy(base_model), config)
    
    models = [
        ModelSpec(
            model_id="math_specialist",
            model=math_model,
            prm=create_dummy_prm(math_model),
            orm=create_dummy_orm(math_model),
            task_types=[TaskType.MATHEMATICAL, TaskType.SCIENTIFIC],
            specialized_domains=['mathematics', 'physics', 'engineering'],
            performance_metrics={
                'mathematical': 0.95,
                'scientific': 0.85,
                'reasoning': 0.80,
                'factual_qa': 0.70
            },
            description="Specialized for mathematical and scientific problem solving"
        ),
        
        ModelSpec(
            model_id="creative_specialist",
            model=creative_model,
            prm=None,  # Creative tasks don't need step-by-step rewards
            orm=create_dummy_orm(creative_model),
            task_types=[TaskType.CREATIVE_WRITING, TaskType.CONVERSATIONAL],
            specialized_domains=['literature', 'arts', 'entertainment'],
            performance_metrics={
                'creative_writing': 0.90,
                'conversational': 0.85,
                'factual_qa': 0.60,
                'reasoning': 0.65
            },
            description="Specialized for creative writing and conversational tasks"
        ),
        
        ModelSpec(
            model_id="reasoning_specialist", 
            model=reasoning_model,
            prm=create_dummy_prm(reasoning_model),
            orm=create_dummy_orm(reasoning_model),
            task_types=[TaskType.REASONING, TaskType.FACTUAL_QA],
            specialized_domains=['philosophy', 'logic', 'analysis'],
            performance_metrics={
                'reasoning': 0.92,
                'factual_qa': 0.88,
                'scientific': 0.75,
                'mathematical': 0.70
            },
            description="Specialized for complex reasoning and analytical tasks"
        ),
        
        ModelSpec(
            model_id="code_specialist",
            model=code_model,
            prm=create_dummy_prm(code_model),
            orm=create_dummy_orm(code_model),
            task_types=[TaskType.CODE_GENERATION, TaskType.REASONING],
            specialized_domains=['programming', 'software', 'algorithms'],
            performance_metrics={
                'code_generation': 0.93,
                'reasoning': 0.85,
                'mathematical': 0.80,
                'factual_qa': 0.70
            },
            description="Specialized for code generation and programming tasks"
        ),
        
        ModelSpec(
            model_id="general_model",
            model=general_model,
            prm=create_dummy_prm(general_model),
            orm=create_dummy_orm(general_model),
            task_types=list(TaskType),  # Handles all task types
            specialized_domains=['general'],
            performance_metrics={
                task_type.value: 0.75 for task_type in TaskType
            },
            description="General-purpose model for all task types"
        )
    ]
    
    return models

def demonstrate_multi_model_inference():
    """Demonstrate multi-model inference with automatic selection"""
    
    print("🤖 Multi-Model Inference Demonstration")
    print("=" * 60)
    
    # Create models and tokenizer
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    
    # Create multi-model engine
    engine = MultiModelInferenceEngine(
        models=models,
        tokenizer=tokenizer,
        default_model_id="general_model"
    )
    
    print(f"✅ Created {len(models)} specialized models:")
    for model in models:
        print(f"   • {model.model_id}: {model.description}")
    
    # Test prompts of different types
    test_prompts = [
        {
            'text': "Solve the quadratic equation: x² - 5x + 6 = 0",
            'expected_model': 'math_specialist'
        },
        {
            'text': "Write a creative short story about a time-traveling detective",
            'expected_model': 'creative_specialist'
        },
        {
            'text': "Analyze the pros and cons of renewable energy adoption",
            'expected_model': 'reasoning_specialist'
        },
        {
            'text': "Write a Python function to implement binary search",
            'expected_model': 'code_specialist'
        },
        {
            'text': "What is the capital of France?",
            'expected_model': 'general_model'
        },
        {
            'text': "Explain the process of photosynthesis in plants",
            'expected_model': 'reasoning_specialist'
        },
        {
            'text': "Debug this code and optimize its performance: def slow_function(n): result = 0; for i in range(n): for j in range(n): result += i * j; return result",
            'expected_model': 'code_specialist'
        }
    ]
    
    results = []
    
    print(f"\n🧪 Testing {len(test_prompts)} different prompts")
    print("=" * 60)
    
    for i, prompt_data in enumerate(test_prompts, 1):
        prompt = prompt_data['text']
        expected = prompt_data['expected_model']
        
        print(f"\n[{i}/{len(test_prompts)}] Testing prompt:")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Expected model: {expected}")
        
        # Generate response
        try:
            result = engine.generate(
                prompt=prompt,
                max_length=150,
                temperature=0.7
            )
            
            selected_model = result['selected_model']
            confidence = result['model_selection_confidence']
            analysis = result['input_analysis']
            
            print(f"Selected model: {selected_model}")
            print(f"Selection confidence: {confidence:.3f}")
            print(f"Task type detected: {analysis['task_type']}")
            print(f"Complexity score: {analysis['complexity_score']:.3f}")
            print(f"Generation time: {result['generation_time']:.3f}s")
            
            # Check if selection was correct
            correct_selection = selected_model == expected
            print(f"Selection accuracy: {'✅ Correct' if correct_selection else '❌ Incorrect'}")
            
            if not correct_selection:
                print(f"   Expected: {expected}, Got: {selected_model}")
            
            results.append({
                'prompt': prompt,
                'expected_model': expected,
                'selected_model': selected_model,
                'correct': correct_selection,
                'confidence': confidence,
                'analysis': analysis,
                'result': result
            })
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            results.append({
                'prompt': prompt,
                'expected_model': expected,
                'selected_model': None,
                'correct': False,
                'error': str(e)
            })
    
    return engine, results

def analyze_model_selection_performance(results: List[Dict]) -> Dict[str, Any]:
    """Analyze the performance of model selection"""
    
    print("\n📊 Model Selection Analysis")
    print("=" * 40)
    
    successful_results = [r for r in results if r.get('selected_model')]
    
    if not successful_results:
        print("❌ No successful results to analyze")
        return {}
    
    # Calculate accuracy
    correct_selections = sum(1 for r in successful_results if r['correct'])
    accuracy = correct_selections / len(successful_results)
    
    print(f"Selection Accuracy: {accuracy:.2%} ({correct_selections}/{len(successful_results)})")
    
    # Analyze by task type
    task_type_performance = {}
    for result in successful_results:
        task_type = result['analysis']['task_type']
        if task_type not in task_type_performance:
            task_type_performance[task_type] = {'correct': 0, 'total': 0}
        
        task_type_performance[task_type]['total'] += 1
        if result['correct']:
            task_type_performance[task_type]['correct'] += 1
    
    print(f"\nPerformance by Task Type:")
    for task_type, stats in task_type_performance.items():
        task_accuracy = stats['correct'] / stats['total']
        print(f"  {task_type}: {task_accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Average confidence analysis
    avg_confidence = np.mean([r['confidence'] for r in successful_results])
    correct_confidence = np.mean([r['confidence'] for r in successful_results if r['correct']])
    incorrect_confidence = np.mean([r['confidence'] for r in successful_results if not r['correct']])
    
    print(f"\nConfidence Analysis:")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Correct selections: {correct_confidence:.3f}")
    print(f"  Incorrect selections: {incorrect_confidence:.3f}")
    
    # Model usage distribution
    model_usage = {}
    for result in successful_results:
        model = result['selected_model']
        model_usage[model] = model_usage.get(model, 0) + 1
    
    print(f"\nModel Usage Distribution:")
    for model, count in sorted(model_usage.items()):
        percentage = (count / len(successful_results)) * 100
        print(f"  {model}: {count} times ({percentage:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'task_type_performance': task_type_performance,
        'avg_confidence': avg_confidence,
        'model_usage': model_usage,
        'total_tests': len(successful_results)
    }

def compare_single_vs_multi_model():
    """Compare performance between single model and multi-model approaches"""
    
    print("\n⚖️ Single vs Multi-Model Comparison")
    print("=" * 50)
    
    # Create models
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    
    # Multi-model engine
    multi_engine = MultiModelInferenceEngine(
        models=models,
        tokenizer=tokenizer,
        default_model_id="general_model"
    )
    
    # Single model engine (using general model)
    general_model = next(m for m in models if m.model_id == "general_model")
    single_engine = create_reward_guided_engine(
        model=general_model.model,
        prm=general_model.prm,
        orm=general_model.orm,
        tokenizer=tokenizer
    )
    
    # Test prompts with varying complexity
    test_prompts = [
        "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1",
        "Write a poem about artificial intelligence",
        "Compare the advantages and disadvantages of electric vehicles",
        "Implement a recursive factorial function in Python",
        "Explain quantum entanglement in simple terms",
        "Solve: 2x + 3y = 12, x - y = 2",
        "Create a dialogue between two characters meeting for the first time"
    ]
    
    comparison_results = {
        'single_model': [],
        'multi_model': []
    }
    
    print(f"Testing {len(test_prompts)} prompts with both approaches...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt[:60]}...")
        
        # Test single model
        try:
            start_time = time.time()
            single_result = single_engine.generate(
                prompt=prompt,
                config=RewardGuidedConfig(
                    search_strategy="beam_search",
                    num_beams=4,
                    use_prm=True,
                    use_orm=True
                ),
                max_length=150
            )
            single_time = time.time() - start_time
            
            comparison_results['single_model'].append({
                'prompt': prompt,
                'time': single_time,
                'success': True,
                'result': single_result
            })
            
            print(f"  Single model: {single_time:.3f}s")
            
        except Exception as e:
            comparison_results['single_model'].append({
                'prompt': prompt,
                'time': 0,
                'success': False,
                'error': str(e)
            })
            print(f"  Single model: Failed ({e})")
        
        # Test multi-model
        try:
            start_time = time.time()
            multi_result = multi_engine.generate(
                prompt=prompt,
                max_length=150
            )
            multi_time = time.time() - start_time
            
            comparison_results['multi_model'].append({
                'prompt': prompt,
                'time': multi_time,
                'success': True,
                'result': multi_result,
                'selected_model': multi_result['selected_model']
            })
            
            print(f"  Multi-model: {multi_time:.3f}s (used {multi_result['selected_model']})")
            
        except Exception as e:
            comparison_results['multi_model'].append({
                'prompt': prompt,
                'time': 0,
                'success': False,
                'error': str(e)
            })
            print(f"  Multi-model: Failed ({e})")
    
    # Analyze comparison results
    single_success = sum(1 for r in comparison_results['single_model'] if r['success'])
    multi_success = sum(1 for r in comparison_results['multi_model'] if r['success'])
    
    single_avg_time = np.mean([r['time'] for r in comparison_results['single_model'] if r['success']])
    multi_avg_time = np.mean([r['time'] for r in comparison_results['multi_model'] if r['success']])
    
    print(f"\n📈 Comparison Results:")
    print(f"Single Model Success Rate: {single_success}/{len(test_prompts)} ({single_success/len(test_prompts):.2%})")
    print(f"Multi-Model Success Rate: {multi_success}/{len(test_prompts)} ({multi_success/len(test_prompts):.2%})")
    print(f"Single Model Avg Time: {single_avg_time:.3f}s")
    print(f"Multi-Model Avg Time: {multi_avg_time:.3f}s")
    
    return comparison_results

def demonstrate_adaptive_routing():
    """Demonstrate advanced adaptive routing with load balancing"""
    
    print("\n🔄 Adaptive Routing Demonstration")
    print("=" * 45)
    
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    
    # Enhanced engine with load balancing
    class LoadBalancedMultiModelEngine(MultiModelInferenceEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_load = {model_id: 0 for model_id in self.models.keys()}
            self.max_concurrent = 3
        
        def select_model(self, input_text: str) -> Tuple[str, InputAnalysis, float]:
            # Get base selection
            primary_model, analysis, confidence = super().select_model(input_text)
            
            # Check load balancing
            if self.model_load[primary_model] >= self.max_concurrent:
                # Find alternative models
                alternatives = []
                for model_id, model_spec in self.models.items():
                    if (self.model_load[model_id] < self.max_concurrent and 
                        analysis.task_type in model_spec.task_types):
                        alternatives.append(model_id)
                
                if alternatives:
                    # Select least loaded alternative
                    alternative = min(alternatives, key=lambda m: self.model_load[m])
                    print(f"🔄 Load balancing: {primary_model} -> {alternative}")
                    return alternative, analysis, confidence * 0.9
            
            return primary_model, analysis, confidence
        
        def generate(self, prompt: str, **kwargs):
            # Simulate load tracking
            selected_model, analysis, confidence = self.select_model(prompt)
            self.model_load[selected_model] += 1
            
            try:
                result = super().generate(prompt, force_model=selected_model, **kwargs)
                return result
            finally:
                self.model_load[selected_model] -= 1
    
    # Create load-balanced engine
    engine = LoadBalancedMultiModelEngine(
        models=models,
        tokenizer=tokenizer,
        default_model_id="general_model"
    )
    
    # Simulate concurrent requests
    concurrent_prompts = [
        "Solve: ∫(x² + 2x)dx",
        "Calculate: lim(x→0) sin(x)/x", 
        "Find roots of: x³ - 6x² + 11x - 6 = 0",
        "Evaluate: Σ(n=1 to ∞) 1/n²",
        "Solve system: 3x + 2y = 7, x - y = 1"
    ]
    
    print(f"Simulating {len(concurrent_prompts)} concurrent math requests...")
    
    concurrent_results = []
    for i, prompt in enumerate(concurrent_prompts):
        print(f"\nRequest {i+1}: {prompt[:40]}...")
        
        result = engine.generate(prompt, max_length=100)
        concurrent_results.append(result)
        
        print(f"  Routed to: {result['selected_model']}")
        print(f"  Current load: {dict(engine.model_load)}")
    
    return engine, concurrent_results

def visualize_multi_model_performance(engine: MultiModelInferenceEngine, 
                                    results: List[Dict],
                                    analysis_results: Dict):
    """Create visualizations for multi-model performance"""
    
    print("\n📊 Creating Performance Visualizations")
    print("=" * 45)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Model Inference Performance Analysis', fontsize=16)
    
    # Model selection accuracy by task type
    task_performance = analysis_results.get('task_type_performance', {})
    if task_performance:
        tasks = list(task_performance.keys())
        accuracies = [task_performance[task]['correct'] / task_performance[task]['total'] 
                     for task in tasks]
        
        bars = ax1.bar(tasks, accuracies, color='skyblue')
        ax1.set_title('Model Selection Accuracy by Task Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom')
    
    # Model usage distribution
    model_usage = analysis_results.get('model_usage', {})
    if model_usage:
        models = list(model_usage.keys())
        usage_counts = list(model_usage.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        wedges, texts, autotexts = ax2.pie(usage_counts, labels=models, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('Model Usage Distribution')
    
    # Performance metrics comparison
    stats = engine.get_model_stats()
    model_perf = stats.get('model_performance', {})
    
    if model_perf:
        model_names = list(model_perf.keys())
        avg_times = [model_perf[model]['average_generation_time'] for model in model_names]
        success_rates = [model_perf[model]['success_rate'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, avg_times, width, label='Avg Time (s)', color='lightcoral')
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, success_rates, width, label='Success Rate', color='lightgreen')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Average Time (s)', color='red')
        ax3_twin.set_ylabel('Success Rate', color='green')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        
        # Add legends
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
    
    # Confidence vs Accuracy scatter plot
    successful_results = [r for r in results if r.get('selected_model')]
    if successful_results:
        confidences = [r['confidence'] for r in successful_results]
        correct_flags = [1 if r['correct'] else 0 for r in successful_results]
        
        # Group by confidence ranges for better visualization
        conf_ranges = np.arange(0, 1.1, 0.1)
        accuracy_by_conf = []
        conf_midpoints = []
        
        for i in range(len(conf_ranges)-1):
            mask = (np.array(confidences) >= conf_ranges[i]) & (np.array(confidences) < conf_ranges[i+1])
            if np.any(mask):
                range_accuracy = np.mean(np.array(correct_flags)[mask])
                accuracy_by_conf.append(range_accuracy)
                conf_midpoints.append((conf_ranges[i] + conf_ranges[i+1]) / 2)
        
        if accuracy_by_conf:
            ax4.scatter(conf_midpoints, accuracy_by_conf, s=100, alpha=0.7, color='gold')
            ax4.plot(conf_midpoints, accuracy_by_conf, '--', alpha=0.5)
            ax4.set_xlabel('Model Selection Confidence')
            ax4.set_ylabel('Selection Accuracy')
            ax4.set_title('Confidence vs Accuracy Correlation')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_model_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Visualization saved as 'multi_model_performance_analysis.png'")
    
    return fig

def save_multi_model_results(engine: MultiModelInferenceEngine,
                           results: List[Dict],
                           analysis_results: Dict,
                           comparison_results: Dict = None):
    """Save comprehensive multi-model experiment results"""
    
    print("\n💾 Saving Multi-Model Results")
    print("=" * 35)
    
    # Get engine statistics
    stats = engine.get_model_stats()
    
    # Prepare comprehensive results
    experiment_data = {
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
        'experiment_type': 'multi_model_inference',
        'model_configurations': {
            model_id: {
                'description': model_spec.description,
                'task_types': [tt.value for tt in model_spec.task_types],
                'specialized_domains': model_spec.specialized_domains,
                'performance_metrics': model_spec.performance_metrics
            }
            for model_id, model_spec in engine.models.items()
        },
        'selection_results': [
            {
                'prompt': r['prompt'],
                'expected_model': r.get('expected_model'),
                'selected_model': r.get('selected_model'),
                'correct_selection': r.get('correct', False),
                'confidence': r.get('confidence', 0),
                'task_type': r.get('analysis', {}).get('task_type'),
                'complexity_score': r.get('analysis', {}).get('complexity_score'),
                'generation_time': r.get('result', {}).get('generation_time', 0)
            }
            for r in results if r.get('selected_model')
        ],
        'analysis_summary': analysis_results,
        'engine_statistics': stats,
        'performance_metrics': {
            'overall_accuracy': analysis_results.get('accuracy', 0),
            'average_confidence': analysis_results.get('avg_confidence', 0),
            'total_tests': analysis_results.get('total_tests', 0),
            'models_used': len(engine.models)
        }
    }
    
    # Add comparison results if available
    if comparison_results:
        experiment_data['single_vs_multi_comparison'] = {
            'single_model_success_rate': sum(1 for r in comparison_results['single_model'] if r['success']) / len(comparison_results['single_model']),
            'multi_model_success_rate': sum(1 for r in comparison_results['multi_model'] if r['success']) / len(comparison_results['multi_model']),
            'single_model_avg_time': np.mean([r['time'] for r in comparison_results['single_model'] if r['success']]),
            'multi_model_avg_time': np.mean([r['time'] for r in comparison_results['multi_model'] if r['success']])
        }
    
    # Save to file
    output_file = Path(f"multi_model_experiment_{experiment_data['timestamp']}.json")
    with open(output_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"📁 Comprehensive results saved to {output_file}")
    
    # Create summary report
    summary_file = Path(f"multi_model_summary_{experiment_data['timestamp']}.txt")
    with open(summary_file, 'w') as f:
        f.write("Multi-Model Inference Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Experiment Date: {experiment_data['timestamp']}\n")
        f.write(f"Total Models: {len(engine.models)}\n")
        f.write(f"Total Tests: {analysis_results.get('total_tests', 0)}\n\n")
        
        f.write("Model Configurations:\n")
        for model_id, config in experiment_data['model_configurations'].items():
            f.write(f"  • {model_id}: {config['description']}\n")
            f.write(f"    Task Types: {', '.join(config['task_types'])}\n")
            f.write(f"    Domains: {', '.join(config['specialized_domains'])}\n\n")
        
        f.write(f"Overall Performance:\n")
        f.write(f"  Selection Accuracy: {analysis_results.get('accuracy', 0):.2%}\n")
        f.write(f"  Average Confidence: {analysis_results.get('avg_confidence', 0):.3f}\n\n")
        
        f.write("Task Type Performance:\n")
        for task_type, perf in analysis_results.get('task_type_performance', {}).items():
            accuracy = perf['correct'] / perf['total']
            f.write(f"  {task_type}: {accuracy:.2%} ({perf['correct']}/{perf['total']})\n")
        
        f.write(f"\nModel Usage:\n")
        for model, count in analysis_results.get('model_usage', {}).items():
            percentage = (count / analysis_results.get('total_tests', 1)) * 100
            f.write(f"  {model}: {count} times ({percentage:.1f}%)\n")
    
    print(f"📄 Summary report saved to {summary_file}")
    
    return output_file, summary_file

def main():
    """Main demonstration function for multi-model inference"""
    
    print("🚀 Multi-Model Reward-Guided Inference Demonstration")
    print("=" * 70)
    print("This enhanced example demonstrates:")
    print("• Automatic model selection based on input analysis")
    print("• Multiple specialized models for different task types")
    print("• Adaptive configuration based on problem complexity")
    print("• Load balancing and routing optimization")
    print("• Performance comparison between single and multi-model approaches")
    print("• Comprehensive analysis and visualization")
    print("=" * 70)
    
    try:
        # Step 1: Demonstrate multi-model inference
        print("\n🎯 Step 1: Multi-Model Inference Testing")
        engine, results = demonstrate_multi_model_inference()
        
        # Step 2: Analyze model selection performance
        print("\n📊 Step 2: Selection Performance Analysis")
        analysis_results = analyze_model_selection_performance(results)
        
        # Step 3: Compare single vs multi-model approaches
        print("\n⚖️ Step 3: Single vs Multi-Model Comparison")
        comparison_results = compare_single_vs_multi_model()
        
        # Step 4: Demonstrate adaptive routing
        print("\n🔄 Step 4: Adaptive Routing with Load Balancing")
        balanced_engine, routing_results = demonstrate_adaptive_routing()
        
        # Step 5: Create visualizations
        print("\n📈 Step 5: Performance Visualization")
        fig = visualize_multi_model_performance(engine, results, analysis_results)
        
        # Step 6: Save comprehensive results
        print("\n💾 Step 6: Saving Results")
        output_file, summary_file = save_multi_model_results(
            engine, results, analysis_results, comparison_results
        )
        
        # Final summary
        print("\n🎉 Multi-Model Experiment Complete!")
        print("=" * 50)
        print(f"✅ Tested {len(engine.models)} specialized models")
        print(f"✅ Evaluated {len(results)} different prompts")
        print(f"✅ Achieved {analysis_results.get('accuracy', 0):.2%} selection accuracy")
        print(f"✅ Generated performance analysis and visualizations")
        print(f"✅ Saved results to {output_file}")
        print(f"✅ Created summary report: {summary_file}")
        
        # Best performing model
        model_usage = analysis_results.get('model_usage', {})
        if model_usage:
            most_used = max(model_usage.keys(), key=lambda k: model_usage[k])
            print(f"\n🏆 Most utilized model: {most_used}")
            print(f"   Usage: {model_usage[most_used]} times ({(model_usage[most_used]/len(results))*100:.1f}%)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if analysis_results.get('accuracy', 0) > 0.8:
            print("   • Model selection system is performing well")
        else:
            print("   • Consider improving input classification accuracy")
        
        if analysis_results.get('avg_confidence', 0) < 0.7:
            print("   • Consider adjusting model selection thresholds")
        else:
            print("   • Model selection confidence is satisfactory")
        
    except Exception as e:
        print(f"❌ Multi-model experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()