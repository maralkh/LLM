#!/usr/bin/env python3
"""
Enhanced Multi-Model Inference with Manifold Learning - Fixed Version
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import UMAP
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path


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
class ManifoldLearningConfig:
    """Configuration for manifold learning"""
    embedding_dim: int = 32
    online_batch_size: int = 16
    memory_size: int = 500
    manifold_method: str = "umap"
    enable_online_learning: bool = True
    enable_clustering: bool = True
    similarity_threshold: float = 0.7


@dataclass
class DataPoint:
    """Represents a data point in the manifold space"""
    text: str
    embedding: np.ndarray
    task_type: TaskType
    selected_model: str
    performance_score: float
    timestamp: float
    complexity: float
    cluster_id: int = -1


@dataclass
class InputAnalysis:
    """Analysis results for input text"""
    task_type: TaskType
    confidence: float
    complexity_score: float
    keywords: List[str]
    domain_indicators: List[str]
    manifold_insights: Dict[str, Any]


class DummyTokenizer:
    """Simple tokenizer for demonstration"""
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


class DummyModel(nn.Module):
    """Simple dummy model for demonstration"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32000, 512)
        self.linear = nn.Linear(512, 32000)
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        return self.linear(embeddings.mean(dim=1))


class ManifoldLearner:
    """Manifold learning for input distribution analysis"""
    
    def __init__(self, config: ManifoldLearningConfig):
        self.config = config
        self.data_points = []
        
        # Feature extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.feature_scaler = StandardScaler()
        
        # Manifold learning
        self.manifold_model = None
        self.clustering_model = MiniBatchKMeans(n_clusters=5, random_state=42)
        self.nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        
        self.manifold_fitted = False
        self.cluster_performance = {}
        
    def extract_features(self, text: str) -> np.ndarray:
        """Extract features from text"""
        try:
            if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()[0]
            else:
                tfidf_features = self.tfidf_vectorizer.fit_transform([text]).toarray()[0]
        except:
            tfidf_features = np.zeros(500)
        
        # Add simple linguistic features
        features = [
            len(text),
            len(text.split()),
            text.count('?'),
            text.count('!'),
            len(re.findall(r'[0-9]', text)) / max(len(text), 1),
            len(re.findall(r'[+\-*/=]', text)) / max(len(text), 1)
        ]
        
        return np.concatenate([tfidf_features, features])
    
    def learn_manifold_offline(self, texts: List[str]) -> None:
        """Learn manifold structure from offline data"""
        print(f"🧠 Learning manifold from {len(texts)} samples...")
        
        features_list = [self.extract_features(text) for text in texts]
        features_matrix = np.array(features_list)
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features_matrix)
        
        # Learn manifold
        self.manifold_model = UMAP(
            n_components=self.config.embedding_dim,
            random_state=42,
            n_neighbors=10
        )
        
        embeddings = self.manifold_model.fit_transform(scaled_features)
        
        # Clustering
        if self.config.enable_clustering:
            cluster_labels = self.clustering_model.fit_predict(embeddings)
            
            for i, (text, embedding, cluster_id) in enumerate(zip(texts, embeddings, cluster_labels)):
                data_point = DataPoint(
                    text=text,
                    embedding=embedding,
                    task_type=TaskType.CONVERSATIONAL,
                    selected_model="unknown",
                    performance_score=0.5,
                    timestamp=time.time(),
                    complexity=0.5,
                    cluster_id=cluster_id
                )
                self.data_points.append(data_point)
        
        self.nn_model.fit(embeddings)
        self.manifold_fitted = True
        print(f"✅ Manifold learning completed")
    
    def get_recommendations(self, text: str) -> Dict[str, Any]:
        """Get recommendations based on manifold analysis"""
        if not self.manifold_fitted:
            return {
                'recommended_models': ['general_model'],
                'manifold_confidence': 0.5,
                'similar_points': 0,
                'cluster_info': {}
            }
        
        features = self.extract_features(text)
        scaled_features = self.feature_scaler.transform([features])
        embedding = self.manifold_model.transform(scaled_features)[0]
        
        # Find similar points
        distances, indices = self.nn_model.kneighbors([embedding])
        similar_points = [self.data_points[i] for i in indices[0] if i < len(self.data_points)]
        
        # Calculate confidence based on similarity
        avg_distance = np.mean(distances[0])
        confidence = max(0.1, 1.0 - avg_distance)
        
        return {
            'recommended_models': ['math_specialist'],  # Simplified
            'manifold_confidence': confidence,
            'similar_points': len(similar_points),
            'cluster_info': {'cluster_id': 0}
        }
    
    def update_online(self, text: str, task_type: TaskType, selected_model: str, performance_score: float):
        """Update with online feedback"""
        if not self.config.enable_online_learning:
            return
        
        features = self.extract_features(text)
        scaled_features = self.feature_scaler.transform([features])
        
        if self.manifold_fitted:
            embedding = self.manifold_model.transform(scaled_features)[0]
        else:
            embedding = np.random.normal(0, 0.1, self.config.embedding_dim)
        
        data_point = DataPoint(
            text=text,
            embedding=embedding,
            task_type=task_type,
            selected_model=selected_model,
            performance_score=performance_score,
            timestamp=time.time(),
            complexity=self._estimate_complexity(text),
            cluster_id=0
        )
        
        self.data_points.append(data_point)
    
    def _estimate_complexity(self, text: str) -> float:
        """Simple complexity estimation"""
        word_count = len(text.split())
        return min(word_count / 50, 1.0)


class InputClassifier:
    """Classify input text to determine task type"""
    
    def __init__(self, manifold_config: ManifoldLearningConfig = None):
        self.manifold_config = manifold_config or ManifoldLearningConfig()
        self.manifold_learner = ManifoldLearner(self.manifold_config)
        
        self.task_patterns = {
            TaskType.MATHEMATICAL: {
                'keywords': ['solve', 'calculate', 'equation', 'derivative', 'integral'],
                'patterns': [r'\d+[x-z]\s*[+\-*/=]', r'[∫∑∏√]']
            },
            TaskType.CREATIVE_WRITING: {
                'keywords': ['story', 'write', 'creative', 'poem', 'character'],
                'patterns': [r'write\s+a\s+story', r'create\s+a\s+character']
            },
            TaskType.CODE_GENERATION: {
                'keywords': ['code', 'function', 'python', 'algorithm', 'implement'],
                'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+']
            },
            TaskType.REASONING: {
                'keywords': ['analyze', 'compare', 'evaluate', 'pros', 'cons'],
                'patterns': [r'pros\s+and\s+cons', r'advantages\s+and\s+disadvantages']
            },
            TaskType.FACTUAL_QA: {
                'keywords': ['what', 'who', 'when', 'where', 'explain'],
                'patterns': [r'^(what|who|when|where|why|how)\s+']
            }
        }
    
    def analyze_input(self, text: str) -> InputAnalysis:
        """Analyze input text"""
        text_lower = text.lower()
        
        # Calculate task scores
        task_scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = 0
            score += sum(2 for keyword in patterns['keywords'] if keyword in text_lower)
            score += sum(3 for pattern in patterns['patterns'] if re.search(pattern, text_lower))
            task_scores[task_type] = score
        
        # Determine best task
        best_task = max(task_scores.keys(), key=lambda k: task_scores[k])
        confidence = task_scores[best_task] / max(sum(task_scores.values()), 1)
        
        # Get manifold recommendations
        manifold_rec = self.manifold_learner.get_recommendations(text)
        
        return InputAnalysis(
            task_type=best_task,
            confidence=confidence,
            complexity_score=min(len(text) / 500, 1.0),
            keywords=text.split()[:3],
            domain_indicators=['general'],
            manifold_insights=manifold_rec
        )
    
    def train_manifold_offline(self, training_data: List[Dict]):
        """Train manifold learning"""
        texts = [item['text'] for item in training_data]
        self.manifold_learner.learn_manifold_offline(texts)
    
    def update_with_feedback(self, text: str, task_type: TaskType, selected_model: str, performance_score: float):
        """Update with feedback"""
        self.manifold_learner.update_online(text, task_type, selected_model, performance_score)


class MultiModelInferenceEngine:
    """Multi-model inference engine with manifold learning"""
    
    def __init__(self, models: List[Dict], tokenizer, default_model_id: str = "general_model",
                 manifold_config: ManifoldLearningConfig = None):
        self.models = {model['id']: model for model in models}
        self.tokenizer = tokenizer
        self.default_model_id = default_model_id
        
        self.manifold_config = manifold_config or ManifoldLearningConfig()
        self.classifier = InputClassifier(self.manifold_config)
        
        self.usage_stats = {model_id: 0 for model_id in self.models.keys()}
        self.performance_history = []
        self.selection_history = deque(maxlen=1000)
        
    def initialize_manifold_learning(self, training_data: List[Dict] = None):
        """Initialize manifold learning"""
        print("🧠 Initializing Manifold Learning System")
        
        if not training_data:
            training_data = self._create_synthetic_training_data()
        
        self.classifier.train_manifold_offline(training_data)
        print("✅ Manifold learning initialization completed")
    
    def _create_synthetic_training_data(self) -> List[Dict]:
        """Create synthetic training data"""
        data = []
        
        # Math examples
        math_examples = [
            "Solve: 2x + 5 = 11",
            "Calculate the derivative of x^2",
            "Find the integral of sin(x)",
            "What is the limit of x as x approaches 0?"
        ]
        
        for text in math_examples:
            data.append({'text': text, 'task_type': 'mathematical'})
        
        # Creative examples
        creative_examples = [
            "Write a story about a robot",
            "Create a poem about AI",
            "Describe a character who travels through time"
        ]
        
        for text in creative_examples:
            data.append({'text': text, 'task_type': 'creative'})
        
        # Code examples
        code_examples = [
            "Write a Python function for sorting",
            "Implement binary search algorithm",
            "Create a class for managing data"
        ]
        
        for text in code_examples:
            data.append({'text': text, 'task_type': 'code'})
        
        return data
    
    def select_model(self, input_text: str) -> Tuple[str, InputAnalysis, float]:
        """Select best model for input"""
        analysis = self.classifier.analyze_input(input_text)
        
        # Simple model selection logic
        model_scores = {}
        for model_id, model_spec in self.models.items():
            score = 0.5  # Base score
            
            # Task compatibility
            if analysis.task_type.value in model_spec.get('task_types', []):
                score += 0.3
            
            # Manifold recommendation
            manifold_rec = analysis.manifold_insights.get('recommended_models', [])
            if model_id in manifold_rec:
                score += analysis.manifold_insights.get('manifold_confidence', 0) * 0.2
            
            model_scores[model_id] = score
        
        best_model_id = max(model_scores.keys(), key=lambda k: model_scores[k])
        confidence = model_scores[best_model_id]
        
        # Fallback to default if confidence too low
        if confidence < 0.3:
            best_model_id = self.default_model_id
            confidence = 0.3
        
        return best_model_id, analysis, confidence
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7,
                 force_model: str = None) -> Dict[str, Any]:
        """Generate text using selected model"""
        start_time = time.time()
        
        if force_model and force_model in self.models:
            selected_model_id = force_model
            analysis = self.classifier.analyze_input(prompt)
            selection_confidence = 1.0
        else:
            selected_model_id, analysis, selection_confidence = self.select_model(prompt)
        
        self.usage_stats[selected_model_id] += 1
        
        # Simulate generation
        generation_time = time.time() - start_time + np.random.uniform(0.1, 0.5)
        
        # Calculate performance score
        performance_score = selection_confidence * np.random.uniform(0.7, 1.0)
        
        # Update manifold learning
        self.classifier.update_with_feedback(
            prompt, analysis.task_type, selected_model_id, performance_score
        )
        
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            'model_id': selected_model_id,
            'task_type': analysis.task_type.value,
            'performance_score': performance_score,
            'generation_time': generation_time,
            'manifold_confidence': analysis.manifold_insights.get('manifold_confidence', 0)
        })
        
        return {
            'text': f"[Generated response to: {prompt[:50]}...]",
            'selected_model': selected_model_id,
            'model_selection_confidence': selection_confidence,
            'generation_time': generation_time,
            'input_analysis': {
                'task_type': analysis.task_type.value,
                'complexity_score': analysis.complexity_score,
                'manifold_insights': analysis.manifold_insights
            }
        }
    
    def analyze_manifold_performance(self) -> Dict[str, Any]:
        """Analyze manifold learning performance"""
        print("\n🧠 Manifold Learning Performance Analysis")
        
        if not self.performance_history:
            return {}
        
        recent_data = self.performance_history[-50:]
        
        manifold_confidences = [p['manifold_confidence'] for p in recent_data]
        performance_scores = [p['performance_score'] for p in recent_data]
        
        if manifold_confidences and performance_scores:
            correlation = np.corrcoef(manifold_confidences, performance_scores)[0, 1]
        else:
            correlation = 0.0
        
        analysis = {
            'total_data_points': len(recent_data),
            'manifold_confidence_correlation': correlation,
            'avg_manifold_confidence': np.mean(manifold_confidences) if manifold_confidences else 0,
            'avg_performance_score': np.mean(performance_scores) if performance_scores else 0,
            'model_selection_accuracy': sum(1 for p in recent_data if p['performance_score'] > 0.7) / len(recent_data)
        }
        
        print(f"Data Points: {analysis['total_data_points']}")
        print(f"Correlation: {correlation:.3f}")
        print(f"Avg Confidence: {analysis['avg_manifold_confidence']:.3f}")
        print(f"Selection Accuracy: {analysis['model_selection_accuracy']:.2%}")
        
        return analysis
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        total_usage = sum(self.usage_stats.values())
        return {
            'usage_distribution': {
                model_id: (count / max(total_usage, 1)) * 100 
                for model_id, count in self.usage_stats.items()
            },
            'total_generations': total_usage
        }
    
    def save_manifold_state(self, filepath: str):
        """Save manifold state"""
        state = {
            'performance_history': self.performance_history,
            'usage_stats': self.usage_stats,
            'manifold_data_points': self.classifier.manifold_learner.data_points[-100:]  # Last 100
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 Manifold state saved to {filepath}")


def create_specialized_models() -> List[Dict]:
    """Create model specifications"""
    return [
        {
            'id': 'math_specialist',
            'model': DummyModel(),
            'task_types': ['mathematical', 'scientific'],
            'description': 'Specialized for math and science'
        },
        {
            'id': 'creative_specialist', 
            'model': DummyModel(),
            'task_types': ['creative_writing'],
            'description': 'Specialized for creative tasks'
        },
        {
            'id': 'code_specialist',
            'model': DummyModel(),
            'task_types': ['code_generation'],
            'description': 'Specialized for coding'
        },
        {
            'id': 'general_model',
            'model': DummyModel(),
            'task_types': ['factual_qa', 'conversational', 'reasoning'],
            'description': 'General purpose model'
        }
    ]


def demonstrate_manifold_learning():
    """Demonstrate manifold learning capabilities"""
    print("🧠 Manifold Learning Demonstration")
    print("=" * 50)
    
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    
    manifold_config = ManifoldLearningConfig(
        embedding_dim=16,
        online_batch_size=8,
        memory_size=100
    )
    
    engine = MultiModelInferenceEngine(
        models=models,
        tokenizer=tokenizer,
        manifold_config=manifold_config
    )
    
    engine.initialize_manifold_learning()
    
    test_prompts = [
        "Calculate the integral of x^2 dx",
        "Write a story about AI",
        "Implement binary search in Python", 
        "What causes earthquakes?",
        "Solve: 3x + 2 = 11",
        "Create a poem about technology"
    ]
    
    print(f"\n🎯 Testing {len(test_prompts)} prompts...")
    
    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] {prompt[:50]}...")
        
        try:
            result = engine.generate(prompt, max_length=100)
            
            print(f"  Model: {result['selected_model']}")
            print(f"  Confidence: {result['model_selection_confidence']:.3f}")
            print(f"  Task: {result['input_analysis']['task_type']}")
            
            manifold_insights = result['input_analysis']['manifold_insights']
            print(f"  Manifold Confidence: {manifold_insights['manifold_confidence']:.3f}")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Analyze performance
    analysis = engine.analyze_manifold_performance()
    
    return engine, results, analysis


def demonstrate_multi_model_inference():
    """Demonstrate traditional multi-model inference"""
    print("\n🎯 Traditional Multi-Model Inference")
    print("=" * 40)
    
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    
    # Create engine without manifold learning
    engine = MultiModelInferenceEngine(models=models, tokenizer=tokenizer)
    
    test_prompts = [
        "Solve: x^2 + 5x + 6 = 0",
        "Write a creative story",
        "Code a sorting function",
        "Explain photosynthesis"
    ]
    
    results = []
    for prompt in test_prompts:
        try:
            result = engine.generate(prompt)
            results.append(result)
        except Exception as e:
            print(f"Failed: {e}")
    
    return engine, results


def compare_manifold_vs_traditional(manifold_results: List[Dict], 
                                  traditional_results: List[Dict],
                                  manifold_analysis: Dict) -> Dict[str, Any]:
    """Compare approaches"""
    print("⚖️ Comparing Approaches")
    
    manifold_perf = np.mean([r.get('model_selection_confidence', 0) for r in manifold_results])
    traditional_perf = np.mean([r.get('model_selection_confidence', 0) for r in traditional_results])
    
    improvement = ((manifold_perf - traditional_perf) / max(traditional_perf, 0.001)) * 100
    
    comparison = {
        'manifold_avg_performance': manifold_perf,
        'traditional_avg_performance': traditional_perf,
        'performance_improvement': improvement,
        'manifold_samples': len(manifold_results),
        'traditional_samples': len(traditional_results)
    }
    
    print(f"Manifold: {manifold_perf:.3f}")
    print(f"Traditional: {traditional_perf:.3f}")
    print(f"Improvement: {improvement:+.2f}%")
    
    return comparison


def main():
    """Enhanced main demonstration function"""
    print("🚀 Advanced Multi-Model Inference with Manifold Learning")
    print("=" * 75)
    
    try:
        # Step 1: Manifold learning demo
        print("\n🧠 Step 1: Manifold Learning System")
        engine, manifold_results, manifold_analysis = demonstrate_manifold_learning()
        
        # Step 2: Traditional approach
        print("\n🎯 Step 2: Traditional Multi-Model Inference")
        traditional_engine, traditional_results = demonstrate_multi_model_inference()
        
        # Step 3: Comparison
        print("\n⚖️ Step 3: Performance Comparison")
        comparison = compare_manifold_vs_traditional(
            manifold_results, traditional_results, manifold_analysis
        )
        
        # Step 4: Save results
        print("\n💾 Step 4: Saving Results")
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        
        results = {
            'timestamp': timestamp,
            'manifold_results': manifold_results[:5],  # Save sample
            'traditional_results': traditional_results[:5],
            'comparison': comparison,
            'manifold_analysis': manifold_analysis
        }
        
        output_file = f"enhanced_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model state
        model_file = f"manifold_model_{timestamp}.pkl"
        engine.save_manifold_state(model_file)
        
        # Final summary
        print("\n🎉 Experiment Complete!")
        print("=" * 30)
        print(f"✅ Manifold tests: {len(manifold_results)}")
        print(f"✅ Traditional tests: {len(traditional_results)}")
        print(f"✅ Performance improvement: {comparison['performance_improvement']:+.1f}%")
        print(f"✅ Results saved to: {output_file}")
        print(f"✅ Model saved to: {model_file}")
        
        correlation = manifold_analysis.get('manifold_confidence_correlation', 0)
        print(f"\n💡 Manifold correlation: {correlation:.3f}")
        
        if correlation > 0.3:
            print("   • Manifold learning is effective")
        else:
            print("   • Consider more training data")
            
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()