# training_infra/data/synthetic.py
"""
Synthetic Data Generation for training language models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import json
import random
import re
import numpy as np
from pathlib import Path
import itertools
from abc import ABC, abstractmethod

from ..inference.engine import InferenceEngine, GenerationConfig
from ..inference.sampling import SamplingConfig

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    # Generation parameters
    num_samples: int = 1000
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: Optional[int] = 50
    
    # Quality control
    min_quality_score: float = 0.7
    use_quality_filter: bool = True
    use_diversity_filter: bool = True
    diversity_threshold: float = 0.8
    
    # Data types
    include_instructions: bool = True
    include_conversations: bool = True
    include_reasoning: bool = True
    include_creative: bool = True
    
    # Templates and prompts
    template_file: Optional[str] = None
    custom_prompts: List[str] = field(default_factory=list)
    
    # Self-improvement
    use_self_critique: bool = True
    critique_iterations: int = 2
    use_constitutional_ai: bool = False
    
    # Augmentation
    use_paraphrasing: bool = True
    use_back_translation: bool = False
    augmentation_ratio: float = 0.3

class SyntheticDataGenerator:
    """Main class for generating synthetic training data"""
    
    def __init__(self, 
                 teacher_model: nn.Module,
                 tokenizer,
                 config: SyntheticDataConfig,
                 quality_model: Optional[nn.Module] = None):
        
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.config = config
        self.quality_model = quality_model
        
        # Create inference engine
        self.engine = InferenceEngine(teacher_model, tokenizer)
        
        # Load templates
        self.templates = self._load_templates()
        
        # Statistics
        self.generation_stats = {
            'total_generated': 0,
            'filtered_out': 0,
            'quality_scores': [],
            'diversity_scores': []
        }
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load generation templates"""
        
        default_templates = {
            'instruction_following': [
                "Write a detailed explanation of {topic}.",
                "Provide step-by-step instructions for {task}.",
                "Compare and contrast {concept1} and {concept2}.",
                "Explain the benefits and drawbacks of {approach}.",
                "Create a comprehensive guide about {subject}."
            ],
            
            'conversation': [
                "Human: Can you help me understand {topic}?\nAssistant:",
                "Human: What's the difference between {concept1} and {concept2}?\nAssistant:",
                "Human: How do I {task}?\nAssistant:",
                "Human: Why is {phenomenon} important?\nAssistant:",
                "Human: Can you give me advice on {situation}?\nAssistant:"
            ],
            
            'reasoning': [
                "Solve this problem step by step: {problem}",
                "Analyze this scenario: {scenario}",
                "What would happen if {hypothetical}?",
                "Explain the reasoning behind {decision}.",
                "Break down this complex issue: {issue}"
            ],
            
            'creative': [
                "Write a short story about {theme}.",
                "Create a poem inspired by {concept}.",
                "Imagine a world where {premise}. Describe it.",
                "Write a dialogue between {character1} and {character2}.",
                "Compose a creative essay on {topic}."
            ],
            
            'technical': [
                "Implement a {algorithm} in Python.",
                "Debug this code: {code_snippet}",
                "Optimize this function: {function}",
                "Explain this technical concept: {concept}",
                "Design a system for {requirement}."
            ]
        }
        
        # Load custom templates if provided
        if self.config.template_file and Path(self.config.template_file).exists():
            with open(self.config.template_file, 'r') as f:
                custom_templates = json.load(f)
                default_templates.update(custom_templates)
        
        return default_templates
    
    def generate_synthetic_dataset(self) -> List[Dict[str, Any]]:
        """Generate a complete synthetic dataset"""
        
        print(f"🔄 Generating {self.config.num_samples} synthetic examples...")
        
        synthetic_data = []
        
        # Generate different types of data
        data_types = []
        if self.config.include_instructions:
            data_types.append('instruction_following')
        if self.config.include_conversations:
            data_types.append('conversation')
        if self.config.include_reasoning:
            data_types.append('reasoning')
        if self.config.include_creative:
            data_types.append('creative')
        
        samples_per_type = self.config.num_samples // len(data_types)
        
        for data_type in data_types:
            type_samples = self._generate_data_type(data_type, samples_per_type)
            synthetic_data.extend(type_samples)
        
        # Add custom prompt samples
        if self.config.custom_prompts:
            custom_samples = self._generate_from_custom_prompts()
            synthetic_data.extend(custom_samples)
        
        # Apply quality filtering
        if self.config.use_quality_filter:
            synthetic_data = self._filter_by_quality(synthetic_data)
        
        # Apply diversity filtering
        if self.config.use_diversity_filter:
            synthetic_data = self._filter_by_diversity(synthetic_data)
        
        # Apply augmentation
        if self.config.use_paraphrasing:
            synthetic_data = self._apply_augmentation(synthetic_data)
        
        print(f"✅ Generated {len(synthetic_data)} high-quality synthetic examples")
        print(f"📊 Statistics: {self.generation_stats}")
        
        return synthetic_data
    
    def _generate_data_type(self, data_type: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples for a specific data type"""
        
        samples = []
        templates = self.templates.get(data_type, [])
        
        if not templates:
            return samples
        
        # Topic/concept lists for filling templates
        topics = self._get_topic_list(data_type)
        
        for i in range(num_samples):
            try:
                # Select random template
                template = random.choice(templates)
                
                # Fill template with random topics
                filled_prompt = self._fill_template(template, topics, data_type)
                
                # Generate response
                response = self._generate_response(filled_prompt)
                
                # Post-process if needed
                if self.config.use_self_critique:
                    response = self._apply_self_critique(filled_prompt, response)
                
                sample = {
                    'prompt': filled_prompt,
                    'response': response,
                    'data_type': data_type,
                    'template': template,
                    'generation_params': {
                        'temperature': self.config.temperature,
                        'top_p': self.config.top_p
                    }
                }
                
                samples.append(sample)
                self.generation_stats['total_generated'] += 1
                
            except Exception as e:
                print(f"⚠️ Failed to generate sample {i}: {e}")
                continue
        
        return samples
    
    def _get_topic_list(self, data_type: str) -> Dict[str, List[str]]:
        """Get topic lists for different data types"""
        
        topics = {
            'topic': [
                'artificial intelligence', 'climate change', 'quantum computing',
                'renewable energy', 'space exploration', 'biotechnology',
                'cybersecurity', 'machine learning', 'blockchain', 'robotics',
                'nanotechnology', 'gene therapy', 'virtual reality', 'solar power'
            ],
            
            'task': [
                'writing a research paper', 'learning a new language', 'starting a business',
                'cooking a healthy meal', 'exercising regularly', 'managing time effectively',
                'building a website', 'investing money', 'learning programming', 'gardening'
            ],
            
            'concept1': [
                'artificial intelligence', 'machine learning', 'supervised learning',
                'neural networks', 'democracy', 'capitalism', 'renewable energy'
            ],
            
            'concept2': [
                'human intelligence', 'traditional programming', 'unsupervised learning',
                'decision trees', 'autocracy', 'socialism', 'fossil fuels'
            ],
            
            'problem': [
                'A train travels 120 km in 2 hours. What is its average speed?',
                'Find the area of a circle with radius 5 meters.',
                'Solve for x: 2x + 5 = 15',
                'Calculate the compound interest on $1000 at 5% for 3 years.',
                'A rectangle has length 8m and width 6m. Find its perimeter.'
            ],
            
            'theme': [
                'friendship in difficult times', 'the last tree on Earth',
                'a world without colors', 'time travel gone wrong',
                'the secret life of objects', 'dreams becoming reality'
            ]
        }
        
        # Add data-type specific topics
        if data_type == 'technical':
            topics.update({
                'algorithm': ['binary search', 'quicksort', 'dijkstra', 'A*', 'dynamic programming'],
                'concept': ['recursion', 'object-oriented programming', 'database normalization']
            })
        
        return topics
    
    def _fill_template(self, template: str, topics: Dict[str, List[str]], data_type: str) -> str:
        """Fill template with random topics"""
        
        filled = template
        
        # Find all placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        for placeholder in placeholders:
            if placeholder in topics:
                replacement = random.choice(topics[placeholder])
                filled = filled.replace(f'{{{placeholder}}}', replacement)
        
        return filled
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using teacher model"""
        
        # Create generation config
        gen_config = GenerationConfig(
            sampling=SamplingConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True
            ),
            max_new_tokens=self.config.max_length
        )
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate
        result = self.engine.generate(input_ids, gen_config)
        
        # Decode response (skip input tokens)
        generated_ids = result['sequences'][0][input_ids.size(1):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def _apply_self_critique(self, prompt: str, response: str) -> str:
        """Apply self-critique to improve response quality"""
        
        improved_response = response
        
        for iteration in range(self.config.critique_iterations):
            # Create critique prompt
            critique_prompt = f"""
            Original Question: {prompt}
            Current Answer: {improved_response}
            
            Please critique this answer and suggest improvements. Focus on:
            1. Accuracy and completeness
            2. Clarity and organization
            3. Helpful details or examples
            
            Provide an improved version of the answer:
            """
            
            try:
                # Generate improved response
                improved_response = self._generate_response(critique_prompt)
                
                # Basic quality check
                if len(improved_response) < len(response) * 0.5:
                    # If significantly shorter, keep original
                    break
                    
            except Exception as e:
                print(f"⚠️ Self-critique failed: {e}")
                break
        
        return improved_response
    
    def _generate_from_custom_prompts(self) -> List[Dict[str, Any]]:
        """Generate samples from custom prompts"""
        
        samples = []
        
        for prompt in self.config.custom_prompts:
            try:
                response = self._generate_response(prompt)
                
                sample = {
                    'prompt': prompt,
                    'response': response,
                    'data_type': 'custom',
                    'template': 'custom_prompt'
                }
                
                samples.append(sample)
                
            except Exception as e:
                print(f"⚠️ Failed to generate from custom prompt: {e}")
                continue
        
        return samples
    
    def _filter_by_quality(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter samples by quality score"""
        
        if not self.quality_model:
            return samples
        
        filtered_samples = []
        
        for sample in samples:
            try:
                quality_score = self._compute_quality_score(sample)
                
                if quality_score >= self.config.min_quality_score:
                    sample['quality_score'] = quality_score
                    filtered_samples.append(sample)
                    self.generation_stats['quality_scores'].append(quality_score)
                else:
                    self.generation_stats['filtered_out'] += 1
                    
            except Exception as e:
                print(f"⚠️ Quality scoring failed: {e}")
                # Keep sample if scoring fails
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _compute_quality_score(self, sample: Dict[str, Any]) -> float:
        """Compute quality score for a sample"""
        
        if not self.quality_model:
            return 1.0
        
        # Combine prompt and response
        full_text = f"{sample['prompt']} {sample['response']}"
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, return_tensors='pt', truncation=True, max_length=512)
        
        # Get quality score
        with torch.no_grad():
            quality_output = self.quality_model.get_reward(tokens)
            quality_score = torch.sigmoid(quality_output).item()  # Normalize to 0-1
        
        return quality_score
    
    def _filter_by_diversity(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter samples to ensure diversity"""
        
        if not samples:
            return samples
        
        # Simple diversity filtering based on response similarity
        filtered_samples = [samples[0]]  # Always keep first sample
        
        for sample in samples[1:]:
            is_diverse = True
            
            for existing_sample in filtered_samples:
                similarity = self._compute_similarity(
                    sample['response'], 
                    existing_sample['response']
                )
                
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered_samples.append(sample)
                self.generation_stats['diversity_scores'].append(1.0)
            else:
                self.generation_stats['filtered_out'] += 1
                self.generation_stats['diversity_scores'].append(0.0)
        
        return filtered_samples
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        
        # Simple token-based similarity
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _apply_augmentation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply data augmentation techniques"""
        
        augmented_samples = samples.copy()
        
        if self.config.use_paraphrasing:
            augmented_samples.extend(self._paraphrase_samples(samples))
        
        if self.config.use_back_translation:
            augmented_samples.extend(self._back_translate_samples(samples))
        
        return augmented_samples
    
    def _paraphrase_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate paraphrased versions of samples"""
        
        paraphrased = []
        num_to_paraphrase = int(len(samples) * self.config.augmentation_ratio)
        
        selected_samples = random.sample(samples, min(num_to_paraphrase, len(samples)))
        
        for sample in selected_samples:
            try:
                paraphrase_prompt = f"""
                Rewrite the following text while preserving its meaning:
                Original: {sample['response']}
                Rewritten:
                """
                
                paraphrased_response = self._generate_response(paraphrase_prompt)
                
                paraphrased_sample = {
                    'prompt': sample['prompt'],
                    'response': paraphrased_response,
                    'data_type': sample['data_type'] + '_paraphrased',
                    'template': sample['template'],
                    'original_response': sample['response']
                }
                
                paraphrased.append(paraphrased_sample)
                
            except Exception as e:
                print(f"⚠️ Paraphrasing failed: {e}")
                continue
        
        return paraphrased
    
    def _back_translate_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply back-translation for augmentation (placeholder implementation)"""
        
        # This would require translation models
        # For now, return empty list
        return []

class ConstitutionalAIGenerator:
    """Constitutional AI approach for synthetic data generation"""
    
    def __init__(self, model, tokenizer, principles: List[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.principles = principles
        self.engine = InferenceEngine(model, tokenizer)
    
    def generate_constitutional_data(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate data following constitutional principles"""
        
        constitutional_data = []
        
        for prompt in prompts:
            # Initial response
            initial_response = self._generate_response(prompt)
            
            # Apply constitutional critique
            improved_response = self._apply_constitutional_critique(prompt, initial_response)
            
            constitutional_data.append({
                'prompt': prompt,
                'initial_response': initial_response,
                'constitutional_response': improved_response,
                'principles_applied': self.principles
            })
        
        return constitutional_data
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using model"""
        
        gen_config = GenerationConfig(
            sampling=SamplingConfig(temperature=0.7, top_p=0.9),
            max_new_tokens=256
        )
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        result = self.engine.generate(input_ids, gen_config)
        
        generated_ids = result['sequences'][0][input_ids.size(1):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    def _apply_constitutional_critique(self, prompt: str, response: str) -> str:
        """Apply constitutional principles to improve response"""
        
        for principle in self.principles:
            critique_prompt = f"""
            Original prompt: {prompt}
            Current response: {response}
            
            Constitutional principle: {principle}
            
            Please revise the response to better follow this principle while maintaining helpfulness and accuracy:
            """
            
            try:
                response = self._generate_response(critique_prompt)
            except Exception as e:
                print(f"⚠️ Constitutional critique failed: {e}")
                continue
        
        return response

# Specialized generators for different domains
class MathDataGenerator:
    """Specialized generator for mathematical problems and solutions"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = InferenceEngine(model, tokenizer)
    
    def generate_math_problems(self, num_problems: int = 100) -> List[Dict[str, Any]]:
        """Generate mathematical problems with step-by-step solutions"""
        
        problem_types = [
            'linear_equations', 'quadratic_equations', 'geometry', 
            'calculus', 'statistics', 'algebra'
        ]
        
        math_data = []
        
        for i in range(num_problems):
            problem_type = random.choice(problem_types)
            problem = self._generate_problem(problem_type)
            solution = self._generate_solution(problem)
            
            math_data.append({
                'problem': problem,
                'solution': solution,
                'problem_type': problem_type,
                'difficulty': self._assess_difficulty(problem)
            })
        
        return math_data
    
    def _generate_problem(self, problem_type: str) -> str:
        """Generate a math problem of specified type"""
        
        templates = {
            'linear_equations': [
                f"Solve for x: {random.randint(2,9)}x + {random.randint(1,20)} = {random.randint(10,50)}",
                f"Find x when {random.randint(3,8)}x - {random.randint(5,15)} = {random.randint(20,40)}"
            ],
            'geometry': [
                f"Find the area of a circle with radius {random.randint(3,12)} meters",
                f"Calculate the perimeter of a rectangle with length {random.randint(5,20)} and width {random.randint(3,15)}"
            ]
        }
        
        return random.choice(templates.get(problem_type, ['Solve this problem.']))
    
    def _generate_solution(self, problem: str) -> str:
        """Generate step-by-step solution"""
        
        solution_prompt = f"""
        Solve this math problem step by step:
        Problem: {problem}
        
        Solution:
        """
        
        gen_config = GenerationConfig(
            sampling=SamplingConfig(temperature=0.3, top_p=0.9),  # Lower temperature for math
            max_new_tokens=300
        )
        
        input_ids = self.tokenizer.encode(solution_prompt, return_tensors='pt')
        result = self.engine.generate(input_ids, gen_config)
        
        generated_ids = result['sequences'][0][input_ids.size(1):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    def _assess_difficulty(self, problem: str) -> str:
        """Assess problem difficulty"""
        
        # Simple heuristic based on problem complexity
        if any(word in problem.lower() for word in ['calculus', 'derivative', 'integral']):
            return 'hard'
        elif any(word in problem.lower() for word in ['quadratic', 'system', 'matrix']):
            return 'medium'
        else:
            return 'easy'

class CodeDataGenerator:
    """Specialized generator for programming problems and solutions"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = InferenceEngine(model, tokenizer)
    
    def generate_coding_problems(self, num_problems: int = 100) -> List[Dict[str, Any]]:
        """Generate coding problems with solutions"""
        
        languages = ['Python', 'JavaScript', 'Java', 'C++']
        problem_types = ['algorithm', 'data_structure', 'debugging', 'optimization']
        
        coding_data = []
        
        for i in range(num_problems):
            language = random.choice(languages)
            problem_type = random.choice(problem_types)
            
            problem = self._generate_coding_problem(language, problem_type)
            solution = self._generate_coding_solution(problem, language)
            
            coding_data.append({
                'problem': problem,
                'solution': solution,
                'language': language,
                'problem_type': problem_type
            })
        
        return coding_data
    
    def _generate_coding_problem(self, language: str, problem_type: str) -> str:
        """Generate a coding problem"""
        
        templates = {
            'algorithm': f"Implement a function in {language} to sort an array using bubble sort",
            'data_structure': f"Create a {language} class for a binary search tree",
            'debugging': f"Debug this {language} code that should calculate factorial",
            'optimization': f"Optimize this {language} function for better performance"
        }
        
        return templates.get(problem_type, f"Write a {language} program")
    
    def _generate_coding_solution(self, problem: str, language: str) -> str:
        """Generate coding solution"""
        
        solution_prompt = f"""
        {problem}
        
        Please provide a complete, working solution with comments:
        """
        
        gen_config = GenerationConfig(
            sampling=SamplingConfig(temperature=0.2, top_p=0.9),  # Low temperature for code
            max_new_tokens=400
        )
        
        input_ids = self.tokenizer.encode(solution_prompt, return_tensors='pt')
        result = self.engine.generate(input_ids, gen_config)
        
        generated_ids = result['sequences'][0][input_ids.size(1):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# Utility functions
def create_synthetic_data_generator(
    teacher_model: nn.Module,
    tokenizer,
    config: SyntheticDataConfig = None,
    quality_model: Optional[nn.Module] = None
) -> SyntheticDataGenerator:
    """Factory function to create synthetic data generator"""
    
    if config is None:
        config = SyntheticDataConfig()
    
    return SyntheticDataGenerator(teacher_model, tokenizer, config, quality_model)

def save_synthetic_dataset(dataset: List[Dict[str, Any]], filepath: str):
    """Save synthetic dataset to file"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(dataset)} samples to {filepath}")

def load_synthetic_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load synthetic dataset from file"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} samples from {filepath}")
    return dataset

def merge_datasets(datasets: List[List[Dict[str, Any]]], 
                  weights: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """Merge multiple synthetic datasets with optional weighting"""
    
    if weights is None:
        weights = [1.0] * len(datasets)
    
    merged = []
    
    for dataset, weight in zip(datasets, weights):
        # Sample based on weight
        num_samples = int(len(dataset) * weight)
        sampled = random.sample(dataset, min(num_samples, len(dataset)))
        merged.extend(sampled)
    
    # Shuffle the merged dataset
    random.shuffle(merged)
    
    return merged

# Export components
__all__ = [
    'SyntheticDataConfig',
    'SyntheticDataGenerator',
    'ConstitutionalAIGenerator',
    'MathDataGenerator',
    'CodeDataGenerator',
    'create_synthetic_data_generator',
    'save_synthetic_dataset',
    'load_synthetic_dataset',
    'merge_datasets'
]