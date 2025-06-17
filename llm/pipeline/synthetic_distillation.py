# training_infra/pipeline/synthetic_distillation.py
"""
Complete pipeline combining synthetic data generation and knowledge distillation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union
import json
import copy
from pathlib import Path
import numpy as np
import time

from ..data.synthetic import (
    SyntheticDataGenerator, SyntheticDataConfig,
    MathDataGenerator, CodeDataGenerator,
    create_synthetic_data_generator
)
from ..distillation.distillation import (
    DistillationTrainer, DistillationConfig,
    ProgressiveDistillationTrainer, SelfDistillationTrainer,
    create_distillation_trainer
)
from ..models.llama import create_llama_7b
from ..config import TrainingConfig

class SyntheticDistillationPipeline:
    """Complete pipeline for synthetic data generation and model distillation"""
    
    def __init__(self, 
                 teacher_model: nn.Module,
                 tokenizer,
                 target_compression: float = 0.5):
        
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.target_compression = target_compression
        
        # Pipeline components
        self.synthetic_generator = None
        self.student_model = None
        self.distillation_trainer = None
        
        # Generated data
        self.synthetic_datasets = {}
        
        # Statistics
        self.pipeline_stats = {
            'data_generation_time': 0.0,
            'distillation_time': 0.0,
            'total_synthetic_samples': 0,
            'final_model_size': 0,
            'compression_achieved': 0.0
        }
    
    def run_complete_pipeline(self,
                            synthetic_config: SyntheticDataConfig = None,
                            distillation_config: DistillationConfig = None,
                            training_config: TrainingConfig = None) -> Dict[str, Any]:
        """
        Run the complete synthetic data generation + distillation pipeline
        
        Args:
            synthetic_config: Configuration for synthetic data generation
            distillation_config: Configuration for knowledge distillation
            training_config: Configuration for training
            
        Returns:
            Dictionary with pipeline results
        """
        
        print("🌟 Starting Complete Synthetic Data + Distillation Pipeline")
        print("=" * 70)
        
        # Step 1: Generate synthetic data
        print("\n📊 Step 1: Synthetic Data Generation")
        synthetic_data = self.generate_synthetic_training_data(synthetic_config)
        
        # Step 2: Create student model
        print("\n🏗️ Step 2: Student Model Creation")
        self.create_student_model()
        
        # Step 3: Knowledge distillation
        print("\n🎓 Step 3: Knowledge Distillation")
        distilled_model = self.distill_knowledge(
            synthetic_data, 
            distillation_config, 
            training_config
        )
        
        # Step 4: Evaluation
        print("\n📈 Step 4: Model Evaluation")
        evaluation_results = self.evaluate_pipeline()
        
        # Step 5: Save results
        print("\n💾 Step 5: Save Results")
        self.save_pipeline_results()
        
        results = {
            'synthetic_data': synthetic_data,
            'student_model': distilled_model,
            'evaluation_results': evaluation_results,
            'pipeline_stats': self.pipeline_stats
        }
        
        print("\n🎉 Pipeline completed successfully!")
        self._print_pipeline_summary()
        
        return results
    
    def generate_synthetic_training_data(self, config: SyntheticDataConfig = None) -> List[Dict[str, Any]]:
        """Generate comprehensive synthetic training data"""
        
        start_time = time.time()
        
        if config is None:
            config = SyntheticDataConfig(
                num_samples=2000,
                temperature=0.8,
                use_quality_filter=True,
                use_diversity_filter=True,
                include_instructions=True,
                include_conversations=True,
                include_reasoning=True,
                include_creative=True
            )
        
        # Create synthetic data generator
        self.synthetic_generator = create_synthetic_data_generator(
            teacher_model=self.teacher_model,
            tokenizer=self.tokenizer,
            config=config
        )
        
        print(f"📋 Generating {config.num_samples} synthetic samples...")
        
        # Generate general synthetic data
        general_data = self.synthetic_generator.generate_synthetic_dataset()
        self.synthetic_datasets['general'] = general_data
        
        # Generate specialized data
        specialized_data = self._generate_specialized_data(config)
        
        # Combine all synthetic data
        all_synthetic_data = general_data + specialized_data
        
        generation_time = time.time() - start_time
        self.pipeline_stats['data_generation_time'] = generation_time
        self.pipeline_stats['total_synthetic_samples'] = len(all_synthetic_data)
        
        print(f"✅ Generated {len(all_synthetic_data)} synthetic samples in {generation_time:.2f}s")
        
        return all_synthetic_data
    
    def _generate_specialized_data(self, config: SyntheticDataConfig) -> List[Dict[str, Any]]:
        """Generate specialized synthetic data for specific domains"""
        
        specialized_data = []
        
        # Math data generation
        if config.include_reasoning:
            print("🧮 Generating mathematical reasoning data...")
            math_generator = MathDataGenerator(self.teacher_model, self.tokenizer)
            math_data = math_generator.generate_math_problems(num_problems=200)
            
            # Convert to standard format
            for item in math_data:
                specialized_data.append({
                    'prompt': item['problem'],
                    'response': item['solution'],
                    'data_type': 'mathematical_reasoning',
                    'difficulty': item['difficulty'],
                    'problem_type': item['problem_type']
                })
            
            self.synthetic_datasets['math'] = math_data
        
        # Code data generation
        if config.include_instructions:
            print("💻 Generating code instruction data...")
            code_generator = CodeDataGenerator(self.teacher_model, self.tokenizer)
            code_data = code_generator.generate_coding_problems(num_problems=150)
            
            # Convert to standard format
            for item in code_data:
                specialized_data.append({
                    'prompt': item['problem'],
                    'response': item['solution'],
                    'data_type': 'code_instruction',
                    'language': item['language'],
                    'problem_type': item['problem_type']
                })
            
            self.synthetic_datasets['code'] = code_data
        
        # Domain-specific conversation data
        conversation_data = self._generate_domain_conversations()
        specialized_data.extend(conversation_data)
        self.synthetic_datasets['domain_conversations'] = conversation_data
        
        print(f"📚 Generated {len(specialized_data)} specialized samples")
        
        return specialized_data
    
    def _generate_domain_conversations(self) -> List[Dict[str, Any]]:
        """Generate domain-specific conversation data"""
        
        domains = {
            'science': [
                "Explain the process of photosynthesis",
                "What causes climate change?",
                "How do vaccines work?",
                "Describe the structure of an atom"
            ],
            'technology': [
                "What is artificial intelligence?",
                "How does blockchain technology work?",
                "Explain quantum computing",
                "What are the benefits of cloud computing?"
            ],
            'health': [
                "What are the benefits of regular exercise?",
                "How can I maintain a healthy diet?",
                "What is the importance of sleep?",
                "How do I manage stress effectively?"
            ]
        }
        
        conversation_data = []
        
        for domain, prompts in domains.items():
            for prompt in prompts:
                try:
                    # Generate response using teacher model
                    from ..inference.engine import GenerationConfig
                    from ..inference.sampling import SamplingConfig
                    
                    gen_config = GenerationConfig(
                        sampling=SamplingConfig(temperature=0.7, top_p=0.9),
                        max_new_tokens=200
                    )
                    
                    input_ids = self.tokenizer.encode(f"Human: {prompt}\nAssistant:", return_tensors='pt')
                    
                    # Use teacher model to generate
                    with torch.no_grad():
                        outputs = self.teacher_model.generate(
                            input_ids,
                            max_new_tokens=200,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][input_ids.size(1):], 
                        skip_special_tokens=True
                    ).strip()
                    
                    conversation_data.append({
                        'prompt': f"Human: {prompt}\nAssistant:",
                        'response': response,
                        'data_type': f'{domain}_conversation',
                        'domain': domain
                    })
                    
                except Exception as e:
                    print(f"⚠️ Failed to generate conversation for '{prompt}': {e}")
                    continue
        
        return conversation_data
    
    def create_student_model(self):
        """Create smaller student model based on teacher"""
        
        print(f"🏗️ Creating student model with {self.target_compression}x compression...")
        
        # Get teacher model parameters
        teacher_config = getattr(self.teacher_model, 'config', None)
        
        if teacher_config:
            # Create smaller config for student
            student_config = copy.deepcopy(teacher_config)
            
            # Reduce model size based on compression ratio
            student_config.hidden_size = int(teacher_config.hidden_size * self.target_compression)
            student_config.intermediate_size = int(teacher_config.intermediate_size * self.target_compression)
            student_config.num_hidden_layers = int(teacher_config.num_hidden_layers * self.target_compression)
            student_config.num_attention_heads = max(1, int(teacher_config.num_attention_heads * self.target_compression))
            
            # Ensure dimensions are compatible
            student_config.hidden_size = (student_config.hidden_size // student_config.num_attention_heads) * student_config.num_attention_heads
            
            print(f"Student model: {student_config.num_hidden_layers} layers, {student_config.hidden_size} hidden size")
            
            # Create student model with smaller config
            from ..models.llama import LlamaForCausalLM
            self.student_model = LlamaForCausalLM(student_config)
        else:
            # Fallback: create a smaller version of teacher
            self.student_model = copy.deepcopy(self.teacher_model)
            # In practice, you'd implement actual model compression here
        
        # Calculate actual compression achieved
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        actual_compression = student_params / teacher_params
        self.pipeline_stats['compression_achieved'] = actual_compression
        self.pipeline_stats['final_model_size'] = student_params
        
        print(f"✅ Student model created: {student_params:,} parameters ({actual_compression:.2f}x of teacher)")
    
    def distill_knowledge(self,
                         synthetic_data: List[Dict[str, Any]],
                         distillation_config: DistillationConfig = None,
                         training_config: TrainingConfig = None) -> nn.Module:
        """Perform knowledge distillation using synthetic data"""
        
        start_time = time.time()
        
        if distillation_config is None:
            distillation_config = DistillationConfig(
                temperature=4.0,
                alpha=0.7,
                beta=0.3,
                use_feature_distillation=True,
                use_progressive=True,
                progressive_epochs=[0, 3, 6],
                progressive_temperatures=[8.0, 4.0, 2.0]
            )
        
        if training_config is None:
            training_config = TrainingConfig(
                epochs=10,
                batch_size=8,
                optimizer=TrainingConfig.OptimizerConfig(lr=5e-5),
                logging=TrainingConfig.LoggingConfig(log_every=50)
            )
        
        # Create dataset from synthetic data
        train_dataset = SyntheticDataset(synthetic_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
        
        # Create distillation trainer
        if distillation_config.use_progressive:
            self.distillation_trainer = ProgressiveDistillationTrainer(
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                config=training_config,
                distillation_config=distillation_config,
                train_dataloader=train_loader
            )
        else:
            self.distillation_trainer = create_distillation_trainer(
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                train_dataloader=train_loader,
                config=training_config,
                distillation_config=distillation_config
            )
        
        print(f"🎓 Starting knowledge distillation for {training_config.epochs} epochs...")
        
        # Train student model
        self.distillation_trainer.fit()
        
        distillation_time = time.time() - start_time
        self.pipeline_stats['distillation_time'] = distillation_time
        
        print(f"✅ Knowledge distillation completed in {distillation_time:.2f}s")
        
        return self.student_model
    
    def evaluate_pipeline(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of the pipeline"""
        
        print("📊 Evaluating pipeline effectiveness...")
        
        # Create test prompts
        test_prompts = [
            "Explain the concept of machine learning",
            "Solve: 3x + 7 = 22",
            "Write a Python function to reverse a string",
            "What are the causes of global warming?",
            "How do you maintain a healthy lifestyle?"
        ]
        
        evaluation_results = {
            'test_prompts': test_prompts,
            'teacher_responses': [],
            'student_responses': [],
            'response_similarities': [],
            'generation_times': {'teacher': [], 'student': []},
            'model_sizes': {
                'teacher_params': sum(p.numel() for p in self.teacher_model.parameters()),
                'student_params': sum(p.numel() for p in self.student_model.parameters())
            }
        }
        
        # Generate responses from both models
        for prompt in test_prompts:
            # Teacher response
            teacher_response, teacher_time = self._generate_timed_response(self.teacher_model, prompt)
            evaluation_results['teacher_responses'].append(teacher_response)
            evaluation_results['generation_times']['teacher'].append(teacher_time)
            
            # Student response
            student_response, student_time = self._generate_timed_response(self.student_model, prompt)
            evaluation_results['student_responses'].append(student_response)
            evaluation_results['generation_times']['student'].append(student_time)
            
            # Calculate similarity
            similarity = self._calculate_response_similarity(teacher_response, student_response)
            evaluation_results['response_similarities'].append(similarity)
        
        # Calculate summary metrics
        evaluation_results['summary'] = {
            'avg_similarity': np.mean(evaluation_results['response_similarities']),
            'avg_teacher_time': np.mean(evaluation_results['generation_times']['teacher']),
            'avg_student_time': np.mean(evaluation_results['generation_times']['student']),
            'speedup_factor': np.mean(evaluation_results['generation_times']['teacher']) / np.mean(evaluation_results['generation_times']['student']),
            'compression_ratio': evaluation_results['model_sizes']['student_params'] / evaluation_results['model_sizes']['teacher_params']
        }
        
        print(f"📈 Evaluation Results:")
        print(f"  Average Similarity: {evaluation_results['summary']['avg_similarity']:.3f}")
        print(f"  Speedup Factor: {evaluation_results['summary']['speedup_factor']:.2f}x")
        print(f"  Compression Ratio: {evaluation_results['summary']['compression_ratio']:.3f}")
        
        return evaluation_results
    
    def _generate_timed_response(self, model: nn.Module, prompt: str) -> Tuple[str, float]:
        """Generate response and measure time"""
        
        start_time = time.time()
        
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][input_ids.size(1):], 
                skip_special_tokens=True
            ).strip()
            
        except Exception as e:
            response = f"[Generation failed: {e}]"
        
        generation_time = time.time() - start_time
        
        return response, generation_time
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses"""
        
        # Simple token-based similarity
        tokens1 = set(response1.lower().split())
        tokens2 = set(response2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def save_pipeline_results(self, save_dir: str = "./synthetic_distillation_results"):
        """Save all pipeline results"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save synthetic datasets
        for dataset_name, dataset in self.synthetic_datasets.items():
            dataset_file = save_path / f"synthetic_{dataset_name}_data.json"
            with open(dataset_file, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
        
        # Save student model
        if self.student_model:
            model_path = save_path / "distilled_student_model.pt"
            torch.save(self.student_model.state_dict(), model_path)
        
        # Save pipeline statistics
        stats_file = save_path / "pipeline_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.pipeline_stats, f, indent=2, default=str)
        
        # Save configurations
        config_file = save_path / "pipeline_config.json"
        pipeline_config = {
            'target_compression': self.target_compression,
            'teacher_model_info': {
                'parameters': sum(p.numel() for p in self.teacher_model.parameters()),
                'architecture': self.teacher_model.__class__.__name__
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(pipeline_config, f, indent=2, default=str)
        
        print(f"💾 Pipeline results saved to: {save_dir}")
    
    def _print_pipeline_summary(self):
        """Print comprehensive pipeline summary"""
        
        print("\n📋 Pipeline Summary")
        print("=" * 50)
        print(f"🕒 Data Generation Time: {self.pipeline_stats['data_generation_time']:.2f}s")
        print(f"🕒 Distillation Time: {self.pipeline_stats['distillation_time']:.2f}s")
        print(f"📊 Total Synthetic Samples: {self.pipeline_stats['total_synthetic_samples']:,}")
        print(f"🗜️ Compression Achieved: {self.pipeline_stats['compression_achieved']:.3f}")
        print(f"📦 Final Model Size: {self.pipeline_stats['final_model_size']:,} parameters")
        
        print(f"\n📈 Performance Gains:")
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = self.pipeline_stats['final_model_size']
        size_reduction = (1 - student_params / teacher_params) * 100
        
        print(f"  Model Size Reduction: {size_reduction:.1f}%")
        print(f"  Memory Savings: ~{size_reduction:.1f}%")
        print(f"  Potential Speedup: ~{1/self.pipeline_stats['compression_achieved']:.1f}x")

class SyntheticDataset(Dataset):
    """Dataset wrapper for synthetic data"""
    
    def __init__(self, synthetic_data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = synthetic_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine prompt and response
        full_text = f"{item['prompt']} {item['response']}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For language modeling
        }

# Advanced pipeline variants
class MultiTeacherDistillationPipeline(SyntheticDistillationPipeline):
    """Pipeline with multiple teacher models for ensemble distillation"""
    
    def __init__(self, teacher_models: List[nn.Module], tokenizer, target_compression: float = 0.5):
        # Use first teacher as primary
        super().__init__(teacher_models[0], tokenizer, target_compression)
        self.teacher_models = teacher_models
        self.ensemble_size = len(teacher_models)
    
    def generate_synthetic_training_data(self, config: SyntheticDataConfig = None) -> List[Dict[str, Any]]:
        """Generate synthetic data using ensemble of teachers"""
        
        all_synthetic_data = []
        
        # Generate data from each teacher
        for i, teacher in enumerate(self.teacher_models):
            print(f"📚 Generating data from teacher {i+1}/{self.ensemble_size}")
            
            # Temporarily set current teacher
            original_teacher = self.teacher_model
            self.teacher_model = teacher
            
            # Generate data
            teacher_data = super().generate_synthetic_training_data(config)
            
            # Tag data with teacher source
            for item in teacher_data:
                item['teacher_source'] = i
            
            all_synthetic_data.extend(teacher_data)
            
            # Restore original teacher
            self.teacher_model = original_teacher
        
        return all_synthetic_data

class DomainAdaptiveDistillationPipeline(SyntheticDistillationPipeline):
    """Pipeline with domain-specific distillation"""
    
    def __init__(self, *args, domains: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.domains = domains or ['general', 'math', 'code', 'science']
        self.domain_weights = {domain: 1.0 for domain in self.domains}
    
    def distill_knowledge(self, synthetic_data: List[Dict[str, Any]], *args, **kwargs) -> nn.Module:
        """Domain-adaptive knowledge distillation"""
        
        # Group data by domain
        domain_data = {}
        for item in synthetic_data:
            domain = item.get('data_type', 'general')
            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append(item)
        
        # Progressive distillation by domain
        for domain, data in domain_data.items():
            if domain in self.domains:
                print(f"🎯 Distilling knowledge for domain: {domain}")
                
                # Create domain-specific dataset
                domain_dataset = SyntheticDataset(data, self.tokenizer)
                domain_loader = DataLoader(domain_dataset, batch_size=8, shuffle=True)
                
                # Domain-specific distillation (simplified)
                # In practice, you'd adjust distillation parameters per domain
                
        return super().distill_knowledge(synthetic_data, *args, **kwargs)

# Complete example function
def run_synthetic_distillation_example():
    """Run complete synthetic data + distillation example"""
    
    print("🌟 Running Complete Synthetic Data + Distillation Example")
    print("=" * 70)
    
    # Create dummy teacher model
    teacher_model = create_llama_7b()
    
    # Dummy tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            self.pad_token_id = 0
        
        def encode(self, text, return_tensors=None):
            tokens = [hash(word) % 32000 for word in text.split()[:100]]
            if return_tensors == 'pt':
                return torch.tensor(tokens).unsqueeze(0)
            return tokens
        
        def decode(self, tokens, skip_special_tokens=True):
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            return f"[Generated response from {len(tokens)} tokens]"
        
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
    
    tokenizer = DummyTokenizer()
    
    # Create pipeline
    pipeline = SyntheticDistillationPipeline(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        target_compression=0.3  # 30% of original size
    )
    
    # Configure synthetic data generation
    synthetic_config = SyntheticDataConfig(
        num_samples=500,  # Small for demo
        temperature=0.8,
        use_quality_filter=True,
        include_instructions=True,
        include_conversations=True,
        include_reasoning=True
    )
    
    # Configure distillation
    distillation_config = DistillationConfig(
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
        use_feature_distillation=True,
        use_progressive=True
    )
    
    # Configure training
    training_config = TrainingConfig(
        epochs=3,  # Short for demo
        batch_size=4,
        optimizer=TrainingConfig.OptimizerConfig(lr=5e-5)
    )
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            synthetic_config=synthetic_config,
            distillation_config=distillation_config,
            training_config=training_config
        )
        
        print("\n🎉 Example completed successfully!")
        
        # Print results summary
        print(f"\n📊 Results Summary:")
        print(f"  Synthetic samples generated: {len(results['synthetic_data'])}")
        print(f"  Student model parameters: {results['pipeline_stats']['final_model_size']:,}")
        print(f"  Compression achieved: {results['pipeline_stats']['compression_achieved']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Export components
__all__ = [
    'SyntheticDistillationPipeline',
    'MultiTeacherDistillationPipeline', 
    'DomainAdaptiveDistillationPipeline',
    'SyntheticDataset',
    'run_synthetic_distillation_example'
]