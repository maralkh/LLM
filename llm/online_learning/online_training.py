# online_training_inference.py
"""
Online Training and Inference Script
Combines reward-guided inference with continuous model improvement
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import threading
import queue
import time
import json
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OnlineConfig:
    """Configuration for online training and inference"""
    # Training settings
    batch_size: int = 8
    learning_rate: float = 1e-5
    training_interval: int = 100  # Train after N inferences
    max_training_queue_size: int = 1000
    
    # Inference settings
    max_inference_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Reward model settings
    reward_weight: float = 0.5
    prm_weight: float = 0.3
    orm_weight: float = 0.7
    
    # Online learning settings
    experience_buffer_size: int = 10000
    min_reward_threshold: float = 0.1
    diversity_penalty: float = 0.1
    
    # System settings
    max_workers: int = 4
    checkpoint_interval: int = 500
    evaluation_interval: int = 200

class ExperienceBuffer:
    """Buffer for storing training experiences"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_experience(self, prompt: str, response: str, reward: float, 
                      prm_scores: List[float] = None, orm_score: float = None,
                      user_feedback: str = None):
        """Add new experience to buffer"""
        experience = {
            'prompt': prompt,
            'response': response,
            'reward': reward,
            'prm_scores': prm_scores or [],
            'orm_score': orm_score,
            'user_feedback': user_feedback,
            'timestamp': time.time()
        }
        
        with self.lock:
            self.buffer.append(experience)
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get a batch of experiences for training"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            
            # Sample with priority to recent high-reward experiences
            experiences = list(self.buffer)
            
            # Sort by reward and recency
            experiences.sort(key=lambda x: x['reward'] + 0.1 * (time.time() - x['timestamp']) / 3600, 
                           reverse=True)
            
            return experiences[:batch_size]
    
    def size(self) -> int:
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        with self.lock:
            self.buffer.clear()

class OnlineTrainingDataset(Dataset):
    """Dataset for online training"""
    
    def __init__(self, experiences: List[Dict], tokenizer, max_length: int = 512):
        self.experiences = experiences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        exp = self.experiences[idx]
        
        # Create full text
        full_text = f"{exp['prompt']} {exp['response']}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'reward': torch.tensor(exp['reward'], dtype=torch.float),
            'orm_score': torch.tensor(exp.get('orm_score', 0.0), dtype=torch.float)
        }

class OnlineTrainer:
    """Online trainer that continuously improves the model"""
    
    def __init__(self, model, reward_model, tokenizer, config: OnlineConfig):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.training_step = 0
        self.device = next(model.parameters()).device
        
        # Training statistics
        self.stats = {
            'total_training_steps': 0,
            'average_reward': 0.0,
            'training_loss': deque(maxlen=100),
            'last_update': time.time()
        }
    
    def train_on_experiences(self, experiences: List[Dict]):
        """Train model on batch of experiences"""
        if not experiences:
            return
        
        # Create dataset and dataloader
        dataset = OnlineTrainingDataset(experiences, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
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
            
            # Compute loss with reward weighting
            base_loss = outputs.loss
            
            # Reward-weighted loss
            reward_weights = torch.sigmoid(batch['reward'])  # Convert to 0-1
            weighted_loss = base_loss * reward_weights.mean()
            
            # Add diversity penalty
            if self.config.diversity_penalty > 0:
                # Simple diversity penalty based on output variance
                logits = outputs.logits
                diversity_penalty = -torch.var(logits, dim=-1).mean()
                weighted_loss += self.config.diversity_penalty * diversity_penalty
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
            num_batches += 1
            self.training_step += 1
        
        # Update statistics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.stats['training_loss'].append(avg_loss)
        self.stats['total_training_steps'] += num_batches
        self.stats['last_update'] = time.time()
        
        logger.info(f"Training step {self.training_step}: Loss = {avg_loss:.4f}")
        
        return avg_loss

class OnlineInferenceEngine:
    """Online inference engine with reward guidance"""
    
    def __init__(self, model, prm_model, orm_model, tokenizer, config: OnlineConfig):
        self.model = model
        self.prm_model = prm_model
        self.orm_model = orm_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # Inference statistics
        self.stats = {
            'total_inferences': 0,
            'average_reward': 0.0,
            'average_length': 0.0,
            'inference_times': deque(maxlen=100)
        }
    
    def generate_with_rewards(self, prompt: str, max_length: int = None) -> Dict[str, Any]:
        """Generate response with reward guidance"""
        start_time = time.time()
        
        max_length = max_length or self.config.max_inference_length
        
        self.model.eval()
        with torch.no_grad():
            # Tokenize prompt
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate multiple candidates
            candidates = []
            for _ in range(3):  # Generate 3 candidates
                output_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_length,
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
            
            # Evaluate candidates with reward models
            best_response = None
            best_score = float('-inf')
            best_prm_scores = []
            best_orm_score = 0.0
            
            for response in candidates:
                full_text = prompt + " " + response
                text_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
                
                # Get PRM scores (step-by-step rewards)
                prm_outputs = self.prm_model.forward(text_ids)
                prm_scores = prm_outputs['step_rewards'][0].cpu().tolist()
                avg_prm_score = np.mean(prm_scores) if prm_scores else 0.0
                
                # Get ORM score (outcome reward)
                attention_mask = torch.ones_like(text_ids)
                orm_outputs = self.orm_model.forward(text_ids, attention_mask)
                orm_score = orm_outputs['overall_reward'][0].item()
                
                # Combined score
                combined_score = (self.config.prm_weight * avg_prm_score + 
                                self.config.orm_weight * orm_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_response = response
                    best_prm_scores = prm_scores
                    best_orm_score = orm_score
        
        # Update statistics
        inference_time = time.time() - start_time
        self.stats['total_inferences'] += 1
        self.stats['inference_times'].append(inference_time)
        self.stats['average_reward'] = (
            (self.stats['average_reward'] * (self.stats['total_inferences'] - 1) + best_score) /
            self.stats['total_inferences']
        )
        self.stats['average_length'] = (
            (self.stats['average_length'] * (self.stats['total_inferences'] - 1) + len(best_response)) /
            self.stats['total_inferences']
        )
        
        return {
            'prompt': prompt,
            'response': best_response,
            'prm_scores': best_prm_scores,
            'orm_score': best_orm_score,
            'combined_score': best_score,
            'inference_time': inference_time,
            'candidates_evaluated': len(candidates)
        }

class OnlineSystem:
    """Main online training and inference system"""
    
    def __init__(self, base_model, prm_model, orm_model, tokenizer, config: OnlineConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        # Models
        self.model = base_model
        self.prm_model = prm_model
        self.orm_model = orm_model
        
        # Components
        self.experience_buffer = ExperienceBuffer(config.experience_buffer_size)
        self.trainer = OnlineTrainer(self.model, self.orm_model, tokenizer, config)
        self.inference_engine = OnlineInferenceEngine(
            self.model, self.prm_model, self.orm_model, tokenizer, config
        )
        
        # Queues for async communication
        self.inference_queue = queue.Queue()
        self.training_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Control flags
        self.running = False
        self.training_thread = None
        self.inference_count = 0
        
        # Statistics
        self.system_stats = {
            'start_time': time.time(),
            'total_requests': 0,
            'successful_inferences': 0,
            'training_sessions': 0,
            'last_checkpoint': time.time()
        }
    
    def start(self):
        """Start the online system"""
        self.running = True
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("🚀 Online Training and Inference System Started")
    
    def stop(self):
        """Stop the online system"""
        self.running = False
        if self.training_thread:
            self.training_thread.join()
        logger.info("🛑 Online System Stopped")
    
    def inference_request(self, prompt: str, user_feedback: str = None) -> Dict[str, Any]:
        """Handle inference request"""
        self.system_stats['total_requests'] += 1
        
        try:
            # Generate response
            result = self.inference_engine.generate_with_rewards(prompt)
            
            # Add to experience buffer
            self.experience_buffer.add_experience(
                prompt=prompt,
                response=result['response'],
                reward=result['combined_score'],
                prm_scores=result['prm_scores'],
                orm_score=result['orm_score'],
                user_feedback=user_feedback
            )
            
            self.inference_count += 1
            self.system_stats['successful_inferences'] += 1
            
            # Trigger training if needed
            if self.inference_count >= self.config.training_interval:
                self._trigger_training()
                self.inference_count = 0
            
            # Checkpoint if needed
            if (time.time() - self.system_stats['last_checkpoint']) > self.config.checkpoint_interval:
                self._save_checkpoint()
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'prompt': prompt,
                'response': "Sorry, I encountered an error generating a response.",
                'error': str(e),
                'combined_score': 0.0
            }
    
    def _training_loop(self):
        """Background training loop"""
        while self.running:
            try:
                # Wait for training trigger or timeout
                time.sleep(1.0)
                
                # Check if training is needed
                buffer_size = self.experience_buffer.size()
                if buffer_size >= self.config.batch_size * 2:
                    # Get experiences for training
                    experiences = self.experience_buffer.get_batch(
                        self.config.batch_size * 4
                    )
                    
                    # Filter high-quality experiences
                    good_experiences = [
                        exp for exp in experiences 
                        if exp['reward'] > self.config.min_reward_threshold
                    ]
                    
                    if good_experiences:
                        logger.info(f"Training on {len(good_experiences)} experiences")
                        
                        # Train model
                        loss = self.trainer.train_on_experiences(good_experiences)
                        self.system_stats['training_sessions'] += 1
                        
                        logger.info(f"Training completed. Loss: {loss:.4f}")
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _trigger_training(self):
        """Trigger immediate training"""
        logger.info("🔄 Triggering training session")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = Path("./online_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save model state
        torch.save(self.model.state_dict(), 
                  checkpoint_dir / f"model_{timestamp}.pt")
        
        # Save statistics
        stats = {
            'system_stats': self.system_stats,
            'trainer_stats': self.trainer.stats,
            'inference_stats': self.inference_engine.stats,
            'buffer_size': self.experience_buffer.size()
        }
        
        with open(checkpoint_dir / f"stats_{timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.system_stats['last_checkpoint'] = time.time()
        logger.info(f"💾 Checkpoint saved at {timestamp}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'system_stats': self.system_stats,
            'trainer_stats': self.trainer.stats,
            'inference_stats': self.inference_engine.stats,
            'buffer_size': self.experience_buffer.size(),
            'running': self.running,
            'uptime': time.time() - self.system_stats['start_time']
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update system configuration"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")

# Example usage and testing
class DummyTokenizer:
    """Dummy tokenizer for testing"""
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
        return f"Generated response based on {len(tokens)} tokens"
    
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

def create_dummy_models():
    """Create dummy models for testing"""
    # Create a simple transformer model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32000, 512)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(512, 8, 2048), 6
            )
            self.output_head = nn.Linear(512, 32000)
        
        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embedding(input_ids)
            x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
            logits = self.output_head(x)
            
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, 32000), labels.view(-1))
            
            return type('Outputs', (), {'logits': logits, 'loss': loss})()
        
        def generate(self, input_ids, max_new_tokens=50, **kwargs):
            # Simple generation - just repeat input with noise
            batch_size, seq_len = input_ids.shape
            new_tokens = torch.randint(0, 32000, (batch_size, max_new_tokens))
            return torch.cat([input_ids, new_tokens], dim=1)
    
    # Create dummy reward models
    class DummyPRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.score_head = nn.Linear(512, 1)
        
        def forward(self, input_ids):
            # Random step rewards
            seq_len = input_ids.size(1)
            step_rewards = torch.rand(input_ids.size(0), seq_len)
            step_confidences = torch.rand(input_ids.size(0), seq_len)
            return {
                'step_rewards': step_rewards,
                'step_confidences': step_confidences
            }
    
    class DummyORM(nn.Module):
        def __init__(self):
            super().__init__()
            self.reward_head = nn.Linear(512, 1)
        
        def forward(self, input_ids, attention_mask):
            # Random overall reward
            batch_size = input_ids.size(0)
            overall_reward = torch.rand(batch_size, 1) * 2 - 1  # -1 to 1
            return {
                'overall_reward': overall_reward,
                'correctness': torch.sigmoid(overall_reward),
                'helpfulness': overall_reward
            }
    
    return DummyModel(), DummyPRM(), DummyORM()

def test_online_system():
    """Test the online system"""
    print("🧪 Testing Online Training and Inference System")
    print("=" * 60)
    
    # Create models
    base_model, prm_model, orm_model = create_dummy_models()
    tokenizer = DummyTokenizer()
    
    # Create config
    config = OnlineConfig(
        batch_size=4,
        training_interval=5,  # Train after 5 inferences
        max_inference_length=100,
        experience_buffer_size=50,
        checkpoint_interval=30
    )
    
    # Create online system
    system = OnlineSystem(base_model, prm_model, orm_model, tokenizer, config)
    
    # Start system
    system.start()
    
    # Test prompts
    test_prompts = [
        "Explain machine learning",
        "Write a story about AI",
        "Solve this math problem: 2x + 5 = 11",
        "Describe the water cycle",
        "What is quantum computing?",
        "How do neural networks work?",
        "Explain climate change",
        "Write a poem about space"
    ]
    
    print("🔄 Running inference tests...")
    
    # Run multiple inference requests
    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\nRequest {i+1}: {prompt}")
        
        # Simulate user feedback
        feedback = "good" if i % 3 == 0 else None
        
        result = system.inference_request(prompt, user_feedback=feedback)
        results.append(result)
        
        print(f"Response: {result['response'][:100]}...")
        print(f"Score: {result.get('combined_score', 0):.3f}")
        print(f"Time: {result.get('inference_time', 0):.2f}s")
        
        # Brief pause between requests
        time.sleep(1)
    
    # Wait for training to complete
    print("\n⏳ Waiting for training to complete...")
    time.sleep(5)
    
    # Get system status
    status = system.get_system_status()
    print("\n📊 System Status:")
    print("-" * 30)
    print(f"Total Requests: {status['system_stats']['total_requests']}")
    print(f"Successful Inferences: {status['system_stats']['successful_inferences']}")
    print(f"Training Sessions: {status['system_stats']['training_sessions']}")
    print(f"Buffer Size: {status['buffer_size']}")
    print(f"Uptime: {status['uptime']:.1f}s")
    print(f"Average Reward: {status['inference_stats']['average_reward']:.3f}")
    
    # Test configuration update
    print("\n🔧 Testing configuration update...")
    system.update_config({
        'temperature': 0.8,
        'training_interval': 3
    })
    
    # Run a few more requests to test updated config
    for prompt in test_prompts[:3]:
        result = system.inference_request(prompt)
        print(f"Updated config result: {result.get('combined_score', 0):.3f}")
    
    # Stop system
    system.stop()
    
    print("\n✅ Online system test completed!")
    return system, results

def run_production_server():
    """Run a production-like server"""
    print("🌐 Starting Production Online System")
    print("=" * 50)
    
    # Create models (in production, load from checkpoints)
    base_model, prm_model, orm_model = create_dummy_models()
    tokenizer = DummyTokenizer()
    
    # Production config
    config = OnlineConfig(
        batch_size=16,
        training_interval=50,
        max_inference_length=256,
        experience_buffer_size=5000,
        checkpoint_interval=300,  # 5 minutes
        max_workers=8
    )
    
    # Create and start system
    system = OnlineSystem(base_model, prm_model, orm_model, tokenizer, config)
    system.start()
    
    print("🚀 Production system started. Press Ctrl+C to stop.")
    
    try:
        # Simulate production load
        import random
        
        prompts = [
            "Explain artificial intelligence",
            "How do I learn programming?",
            "What is the meaning of life?",
            "Describe renewable energy",
            "Write a short story",
            "Solve math problems",
            "Explain quantum physics",
            "Help with coding"
        ]
        
        request_count = 0
        while True:
            # Simulate incoming requests
            prompt = random.choice(prompts)
            feedback = random.choice([None, None, None, "good", "bad"])  # 1/3 have feedback
            
            result = system.inference_request(prompt, user_feedback=feedback)
            request_count += 1
            
            if request_count % 10 == 0:
                status = system.get_system_status()
                print(f"Processed {request_count} requests. "
                      f"Buffer: {status['buffer_size']}, "
                      f"Training sessions: {status['system_stats']['training_sessions']}")
            
            # Simulate realistic request intervals
            time.sleep(random.uniform(0.5, 2.0))
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down production system...")
        system.stop()
        
        # Final statistics
        final_status = system.get_system_status()
        print("\n📈 Final Statistics:")
        print(f"Total Requests: {final_status['system_stats']['total_requests']}")
        print(f"Training Sessions: {final_status['system_stats']['training_sessions']}")
        print(f"Uptime: {final_status['uptime']:.1f}s")
        
        print("✅ Production system stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "production":
        run_production_server()
    else:
        test_online_system()