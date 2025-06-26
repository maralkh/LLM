# file_based_training_inference.py
"""
File-Based Online Training and Inference System
Reads training data from files and automatically processes them
"""

import torch
import torch.nn as nn
import json
import csv
import threading
import time
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
import pandas as pd
import random
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileBasedConfig:
    """Configuration for file-based training system"""
    # File paths
    training_data_dir: str = "./training_data"
    inference_requests_file: str = "./inference_requests.jsonl"
    results_output_dir: str = "./results"
    checkpoints_dir: str = "./checkpoints"
    
    # Data processing
    batch_size: int = 16
    max_sequence_length: int = 512
    data_refresh_interval: int = 30  # seconds
    
    # Training settings
    learning_rate: float = 1e-5
    training_epochs_per_batch: int = 1
    min_data_size_for_training: int = 10
    
    # Inference settings
    max_inference_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_candidates: int = 3
    
    # Reward model settings
    prm_weight: float = 0.3
    orm_weight: float = 0.7
    quality_threshold: float = 0.1
    
    # System settings
    auto_training: bool = True
    save_intermediate_results: bool = True
    max_cache_size: int = 10000

class DataFileProcessor:
    """Processes different types of data files"""
    
    def __init__(self, config: FileBasedConfig):
        self.config = config
        self.data_cache = {}
        self.last_processed_times = {}
    
    def load_training_data(self, data_dir: str) -> List[Dict]:
        """Load training data from various file formats"""
        data_path = Path(data_dir)
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Created training data directory: {data_dir}")
            return []
        
        all_data = []
        
        # Process different file types
        for file_path in data_path.glob("**/*"):
            if file_path.is_file():
                try:
                    data = self._process_file(file_path)
                    if data:
                        all_data.extend(data)
                        logger.info(f"Loaded {len(data)} examples from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Total loaded data: {len(all_data)} examples")
        return all_data
    
    def _process_file(self, file_path: Path) -> List[Dict]:
        """Process individual file based on extension"""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.json':
            return self._process_json_file(file_path)
        elif file_ext == '.jsonl':
            return self._process_jsonl_file(file_path)
        elif file_ext == '.csv':
            return self._process_csv_file(file_path)
        elif file_ext == '.txt':
            return self._process_txt_file(file_path)
        elif file_ext == '.pkl':
            return self._process_pickle_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []
    
    def _process_json_file(self, file_path: Path) -> List[Dict]:
        """Process JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return self._normalize_data_format(data)
        elif isinstance(data, dict):
            return self._normalize_data_format([data])
        else:
            logger.error(f"Invalid JSON format in {file_path}")
            return []
    
    def _process_jsonl_file(self, file_path: Path) -> List[Dict]:
        """Process JSONL file (one JSON per line)"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in {file_path} line {line_num}: {e}")
        
        return self._normalize_data_format(data)
    
    def _process_csv_file(self, file_path: Path) -> List[Dict]:
        """Process CSV file"""
        try:
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
            return self._normalize_data_format(data)
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {e}")
            return []
    
    def _process_txt_file(self, file_path: Path) -> List[Dict]:
        """Process text file (assume each line is a prompt)"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    data.append({
                        'prompt': line,
                        'source_file': str(file_path),
                        'line_number': line_num
                    })
        
        return self._normalize_data_format(data)
    
    def _process_pickle_file(self, file_path: Path) -> List[Dict]:
        """Process pickle file"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list):
                return self._normalize_data_format(data)
            elif isinstance(data, dict):
                return self._normalize_data_format([data])
            else:
                logger.error(f"Invalid pickle format in {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error reading pickle {file_path}: {e}")
            return []
    
    def _normalize_data_format(self, data: List[Dict]) -> List[Dict]:
        """Normalize data to standard format"""
        normalized = []
        
        for item in data:
            normalized_item = {}
            
            # Extract prompt (various possible field names)
            prompt_fields = ['prompt', 'question', 'input', 'text', 'query']
            for field in prompt_fields:
                if field in item:
                    normalized_item['prompt'] = str(item[field])
                    break
            
            # Extract response/answer (if available)
            response_fields = ['response', 'answer', 'output', 'completion', 'target']
            for field in response_fields:
                if field in item:
                    normalized_item['response'] = str(item[field])
                    break
            
            # Extract rewards/scores (if available)
            reward_fields = ['reward', 'score', 'quality', 'rating']
            for field in reward_fields:
                if field in item:
                    try:
                        normalized_item['reward'] = float(item[field])
                    except (ValueError, TypeError):
                        pass
                    break
            
            # Extract preferences (for DPO-style data)
            if 'chosen' in item:
                normalized_item['chosen'] = str(item['chosen'])
            if 'rejected' in item:
                normalized_item['rejected'] = str(item['rejected'])
            
            # Extract metadata
            normalized_item['metadata'] = {
                k: v for k, v in item.items() 
                if k not in ['prompt', 'response', 'answer', 'chosen', 'rejected', 'reward']
            }
            
            # Only add if we have at least a prompt
            if 'prompt' in normalized_item:
                normalized.append(normalized_item)
        
        return normalized
    
    def load_inference_requests(self, file_path: str) -> List[Dict]:
        """Load inference requests from file"""
        requests_path = Path(file_path)
        if not requests_path.exists():
            return []
        
        try:
            if file_path.endswith('.jsonl'):
                return self._process_jsonl_file(requests_path)
            elif file_path.endswith('.json'):
                return self._process_json_file(requests_path)
            elif file_path.endswith('.csv'):
                return self._process_csv_file(requests_path)
            else:
                return self._process_txt_file(requests_path)
        except Exception as e:
            logger.error(f"Failed to load inference requests: {e}")
            return []

class FileWatcher(FileSystemEventHandler):
    """Watches for file changes and triggers data reload"""
    
    def __init__(self, callback):
        self.callback = callback
        self.last_triggered = 0
        self.debounce_time = 2  # seconds
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Debounce rapid file changes
        current_time = time.time()
        if current_time - self.last_triggered < self.debounce_time:
            return
        
        self.last_triggered = current_time
        logger.info(f"File changed: {event.src_path}")
        self.callback()

class FileBasedTrainingSystem:
    """Main file-based training and inference system"""
    
    def __init__(self, base_model, prm_model, orm_model, tokenizer, config: FileBasedConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        # Models
        self.model = base_model
        self.prm_model = prm_model
        self.orm_model = orm_model
        self.device = next(base_model.parameters()).device
        
        # Components
        self.data_processor = DataFileProcessor(config)
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
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'total_training_batches': 0,
            'total_inferences': 0,
            'data_reload_count': 0,
            'last_training_time': 0,
            'average_training_loss': deque(maxlen=100),
            'average_inference_score': deque(maxlen=100)
        }
        
        # Create output directories
        Path(config.results_output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start the file-based system"""
        self.running = True
        
        # Initial data load
        self._load_training_data()
        
        # Start file watcher
        self._start_file_watcher()
        
        # Start training thread
        if self.config.auto_training:
            self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
            self.training_thread.start()
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        logger.info("🚀 File-Based Training System Started")
        logger.info(f"📁 Watching directory: {self.config.training_data_dir}")
        logger.info(f"📥 Inference requests: {self.config.inference_requests_file}")
        logger.info(f"📤 Results output: {self.config.results_output_dir}")
    
    def stop(self):
        """Stop the system"""
        self.running = False
        
        if self.training_thread:
            self.training_thread.join()
        if self.inference_thread:
            self.inference_thread.join()
        if self.file_watcher_thread:
            self.file_watcher_thread.join()
        
        logger.info("🛑 File-Based System Stopped")
    
    def _start_file_watcher(self):
        """Start watching files for changes"""
        def watch_files():
            try:
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
                logger.info("📁 File watcher started")
                
                while self.running:
                    time.sleep(1)
                
                observer.stop()
                observer.join()
                
            except Exception as e:
                logger.error(f"File watcher error: {e}")
        
        self.file_watcher_thread = threading.Thread(target=watch_files, daemon=True)
        self.file_watcher_thread.start()
    
    def _on_file_change(self):
        """Handle file change events"""
        logger.info("📁 Files changed, reloading data...")
        self._load_training_data()
        self._load_inference_requests()
    
    def _load_training_data(self):
        """Load training data from files"""
        try:
            new_data = self.data_processor.load_training_data(self.config.training_data_dir)
            
            if new_data:
                # Filter and enhance data
                processed_data = self._enhance_training_data(new_data)
                self.training_data = processed_data
                self.stats['data_reload_count'] += 1
                
                logger.info(f"📊 Loaded {len(self.training_data)} training examples")
            else:
                logger.warning("⚠️  No training data found")
                
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
    
    def _enhance_training_data(self, data: List[Dict]) -> List[Dict]:
        """Enhance training data with reward scores if missing"""
        enhanced_data = []
        
        for item in data.copy():
            # If no reward is provided, estimate it
            if 'reward' not in item and 'response' in item:
                estimated_reward = self._estimate_reward(item['prompt'], item['response'])
                item['reward'] = estimated_reward
            
            # Default reward if still missing
            if 'reward' not in item:
                item['reward'] = 0.0
            
            # Add quality indicators
            item['quality_score'] = self._assess_quality(item)
            
            enhanced_data.append(item)
        
        # Sort by quality and keep best examples
        enhanced_data.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Limit cache size
        if len(enhanced_data) > self.config.max_cache_size:
            enhanced_data = enhanced_data[:self.config.max_cache_size]
        
        return enhanced_data
    
    def _estimate_reward(self, prompt: str, response: str) -> float:
        """Estimate reward using trained reward models"""
        try:
            full_text = f"{prompt} {response}"
            text_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                # Get PRM scores
                prm_outputs = self.prm_model.forward(text_ids)
                prm_scores = prm_outputs['step_rewards'][0].cpu().tolist()
                avg_prm_score = np.mean(prm_scores) if prm_scores else 0.0
                
                # Get ORM score
                attention_mask = torch.ones_like(text_ids)
                orm_outputs = self.orm_model.forward(text_ids, attention_mask)
                orm_score = orm_outputs['overall_reward'][0].item()
                
                # Combined score
                combined_score = (self.config.prm_weight * avg_prm_score + 
                                self.config.orm_weight * orm_score)
                
                return combined_score
                
        except Exception as e:
            logger.error(f"Reward estimation failed: {e}")
            return 0.0
    
    def _assess_quality(self, item: Dict) -> float:
        """Assess overall quality of data item"""
        quality_score = 0.0
        
        # Length-based quality
        if 'prompt' in item:
            prompt_len = len(item['prompt'].split())
            quality_score += min(prompt_len / 50, 1.0) * 0.2  # Prefer reasonable length prompts
        
        if 'response' in item:
            response_len = len(item['response'].split())
            quality_score += min(response_len / 100, 1.0) * 0.3  # Prefer substantial responses
        
        # Reward-based quality
        if 'reward' in item:
            quality_score += max(0, item['reward']) * 0.5
        
        return min(quality_score, 1.0)
    
    def _load_inference_requests(self):
        """Load inference requests from file"""
        try:
            requests = self.data_processor.load_inference_requests(
                self.config.inference_requests_file
            )
            
            # Add new requests to queue
            for request in requests:
                if 'prompt' in request:
                    self.pending_inference_requests.put(request)
            
            if requests:
                logger.info(f"📥 Loaded {len(requests)} inference requests")
                
        except Exception as e:
            logger.error(f"Failed to load inference requests: {e}")
    
    def _training_loop(self):
        """Main training loop"""
        logger.info("🎓 Training loop started")
        
        while self.running:
            try:
                if len(self.training_data) >= self.config.min_data_size_for_training:
                    self._train_on_data()
                    self.stats['last_training_time'] = time.time()
                
                # Wait before next training cycle
                time.sleep(self.config.data_refresh_interval)
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                time.sleep(5)
    
    def _train_on_data(self):
        """Train model on available data"""
        # Sample training batch
        batch_size = min(self.config.batch_size, len(self.training_data))
        
        # Weighted sampling based on quality
        weights = [item['quality_score'] for item in self.training_data]
        total_weight = sum(weights)
        
        if total_weight == 0:
            # Uniform sampling if no weights
            batch_data = random.sample(self.training_data, batch_size)
        else:
            # Weighted sampling
            probabilities = [w / total_weight for w in weights]
            batch_indices = np.random.choice(
                len(self.training_data), 
                size=batch_size, 
                p=probabilities, 
                replace=False
            )
            batch_data = [self.training_data[i] for i in batch_indices]
        
        # Create training dataset
        class FileBatchDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                
                # Use prompt + response if available, otherwise just prompt
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
                    'reward': torch.tensor(item.get('reward', 0.0), dtype=torch.float)
                }
        
        # Train
        dataset = FileBatchDataset(batch_data, self.tokenizer, self.config.max_sequence_length)
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
                
                # Reward-weighted loss
                base_loss = outputs.loss
                reward_weights = torch.sigmoid(batch['reward'])
                weighted_loss = base_loss * reward_weights.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += weighted_loss.item()
        
        # Update statistics
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        self.stats['average_training_loss'].append(avg_loss)
        self.stats['total_training_batches'] += 1
        
        logger.info(f"🎓 Training batch completed. Loss: {avg_loss:.4f}, Examples: {len(batch_data)}")
        
        # Save checkpoint periodically
        if self.stats['total_training_batches'] % 10 == 0:
            self._save_checkpoint()
    
    def _inference_loop(self):
        """Main inference loop"""
        logger.info("🔮 Inference loop started")
        
        while self.running:
            try:
                # Load new requests periodically
                self._load_inference_requests()
                
                # Process pending requests
                while not self.pending_inference_requests.empty():
                    request = self.pending_inference_requests.get()
                    result = self._process_inference_request(request)
                    
                    if result:
                        self.processed_results.append(result)
                        self._save_inference_result(result)
                
                time.sleep(2)  # Short sleep between checks
                
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                time.sleep(5)
    
    def _process_inference_request(self, request: Dict) -> Optional[Dict]:
        """Process single inference request"""
        try:
            prompt = request['prompt']
            
            # Generate response with reward guidance
            result = self._generate_with_rewards(prompt)
            
            # Add request metadata
            result.update({
                'request_id': request.get('id', f"req_{int(time.time())}"),
                'timestamp': time.time(),
                'metadata': request.get('metadata', {})
            })
            
            self.stats['total_inferences'] += 1
            self.stats['average_inference_score'].append(result.get('combined_score', 0.0))
            
            logger.info(f"🔮 Processed inference: {prompt[:50]}... Score: {result.get('combined_score', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Inference processing failed: {e}")
            return None
    
    def _generate_with_rewards(self, prompt: str) -> Dict[str, Any]:
        """Generate response with reward guidance"""
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
            
            # Evaluate candidates
            best_response = None
            best_score = float('-inf')
            best_details = {}
            
            for response in candidates:
                full_text = f"{prompt} {response}"
                text_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
                
                # PRM evaluation
                prm_outputs = self.prm_model.forward(text_ids)
                prm_scores = prm_outputs['step_rewards'][0].cpu().tolist()
                avg_prm_score = np.mean(prm_scores) if prm_scores else 0.0
                
                # ORM evaluation
                attention_mask = torch.ones_like(text_ids)
                orm_outputs = self.orm_model.forward(text_ids, attention_mask)
                orm_score = orm_outputs['overall_reward'][0].item()
                
                # Combined score
                combined_score = (self.config.prm_weight * avg_prm_score + 
                                self.config.orm_weight * orm_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_response = response
                    best_details = {
                        'prm_scores': prm_scores,
                        'avg_prm_score': avg_prm_score,
                        'orm_score': orm_score,
                        'combined_score': combined_score
                    }
        
        return {
            'prompt': prompt,
            'response': best_response,
            'candidates_count': len(candidates),
            **best_details
        }
    
    def _save_inference_result(self, result: Dict):
        """Save inference result to file"""
        if not self.config.save_intermediate_results:
            return
        
        try:
            output_file = Path(self.config.results_output_dir) / f"inference_results_{int(time.time() / 3600)}.jsonl"
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save inference result: {e}")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        try:
            timestamp = int(time.time())
            checkpoint_path = Path(self.config.checkpoints_dir) / f"model_checkpoint_{timestamp}.pt"
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'stats': dict(self.stats),
                'config': self.config.__dict__
            }, checkpoint_path)
            
            # Also save as latest
            latest_path = Path(self.config.checkpoints_dir) / "latest_checkpoint.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'stats': dict(self.stats),
                'config': self.config.__dict__
            }, latest_path)
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'stats' in checkpoint:
                self.stats.update(checkpoint['stats'])
            
            logger.info(f"📂 Loaded checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'stats': dict(self.stats),
            'training_data_size': len(self.training_data),
            'pending_requests': self.pending_inference_requests.qsize(),
            'processed_results': len(self.processed_results),
            'running': self.running,
            'uptime': time.time() - self.stats['start_time'],
            'config': self.config.__dict__
        }
    
    def export_results(self, output_path: str):
        """Export all processed results"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📤 Results exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

# Utility functions for creating sample data files
def create_sample_data_files():
    """Create sample data files for testing"""
    
    # Create directories
    Path("./training_data").mkdir(exist_ok=True)
    Path("./inference_requests").mkdir(exist_ok=True)
    Path("./results").mkdir(exist_ok=True)
    
    # Sample training data - JSON format
    training_json = [
        {
            "prompt": "Explain machine learning in simple terms",
            "response": "Machine learning is like teaching computers to learn patterns from data, similar to how humans learn from experience.",
            "reward": 0.8,
            "quality": "high"
        },
        {
            "prompt": "How do neural networks work?",
            "response": "Neural networks are inspired by the human brain, consisting of interconnected nodes that process and transmit information.",
            "reward": 0.9,
            "quality": "high"
        },
        {
            "prompt": "What is artificial intelligence?",
            "response": "AI is the simulation of human intelligence in machines programmed to think and act like humans.",
            "reward": 0.7,
            "quality": "medium"
        }
    ]
    
    with open("./training_data/ml_qa.json", "w") as f:
        json.dump(training_json, f, indent=2)
    
    # Sample training data - JSONL format
    training_jsonl = [
        {"prompt": "Solve: 2x + 5 = 11", "response": "First subtract 5: 2x = 6, then divide by 2: x = 3", "reward": 0.95},
        {"prompt": "What is photosynthesis?", "response": "Process where plants convert sunlight into energy using chlorophyll", "reward": 0.8},
        {"prompt": "Explain gravity", "response": "Force that attracts objects toward each other, stronger with more mass", "reward": 0.85}
    ]
    
    with open("./training_data/science_qa.jsonl", "w") as f:
        for item in training_jsonl:
            f.write(json.dumps(item) + "\n")
    
    # Sample training data - CSV format
    csv_data = [
        ["prompt", "response", "reward", "category"],
        ["What is Python?", "Programming language known for simplicity and readability", "0.9", "programming"],
        ["How to learn coding?", "Start with basics, practice daily, build projects, join communities", "0.85", "education"],
        ["What is debugging?", "Process of finding and fixing errors in computer programs", "0.8", "programming"]
    ]
    
    with open("./training_data/programming_qa.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    # Sample preference data for DPO
    preference_data = [
        {
            "prompt": "Explain climate change",
            "chosen": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities like burning fossil fuels.",
            "rejected": "Climate change is when weather gets different sometimes.",
            "category": "science"
        },
        {
            "prompt": "How to stay healthy?",
            "chosen": "Maintain a balanced diet, exercise regularly, get adequate sleep, manage stress, and have regular medical checkups.",
            "rejected": "Just eat whatever and hope for the best.",
            "category": "health"
        }
    ]
    
    with open("./training_data/preferences.json", "w") as f:
        json.dump(preference_data, f, indent=2)
    
    # Sample text file with prompts
    text_prompts = [
        "Explain quantum computing",
        "Write a short story about robots",
        "How do vaccines work?",
        "What is blockchain technology?",
        "Describe the water cycle"
    ]
    
    with open("./training_data/prompts.txt", "w") as f:
        for prompt in text_prompts:
            f.write(prompt + "\n")
    
    # Sample inference requests
    inference_requests = [
        {"id": "req_001", "prompt": "What is deep learning?", "priority": "high"},
        {"id": "req_002", "prompt": "Explain renewable energy", "priority": "medium"},
        {"id": "req_003", "prompt": "How does the internet work?", "priority": "low"},
        {"id": "req_004", "prompt": "What is cryptocurrency?", "priority": "medium"}
    ]
    
    with open("./inference_requests.jsonl", "w") as f:
        for req in inference_requests:
            f.write(json.dumps(req) + "\n")
    
    print("📁 Sample data files created:")
    print("  ./training_data/ml_qa.json")
    print("  ./training_data/science_qa.jsonl") 
    print("  ./training_data/programming_qa.csv")
    print("  ./training_data/preferences.json")
    print("  ./training_data/prompts.txt")
    print("  ./inference_requests.jsonl")

def create_dummy_models():
    """Create dummy models for testing"""
    class DummyTokenizer:
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
            return f"Generated response about the topic discussed ({len(tokens)} tokens)"
        
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
            batch_size, seq_len = input_ids.shape
            new_tokens = torch.randint(1, 32000, (batch_size, max_new_tokens))
            return torch.cat([input_ids, new_tokens], dim=1)
    
    class DummyPRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.score_head = nn.Linear(512, 1)
        
        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            # Generate realistic step rewards (generally increasing)
            base_scores = torch.linspace(0.3, 0.8, seq_len).unsqueeze(0).repeat(input_ids.size(0), 1)
            noise = torch.randn_like(base_scores) * 0.1
            step_rewards = torch.clamp(base_scores + noise, 0, 1)
            
            step_confidences = torch.rand(input_ids.size(0), seq_len) * 0.5 + 0.5
            
            return {
                'step_rewards': step_rewards,
                'step_confidences': step_confidences
            }
    
    class DummyORM(nn.Module):
        def __init__(self):
            super().__init__()
            self.reward_head = nn.Linear(512, 1)
        
        def forward(self, input_ids, attention_mask):
            batch_size = input_ids.size(0)
            # Generate realistic overall rewards
            base_reward = torch.rand(batch_size, 1) * 1.2 - 0.2  # -0.2 to 1.0
            
            return {
                'overall_reward': base_reward,
                'correctness': torch.sigmoid(base_reward + 0.2),
                'helpfulness': torch.tanh(base_reward)
            }
    
    return DummyModel(), DummyPRM(), DummyORM(), DummyTokenizer()

def test_file_based_system():
    """Test the file-based system"""
    print("🧪 Testing File-Based Training and Inference System")
    print("=" * 60)
    
    # Create sample data files
    create_sample_data_files()
    
    # Create models
    base_model, prm_model, orm_model, tokenizer = create_dummy_models()
    
    # Create config
    config = FileBasedConfig(
        training_data_dir="./training_data",
        inference_requests_file="./inference_requests.jsonl",
        results_output_dir="./results",
        batch_size=4,
        data_refresh_interval=10,
        auto_training=True,
        save_intermediate_results=True
    )
    
    # Create system
    system = FileBasedTrainingSystem(base_model, prm_model, orm_model, tokenizer, config)
    
    # Start system
    system.start()
    
    print("🔄 System started, processing files...")
    
    # Let it run for a while
    time.sleep(15)
    
    # Check status
    status = system.get_system_status()
    print("\n📊 System Status:")
    print("-" * 30)
    print(f"Training Data Loaded: {status['training_data_size']}")
    print(f"Training Batches: {status['stats']['total_training_batches']}")
    print(f"Total Inferences: {status['stats']['total_inferences']}")
    print(f"Pending Requests: {status['pending_requests']}")
    print(f"Processed Results: {status['processed_results']}")
    print(f"Data Reloads: {status['stats']['data_reload_count']}")
    print(f"Uptime: {status['uptime']:.1f}s")
    
    # Add more inference requests dynamically
    print("\n📥 Adding more inference requests...")
    additional_requests = [
        {"id": "req_005", "prompt": "What is machine learning?"},
        {"id": "req_006", "prompt": "Explain neural networks"},
        {"id": "req_007", "prompt": "How does AI work?"}
    ]
    
    with open("./inference_requests.jsonl", "a") as f:
        for req in additional_requests:
            f.write(json.dumps(req) + "\n")
    
    # Wait for processing
    time.sleep(10)
    
    # Add more training data
    print("\n📚 Adding more training data...")
    new_training_data = [
        {"prompt": "What is data science?", "response": "Field that uses scientific methods to extract insights from data", "reward": 0.85},
        {"prompt": "Explain algorithms", "response": "Step-by-step procedures for solving problems or completing tasks", "reward": 0.9}
    ]
    
    with open("./training_data/additional_data.jsonl", "w") as f:
        for item in new_training_data:
            f.write(json.dumps(item) + "\n")
    
    # Wait for file watcher to detect changes
    time.sleep(5)
    
    # Final status check
    final_status = system.get_system_status()
    print("\n📈 Final Status:")
    print("-" * 30)
    print(f"Training Data: {final_status['training_data_size']}")
    print(f"Training Batches: {final_status['stats']['total_training_batches']}")
    print(f"Inferences: {final_status['stats']['total_inferences']}")
    print(f"Data Reloads: {final_status['stats']['data_reload_count']}")
    
    # Export results
    system.export_results("./final_results.json")
    
    # Stop system
    system.stop()
    
    print("\n✅ File-based system test completed!")
    print("📁 Check the following directories for outputs:")
    print("  ./results/ - Inference results")
    print("  ./checkpoints/ - Model checkpoints")
    print("  ./final_results.json - Exported results")
    
    return system

def run_production_file_system():
    """Run production file-based system"""
    print("🌐 Starting Production File-Based System")
    print("=" * 50)
    
    # Check if sample data exists, create if not
    if not Path("./training_data").exists():
        print("📁 Creating sample data files...")
        create_sample_data_files()
    
    # Create models
    base_model, prm_model, orm_model, tokenizer = create_dummy_models()
    
    # Production config
    config = FileBasedConfig(
        training_data_dir="./training_data",
        inference_requests_file="./inference_requests.jsonl",
        results_output_dir="./results",
        checkpoints_dir="./checkpoints",
        batch_size=16,
        data_refresh_interval=60,  # Check every minute
        auto_training=True,
        save_intermediate_results=True,
        max_cache_size=50000
    )
    
    # Create and start system
    system = FileBasedTrainingSystem(base_model, prm_model, orm_model, tokenizer, config)
    
    # Try to load latest checkpoint
    latest_checkpoint = Path(config.checkpoints_dir) / "latest_checkpoint.pt"
    if latest_checkpoint.exists():
        print(f"📂 Loading checkpoint: {latest_checkpoint}")
        system.load_checkpoint(str(latest_checkpoint))
    
    system.start()
    
    print("🚀 Production system started")
    print("📁 Monitoring directories:")
    print(f"  Training data: {config.training_data_dir}")
    print(f"  Inference requests: {config.inference_requests_file}")
    print(f"  Results: {config.results_output_dir}")
    print(f"  Checkpoints: {config.checkpoints_dir}")
    print("\nAdd files to training_data/ directory and requests to inference_requests.jsonl")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Print status every 30 seconds
            time.sleep(30)
            status = system.get_system_status()
            
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Data: {status['training_data_size']}, "
                  f"Trained: {status['stats']['total_training_batches']}, "
                  f"Inferences: {status['stats']['total_inferences']}, "
                  f"Pending: {status['pending_requests']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down production system...")
        system.stop()
        
        # Export final results
        timestamp = int(time.time())
        system.export_results(f"./production_results_{timestamp}.json")
        
        # Final stats
        final_status = system.get_system_status()
        print("\n📈 Production Run Summary:")
        print(f"Uptime: {final_status['uptime']:.1f}s")
        print(f"Training Batches: {final_status['stats']['total_training_batches']}")
        print(f"Total Inferences: {final_status['stats']['total_inferences']}")
        print(f"Data Reloads: {final_status['stats']['data_reload_count']}")
        
        if final_status['stats']['average_training_loss']:
            avg_loss = np.mean(list(final_status['stats']['average_training_loss']))
            print(f"Average Training Loss: {avg_loss:.4f}")
        
        if final_status['stats']['average_inference_score']:
            avg_score = np.mean(list(final_status['stats']['average_inference_score']))
            print(f"Average Inference Score: {avg_score:.4f}")
        
        print("✅ Production system stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "production":
            run_production_file_system()
        elif sys.argv[1] == "create_data":
            create_sample_data_files()
            print("✅ Sample data files created!")
        else:
            print("Usage: python file_based_training_inference.py [production|create_data]")
    else:
        test_file_based_system()