# training_infra/inference/engine.py
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import time
from dataclasses import dataclass, field
from .sampling import SamplingConfig, create_sampler, BaseSampler

@dataclass
class GenerationConfig:
    """Complete configuration for text generation"""
    # Inherited from SamplingConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # Generation specific
    max_length: int = 100
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    
    # Stopping criteria
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    stop_strings: Optional[List[str]] = None
    
    # Batching and efficiency
    batch_size: int = 1
    use_cache: bool = True
    
    # Stream generation
    stream: bool = False
    stream_callback: Optional[Callable] = None
    
    # Performance monitoring
    profile: bool = False
    return_dict_in_generate: bool = False
    output_scores: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
    
    # Safety and filtering
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Optional[List[List[int]]] = None
    
    def __post_init__(self):
        if self.max_new_tokens is None:
            self.max_new_tokens = self.max_length
        if self.min_new_tokens is None:
            self.min_new_tokens = self.min_length

class InferenceEngine:
    """Complete inference engine for language models"""
    
    def __init__(self, model: torch.nn.Module, tokenizer=None, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Performance tracking
        self.stats = {
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'num_generations': 0
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main generation function with comprehensive sampling support
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            config: Generation configuration
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        start_time = time.time()
        
        # Setup
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # Initialize generation state
        generated_tokens = []
        past_key_values = None
        current_length = input_length
        
        # Create sampler
        sampler = create_sampler(config.sampling)
        
        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Stopping criteria
        stop_criteria = self._create_stopping_criteria(config, batch_size, device)
        
        # Generation loop
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Store outputs if requested
        all_scores = [] if config.output_scores else None
        all_attentions = [] if config.output_attentions else None
        all_hidden_states = [] if config.output_hidden_states else None
        
        while True:
            # Prepare model inputs
            model_inputs = self._prepare_inputs_for_generation(
                input_ids, past_key_values, attention_mask, config.use_cache
            )
            
            # Forward pass
            outputs = self.model(**model_inputs)
            
            # Extract logits and optional outputs
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            if config.use_cache:
                past_key_values = outputs.past_key_values
            
            # Store intermediate outputs
            if config.output_scores:
                all_scores.append(logits)
            if config.output_attentions and hasattr(outputs, 'attentions'):
                all_attentions.append(outputs.attentions)
            if config.output_hidden_states and hasattr(outputs, 'hidden_states'):
                all_hidden_states.append(outputs.hidden_states)
            
            # Apply logits processors
            logits = self._apply_logits_processors(logits, input_ids, config)
            
            # Sample next token
            next_tokens = sampler.sample(
                logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
            
            # Update sequences
            next_tokens = next_tokens * unfinished_sequences + config.sampling.pad_token_id * (1 - unfinished_sequences)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            
            generated_tokens.append(next_tokens)
            current_length += 1
            
            # Stream callback
            if config.stream and config.stream_callback:
                config.stream_callback(next_tokens, input_ids)
            
            # Check stopping criteria
            unfinished_sequences = self._update_stopping_criteria(
                unfinished_sequences, next_tokens, current_length, stop_criteria, config
            )
            
            # Break if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
            
            # Break if max length reached
            if current_length >= config.max_length:
                break
        
        # Prepare output
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Update stats
        self.stats['total_tokens_generated'] += len(generated_tokens) * batch_size
        self.stats['total_time'] += generation_time
        self.stats['num_generations'] += 1
        
        # Create output dictionary
        output = {
            'sequences': input_ids,
            'generated_tokens': torch.stack(generated_tokens, dim=1) if generated_tokens else torch.empty((batch_size, 0), device=device),
            'generation_time': generation_time,
            'tokens_per_second': len(generated_tokens) / generation_time if generation_time > 0 else 0
        }
        
        if config.output_scores:
            output['scores'] = torch.stack(all_scores, dim=1)
        if config.output_attentions:
            output['attentions'] = all_attentions
        if config.output_hidden_states:
            output['hidden_states'] = all_hidden_states
            
        return output
    
    def _prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple],
        attention_mask: Optional[torch.Tensor],
        use_cache: bool
    ) -> Dict[str, Any]:
        """Prepare inputs for model forward pass"""
        
        model_inputs = {
            'input_ids': input_ids,
            'use_cache': use_cache
        }
        
        if past_key_values is not None:
            model_inputs['past_key_values'] = past_key_values
            # Only use last token when using cache
            model_inputs['input_ids'] = input_ids[:, -1:]
        
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask
        
        return model_inputs
    
    def _apply_logits_processors(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """Apply various logits processors"""
        
        # Repetition penalty
        if config.sampling.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, input_ids, config.sampling.repetition_penalty)
        
        # Frequency penalty
        if config.sampling.frequency_penalty != 0.0:
            logits = self._apply_frequency_penalty(logits, input_ids, config.sampling.frequency_penalty)
        
        # Presence penalty
        if config.sampling.presence_penalty != 0.0:
            logits = self._apply_presence_penalty(logits, input_ids, config.sampling.presence_penalty)
        
        # Bad words filter
        if config.bad_words_ids:
            logits = self._apply_bad_words_filter(logits, config.bad_words_ids)
        
        # Force words
        if config.force_words_ids:
            logits = self._apply_force_words(logits, input_ids, config.force_words_ids)
        
        return logits
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        if penalty == 1.0:
            return logits
        
        batch_size = logits.shape[0]
        for batch_idx in range(batch_size):
            for token in input_ids[batch_idx].unique():
                if token < logits.shape[-1]:
                    if logits[batch_idx, token] < 0:
                        logits[batch_idx, token] *= penalty
                    else:
                        logits[batch_idx, token] /= penalty
        
        return logits
    
    def _apply_frequency_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """Apply frequency penalty to logits"""
        if penalty == 0.0:
            return logits
        
        batch_size = logits.shape[0]
        for batch_idx in range(batch_size):
            # Count token frequencies
            unique_tokens, counts = input_ids[batch_idx].unique(return_counts=True)
            
            for token, count in zip(unique_tokens, counts):
                if token < logits.shape[-1]:
                    logits[batch_idx, token] -= penalty * count
        
        return logits
    
    def _apply_presence_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """Apply presence penalty to logits"""
        if penalty == 0.0:
            return logits
        
        batch_size = logits.shape[0]
        for batch_idx in range(batch_size):
            # Get unique tokens (presence)
            unique_tokens = input_ids[batch_idx].unique()
            
            for token in unique_tokens:
                if token < logits.shape[-1]:
                    logits[batch_idx, token] -= penalty
        
        return logits
    
    def _apply_bad_words_filter(self, logits: torch.Tensor, bad_words_ids: List[List[int]]) -> torch.Tensor:
        """Filter out bad words by setting their logits to -inf"""
        for bad_word in bad_words_ids:
            if len(bad_word) == 1:  # Single token bad word
                token_id = bad_word[0]
                if token_id < logits.shape[-1]:
                    logits[:, token_id] = float('-inf')
        
        return logits
    
    def _apply_force_words(self, logits: torch.Tensor, input_ids: torch.Tensor, force_words_ids: List[List[int]]) -> torch.Tensor:
        """Force certain words by boosting their logits"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated logic to handle multi-token forced words
        for force_word in force_words_ids:
            if len(force_word) == 1:  # Single token forced word
                token_id = force_word[0]
                if token_id < logits.shape[-1]:
                    logits[:, token_id] += 10.0  # Boost logit
        
        return logits
    
    def _create_stopping_criteria(self, config: GenerationConfig, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """Create stopping criteria"""
        criteria = {
            'max_length': config.max_length,
            'max_new_tokens': config.max_new_tokens,
            'min_length': config.min_length,
            'min_new_tokens': config.min_new_tokens,
            'eos_token_id': config.eos_token_id,
            'stop_strings': config.stop_strings,
        }
        return criteria
    
    def _update_stopping_criteria(
        self,
        unfinished_sequences: torch.Tensor,
        next_tokens: torch.Tensor,
        current_length: int,
        stop_criteria: Dict[str, Any],
        config: GenerationConfig
    ) -> torch.Tensor:
        """Update which sequences should continue generating"""
        
        # Check EOS tokens
        if stop_criteria['eos_token_id'] is not None:
            if isinstance(stop_criteria['eos_token_id'], int):
                eos_ids = [stop_criteria['eos_token_id']]
            else:
                eos_ids = stop_criteria['eos_token_id']
            
            for eos_id in eos_ids:
                unfinished_sequences = unfinished_sequences * (next_tokens != eos_id).long()
        
        # Check minimum length
        if current_length < stop_criteria['min_length']:
            # Don't stop any sequences yet
            pass
        
        return unfinished_sequences
    
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """Generator for streaming text generation"""
        config.stream = True
        
        # Store generated tokens for yielding
        generated_tokens = []
        
        def stream_callback(tokens, full_sequence):
            generated_tokens.append(tokens)
            return tokens
        
        config.stream_callback = stream_callback
        
        # Start generation in a separate thread or async manner
        result = self.generate(input_ids, config, attention_mask)
        
        # Yield tokens as they're generated
        for tokens in generated_tokens:
            yield tokens
    
    def batch_generate(
        self,
        input_ids_list: List[torch.Tensor],
        configs: List[GenerationConfig],
        attention_masks: Optional[List[torch.Tensor]] = None
    ) -> List[Dict[str, Any]]:
        """Generate for multiple inputs with different configs"""
        results = []
        
        for i, (input_ids, config) in enumerate(zip(input_ids_list, configs)):
            attention_mask = attention_masks[i] if attention_masks else None
            result = self.generate(input_ids, config, attention_mask)
            results.append(result)
        
        return results
    
    def generate_with_guidance(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        guidance_model: Optional[torch.nn.Module] = None,
        guidance_scale: float = 1.5,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate with classifier-free guidance"""
        if guidance_model is None or guidance_scale == 1.0:
            return self.generate(input_ids, config, attention_mask)
        
        # This is a simplified implementation of guidance
        # In practice, you'd need more sophisticated logic
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # Duplicate inputs for guided and unguided generation
        guided_input_ids = input_ids.repeat(2, 1)
        if attention_mask is not None:
            guided_attention_mask = attention_mask.repeat(2, 1)
        else:
            guided_attention_mask = None
        
        # Generate with both models
        guided_config = GenerationConfig(**config.__dict__)
        guided_config.batch_size = batch_size * 2
        
        # Custom generation loop with guidance
        return self._generate_with_guidance_loop(
            guided_input_ids, guided_config, guidance_model, 
            guidance_scale, guided_attention_mask
        )
    
    def _generate_with_guidance_loop(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        guidance_model: torch.nn.Module,
        guidance_scale: float,
        attention_mask: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Internal guidance generation loop"""
        # Simplified implementation
        # In practice, you'd implement proper CFG logic here
        return self.generate(input_ids[:input_ids.shape[0]//2], config, 
                           attention_mask[:attention_mask.shape[0]//2] if attention_mask is not None else None)
    
    def benchmark(self, input_ids: torch.Tensor, config: GenerationConfig, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark generation performance"""
        times = []
        tokens_per_second = []
        
        for _ in range(num_runs):
            start_time = time.time()
            result = self.generate(input_ids, config)
            end_time = time.time()
            
            generation_time = end_time - start_time
            num_tokens = result['generated_tokens'].shape[1]
            
            times.append(generation_time)
            if generation_time > 0:
                tokens_per_second.append(num_tokens / generation_time)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'avg_tokens_per_second': sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0,
            'max_tokens_per_second': max(tokens_per_second) if tokens_per_second else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        avg_tokens_per_second = (
            self.stats['total_tokens_generated'] / self.stats['total_time'] 
            if self.stats['total_time'] > 0 else 0
        )
        
        return {
            **self.stats,
            'avg_tokens_per_second': avg_tokens_per_second
        }
    
    def reset_stats(self):
        """Reset generation statistics"""
        self.stats = {
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'num_generations': 0
        }

# High-level generation functions
def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    **kwargs
) -> str:
    """High-level text generation function"""
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Create config
    sampling_config = SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        **kwargs
    )
    
    generation_config = GenerationConfig(
        sampling=sampling_config,
        max_length=max_length
    )
    
    # Create engine and generate
    engine = InferenceEngine(model, tokenizer)
    result = engine.generate(input_ids, generation_config)
    
    # Decode result
    generated_ids = result['sequences'][0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

def chat_generate(
    model: torch.nn.Module,
    tokenizer,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    max_length: int = 500,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Generate chat response"""
    
    # Format chat messages
    if system_prompt:
        prompt = f"System: {system_prompt}\n"
    else:
        prompt = ""
    
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        prompt += f"{role.capitalize()}: {content}\n"
    
    prompt += "Assistant:"
    
    return generate_text(
        model, tokenizer, prompt, max_length=max_length, 
        temperature=temperature, **kwargs
    )

def batch_generate_texts(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    configs: Optional[List[GenerationConfig]] = None,
    **kwargs
) -> List[str]:
    """Generate multiple texts in batch"""
    
    if configs is None:
        # Create default config for all prompts
        default_config = GenerationConfig(
            sampling=SamplingConfig(**kwargs)
        )
        configs = [default_config] * len(prompts)
    
    # Tokenize all prompts
    input_ids_list = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        input_ids_list.append(input_ids)
    
    # Generate
    engine = InferenceEngine(model, tokenizer)
    results = engine.batch_generate(input_ids_list, configs)
    
    # Decode results
    generated_texts = []
    for i, result in enumerate(results):
        input_length = input_ids_list[i].shape[1]
        generated_ids = result['sequences'][0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

# Example usage functions
def create_creative_writing_config() -> GenerationConfig:
    """Config optimized for creative writing"""
    return GenerationConfig(
        sampling=SamplingConfig(
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.05,
            frequency_penalty=0.1
        ),
        max_new_tokens=500
    )

def create_code_generation_config() -> GenerationConfig:
    """Config optimized for code generation"""
    return GenerationConfig(
        sampling=SamplingConfig(
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1
        ),
        max_new_tokens=200
    )

def create_chat_config() -> GenerationConfig:
    """Config optimized for chat"""
    return GenerationConfig(
        sampling=SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.1
        ),
        max_new_tokens=300
    )

def create_factual_config() -> GenerationConfig:
    """Config optimized for factual responses"""
    return GenerationConfig(
        sampling=SamplingConfig(
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        max_new_tokens=150
    )

class BatchInferenceEngine(InferenceEngine):
    """Optimized engine for batch inference"""
    
    def __init__(self, model, tokenizer=None, device=None, batch_size=8):
        super().__init__(model, tokenizer, device)
        self.default_batch_size = batch_size
    
    def generate_batch(self, prompts: List[str], config: GenerationConfig = None, **kwargs):
        """Generate for multiple prompts efficiently"""
        if config is None:
            config = GenerationConfig()
        
        # Tokenize all prompts
        input_ids_list = []
        for prompt in prompts:
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            else:
                input_ids = torch.tensor([prompt])  # Placeholder
            input_ids_list.append(input_ids)
        
        # Batch process
        results = []
        for i in range(0, len(input_ids_list), self.default_batch_size):
            batch = input_ids_list[i:i + self.default_batch_size]
            
            # Process batch
            for input_ids in batch:
                result = self.generate(input_ids, config)
                results.append(result)
        
        return results

class StreamingInferenceEngine(InferenceEngine):
    """Engine for streaming text generation"""
    
    def __init__(self, model, tokenizer=None, device=None):
        super().__init__(model, tokenizer, device)
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None, **kwargs):
        """Generate text with streaming output"""
        if config is None:
            config = GenerationConfig()
        
        # Enable streaming
        config.stream = True
        
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        else:
            input_ids = torch.tensor([[1]])  # Placeholder
        
        # Generator for streaming
        def token_generator():
            result = self.generate(input_ids, config)
            generated_tokens = result['generated_tokens']
            
            for i in range(generated_tokens.shape[1]):
                token = generated_tokens[0, i]
                if self.tokenizer:
                    text = self.tokenizer.decode([token])
                else:
                    text = str(token.item())
                yield text
        
        return token_generator()
    
    def stream_chat(self, messages: List[Dict[str, str]], config: GenerationConfig = None):
        """Stream chat responses"""
        # Format messages into prompt
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant: "
        
        return self.generate_stream(prompt, config)