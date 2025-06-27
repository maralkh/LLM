# training_infra/distributed/microbatch_scheduler.py
"""
Microbatch scheduler for pipeline parallelism in distributed LLaMA training.
"""

import time
import torch
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging


@dataclass
class ScheduleStep:
    """Represents a single step in the pipeline schedule"""
    step_type: str  # 'forward', 'backward', 'optimizer_step', 'communication'
    microbatch_id: Optional[int] = None
    stage_id: Optional[int] = None
    phase: Optional[str] = None  # 'warmup', 'steady', 'cooldown', 'sync'
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineSchedule(ABC):
    """Abstract base class for pipeline schedules"""
    
    def __init__(
        self,
        num_microbatches: int,
        num_pipeline_stages: int,
        current_stage: int
    ):
        self.num_microbatches = num_microbatches
        self.num_pipeline_stages = num_pipeline_stages
        self.current_stage = current_stage
        
    @abstractmethod
    def generate_schedule(self) -> List[ScheduleStep]:
        """Generate the execution schedule for current pipeline stage"""
        pass
    
    @abstractmethod
    def get_schedule_name(self) -> str:
        """Get the name of this schedule"""
        pass


class GPipeSchedule(PipelineSchedule):
    """GPipe schedule: all forwards then all backwards"""
    
    def generate_schedule(self) -> List[ScheduleStep]:
        schedule = []
        
        # Forward phase: process all microbatches
        for mb_id in range(self.num_microbatches):
            schedule.append(ScheduleStep(
                step_type='forward',
                microbatch_id=mb_id,
                stage_id=self.current_stage,
                phase='forward'
            ))
        
        # Backward phase: process all microbatches in reverse order
        for mb_id in range(self.num_microbatches - 1, -1, -1):
            schedule.append(ScheduleStep(
                step_type='backward',
                microbatch_id=mb_id,
                stage_id=self.current_stage,
                phase='backward'
            ))
        
        # Synchronization step
        schedule.append(ScheduleStep(
            step_type='optimizer_step',
            stage_id=self.current_stage,
            phase='sync'
        ))
        
        return schedule
    
    def get_schedule_name(self) -> str:
        return "GPipe"


class OneForwardOneBackwardSchedule(PipelineSchedule):
    """1F1B schedule: interleaved forwards and backwards"""
    
    def generate_schedule(self) -> List[ScheduleStep]:
        schedule = []
        
        # Warmup phase: forward passes to fill pipeline
        warmup_steps = min(self.num_pipeline_stages - 1 - self.current_stage, self.num_microbatches)
        
        for mb_id in range(warmup_steps):
            schedule.append(ScheduleStep(
                step_type='forward',
                microbatch_id=mb_id,
                stage_id=self.current_stage,
                phase='warmup'
            ))
        
        # Steady state: 1F1B pattern
        steady_steps = self.num_microbatches - warmup_steps
        
        for i in range(steady_steps):
            forward_mb_id = warmup_steps + i
            backward_mb_id = i
            
            # Forward pass for new microbatch
            if forward_mb_id < self.num_microbatches:
                schedule.append(ScheduleStep(
                    step_type='forward',
                    microbatch_id=forward_mb_id,
                    stage_id=self.current_stage,
                    phase='steady'
                ))
            
            # Backward pass for older microbatch
            schedule.append(ScheduleStep(
                step_type='backward',
                microbatch_id=backward_mb_id,
                stage_id=self.current_stage,
                phase='steady'
            ))
        
        # Cooldown phase: remaining backward passes
        remaining_backwards = warmup_steps
        for i in range(remaining_backwards):
            backward_mb_id = steady_steps + i
            schedule.append(ScheduleStep(
                step_type='backward',
                microbatch_id=backward_mb_id,
                stage_id=self.current_stage,
                phase='cooldown'
            ))
        
        # Synchronization step
        schedule.append(ScheduleStep(
            step_type='optimizer_step',
            stage_id=self.current_stage,
            phase='sync'
        ))
        
        return schedule
    
    def get_schedule_name(self) -> str:
        return "1F1B"


class ChimeraSchedule(PipelineSchedule):
    """Chimera schedule: adaptive scheduling based on memory pressure"""
    
    def __init__(self, *args, memory_threshold_gb: float = 16.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_threshold_gb = memory_threshold_gb
        self._use_1f1b = True  # Start with 1F1B
        
    def generate_schedule(self) -> List[ScheduleStep]:
        # Decide between 1F1B and GPipe based on memory pressure
        current_memory_gb = self._get_current_memory_usage()
        
        if current_memory_gb > self.memory_threshold_gb:
            self._use_1f1b = False  # Switch to GPipe for lower memory
        else:
            self._use_1f1b = True   # Use 1F1B for better efficiency
        
        if self._use_1f1b:
            scheduler = OneForwardOneBackwardSchedule(
                self.num_microbatches,
                self.num_pipeline_stages,
                self.current_stage
            )
        else:
            scheduler = GPipeSchedule(
                self.num_microbatches,
                self.num_pipeline_stages,
                self.current_stage
            )
        
        schedule = scheduler.generate_schedule()
        
        # Add metadata about which schedule was used
        for step in schedule:
            step.metadata['actual_schedule'] = "1F1B" if self._use_1f1b else "GPipe"
            step.metadata['memory_gb'] = current_memory_gb
        
        return schedule
    
    def _get_current_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def get_schedule_name(self) -> str:
        return f"Chimera ({'1F1B' if self._use_1f1b else 'GPipe'})"


class InterleavedSchedule(PipelineSchedule):
    """Interleaved schedule with virtual pipeline stages"""
    
    def __init__(self, *args, num_virtual_stages: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_virtual_stages = num_virtual_stages
        
    def generate_schedule(self) -> List[ScheduleStep]:
        schedule = []
        
        # Distribute microbatches across virtual stages
        microbatches_per_virtual_stage = self.num_microbatches // self.num_virtual_stages
        
        for virtual_stage in range(self.num_virtual_stages):
            start_mb = virtual_stage * microbatches_per_virtual_stage
            end_mb = start_mb + microbatches_per_virtual_stage
            
            # Forward passes for this virtual stage
            for mb_id in range(start_mb, min(end_mb, self.num_microbatches)):
                schedule.append(ScheduleStep(
                    step_type='forward',
                    microbatch_id=mb_id,
                    stage_id=self.current_stage,
                    phase='forward',
                    metadata={'virtual_stage': virtual_stage}
                ))
            
            # Backward passes for this virtual stage
            for mb_id in range(min(end_mb, self.num_microbatches) - 1, start_mb - 1, -1):
                schedule.append(ScheduleStep(
                    step_type='backward',
                    microbatch_id=mb_id,
                    stage_id=self.current_stage,
                    phase='backward',
                    metadata={'virtual_stage': virtual_stage}
                ))
        
        # Synchronization step
        schedule.append(ScheduleStep(
            step_type='optimizer_step',
            stage_id=self.current_stage,
            phase='sync'
        ))
        
        return schedule
    
    def get_schedule_name(self) -> str:
        return f"Interleaved-{self.num_virtual_stages}V"


@dataclass
class MicrobatchMetrics:
    """Metrics for microbatch execution"""
    total_forward_time: float = 0.0
    total_backward_time: float = 0.0
    total_communication_time: float = 0.0
    total_bubble_time: float = 0.0
    num_microbatches_processed: int = 0
    memory_peak_gb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    
    @property
    def avg_forward_time(self) -> float:
        return self.total_forward_time / max(self.num_microbatches_processed, 1)
    
    @property
    def avg_backward_time(self) -> float:
        return self.total_backward_time / max(self.num_microbatches_processed, 1)
    
    @property
    def bubble_ratio(self) -> float:
        total_compute = self.total_forward_time + self.total_backward_time
        return self.total_bubble_time / max(total_compute, 1e-8)
    
    @property
    def efficiency(self) -> float:
        """Pipeline efficiency (1 - bubble_ratio)"""
        return 1.0 - self.bubble_ratio


class MicrobatchScheduler:
    """Main microbatch scheduler for pipeline parallelism"""
    
    def __init__(
        self,
        num_microbatches: int,
        microbatch_size: int,
        num_pipeline_stages: int,
        current_stage: int,
        schedule_type: str = "1f1b",
        virtual_stages: Optional[int] = None,
        enable_profiling: bool = True
    ):
        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size
        self.num_pipeline_stages = num_pipeline_stages
        self.current_stage = current_stage
        self.schedule_type = schedule_type
        self.virtual_stages = virtual_stages
        self.enable_profiling = enable_profiling
        
        # Metrics tracking
        self.metrics = MicrobatchMetrics()
        self.step_times = {}
        self.current_schedule = None
        
        # Create schedule
        self.schedule = self._create_schedule()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _create_schedule(self) -> PipelineSchedule:
        """Create the appropriate schedule based on configuration"""
        
        if self.schedule_type.lower() == "gpipe":
            return GPipeSchedule(
                self.num_microbatches,
                self.num_pipeline_stages,
                self.current_stage
            )
        elif self.schedule_type.lower() == "1f1b":
            return OneForwardOneBackwardSchedule(
                self.num_microbatches,
                self.num_pipeline_stages,
                self.current_stage
            )
        elif self.schedule_type.lower() == "chimera":
            return ChimeraSchedule(
                self.num_microbatches,
                self.num_pipeline_stages,
                self.current_stage
            )
        elif self.schedule_type.lower() == "interleaved":
            virtual_stages = self.virtual_stages or 2
            return InterleavedSchedule(
                self.num_microbatches,
                self.num_pipeline_stages,
                self.current_stage,
                num_virtual_stages=virtual_stages
            )
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_execution_schedule(self) -> List[ScheduleStep]:
        """Get the execution schedule for current iteration"""
        self.current_schedule = self.schedule.generate_schedule()
        return self.current_schedule
    
    def execute_schedule(
        self,
        forward_fn,
        backward_fn,
        optimizer_step_fn,
        microbatch_data: List[Any]
    ) -> Dict[str, Any]:
        """Execute the complete pipeline schedule"""
        
        results = {
            'losses': [],
            'total_time': 0.0,
            'schedule_name': self.schedule.get_schedule_name()
        }
        
        schedule = self.get_execution_schedule()
        start_time = time.time()
        
        # Track step execution times
        step_start_times = {}
        
        for step in schedule:
            step_start = time.time()
            
            try:
                if step.step_type == 'forward':
                    result = self._execute_forward_step(
                        step, forward_fn, microbatch_data
                    )
                    if result is not None:
                        results['losses'].append(result)
                        
                elif step.step_type == 'backward':
                    self._execute_backward_step(step, backward_fn)
                    
                elif step.step_type == 'optimizer_step':
                    self._execute_optimizer_step(step, optimizer_step_fn)
                    
                step_time = time.time() - step_start
                
                # Record timing
                if self.enable_profiling:
                    self._record_step_timing(step, step_time)
                    
            except Exception as e:
                self.logger.error(f"Error executing step {step}: {e}")
                raise
        
        results['total_time'] = time.time() - start_time
        
        # Update metrics
        if self.enable_profiling:
            self._update_metrics(results)
        
        return results
    
    def _execute_forward_step(
        self,
        step: ScheduleStep,
        forward_fn,
        microbatch_data: List[Any]
    ) -> Optional[torch.Tensor]:
        """Execute a forward step"""
        
        microbatch_id = step.microbatch_id
        if microbatch_id >= len(microbatch_data):
            self.logger.warning(f"Microbatch {microbatch_id} not available")
            return None
        
        microbatch = microbatch_data[microbatch_id]
        
        # Execute forward pass
        with torch.cuda.nvtx.range(f"Forward-MB{microbatch_id}") if torch.cuda.is_available() else torch.no_grad():
            result = forward_fn(microbatch, microbatch_id)
        
        return result
    
    def _execute_backward_step(
        self,
        step: ScheduleStep,
        backward_fn
    ):
        """Execute a backward step"""
        
        microbatch_id = step.microbatch_id
        
        # Execute backward pass
        with torch.cuda.nvtx.range(f"Backward-MB{microbatch_id}") if torch.cuda.is_available() else torch.no_grad():
            backward_fn(microbatch_id)
    
    def _execute_optimizer_step(
        self,
        step: ScheduleStep,
        optimizer_step_fn
    ):
        """Execute optimizer step"""
        
        with torch.cuda.nvtx.range("OptimizerStep") if torch.cuda.is_available() else torch.no_grad():
            optimizer_step_fn()
    
    def _record_step_timing(self, step: ScheduleStep, step_time: float):
        """Record timing for a step"""
        
        step_type = step.step_type
        phase = step.phase or 'unknown'
        
        if step_type not in self.step_times:
            self.step_times[step_type] = {}
        
        if phase not in self.step_times[step_type]:
            self.step_times[step_type][phase] = []
        
        self.step_times[step_type][phase].append(step_time)
        
        # Update running metrics
        if step_type == 'forward':
            self.metrics.total_forward_time += step_time
        elif step_type == 'backward':
            self.metrics.total_backward_time += step_time
    
    def _update_metrics(self, results: Dict[str, Any]):
        """Update performance metrics"""
        
        self.metrics.num_microbatches_processed += self.num_microbatches
        
        # Update memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.metrics.memory_peak_gb = max(self.metrics.memory_peak_gb, current_memory)
        
        # Calculate throughput (tokens per second)
        if results['total_time'] > 0:
            total_tokens = self.num_microbatches * self.microbatch_size * 512  # Assume 512 seq length
            self.metrics.throughput_tokens_per_sec = total_tokens / results['total_time']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        return {
            'schedule_name': self.schedule.get_schedule_name(),
            'num_microbatches': self.num_microbatches,
            'microbatch_size': self.microbatch_size,
            'avg_forward_time_ms': self.metrics.avg_forward_time * 1000,
            'avg_backward_time_ms': self.metrics.avg_backward_time * 1000,
            'bubble_ratio': self.metrics.bubble_ratio,
            'efficiency': self.metrics.efficiency,
            'throughput_tokens_per_sec': self.metrics.throughput_tokens_per_sec,
            'memory_peak_gb': self.metrics.memory_peak_gb,
            'total_microbatches_processed': self.metrics.num_microbatches_processed
        }
    
    def print_metrics(self):
        """Print detailed performance metrics"""
        
        metrics = self.get_metrics()
        
        print(f"\n📊 Pipeline Schedule Metrics ({metrics['schedule_name']})")
        print("=" * 60)
        print(f"Microbatches: {metrics['num_microbatches']} x {metrics['microbatch_size']}")
        print(f"Average Forward Time: {metrics['avg_forward_time_ms']:.2f} ms")
        print(f"Average Backward Time: {metrics['avg_backward_time_ms']:.2f} ms")
        print(f"Pipeline Efficiency: {metrics['efficiency']:.1%}")
        print(f"Bubble Ratio: {metrics['bubble_ratio']:.1%}")
        print(f"Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
        print(f"Peak Memory: {metrics['memory_peak_gb']:.2f} GB")
        
        # Detailed timing breakdown
        if self.step_times:
            print(f"\n⏱️  Detailed Timing Breakdown:")
            for step_type, phases in self.step_times.items():
                print(f"  {step_type.capitalize()}:")
                for phase, times in phases.items():
                    avg_time = sum(times) / len(times) * 1000
                    print(f"    {phase}: {avg_time:.2f} ms (avg over {len(times)} steps)")
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = MicrobatchMetrics()
        self.step_times = {}
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def optimize_schedule(
        self,
        target_memory_gb: Optional[float] = None,
        target_efficiency: Optional[float] = None
    ) -> str:
        """Optimize schedule based on targets"""
        
        current_metrics = self.get_metrics()
        
        # If memory is too high, switch to more memory-efficient schedule
        if (target_memory_gb and 
            current_metrics['memory_peak_gb'] > target_memory_gb):
            
            if self.schedule_type != "gpipe":
                self.schedule_type = "gpipe"
                self.schedule = self._create_schedule()
                self.logger.info("Switched to GPipe schedule for lower memory usage")
                return "gpipe"
        
        # If efficiency is too low, switch to more efficient schedule
        if (target_efficiency and 
            current_metrics['efficiency'] < target_efficiency):
            
            if self.schedule_type != "1f1b":
                self.schedule_type = "1f1b"
                self.schedule = self._create_schedule()
                self.logger.info("Switched to 1F1B schedule for higher efficiency")
                return "1f1b"
        
        return self.schedule_type
    
    def visualize_schedule(self, save_path: Optional[str] = None):
        """Visualize the pipeline schedule"""
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            self.logger.warning("Matplotlib not available for schedule visualization")
            return
        
        schedule = self.get_execution_schedule()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            'forward': 'lightblue',
            'backward': 'lightcoral',
            'optimizer_step': 'lightgreen',
            'communication': 'lightyellow'
        }
        
        y_pos = 0
        step_width = 1.0
        
        for i, step in enumerate(schedule):
            color = colors.get(step.step_type, 'lightgray')
            
            rect = patches.Rectangle(
                (i * step_width, y_pos),
                step_width * 0.9,
                1,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            
            ax.add_patch(rect)
            
            # Add labels
            label = f"{step.step_type[:1].upper()}"
            if step.microbatch_id is not None:
                label += f"{step.microbatch_id}"
            
            ax.text(
                i * step_width + step_width/2,
                y_pos + 0.5,
                label,
                ha='center',
                va='center',
                fontsize=8
            )
        
        ax.set_xlim(0, len(schedule) * step_width)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('Execution Steps')
        ax.set_ylabel('Pipeline Stage')
        ax.set_title(f'{self.schedule.get_schedule_name()} Schedule Visualization')
        
        # Legend
        legend_elements = [
            patches.Patch(color=color, label=step_type.replace('_', ' ').title())
            for step_type, color in colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Schedule visualization saved to {save_path}")
        
        plt.show()


class BatchSplitter:
    """Utility for splitting batches into microbatches"""
    
    @staticmethod
    def split_batch(
        batch: Any,
        microbatch_size: int,
        num_microbatches: int
    ) -> List[Any]:
        """Split a batch into microbatches"""
        
        microbatches = []
        
        if isinstance(batch, dict):
            # Handle dictionary batch (e.g., {"input_ids": tensor, "labels": tensor})
            batch_size = list(batch.values())[0].size(0)
            
            for i in range(num_microbatches):
                start_idx = i * microbatch_size
                end_idx = min((i + 1) * microbatch_size, batch_size)
                
                if start_idx >= batch_size:
                    break
                
                microbatch = {}
                for key, value in batch.items():
                    microbatch[key] = value[start_idx:end_idx]
                
                microbatches.append(microbatch)
                
        elif isinstance(batch, (list, tuple)):
            # Handle list/tuple batch
            batch_size = batch[0].size(0)
            
            for i in range(num_microbatches):
                start_idx = i * microbatch_size
                end_idx = min((i + 1) * microbatch_size, batch_size)
                
                if start_idx >= batch_size:
                    break
                
                microbatch = []
                for tensor in batch:
                    microbatch.append(tensor[start_idx:end_idx])
                
                microbatches.append(tuple(microbatch) if isinstance(batch, tuple) else microbatch)
                
        elif isinstance(batch, torch.Tensor):
            # Handle single tensor batch
            batch_size = batch.size(0)
            
            for i in range(num_microbatches):
                start_idx = i * microbatch_size
                end_idx = min((i + 1) * microbatch_size, batch_size)
                
                if start_idx >= batch_size:
                    break
                
                microbatches.append(batch[start_idx:end_idx])
        
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        return microbatches
    
    @staticmethod
    def calculate_optimal_microbatch_size(
        total_batch_size: int,
        num_pipeline_stages: int,
        memory_limit_gb: float,
        model_memory_gb: float
    ) -> Tuple[int, int]:
        """Calculate optimal microbatch size and count"""
        
        # Available memory for activations
        available_memory_gb = memory_limit_gb - model_memory_gb
        
        # Target: 4-8 microbatches per pipeline stage
        target_microbatches = num_pipeline_stages * 6
        
        # Calculate microbatch size
        microbatch_size = max(1, total_batch_size // target_microbatches)
        num_microbatches = total_batch_size // microbatch_size
        
        # Adjust if microbatches don't fit in memory
        estimated_activation_memory_per_mb = 0.1  # Rough estimate in GB
        
        while (num_microbatches * estimated_activation_memory_per_mb > available_memory_gb and 
               microbatch_size > 1):
            microbatch_size += 1
            num_microbatches = total_batch_size // microbatch_size
        
        return microbatch_size, num_microbatches


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic 1F1B schedule
    scheduler = MicrobatchScheduler(
        num_microbatches=8,
        microbatch_size=2,
        num_pipeline_stages=4,
        current_stage=1,
        schedule_type="1f1b"
    )
    
    schedule = scheduler.get_execution_schedule()
    print(f"Generated {len(schedule)} steps for {scheduler.schedule.get_schedule_name()}")
    
    for i, step in enumerate(schedule):
        print(f"Step {i}: {step.step_type} MB{step.microbatch_id} ({step.phase})")
    
    # Example 2: Compare different schedules
    schedule_types = ["gpipe", "1f1b", "chimera"]
    
    print(f"\n📋 Schedule Comparison:")
    for schedule_type in schedule_types:
        scheduler = MicrobatchScheduler(
            num_microbatches=8,
            microbatch_size=2,
            num_pipeline_stages=4,
            current_stage=1,
            schedule_type=schedule_type
        )
        
        schedule = scheduler.get_execution_schedule()
        forward_steps = len([s for s in schedule if s.step_type == 'forward'])
        backward_steps = len([s for s in schedule if s.step_type == 'backward'])
        
        print(f"  {schedule_type.upper()}: {len(schedule)} total steps "
              f"({forward_steps}F + {backward_steps}B)")
    
    # Example 3: Batch splitting
    print(f"\n✂️  Batch Splitting Example:")
    batch = torch.randn(16, 512, 4096)  # [batch_size, seq_len, hidden_size]
    
    microbatch_size, num_microbatches = BatchSplitter.calculate_optimal_microbatch_size(
        total_batch_size=16,
        num_pipeline_stages=4,
        memory_limit_gb=40,
        model_memory_gb=20
    )
    
    print(f"Optimal config: {num_microbatches} microbatches of size {microbatch_size}")
    
    microbatches = BatchSplitter.split_batch(batch, microbatch_size, num_microbatches)
    print(f"Split batch {batch.shape} into {len(microbatches)} microbatches")
    for i, mb in enumerate(microbatches):
        print(f"  Microbatch {i}: {mb.shape}")