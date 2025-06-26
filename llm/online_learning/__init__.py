"""
Online learning module for continuous model training and adaptation.
"""

try:
    from .enhanced_online_training import (
        EnhancedOnlineConfig,
        EnhancedDataProcessor,
        AdaptiveParameterManager,
        EnhancedFileBasedTrainingSystem,
        test_enhanced_online_system,
        run_production_enhanced_system,

    )
except ImportError:
    print("Warning: enhanced_online_training not found")

try:
    from .file_based_training_inference import (
        FileBasedConfig,
        DataFileProcessor,
        FileWatcher,
        FileBasedTrainingSystem,
        create_sample_data_files,
        create_dummy_models,
        test_file_based_system,
        run_production_file_system,
    )
except ImportError:
    print("Warning: file_based_training_inference not found")

try:
    from .multi_model_learning import (
        ManifoldLearningConfig,
        DataPoint,
        InputAnalysis,
        DummyTokenizer,
        DummyModel,
        ManifoldLearner,
        InputClassifier,
        MultiModelInferenceEngine,
        create_specialized_models,
        demonstrate_manifold_learning,
        demonstrate_multi_model_inference,
        compare_manifold_vs_traditional,

    )
except ImportError:
    print("Warning: multi_model_learning not found")

try:
    from .online_training import (
        OnlineConfig,
        ExperienceBuffer,
        OnlineTrainingDataset,
        OnlineTrainer,
        OnlineInferenceEngine,
        OnlineSystem,
        DummyTokenizer,
        create_dummy_models,
        test_online_system,
        run_production_server,

    )
except ImportError:
    print("Warning: online_training not found")

__all__ = [
   # Enhanced online training
   "EnhancedOnlineConfig",
   "EnhancedDataProcessor", 
   "AdaptiveParameterManager",
   "EnhancedFileBasedTrainingSystem",
   "test_enhanced_online_system",
   "run_production_enhanced_system",
   
   # File-based training and inference
   "FileBasedConfig",
   "DataFileProcessor",
   "FileWatcher",
   "FileBasedTrainingSystem", 
   "create_sample_data_files",
   "create_dummy_models",
   "test_file_based_system",
   "run_production_file_system",
   
   # Multi-model learning
   "ManifoldLearningConfig",
   "DataPoint",
   "InputAnalysis", 
   "DummyTokenizer",
   "DummyModel",
   "ManifoldLearner",
   "InputClassifier",
   "MultiModelInferenceEngine",
   "create_specialized_models",
   "demonstrate_manifold_learning", 
   "demonstrate_multi_model_inference",
   "compare_manifold_vs_traditional",
   
   # Online training
   "OnlineConfig",
   "ExperienceBuffer",
   "OnlineTrainingDataset",
   "OnlineTrainer", 
   "OnlineInferenceEngine",
   "OnlineSystem",
   "test_online_system",
   "run_production_server",
]