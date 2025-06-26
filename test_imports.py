#!/usr/bin/env python3
"""
Complete Import Testing Script for Training Infrastructure

This script tests all imports and helps identify missing files or circular dependencies.
Run this script to validate your import structure.
"""

import sys
import importlib
import traceback
from pathlib import Path
import argparse

def test_import(module_name, description=""):
    """Test importing a module and catch any errors."""
    module = importlib.import_module(module_name)
    print(f"✅ {module_name} - {description}")
    return True, module
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True, module
    except ImportError as e:
        print(f"❌ {module_name} - ImportError: {e}")
        return False, None
    except Exception as e:
        print(f"⚠️  {module_name} - Error: {e}")
        return False, None

def test_function_import(module_name, function_name, description=""):
    """Test importing a specific function from a module."""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        print(f"✅ {module_name}.{function_name} - {description}")
        return True, func
    except ImportError as e:
        print(f"❌ {module_name}.{function_name} - ImportError: {e}")
        return False, None
    except AttributeError as e:
        print(f"❌ {module_name}.{function_name} - AttributeError: {e}")
        return False, None
    except Exception as e:
        print(f"⚠️  {module_name}.{function_name} - Error: {e}")
        return False, None

def test_class_import(module_name, class_name, description=""):
    """Test importing a specific class from a module."""
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        print(f"✅ {module_name}.{class_name} - {description}")
        return True, cls
    except ImportError as e:
        print(f"❌ {module_name}.{class_name} - ImportError: {e}")
        return False, None
    except AttributeError as e:
        print(f"❌ {module_name}.{class_name} - AttributeError: {e}")
        return False, None
    except Exception as e:
        print(f"⚠️  {module_name}.{class_name} - Error: {e}")
        return False, None

def test_circular_imports(package_name):
    """Test for circular import issues."""
    print(f"\n{'='*60}")
    print("Testing for Circular Imports")
    print('='*60)
    
    try:
        # Clear the module cache
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith(package_name)]
        for module in modules_to_clear:
            del sys.modules[module]
        
        # Try importing the main package
        main_module = importlib.import_module(package_name)
        print(f"✅ No circular import detected in {package_name}")
        return True
    except Exception as e:
        print(f"❌ Circular import detected: {e}")
        return False

def check_file_structure(package_name):
    """Check if the required file structure exists."""
    print(f"\n{'='*60}")
    print("Checking File Structure")
    print('='*60)
    
    base_path = Path(package_name)
    if not base_path.exists():
        print(f"❌ Package directory '{package_name}' not found")
        return False
    
    required_files = [
        f"{package_name}/__init__.py",
        f"{package_name}/config.py",
        f"{package_name}/trainer.py",
        f"{package_name}/models/__init__.py",
        f"{package_name}/inference/__init__.py",
        f"{package_name}/rlhf/__init__.py",
        f"{package_name}/data/__init__.py",
        f"{package_name}/data/distillation.py",
        f"{package_name}/pipeline/__init__.py",
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - File not found")
    
    print(f"\n📊 File Structure Summary:")
    print(f"   ✅ Existing files: {len(existing_files)}")
    print(f"   ❌ Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n📝 Missing files to create:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    return len(missing_files) == 0

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test imports for training infrastructure')
    parser.add_argument('--package', default='llm', 
                       help='Package name to test (default: llm)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with full error traces')
    parser.add_argument('--structure-only', action='store_true',
                       help='Only check file structure, skip import tests')
    
    args = parser.parse_args()
    package_name = args.package
    
    print(f"🚀 Testing {package_name.title()} Infrastructure Imports\n")
    
    # Check file structure first
    structure_ok = check_file_structure(package_name)
    if not structure_ok:
        print(f"\n⚠️  File structure issues detected. Run the setup script first.")
        if not args.structure_only:
            print("Continuing with import tests anyway...\n")
    
    if args.structure_only:
        return
    
    # Test for circular imports first
    circular_ok = test_circular_imports(package_name)
    
    # Test basic package structure
    print("=" * 60)
    print("Testing Basic Package Structure")
    print("=" * 60)
    
    # Test main package
    success, pkg = test_import(package_name, "Main package")
    if not success:
        print(f"❌ Main package import failed. Check if package is installed or in PYTHONPATH.")
        print("💡 Try: pip install -e . (from the project root)")
        return
    
    # Test submodules
    modules_to_test = [
        (f"{package_name}.config", "Configuration classes"),
        (f"{package_name}.trainer", "Core training logic"),
        (f"{package_name}.logger", "Logging utilities"),
        (f"{package_name}.callbacks", "Callback system"),
        (f"{package_name}.utils", "Utility functions"),
        (f"{package_name}.advanced", "Integration module"),
    ]
    
    for module_name, description in modules_to_test:
        test_import(module_name, description)
    
    print("\n" + "=" * 60)
    print("Testing Models Module")
    print("=" * 60)
    
    # Test models module
    test_import(f"{package_name}.models", "Models package")
    test_import(f"{package_name}.models.llama", "LLaMA architectures")
    test_import(f"{package_name}.models.moe", "Mixture of Experts")
    
    # Test model functions
    model_functions = [
        (f"{package_name}.models", "create_llama_7b", "LLaMA 7B creator"),
        (f"{package_name}.models", "create_llama_13b", "LLaMA 13B creator"),
        (f"{package_name}.models", "create_llama_moe_7b", "LLaMA MoE 7B creator"),
        (f"{package_name}.models", "create_model", "Generic model creator"),
        (f"{package_name}.models", "list_models", "List available models"),
    ]
    
    for module_name, func_name, description in model_functions:
        test_function_import(module_name, func_name, description)
    
    # Test model classes
    model_classes = [
        (f"{package_name}.models", "LlamaModel", "LLaMA model class"),
        (f"{package_name}.models", "LlamaMoEModel", "LLaMA MoE model class"),
        (f"{package_name}.models", "LlamaConfig", "LLaMA configuration"),
    ]
    
    for module_name, class_name, description in model_classes:
        test_class_import(module_name, class_name, description)
    
    print("\n" + "=" * 60)
    print("Testing Inference Module")
    print("=" * 60)
    
    # Test inference module
    test_import(f"{package_name}.inference", "Inference package")
    test_import(f"{package_name}.inference.engine", "Inference engine")
    test_import(f"{package_name}.inference.sampling", "Sampling methods")
    test_import(f"{package_name}.inference.reward_guided", "Reward-guided inference")
    
    # Test inference functions
    inference_functions = [
        (f"{package_name}.inference", "create_sampler", "Sampler creator"),
        (f"{package_name}.inference", "create_reward_guided_engine", "Reward-guided engine creator"),
        (f"{package_name}.inference", "get_sampling_strategy", "Sampling strategy getter"),
    ]
    
    for module_name, func_name, description in inference_functions:
        test_function_import(module_name, func_name, description)
    
    # Test inference classes
    inference_classes = [
        (f"{package_name}.inference", "InferenceEngine", "Main inference engine"),
        (f"{package_name}.inference", "SamplingConfig", "Sampling configuration"),
        (f"{package_name}.inference", "RewardGuidedConfig", "Reward-guided configuration"),
        (f"{package_name}.inference", "TopKSampler", "Top-K sampling"),
        (f"{package_name}.inference", "TopPSampler", "Top-P sampling"),
        (f"{package_name}.inference", "GreedySampler", "Greedy sampling"),
        (f"{package_name}.inference", "TemperatureSampler", "Temperature sampling"),
    ]
    
    for module_name, class_name, description in inference_classes:
        test_class_import(module_name, class_name, description)
    
    print("\n" + "=" * 60)
    print("Testing RLHF Module")
    print("=" * 60)
    
    # Test RLHF module
    test_import(f"{package_name}.rlhf", "RLHF package")
    test_import(f"{package_name}.rlhf.reward_model", "Reward model training")
    test_import(f"{package_name}.rlhf.ppo", "PPO implementation")
    test_import(f"{package_name}.rlhf.dpo", "DPO implementation")
    test_import(f"{package_name}.rlhf.grpo", "GRPO implementation")
    test_import(f"{package_name}.rlhf.prm_orm_training", "PRM/ORM training")
    
    # Test RLHF functions
    rlhf_functions = [
        (f"{package_name}.rlhf", "train_full_rlhf_pipeline", "Complete RLHF pipeline"),
        (f"{package_name}.rlhf", "train_process_reward_model", "Process reward model training"),
        (f"{package_name}.rlhf", "train_outcome_reward_model", "Outcome reward model training"),
        (f"{package_name}.rlhf", "get_rlhf_trainer", "RLHF trainer getter"),
    ]
    
    for module_name, func_name, description in rlhf_functions:
        test_function_import(module_name, func_name, description)
    
    # Test RLHF classes
    rlhf_classes = [
        (f"{package_name}.rlhf", "PPOTrainer", "PPO trainer"),
        (f"{package_name}.rlhf", "DPOTrainer", "DPO trainer"),
        (f"{package_name}.rlhf", "GRPOTrainer", "GRPO trainer"),
        (f"{package_name}.rlhf", "RewardModel", "Reward model"),
        (f"{package_name}.rlhf", "ProcessRewardModel", "Process reward model"),
        (f"{package_name}.rlhf", "OutcomeRewardModel", "Outcome reward model"),
    ]
    
    for module_name, class_name, description in rlhf_classes:
        test_class_import(module_name, class_name, description)
    
    print("\n" + "=" * 60)
    print("Testing Data Module (includes Distillation)")
    print("=" * 60)
    
    # Test data module
    test_import(f"{package_name}.data", "Data package")
    test_import(f"{package_name}.data.synthetic", "Synthetic data generation")
    test_import(f"{package_name}.data.distillation", "Knowledge distillation")
    
    # Test data functions
    data_functions = [
        (f"{package_name}.data", "create_synthetic_data_generator", "Synthetic data generator creator"),
        (f"{package_name}.data", "get_data_generator", "Data generator getter"),
        (f"{package_name}.data", "list_data_generators", "List available generators"),
        (f"{package_name}.data", "compress_model_with_distillation", "Model compression"),
        (f"{package_name}.data", "evaluate_distillation_quality", "Distillation evaluation"),
        (f"{package_name}.data", "get_distillation_trainer", "Distillation trainer getter"),
        (f"{package_name}.data", "list_distillation_methods", "List distillation methods"),
    ]
    
    for module_name, func_name, description in data_functions:
        test_function_import(module_name, func_name, description)
    
    # Test data classes (including distillation)
    data_classes = [
        (f"{package_name}.data", "SyntheticDataGenerator", "General synthetic data generator"),
        (f"{package_name}.data", "SyntheticDataConfig", "Synthetic data configuration"),
        (f"{package_name}.data", "MathDataGenerator", "Math data generator"),
        (f"{package_name}.data", "CodeDataGenerator", "Code data generator"),
        (f"{package_name}.data", "ConstitutionalAIGenerator", "Constitutional AI generator"),
        (f"{package_name}.data", "InstructionDataGenerator", "Instruction data generator"),
        (f"{package_name}.data", "ConversationDataGenerator", "Conversation data generator"),
        (f"{package_name}.data", "DistillationTrainer", "Base distillation trainer"),
        (f"{package_name}.data", "DistillationConfig", "Distillation configuration"),
        (f"{package_name}.data", "ProgressiveDistillationTrainer", "Progressive distillation"),
        (f"{package_name}.data", "ResponseDistillationTrainer", "Response distillation"),
        (f"{package_name}.data", "FeatureDistillationTrainer", "Feature distillation"),
        (f"{package_name}.data", "AttentionDistillationTrainer", "Attention distillation"),
    ]
    
    for module_name, class_name, description in data_classes:
        test_class_import(module_name, class_name, description)
    
    print("\n" + "=" * 60)
    print("Testing Pipeline Module")
    print("=" * 60)
    
    # Test pipeline module
    test_import(f"{package_name}.pipeline", "Pipeline package")
    test_import(f"{package_name}.pipeline.synthetic_distillation", "Synthetic distillation pipelines")
    
    # Test pipeline functions
    pipeline_functions = [
        (f"{package_name}.pipeline", "create_pipeline", "Pipeline creator"),
        (f"{package_name}.pipeline", "list_pipelines", "List available pipelines"),
    ]
    
    for module_name, func_name, description in pipeline_functions:
        test_function_import(module_name, func_name, description)
    
    # Test pipeline classes
    pipeline_classes = [
        (f"{package_name}.pipeline", "SyntheticDistillationPipeline", "Synthetic distillation pipeline"),
        (f"{package_name}.pipeline", "DomainAdaptiveDistillationPipeline", "Domain adaptive pipeline"),
        (f"{package_name}.pipeline", "MultiTeacherDistillationPipeline", "Multi-teacher pipeline"),
        (f"{package_name}.pipeline", "ProductionCompressionPipeline", "Production compression pipeline"),
    ]
    
    for module_name, class_name, description in pipeline_classes:
        test_class_import(module_name, class_name, description)
    
    print("\n" + "=" * 60)
    print("Testing Main Package Exports")
    print("=" * 60)
    
    # Test main package exports
    main_exports = [
        (package_name, "TrainingConfig", "Training configuration"),
        (package_name, "Trainer", "Main trainer"),
        (package_name, "AdvancedLlamaTrainer", "Advanced LLaMA trainer"),
        (package_name, "setup_logging", "Logging setup"),
        (package_name, "get_logger", "Logger getter"),
    ]
    
    for module_name, export_name, description in main_exports:
        test_function_import(module_name, export_name, description)
    
    print("\n" + "=" * 60)
    print("Testing Registry Functions")
    print("=" * 60)
    
    # Test registry functions
    try:
        module = importlib.import_module(f"{package_name}.models")
        if hasattr(module, 'list_models'):
            models = module.list_models()
            print(f"✅ Available models: {', '.join(models) if models else 'None (not implemented)'}")
        else:
            print(f"❌ list_models() function not found in {package_name}.models")
    except Exception as e:
        print(f"❌ list_models() failed: {e}")
    
    try:
        module = importlib.import_module(f"{package_name}.inference")
        if hasattr(module, 'SAMPLING_STRATEGIES'):
            strategies = list(module.SAMPLING_STRATEGIES.keys())
            print(f"✅ Available sampling strategies: {', '.join(strategies) if strategies else 'None (not implemented)'}")
        else:
            print(f"❌ SAMPLING_STRATEGIES registry not found in {package_name}.inference")
    except Exception as e:
        print(f"❌ Sampling strategies registry failed: {e}")
    
    try:
        module = importlib.import_module(f"{package_name}.rlhf")
        if hasattr(module, 'RLHF_METHODS'):
            methods = list(module.RLHF_METHODS.keys())
            print(f"✅ Available RLHF methods: {', '.join(methods) if methods else 'None (not implemented)'}")
        else:
            print(f"❌ RLHF_METHODS registry not found in {package_name}.rlhf")
    except Exception as e:
        print(f"❌ RLHF methods registry failed: {e}")
    
    try:
        module = importlib.import_module(f"{package_name}.data")
        if hasattr(module, 'DATA_GENERATORS'):
            generators = list(module.DATA_GENERATORS.keys())
            print(f"✅ Available data generators: {', '.join(generators) if generators else 'None (not implemented)'}")
        else:
            print(f"❌ DATA_GENERATORS registry not found in {package_name}.data")
    except Exception as e:
        print(f"❌ Data generators registry failed: {e}")
    
    try:
        module = importlib.import_module(f"{package_name}.data")
        if hasattr(module, 'DISTILLATION_METHODS'):
            methods = list(module.DISTILLATION_METHODS.keys())
            print(f"✅ Available distillation methods: {', '.join(methods) if methods else 'None (not implemented)'}")
        else:
            print(f"❌ DISTILLATION_METHODS registry not found in {package_name}.data")
    except Exception as e:
        print(f"❌ Distillation methods registry failed: {e}")
    
    print("\n" + "=" * 60)
    print("Testing Integration Patterns")
    print("=" * 60)
    
    # Test common integration patterns
    integration_tests = [
        (f"from {package_name} import TrainingConfig, Trainer", "Basic training imports"),
        (f"from {package_name}.models import create_model", "Model creation import"),
        (f"from {package_name}.inference import create_sampler, SamplingConfig", "Inference imports"),
        (f"from {package_name}.rlhf import train_full_rlhf_pipeline", "RLHF pipeline import"),
        (f"from {package_name}.data import create_synthetic_data_generator", "Data generation import"),
        (f"from {package_name}.data import compress_model_with_distillation", "Distillation import"),
        (f"from {package_name}.pipeline import create_pipeline", "Pipeline import"),
    ]
    
    for import_statement, description in integration_tests:
        try:
            exec(import_statement)
            print(f"✅ {description}")
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
            if args.verbose:
                print(f"   Full traceback: {traceback.format_exc()}")
    
    print("\n" + "=" * 60)
    print("Testing Version and Metadata")
    print("=" * 60)
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Not set')
        author = getattr(module, '__author__', 'Not set')
        print(f"✅ Package version: {version}")
        print(f"✅ Package author: {author}")
        
        if hasattr(module, '__all__'):
            all_exports = module.__all__
            print(f"✅ __all__ exports: {len(all_exports)} items")
            if args.verbose:
                print(f"   Exports: {', '.join(all_exports)}")
        else:
            print(f"⚠️  __all__ not defined (recommended for public API)")
    except Exception as e:
        print(f"❌ Version/metadata check failed: {e}")
    
    print("\n" + "=" * 60)
    print("Summary and Recommendations")
    print("=" * 60)
    
    print(f"""
📋 Import Testing Complete for {package_name}!

🔧 Common Issues and Solutions:

1. Missing __init__.py files:
   - Ensure every directory has an __init__.py file
   - Check that __init__.py files have proper imports

2. Circular imports:
   - Use function-level imports when needed
   - Consider moving shared code to a separate module
   - Use TYPE_CHECKING for type hints

3. Missing dependencies:
   - Install required packages (torch, transformers, etc.)
   - Check requirements.txt

4. PYTHONPATH issues:
   - Ensure your package is installed: pip install -e .
   - Or add to PYTHONPATH: export PYTHONPATH=$PYTHONPATH:/path/to/your/package

5. Missing implementations:
   - Create placeholder implementations for missing classes/functions
   - Use NotImplementedError for methods to be implemented later

🚀 Next Steps:
1. Fix any failed imports shown above
2. Implement missing classes and functions
3. Add proper error handling and documentation
4. Create unit tests for each module
5. Set up proper package installation with setup.py

💡 Tips:
- Start with core modules (config, trainer, models)
- Implement basic functionality first, then add advanced features
- Use type hints for better IDE support
- Follow PEP 8 naming conventions

🧪 Testing Commands:
- Test specific package: python test_imports.py --package your_package_name
- Check structure only: python test_imports.py --structure-only
- Verbose output: python test_imports.py --verbose
""")


if __name__ == "__main__":
    main()