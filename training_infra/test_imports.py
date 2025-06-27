#!/usr/bin/env python3
"""
Import Testing Script for Training Infrastructure

This script tests all imports and helps identify missing files or dependencies.
Run this script to validate your package structure.

Usage:
    python test_imports.py
    python test_imports.py --verbose
    python test_imports.py --check-deps
"""

import sys
import importlib
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

def test_import(module_name: str, description: str = "") -> Tuple[bool, Optional[object]]:
    """Test importing a module and catch any errors."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name:<30} - {description}")
        return True, module
    except ImportError as e:
        print(f"❌ {module_name:<30} - ImportError: {e}")
        return False, None
    except Exception as e:
        print(f"⚠️  {module_name:<30} - Error: {e}")
        return False, None

def test_function_import(module_name: str, function_name: str, description: str = "") -> Tuple[bool, Optional[object]]:
    """Test importing a specific function from a module."""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        print(f"✅ {module_name}.{function_name:<20} - {description}")
        return True, func
    except ImportError as e:
        print(f"❌ {module_name}.{function_name:<20} - ImportError: {e}")
        return False, None
    except AttributeError as e:
        print(f"❌ {module_name}.{function_name:<20} - AttributeError: {e}")
        return False, None
    except Exception as e:
        print(f"⚠️  {module_name}.{function_name:<20} - Error: {e}")
        return False, None

def check_dependencies() -> bool:
    """Check if core dependencies are installed."""
    print(f"\n{'='*60}")
    print("Checking Core Dependencies")
    print('='*60)
    
    core_deps = [
        ("torch", "PyTorch for deep learning"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("accelerate", "HuggingFace Accelerate"),
        ("peft", "Parameter Efficient Fine-Tuning"),
        ("yaml", "YAML configuration support"),
        ("numpy", "Numerical computing"),
        ("tqdm", "Progress bars"),
    ]
    
    all_good = True
    for package, description in core_deps:
        success, _ = test_import(package, description)
        if not success:
            all_good = False
    
    if all_good:
        print(f"\n🎉 All core dependencies are installed!")
    else:
        print(f"\n⚠️  Some dependencies are missing. Install with:")
        print(f"   pip install -r requirements.txt")
    
    return all_good

def check_file_structure() -> bool:
    """Check if the required file structure exists."""
    print(f"\n{'='*60}")
    print("Checking File Structure")
    print('='*60)
    
    base_path = Path("training_infra")
    required_files = [
        "training_infra/__init__.py",
        "setup.py",
        "requirements.txt",
        "test_imports.py",
    ]
    
    # Will be added in later phases
    future_files = [
        "training_infra/config/__init__.py",
        "training_infra/models/__init__.py", 
        "training_infra/training/__init__.py",
        "training_infra/utils/__init__.py",
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
    
    print(f"\n📋 Future files (will be added in later phases):")
    for file_path in future_files:
        status = "✅" if Path(file_path).exists() else "⏳"
        print(f"{status} {file_path}")
    
    print(f"\n📊 File Structure Summary:")
    print(f"   ✅ Required files: {len(existing_files)}/{len(required_files)}")
    print(f"   ❌ Missing files: {len(missing_files)}")
    
    return len(missing_files) == 0

def test_basic_functionality() -> bool:
    """Test basic functionality of the package."""
    print(f"\n{'='*60}")
    print("Testing Basic Package Functionality")
    print('='*60)
    
    # Test main package import
    success, pkg = test_import("training_infra", "Main package")
    if not success:
        return False
    
    # Test basic functions
    functions_to_test = [
        ("quick_start", "Quick start function"),
        ("load_model", "Model loading function"),
        ("check_dependencies", "Dependency checker"),
        ("info", "Package information"),
    ]
    
    all_good = True
    for func_name, description in functions_to_test:
        success, func = test_function_import("training_infra", func_name, description)
        if not success:
            all_good = False
    
    # Test basic functionality
    if all_good:
        try:
            import training_infra
            print(f"\n🧪 Testing basic functionality:")
            
            # Test info function
            print(f"📋 Package info:")
            training_infra.info()
            
            # Test dependency check
            print(f"\n🔍 Dependency check:")
            deps_ok = training_infra.check_dependencies()
            
            # Test quick_start (should warn about missing components)
            print(f"\n🚀 Testing quick_start (should show warning):")
            result = training_infra.quick_start("tiny_llama3_150m", "standard", 1)
            print(f"   Result: {result} (expected None with warning)")
            
        except Exception as e:
            print(f"❌ Error testing functionality: {e}")
            all_good = False
    
    return all_good

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test imports for training infrastructure')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--check-deps', action='store_true',
                       help='Only check dependencies')
    parser.add_argument('--check-structure', action='store_true',
                       help='Only check file structure')
    
    args = parser.parse_args()
    
    print(f"🚀 Testing Training Infrastructure Package\n")
    
    if args.check_deps:
        check_dependencies()
        return
    
    if args.check_structure:
        check_file_structure()
        return
    
    # Run all tests
    print(f"Phase 1.1: Basic Project Setup Test")
    print(f"Expected: Package imports, basic functions available")
    print(f"Note: Advanced features will show warnings (this is expected)")
    
    # Test dependency installation
    deps_ok = check_dependencies()
    
    # Test file structure  
    structure_ok = check_file_structure()
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE 1.1 TEST SUMMARY")
    print('='*60)
    
    tests = [
        ("Dependencies", deps_ok),
        ("File Structure", structure_ok), 
        ("Basic Functionality", func_ok),
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 PHASE 1.1 COMPLETE!")
        print(f"✅ Package structure is ready")
        print(f"✅ Basic imports work")
        print(f"✅ Ready to move to Phase 1.2 (Configuration)")
    else:
        print(f"\n⚠️  PHASE 1.1 NEEDS WORK")
        print(f"💡 Fix the failed tests before proceeding")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)