#!/usr/bin/env python3
"""
Import Testing Script for Training Infrastructure

This script tests all imports and helps identify missing files or circular dependencies.
Run this script to validate your import structure.
"""

import sys
import importlib
import traceback
from pathlib import Path

def test_import(module_name, description=""):
    """Test importing a module and catch any errors."""
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

def main():
    """Main testing function."""
    print("🚀 Testing Training Infrastructure Imports\n")
    
    # Test basic package structure
    print("=" * 60)
    print("Testing Basic Package Structure")
    print("=" * 60)
    
    # Test main package
    success, pkg = test_import("llm", "Main package")
    if not success:
        print("❌ Main package import failed. Check if package is installed or in PYTHONPATH.")
        return
    
    # Test submodules
    modules_to_test = [
        ("llm.config", "Configuration classes"),
        ("llm.trainer", "Core training logic"),
        ("llm.logger", "Logging utilities"),
        ("llm.callbacks", "Callback system"),
        ("llm.utils", "Utility functions"),
        ("llm.advanced", "Integration module"),
    ]
    
    for module_name, description in modules_to_test:
        test_import(module_name, description)
    
    print("\n" + "=" * 60)
    print("Testing Models Module")
    print("=" * 60)
    
    # Test models module
    test_import("llm.models", "Models package")
    test_import("llm.models.llama", "LLaMA architectures")
    test_import("llm.models.moe", "Mixture of Experts")
    
    # Test model functions
    model_functions = [
        ("llm.models", "create_llama_7b", "LLaMA 7B creator"),
        ("llm.models", "create_llama_13b", "LLaMA 13B creator"),
        ("llm.models", "create_llama_moe