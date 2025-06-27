# setup.py
"""
Setup script for Training Infrastructure package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Advanced distributed training framework for LLaMA models"

# Read version from __init__.py
def get_version():
    version_file = os.path.join("training_infra", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split('"')[1]
    return "0.1.0"

# Core dependencies - minimal for Phase 1
core_requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0", 
    "datasets>=2.10.0",
    "accelerate>=0.20.0",
    "peft>=0.4.0",
    "safetensors>=0.3.0",
    "PyYAML>=6.0",
    "numpy>=1.21.0",
    "tqdm>=4.64.0",
    "tensorboard>=2.12.0",
    "wandb>=0.15.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0", 
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

# Advanced features (optional)
advanced_requirements = [
    "flash-attn>=2.0.0",  # Flash Attention
    "deepspeed>=0.9.0",   # DeepSpeed integration
    "fairscale>=0.4.0",   # FairScale for parallelism
    "triton>=2.0.0",      # Triton for custom kernels
]

# All optional dependencies
extras_require = {
    "dev": dev_requirements,
    "advanced": advanced_requirements,
    "all": dev_requirements + advanced_requirements,
}

setup(
    name="training-infra",
    version=get_version(),
    author="Training Infrastructure Team",
    author_email="",
    description="Advanced distributed training framework for LLaMA models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/training-infra",
    
    packages=find_packages(),
    include_package_data=True,
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "training-infra=training_infra.cli:main",
        ],
    },
    
    # Package data
    package_data={
        "training_infra": [
            "configs/**/*.yaml",
            "configs/**/*.yml", 
            "configs/**/*.json",
        ],
    },
    
    # Keywords for discoverability
    keywords=[
        "llama", "transformer", "distributed-training", 
        "deep-learning", "pytorch", "nlp", "ai",
        "machine-learning", "fine-tuning", "rlhf"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-repo/training-infra/issues",
        "Source": "https://github.com/your-repo/training-infra",
        "Documentation": "https://training-infra.readthedocs.io/",
    },
)