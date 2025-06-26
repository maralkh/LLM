# setup.py
from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="training-infra",
    version="1.0.0",
    author="Maral Khosroshahi, Claude Sonnet",
    author_email="maral.khosroshahi@gmail.com",
    description="Production-ready training and inference code for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maralkh/training-infra",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
    install_requires=requirements,
    extras_require={
        "wandb": ["wandb>=0.13.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "transformers": ["transformers>=4.20.0"],
        "vision": ["torchvision>=0.13.0", "pillow>=9.0.0"],
    },
    entry_points={
        "console_scripts": [
            "training-infra=training_infra.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "training_infra": ["configs/*.yaml"],
    },
)
