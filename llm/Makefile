
# Makefile
.PHONY: install install-dev test lint format clean build upload

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=training_infra

lint:
	flake8 training_infra tests
	mypy training_infra

format:
	black training_infra tests examples
	isort training_infra tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Quick start for development
dev-setup: install-dev
	echo "Development environment ready!"
	echo "Run 'make test' to run tests"
	echo "Run 'make lint' to check code quality"
	echo "Run 'make format' to format code"

# ---