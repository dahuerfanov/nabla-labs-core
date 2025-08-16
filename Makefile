.PHONY: help install install-dev test lint format clean docs build dist

help:  ## Show this help message
	@echo "Nabla Labs Core - Development Commands"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e .

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=nabla_labs_core --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 nabla_labs_core/ tests/ examples/
	mypy nabla_labs_core/

format:  ## Format code with Black and sort imports
	black nabla_labs_core/ tests/ examples/
	isort nabla_labs_core/ tests/ examples/

check-format:  ## Check if code is properly formatted
	black --check nabla_labs_core/ tests/ examples/
	isort --check-only nabla_labs_core/ tests/ examples/

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && make html

build:  ## Build distribution packages
	python setup.py sdist bdist_wheel

dist: clean build  ## Clean and build distribution packages

check: format lint test  ## Run all quality checks

pre-commit:  ## Install pre-commit hooks
	pre-commit install

examples:  ## Run example scripts
	python examples/basic_usage.py
	python examples/custom_dataset.py

install-examples:  ## Install example dependencies
	pip install matplotlib jupyter

all: install-dev format lint test  ## Install, format, lint, and test
