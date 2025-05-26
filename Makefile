.PHONY: help install install-dev test lint format security clean setup-env train predict serve docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code"
	@echo "  security     Run security checks"
	@echo "  clean        Clean up temporary files"
	@echo "  setup-env    Setup development environment"
	@echo "  train        Train models"
	@echo "  predict      Run predictions"
	@echo "  serve        Start API server"
	@echo "  docs         Generate documentation"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Environment setup
setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  Unix/MacOS: source venv/bin/activate"

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Security
security:
	bandit -r src/
	safety check

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# ML Operations
train:
	python -m src.training.train

predict:
	python -m src.inference.predict

evaluate:
	python -m src.evaluation.evaluate

# API
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

docs-serve:
	python -m http.server 8080 --directory docs/_build/html

# Data processing
process-data:
	python -m src.data.process

# MLflow
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Jupyter
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Docker (if using containers)
docker-build:
	docker build -t kb-nova-pipeline .

docker-run:
	docker run -p 8000:8000 kb-nova-pipeline

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files 