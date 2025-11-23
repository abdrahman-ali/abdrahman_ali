.PHONY: help install install-dev test clean lint format docker-build docker-up jupyter serve mlflow

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt

install-minimal: ## Install minimal dependencies
	pip install -r requirements-minimal.txt

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all Docker services
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down

jupyter: ## Start Jupyter Lab
	docker-compose up jupyter

serve: ## Start model serving API
	python main.py --mode serve

serve-docker: ## Start model serving API with Docker
	docker-compose up api

mlflow: ## Start MLflow tracking UI
	mlflow ui --port 5000

mlflow-docker: ## Start MLflow tracking UI with Docker
	docker-compose up mlflow

train: ## Run training pipeline
	python main.py --mode train

eda: ## Run exploratory data analysis
	python main.py --mode eda

full: ## Run full pipeline
	python main.py --mode full

setup: ## Initial setup (install + create directories)
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/interim data/external
	mkdir -p models/saved_models models/checkpoints
	mkdir -p logs
	mkdir -p docs/reports/figures
	@echo "Setup complete! Add your data to data/raw/ and run 'make full'"
