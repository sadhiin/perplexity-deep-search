.PHONY: help install dev test lint format clean docker-build docker-up docker-down deploy

help: ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

dev: ## Install development dependencies
	uv sync --extra dev

# test: ## Run tests
# 	uv run python scripts/run_tests.py

lint: ## Run linting
	uv run black --check app tests
	uv run isort --check app tests
	uv run flake8 app tests
	uv run mypy app

format: ## Format code
	uv run black app tests
	uv run isort app tests

clean: ## Clean cache and build files
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	docker compose down
	docker compose prune -f
# start-dev: ## Start development environment
# 	uv run python scripts/start_dev.py

start-worker: ## Start worker process
	uv run python scripts/start_worker.py

docker-build: ## Build Docker images
	docker build -f docker/Dockerfile.backend -t perplexity-deep-search:v1 .

docker-up: ## Start Docker services
	docker-compose -f docker-compose.yml up --build

docker-up-prod: ## Start Docker services
	docker-compose -f docker-compose.yml up -d
	
docker-down: ## Stop Docker services
	docker-compose -f docker-compose.yml down -v

docker-dev: ## Start development Docker services
	docker-compose -f docker-compose.yml up

# deploy: ## Deploy to production
# 	./scripts/deploy.sh

logs: ## Show application logs
	docker-compose -f docker-compose.yml logs -f app

migrate: ## Run database migrations
	uv run alembic upgrade head

migrate-create: ## Create new migration
	uv run alembic revision --autogenerate -m "$(name)"