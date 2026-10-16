.PHONY: install test lint up down logs

install:  ## Install backend + dev dependencies
	pip install -r backend/requirements-dev.txt

test:  ## Run the test suite with coverage (no API keys needed)
	cd backend && pytest --cov=app --cov-report=term-missing

lint:
	cd backend && ruff check .

up:  ## Start nginx + 3 backend replicas + redis
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f city-assistant-backend
