.PHONY: install test lint ui up down logs

install:  ## Install backend + frontend dependencies
	pip install -r backend/requirements-dev.txt
	cd frontend && npm ci

test:  ## Run backend + frontend tests (no API keys needed)
	cd backend && pytest --cov=app --cov-report=term-missing
	cd frontend && npm test

ui:  ## Frontend dev server on :5173, proxies /api to a backend on :8000
	cd frontend && npm run dev

lint:
	cd backend && ruff check .

up:  ## Start nginx + 3 backend replicas + redis
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f city-assistant-backend
