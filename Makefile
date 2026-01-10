.PHONY: install install-dev test cov lint format typecheck clean

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	uv run pytest

cov:
	uv run pytest --cov=src --cov-report=term-missing

cov-html:
	uv run pytest --cov=src --cov-report=html
	@echo "Open htmlcov/index.html"

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
