.PHONY: test lint format setup clean install

# Development commands
setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run mypy src tests

format:
	uv run black src tests
	uv run ruff check --fix src tests

clean:
	rm -rf build/ dist/ *.egg-info/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	uv clean

install:
	uv sync --no-dev

# TDD cycle helpers
red:
	uv run pytest --tb=short -v

green:
	uv run pytest --tb=short

refactor:
	uv run pytest --tb=short && make lint