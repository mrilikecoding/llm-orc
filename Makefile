.PHONY: test lint format setup clean install

# Development commands
setup:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests
	mypy src tests

format:
	black src tests
	ruff --fix src tests

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

install:
	pip install -e .

# TDD cycle helpers
red:
	pytest --tb=short -v

green:
	pytest --tb=short

refactor:
	pytest --tb=short && make lint