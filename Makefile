.PHONY: test test-watch lint format lint-check setup clean install help push workflow-status watch-workflows status red green refactor roadmap

# Help target
help:
	@echo "llm-orc Makefile targets:"
	@echo "  setup           Setup development environment"
	@echo "  test            Run tests"
	@echo "  test-watch      Run tests in watch mode"
	@echo "  lint            Run linting checks (mypy + ruff)"
	@echo "  format          Format code with ruff"
	@echo "  lint-check      Same as lint (compatibility)"
	@echo "  push            Push changes with workflow monitoring"
	@echo "  workflow-status Check CI workflow status"
	@echo "  watch-workflows Watch active workflows"
	@echo "  status          Show git status"
	@echo "  install         Install production dependencies"
	@echo "  clean           Clean build artifacts"
	@echo "  red             TDD: Run tests with short traceback"
	@echo "  green           TDD: Run tests with short traceback"
	@echo "  refactor        TDD: Run tests + lint"
	@echo "  roadmap         Show current development roadmap"

# Development commands
setup:
	uv sync
	@echo "✅ Development environment setup complete"

test:
	uv run pytest

test-watch:
	@echo "Running tests in watch mode..."
	uv run pytest-watch

lint:
	uv run mypy src tests
	uv run ruff check src tests

lint-check: lint

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

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

# Git operations with CI monitoring
push:
	@echo "Pushing changes with workflow monitoring..."
	@git push && gh run list || echo "No workflows found or gh not available"

workflow-status:
	@echo "Checking workflow status..."
	@gh run list --limit 5 || echo "No workflows found or gh not available"

watch-workflows:
	@echo "Watching workflows..."
	@gh run watch || echo "No active workflows or gh not available"

status:
	@echo "Git status:"
	@git status

roadmap:
	@echo "🗺️ Current Roadmap and Strategic Priorities:"
	@gh issue view 9