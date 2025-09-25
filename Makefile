.PHONY: test test-watch lint lint-fix format lint-check security dead-code setup clean install help push workflow-status watch-workflows status red green refactor pre-commit roadmap validate-contracts-core validate-contracts-examples validate-contracts-community validate-contracts-all

# Help target
help:
	@echo "llm-orc Makefile targets:"
	@echo "  setup           Setup development environment"
	@echo "  test            Run tests"
	@echo "  test-watch      Run tests in watch mode"
	@echo "  lint            Run linting checks (mypy + ruff + format check + complexity + security + dead code)"
	@echo "  lint-fix        Run linting checks and auto-fix issues"
	@echo "  format          Format code with ruff"
	@echo "  lint-check      Same as lint (compatibility)"
	@echo "  security        Run security analysis with bandit"
	@echo "  dead-code       Run dead code analysis with vulture"
	@echo "  pre-commit      Run all CI checks locally before commit"
	@echo "  push            Push changes with pre-commit checks and workflow monitoring"
	@echo "  workflow-status Check CI workflow status"
	@echo "  watch-workflows Watch active workflows"
	@echo "  status          Show git status"
	@echo "  install         Install production dependencies"
	@echo "  clean           Clean build artifacts"
	@echo "  red             TDD: Run tests with short traceback"
	@echo "  green           TDD: Run tests with short traceback"
	@echo "  refactor        TDD: Run tests + lint"
	@echo "  roadmap         Show current development roadmap"
	@echo "  validate-contracts-core       Validate core primitive contracts"
	@echo "  validate-contracts-examples   Validate example script contracts"
	@echo "  validate-contracts-community  Validate community script contracts"
	@echo "  validate-contracts-all        Validate all script contracts"

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
	uv run ruff format --check src tests
	uv run complexipy --max-complexity-allowed 15 src
	uv run bandit -r src/ --quiet --severity-level medium
	uv run vulture src/ --min-confidence 80

lint-fix:
	uv run mypy src tests
	uv run ruff check --fix src tests
	uv run ruff format src tests

lint-check: lint

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

security:
	@echo "Running security analysis with bandit..."
	uv run bandit -r src/ --quiet --severity-level medium

dead-code:
	@echo "Running dead code analysis with vulture..."
	uv run vulture src/ --min-confidence 80

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

# Pre-commit checks (runs all CI checks locally)
pre-commit:
	@echo "Running pre-commit checks..."
	make test
	make lint
	make validate-contracts-core
	@echo "✅ All pre-commit checks passed"

# Git operations with CI monitoring
push:
	@echo "Running pre-commit checks before push..."
	@make pre-commit
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

# ADR-003 Contract Validation targets
validate-contracts-core:
	@echo "Validating core primitive contracts..."
	uv run python -m llm_orc.testing.contract_validator --directory .llm-orc/scripts/primitives --level core --verbose

validate-contracts-examples:
	@echo "Validating example script contracts..."
	uv run python -m llm_orc.testing.contract_validator --directory .llm-orc/scripts/examples --level examples --verbose

validate-contracts-community:
	@echo "Validating community script contracts..."
	uv run python -m llm_orc.testing.contract_validator --directory .llm-orc/scripts/community --level community --verbose

validate-contracts-all:
	@echo "Validating all script contracts..."
	@make validate-contracts-core
	@make validate-contracts-examples
	@make validate-contracts-community