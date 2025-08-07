# LLM Orchestra Development Commands

## Essential Development Workflow

### Setup and Environment
```bash
make setup          # Setup development environment with uv sync
make install         # Install production dependencies only
make clean           # Clean build artifacts and caches
```

### Testing Commands
```bash
make test            # Run full test suite
make test-watch      # Run tests in watch mode for development
make red             # TDD Red phase: Run tests with short traceback
make green           # TDD Green phase: Run tests with short traceback  
make refactor        # TDD Refactor phase: Run tests + linting
```

### Code Quality
```bash
make lint            # Full linting suite (mypy + ruff + complexity + security)
make lint-fix        # Auto-fix linting issues where possible
make format          # Format code with ruff
make lint-check      # Same as lint (compatibility alias)
```

### Advanced Quality Checks
```bash
make security        # Run security analysis with bandit
make dead-code       # Run dead code analysis with vulture
```

## Git Workflow Commands

### Pre-commit and Push
```bash
make pre-commit      # Run all CI checks locally before commit
make push            # Pre-commit checks + push + workflow monitoring
```

### Git and CI Monitoring
```bash
make status          # Show git status
make workflow-status # Check GitHub Actions workflow status
make watch-workflows # Watch active GitHub workflows
```

## TDD Development Cycle

### Red → Green → Refactor
```bash
# 1. Red Phase: Write failing test
make red             # Run tests, expect failure

# 2. Green Phase: Make test pass
make green           # Run tests, should pass now

# 3. Refactor Phase: Improve code structure
make refactor        # Run tests + linting to verify no regressions
```

### TDD Best Practices
- Use `make red` to verify tests fail initially
- Use `make green` for quick feedback during implementation
- Use `make refactor` to ensure code quality after changes
- Always run full `make test` before commits

## Quality Assurance Pipeline

### Full Quality Check
```bash
make pre-commit      # Complete validation before any commit
```

This runs:
1. Full test suite (`make test`)
2. Complete linting (`make lint`)
   - Type checking (mypy)
   - Code linting (ruff check)
   - Format checking (ruff format --check)
   - Complexity analysis (complexipy)
   - Security analysis (bandit)
   - Dead code detection (vulture)

### Individual Quality Tools

#### Type Checking
```bash
uv run mypy src tests        # Type checking with mypy strict mode
```

#### Code Linting and Formatting
```bash
uv run ruff check src tests  # Linting checks
uv run ruff format src tests # Code formatting
uv run ruff check --fix src tests  # Auto-fix issues
```

#### Complexity Analysis
```bash
uv run complexipy --max-complexity-allowed 15 src  # Complexity limits
```

#### Security Analysis
```bash
uv run bandit -r src/ --quiet --severity-level medium  # Security scanning
```

#### Dead Code Detection
```bash
uv run vulture src/ --min-confidence 80  # Unused code detection
```

## Development Environment

### Package Management
```bash
uv sync              # Install all dependencies (including dev)
uv sync --dev        # Explicit dev dependencies
uv sync --no-dev     # Production dependencies only
uv clean             # Clean package cache
```

### Virtual Environment
- Automatically managed by `uv sync`
- Located at `.venv/` in project root
- All `uv run` commands use this environment

## GitHub Integration

### Workflow Monitoring
```bash
gh run list          # List recent workflow runs
gh run list --limit 5  # Last 5 runs
gh run watch         # Watch active workflows
gh run view <run-id> # View specific workflow run
```

### Issue and Project Management
```bash
make roadmap         # View current strategic roadmap (Issue #9)
gh issue list        # List open issues
gh issue view <number>  # View specific issue
```

## Commit and Push Workflow

### Safe Push Process
```bash
# 1. Check current status
make status

# 2. Run all quality checks
make pre-commit

# 3. Commit changes
git add .
git commit -m "feat: implement feature (closes #123)"

# 4. Push with monitoring
make push
```

### Post-Push Monitoring
```bash
# Watch workflows after push
make watch-workflows

# Check workflow status
make workflow-status
```

## Performance and Optimization

### Test Performance
```bash
# Run performance tests specifically
uv run pytest tests/performance/ -v

# Benchmark parallel execution
uv run pytest tests/performance/test_parallel_execution.py --benchmark-only
```

### Development Performance
```bash
# Fast feedback loop
make test-watch      # Automatic test re-running

# Quick TDD cycle
make red && make green && make refactor
```

## Configuration and Setup

### Initial Project Setup
```bash
# 1. Clone repository
git clone <repo-url>
cd llm-orc

# 2. Setup development environment
make setup

# 3. Verify installation
uv run llm-orc --version

# 4. Run tests to verify setup
make test
```

### Troubleshooting Setup
```bash
# Clean and restart
make clean
make setup

# Check environment
uv run python --version
uv run which python

# Verify dependencies
uv run pip list
```

## Integration with IDEs

### VS Code Integration
- Use Python interpreter from `.venv/bin/python`
- Configure test discovery for pytest
- Set up linting with ruff and mypy
- Use `make test-watch` in integrated terminal

### Command Line Development
```bash
# Fast development cycle
make test-watch &     # Background test watching
make green           # Quick test runs
make format          # Code formatting
```

## Continuous Integration

### Local CI Simulation
```bash
# Run exactly what CI runs
make pre-commit

# Individual CI steps
make test            # Test suite
make lint            # Code quality
make security        # Security checks  
make dead-code       # Dead code analysis
```

### CI Debugging
```bash
# Check failing workflows
make workflow-status

# Watch specific workflow
gh run watch <run-id>

# View workflow logs
gh run view <run-id> --log
```