#!/bin/bash

# Test Coverage Gate Hook
# Ensures adequate test coverage before commits

# Exit early if not in Python project
if ! command -v uv &> /dev/null || [ ! -f "pyproject.toml" ]; then
    exit 0
fi

# Check if we're in a commit context (git add was used)
if ! git diff --cached --quiet 2>/dev/null; then
    echo "🧪 Checking test coverage before commit..."
    
    # Run tests with coverage (using project's 95% requirement)
    COVERAGE_RESULT=$(uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95 --quiet 2>&1 || echo "COVERAGE_FAILED")
    
    if echo "$COVERAGE_RESULT" | grep -q "COVERAGE_FAILED\|FAILED.*coverage"; then
        echo "❌ Test coverage below 95% threshold"
        echo ""
        echo "Coverage report:"
        # Show just the summary line
        echo "$COVERAGE_RESULT" | grep -A 5 -B 5 "TOTAL\|coverage" || echo "$COVERAGE_RESULT" | tail -10
        echo ""
        echo "Options:"
        echo "1. Add tests to improve coverage (recommended)"
        echo "2. View detailed coverage report (v)?"
        # Check if stdin is a terminal (interactive mode)
        if [ -t 0 ]; then
            echo "3. Commit anyway (not recommended) (c)?"
            echo "4. Abort commit (any other key)"
            read -r -n 1 response
            echo ""

            case $response in
            [Vv])
                echo "📊 Detailed coverage report:"
                uv run pytest --cov=src --cov-report=term-missing --quiet 2>/dev/null | grep -A 50 "Name\|TOTAL" || echo "Run 'make test' for full report"
                echo ""
                echo "Add tests for uncovered lines and try again."
                exit 1
                ;;
            [Cc])
                echo "⚠️  Proceeding with insufficient coverage (technical debt created)"
                ;;
            *)
                echo "✋ Commit blocked - please add tests to reach 95% coverage"
                echo "Run 'make test' to see detailed coverage report"
                exit 1
                ;;
            esac
        else
            # Non-interactive mode: provide information without blocking
            echo "⚠️  Non-interactive mode: Low test coverage detected"
            echo "   Current coverage: $COVERAGE%"
            echo "   Consider adding more tests"
        fi
    else
        # Check if there are any test failures
        if echo "$COVERAGE_RESULT" | grep -q "FAILED\|ERROR"; then
            echo "❌ Tests are failing"
            echo ""
            echo "Options:"
            # Check if stdin is a terminal (interactive mode)
            if [ -t 0 ]; then
                echo "1. View test failures (v)?"
                echo "2. Fix tests first (any other key)"
                read -r -n 1 response
                echo ""

                case $response in
                [Vv])
                    echo "🔍 Test failure details:"
                    uv run pytest -v --tb=short 2>/dev/null || echo "Run 'make test' for detailed output"
                    ;;
                esac
            else
                # Non-interactive mode: report failures
                echo "⚠️  Non-interactive mode: Tests are failing"
                echo "   Run 'make test' to see failures"
            fi
            
            echo "✋ Commit blocked - please fix failing tests first"
            exit 1
        else
            echo "✅ Test coverage meets requirements (≥95%)"
        fi
    fi
else
    # Not in a commit context, just run a quick check
    CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null | grep '^src/.*\.py$' || echo "")
    
    if [ -n "$CHANGED_FILES" ]; then
        echo "💡 Tip: Run 'make test' to check coverage for your changes"
    fi
fi

exit 0