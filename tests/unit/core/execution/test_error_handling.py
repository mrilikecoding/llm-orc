"""Unit tests for error handling and exception chaining.

This module contains tests for proper error handling patterns used by script agents,
ensuring proper exception chaining as outlined in ADR-003.

Migrated from: tests/test_issue_24_units.py::test_error_handling
Related BDD: tests/bdd/features/issue-24-script-agents.feature (error handling)
Related ADR: docs/adrs/003-exception-chaining-patterns.md
"""

import pytest
from pydantic import ValidationError

from llm_orc.schemas.script_agent import ScriptAgentOutput


class TestScriptErrorHandling:
    """Unit tests for script agent error handling and exception chaining."""

    def test_error_handling(self) -> None:
        """Test error handling and exception chaining patterns.

        Originally from BDD scenario: Script execution handles errors with chaining
        Tests proper exception chaining according to ADR-003.
        """

        # Test proper exception chaining (ADR-003)
        class ScriptExecutionError(Exception):
            """Custom exception for script execution failures."""

            pass

        def execute_script(script_content: str) -> str:
            """Simulate script execution with error handling."""
            try:
                # Simulate script parsing error
                if "syntax_error" in script_content:
                    raise SyntaxError("Invalid Python syntax at line 5")

                # Simulate runtime error
                if "runtime_error" in script_content:
                    _ = 1 / 0  # Will raise ZeroDivisionError

                return "success"

            except SyntaxError as e:
                # Proper exception chaining (ADR-003)
                raise ScriptExecutionError(
                    f"Failed to parse script: {script_content[:20]}"
                ) from e

            except ZeroDivisionError as e:
                # Chain the original exception
                raise ScriptExecutionError("Script encountered runtime error") from e

        # Test syntax error handling with chaining
        with pytest.raises(ScriptExecutionError) as exc_info:
            execute_script("print('hello') syntax_error")

        assert "Failed to parse script" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, SyntaxError)

        # Test runtime error handling with chaining
        with pytest.raises(ScriptExecutionError) as exc_info:
            execute_script("calculate() runtime_error")

        assert "runtime error" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

        # Test error output schema
        error_output = ScriptAgentOutput(
            success=False,
            data=None,
            error="Script execution failed: Invalid input format",
        )

        assert error_output.success is False
        assert error_output.error is not None
        assert "Invalid input format" in error_output.error
        assert error_output.data is None

        # Test validation errors are properly caught
        try:
            # This will raise ValidationError
            ScriptAgentOutput(success="not_a_bool")  # type: ignore[arg-type]
        except ValidationError as e:
            error_output = ScriptAgentOutput(
                success=False, error=f"Validation failed: {str(e).split(' ', 1)[0]}"
            )
            assert error_output.success is False
            assert error_output.error is not None
            assert "Validation failed" in error_output.error
