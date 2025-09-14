"""Unit tests for script agent schemas.

This module contains tests for the Pydantic schemas defined in ADR-001,
ensuring proper validation and serialization behavior.

Migrated from: tests/test_issue_24_units.py::test_schema_validation
Related BDD: tests/bdd/features/issue-24-script-agents.feature (JSON I/O contract)
"""

import pytest
from pydantic import ValidationError

from llm_orc.schemas.script_agent import ScriptAgentInput, ScriptAgentOutput


class TestScriptAgentSchemas:
    """Unit tests for script agent Pydantic schemas (ADR-001)."""

    def test_schema_validation(self) -> None:
        """Test Pydantic schema validation logic.

        Originally from BDD scenario: Script agent executes with JSON I/O contract
        Tests the core Pydantic schema validation for script agent communication.
        """
        # Test valid input schema creation
        valid_input = ScriptAgentInput(
            agent_name="test-agent",
            input_data="test data",
            context={"key": "value"},
            dependencies={"dep": "value"},
        )
        assert valid_input.agent_name == "test-agent"
        assert valid_input.input_data == "test data"
        assert valid_input.context == {"key": "value"}
        assert valid_input.dependencies == {"dep": "value"}

        # Test minimal input (required fields only)
        minimal_input = ScriptAgentInput(agent_name="minimal", input_data="data")
        assert minimal_input.context == {}
        assert minimal_input.dependencies == {}

        # Test invalid input (missing required fields)
        with pytest.raises(ValidationError):
            ScriptAgentInput()  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            ScriptAgentInput(agent_name="test")  # type: ignore[call-arg] # missing input_data

        # Test valid output schema
        valid_output = ScriptAgentOutput(
            success=True, data={"result": "test"}, error=None, agent_requests=[]
        )
        assert valid_output.success is True
        assert valid_output.data == {"result": "test"}
        assert valid_output.error is None
        assert valid_output.agent_requests == []

        # Test error output
        error_output = ScriptAgentOutput(success=False, error="Test error message")
        assert error_output.success is False
        assert error_output.error == "Test error message"
        assert error_output.data is None
