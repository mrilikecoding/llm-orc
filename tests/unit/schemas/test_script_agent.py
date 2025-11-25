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

    def test_json_serialization(self) -> None:
        """Test JSON serialization/deserialization.

        Migrated from: tests/test_issue_24_units.py::test_json_serialization
        Related BDD: tests/bdd/features/issue-24-script-agents.feature (JSON I/O)
        """
        import json

        from llm_orc.schemas.script_agent import (
            AgentRequest,
            ScriptAgentInput,
            ScriptAgentOutput,
        )

        # Test input serialization to JSON
        input_data = ScriptAgentInput(
            agent_name="test-agent",
            input_data="test data",
            context={"key": "value", "number": 42},
            dependencies={"dep1": "value1", "dep2": "value2"},
        )

        # Serialize to JSON string
        json_str = input_data.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize back
        parsed_dict = json.loads(json_str)
        reconstructed = ScriptAgentInput(**parsed_dict)
        assert reconstructed == input_data
        assert reconstructed.agent_name == "test-agent"
        assert reconstructed.context["number"] == 42

        # Test output serialization with nested AgentRequest
        output_data = ScriptAgentOutput(
            success=True,
            data={"result": [1, 2, 3], "nested": {"key": "value"}},
            error=None,
            agent_requests=[
                AgentRequest(
                    target_agent_type="helper-agent",
                    parameters={"task": "help me", "urgency": "high"},
                    priority=1,
                )
            ],
        )

        # Serialize and deserialize
        output_json = output_data.model_dump_json()
        output_dict = json.loads(output_json)
        reconstructed_output = ScriptAgentOutput(**output_dict)
        assert reconstructed_output.success is True
        assert reconstructed_output.data["result"] == [1, 2, 3]
        assert len(reconstructed_output.agent_requests) == 1
        assert (
            reconstructed_output.agent_requests[0].target_agent_type == "helper-agent"
        )

        # Test edge cases: empty, None, special characters
        edge_case = ScriptAgentInput(
            agent_name="test-with-special-chars-éñ",
            input_data='{"nested": "json"}',
            context={},
            dependencies={},
        )
        edge_json = edge_case.model_dump_json()
        edge_reconstructed = ScriptAgentInput.model_validate_json(edge_json)
        assert edge_reconstructed.agent_name == "test-with-special-chars-éñ"
        assert edge_reconstructed.input_data == '{"nested": "json"}'

    def test_input_validation(self) -> None:
        """Test input validation from BDD scenario.

        Migrated from: tests/test_issue_24_units.py::test_validate_input
        Related BDD: tests/bdd/features/issue-24-script-agents.feature (validation)
        """
        import json

        from pydantic import ValidationError

        from llm_orc.schemas.script_agent import ScriptAgentInput

        # Test validation of required fields
        with pytest.raises(ValidationError) as exc_info:
            ScriptAgentInput()  # type: ignore[call-arg]

        assert "agent_name" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)

        # Test partial input validation
        with pytest.raises(ValidationError) as exc_info:
            ScriptAgentInput(agent_name="test")  # type: ignore[call-arg]

        assert "input_data" in str(exc_info.value)

        # Test field type validation
        with pytest.raises(ValidationError) as exc_info:
            ScriptAgentInput(
                agent_name=123,  # type: ignore[arg-type]
                input_data="data",
            )

        assert "Input should be a valid string" in str(exc_info.value)

        # Test valid input passes validation
        valid_input = ScriptAgentInput(
            agent_name="validator",
            input_data="validate this",
            context={"env": "test"},
            dependencies={"lib": "1.0"},
        )
        assert valid_input.agent_name == "validator"

        # Test JSON validation for schema-based execution
        # Invalid JSON string should raise when parsed
        invalid_json = "not json"
        try:
            json.loads(invalid_json)
            pytest.fail("Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            pass  # Expected

        # Valid JSON can be parsed and used with schemas
        valid_json = '{"agent_name": "test", "input_data": "data"}'
        parsed = json.loads(valid_json)
        input_from_json = ScriptAgentInput(**parsed)
        assert input_from_json.agent_name == "test"
        assert input_from_json.input_data == "data"
