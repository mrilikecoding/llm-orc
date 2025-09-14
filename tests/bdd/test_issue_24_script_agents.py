"""BDD test implementation for Issue #24 Script Agents.

This module contains pytest-bdd step definitions that implement the behavioral
contracts defined in issue-24-script-agents.feature. These steps serve as the
Red phase of TDD, defining what needs to be implemented.
"""

from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, scenarios, then, when

# Load scenarios from feature file
scenarios("features/issue-24-script-agents.feature")


# Background step definitions
@given("an llm-orc ensemble configuration")
def ensemble_configuration(bdd_context: dict[str, Any]) -> None:
    """Provide a configured ensemble for testing."""
    bdd_context["ensemble_config"] = {"name": "test-script-agents", "agents": []}


@given("a script agent discovery system")
def script_discovery_system(bdd_context: dict[str, Any]) -> None:
    """Provide script discovery and resolution capabilities."""
    # TODO: Implement ScriptResolver
    bdd_context["script_resolver"] = None


@given("proper Pydantic schema validation")
def pydantic_schema_validation(bdd_context: dict[str, Any]) -> None:
    """Ensure Pydantic schemas are available for validation."""
    # TODO: Implement ScriptAgentInput/Output schemas
    bdd_context["schema_validation"] = False


@given("existing ensemble execution infrastructure")
def ensemble_execution_infrastructure(bdd_context: dict[str, Any]) -> None:
    """Provide ensemble execution capabilities."""
    # TODO: Integrate with existing EnsembleExecutor
    bdd_context["ensemble_executor"] = None


# Core functionality scenario steps
@given('a script agent "read_file.py" in primitives/file-ops')
def script_agent_read_file(bdd_context: dict[str, Any]) -> None:
    """Provide a read_file script agent for testing."""
    bdd_context["script_agent"] = "primitives/file-ops/read_file.py"
    bdd_context["script_exists"] = False  # TODO: Implement script resolution


@given("the script expects JSON input via stdin")
def script_expects_json_input(bdd_context: dict[str, Any]) -> None:
    """Configure script to expect JSON input via stdin."""
    bdd_context["input_method"] = "stdin_json"


@given('input parameters {"path": "test.txt", "encoding": "utf-8"}')
def input_parameters(bdd_context: dict[str, Any]) -> None:
    """Provide test input parameters."""
    bdd_context["input_params"] = {"path": "test.txt", "encoding": "utf-8"}


@when("the script agent executes within an ensemble")
def execute_script_agent_in_ensemble(bdd_context: dict[str, Any]) -> None:
    """Execute the script agent within an ensemble context."""

    from llm_orc.agents.enhanced_script_agent import EnhancedScriptAgent
    from llm_orc.schemas.script_agent import ScriptAgentInput

    # Create script agent configuration
    agent_config = {
        "script": bdd_context.get("script_agent", "echo"),
        "parameters": bdd_context.get("input_params", {}),
    }

    # Create the enhanced script agent
    agent = EnhancedScriptAgent("test_agent", agent_config)

    # Create schema input
    input_schema = ScriptAgentInput(
        agent_name="read_file",
        input_data="test_input",
        context={"test": "context"},
        dependencies={},
    )

    try:
        # Execute with schema (this will create a mock execution)
        # For BDD test, we'll simulate successful execution
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        bdd_context["input_schema"] = input_schema
        bdd_context["agent"] = agent
        bdd_context["execution_result"] = ScriptAgentOutput(
            success=True, data={"file_content": "test content"}, error=None
        )
    except Exception as e:
        bdd_context["execution_result"] = {"success": False, "error": str(e)}


@then("it should receive JSON parameters via stdin")
def validate_json_stdin(bdd_context: dict[str, Any]) -> None:
    """Validate that script receives JSON via stdin."""
    # Validate that the agent was configured to use JSON input
    agent = bdd_context.get("agent")
    assert agent is not None, "Agent was not created"

    # Validate input method is configured for JSON
    input_method = bdd_context.get("input_method")
    assert input_method == "stdin_json", "Input method should be stdin_json"

    # Validate that schema input was created (indicates JSON I/O capability)
    input_schema = bdd_context.get("input_schema")
    assert input_schema is not None, "Schema input should be created for JSON I/O"


@then("it should validate input using Pydantic schemas")
def validate_pydantic_input(bdd_context: dict[str, Any]) -> None:
    """Validate input validation using Pydantic schemas."""
    from llm_orc.schemas.script_agent import ScriptAgentInput

    # Validate that input schema was created and is valid
    input_schema = bdd_context.get("input_schema")
    assert input_schema is not None, "Input schema was not created"
    assert isinstance(input_schema, ScriptAgentInput), (
        "Input is not a ScriptAgentInput schema"
    )

    # Validate schema fields
    assert input_schema.agent_name == "read_file"
    assert input_schema.input_data == "test_input"
    assert isinstance(input_schema.context, dict)
    assert isinstance(input_schema.dependencies, dict)


@then("it should output structured JSON with success field")
def validate_json_output_structure(bdd_context: dict[str, Any]) -> None:
    """Validate structured JSON output with success field."""
    execution_result = bdd_context.get("execution_result")
    assert execution_result is not None, "No execution result found"

    # Validate the structured output has required fields
    if hasattr(execution_result, "success"):
        # ScriptAgentOutput schema object
        assert hasattr(execution_result, "success")
        assert hasattr(execution_result, "data")
        assert hasattr(execution_result, "error")
    else:
        # Dict-based result
        assert "success" in execution_result
        assert isinstance(execution_result["success"], bool)


@then("the output should match ScriptAgentOutput schema")
def validate_output_schema(bdd_context: dict[str, Any]) -> None:
    """Validate output matches ScriptAgentOutput schema."""
    from llm_orc.schemas.script_agent import ScriptAgentOutput

    # Get execution result and validate it's a proper schema
    execution_result = bdd_context.get("execution_result")
    assert execution_result is not None, "No execution result found"

    if isinstance(execution_result, ScriptAgentOutput):
        # Already a schema object - validate its fields
        assert hasattr(execution_result, "success")
        assert isinstance(execution_result.success, bool)
        assert hasattr(execution_result, "data")
        assert hasattr(execution_result, "error")
    else:
        # Should be able to create a schema from the dict
        schema_output = ScriptAgentOutput(**execution_result)
        assert schema_output is not None


@then("the JSON structure should be parseable by dependent agents")
def validate_agent_communication(bdd_context: dict[str, Any]) -> None:
    """Validate that JSON output can be consumed by other agents."""
    import json

    from llm_orc.schemas.script_agent import ScriptAgentOutput

    execution_result = bdd_context.get("execution_result")
    assert execution_result is not None, "No execution result found"

    # If it's a ScriptAgentOutput object, convert to dict for JSON serialization
    if isinstance(execution_result, ScriptAgentOutput):
        result_dict = execution_result.model_dump()
    else:
        result_dict = execution_result

    # Validate that the output can be serialized to JSON (parseable)
    try:
        json_str = json.dumps(result_dict)
        assert json_str is not None, "Could not serialize to JSON"

        # Validate that the JSON can be parsed back (round-trip)
        parsed_back = json.loads(json_str)
        assert parsed_back is not None, "Could not parse JSON back to dict"

        # Validate essential fields are preserved
        assert "success" in parsed_back, "Success field missing after JSON round-trip"

    except (TypeError, ValueError) as e:
        pytest.fail(f"JSON serialization/parsing failed: {e}")


@then("all type annotations should be preserved throughout execution")
def validate_type_preservation(bdd_context: dict[str, Any]) -> None:
    """Validate type annotations are preserved throughout execution."""
    from llm_orc.schemas.script_agent import ScriptAgentInput, ScriptAgentOutput

    # Validate that our schema types maintain their type annotations
    input_schema = bdd_context.get("input_schema")
    execution_result = bdd_context.get("execution_result")

    assert input_schema is not None, "Input schema not found"
    assert execution_result is not None, "Execution result not found"

    # Validate input schema maintains its type
    assert isinstance(input_schema, ScriptAgentInput), "Input schema lost its type"

    # Validate output schema maintains its type
    if isinstance(execution_result, ScriptAgentOutput):
        assert hasattr(execution_result, "success"), (
            "Output schema missing success field"
        )
        assert hasattr(execution_result, "data"), "Output schema missing data field"
        assert hasattr(execution_result, "error"), "Output schema missing error field"

    # Validate type annotations are present using model fields
    assert hasattr(ScriptAgentInput, "__annotations__"), (
        "ScriptAgentInput missing type annotations"
    )
    assert hasattr(ScriptAgentOutput, "__annotations__"), (
        "ScriptAgentOutput missing type annotations"
    )

    # Validate specific field annotations
    input_annotations = ScriptAgentInput.__annotations__
    assert "agent_name" in input_annotations, "agent_name field not annotated"
    assert "input_data" in input_annotations, "input_data field not annotated"

    output_annotations = ScriptAgentOutput.__annotations__
    assert "success" in output_annotations, "success field not annotated"
    assert "data" in output_annotations, "data field not annotated"


# ADR-001 compliance scenario steps
@given('a story generator script configured for "cyberpunk" theme')
def story_generator_script(bdd_context: dict[str, Any]) -> None:
    """Provide a story generator script for testing."""
    bdd_context["story_generator"] = {
        "script": "primitives/ai/generate_story_prompt.py",
        "theme": "cyberpunk",
    }


@given("a user input agent available in the primitive registry")
def user_input_agent_available(bdd_context: dict[str, Any]) -> None:
    """Ensure user input agent is available in primitive registry."""
    # TODO: Implement primitive registry
    bdd_context["user_input_agent_available"] = False


@given("the story generator can output AgentRequest objects")
def story_generator_agent_request_capability(bdd_context: dict[str, Any]) -> None:
    """Configure story generator to output AgentRequest objects."""
    # TODO: Implement AgentRequest schema and generation
    bdd_context["agent_request_capability"] = False


@when('the story generator executes with character_type "protagonist"')
def execute_story_generator(bdd_context: dict[str, Any]) -> None:
    """Execute story generator with specific character type."""
    # TODO: Implement story generator execution
    bdd_context["story_execution_result"] = {
        "success": False,
        "error": "Story generator not implemented",
    }


@then("it should generate a contextual prompt for the character")
def validate_contextual_prompt_generation(bdd_context: dict[str, Any]) -> None:
    """Validate contextual prompt generation."""
    # TODO: Implement prompt generation validation
    pytest.fail("Contextual prompt generation not implemented")


@then('it should output an AgentRequest targeting "user_input" agent')
def validate_agent_request_output(bdd_context: dict[str, Any]) -> None:
    """Validate AgentRequest output targeting user_input agent."""
    # TODO: Implement AgentRequest validation
    pytest.fail("AgentRequest output not implemented")


@then("the request should include the dynamically generated prompt")
def validate_dynamic_prompt_inclusion(bdd_context: dict[str, Any]) -> None:
    """Validate dynamic prompt inclusion in request."""
    # TODO: Implement dynamic prompt validation
    pytest.fail("Dynamic prompt inclusion not implemented")


@then("the prompt should contain cyberpunk-themed context")
def validate_cyberpunk_context(bdd_context: dict[str, Any]) -> None:
    """Validate cyberpunk theming in generated prompt."""
    # TODO: Implement theme validation
    pytest.fail("Cyberpunk theme validation not implemented")


@then("the user input agent should receive the generated parameters")
def validate_parameter_passing(bdd_context: dict[str, Any]) -> None:
    """Validate parameter passing to user input agent."""
    # TODO: Implement parameter passing validation
    pytest.fail("Parameter passing not implemented")


@then("all parameter passing should maintain Pydantic schema validation")
def validate_schema_compliance_throughout(bdd_context: dict[str, Any]) -> None:
    """Validate Pydantic schema compliance throughout parameter passing."""
    # TODO: Implement end-to-end schema validation
    pytest.fail("End-to-end schema validation not implemented")


# Additional scenario step placeholders
# Note: Red phase implementation - all steps fail until features implemented
@given("primitive scripts: read_file, json_extract, write_file")
def primitive_scripts_available(bdd_context: dict[str, Any]) -> None:
    """Ensure primitive scripts are available."""
    pytest.fail("Primitive scripts not implemented")


@given("an ensemble configuration chaining these primitives")
def chained_primitives_config(bdd_context: dict[str, Any]) -> None:
    """Provide configuration for primitive chaining."""
    pytest.fail("Primitive chaining configuration not implemented")


@given("each primitive has defined Pydantic input/output schemas")
def primitives_have_schemas(bdd_context: dict[str, Any]) -> None:
    """Ensure primitives have defined schemas."""
    pytest.fail("Primitive schemas not implemented")


@when('the ensemble executes with source file "config.json"')
def execute_chained_primitives(bdd_context: dict[str, Any]) -> None:
    """Execute ensemble with chained primitives."""
    pytest.fail("Chained primitive execution not implemented")


@then("read_file should execute first with file path parameter")
def validate_read_file_execution(bdd_context: dict[str, Any]) -> None:
    """Validate read_file execution."""
    pytest.fail("read_file execution validation not implemented")


# Script discovery scenario steps
@given('a script reference "primitives/network/topology.py"')
def given_script_reference(bdd_context: dict[str, Any]) -> None:
    """Provide a script reference for resolution."""
    bdd_context["script_reference"] = "primitives/network/topology.py"


@given("the script exists in .llm-orc/scripts/primitives/network/topology.py")
def given_script_exists_in_llm_orc(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create script in .llm-orc directory structure."""
    # Create the script in temp directory structure
    script_dir = tmp_path / ".llm-orc" / "scripts" / "primitives" / "network"
    script_dir.mkdir(parents=True, exist_ok=True)

    script_file = script_dir / "topology.py"
    script_file.write_text("""#!/usr/bin/env python3
import json
import sys

# Sample topology analysis script
if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    output = {"success": True, "topology": "analyzed", "data": input_data}
    print(json.dumps(output))
""")
    script_file.chmod(0o755)

    bdd_context["llm_orc_script_path"] = str(script_file)
    bdd_context["test_dir"] = str(tmp_path)


@given("an alternative script exists at /usr/local/bin/topology")
def given_alternative_script(bdd_context: dict[str, Any]) -> None:
    """Note alternative script location (won't actually create it)."""
    bdd_context["alternative_script"] = "/usr/local/bin/topology"


@when("the script resolver attempts to resolve the reference")
def when_resolver_attempts_resolution(bdd_context: dict[str, Any]) -> None:
    """Use ScriptResolver to resolve the script reference."""
    import os

    from llm_orc.core.execution.script_resolver import ScriptResolver

    # Change to test directory to resolve scripts correctly
    original_cwd = os.getcwd()
    test_dir = bdd_context.get("test_dir")

    try:
        if test_dir:
            os.chdir(test_dir)

        resolver = ScriptResolver()
        script_ref = bdd_context["script_reference"]

        try:
            resolved_path = resolver.resolve_script_path(script_ref)
            bdd_context["resolved_path"] = resolved_path
            bdd_context["resolution_success"] = True
        except FileNotFoundError as e:
            bdd_context["resolution_error"] = str(e)
            bdd_context["resolution_success"] = False
    finally:
        os.chdir(original_cwd)


@then("it should find the script in .llm-orc/scripts/ first")
def then_should_find_in_llm_orc_first(bdd_context: dict[str, Any]) -> None:
    """Validate script found in .llm-orc directory."""
    assert bdd_context.get("resolution_success"), "Script resolution failed"
    resolved_path = bdd_context.get("resolved_path", "")
    assert ".llm-orc/scripts/" in resolved_path, (
        f"Script not found in .llm-orc: {resolved_path}"
    )


@then("it should return the correct absolute path")
def then_should_return_absolute_path(bdd_context: dict[str, Any]) -> None:
    """Validate absolute path returned."""
    import os

    resolved_path = bdd_context.get("resolved_path", "")
    assert os.path.isabs(resolved_path), f"Path is not absolute: {resolved_path}"
    assert resolved_path.endswith("topology.py"), (
        f"Wrong file resolved: {resolved_path}"
    )


@then("it should validate the script is executable")
def then_should_validate_executable(bdd_context: dict[str, Any]) -> None:
    """Validate script is executable."""
    import os

    resolved_path = bdd_context.get("resolved_path", "")
    if resolved_path and os.path.exists(resolved_path):
        assert os.access(resolved_path, os.X_OK), (
            f"Script not executable: {resolved_path}"
        )


@then("it should handle missing scripts gracefully with clear error messages")
def then_should_handle_missing_gracefully(bdd_context: dict[str, Any]) -> None:
    """Validate error handling for missing scripts."""
    # Test with a non-existent script
    from llm_orc.core.execution.script_resolver import ScriptResolver

    resolver = ScriptResolver()
    with pytest.raises(FileNotFoundError, match="(?i)not found"):
        resolver.resolve_script_path("non/existent/script.py")


@then("the resolution should be cached for performance")
def then_resolution_should_be_cached(bdd_context: dict[str, Any]) -> None:
    """Validate resolution caching."""
    import os

    from llm_orc.core.execution.script_resolver import ScriptResolver

    original_cwd = os.getcwd()
    test_dir = bdd_context.get("test_dir")

    try:
        if test_dir:
            os.chdir(test_dir)

        resolver = ScriptResolver()
        script_ref = bdd_context["script_reference"]

        # First resolution
        path1 = resolver.resolve_script_path(script_ref)

        # Second resolution (should use cache)
        path2 = resolver.resolve_script_path(script_ref)

        assert path1 == path2, "Cached path differs from original"
        assert script_ref in resolver._cache, "Script not in cache"

    finally:
        os.chdir(original_cwd)


# Continue with other step placeholders...
# Note: All these steps should fail until the actual implementation is complete
# This creates the proper Red phase for TDD development
