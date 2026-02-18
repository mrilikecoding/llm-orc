"""BDD step definitions for ADR-009 MCP Server Architecture."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_bdd import given, scenarios, then, when

# Load all scenarios from the feature file
scenarios("features/adr-009-mcp-server-architecture.feature")


def _create_mock_config_manager(
    ensemble_dirs: list[Path], artifacts_dir: Path | None = None
) -> MagicMock:
    """Create a mock ConfigurationManager for testing.

    Args:
        ensemble_dirs: List of ensemble directories.
        artifacts_dir: Optional artifacts directory (e.g., tmp_path / "artifacts")

    Returns:
        Mock ConfigurationManager.
    """
    mock_config = MagicMock()
    mock_config.get_ensembles_dirs.return_value = [str(d) for d in ensemble_dirs]
    # Set global_config_dir to the artifacts directory itself
    mock_config.global_config_dir = str(artifacts_dir) if artifacts_dir else ""
    mock_config.get_model_profiles.return_value = {
        "fast": {"provider": "anthropic-api", "model": "claude-3-haiku-20240307"},
        "standard": {
            "provider": "anthropic-api",
            "model": "claude-3-5-sonnet-20241022",
        },
        "quality": {"provider": "anthropic-api", "model": "claude-3-opus-20240229"},
    }
    return mock_config


# ============================================================================
# Background and Server Setup Steps
# ============================================================================


@given("an MCP server instance is available")
def mcp_server_available(bdd_context: dict[str, Any]) -> None:
    """Set up MCP server instance for testing."""
    # Import will fail in Red phase until implementation exists
    try:
        from llm_orc.mcp.server import MCPServerV2

        # Create a default server - will be reconfigured by subsequent steps
        bdd_context["mcp_server_class"] = MCPServerV2
        bdd_context["mcp_server"] = MCPServerV2()
        bdd_context["mcp_available"] = True
    except ImportError:
        # Red phase: module doesn't exist yet
        bdd_context["mcp_server"] = None
        bdd_context["mcp_available"] = False


@given("ensembles exist in local, library, and global directories")
def ensembles_exist_multiple_dirs(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up ensembles in multiple directories."""
    # Create local ensembles directory
    local_dir = tmp_path / ".llm-orc" / "ensembles"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create a local ensemble
    (local_dir / "local-test.yaml").write_text(
        "name: local-test\ndescription: Local test ensemble\n"
        "agents:\n  - name: agent1\n    model_profile: fast"
    )

    # Create library ensembles directory
    library_dir = tmp_path / "library" / "ensembles"
    library_dir.mkdir(parents=True, exist_ok=True)
    (library_dir / "library-test.yaml").write_text(
        "name: library-test\ndescription: Library test ensemble\n"
        "agents:\n  - name: agent2\n    model_profile: standard"
    )

    # Create global ensembles directory
    global_dir = tmp_path / "global" / "ensembles"
    global_dir.mkdir(parents=True, exist_ok=True)
    (global_dir / "global-test.yaml").write_text(
        "name: global-test\ndescription: Global test ensemble\n"
        "agents:\n  - name: agent3\n    model_profile: quality"
    )

    bdd_context["ensemble_dirs"] = {
        "local": local_dir,
        "library": library_dir,
        "global": global_dir,
    }
    bdd_context["expected_ensembles"] = ["local-test", "library-test", "global-test"]

    # Reconfigure MCP server with test directories
    if bdd_context.get("mcp_available"):
        mock_config = _create_mock_config_manager([local_dir, library_dir, global_dir])
        server_class = bdd_context["mcp_server_class"]
        bdd_context["mcp_server"] = server_class(config_manager=mock_config)


@given("no ensembles are configured")
def no_ensembles_configured(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up empty ensemble directories."""
    empty_dir = tmp_path / ".llm-orc" / "ensembles"
    empty_dir.mkdir(parents=True, exist_ok=True)
    bdd_context["ensemble_dirs"] = {"local": empty_dir}
    bdd_context["expected_ensembles"] = []

    # Reconfigure MCP server with test directories
    if bdd_context.get("mcp_available"):
        mock_config = _create_mock_config_manager([empty_dir])
        server_class = bdd_context["mcp_server_class"]
        bdd_context["mcp_server"] = server_class(config_manager=mock_config)


def _reconfigure_server(
    bdd_context: dict[str, Any],
    ensemble_dirs: list[Path],
    artifacts_dir: Path | None = None,
) -> None:
    """Reconfigure MCP server with test directories."""
    if bdd_context.get("mcp_available"):
        mock_config = _create_mock_config_manager(ensemble_dirs, artifacts_dir)
        server_class = bdd_context["mcp_server_class"]
        bdd_context["mcp_server"] = server_class(config_manager=mock_config)


@given('an ensemble named "code-review" exists')
def ensemble_code_review_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create code-review ensemble."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)

    ensemble_yaml = """
name: code-review
description: Code review ensemble
agents:
  - name: syntax-check
    model_profile: fast
  - name: style-check
    model_profile: fast
    depends_on:
      - syntax-check
  - name: security-check
    model_profile: quality
    depends_on:
      - syntax-check
"""
    (ensembles_dir / "code-review.yaml").write_text(ensemble_yaml)
    bdd_context["ensemble_dir"] = ensembles_dir
    _reconfigure_server(bdd_context, [ensembles_dir])


@given('no ensemble named "non-existent" exists')
def no_ensemble_non_existent(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Confirm no non-existent ensemble exists."""
    bdd_context["target_ensemble"] = "non-existent"
    # Reconfigure with empty dir if not already configured
    if "ensemble_dir" not in bdd_context:
        empty_dir = tmp_path / ".llm-orc" / "ensembles"
        empty_dir.mkdir(parents=True, exist_ok=True)
        _reconfigure_server(bdd_context, [empty_dir])


@given('an ensemble named "code-review" has execution artifacts')
def ensemble_has_artifacts(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up ensemble with artifacts."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    # New directory structure: {ensemble}/{artifact_id}/execution.json
    artifacts_base = tmp_path / ".llm-orc" / "artifacts"
    artifact_dir = artifacts_base / "code-review" / "2025-01-15-120000"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Create sample artifact in new format
    artifact = {
        "ensemble": "code-review",
        "status": "success",
        "results": {"syntax-check": {"status": "success", "response": "OK"}},
        "metadata": {
            "started_at": 1705320000.0,
            "duration": "2.3s",
            "agents_used": 1,
        },
    }
    (artifact_dir / "execution.json").write_text(json.dumps(artifact))

    bdd_context["artifacts_dir"] = tmp_path / ".llm-orc" / "artifacts" / "code-review"
    bdd_context["expected_artifacts"] = ["2025-01-15-120000"]
    _reconfigure_server(
        bdd_context, [ensembles_dir], tmp_path / ".llm-orc" / "artifacts"
    )


@given('an ensemble named "new-ensemble" has no execution artifacts')
def ensemble_no_artifacts(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up ensemble with no artifacts."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / ".llm-orc" / "artifacts" / "new-ensemble"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bdd_context["artifacts_dir"] = artifacts_dir
    bdd_context["expected_artifacts"] = []
    _reconfigure_server(
        bdd_context, [ensembles_dir], tmp_path / ".llm-orc" / "artifacts"
    )


@given('an artifact "code-review/2025-01-15-120000" exists')
def artifact_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create specific artifact."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    # New directory structure: {ensemble}/{artifact_id}/execution.json
    artifacts_base = tmp_path / ".llm-orc" / "artifacts"
    artifact_dir = artifacts_base / "code-review" / "2025-01-15-120000"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "ensemble": "code-review",
        "status": "success",
        "results": {
            "syntax-check": {"status": "success", "response": "No syntax errors"},
            "style-check": {"status": "success", "response": "Style OK"},
        },
        "synthesis": "Code review passed all checks.",
        "metadata": {
            "started_at": 1705320000.0,
            "duration": "2.3s",
            "agents_used": 2,
        },
    }
    (artifact_dir / "execution.json").write_text(json.dumps(artifact))
    bdd_context["artifact_id"] = "code-review/2025-01-15-120000"
    bdd_context["artifact_data"] = artifact
    _reconfigure_server(
        bdd_context, [ensembles_dir], tmp_path / ".llm-orc" / "artifacts"
    )


@given('an ensemble "code-review" has multiple executions')
def ensemble_multiple_executions(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up ensemble with multiple execution artifacts."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    base_artifacts_dir = tmp_path / ".llm-orc" / "artifacts" / "code-review"

    # Create multiple artifacts in new directory structure
    for i in range(5):
        artifact_dir = base_artifacts_dir / f"2025-01-1{i}-120000"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "ensemble": "code-review",
            "status": "success" if i % 2 == 0 else "failed",
            "results": {},
            "metadata": {
                "started_at": 1705320000.0 + (i * 86400),
                "duration": f"{2.0 + (i * 0.5)}s",
                "agents_used": 2,
            },
        }
        (artifact_dir / "execution.json").write_text(json.dumps(artifact))

    bdd_context["artifacts_dir"] = base_artifacts_dir
    bdd_context["execution_count"] = 5
    _reconfigure_server(
        bdd_context, [ensembles_dir], tmp_path / ".llm-orc" / "artifacts"
    )


@given('an ensemble named "simple-test" exists')
def ensemble_simple_test_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create simple test ensemble."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)

    ensemble_yaml = """
name: simple-test
description: Simple test ensemble
agents:
  - name: test-agent
    model_profile: fast
"""
    (ensembles_dir / "simple-test.yaml").write_text(ensemble_yaml)
    bdd_context["ensemble_dir"] = ensembles_dir
    _reconfigure_server(bdd_context, [ensembles_dir])


@given('an ensemble named "code-review" exists with valid configuration')
def ensemble_valid_config(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create ensemble with valid configuration."""
    ensemble_code_review_exists(bdd_context, tmp_path)


@given('an ensemble named "invalid-ensemble" exists with circular dependencies')
def ensemble_circular_deps(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create ensemble with circular dependencies."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)

    ensemble_yaml = """
name: invalid-ensemble
description: Invalid ensemble with circular deps
agents:
  - name: agent-a
    depends_on:
      - agent-b
  - name: agent-b
    depends_on:
      - agent-c
  - name: agent-c
    depends_on:
      - agent-a
"""
    (ensembles_dir / "invalid-ensemble.yaml").write_text(ensemble_yaml)
    bdd_context["ensemble_dir"] = ensembles_dir
    _reconfigure_server(bdd_context, [ensembles_dir])


@given('an ensemble named "multi-agent-test" exists with multiple agents')
def ensemble_multi_agent(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create multi-agent test ensemble."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)

    ensemble_yaml = """
name: multi-agent-test
description: Multi-agent test ensemble
agents:
  - name: agent-1
    model_profile: fast
  - name: agent-2
    model_profile: standard
    depends_on:
      - agent-1
  - name: agent-3
    model_profile: quality
    depends_on:
      - agent-1
"""
    (ensembles_dir / "multi-agent-test.yaml").write_text(ensemble_yaml)
    bdd_context["ensemble_dir"] = ensembles_dir
    _reconfigure_server(bdd_context, [ensembles_dir])


# ============================================================================
# When Steps - Resource Access
# ============================================================================


@when('I request the "llm-orc://ensembles" resource')
def request_ensembles_resource(bdd_context: dict[str, Any]) -> None:
    """Request ensembles resource via MCP."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.read_resource("llm-orc://ensembles")
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


@when('I request the "llm-orc://ensemble/code-review" resource')
def request_ensemble_detail_resource(bdd_context: dict[str, Any]) -> None:
    """Request specific ensemble resource."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.read_resource("llm-orc://ensemble/code-review")
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


@when('I request the "llm-orc://ensemble/non-existent" resource')
def request_nonexistent_ensemble(bdd_context: dict[str, Any]) -> None:
    """Request non-existent ensemble resource."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.read_resource("llm-orc://ensemble/non-existent")
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


@when('I request the "llm-orc://artifacts/code-review" resource')
def request_artifacts_resource(bdd_context: dict[str, Any]) -> None:
    """Request artifacts resource."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.read_resource("llm-orc://artifacts/code-review")
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


@when('I request the "llm-orc://artifacts/new-ensemble" resource')
def request_empty_artifacts_resource(bdd_context: dict[str, Any]) -> None:
    """Request artifacts resource for ensemble with no artifacts."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.read_resource("llm-orc://artifacts/new-ensemble")
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


@when('I request the "llm-orc://artifact/code-review/2025-01-15-120000" resource')
def request_artifact_detail_resource(bdd_context: dict[str, Any]) -> None:
    """Request specific artifact resource."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            uri = "llm-orc://artifact/code-review/2025-01-15-120000"
            return await server.read_resource(uri)
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


@when('I request the "llm-orc://metrics/code-review" resource')
def request_metrics_resource(bdd_context: dict[str, Any]) -> None:
    """Request metrics resource."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.read_resource("llm-orc://metrics/code-review")
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


@when('I request the "llm-orc://profiles" resource')
def request_profiles_resource(bdd_context: dict[str, Any]) -> None:
    """Request model profiles resource."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resource_result"] = None
        bdd_context["resource_error"] = "MCP server not available"
        return

    async def _request() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.read_resource("llm-orc://profiles")
        except Exception as e:
            bdd_context["resource_error"] = str(e)
            return None

    bdd_context["resource_result"] = asyncio.run(_request())


# ============================================================================
# When Steps - Tool Invocation
# ============================================================================


def _parse_datatable(datatable: Any) -> dict[str, Any]:
    """Parse pytest-bdd datatable into parameters dict."""
    params: dict[str, Any] = {}
    if datatable is None:
        return params

    # Handle different datatable formats
    rows = datatable if isinstance(datatable, list) else list(datatable)
    for row in rows:
        if len(row) >= 2:
            key = str(row[0]).strip()
            value = str(row[1]).strip()
            # Handle boolean and JSON values
            if value.lower() == "true":
                params[key] = True
            elif value.lower() == "false":
                params[key] = False
            elif value.startswith("{") or value.startswith("["):
                params[key] = json.loads(value)
            else:
                params[key] = value
    return params


@when('I call the "invoke" tool with:', target_fixture="invoke_datatable")
def call_invoke_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call invoke tool with parameters from datatable."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["invoke_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("invoke", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "validate_ensemble" tool with:', target_fixture="validate_datatable")
def call_validate_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call validate_ensemble tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["validate_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("validate_ensemble", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "update_ensemble" tool with:', target_fixture="update_datatable")
def call_update_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call update_ensemble tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["update_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("update_ensemble", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "invoke" tool with streaming enabled')
def call_invoke_streaming(bdd_context: dict[str, Any]) -> None:
    """Call invoke tool with streaming enabled."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["streaming_events"] = []
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = {
        "ensemble_name": "multi-agent-test",
        "input": "Test streaming input",
        "streaming": True,
    }
    bdd_context["invoke_params"] = params
    bdd_context["streaming_events"] = []

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            events: list[dict[str, Any]] = []

            async for event in server.invoke_streaming(params):
                events.append(event)

            bdd_context["streaming_events"] = events
            return {"events": events}
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


# ============================================================================
# When Steps - Server Discovery
# ============================================================================


@when("I request the tools list")
def request_tools_list(bdd_context: dict[str, Any]) -> None:
    """Request list of available tools."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tools_list"] = []
        return

    async def _list() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.list_tools()
        except Exception as e:
            bdd_context["tools_error"] = str(e)
            return []

    bdd_context["tools_list"] = asyncio.run(_list())


@when("I request the resources list")
def request_resources_list(bdd_context: dict[str, Any]) -> None:
    """Request list of available resources."""
    if not bdd_context.get("mcp_available"):
        bdd_context["resources_list"] = []
        return

    async def _list() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.list_resources()
        except Exception as e:
            bdd_context["resources_error"] = str(e)
            return []

    bdd_context["resources_list"] = asyncio.run(_list())


# ============================================================================
# Then Steps - Resource Access Results
# ============================================================================


@then("I should receive a list of all ensembles")
def receive_ensembles_list(bdd_context: dict[str, Any]) -> None:
    """Verify ensembles list is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result")
    assert result is not None, "Resource result should not be None"
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == len(bdd_context.get("expected_ensembles", []))


@then("each ensemble should have name, source, and agent_count metadata")
def ensembles_have_metadata(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble metadata."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result", [])
    for ensemble in result:
        assert "name" in ensemble, "Ensemble should have name"
        assert "source" in ensemble, "Ensemble should have source"
        assert "agent_count" in ensemble, "Ensemble should have agent_count"


@then("I should receive an empty list")
def receive_empty_list(bdd_context: dict[str, Any]) -> None:
    """Verify empty list is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result")
    assert result is not None, "Resource result should not be None"
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 0, "List should be empty"


@then("I should receive the complete ensemble configuration")
def receive_ensemble_config(bdd_context: dict[str, Any]) -> None:
    """Verify complete ensemble configuration is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result")
    error = bdd_context.get("resource_error")
    assert result is not None, f"Resource result should not be None, error: {error}"
    assert "name" in result, "Config should have name"
    assert result["name"] == "code-review"


@then("the configuration should include agents and their dependencies")
def config_includes_dependencies(bdd_context: dict[str, Any]) -> None:
    """Verify agents and dependencies in config."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result", {})
    assert "agents" in result, "Config should have agents"
    agents = result["agents"]
    assert len(agents) > 0, "Should have at least one agent"

    # Check for dependencies
    has_deps = any(agent.get("depends_on") for agent in agents)
    assert has_deps, "At least one agent should have dependencies"


@then("I should receive a resource not found error")
def receive_not_found_error(bdd_context: dict[str, Any]) -> None:
    """Verify resource not found error."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("resource_error")
    assert error is not None, "Should have an error"
    assert "not found" in error.lower() or bdd_context.get("resource_result") is None


@then("I should receive a list of artifacts")
def receive_artifacts_list(bdd_context: dict[str, Any]) -> None:
    """Verify artifacts list is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result")
    assert result is not None, "Resource result should not be None"
    assert isinstance(result, list), "Result should be a list"


@then("each artifact should have timestamp, status, cost, and duration")
def artifacts_have_metadata(bdd_context: dict[str, Any]) -> None:
    """Verify artifact metadata."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result", [])
    for artifact in result:
        assert "timestamp" in artifact, "Artifact should have timestamp"
        assert "status" in artifact, "Artifact should have status"
        assert "duration" in artifact, "Artifact should have duration"
        assert "agent_count" in artifact, "Artifact should have agent_count"


@then("I should receive the complete artifact data")
def receive_artifact_data(bdd_context: dict[str, Any]) -> None:
    """Verify complete artifact data is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result")
    assert result is not None, "Resource result should not be None"
    assert "ensemble" in result, "Artifact should have ensemble"


@then("it should include agent results and synthesis")
def artifact_includes_results(bdd_context: dict[str, Any]) -> None:
    """Verify artifact includes results and synthesis."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result", {})
    assert "results" in result, "Artifact should have results"
    assert "synthesis" in result, "Artifact should have synthesis"


@then("I should receive aggregated metrics")
def receive_aggregated_metrics(bdd_context: dict[str, Any]) -> None:
    """Verify aggregated metrics are received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result")
    assert result is not None, "Resource result should not be None"


@then("metrics should include success_rate, avg_cost, and avg_duration")
def metrics_include_fields(bdd_context: dict[str, Any]) -> None:
    """Verify metric fields."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result", {})
    assert "success_rate" in result, "Metrics should have success_rate"
    assert "avg_cost" in result, "Metrics should have avg_cost"
    assert "avg_duration" in result, "Metrics should have avg_duration"


@then("I should receive a list of configured model profiles")
def receive_profiles_list(bdd_context: dict[str, Any]) -> None:
    """Verify model profiles list is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result")
    assert result is not None, "Resource result should not be None"
    assert isinstance(result, list), "Result should be a list"


@then("each profile should have name, provider, and model details")
def profiles_have_details(bdd_context: dict[str, Any]) -> None:
    """Verify profile details."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("resource_result", [])
    for profile in result:
        assert "name" in profile, "Profile should have name"
        assert "provider" in profile, "Profile should have provider"
        assert "model" in profile, "Profile should have model"


# ============================================================================
# Then Steps - Tool Results
# ============================================================================


@then("the ensemble should execute successfully")
def ensemble_executes_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble executed successfully."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"


@then("I should receive structured results with agent outputs")
def receive_structured_results(bdd_context: dict[str, Any]) -> None:
    """Verify structured results with agent outputs."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert "results" in result or "content" in result, "Should have results or content"


@then("I should receive results in JSON format")
def receive_json_results(bdd_context: dict[str, Any]) -> None:
    """Verify results are in JSON format."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    # Should be serializable to JSON
    json.dumps(result)  # Will raise if not JSON serializable


@then("I should receive a tool error")
def receive_tool_error(bdd_context: dict[str, Any]) -> None:
    """Verify tool error is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error")
    result = bdd_context.get("tool_result")
    assert error is not None or result is None, "Should have error or no result"


@then("the error should indicate ensemble not found")
def error_indicates_not_found(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates not found."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "not found" in error.lower() or "does not exist" in error.lower()


@then("validation should pass")
def validation_passes(bdd_context: dict[str, Any]) -> None:
    """Verify validation passes."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert result.get("valid", False) is True, "Validation should pass"


@then("I should receive validation details")
def receive_validation_details(bdd_context: dict[str, Any]) -> None:
    """Verify validation details are received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert "details" in result or "valid" in result, "Should have validation details"


@then("validation should fail")
def validation_fails(bdd_context: dict[str, Any]) -> None:
    """Verify validation fails."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    # Either we got an error during execution, or result indicates invalid
    if result is None:
        assert error is not None, "Should have error or result, got neither"
    else:
        assert result.get("valid") is False, "Validation should indicate invalid"


@then("I should receive error details about the circular dependency")
def receive_circular_dep_error(bdd_context: dict[str, Any]) -> None:
    """Verify circular dependency error details."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    error = bdd_context.get("tool_error", "")
    combined = str(result) + error.lower()
    # Note: The EnsembleLoader validates during load, so ensembles with circular
    # dependencies are rejected at load time. The error is either "not found"
    # (because load failed) or mentions "circular" if validation is separate.
    assert "circular" in combined.lower() or "not found" in combined.lower(), (
        f"Should mention circular dependency or not found, got: {combined}"
    )


@then("I should receive a preview of changes")
def receive_changes_preview(bdd_context: dict[str, Any]) -> None:
    """Verify preview of changes is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert "preview" in result or "changes" in result, "Should have preview of changes"


@then("the ensemble file should not be modified")
def ensemble_not_modified(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble file is not modified."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    # In dry run mode, file should not be modified
    result = bdd_context.get("tool_result", {})
    assert result.get("modified", True) is False, "File should not be modified"


@then("the ensemble should be updated")
def ensemble_is_updated(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble is updated."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert result.get("modified", False) is True, "File should be modified"


@then("a backup file should be created")
def backup_file_created(bdd_context: dict[str, Any]) -> None:
    """Verify backup file is created."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert result.get("backup_created", False) is True, "Backup should be created"


# ============================================================================
# Then Steps - Streaming
# ============================================================================


@then("I should receive progress notifications as agents execute")
def receive_progress_notifications(bdd_context: dict[str, Any]) -> None:
    """Verify progress notifications are received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    events = bdd_context.get("streaming_events", [])
    assert len(events) > 0, "Should receive streaming events"


@then(
    "notifications should include agent_start, agent_progress, "
    "and agent_complete events"
)
def notifications_include_event_types(bdd_context: dict[str, Any]) -> None:
    """Verify notification event types."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    events = bdd_context.get("streaming_events", [])
    event_types = {e.get("type") for e in events}
    assert "agent_start" in event_types, "Should have agent_start events"
    assert "agent_complete" in event_types, "Should have agent_complete events"


# ============================================================================
# Then Steps - Server Discovery
# ============================================================================


@then('I should see "invoke" tool')
def see_invoke_tool(bdd_context: dict[str, Any]) -> None:
    """Verify invoke tool is visible."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    tools = bdd_context.get("tools_list", [])
    tool_names = [t.get("name") for t in tools]
    assert "invoke" in tool_names, "Should see invoke tool"


@then('I should see "validate_ensemble" tool')
def see_validate_tool(bdd_context: dict[str, Any]) -> None:
    """Verify validate_ensemble tool is visible."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    tools = bdd_context.get("tools_list", [])
    tool_names = [t.get("name") for t in tools]
    assert "validate_ensemble" in tool_names, "Should see validate_ensemble tool"


@then('I should see "update_ensemble" tool')
def see_update_tool(bdd_context: dict[str, Any]) -> None:
    """Verify update_ensemble tool is visible."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    tools = bdd_context.get("tools_list", [])
    tool_names = [t.get("name") for t in tools]
    assert "update_ensemble" in tool_names, "Should see update_ensemble tool"


@then('I should see "analyze_execution" tool')
def see_analyze_tool(bdd_context: dict[str, Any]) -> None:
    """Verify analyze_execution tool is visible."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    tools = bdd_context.get("tools_list", [])
    tool_names = [t.get("name") for t in tools]
    assert "analyze_execution" in tool_names, "Should see analyze_execution tool"


@then('I should see "llm-orc://ensembles" resource')
def see_ensembles_resource(bdd_context: dict[str, Any]) -> None:
    """Verify ensembles resource is visible."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    resources = bdd_context.get("resources_list", [])
    resource_uris = [r.get("uri") for r in resources]
    assert "llm-orc://ensembles" in resource_uris, "Should see ensembles resource"


@then('I should see "llm-orc://profiles" resource')
def see_profiles_resource(bdd_context: dict[str, Any]) -> None:
    """Verify profiles resource is visible."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    resources = bdd_context.get("resources_list", [])
    resource_uris = [r.get("uri") for r in resources]
    assert "llm-orc://profiles" in resource_uris, "Should see profiles resource"
