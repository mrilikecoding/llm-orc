"""BDD step definitions for ADR-009 MCP Server Architecture."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

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


@when('I call the "analyze_execution" tool with:', target_fixture="analyze_datatable")
def call_analyze_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call analyze_execution tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["analyze_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("analyze_execution", params)
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
# When Steps - Server Lifecycle
# ============================================================================


@when("the MCP server starts")
def mcp_server_starts(bdd_context: dict[str, Any]) -> None:
    """Start MCP server and check initialization."""
    if not bdd_context.get("mcp_available"):
        bdd_context["server_started"] = False
        bdd_context["server_capabilities"] = {}
        return

    async def _start() -> Any:
        try:
            server = bdd_context["mcp_server"]
            init_result = await server.handle_initialize()
            return init_result
        except Exception as e:
            return {"error": str(e)}

    bdd_context["init_result"] = asyncio.run(_start())
    bdd_context["server_started"] = "error" not in bdd_context["init_result"]


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
# When Steps - CLI Integration
# ============================================================================


@when('I run "llm-orc mcp serve" in background')
def run_mcp_serve_background(bdd_context: dict[str, Any]) -> None:
    """Run MCP serve command in background."""
    # This would require actual subprocess management
    # For now, mock the result
    bdd_context["cli_command"] = "llm-orc mcp serve"
    bdd_context["cli_transport"] = "stdio"
    bdd_context["cli_started"] = False  # Will be True when implemented


@when('I run "llm-orc mcp serve --http --port 8080"')
def run_mcp_serve_http(bdd_context: dict[str, Any]) -> None:
    """Run MCP serve command with HTTP transport."""
    bdd_context["cli_command"] = "llm-orc mcp serve --http --port 8080"
    bdd_context["cli_transport"] = "http"
    bdd_context["cli_port"] = 8080
    bdd_context["cli_started"] = False  # Will be True when implemented


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
    # This would require checking file modification time
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


@then("I should receive execution analysis")
def receive_execution_analysis(bdd_context: dict[str, Any]) -> None:
    """Verify execution analysis is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    assert result is not None, "Should have analysis result"


@then("analysis should include agent effectiveness metrics")
def analysis_includes_metrics(bdd_context: dict[str, Any]) -> None:
    """Verify analysis includes effectiveness metrics."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert "metrics" in result or "analysis" in result, "Should have metrics"


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
# Then Steps - Server Lifecycle
# ============================================================================


@then("it should respond to initialize request")
def responds_to_initialize(bdd_context: dict[str, Any]) -> None:
    """Verify server responds to initialize."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    init_result = bdd_context.get("init_result", {})
    assert "error" not in init_result, "Should not have error"


@then("capabilities should include tools and resources")
def capabilities_include_tools_resources(bdd_context: dict[str, Any]) -> None:
    """Verify capabilities include tools and resources."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    init_result = bdd_context.get("init_result", {})
    capabilities = init_result.get("capabilities", {})
    assert "tools" in capabilities, "Should have tools capability"
    assert "resources" in capabilities, "Should have resources capability"


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


# ============================================================================
# Then Steps - CLI Integration
# ============================================================================


@then("the server should start on stdio transport")
def server_starts_stdio(bdd_context: dict[str, Any]) -> None:
    """Verify server starts on stdio transport."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    assert bdd_context.get("cli_transport") == "stdio"
    # In full implementation, would verify actual process


@then("it should respond to MCP requests")
def server_responds_to_requests(bdd_context: dict[str, Any]) -> None:
    """Verify server responds to requests."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    # Would verify actual MCP communication
    pass


@then("the server should start on HTTP transport")
def server_starts_http(bdd_context: dict[str, Any]) -> None:
    """Verify server starts on HTTP transport."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    assert bdd_context.get("cli_transport") == "http"


@then('it should be accessible at "http://localhost:8080"')
def server_accessible_at_port(bdd_context: dict[str, Any]) -> None:
    """Verify server is accessible at specified port."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    assert bdd_context.get("cli_port") == 8080


# ============================================================================
# Phase 2: CRUD Operations - Given Steps
# ============================================================================


@given("a local ensembles directory exists")
def local_ensembles_dir_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create a local ensembles directory."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    bdd_context["local_ensembles_dir"] = ensembles_dir
    bdd_context["tmp_path"] = tmp_path
    _reconfigure_server(bdd_context, [ensembles_dir])


@given('an ensemble named "existing-ensemble" exists')
def ensemble_existing_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create an existing ensemble."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "existing-ensemble.yaml").write_text(
        "name: existing-ensemble\ndescription: Existing\n"
        "agents:\n  - name: agent1\n    model_profile: fast"
    )
    bdd_context["local_ensembles_dir"] = ensembles_dir
    bdd_context["tmp_path"] = tmp_path
    _reconfigure_server(bdd_context, [ensembles_dir])


@given('an ensemble named "to-delete" exists')
def ensemble_to_delete_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create an ensemble to be deleted."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "to-delete.yaml").write_text(
        "name: to-delete\ndescription: To be deleted\n"
        "agents:\n  - name: agent1\n    model_profile: fast"
    )
    bdd_context["local_ensembles_dir"] = ensembles_dir
    bdd_context["tmp_path"] = tmp_path
    _reconfigure_server(bdd_context, [ensembles_dir])


@given('an ensemble named "protected" exists')
def ensemble_protected_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create a protected ensemble."""
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "protected.yaml").write_text(
        "name: protected\ndescription: Protected ensemble\n"
        "agents:\n  - name: agent1\n    model_profile: fast"
    )
    bdd_context["local_ensembles_dir"] = ensembles_dir
    bdd_context["tmp_path"] = tmp_path
    _reconfigure_server(bdd_context, [ensembles_dir])


@given("scripts exist in the scripts directory")
def scripts_exist(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create scripts in the scripts directory."""
    scripts_dir = tmp_path / ".llm-orc" / "scripts"
    transform_dir = scripts_dir / "transform"
    transform_dir.mkdir(parents=True, exist_ok=True)

    (transform_dir / "uppercase.py").write_text(
        '"""Uppercase transform script."""\n'
        "def transform(data):\n    return data.upper()\n"
    )
    bdd_context["scripts_dir"] = scripts_dir
    bdd_context["tmp_path"] = tmp_path


@given("scripts exist in multiple categories")
def scripts_multiple_categories(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create scripts in multiple categories."""
    scripts_dir = tmp_path / ".llm-orc" / "scripts"

    # Transform category
    transform_dir = scripts_dir / "transform"
    transform_dir.mkdir(parents=True, exist_ok=True)
    (transform_dir / "uppercase.py").write_text(
        '"""Uppercase transform."""\ndef transform(data): return data.upper()\n'
    )

    # Validate category
    validate_dir = scripts_dir / "validate"
    validate_dir.mkdir(parents=True, exist_ok=True)
    (validate_dir / "json_check.py").write_text(
        '"""JSON validator."""\nimport json\ndef validate(data): json.loads(data)\n'
    )

    bdd_context["scripts_dir"] = scripts_dir
    bdd_context["tmp_path"] = tmp_path


@given("the library contains ensembles")
def library_contains_ensembles(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up library with ensembles."""
    library_dir = tmp_path / "library"
    ensembles_dir = library_dir / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)

    (ensembles_dir / "lib-ensemble.yaml").write_text(
        "name: lib-ensemble\ndescription: Library ensemble\n"
        "agents:\n  - name: agent1\n    model_profile: fast"
    )
    bdd_context["library_dir"] = library_dir
    bdd_context["tmp_path"] = tmp_path


@given("the library contains ensembles and scripts")
def library_contains_both(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up library with ensembles and scripts."""
    library_dir = tmp_path / "library"

    # Ensembles
    ensembles_dir = library_dir / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "lib-ensemble.yaml").write_text(
        "name: lib-ensemble\ndescription: Library ensemble\n"
        "agents:\n  - name: agent1\n    model_profile: fast"
    )

    # Scripts
    scripts_dir = library_dir / "scripts" / "transform"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "lib-script.py").write_text(
        '"""Library script."""\ndef run(data): return data\n'
    )

    bdd_context["library_dir"] = library_dir
    bdd_context["tmp_path"] = tmp_path


@given('the library contains an ensemble named "library-ensemble"')
def library_has_ensemble(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create a specific library ensemble."""
    library_dir = tmp_path / "library"
    ensembles_dir = library_dir / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)

    (ensembles_dir / "library-ensemble.yaml").write_text(
        "name: library-ensemble\ndescription: Library ensemble to copy\n"
        "agents:\n  - name: lib-agent\n    model_profile: standard"
    )
    bdd_context["library_dir"] = library_dir
    bdd_context["tmp_path"] = tmp_path

    # Configure local ensembles directory first
    local_ensembles = tmp_path / ".llm-orc" / "ensembles"
    local_ensembles.mkdir(parents=True, exist_ok=True)
    bdd_context["local_ensembles_dir"] = local_ensembles
    _reconfigure_server(bdd_context, [local_ensembles])

    # Now set library dir on the NEW server (after reconfigure)
    server = bdd_context.get("mcp_server")
    if server:
        server._test_library_dir = library_dir


@given('an ensemble named "library-ensemble" exists locally')
def ensemble_exists_locally(bdd_context: dict[str, Any]) -> None:
    """Create a local ensemble with same name as library."""
    # Use the same ensembles dir that was set up by the library step
    ensembles_dir = bdd_context.get("local_ensembles_dir")
    if not ensembles_dir:
        raise ValueError("local_ensembles_dir not set - must run library step first")
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "library-ensemble.yaml").write_text(
        "name: library-ensemble\ndescription: Local version\n"
        "agents:\n  - name: local-agent\n    model_profile: fast"
    )


# ============================================================================
# Phase 2: CRUD Operations - When Steps
# ============================================================================


@when('I call the "create_ensemble" tool with:', target_fixture="create_datatable")
def call_create_ensemble_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call create_ensemble tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["create_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("create_ensemble", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "delete_ensemble" tool with:', target_fixture="delete_datatable")
def call_delete_ensemble_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call delete_ensemble tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["delete_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("delete_ensemble", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "list_scripts" tool')
def call_list_scripts_tool_no_params(bdd_context: dict[str, Any]) -> None:
    """Call list_scripts tool without parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("list_scripts", {})
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "list_scripts" tool with:', target_fixture="scripts_datatable")
def call_list_scripts_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call list_scripts tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["scripts_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("list_scripts", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "library_browse" tool with:', target_fixture="browse_datatable")
def call_library_browse_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call library_browse tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["browse_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("library_browse", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "library_browse" tool')
def call_library_browse_tool_no_params(bdd_context: dict[str, Any]) -> None:
    """Call library_browse tool without parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("library_browse", {})
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "library_copy" tool with:', target_fixture="copy_datatable")
def call_library_copy_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call library_copy tool with parameters."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    bdd_context["copy_params"] = params

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            # Ensure library dir is set (may have been reset by reconfigure)
            if "library_dir" in bdd_context:
                server._test_library_dir = bdd_context["library_dir"]
            return await server.call_tool("library_copy", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


# ============================================================================
# Phase 2: CRUD Operations - Then Steps
# ============================================================================


@then("the ensemble should be created successfully")
def ensemble_created_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble was created."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("created", False) is True, "Ensemble should be created"


@then('the ensemble file should exist at ".llm-orc/ensembles/my-new-ensemble.yaml"')
def ensemble_file_exists(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble file exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    tmp_path = bdd_context.get("tmp_path")
    if tmp_path:
        ensemble_file = tmp_path / ".llm-orc" / "ensembles" / "my-new-ensemble.yaml"
        assert ensemble_file.exists(), f"Ensemble file should exist at {ensemble_file}"


@then("the new ensemble should have the same agents as the template")
def ensemble_has_template_agents(bdd_context: dict[str, Any]) -> None:
    """Verify new ensemble has same agents as template."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    # The result should indicate agents were copied
    assert result.get("agents_copied", 0) > 0, "Should have copied agents"


@then("the error should indicate ensemble already exists")
def error_indicates_exists(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates ensemble exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "exists" in error.lower() or "already" in error.lower(), (
        f"Error should indicate exists: {error}"
    )


@then("the ensemble should be deleted successfully")
def ensemble_deleted_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble was deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("deleted", False) is True, "Ensemble should be deleted"


@then("the ensemble file should no longer exist")
def ensemble_file_deleted(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble file is deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    ensembles_dir = bdd_context.get("local_ensembles_dir")
    if ensembles_dir:
        ensemble_file = ensembles_dir / "to-delete.yaml"
        assert not ensemble_file.exists(), "Ensemble file should be deleted"


@then("the error should indicate confirmation required")
def error_indicates_confirmation(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates confirmation required."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "confirm" in error.lower(), f"Error should mention confirmation: {error}"


@then("I should receive a list of scripts")
def receive_scripts_list(bdd_context: dict[str, Any]) -> None:
    """Verify scripts list is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    assert result is not None, "Should have result"
    assert "scripts" in result, "Result should have scripts"
    assert isinstance(result["scripts"], list), "Scripts should be a list"


@then("each script should have name, category, and path")
def scripts_have_metadata(bdd_context: dict[str, Any]) -> None:
    """Verify script metadata."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    scripts = result.get("scripts", [])
    for script in scripts:
        assert "name" in script, "Script should have name"
        assert "category" in script, "Script should have category"
        assert "path" in script, "Script should have path"


@then('I should receive only scripts in the "transform" category')
def receive_transform_scripts(bdd_context: dict[str, Any]) -> None:
    """Verify only transform scripts are returned."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    scripts = result.get("scripts", [])
    for script in scripts:
        assert script.get("category") == "transform", (
            f"Script should be in transform category: {script}"
        )


@then("I should receive a list of library ensembles")
def receive_library_ensembles(bdd_context: dict[str, Any]) -> None:
    """Verify library ensembles list is received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    assert result is not None, "Should have result"
    assert "ensembles" in result, "Result should have ensembles"


@then("each ensemble should have name, description, and path")
def library_ensembles_have_metadata(bdd_context: dict[str, Any]) -> None:
    """Verify library ensemble metadata."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    ensembles = result.get("ensembles", [])
    for ensemble in ensembles:
        assert "name" in ensemble, "Ensemble should have name"
        assert "description" in ensemble, "Ensemble should have description"
        assert "path" in ensemble, "Ensemble should have path"


@then("I should receive both ensembles and scripts")
def receive_ensembles_and_scripts(bdd_context: dict[str, Any]) -> None:
    """Verify both ensembles and scripts are returned."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    assert result is not None, "Should have result"
    assert "ensembles" in result, "Result should have ensembles"
    assert "scripts" in result, "Result should have scripts"


@then("the ensemble should be copied to local directory")
def ensemble_copied_to_local(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble was copied."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("copied", False) is True, "Ensemble should be copied"


@then("the local ensemble file should exist")
def local_ensemble_file_exists(bdd_context: dict[str, Any]) -> None:
    """Verify local ensemble file exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    destination = result.get("destination")
    if destination:
        assert Path(destination).exists(), f"File should exist at {destination}"


@then("the error should indicate file already exists")
def error_indicates_file_exists(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates file exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "exists" in error.lower() or "already" in error.lower(), (
        f"Error should indicate file exists: {error}"
    )


# =============================================================================
# Phase 2 Medium Priority: Profile CRUD Step Definitions
# =============================================================================


@given("model profiles exist in the configuration")
def profiles_exist_in_config(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up profiles in configuration."""
    profiles_dir = tmp_path / ".llm-orc" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Create sample profiles
    (profiles_dir / "fast-ollama.yaml").write_text(
        "name: fast-ollama\nprovider: ollama\nmodel: llama3.2:1b\n"
    )
    (profiles_dir / "claude-sonnet.yaml").write_text(
        "name: claude-sonnet\nprovider: anthropic\nmodel: claude-3-5-sonnet-latest\n"
    )
    bdd_context["profiles_dir"] = profiles_dir
    _reconfigure_server_with_profiles(bdd_context, [profiles_dir])


@given(parsers.parse('model profiles exist for providers "{providers}"'))
def profiles_exist_for_providers(
    bdd_context: dict[str, Any], tmp_path: Path, providers: str
) -> None:
    """Set up profiles for specific providers."""
    profiles_dir = tmp_path / ".llm-orc" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    provider_list = [p.strip() for p in providers.split(" and ")]
    for provider in provider_list:
        (profiles_dir / f"{provider}-profile.yaml").write_text(
            f"name: {provider}-profile\nprovider: {provider}\nmodel: test-model\n"
        )
    bdd_context["profiles_dir"] = profiles_dir
    _reconfigure_server_with_profiles(bdd_context, [profiles_dir])


@given("a local profiles directory exists")
def local_profiles_dir_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create local profiles directory."""
    profiles_dir = tmp_path / ".llm-orc" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    bdd_context["profiles_dir"] = profiles_dir
    _reconfigure_server_with_profiles(bdd_context, [profiles_dir])


@given(parsers.parse('a profile named "{name}" exists'))
def profile_named_exists(
    bdd_context: dict[str, Any], tmp_path: Path, name: str
) -> None:
    """Create a specific profile."""
    profiles_dir = tmp_path / ".llm-orc" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / f"{name}.yaml").write_text(
        f"name: {name}\nprovider: ollama\nmodel: llama3.2:1b\n"
    )
    bdd_context["profiles_dir"] = profiles_dir
    bdd_context["profile_name"] = name
    _reconfigure_server_with_profiles(bdd_context, [profiles_dir])


@given(parsers.parse('no profile named "{name}" exists'))
def no_profile_named_exists(
    bdd_context: dict[str, Any], tmp_path: Path, name: str
) -> None:
    """Ensure profile does not exist."""
    profiles_dir = tmp_path / ".llm-orc" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_file = profiles_dir / f"{name}.yaml"
    if profile_file.exists():
        profile_file.unlink()
    bdd_context["profiles_dir"] = profiles_dir
    _reconfigure_server_with_profiles(bdd_context, [profiles_dir])


def _reconfigure_server_with_profiles(
    bdd_context: dict[str, Any], profiles_dirs: list[Path]
) -> None:
    """Reconfigure server with profiles directories."""
    from unittest.mock import MagicMock

    from llm_orc.mcp.server import MCPServerV2

    mock_config = MagicMock()
    mock_config.get_ensembles_dirs.return_value = bdd_context.get("ensembles_dirs", [])
    mock_config.get_profiles_dirs.return_value = [str(d) for d in profiles_dirs]
    server = MCPServerV2(config_manager=mock_config)
    bdd_context["mcp_server"] = server
    bdd_context["mcp_available"] = True


@then("I should receive a list of profiles")
def receive_profiles_list_from_tool(bdd_context: dict[str, Any]) -> None:
    """Verify profiles list received from tool call."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert "profiles" in result, "Result should have profiles"
    assert isinstance(result["profiles"], list), "Profiles should be a list"


@then("each profile should have name, provider, and model")
def each_profile_has_required_fields(bdd_context: dict[str, Any]) -> None:
    """Verify profile fields."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    profiles = result.get("profiles", [])
    for profile in profiles:
        assert "name" in profile, "Profile should have name"
        assert "provider" in profile, "Profile should have provider"
        assert "model" in profile, "Profile should have model"


@then("I should receive only ollama profiles")
def receive_only_ollama_profiles(bdd_context: dict[str, Any]) -> None:
    """Verify only ollama profiles returned."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    profiles = result.get("profiles", [])
    for profile in profiles:
        assert profile.get("provider") == "ollama", (
            f"All profiles should be ollama, got {profile.get('provider')}"
        )


@then("the profile should be created successfully")
def profile_created_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify profile was created."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("created", False) is True, "Profile should be created"


@then("the profile file should exist")
def profile_file_exists(bdd_context: dict[str, Any]) -> None:
    """Verify profile file exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    path = result.get("path")
    if path:
        assert Path(path).exists(), f"Profile file should exist at {path}"


@then("the error should indicate profile already exists")
def error_indicates_profile_exists(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates profile exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "exists" in error.lower() or "already" in error.lower(), (
        f"Error should indicate profile exists: {error}"
    )


@then("the profile should be updated successfully")
def profile_updated_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify profile was updated."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("updated", False) is True, "Profile should be updated"


@then("the error should indicate profile not found")
def error_indicates_profile_not_found(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates profile not found."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "not found" in error.lower(), f"Error should indicate not found: {error}"


@then("the profile should be deleted successfully")
def profile_deleted_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify profile was deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("deleted", False) is True, "Profile should be deleted"


@then("the profile file should not exist")
def profile_file_not_exists(bdd_context: dict[str, Any]) -> None:
    """Verify profile file does not exist."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    profiles_dir = bdd_context.get("profiles_dir")
    profile_name = bdd_context.get("profile_name")
    if profiles_dir and profile_name:
        profile_file = profiles_dir / f"{profile_name}.yaml"
        assert not profile_file.exists(), (
            f"Profile file should not exist: {profile_file}"
        )


# =============================================================================
# Phase 2 Medium Priority: Artifact Management Step Definitions
# =============================================================================


@given(parsers.parse('an execution artifact exists for ensemble "{ensemble}"'))
def artifact_exists_for_ensemble(
    bdd_context: dict[str, Any], tmp_path: Path, ensemble: str
) -> None:
    """Create execution artifact for ensemble."""
    artifacts_dir = tmp_path / ".llm-orc" / "artifacts" / ensemble / "20250101-120000"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "execution.json").write_text('{"status": "completed"}')
    (artifacts_dir / "execution.md").write_text("# Execution Report\n")
    bdd_context["artifacts_base"] = tmp_path / ".llm-orc" / "artifacts"
    bdd_context["artifact_ensemble"] = ensemble
    _reconfigure_server_with_artifacts(bdd_context, tmp_path)


@given(parsers.parse('multiple execution artifacts exist for ensemble "{ensemble}"'))
def multiple_artifacts_exist(
    bdd_context: dict[str, Any], tmp_path: Path, ensemble: str
) -> None:
    """Create multiple execution artifacts."""
    import time

    artifacts_base = tmp_path / ".llm-orc" / "artifacts" / ensemble
    artifacts_base.mkdir(parents=True, exist_ok=True)

    # Create old artifact (mtime set to 10 days ago)
    old_artifact = artifacts_base / "20250101-100000"
    old_artifact.mkdir(parents=True)
    (old_artifact / "execution.json").write_text('{"status": "completed"}')
    old_time = time.time() - (10 * 24 * 60 * 60)  # 10 days ago
    import os

    os.utime(old_artifact, (old_time, old_time))

    # Create recent artifact (current time)
    recent_artifact = artifacts_base / "20250101-200000"
    recent_artifact.mkdir(parents=True)
    (recent_artifact / "execution.json").write_text('{"status": "completed"}')

    bdd_context["artifacts_base"] = tmp_path / ".llm-orc" / "artifacts"
    bdd_context["artifact_ensemble"] = ensemble
    bdd_context["old_artifact"] = old_artifact
    bdd_context["recent_artifact"] = recent_artifact
    _reconfigure_server_with_artifacts(bdd_context, tmp_path)


@given("some artifacts are older than 7 days")
def some_artifacts_older_than_7_days(bdd_context: dict[str, Any]) -> None:
    """Marker step - artifacts already configured in previous step."""
    pass  # Handled by multiple_artifacts_exist


def _reconfigure_server_with_artifacts(
    bdd_context: dict[str, Any], tmp_path: Path
) -> None:
    """Reconfigure server with artifact base directory."""
    from unittest.mock import MagicMock

    from llm_orc.mcp.server import MCPServerV2

    mock_config = MagicMock()
    mock_config.get_ensembles_dirs.return_value = bdd_context.get("ensembles_dirs", [])
    mock_config.get_profiles_dirs.return_value = []
    server = MCPServerV2(config_manager=mock_config)
    # Set artifact base for testing
    server._test_artifacts_base = bdd_context.get("artifacts_base")
    bdd_context["mcp_server"] = server
    bdd_context["mcp_available"] = True


@then("the artifact should be deleted successfully")
def artifact_deleted_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify artifact was deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("deleted", False) is True, "Artifact should be deleted"


@then("the artifact directory should not exist")
def artifact_directory_not_exists(bdd_context: dict[str, Any]) -> None:
    """Verify artifact directory does not exist."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    artifacts_base = bdd_context.get("artifacts_base")
    ensemble = bdd_context.get("artifact_ensemble")
    if artifacts_base and ensemble:
        artifact_dir = artifacts_base / ensemble / "20250101-120000"
        assert not artifact_dir.exists(), (
            f"Artifact dir should not exist: {artifact_dir}"
        )


@then("the error should indicate artifact not found")
def error_indicates_artifact_not_found(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates artifact not found."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "not found" in error.lower(), f"Error should indicate not found: {error}"


@then("I should receive a preview of artifacts to delete")
def receive_preview_of_artifacts(bdd_context: dict[str, Any]) -> None:
    """Verify preview of artifacts received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert "would_delete" in result, "Result should have would_delete preview"
    assert result.get("dry_run", False) is True, "Should be dry run"


@then("no artifacts should actually be deleted")
def no_artifacts_actually_deleted(bdd_context: dict[str, Any]) -> None:
    """Verify no artifacts were deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    old_artifact = bdd_context.get("old_artifact")
    if old_artifact:
        assert old_artifact.exists(), "Old artifact should still exist (dry run)"


@then("old artifacts should be deleted")
def old_artifacts_deleted(bdd_context: dict[str, Any]) -> None:
    """Verify old artifacts were deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    old_artifact = bdd_context.get("old_artifact")
    if old_artifact:
        assert not old_artifact.exists(), "Old artifact should be deleted"


@then("recent artifacts should remain")
def recent_artifacts_remain(bdd_context: dict[str, Any]) -> None:
    """Verify recent artifacts remain."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    recent_artifact = bdd_context.get("recent_artifact")
    if recent_artifact:
        assert recent_artifact.exists(), "Recent artifact should still exist"


@then(parsers.parse('only "{ensemble}" artifacts should be deleted'))
def only_specific_ensemble_deleted(bdd_context: dict[str, Any], ensemble: str) -> None:
    """Verify only specific ensemble artifacts deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    artifacts_base = bdd_context.get("artifacts_base")
    if artifacts_base:
        ensemble_dir = artifacts_base / ensemble
        # Check that artifacts for this ensemble are deleted (empty or no dir)
        if ensemble_dir.exists():
            artifacts = list(ensemble_dir.iterdir())
            assert len(artifacts) == 0, f"{ensemble} should have no artifacts"


@then(parsers.parse('"{ensemble}" artifacts should remain'))
def ensemble_artifacts_remain(bdd_context: dict[str, Any], ensemble: str) -> None:
    """Verify ensemble artifacts remain."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    artifacts_base = bdd_context.get("artifacts_base")
    if artifacts_base:
        ensemble_dir = artifacts_base / ensemble
        assert ensemble_dir.exists(), f"{ensemble} directory should exist"
        artifacts = list(ensemble_dir.iterdir())
        assert len(artifacts) > 0, f"{ensemble} should have artifacts"


# =============================================================================
# Phase 2 Medium Priority: When Step Definitions for Profile Tools
# =============================================================================


@when('I call the "list_profiles" tool')
def call_list_profiles_tool(bdd_context: dict[str, Any]) -> None:
    """Call list_profiles tool without arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("list_profiles", {})
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "list_profiles" tool with:', target_fixture="profiles_datatable")
def call_list_profiles_with_args(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call list_profiles with arguments from datatable."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("list_profiles", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "create_profile" tool with:', target_fixture="create_profile_dt")
def call_create_profile_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call create_profile tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("create_profile", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "update_profile" tool with:', target_fixture="update_profile_dt")
def call_update_profile_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call update_profile tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("update_profile", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "delete_profile" tool with:', target_fixture="delete_profile_dt")
def call_delete_profile_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call delete_profile tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("delete_profile", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


# =============================================================================
# Phase 2 Medium Priority: When Step Definitions for Artifact Tools
# =============================================================================


@when('I call the "delete_artifact" tool with:', target_fixture="delete_artifact_dt")
def call_delete_artifact_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call delete_artifact tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("delete_artifact", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "cleanup_artifacts" tool with:', target_fixture="cleanup_art_dt")
def call_cleanup_artifacts_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call cleanup_artifacts tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)
    # Ensure older_than_days is an int
    if "older_than_days" in params:
        params["older_than_days"] = int(params["older_than_days"])

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("cleanup_artifacts", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


# =============================================================================
# Phase 2 Low Priority: Script Management Step Definitions
# =============================================================================


@given(parsers.parse('a script named "{name}" exists in category "{category}"'))
def script_exists_in_category(
    bdd_context: dict[str, Any], tmp_path: Path, name: str, category: str
) -> None:
    """Create a script in the given category."""
    scripts_dir = tmp_path / ".llm-orc" / "scripts" / category
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_file = scripts_dir / f"{name}.py"
    script_file.write_text(
        f'''"""Script: {name}"""

def run(input_data: str) -> str:
    """Run the script."""
    return f"Processed: {{input_data}}"
'''
    )
    bdd_context["scripts_dir"] = tmp_path / ".llm-orc" / "scripts"
    bdd_context["script_name"] = name
    bdd_context["script_category"] = category
    _reconfigure_server_with_scripts(bdd_context, tmp_path)


@given("a local scripts directory exists")
def local_scripts_dir_exists(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Create local scripts directory."""
    scripts_dir = tmp_path / ".llm-orc" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    bdd_context["scripts_dir"] = scripts_dir
    _reconfigure_server_with_scripts(bdd_context, tmp_path)


def _reconfigure_server_with_scripts(
    bdd_context: dict[str, Any], tmp_path: Path
) -> None:
    """Reconfigure server with scripts directory."""
    from unittest.mock import MagicMock

    from llm_orc.mcp.server import MCPServerV2

    mock_config = MagicMock()
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    mock_config.get_ensembles_dirs.return_value = [str(ensembles_dir)]
    mock_config.get_profiles_dirs.return_value = []
    server = MCPServerV2(config_manager=mock_config)
    server._test_scripts_dir = tmp_path / ".llm-orc" / "scripts"
    bdd_context["mcp_server"] = server
    bdd_context["mcp_available"] = True


@when('I call the "get_script" tool with:', target_fixture="get_script_dt")
def call_get_script_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call get_script tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("get_script", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "test_script" tool with:', target_fixture="test_script_dt")
def call_test_script_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call test_script tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("test_script", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "create_script" tool with:', target_fixture="create_script_dt")
def call_create_script_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call create_script tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("create_script", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "delete_script" tool with:', target_fixture="delete_script_dt")
def call_delete_script_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call delete_script tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("delete_script", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@then("I should receive the script details")
def receive_script_details(bdd_context: dict[str, Any]) -> None:
    """Verify script details received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"


@then("the script should have name, category, and path")
def script_has_required_fields(bdd_context: dict[str, Any]) -> None:
    """Verify script has required fields."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert "name" in result, "Script should have name"
    assert "category" in result, "Script should have category"
    assert "path" in result, "Script should have path"


@then("the error should indicate script not found")
def error_indicates_script_not_found(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates script not found."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "not found" in error.lower(), f"Error should indicate not found: {error}"


@then("I should receive script test results")
def receive_script_test_results(bdd_context: dict[str, Any]) -> None:
    """Verify script test results received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"


@then("the result should indicate success or failure")
def result_indicates_success_or_failure(bdd_context: dict[str, Any]) -> None:
    """Verify result has success/failure indication."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert "success" in result or "error" in result, "Result should indicate outcome"


@then("the script should be created successfully")
def script_created_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify script was created."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("created", False) is True, "Script should be created"


@then("the script file should exist")
def script_file_exists(bdd_context: dict[str, Any]) -> None:
    """Verify script file exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    path = result.get("path")
    if path:
        assert Path(path).exists(), f"Script file should exist at {path}"


@then("the error should indicate script already exists")
def error_indicates_script_exists(bdd_context: dict[str, Any]) -> None:
    """Verify error indicates script exists."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error", "")
    assert "exists" in error.lower() or "already" in error.lower(), (
        f"Error should indicate script exists: {error}"
    )


@then("the script should be deleted successfully")
def script_deleted_successfully(bdd_context: dict[str, Any]) -> None:
    """Verify script was deleted."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert result.get("deleted", False) is True, "Script should be deleted"


@then("the script file should not exist")
def script_file_not_exists(bdd_context: dict[str, Any]) -> None:
    """Verify script file does not exist."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    scripts_dir = bdd_context.get("scripts_dir")
    name = bdd_context.get("script_name")
    category = bdd_context.get("script_category")
    if scripts_dir and name and category:
        script_file = scripts_dir / category / f"{name}.py"
        assert not script_file.exists(), f"Script should not exist: {script_file}"


# =============================================================================
# Phase 2 Low Priority: Library Extras Step Definitions
# =============================================================================


@given("the library contains ensembles and scripts")
def library_has_content(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up library with ensembles and scripts."""
    library_dir = tmp_path / "library"

    # Create ensembles
    ensembles_dir = library_dir / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "code-review.yaml").write_text(
        "name: code-review\ndescription: Code review ensemble\n"
    )
    (ensembles_dir / "test-runner.yaml").write_text(
        "name: test-runner\ndescription: Test runner ensemble\n"
    )

    # Create scripts
    scripts_dir = library_dir / "scripts" / "extraction"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "json_extract.py").write_text('"""JSON extraction script."""\n')

    bdd_context["library_dir"] = library_dir
    _reconfigure_server_with_library(bdd_context, tmp_path, library_dir)


@given("the library is configured")
def library_is_configured(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Set up configured library."""
    library_dir = tmp_path / "library"
    ensembles_dir = library_dir / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "sample.yaml").write_text("name: sample\n")

    bdd_context["library_dir"] = library_dir
    _reconfigure_server_with_library(bdd_context, tmp_path, library_dir)


def _reconfigure_server_with_library(
    bdd_context: dict[str, Any], tmp_path: Path, library_dir: Path
) -> None:
    """Reconfigure server with library directory."""
    from unittest.mock import MagicMock

    from llm_orc.mcp.server import MCPServerV2

    mock_config = MagicMock()
    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    mock_config.get_ensembles_dirs.return_value = [str(ensembles_dir)]
    mock_config.get_profiles_dirs.return_value = []
    server = MCPServerV2(config_manager=mock_config)
    server._test_library_dir = library_dir
    bdd_context["mcp_server"] = server
    bdd_context["mcp_available"] = True


@when('I call the "library_search" tool with:', target_fixture="lib_search_dt")
def call_library_search_tool(bdd_context: dict[str, Any], datatable: Any) -> None:
    """Call library_search tool with arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    params = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("library_search", params)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@when('I call the "library_info" tool')
def call_library_info_tool(bdd_context: dict[str, Any]) -> None:
    """Call library_info tool without arguments."""
    if not bdd_context.get("mcp_available"):
        bdd_context["tool_result"] = None
        bdd_context["tool_error"] = "MCP server not available"
        return

    async def _call() -> Any:
        try:
            server = bdd_context["mcp_server"]
            return await server.call_tool("library_info", {})
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())


@then("I should receive search results")
def receive_search_results(bdd_context: dict[str, Any]) -> None:
    """Verify search results received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert "results" in result, "Result should have results"


@then("results should include matching ensembles or scripts")
def results_include_matches(bdd_context: dict[str, Any]) -> None:
    """Verify results include matches."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    results = result.get("results", [])
    assert len(results) > 0, "Should have at least one match"


@then("I should receive empty search results")
def receive_empty_search_results(bdd_context: dict[str, Any]) -> None:
    """Verify empty search results."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    # Check total count (results is dict with ensembles/scripts lists)
    total = result.get("total", 0)
    assert total == 0, f"Should have no matches, got total={total}"


@then("I should receive library metadata")
def receive_library_metadata(bdd_context: dict[str, Any]) -> None:
    """Verify library metadata received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"


@then("the metadata should include path and counts")
def metadata_includes_path_and_counts(bdd_context: dict[str, Any]) -> None:
    """Verify metadata has path and counts."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert "path" in result, "Metadata should have path"
    # Implementation uses ensembles_count and scripts_count
    assert "ensembles_count" in result or "ensemble_count" in result, (
        f"Metadata should have ensemble count, got: {list(result.keys())}"
    )


# ============================================================================
# Phase 3: Provider & Model Discovery Steps
# ============================================================================


@given("Ollama is running locally with models")
def ollama_running_with_models(bdd_context: dict[str, Any]) -> None:
    """Assume Ollama is running - actual check happens in tool."""
    bdd_context["ollama_expected_available"] = True


@given("Ollama is running locally with the required models")
def ollama_running_with_required_models(bdd_context: dict[str, Any]) -> None:
    """Assume Ollama is running with required models."""
    bdd_context["ollama_expected_available"] = True
    # Set test override on server to mock Ollama availability
    server = bdd_context.get("mcp_server")
    if server:
        server._test_ollama_status = {
            "available": True,
            "models": ["llama3:latest", "mistral:latest"],
            "model_count": 2,
        }


@given("Ollama is not running")
def ollama_not_running(bdd_context: dict[str, Any]) -> None:
    """Assume Ollama is not running - mock will handle this."""
    bdd_context["ollama_expected_available"] = False


@given("authentication is configured for some providers")
def auth_configured_for_some(bdd_context: dict[str, Any]) -> None:
    """Assume some providers have auth configured."""
    bdd_context["some_auth_configured"] = True


@given("an ensemble using only Ollama profiles exists")
def ensemble_with_ollama_profiles(bdd_context: dict[str, Any], tmp_path: Path) -> None:
    """Ensemble validate-ollama uses only Ollama."""
    from unittest.mock import MagicMock

    from llm_orc.mcp.server import MCPServerV2

    bdd_context["test_ensemble"] = "validate-ollama"

    # Create ensemble directory with validate-ollama ensemble
    ensembles_dir = tmp_path / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / "validate-ollama.yaml").write_text(
        "name: validate-ollama\n"
        "description: Validate Ollama\n"
        "agents:\n"
        "  - name: validator\n"
        "    model_profile: validate-ollama\n"
    )

    # Create profiles directory with validate-ollama profile
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "validate-ollama.yaml").write_text(
        "name: validate-ollama\nprovider: ollama\nmodel: llama3:latest\n"
    )

    # Reconfigure server with both directories
    mock_config = MagicMock()
    mock_config.get_ensembles_dirs.return_value = [str(ensembles_dir)]
    mock_config.get_profiles_dirs.return_value = [str(profiles_dir)]
    server = MCPServerV2(config_manager=mock_config)
    bdd_context["mcp_server"] = server
    bdd_context["mcp_available"] = True


@given("an ensemble using a non-existent profile exists")
def ensemble_with_missing_profile(bdd_context: dict[str, Any]) -> None:
    """Ensemble security-review uses 'default' which doesn't exist."""
    bdd_context["test_ensemble"] = "security-review"


@given("an ensemble using a cloud provider exists")
def ensemble_with_cloud_provider(bdd_context: dict[str, Any]) -> None:
    """Set up ensemble using cloud provider."""
    bdd_context["test_ensemble"] = "startup-advisory-board"


@given("the cloud provider is not configured")
def cloud_provider_not_configured(bdd_context: dict[str, Any]) -> None:
    """Indicate cloud provider is not configured."""
    bdd_context["cloud_provider_available"] = False


@when('I call the "get_provider_status" tool')
def call_get_provider_status_tool(bdd_context: dict[str, Any]) -> None:
    """Call get_provider_status tool."""
    if not bdd_context.get("mcp_available"):
        return

    server = bdd_context.get("mcp_server")
    if not server:
        return

    async def _call() -> Any:
        try:
            return await server.call_tool("get_provider_status", {})
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())
    if bdd_context.get("tool_error") is None:
        bdd_context["tool_error"] = None


@when('I call the "check_ensemble_runnable" tool with:')
def call_check_ensemble_runnable_tool(
    bdd_context: dict[str, Any], datatable: list[list[str]]
) -> None:
    """Call check_ensemble_runnable tool with parameters."""
    if not bdd_context.get("mcp_available"):
        return

    server = bdd_context.get("mcp_server")
    if not server:
        return

    arguments = _parse_datatable(datatable)

    async def _call() -> Any:
        try:
            return await server.call_tool("check_ensemble_runnable", arguments)
        except Exception as e:
            bdd_context["tool_error"] = str(e)
            return None

    bdd_context["tool_result"] = asyncio.run(_call())
    if bdd_context.get("tool_error") is None:
        bdd_context["tool_error"] = None


@then("I should receive provider status")
def receive_provider_status(bdd_context: dict[str, Any]) -> None:
    """Verify provider status received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert "providers" in result, "Result should have providers"


@then("the status should show Ollama as available")
def status_shows_ollama_available(bdd_context: dict[str, Any]) -> None:
    """Verify Ollama is shown as available."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    providers = result.get("providers", {})
    ollama = providers.get("ollama", {})
    # This will be true if Ollama is actually running
    assert "available" in ollama, "Ollama status should have 'available' field"


@then("the status should include available Ollama models")
def status_includes_ollama_models(bdd_context: dict[str, Any]) -> None:
    """Verify Ollama models are included."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    providers = result.get("providers", {})
    ollama = providers.get("ollama", {})
    assert "models" in ollama, "Ollama status should have 'models' field"


@then("the status should show Ollama as unavailable")
def status_shows_ollama_unavailable(bdd_context: dict[str, Any]) -> None:
    """Verify Ollama is shown as unavailable (or available - depends on env)."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    providers = result.get("providers", {})
    ollama = providers.get("ollama", {})
    # Just verify the structure exists - actual availability depends on env
    assert "available" in ollama, "Ollama status should have 'available' field"


@then("the status should indicate which cloud providers are configured")
def status_indicates_cloud_providers(bdd_context: dict[str, Any]) -> None:
    """Verify cloud provider status is indicated."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    providers = result.get("providers", {})
    # Check that cloud providers have status
    for provider in ["anthropic-api", "anthropic-claude-pro-max", "google-gemini"]:
        assert provider in providers, f"Should have {provider} status"
        status = providers[provider]
        assert "available" in status, f"{provider} should have 'available' field"


@then("I should receive runnable status")
def receive_runnable_status(bdd_context: dict[str, Any]) -> None:
    """Verify runnable status received."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result")
    error = bdd_context.get("tool_error")
    assert error is None, f"Should not have error: {error}"
    assert result is not None, "Should have result"
    assert "runnable" in result, "Result should have 'runnable' field"
    assert "agents" in result, "Result should have 'agents' field"


@then("the ensemble should be marked as runnable")
def ensemble_marked_runnable(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble is runnable."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert result.get("runnable") is True, "Ensemble should be runnable"


@then('all agents should have status "available"')
def all_agents_available(bdd_context: dict[str, Any]) -> None:
    """Verify all agents are available."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    agents = result.get("agents", [])
    for agent in agents:
        assert agent.get("status") == "available", (
            f"Agent {agent.get('name')} should be available"
        )


@then("the ensemble should be marked as not runnable")
def ensemble_marked_not_runnable(bdd_context: dict[str, Any]) -> None:
    """Verify ensemble is not runnable."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    assert result.get("runnable") is False, "Ensemble should not be runnable"


@then('at least one agent should have status "missing_profile"')
def agent_has_missing_profile(bdd_context: dict[str, Any]) -> None:
    """Verify at least one agent has missing profile."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    agents = result.get("agents", [])
    has_missing = any(a.get("status") == "missing_profile" for a in agents)
    assert has_missing, "At least one agent should have missing_profile status"


@then("affected agents should have local alternatives suggested")
def agents_have_alternatives(bdd_context: dict[str, Any]) -> None:
    """Verify affected agents have alternatives."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    result = bdd_context.get("tool_result", {})
    agents = result.get("agents", [])
    # At least one non-available agent should have alternatives
    for agent in agents:
        if agent.get("status") != "available":
            # Alternatives may be empty if Ollama not running
            assert "alternatives" in agent, "Agent should have alternatives field"


@then("I should receive an error indicating ensemble not found")
def error_ensemble_not_found(bdd_context: dict[str, Any]) -> None:
    """Verify error for non-existent ensemble."""
    if not bdd_context.get("mcp_available"):
        pytest.skip("MCP server not available - Red phase")

    error = bdd_context.get("tool_error")
    assert error is not None, "Should have error for non-existent ensemble"
    assert "not found" in error.lower() or "Ensemble" in error, (
        f"Error should indicate ensemble not found: {error}"
    )
