"""The MCP ``invoke`` tool through ``/mcp`` over one real OrchestraService
on temp project, state and global dirs (Arc 5, Task 10 and its
integration items). Nothing here opens a connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.web.server import create_app
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _yaml,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_mcp import _ACCEPT_HEADERS, _initialize

_MARK = (
    "import json, sys\n"
    "sys.stdin.read()\n"
    'print(json.dumps({"success": True, "data": "hello"}))\n'
)

ToolCall = Callable[[str, dict[str, Any]], dict[str, Any]]


def _answer(
    client: TestClient, session_id: str, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """The JSON-RPC answer to a tool call. A streaming tool sends its
    progress notifications first, in the same event stream."""
    response = client.post(
        "/mcp",
        headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    messages = [
        json.loads(line[len("data:") :])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    (answer,) = [m for m in messages if m.get("id") == 9]
    return dict(answer)


@pytest.fixture
def tool(
    service: OrchestraService, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ToolCall]:
    """Call an MCP tool over ``/mcp`` on ``service``; the structured result."""
    monkeypatch.setattr(web_api, "_orchestra_service", service)
    with TestClient(create_app()) as client:
        _, session_id = _initialize(client)

        def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            body = _answer(client, session_id, name, arguments)
            assert "error" not in body, body
            result: dict[str, Any] = body["result"]["structuredContent"]
            return result

        yield call


def _raw_root(project: Path) -> None:
    scripts = project / ".llm-orc" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "say.py").write_text(_MARK)
    _yaml(
        project / ".llm-orc" / "ensembles" / "raw.yaml",
        {
            "name": "raw",
            "description": "raw",
            "raw_output": True,
            "agents": [{"name": "a", "script": "say.py"}],
        },
    )


class TestRawOutput:
    def test_the_mcp_result_carries_raw_output_as_rest_does(
        self, project: Path, service: OrchestraService, tool: ToolCall
    ) -> None:
        _raw_root(project)

        over_mcp = tool("invoke", {"ensemble_name": "raw", "input_data": "hi"})
        with TestClient(create_app()) as client:
            over_rest = client.post(
                "/api/ensembles/execute", json={"ensemble_name": "raw", "input": "hi"}
            ).json()

        assert over_rest["raw_output"] is True
        assert over_mcp["raw_output"] is True
