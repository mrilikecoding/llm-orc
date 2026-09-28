"""Tests for the MCP streamable HTTP mount at /mcp.

Proves the design in docs/plans/2026-09-16-mcp-in-serve.md: the serve
mounts the MCP tool set at /mcp, sharing one OrchestraService with the
REST API, with no auth (the tailnet is the boundary).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route

import llm_orc.web.api as web_api
from llm_orc.mcp.server import MCPServer
from llm_orc.web.server import create_app

_ACCEPT_HEADERS = {"Accept": "application/json, text/event-stream"}


@pytest.fixture(autouse=True)
def _reset_shared_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a fresh OrchestraService.

    It's a process-wide singleton (web/api/__init__.py); reset before
    every test so create_app() builds a fresh one instead of picking
    up project state left behind by an earlier test. create_app()
    builds its own MCPServer per call, so there's no session-manager
    reuse concern here.
    """
    monkeypatch.setattr(web_api, "_orchestra_service", None)


def _parse_rpc_body(response: Any) -> dict[str, Any]:
    """Parse a JSON-RPC response that may come back as SSE or JSON."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return dict(json.loads(line[len("data:") :].strip()))
        raise AssertionError(f"No data line in SSE body: {response.text!r}")
    return dict(response.json())


def _initialize(client: TestClient) -> tuple[dict[str, Any], str]:
    """Run the initialize + notifications/initialized handshake.

    Returns the initialize result body and the mcp-session-id for use
    on subsequent requests.
    """
    response = client.post(
        "/mcp",
        headers=_ACCEPT_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1"},
            },
        },
    )
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]

    initialized = client.post(
        "/mcp",
        headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert initialized.status_code == 202

    return _parse_rpc_body(response), session_id


def _call_tool(
    client: TestClient,
    session_id: str,
    name: str,
    arguments: dict[str, Any],
    rpc_id: int = 9,
) -> dict[str, Any]:
    """Call one MCP tool and return the parsed JSON-RPC body."""
    response = client.post(
        "/mcp",
        headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    return _parse_rpc_body(response)


class TestMcpMount:
    """POST /mcp speaks MCP JSON-RPC over the streamable HTTP transport."""

    def test_initialize_returns_server_info(self) -> None:
        with TestClient(create_app()) as client:
            result, _ = _initialize(client)

        assert result["result"]["serverInfo"]["name"] == "llm-orc"


class TestMcpToolParity:
    """tools/list over /mcp exposes the full registered tool set."""

    def test_tools_list_matches_full_registered_set(self) -> None:
        with TestClient(create_app()) as client:
            _, session_id = _initialize(client)

            response = client.post(
                "/mcp",
                headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )

        body = _parse_rpc_body(response)
        mounted_names = {tool["name"] for tool in body["result"]["tools"]}

        registered = asyncio.run(MCPServer()._mcp.list_tools())  # noqa: SLF001
        expected_names = {tool.name for tool in registered}

        assert mounted_names == expected_names
        # Guard against the comparison being trivially true: the full
        # set is much bigger than MCPServer.list_tools()'s stale,
        # hand-maintained subset (invoke, validate_ensemble,
        # update_ensemble, analyze_execution).
        assert len(expected_names) > 4


class TestMcpSharedService:
    """/mcp and /api/ensembles share one OrchestraService, not two.

    Listing ensembles from a shared cwd (the earlier version of this
    test) is a wrong-accept: two independent OrchestraService
    instances constructed against the same cwd list the same
    ensembles anyway, so it passes even when the wiring is split. This
    version discriminates: it changes project via `set_project` over
    /mcp, pointing at a directory REST was never told about. REST only
    sees the new ensemble if it reads through the same service
    instance the MCP tool call mutated.
    """

    def test_set_project_over_mcp_is_visible_to_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        other_project = tmp_path / "other-project"
        other_ensembles = other_project / ".llm-orc" / "ensembles"
        other_ensembles.mkdir(parents=True)
        (other_ensembles / "only-in-other.yaml").write_text(
            "name: only-in-other\ndescription: Only in other\nagents: []\n"
        )

        with TestClient(create_app()) as client:
            _, session_id = _initialize(client)

            set_project_response = client.post(
                "/mcp",
                headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "set_project",
                        "arguments": {"path": str(other_project)},
                    },
                },
            )
            rest_response = client.get("/api/ensembles")

        set_project_body = _parse_rpc_body(set_project_response)
        status = set_project_body["result"]["structuredContent"]["status"]
        assert status == "ok"

        rest_names = {ensemble["name"] for ensemble in rest_response.json()}
        assert "only-in-other" in rest_names


class TestMcpHostHeaderGuard:
    """The serve sits behind a reverse proxy; a real Host header must pass."""

    _REMOTE_HOST = "llm-orc.remote.example"

    def test_dns_rebinding_guard_can_reject(self) -> None:
        """Sanity check: the guard mechanism itself does reject when on.

        Proves the test 4 methodology detects a real rejection -- built
        directly against FastMCP/Starlette (not MCPServer, which never
        enables this guard).
        """
        mcp = FastMCP(
            "llm-orc-test",
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=["127.0.0.1:*", "localhost:*"],
            ),
        )
        mcp_app = mcp.streamable_http_app()
        app = Starlette(
            routes=[Route("/mcp", endpoint=mcp_app)],
            lifespan=mcp_app.router.lifespan_context,
        )

        with TestClient(app) as client:
            response = client.post(
                "/mcp",
                headers={**_ACCEPT_HEADERS, "Host": self._REMOTE_HOST},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "t", "version": "0.1"},
                    },
                },
            )

        assert response.status_code == 421

    def test_reverse_proxied_host_header_is_not_rejected(self) -> None:
        """The actual /mcp mount: on by default, no flag, no auth."""
        with TestClient(create_app()) as client:
            response = client.post(
                "/mcp",
                headers={**_ACCEPT_HEADERS, "Host": self._REMOTE_HOST},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "t", "version": "0.1"},
                    },
                },
            )

        assert response.status_code not in (400, 421)
        assert response.status_code == 200


class TestCreateAppSessionManagerLifecycle:
    """Each create_app() owns its own FastMCP session manager.

    FastMCP's streamable HTTP session manager can only run() once per
    instance; a shared MCPServer singleton across create_app() calls
    would raise on the second app's lifespan startup. create_app()
    builds a fresh MCPServer per call, so two apps in the same process
    must both enter and exit their lifespans without error.
    """

    def test_two_create_app_instances_in_sequence(self) -> None:
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200

        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200


class TestMcpCrudScope:
    """scope reaches the handler through the FastMCP tool signature."""

    def test_create_profile_global_over_mcp_lands_in_global_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_orc.core.config.config_manager import resolve_global_config_dir

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)

        with TestClient(create_app()) as client:
            _, session_id = _initialize(client)
            response = client.post(
                "/mcp",
                headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "create_profile",
                        "arguments": {
                            "name": "remote-prof",
                            "provider": "llama-server",
                            "model": "qwen3-8b",
                            "scope": "global",
                        },
                    },
                },
            )
            listed = client.get("/api/profiles")

        body = _parse_rpc_body(response)
        assert "error" not in body, body
        written = Path(body["result"]["structuredContent"]["path"])
        assert written == resolve_global_config_dir() / "profiles" / "remote-prof.yaml"
        assert written.exists()
        assert not (tmp_path / ".llm-orc" / "profiles" / "remote-prof.yaml").exists()
        assert "remote-prof" in {p["name"] for p in listed.json()}

    def test_every_crud_tool_advertises_scope(self) -> None:
        registered = asyncio.run(MCPServer()._mcp.list_tools())  # noqa: SLF001
        by_name = {tool.name: tool for tool in registered}
        for name in (
            "create_ensemble",
            "update_ensemble",
            "delete_ensemble",
            "create_profile",
            "update_profile",
            "delete_profile",
            "create_script",
            "delete_script",
        ):
            scope_schema = by_name[name].inputSchema["properties"]["scope"]
            assert scope_schema.get("enum") == ["project", "global"], (
                name,
                scope_schema,
            )

    def test_create_ensemble_global_over_mcp_lands_in_global_dir_and_lists_as_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_orc.core.config.config_manager import resolve_global_config_dir

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".llm-orc" / "ensembles").mkdir(parents=True)

        with TestClient(create_app()) as client:
            _, session_id = _initialize(client)
            body = _call_tool(
                client,
                session_id,
                "create_ensemble",
                {
                    "name": "remote-made",
                    "agents": [{"name": "writer", "model_profile": "local-qwen3-8b"}],
                    "scope": "global",
                },
            )
            listed = client.get("/api/ensembles")

        assert "error" not in body, body
        written = Path(body["result"]["structuredContent"]["path"])
        assert written == resolve_global_config_dir() / "ensembles" / "remote-made.yaml"
        assert written.exists()
        assert not (tmp_path / ".llm-orc" / "ensembles" / "remote-made.yaml").exists()
        entry = next(e for e in listed.json() if e["name"] == "remote-made")
        assert entry["source"] == "global"


def _setup_delete_ensemble(global_dir: Path) -> dict[str, Any]:
    target = global_dir / "ensembles" / "only-global.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("name: only-global\nagents: []\n")
    return {"ensemble_name": "only-global", "confirm": True, "scope": "global"}


def _check_delete_ensemble(
    global_dir: Path,
    client: TestClient,
    session_id: str,
    body: dict[str, Any],
) -> None:
    assert "error" not in body, body
    assert body["result"].get("isError") is not True, body
    assert not (global_dir / "ensembles" / "only-global.yaml").exists()


def _setup_update_ensemble(global_dir: Path) -> dict[str, Any]:
    target = global_dir / "ensembles" / "only-global.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("name: only-global\nagents: []\n")
    return {
        "ensemble_name": "only-global",
        "changes": {"add_agents": []},
        "dry_run": False,
        "scope": "global",
    }


def _check_update_ensemble(
    global_dir: Path,
    client: TestClient,
    session_id: str,
    body: dict[str, Any],
) -> None:
    assert "error" not in body, body
    assert body["result"].get("isError") is not True, body
    assert body["result"]["structuredContent"]["modified"] is True

    unscoped = _call_tool(
        client,
        session_id,
        "update_ensemble",
        {
            "ensemble_name": "only-global",
            "changes": {"add_agents": []},
            "dry_run": False,
        },
        rpc_id=10,
    )
    assert unscoped["result"]["isError"] is True
    assert "global tier" in unscoped["result"]["content"][0]["text"]


def _setup_delete_profile(global_dir: Path) -> dict[str, Any]:
    target = global_dir / "profiles" / "only-global.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("name: only-global\nprovider: llama-server\nmodel: m\n")
    return {"name": "only-global", "confirm": True, "scope": "global"}


def _check_delete_profile(
    global_dir: Path,
    client: TestClient,
    session_id: str,
    body: dict[str, Any],
) -> None:
    assert "error" not in body, body
    assert body["result"].get("isError") is not True, body
    assert not (global_dir / "profiles" / "only-global.yaml").exists()


def _setup_update_profile(global_dir: Path) -> dict[str, Any]:
    target = global_dir / "profiles" / "only-global.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("name: only-global\nprovider: llama-server\nmodel: original\n")
    return {
        "name": "only-global",
        "changes": {"model": "changed"},
        "scope": "global",
    }


def _check_update_profile(
    global_dir: Path,
    client: TestClient,
    session_id: str,
    body: dict[str, Any],
) -> None:
    assert "error" not in body, body
    assert body["result"].get("isError") is not True, body
    target = global_dir / "profiles" / "only-global.yaml"
    assert yaml.safe_load(target.read_text())["model"] == "changed"


def _setup_create_script(global_dir: Path) -> dict[str, Any]:
    return {"name": "remote-made", "category": "util", "scope": "global"}


def _check_create_script(
    global_dir: Path,
    client: TestClient,
    session_id: str,
    body: dict[str, Any],
) -> None:
    assert "error" not in body, body
    assert body["result"].get("isError") is not True, body
    assert (global_dir / "scripts" / "util" / "remote-made.py").exists()


def _setup_delete_script(global_dir: Path) -> dict[str, Any]:
    target = global_dir / "scripts" / "util" / "only-global.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('"""Only in global."""\n')
    return {
        "name": "only-global",
        "category": "util",
        "confirm": True,
        "scope": "global",
    }


def _check_delete_script(
    global_dir: Path,
    client: TestClient,
    session_id: str,
    body: dict[str, Any],
) -> None:
    assert "error" not in body, body
    assert body["result"].get("isError") is not True, body
    assert not (global_dir / "scripts" / "util" / "only-global.py").exists()


class TestMcpCrudScopeForwarding:
    """Every scope-taking tool forwards `scope` to its handler, not just two."""

    @pytest.mark.parametrize(
        ("tool", "setup", "check"),
        [
            pytest.param(
                "delete_ensemble",
                _setup_delete_ensemble,
                _check_delete_ensemble,
                id="delete_ensemble",
            ),
            pytest.param(
                "update_ensemble",
                _setup_update_ensemble,
                _check_update_ensemble,
                id="update_ensemble",
            ),
            pytest.param(
                "delete_profile",
                _setup_delete_profile,
                _check_delete_profile,
                id="delete_profile",
            ),
            pytest.param(
                "update_profile",
                _setup_update_profile,
                _check_update_profile,
                id="update_profile",
            ),
            pytest.param(
                "create_script",
                _setup_create_script,
                _check_create_script,
                id="create_script",
            ),
            pytest.param(
                "delete_script",
                _setup_delete_script,
                _check_delete_script,
                id="delete_script",
            ),
        ],
    )
    def test_tool_forwards_scope_to_the_global_tier(
        self,
        tool: str,
        setup: Any,
        check: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llm_orc.core.config.config_manager import resolve_global_config_dir

        monkeypatch.chdir(tmp_path)
        global_dir = resolve_global_config_dir()
        arguments = setup(global_dir)

        with TestClient(create_app()) as client:
            _, session_id = _initialize(client)
            body = _call_tool(client, session_id, tool, arguments)
            check(global_dir, client, session_id, body)
