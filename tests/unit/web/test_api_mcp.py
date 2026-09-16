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
def _reset_shared_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a fresh OrchestraService/MCPServer.

    Both are process-wide singletons (web/api/__init__.py), and
    FastMCP's streamable HTTP session manager can only run() once per
    instance -- reusing one across tests that each enter the app's
    lifespan would raise on the second test. Reset before every test
    so create_app() builds fresh ones.
    """
    monkeypatch.setattr(web_api, "_mcp_server", None)
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
    """/mcp and /api/ensembles agree: one OrchestraService behind both."""

    def test_list_ensembles_matches_rest_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "demo.yaml").write_text(
            "name: demo\ndescription: Demo ensemble\nagents: []\n"
        )

        with TestClient(create_app()) as client:
            _, session_id = _initialize(client)

            mcp_response = client.post(
                "/mcp",
                headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "list_ensembles", "arguments": {}},
                },
            )
            rest_response = client.get("/api/ensembles")

        mcp_body = _parse_rpc_body(mcp_response)
        structured = mcp_body["result"]["structuredContent"]
        mcp_names = {ensemble["name"] for ensemble in structured["result"]}

        rest_names = {ensemble["name"] for ensemble in rest_response.json()}

        assert "demo" in mcp_names
        assert "demo" in rest_names
        assert mcp_names == rest_names


class TestMcpHostHeaderGuard:
    """The serve sits behind a reverse proxy; a real Host header must pass."""

    _REMOTE_HOST = "llm-orc.homelab.nate.green"

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
