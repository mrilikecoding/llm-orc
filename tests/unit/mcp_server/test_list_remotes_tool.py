"""The MCP ``list_remotes`` tool (Arc 6, part 1): the real tool on a
relaying server (the stdio command's build) and over ``/mcp`` on a mount,
which does not relay. The transport seam is the second real service or
canned answers; nothing here opens a connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import yaml

from llm_orc.services import remote_run
from llm_orc.web import server as web_server
from tests.unit.cli.test_invoke_remote import (  # noqa: F401
    REMOTE_URL,
    Remote,
    in_project,
    remote,
    remotes,
)
from tests.unit.mcp_server.test_invoke_tool import (  # noqa: F401
    ToolCall,
    stdio_tool,
    tool,
)
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    listing,
    project,
    service,
    state_dir,
)


class TestOnARelayingServer:
    def test_a_configured_name_reads_reachable_with_the_second_services_version(
        self, remote: Remote, stdio_tool: ToolCall, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web_server, "get_version", lambda: "9.9.9")

        result = stdio_tool("list_remotes", {})

        assert result == {
            "remotes": [
                {
                    "name": "remote-host",
                    "url": REMOTE_URL,
                    "reachable": True,
                    "version": "9.9.9",
                }
            ]
        }

    def test_a_connection_error_reads_unreachable_with_the_error(
        self,
        remotes: None,
        in_project: Path,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(refuse))

        result = stdio_tool("list_remotes", {})

        (row,) = result["remotes"]
        assert row["reachable"] is False
        assert row["error"] == "connection refused"

    def test_a_project_config_remotes_key_is_not_read(
        self,
        remotes: None,
        in_project: Path,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (in_project / ".llm-orc" / "config.yaml").write_text(
            yaml.safe_dump({"remotes": {"from-repo": {"url": "https://repo.example"}}})
        )
        monkeypatch.setattr(
            remote_run,
            "transport",
            httpx.MockTransport(lambda r: httpx.Response(200, json={"version": "1"})),
        )

        result = stdio_tool("list_remotes", {})

        assert [row["name"] for row in result["remotes"]] == ["remote-host"]

    def test_none_configured_is_an_empty_list(
        self, in_project: Path, stdio_tool: ToolCall
    ) -> None:
        assert stdio_tool("list_remotes", {}) == {"remotes": []}

    def test_a_malformed_remotes_key_is_an_invalid_request_envelope(
        self, in_project: Path, stdio_tool: ToolCall
    ) -> None:
        path = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"remotes": ["x"]}))

        result = stdio_tool("list_remotes", {})

        assert result["status"] == "error"
        assert result["error"]["kind"] == "invalid_request"
        assert "must be a mapping" in result["error"]["message"]


class TestOnAMountedServer:
    def test_it_answers_that_this_serve_does_not_relay_and_probes_nothing(
        self,
        remotes: None,
        in_project: Path,
        tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"version": "1"})

        monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))

        result = tool("list_remotes", {})

        assert result["status"] == "error"
        assert result["error"]["kind"] == "invalid_request"
        assert "does not relay" in result["error"]["message"]
        assert seen == []
