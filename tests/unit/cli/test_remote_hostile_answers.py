"""What a remote can do to the client: answer with a body that breaks a
parser, sit on the line, or be configured with a URL the HTTP library
cannot encode. A probe never fails the list, and a run or preflight ends
in one line. The real surfaces on canned transports; nothing here opens a
connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from click.testing import CliRunner, Result

from llm_orc.cli import cli
from llm_orc.services import remote_probe, remote_run
from tests.unit.cli.test_invoke_remote import (  # noqa: F401
    REMOTE_URL,
    Canned,
    _canned,
    _write_top,
    in_project,
    remotes,
)
from tests.unit.mcp_server.test_invoke_tool import (  # noqa: F401
    ToolCall,
    stdio_tool,
)
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    listing,
    project,
    service,
    state_dir,
)

DEEP = "[" * 200000
BAD_IDNA = "http://xn--zz.example"


def _global_config(data: dict[str, Any]) -> None:
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _two_remotes(bad_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``a`` is the bad one and ``b`` answers a health document."""
    _global_config(
        {"remotes": {"a": {"url": bad_url}, "b": {"url": "https://b.example"}}}
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(200, text=DEEP)
        return httpx.Response(200, json={"version": "1.0.0"})

    monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))


BAD_REMOTES = [
    pytest.param("https://a.example", id="deep-body"),
    pytest.param(BAD_IDNA, id="bad-idna-url"),
]


class TestOneBadRemoteDoesNotFailTheList:
    @pytest.mark.parametrize("bad_url", BAD_REMOTES)
    def test_the_command_prints_both_rows(
        self, in_project: Path, monkeypatch: pytest.MonkeyPatch, bad_url: str
    ) -> None:
        _two_remotes(bad_url, monkeypatch)

        result = CliRunner().invoke(cli, ["remotes", "--output-format", "json"])

        assert result.exit_code == 0, result.output
        rows = {row["name"]: row for row in json.loads(result.stdout)}
        assert rows["a"]["reachable"] is False
        assert rows["a"]["error"]
        assert rows["b"]["reachable"] is True

    @pytest.mark.parametrize("bad_url", BAD_REMOTES)
    def test_list_remotes_returns_both_rows(
        self,
        in_project: Path,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        bad_url: str,
    ) -> None:
        _two_remotes(bad_url, monkeypatch)

        result = stdio_tool("list_remotes", {})

        rows = {row["name"]: row for row in result["remotes"]}
        assert rows["a"]["reachable"] is False
        assert rows["b"]["reachable"] is True

    def test_a_body_over_the_cap_is_not_a_health_document(
        self, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _two_remotes("https://a.example", monkeypatch)

        result = CliRunner().invoke(cli, ["remotes", "--output-format", "json"])

        rows = {row["name"]: row for row in json.loads(result.stdout)}
        assert "not with an llm-orc health document" in rows["a"]["error"]

    def test_a_valid_document_over_the_cap_is_not_read(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = json.dumps({"version": "1.0.0", "pad": "x" * 70000})
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text=body))
        monkeypatch.setattr(remote_run, "transport", transport)

        result = CliRunner().invoke(cli, ["remotes", "--output-format", "json"])

        (row,) = json.loads(result.stdout)
        assert row["reachable"] is False
        assert "not with an llm-orc health document" in row["error"]


class _Trickle(httpx.AsyncByteStream):
    """Sends a byte now and then, for far longer than the deadline."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(1000):
            await asyncio.sleep(0.05)
            yield b" "


class TestTheDeadline:
    def test_a_remote_that_trickles_bytes_ends_within_the_deadline(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(remote_probe, "PROBE_DEADLINE_S", 0.3)

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Trickle())

        monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))
        started = time.monotonic()

        result = CliRunner().invoke(cli, ["remotes", "--output-format", "json"])

        assert time.monotonic() - started < 2
        (row,) = json.loads(result.stdout)
        assert row["reachable"] is False
        assert "no answer within" in row["error"]


class TestARunOrPreflightOnAHostileRemote:
    def test_a_huge_nested_body_is_one_line(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, Canned(200, DEEP))

        result = CliRunner().invoke(
            cli, ["invoke", "top", "hi", "--remote", "remote-host"]
        )

        _one_remote_line(result)

    @pytest.mark.parametrize("flags", [[], ["--preflight"]])
    def test_a_url_the_http_library_cannot_encode_is_one_line(
        self,
        in_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        flags: list[str],
    ) -> None:
        _write_top(in_project)
        _global_config({"remotes": {"remote-host": {"url": BAD_IDNA}}})
        _canned(monkeypatch, Canned(200, "{}"))

        result = CliRunner().invoke(
            cli, ["invoke", "top", "hi", "--remote", "remote-host", *flags]
        )

        _one_remote_line(result)


ROW = {"kind": "profile", "name": "seat", "status": "ready", "via": ["top"]}
BAD_PREFLIGHTS = [
    pytest.param(
        {"runnable": True, "dependencies": [], "bindings": ["x"]},
        id="bindings-a-list",
    ),
    pytest.param(
        {"runnable": False, "dependencies": "abc", "bindings": {}},
        id="dependencies-a-string",
    ),
    pytest.param(
        {"runnable": False, "dependencies": [{**ROW, "via": 5}], "bindings": {}},
        id="via-an-int",
    ),
    pytest.param(
        {"runnable": False, "dependencies": [{**ROW, "name": 7}], "bindings": {}},
        id="name-an-int",
    ),
    pytest.param(
        {"runnable": True, "dependencies": ["y"], "bindings": {}},
        id="row-a-string",
    ),
    pytest.param(
        {"runnable": True, "dependencies": [], "bindings": {"k": 1}},
        id="binding-value-an-int",
    ),
    pytest.param(
        {"error": {"kind": "invalid_request", "message": "m", "dependencies": ["y"]}},
        id="envelope-dependencies-strings",
    ),
    pytest.param(
        {"error": {"kind": "invalid_request", "message": 5, "dependencies": []}},
        id="envelope-message-an-int",
    ),
]


class TestAPreflightAnswerIsCheckedForShape:
    @pytest.mark.parametrize("body", BAD_PREFLIGHTS)
    def test_a_body_that_would_crash_the_display_is_not_a_preflight_answer(
        self,
        in_project: Path,
        remotes: None,
        monkeypatch: pytest.MonkeyPatch,
        body: dict[str, Any],
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, Canned(200, json.dumps(body)))

        result = CliRunner().invoke(
            cli,
            [
                "invoke",
                "top",
                "--remote",
                "remote-host",
                "--preflight",
                "--output-format",
                "text",
            ],
        )

        _one_remote_line(result)
        assert "not a preflight document" in result.stderr


def _one_remote_line(result: Result) -> None:
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    (error,) = [x for x in result.stderr.splitlines() if x.startswith("Error:")]
    assert "Remote 'remote-host'" in error
    assert result.stdout == ""
