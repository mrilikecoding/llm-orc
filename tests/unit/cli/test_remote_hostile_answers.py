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
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.remotes import RemoteError
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


ESCAPE = "\x1b]0;pwned\x07"


def _no_control_characters(text: str) -> None:
    assert "\x1b" not in text
    assert "\x07" not in text


class TestTextModeStripsControlCharacters:
    def _preflight(self, *extra: str) -> Result:
        return CliRunner().invoke(
            cli,
            ["invoke", "top", "--remote", "remote-host", "--preflight", *extra],
        )

    def test_a_preflight_table_cell_from_the_remote(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        row = {**ROW, "name": ESCAPE, "via": [ESCAPE], "status": ESCAPE}
        document = {
            "runnable": False,
            "dependencies": [row],
            "bindings": {ESCAPE: ESCAPE},
        }
        _canned(monkeypatch, Canned(200, json.dumps(document)))

        result = self._preflight("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "pwned" in result.stdout
        _no_control_characters(result.stdout)

    def test_a_refusal_table_and_message_from_the_remote(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        error = {
            "kind": ESCAPE,
            "message": ESCAPE,
            "dependencies": [{**ROW, "name": ESCAPE}],
        }
        _canned(monkeypatch, Canned(200, json.dumps({"error": error})))

        result = self._preflight("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "pwned" in result.stdout
        _no_control_characters(result.stdout)

    def test_a_refusal_to_a_run_from_the_remote(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        error = {"kind": "not_equipped", "message": ESCAPE, "dependencies": [ROW]}
        envelope = {"status": "error", "has_errors": True, "error": error}
        _canned(monkeypatch, Canned(200, json.dumps(envelope)))

        result = CliRunner().invoke(
            cli,
            [
                "invoke",
                "top",
                "hi",
                "--remote",
                "remote-host",
                "--output-format",
                "text",
            ],
        )

        assert result.exit_code == 1, result.output
        assert "pwned" in result.stdout
        _no_control_characters(result.stdout)

    def test_a_probe_row_name_and_url_in_text_mode(
        self, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _global_config({"remotes": {ESCAPE: {"url": "https://a.example/\x1b[2J"}}})
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text="x"))
        monkeypatch.setattr(remote_run, "transport", transport)

        result = CliRunner().invoke(cli, ["remotes", "--output-format", "text"])

        assert result.exit_code == 0, result.output
        assert "pwned" in result.stdout
        _no_control_characters(result.stdout)


class TestANameIsNotAUrl:
    def test_a_name_with_a_scheme_separator_is_malformed_and_probes_nothing(
        self, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _global_config({"remotes": {"http://evil.example": {"url": REMOTE_URL}}})
        seen: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"version": "1"})

        monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))

        result = CliRunner().invoke(cli, ["remotes", "--output-format", "json"])

        assert result.exit_code == 1, result.output
        assert "malformed" in result.output
        assert seen == []

    def test_the_config_manager_refuses_it(self, in_project: Path) -> None:
        _global_config({"remotes": {"a://b": {"url": REMOTE_URL}}})

        with pytest.raises(RemoteError, match="malformed"):
            ConfigurationManager(provision=False).remotes()


SECRET = "s3cret"
MASKED = "http://user:***@a.example"
WITH_USERINFO = f"http://user:{SECRET}@a.example"


class TestUserinfoIsMasked:
    @pytest.fixture
    def configured(self, in_project: Path) -> None:
        _global_config({"remotes": {"a": {"url": WITH_USERINFO}}})

    @pytest.mark.parametrize("fmt", ["text", "json"])
    def test_the_remotes_command_shows_the_password_masked(
        self,
        configured: None,
        monkeypatch: pytest.MonkeyPatch,
        fmt: str,
    ) -> None:
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, json={"version": "1"})
        )
        monkeypatch.setattr(remote_run, "transport", transport)

        result = CliRunner().invoke(cli, ["remotes", "--output-format", fmt])

        assert result.exit_code == 0, result.output
        assert SECRET not in result.output
        assert MASKED in result.stdout

    def test_list_remotes_rows_carry_the_masked_url(
        self,
        configured: None,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, json={"version": "1"})
        )
        monkeypatch.setattr(remote_run, "transport", transport)

        result = stdio_tool("list_remotes", {})

        assert SECRET not in json.dumps(result)
        assert result["remotes"][0]["url"] == MASKED

    def test_a_probe_error_that_echoes_the_url_is_masked(
        self, in_project: Path
    ) -> None:
        _global_config({"remotes": {"a": {"url": f"ftp://user:{SECRET}@a.example"}}})

        result = CliRunner().invoke(cli, ["remotes", "--output-format", "json"])

        assert SECRET not in result.output
        assert "ftp://user:***@a.example" in result.output

    def test_the_ticker_line_and_a_connection_error_are_masked(
        self, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, httpx.ConnectError("refused"))

        result = CliRunner().invoke(
            cli, ["invoke", "top", "hi", "--remote", WITH_USERINFO]
        )

        assert result.exit_code == 1, result.output
        assert SECRET not in result.output
        assert f"Running on {MASKED}... " in result.stderr
        assert f"could not reach {MASKED}" in result.stderr

    def test_a_refused_url_is_masked_in_the_message(
        self, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)

        result = CliRunner().invoke(
            cli,
            ["invoke", "top", "hi", "--remote", f"ftp://user:{SECRET}@a.example"],
        )

        assert result.exit_code == 1, result.output
        assert SECRET not in result.output
        assert "ftp://user:***@a.example" in result.output


HINT = (
    "the remote may be older than 0.25.0 and have no preflight route; "
    "`llm-orc remotes` shows its version"
)


class TestAnOlderRemoteHasNoPreflightRoute:
    @pytest.mark.parametrize("status", [404, 405])
    def test_the_message_carries_the_hint(
        self,
        in_project: Path,
        remotes: None,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, Canned(status, '{"detail": "Method Not Allowed"}'))

        result = CliRunner().invoke(
            cli, ["invoke", "top", "--remote", "remote-host", "--preflight"]
        )

        assert result.exit_code == 1, result.output
        assert f"Remote 'remote-host' ({status})" in result.stderr
        assert HINT in result.stderr

    def test_a_run_refused_with_405_gets_no_hint(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, Canned(405, "no"))

        result = CliRunner().invoke(
            cli, ["invoke", "top", "hi", "--remote", "remote-host"]
        )

        assert result.exit_code == 1, result.output
        assert "older than" not in result.stderr


def _one_remote_line(result: Result) -> None:
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    (error,) = [x for x in result.stderr.splitlines() if x.startswith("Error:")]
    assert "Remote 'remote-host'" in error
    assert result.stdout == ""
