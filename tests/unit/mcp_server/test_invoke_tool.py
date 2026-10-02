"""The MCP ``invoke`` tool through ``/mcp`` over one real OrchestraService
on temp project, state and global dirs (Arc 5, Task 10 and its
integration items). Nothing here opens a connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.mcp.server import MCPServer
from llm_orc.services import remote_run
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.web.server import create_app
from tests.unit.cli.test_invoke_remote import (  # noqa: F401
    REMOTE_URL,
    Canned,
    Delayed,
    Remote,
    _canned,
    _write_top,
    in_project,
    remote,
    remotes,
)
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    _gone,
    _profile,
    _read_pid,
    _runs,
    _yaml,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_bundles import _pack
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


@pytest.fixture
def stdio_tool(service: OrchestraService) -> ToolCall:
    """Call an MCP tool on a server built the way ``llm-orc mcp serve``
    builds it (no mount, nothing to refuse ``remote``); the structured
    result."""
    server = MCPServer(service=service)

    def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        _, structured = asyncio.run(server._mcp.call_tool(name, arguments))
        assert isinstance(structured, dict)
        return structured

    return call


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


def _where_script(project: Path, record: Path) -> None:
    """A script that records the directory it runs from, so a test can
    tell which host ran it."""
    scripts = project / ".llm-orc" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "where.py").write_text(
        "import json, os, sys\n"
        "sys.stdin.read()\n"
        "here = os.path.dirname(os.path.abspath(__file__))\n"
        f'open(r"{record}", "w").write(here)\n'
        'print(json.dumps({"success": True, "data": "ran"}))\n'
    )
    _ensemble(project / ".llm-orc", "top", [{"name": "first", "script": "where.py"}])


class TestRemoteRun:
    def test_a_local_named_root_runs_on_the_second_service(
        self,
        project: Path,
        remote: Remote,
        stdio_tool: ToolCall,
        tmp_path: Path,
    ) -> None:
        record = tmp_path / "where.txt"
        _where_script(project, record)

        result = stdio_tool(
            "invoke",
            {"ensemble_name": "top", "input_data": "hi", "remote": "remote-host"},
        )

        assert result["status"] == "success", result
        assert "metadata" in result
        assert result["raw_output"] is False
        assert len(remote.calls) == 1
        assert Path(record.read_text()).is_relative_to(remote.state / "runs")

    def test_a_profile_the_remote_lacks_returns_its_not_equipped_envelope(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project, with_profile=True)

        result = stdio_tool(
            "invoke",
            {"ensemble_name": "top", "input_data": "hi", "remote": "remote-host"},
        )

        assert result["status"] == "error"
        assert result["error"]["kind"] == "not_equipped"
        rows = {d["name"]: d["status"] for d in result["error"]["dependencies"]}
        assert rows["seat"] == "missing_profile"

    def test_bind_travels_and_the_run_carries_its_bindings(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project, with_profile=True)
        _profile(remote.project / ".llm-orc", "other", model="mock-other")

        result = stdio_tool(
            "invoke",
            {
                "ensemble_name": "top",
                "input_data": "hi",
                "remote": "remote-host",
                "bind": {"seat": "other"},
            },
        )

        assert result["status"] == "success", result
        assert result["bindings"] == {"seat": "other"}

    def test_with_profiles_ships_the_definition(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project, with_profile=True)

        result = stdio_tool(
            "invoke",
            {
                "ensemble_name": "top",
                "input_data": "hi",
                "remote": "remote-host",
                "with_profiles": ["seat"],
            },
        )

        assert result["status"] == "success", result
        assert "seat" in remote.calls[0][1]["profiles"]

    def test_persist_global_leaves_the_bundle_on_the_second_service(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project)

        result = stdio_tool(
            "invoke",
            {
                "ensemble_name": "top",
                "input_data": "hi",
                "remote": "remote-host",
                "persist": "global",
            },
        )

        assert result["persisted"] == "top", result
        assert (remote.config / "llm-orc" / "bundles" / "top.json").is_file()
        assert not (resolve_global_config_dir() / "bundles").exists()

    def test_the_remote_name_is_a_name_in_the_config_or_a_url(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project)

        result = stdio_tool(
            "invoke",
            {"ensemble_name": "top", "input_data": "hi", "remote": REMOTE_URL},
        )

        assert result["status"] == "success", result


class TestPersistWithoutRemote:
    def test_an_inline_root_persists_in_the_local_global_config(
        self, project: Path, tool: ToolCall
    ) -> None:
        request = {k: v for k, v in _pack().items() if k != "input"}

        result = tool("invoke", {**request, "input_data": "hi", "persist": "global"})

        assert result["persisted"] == "pack", result
        assert (resolve_global_config_dir() / "bundles" / "pack.json").is_file()


class TestARemoteThatDoesNotAnswer:
    def test_an_unknown_remote_is_invalid_and_lists_the_known_names(
        self,
        project: Path,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        remotes: None,
    ) -> None:
        _write_top(project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = stdio_tool(
            "invoke",
            {"ensemble_name": "top", "input_data": "hi", "remote": "nowhere"},
        )

        assert result["error"]["kind"] == "invalid_request"
        assert "known remotes: remote-host" in result["error"]["message"]
        assert calls == []

    @pytest.mark.parametrize(
        "answer",
        [
            httpx.ConnectError("connection refused"),
            Canned(200, "<!doctype html><title>llm-orc</title>"),
        ],
        ids=["unreachable", "html"],
    )
    def test_no_answer_or_no_result_is_a_remote_error_naming_it(
        self,
        project: Path,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        remotes: None,
        answer: Canned | BaseException,
    ) -> None:
        _write_top(project)
        _canned(monkeypatch, answer)

        result = stdio_tool(
            "invoke",
            {"ensemble_name": "top", "input_data": "hi", "remote": "remote-host"},
        )

        assert result["status"] == "error"
        assert result["has_errors"] is True
        assert result["results"] == {}
        assert result["deliverable"] is None
        assert result["error"]["kind"] == "remote_error"
        assert result["error"]["dependencies"] == []
        assert "remote-host" in result["error"]["message"]


class TestTheEventLoop:
    async def test_it_is_not_blocked_while_a_remote_run_is_out(
        self,
        project: Path,
        service: OrchestraService,
        remote: Remote,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_top(project)
        monkeypatch.setattr(remote_run, "transport", Delayed(remote.transport, 0.5))
        call = asyncio.create_task(
            MCPServer(service=service)._mcp.call_tool(
                "invoke",
                {"ensemble_name": "top", "input_data": "hi", "remote": "remote-host"},
            )
        )
        ticks = 0
        while not call.done():
            await asyncio.sleep(0.01)
            ticks += 1

        assert len(remote.calls) == 1
        assert ticks >= 10
        await call


def _threads_in_the_transport() -> list[str]:
    """Threads whose stack is inside the remote transport: one blocked
    in a POST whose caller has gone. Idle pool workers do not count."""
    found: list[str] = []
    for ident, top in sys._current_frames().items():
        files = []
        frame: FrameType | None = top
        while frame is not None:
            files.append(frame.f_code.co_filename)
            frame = frame.f_back
        if any(f.endswith("remote_run.py") for f in files):
            found.append(str(ident))
    return found


class TestCancellingARemoteRun:
    """Cancelling the tool call closes the connection, so the second
    service sees the disconnect, kills the script's process group and
    removes the run's layer, long before the script's own end."""

    @pytest.fixture
    def pids(self, tmp_path: Path) -> Iterator[dict[str, Path]]:
        marks = {"script": tmp_path / "pid.txt", "child": tmp_path / "child.txt"}
        yield marks
        for mark in marks.values():
            if mark.exists() and mark.read_text().strip():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(mark.read_text()), signal.SIGKILL)

    def _sleeping_root(self, project: Path, pids: dict[str, Path]) -> None:
        scripts = project / ".llm-orc" / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "nap.py").write_text(
            "import os, subprocess, sys, time\n"
            "sys.stdin.read()\n"
            "kid = subprocess.Popen(['sleep', '30'])\n"
            f'open(r"{pids["child"]}", "w").write(str(kid.pid))\n'
            f'open(r"{pids["script"]}", "w").write(str(os.getpid()))\n'
            "time.sleep(30)\n"
        )
        _ensemble(project / ".llm-orc", "top", [{"name": "a", "script": "nap.py"}])

    async def test_the_remote_run_is_killed_and_nothing_still_waits(
        self,
        project: Path,
        service: OrchestraService,
        remote: Remote,
        pids: dict[str, Path],
    ) -> None:
        self._sleeping_root(project, pids)
        call = asyncio.create_task(
            MCPServer(service=service)._mcp.call_tool(
                "invoke",
                {"ensemble_name": "top", "input_data": "hi", "remote": "remote-host"},
            )
        )
        script_pid = await _read_pid(pids["script"])
        child_pid = await _read_pid(pids["child"])
        started = time.monotonic()

        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        assert await _gone(script_pid)
        assert await _gone(child_pid)
        assert _runs(remote.state) == []
        assert time.monotonic() - started < 5
        assert asyncio.all_tasks() == {asyncio.current_task()}
        assert _threads_in_the_transport() == []


INLINE_ROOT = {"name": "inline", "agents": [{"name": "a", "script": "echo hi"}]}


class TestARefusalBeforeAnythingIsSent:
    @pytest.mark.parametrize(
        "inline",
        [
            {"ensemble": INLINE_ROOT},
            {"ensembles": {"kid": INLINE_ROOT}},
            {"profiles": {"p": {"provider": "llama-server", "model": "m"}}},
            {"scripts": {"tools/x.py": "print(1)"}},
        ],
        ids=["ensemble", "ensembles", "profiles", "scripts"],
    )
    def test_an_inline_part_with_remote_is_invalid_and_nothing_is_sent(
        self,
        project: Path,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        remotes: None,
        inline: dict[str, Any],
    ) -> None:
        _write_top(project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = stdio_tool(
            "invoke",
            {
                "ensemble_name": "top",
                "input_data": "hi",
                "remote": "remote-host",
                **inline,
            },
        )

        assert result["error"]["kind"] == "invalid_request"
        assert "send it to the remote itself" in result["error"]["message"]
        assert calls == []

    def test_with_profiles_without_remote_is_invalid(
        self, project: Path, stdio_tool: ToolCall, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = stdio_tool(
            "invoke",
            {"ensemble_name": "top", "input_data": "hi", "with_profiles": ["seat"]},
        )

        assert result["error"]["kind"] == "invalid_request"
        assert "with_profiles needs remote" in result["error"]["message"]
        assert calls == []

    def test_remote_without_a_name_is_invalid(
        self,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        remotes: None,
    ) -> None:
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = stdio_tool("invoke", {"input_data": "hi", "remote": "remote-host"})

        assert result["error"]["kind"] == "invalid_request"
        assert "ensemble_name" in result["error"]["message"]
        assert calls == []


class TestAServeIsNotARelay:
    @pytest.mark.parametrize(
        "extra",
        [
            {"remote": "remote-host"},
            {"remote": REMOTE_URL},
            {"with_profiles": ["seat"]},
        ],
        ids=["name", "url", "with_profiles"],
    )
    def test_the_mounted_mcp_refuses_remote_and_sends_nothing(
        self,
        project: Path,
        tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        remotes: None,
        extra: dict[str, Any],
    ) -> None:
        _write_top(project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = tool("invoke", {"ensemble_name": "top", "input_data": "hi", **extra})

        assert result["status"] == "error"
        assert result["error"]["kind"] == "invalid_request"
        assert "does not relay" in result["error"]["message"]
        assert "call that remote yourself" in result["error"]["message"]
        assert calls == []


class TestRestDoesNotGainRemote:
    def test_a_remote_key_on_the_execute_endpoint_is_still_a_422(
        self, project: Path, service: OrchestraService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(project)
        monkeypatch.setattr(web_api, "_orchestra_service", service)

        with TestClient(create_app()) as client:
            response = client.post(
                "/api/ensembles/execute",
                json={"ensemble_name": "top", "input": "hi", "remote": "remote-host"},
            )

        assert response.status_code == 422
        assert "remote" in response.text
