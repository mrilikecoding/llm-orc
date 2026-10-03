"""A REST run is tied to its connection (Arc 5, Task 5).

Starlette's TestClient cannot simulate a client that leaves mid-request,
so these pins call the ASGI app directly with their own ``receive`` and
``send``. The script is injected and sleeps, over a real OrchestraService
on temp project and state dirs.
"""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import llm_orc.web.api as web_api
from llm_orc.services.handlers.execution_handler import ExecutionHandler
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.web.server import create_app
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    INLINE,
    _ensemble,
    _gone,
    _inline_request,
    _read_pid,
    _runs,
    listing,
    project,
    service,
    state_dir,
)

Message = dict[str, Any]


@pytest.fixture
def app(service: OrchestraService, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(web_api, "_orchestra_service", service)
    return create_app()


@pytest.fixture
def pid_file(tmp_path: Path) -> Iterator[Path]:
    """Where the script writes its pid; a survivor is killed afterwards."""
    path = tmp_path / "pid.txt"
    yield path
    if path.exists() and path.read_text().strip():
        try:
            os.kill(int(path.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _sleeping_request(pid_file: Path, nap: float) -> dict[str, Any]:
    script = (
        "import os, sys, time\n"
        "sys.stdin.read()\n"
        f'open(r"{pid_file}", "w").write(str(os.getpid()))\n'
        f"time.sleep({nap})\n"
        'print(\'{"success": true, "data": {"tag": "done"}}\')\n'
    )
    return _inline_request(scripts={"probe/x.py": script})


def _scope(path: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "server": ("testserver", 80),
        "client": ("testclient", 5000),
        "scheme": "http",
    }


class Wire:
    """One request's ``receive`` and ``send``: the body, then silence until
    ``leave`` is called, then ``http.disconnect``."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode()
        self._sent_body = False
        self._left = asyncio.Event()
        self.sent: list[Message] = []

    def leave(self) -> None:
        self._left.set()

    async def receive(self) -> Message:
        if not self._sent_body:
            self._sent_body = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        await self._left.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: Message) -> None:
        self.sent.append(message)

    def status(self) -> int | None:
        starts = [m for m in self.sent if m["type"] == "http.response.start"]
        return int(starts[0]["status"]) if starts else None

    def json_body(self) -> dict[str, Any]:
        raw = b"".join(m.get("body", b"") for m in self.sent)
        return dict(json.loads(raw))


ROUTES = ["/api/ensembles/execute", "/api/ensembles/inline-top/execute"]


def _body_for(path: str, request: dict[str, Any]) -> dict[str, Any]:
    if path == "/api/ensembles/execute":
        return request
    # The named route takes no root; the ensemble is installed in the test.
    return {k: v for k, v in request.items() if k != "ensemble"}


@pytest.fixture
def call(app: Any, project: Path) -> Callable[[str, dict[str, Any]], Wire]:
    """Start a request as a task; return its wire."""

    def start(path: str, request: dict[str, Any]) -> Wire:
        wire = Wire(_body_for(path, request))
        wire.task = asyncio.create_task(  # type: ignore[attr-defined]
            app(_scope(path), wire.receive, wire.send)
        )
        return wire

    return start


class TestADisconnectCancelsTheRun:
    @pytest.mark.parametrize("path", ROUTES)
    async def test_the_script_group_is_gone_and_runs_is_empty(
        self,
        call: Callable[[str, dict[str, Any]], Wire],
        project: Path,
        state_dir: Path,
        pid_file: Path,
        path: str,
    ) -> None:
        _install_named(project)
        wire = call(path, _sleeping_request(pid_file, 20))
        pid = await _read_pid(pid_file)

        started = time.monotonic()
        wire.leave()
        await asyncio.wait_for(wire.task, timeout=10)  # type: ignore[attr-defined]

        assert await _gone(pid)
        assert _runs(state_dir) == []
        assert time.monotonic() - started < 5

    @pytest.mark.parametrize("path", ROUTES)
    async def test_the_route_answers_499_and_logs_no_error(
        self,
        call: Callable[[str, dict[str, Any]], Wire],
        project: Path,
        pid_file: Path,
        caplog: pytest.LogCaptureFixture,
        path: str,
    ) -> None:
        _install_named(project)
        wire = call(path, _sleeping_request(pid_file, 20))
        await _read_pid(pid_file)

        wire.leave()
        await asyncio.wait_for(wire.task, timeout=10)  # type: ignore[attr-defined]

        assert wire.status() == 499
        assert [r for r in caplog.records if r.levelname == "ERROR"] == []


class TestAClientThatStaysGetsItsResult:
    @pytest.mark.parametrize("path", ROUTES)
    async def test_a_finished_run_answers_in_full(
        self,
        call: Callable[[str, dict[str, Any]], Wire],
        project: Path,
        state_dir: Path,
        pid_file: Path,
        path: str,
    ) -> None:
        _install_named(project)
        wire = call(path, _sleeping_request(pid_file, 1))

        await asyncio.wait_for(wire.task, timeout=20)  # type: ignore[attr-defined]

        assert wire.status() == 200
        body = wire.json_body()
        assert body["status"] == "success"
        assert json.loads(body["results"]["s"]["response"])["data"] == {"tag": "done"}
        assert _runs(state_dir) == []

    async def test_a_leaving_client_does_not_cancel_a_concurrent_run(
        self,
        call: Callable[[str, dict[str, Any]], Wire],
        pid_file: Path,
        tmp_path: Path,
    ) -> None:
        path = "/api/ensembles/execute"
        stays = call(path, _sleeping_request(tmp_path / "stay.txt", 1.5))
        leaves = call(path, _sleeping_request(pid_file, 20))
        await _read_pid(pid_file)
        await _read_pid(tmp_path / "stay.txt")

        leaves.leave()
        await asyncio.wait_for(leaves.task, timeout=10)  # type: ignore[attr-defined]
        await asyncio.wait_for(stays.task, timeout=20)  # type: ignore[attr-defined]

        assert leaves.status() == 499
        assert stays.status() == 200
        assert stays.json_body()["status"] == "success"


def _install_named(project: Path) -> None:
    """The named route needs the inline ensemble installed in the project."""
    _ensemble(project / ".llm-orc", "inline-top", INLINE["agents"])


class TestAPreflightIsTiedToItsConnectionToo:
    async def test_a_disconnect_during_the_gate_cancels_it_and_removes_the_layer(
        self,
        app: Any,
        state_dir: Path,
        pid_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entered = asyncio.Event()

        async def slow_gate(self: Any, *args: Any, **kwargs: Any) -> Any:
            assert _runs(state_dir), "the layer exists while the gate runs"
            entered.set()
            await asyncio.sleep(30)

        monkeypatch.setattr(ExecutionHandler, "_gate", slow_gate)
        path = "/api/ensembles/preflight"
        wire = Wire(_sleeping_request(pid_file, 20))
        task = asyncio.create_task(app(_scope(path), wire.receive, wire.send))
        await asyncio.wait_for(entered.wait(), timeout=10)

        started = time.monotonic()
        wire.leave()
        await asyncio.wait_for(task, timeout=10)

        assert wire.status() == 499
        assert _runs(state_dir) == []
        assert time.monotonic() - started < 5
