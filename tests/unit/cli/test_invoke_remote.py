"""``llm-orc invoke --remote`` (Arc 5, Task 8). The real command through
``CliRunner`` on a temp project. The one transport seam,
``remote_run.transport``, is an ``httpx`` transport: the second real
service's ASGI app (its own project, config and state dirs), or canned
responses. The real client code runs on it, and nothing here opens a
connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest
import yaml
from click.testing import CliRunner, Result

import llm_orc.web.api as web_api
from llm_orc.cli import cli
from llm_orc.services import remote_run
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.web.server import create_app
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    _profile,
    listing,
    project,
    state_dir,
)

REMOTE_URL = "https://llm-orc.remote.example"

_MARK = (
    "import json, os, sys\n"
    "sys.stdin.read()\n"
    "here = os.path.dirname(os.path.abspath(__file__))\n"
    'open(os.path.join(here, "ran.marker"), "w").write("ran")\n'
    'print(json.dumps({"success": True, "data": "hello from the remote"}))\n'
)


@pytest.fixture
def in_project(project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The cwd is the project, as it is for a person running the CLI."""
    monkeypatch.chdir(project)
    return project


@pytest.fixture
def remotes(in_project: Path) -> None:
    """``remote-host`` is a name in the caller's global config."""
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"remotes": {"remote-host": {"url": REMOTE_URL}}}))


class _Recording(httpx.AsyncBaseTransport):
    """Records each request as (url, JSON body), then passes it on."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> None:
        self._inner = inner
        self._calls = calls

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._calls.append((str(request.url), json.loads(request.content)))
        assert str(request.url) == REMOTE_URL + "/api/ensembles/execute"
        return await self._inner.handle_async_request(request)


class Delayed(httpx.AsyncBaseTransport):
    """Holds each request for ``seconds`` (the loop stays free) before
    passing it on."""

    def __init__(self, inner: httpx.AsyncBaseTransport, seconds: float) -> None:
        self._inner = inner
        self._seconds = seconds

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self._seconds)
        return await self._inner.handle_async_request(request)


class Remote:
    """A second real service on its own project, config and state dirs."""

    def __init__(self, base: Path) -> None:
        self.project = base / "remote-proj"
        self.config = base / "remote-xdg"
        self.state = base / "remote-state"
        (self.project / ".llm-orc" / "ensembles").mkdir(parents=True)
        self.config.mkdir()
        self.state.mkdir()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.transport: httpx.AsyncBaseTransport = _Recording(
            httpx.ASGITransport(app=self.asgi), self.calls
        )
        with self._env():
            self.service = OrchestraService()
            status = self.service.handle_set_project(str(self.project))["status"]
        assert status == "ok"

    @contextmanager
    def _env(self) -> Iterator[None]:
        env = {
            "XDG_CONFIG_HOME": str(self.config),
            "LLM_ORC_STATE_DIR": str(self.state),
        }
        with mock.patch.dict(os.environ, env):
            yield

    async def asgi(self, scope: Any, receive: Any, send: Any) -> None:
        """The second service's REST app, with its own service and dirs
        in place for as long as it handles the request."""
        previous = web_api._orchestra_service
        web_api._orchestra_service = self.service
        try:
            with self._env():
                await create_app()(scope, receive, send)
        finally:
            web_api._orchestra_service = previous

    def post_rest(self, request: dict[str, Any]) -> Any:
        """A direct REST call to the second service, outside the seam;
        the JSON answer."""

        async def call() -> Any:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.asgi), base_url=REMOTE_URL
            ) as client:
                response = await client.post(remote_run.EXECUTE_PATH, json=request)
                return response.json()

        return asyncio.run(call())

    def tree(self) -> dict[str, str]:
        """Every file and empty directory under the remote's three trees,
        by content hash. The state dir's empty ``runs/`` is where run
        layers live and stays after a run; a run left in it shows. The
        serve's encryption key is made by its first run of any kind: serve
        state, not the run's, so it is left out."""
        found: dict[str, str] = {}
        for root in (self.project, self.config, self.state):
            for path in sorted(root.rglob("*")):
                if path.name == ".encryption_key":
                    continue
                if path.is_file():
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    found[str(path)] = digest
                elif path.name != "runs" and not any(path.iterdir()):
                    found[str(path) + "/"] = "empty directory"
        return found


@pytest.fixture
def remote(
    tmp_path: Path, remotes: None, listing: Any, monkeypatch: pytest.MonkeyPatch
) -> Remote:
    remote = Remote(tmp_path)
    monkeypatch.setattr(remote_run, "transport", remote.transport)
    return remote


class Canned:
    """A response the seam returns, in place of a serve."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)


def _canned(
    monkeypatch: pytest.MonkeyPatch, answer: Canned | BaseException
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append((str(request.url), json.loads(request.content)))
        if isinstance(answer, BaseException):
            raise answer
        return httpx.Response(answer.status_code, text=answer.text)

    monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))
    return calls


def _write_top(project: Path, *, with_profile: bool = False) -> Path:
    """A root with a script that writes a marker next to itself, and an
    agent on profile ``seat``, which only the caller's project holds."""
    scripts = project / ".llm-orc" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "mark.py").write_text(_MARK)
    agents: list[dict[str, Any]] = [{"name": "first", "script": "mark.py"}]
    if with_profile:
        _profile(project / ".llm-orc", "seat")
        agents.append({"name": "w", "model_profile": "seat", "depends_on": ["first"]})
    _ensemble(project / ".llm-orc", "top", agents)
    return scripts / "ran.marker"


def _write_top_with_remote_only_child(project: Path, remote: Remote) -> None:
    """A root whose only agent runs the child ``kids/only``, which exists
    on the remote and not in the caller's project."""
    _ensemble(project / ".llm-orc", "top", [{"name": "k", "ensemble": "kids/only"}])
    scripts = remote.project / ".llm-orc" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "say.py").write_text(_MARK)
    _ensemble(
        remote.project / ".llm-orc",
        "kids/only",
        [{"name": "a", "script": "say.py"}],
    )


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(cli, ["invoke", "top", "hi", *args])


def _remote_invoke(*args: str) -> Result:
    return _invoke("--remote", "remote-host", *args)


class TestARemoteRun:
    @pytest.mark.parametrize("fmt", ["text", "json", "rich"])
    def test_a_run_prints_the_remotes_deliverable_and_exits_0(
        self, in_project: Path, remote: Remote, fmt: str
    ) -> None:
        _write_top(in_project)
        args = [] if fmt == "rich" else ["--output-format", fmt]

        result = _remote_invoke(*args)

        assert result.exit_code == 0, result.output
        assert "hello from the remote" in result.stdout
        assert len(remote.calls) == 1

    def test_json_mode_prints_one_document_and_nothing_else(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        result = _remote_invoke("--output-format", "json")

        document = json.loads(result.stdout)
        assert document["status"] == "success"
        assert document["deliverable"] is not None
        assert result.stderr == ""

    def test_nothing_ran_locally_and_nothing_persisted_on_the_remote(
        self, in_project: Path, remote: Remote
    ) -> None:
        local_marker = _write_top(in_project)
        before = remote.tree()

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 0, result.output
        assert not local_marker.exists()
        assert remote.tree() == before

    def test_the_request_carries_the_input_and_the_flags(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        _remote_invoke("--output-format", "text", "--pull", "--persist", "global")

        _, request = remote.calls[0]
        assert request["input"] == "hi"
        assert request["pull"] is True
        assert request["persist"] == "global"


class TestWhatWasLeftToTheRemote:
    LINE = "Left to the remote (not found locally): ensemble 'kids/only'\n"

    @pytest.mark.parametrize("fmt", ["text", "json", "rich"])
    def test_the_line_is_on_stderr_and_stdout_is_the_result_alone(
        self, in_project: Path, remote: Remote, fmt: str
    ) -> None:
        _write_top_with_remote_only_child(in_project, remote)
        args = [] if fmt == "rich" else ["--output-format", fmt]

        result = _remote_invoke(*args)

        assert result.exit_code == 0, result.output
        assert self.LINE in result.stderr
        assert "Left to the remote" not in result.stdout
        assert "hello from the remote" in result.stdout
        if fmt == "json":
            assert json.loads(result.stdout)["status"] == "success"

    def test_a_closure_that_ships_whole_prints_no_line(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        result = _remote_invoke("--output-format", "text")

        assert "Left to the remote" not in result.stderr
        assert "Left to the remote" not in result.stdout

    def test_several_labels_are_joined_on_one_line(
        self, in_project: Path, remote: Remote
    ) -> None:
        _ensemble(
            in_project / ".llm-orc",
            "top",
            [
                {"name": "k", "ensemble": "kids/only"},
                {"name": "g", "script": "tools/y.py"},
            ],
        )

        result = _remote_invoke("--output-format", "text")

        assert (
            "Left to the remote (not found locally): "
            "ensemble 'kids/only', script 'tools/y.py'\n"
        ) in result.stderr


class TestPersistingOnTheRemote:
    def test_the_bundle_lands_on_the_remote_and_runs_by_name_over_rest(
        self, in_project: Path, remote: Remote
    ) -> None:
        local_marker = _write_top(in_project)
        caller_config = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc"

        result = _remote_invoke("--output-format", "json", "--persist", "global")

        bundle = remote.config / "llm-orc" / "bundles" / "top.json"
        assert result.exit_code == 0, result.output
        assert bundle.is_file()
        before = remote.tree()
        by_name = remote.post_rest({"ensemble_name": "top", "input": "hi"})
        assert by_name["status"] == "success"
        assert by_name["deliverable"] == json.loads(result.stdout)["deliverable"]
        assert hashlib.sha256(bundle.read_bytes()).hexdigest() == before[str(bundle)]
        assert not local_marker.exists()
        assert not (caller_config / "bundles").exists()
        assert not (in_project / ".llm-orc" / "ensembles" / "top.json").exists()

    @pytest.mark.parametrize("fmt", ["text", "rich"])
    def test_the_stored_bundle_is_named_next_to_the_other_run_lines(
        self, in_project: Path, remote: Remote, fmt: str
    ) -> None:
        _write_top(in_project)
        args = [] if fmt == "rich" else ["--output-format", fmt]

        result = _remote_invoke("--persist", "global", *args)

        assert result.exit_code == 0, result.output
        assert "Bundle persisted: top" in result.stdout

    def test_json_mode_carries_every_key_of_the_remotes_document(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        result = _remote_invoke("--output-format", "json", "--persist", "global")

        document = json.loads(result.stdout)
        assert document["persisted"] == "top"
        assert document["raw_output"] is False

    def test_without_persist_no_line_and_no_key(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        text = _remote_invoke("--output-format", "text")
        as_json = _remote_invoke("--output-format", "json")

        assert "Bundle persisted" not in text.stdout
        assert "persisted" not in json.loads(as_json.stdout)

    def test_a_name_the_remotes_tiers_resolve_is_refused_with_its_message(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)
        _ensemble(remote.project / ".llm-orc", "top", [{"name": "x", "script": "x.py"}])
        before = remote.tree()

        result = _remote_invoke("--output-format", "text", "--persist", "global")

        assert result.exit_code == 1, result.output
        assert "cannot persist 'top'" in result.output
        assert "already has an ensemble of that name" in result.output
        assert remote.tree() == before
        assert not (remote.config / "llm-orc" / "bundles").exists()


class TestAProfileTheRemoteLacks:
    def test_it_prints_the_dependency_table_and_exits_1(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project, with_profile=True)

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "missing_profile" in result.stdout
        assert "seat" in result.stdout

    def test_bind_runs_it_and_shows_the_binding(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project, with_profile=True)
        _profile(remote.project / ".llm-orc", "other", model="mock-other")

        result = _remote_invoke("--bind", "seat=other", "--output-format", "text")

        assert result.exit_code == 0, result.output
        assert "Bindings applied: seat -> other" in result.stdout
        assert remote.calls[0][1]["bind"] == {"seat": "other"}

    def test_with_profile_ships_the_definition_and_it_runs(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project, with_profile=True)
        before = remote.tree()

        result = _remote_invoke("--with-profile", "seat", "--output-format", "text")

        assert result.exit_code == 0, result.output
        assert "seat" in remote.calls[0][1]["profiles"]
        assert remote.tree() == before


class TestAnAnswerThatIsNotAResult:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            (Canned(200, "<!doctype html><title>llm-orc</title>"), "200"),
            (Canned(200, '{"detail": "no status here"}'), "200"),
            (Canned(200, '{"status": "success", "has_errors": "no"}'), "200"),
            (Canned(404, '{"detail": "Not Found"}'), "404"),
            (
                Canned(422, '{"detail": "extra inputs are not permitted: persist"}'),
                "extra inputs are not permitted: persist",
            ),
            (httpx.ConnectError("connection refused"), "connection refused"),
        ],
        ids=["html", "no-status", "no-has-errors", "404", "422", "unreachable"],
    )
    def test_it_exits_1_and_names_the_remote(
        self,
        in_project: Path,
        remotes: None,
        monkeypatch: pytest.MonkeyPatch,
        answer: Canned | BaseException,
        expected: str,
    ) -> None:
        _write_top(in_project)
        calls = _canned(monkeypatch, answer)

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "remote-host" in result.output
        assert expected in result.output
        assert len(calls) == 1
        assert result.stdout == ""


class TestAMalformedRemoteUrl:
    """``httpx.InvalidURL`` is not an ``httpx.HTTPError``. Nothing was
    sent, so the answer is ``invalid_request``, never a traceback."""

    BAD = "http://[::1"

    def test_a_url_given_to_the_flag_exits_1_with_a_message(
        self, in_project: Path
    ) -> None:
        _write_top(in_project)

        result = _invoke("--remote", self.BAD, "--output-format", "text")

        assert result.exit_code == 1, result.output
        assert self.BAD in result.output
        assert "Traceback" not in result.output
        assert isinstance(result.exception, SystemExit)

    def test_a_url_in_the_config_exits_1_with_a_message(self, in_project: Path) -> None:
        _write_top(in_project)
        path = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"remotes": {"remote-host": {"url": self.BAD}}}))

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "remote-host" in result.output
        assert "Traceback" not in result.output
        assert isinstance(result.exception, SystemExit)

    def test_a_tab_given_to_the_flag_exits_1_naming_the_character(
        self, in_project: Path
    ) -> None:
        _write_top(in_project)

        result = _invoke("--remote", REMOTE_URL + "\t", "--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "not allowed in a URL" in result.output
        assert "'\\t'" in result.output
        assert "Traceback" not in result.output
        assert isinstance(result.exception, SystemExit)

    def test_a_config_url_ending_in_a_newline_exits_1_naming_it(
        self, in_project: Path
    ) -> None:
        _write_top(in_project)
        path = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"remotes:\n  remote-host:\n    url: |\n      {REMOTE_URL}\n")

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "remote-host" in result.output
        assert "not allowed in a URL" in result.output
        assert "'\\n'" in result.output
        assert isinstance(result.exception, SystemExit)

    def test_a_url_the_resolver_passes_and_httpx_refuses_exits_1(
        self, in_project: Path
    ) -> None:
        """The catch in ``run_remote`` is the named guard for this class."""
        _write_top(in_project)

        result = _invoke("--remote", "http://\u2603", "--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "not a usable URL" in result.output
        assert "Traceback" not in result.output
        assert isinstance(result.exception, SystemExit)


# Environments in which building ``httpx.AsyncClient(trust_env=True)``
# raises before anything is sent: (variable, value, what the message must
# name). ``HTTP_PROXY`` is read for an http URL only by the client, but a
# malformed one fails the build for every URL.
CLIENT_SETUP_FAILURES = [
    ("ALL_PROXY", "socks5://127.0.0.1:1080", "ALL_PROXY"),
    ("HTTPS_PROXY", "ftp://proxy.example", "HTTPS_PROXY"),
    ("SSL_CERT_FILE", "/nonexistent/ca.pem", "SSL_CERT_FILE"),
    ("HTTP_PROXY", "http://[::1", "HTTP_PROXY"),
]
CLIENT_SETUP_IDS = ["socks-proxy", "ftp-proxy", "stale-ca-file", "bad-http-proxy"]
# A remote the resolver refuses: (value given to --remote, what is wrong).
BAD_REMOTE_URLS = [
    ("http://127.0.0.1:80800", "port"),
    ("ftp://host.example", "http"),
    ("http:///x", "host"),
    ("http://[::1", "http://[::1"),
]


@pytest.fixture
def no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that reaches the real transport fails the test loudly."""

    async def refuse(*_: Any, **__: Any) -> httpx.Response:
        raise AssertionError("a connection was opened")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


@pytest.mark.usefixtures("no_connection")
class TestAClientThatCannotBeBuilt:
    """The real client, no transport seam: the environment's proxy and TLS
    settings can make building it raise. Nothing was sent, so the answer
    is a message naming the likely cause, never a traceback."""

    @pytest.mark.parametrize(
        ("variable", "value", "named"), CLIENT_SETUP_FAILURES, ids=CLIENT_SETUP_IDS
    )
    def test_it_exits_1_and_points_at_the_environment(
        self,
        in_project: Path,
        remotes: None,
        monkeypatch: pytest.MonkeyPatch,
        variable: str,
        value: str,
        named: str,
    ) -> None:
        _write_top(in_project)
        monkeypatch.setenv(variable, value)

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output
        assert "could not be set up" in result.output
        assert named in result.output
        assert "nothing was sent" in result.output
        assert "remote-host" in result.output


@pytest.mark.usefixtures("no_connection")
class TestARemoteTheResolverRefuses:
    @pytest.mark.parametrize(
        ("bad", "named"), BAD_REMOTE_URLS, ids=["port", "scheme", "host", "bracket"]
    )
    def test_it_exits_1_before_a_client_exists(
        self, in_project: Path, bad: str, named: str
    ) -> None:
        _write_top(in_project)

        result = _invoke("--remote", bad, "--output-format", "text")

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output
        assert named in result.output
        assert "could not be set up" not in result.output


class TestAnExceptionGroupFromThePost:
    def test_it_is_a_remote_error_naming_the_first_leaf(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)

        async def post(*_: Any) -> httpx.Response:
            raise ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [OverflowError("bind failed"), ValueError("other")],
            )

        monkeypatch.setattr(remote_run, "post_run", post)

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit)
        assert "could not reach" in result.output
        assert "bind failed" in result.output
        assert "other" not in result.output


class TestTheExitCodeFollowsTheStatus:
    @pytest.mark.parametrize("fmt", ["text", "json", "rich"])
    def test_status_error_exits_1_even_when_has_errors_is_false(
        self,
        in_project: Path,
        remotes: None,
        monkeypatch: pytest.MonkeyPatch,
        fmt: str,
    ) -> None:
        _write_top(in_project)
        answer = Canned(
            200,
            json.dumps({"status": "error", "has_errors": False, "results": {}}),
        )
        _canned(monkeypatch, answer)
        args = [] if fmt == "rich" else ["--output-format", fmt]

        result = _remote_invoke(*args)

        assert result.exit_code == 1, result.output

    def test_status_success_with_has_errors_true_exits_1(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        answer = Canned(
            200,
            json.dumps({"status": "success", "has_errors": True, "results": {}}),
        )
        _canned(monkeypatch, answer)

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 1, result.output


class TestRefusedBeforeAnythingIsSent:
    def _assert_refused(self, result: Result, calls: list[Any], expected: str) -> None:
        assert result.exit_code == 1, result.output
        assert expected in result.output
        assert calls == []

    def test_max_concurrent_is_refused(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = _remote_invoke("--max-concurrent", "2")

        self._assert_refused(result, calls, "--max-concurrent")

    def test_an_interactive_script_is_refused(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripts = in_project / ".llm-orc" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "get_user_input.py").write_text("print(input())\n")
        _ensemble(
            in_project / ".llm-orc",
            "top",
            [{"name": "ask", "script": "get_user_input.py"}],
        )
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = _remote_invoke()

        self._assert_refused(result, calls, "interactive")

    def test_an_unknown_remote_lists_the_known_names(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = _invoke("--remote", "nowhere")

        self._assert_refused(result, calls, "known remotes: remote-host")

    def test_a_closure_that_cannot_ship_is_refused(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = _remote_invoke("--with-profile", "no-such-profile")

        self._assert_refused(result, calls, "no-such-profile")

    def test_an_ensemble_that_is_not_there_is_refused(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = _remote_invoke()

        self._assert_refused(result, calls, "not found")


class TestFlagsThatNeedARemote:
    @pytest.mark.parametrize(
        "flag", [["--with-profile", "seat"], ["--persist", "global"]]
    )
    def test_without_remote_they_are_usage_errors(
        self, in_project: Path, flag: list[str]
    ) -> None:
        _write_top(in_project)

        result = _invoke(*flag)

        assert result.exit_code == 2
        assert "--remote" in result.output

    def test_persist_takes_only_global(self, in_project: Path) -> None:
        result = _remote_invoke("--persist", "project")

        assert result.exit_code == 2

    def test_help_has_an_example_of_each_new_flag(self) -> None:
        result = CliRunner().invoke(cli, ["invoke", "--help"])

        for example in (
            "--remote remote-host",
            "--with-profile",
            "--persist global",
        ):
            assert example in result.output


@pytest.fixture
def default_sigint() -> Iterator[None]:
    """The default SIGINT handler for the test, then the one found. asyncio
    installs its own handler only over the default one, so a handler an
    earlier test on the same worker left behind would take the signal."""
    saved = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, saved)


class TestWhileWaiting:
    def test_rich_mode_shows_the_remote_and_the_time_on_stderr(
        self, in_project: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        monkeypatch.setattr("llm_orc.cli_commands.WAIT_TICK_S", 0.01)
        monkeypatch.setattr(remote_run, "transport", Delayed(remote.transport, 0.1))

        result = _remote_invoke()

        assert result.exit_code == 0, result.output
        assert "remote-host" in result.stderr
        assert "s" in result.stderr

    @pytest.mark.parametrize("fmt", ["text", "json"])
    def test_text_and_json_print_nothing_on_stderr(
        self, in_project: Path, remote: Remote, fmt: str
    ) -> None:
        _write_top(in_project)

        result = _remote_invoke("--output-format", fmt)

        assert result.stderr == ""

    def test_ctrl_c_exits_130(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, KeyboardInterrupt())

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 130

    @pytest.mark.usefixtures("default_sigint")
    def test_ctrl_c_closes_the_connection(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real SIGINT while the request is out: the awaiting task is
        cancelled, so the transport sees the cancellation (a real client
        closes its connection there) and the command exits 130."""
        _write_top(in_project)
        closed: list[bool] = []

        class Hung(httpx.AsyncBaseTransport):
            async def handle_async_request(
                self, request: httpx.Request
            ) -> httpx.Response:
                signal.raise_signal(signal.SIGINT)
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    closed.append(True)
                    raise
                raise AssertionError("the request was not cancelled")

        monkeypatch.setattr(remote_run, "transport", Hung())

        result = _remote_invoke("--output-format", "text")

        assert result.exit_code == 130
        assert closed == [True]


def test_the_post_is_one_request_with_a_connect_timeout_and_no_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))

    asyncio.run(
        remote_run.post_run(
            remote_run.build_client("remote-host"),
            REMOTE_URL + "/api/ensembles/execute",
            {"a": 1},
        )
    )

    (request,) = seen
    assert str(request.url) == REMOTE_URL + "/api/ensembles/execute"
    assert json.loads(request.content) == {"a": 1}
    assert request.extensions["timeout"] == {
        "connect": 10,
        "read": None,
        "write": None,
        "pool": None,
    }


def test_a_redirect_is_not_followed_and_is_not_a_result(
    in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_top(in_project)
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://other.example/x"})

    monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))

    result = _remote_invoke("--output-format", "text")

    assert result.exit_code == 1, result.output
    assert "307" in result.output
    assert "https://other.example/x" in result.output
    assert seen == [REMOTE_URL + "/api/ensembles/execute"]


def test_a_redirects_location_is_capped_and_stripped_of_control_characters(
    in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_top(in_project)
    location = "https://other.example/\x85" + "a" * 300

    def handle(request: httpx.Request) -> httpx.Response:
        raw = [(b"location", location.encode("latin-1"))]
        return httpx.Response(307, headers=raw)

    monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))

    result = _remote_invoke("--output-format", "text")

    assert result.exit_code == 1, result.output
    assert "https://other.example/" + "a" * 20 in result.output
    assert "a" * 200 not in result.output
    assert "\x85" not in result.output
