"""One-run injection through the real surfaces (Arc 4, Task 7).

REST (both execute routes) and the MCP ``invoke`` tool, over one real
OrchestraService on temp project, state and global dirs. The service-level
pins live in tests/unit/services/test_one_run_injection.py; these prove the
request keys survive each surface's own parsing.
"""

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
    INLINE,
    _bound_ensemble,
    _ensemble,
    _inline_request,
    _kinds,
    _profile,
    _runs,
    _tag,
    _trees,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_mcp import _ACCEPT_HEADERS, _initialize

Call = Callable[[str | None, dict[str, Any]], dict[str, Any]]

INLINE_SURFACES = ["rest-new", "mcp"]
NAMED_SURFACES = ["rest-new", "rest-named", "mcp"]


@pytest.fixture
def client(
    service: OrchestraService, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """The app over the test's service, shared by REST and the /mcp mount."""
    monkeypatch.setattr(web_api, "_orchestra_service", service)
    with TestClient(create_app()) as test_client:
        yield test_client


def _mcp_result(
    client: TestClient, session_id: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """The tool result: the last JSON-RPC message of the SSE reply.

    A refused run reports on the log channel first, so the shared
    first-message parser would return the notification.
    """
    response = client.post(
        "/mcp",
        headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "invoke", "arguments": arguments},
        },
    )
    messages = [
        json.loads(line[len("data:") :])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    final = [m for m in messages if m.get("id") == 9]
    assert len(final) == 1, messages
    assert "result" in final[0], messages
    return dict(final[0]["result"]["structuredContent"])


def _caller(client: TestClient, surface: str) -> Call:
    """One way to make a run call: ``call(named_root, request)``.

    ``request`` carries every key except the root; an inline root is the
    request's own ``ensemble``. The result is the run's result dict.
    """
    if surface == "mcp":
        _, session_id = _initialize(client)

        def call_mcp(named: str | None, request: dict[str, Any]) -> dict[str, Any]:
            arguments = {k: v for k, v in request.items() if k != "input"}
            arguments["input_data"] = request["input"]
            if named:
                arguments["ensemble_name"] = named
            return _mcp_result(client, session_id, arguments)

        return call_mcp

    def call_rest(named: str | None, request: dict[str, Any]) -> dict[str, Any]:
        if surface == "rest-named":
            assert named is not None
            response = client.post(f"/api/ensembles/{named}/execute", json=request)
        else:
            body = {**request, "ensemble_name": named} if named else request
            response = client.post("/api/ensembles/execute", json=body)
        assert response.status_code == 200, response.text
        return dict(response.json())

    return call_rest


def _warm(client: TestClient, project: Path) -> None:
    # The first run of any kind makes the serve's credential storage: serve
    # state, not the injection's. Start from a serve that has served once.
    _ensemble(project / ".llm-orc", "warm", [{"name": "s", "script": "echo hi"}])
    response = client.post("/api/ensembles/warm/execute", json={"input": "hi"})
    assert response.status_code == 200, response.text


def _marker_source(marker: Path) -> str:
    return (
        "import json, sys\n"
        "sys.stdin.read()\n"
        f'open(r"{marker}", "a").write("ran")\n'
        'print(json.dumps({"success": True, "data": {"ok": True}}))\n'
    )


def _bound_inline(marker: Path) -> dict[str, Any]:
    """An inline ensemble naming profile ``a`` after a phase-0 marker."""
    return {
        "ensemble": {
            "name": "inline-bound",
            "description": "inline",
            "agents": [
                {"name": "first", "script": "probe/mark.py"},
                {"name": "w", "model_profile": "a", "depends_on": ["first"]},
            ],
        },
        "scripts": {"probe/mark.py": _marker_source(marker)},
        "profiles": {"b": {"provider": "llama-server", "model": "mock-b"}},
        "input": "hi",
    }


class TestInlineRoot:
    @pytest.mark.parametrize("surface", INLINE_SURFACES)
    def test_it_runs_and_every_tree_is_the_same_afterwards(
        self, surface: str, client: TestClient, project: Path, state_dir: Path
    ) -> None:
        _warm(client, project)
        before = _trees(project, state_dir)

        result = _caller(client, surface)(None, _inline_request())

        assert result["status"] == "success", result
        assert _tag(result, "s") == "injected"
        assert _trees(project, state_dir) == before
        assert _runs(state_dir) == []

    @pytest.mark.parametrize("surface", INLINE_SURFACES)
    def test_a_missing_profile_refuses_and_the_same_call_with_bind_runs(
        self, surface: str, client: TestClient, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"
        call = _caller(client, surface)

        refused = call(None, _bound_inline(marker))

        assert refused["status"] == "error"
        assert refused["results"] == {}
        assert refused["error"]["kind"] == "not_equipped"
        assert _kinds(refused)["a"] == "missing_profile"
        assert not marker.exists()

        result = call(None, {**_bound_inline(marker), "bind": {"a": "b"}})

        assert result["status"] == "success", result
        assert result["bindings"] == {"a": "b"}
        assert marker.exists()


class TestNamedRoot:
    @pytest.mark.parametrize("surface", NAMED_SURFACES)
    def test_injected_profile_and_script_run_and_leave_every_tree_alone(
        self, surface: str, client: TestClient, project: Path, state_dir: Path
    ) -> None:
        _warm(client, project)
        _ensemble(project / ".llm-orc", "named-top", INLINE["agents"])
        before = _trees(project, state_dir)
        request = _inline_request()
        body = {k: request[k] for k in ("profiles", "scripts", "input")}

        result = _caller(client, surface)("named-top", body)

        assert result["status"] == "success", result
        assert _tag(result, "s") == "injected"
        # An installed root keeps its own run artifact; nothing else changes.
        after = _trees(project, state_dir)
        after["state"] = {
            k: v for k, v in after["state"].items() if "artifacts/named-top/" not in k
        }
        assert after == before
        assert _runs(state_dir) == []

    @pytest.mark.parametrize("surface", NAMED_SURFACES)
    def test_a_missing_profile_refuses_and_the_same_call_with_bind_runs(
        self,
        surface: str,
        client: TestClient,
        project: Path,
        tmp_path: Path,
    ) -> None:
        marker = _bound_ensemble(project, tmp_path)
        _profile(project / ".llm-orc", "b", model="mock-other")
        call = _caller(client, surface)

        refused = call("top", {"input": "hi"})

        assert refused["error"]["kind"] == "not_equipped"
        assert _kinds(refused)["a"] == "missing_profile"
        assert not marker.exists()

        result = call("top", {"input": "hi", "bind": {"a": "b"}})

        assert result["status"] == "success", result
        assert result["bindings"] == {"a": "b"}
        assert marker.exists()


class TestRequestBodies:
    @pytest.mark.parametrize(
        "path", ["/api/ensembles/execute", "/api/ensembles/top/execute"]
    )
    def test_a_misspelled_key_is_a_422_and_nothing_runs(
        self, path: str, client: TestClient, project: Path, state_dir: Path
    ) -> None:
        _warm(client, project)
        before = _trees(project, state_dir)
        body = {**_inline_request(), "profile": {"x": {}}}
        if path.endswith("/top/execute"):
            body = {"input": "hi", "bnid": {"a": "b"}}

        response = client.post(path, json=body)

        assert response.status_code == 422, response.text
        assert _trees(project, state_dir) == before

    @pytest.mark.parametrize("root_key", ["ensemble_name", "ensemble"])
    def test_a_root_in_the_body_of_the_named_route_is_a_422(
        self, root_key: str, client: TestClient
    ) -> None:
        root: Any = INLINE if root_key == "ensemble" else "other"

        response = client.post(
            "/api/ensembles/top/execute", json={"input": "hi", root_key: root}
        )

        assert response.status_code == 422, response.text

    def test_get_execute_still_reads_an_ensemble_named_execute(
        self, client: TestClient, project: Path
    ) -> None:
        _ensemble(project / ".llm-orc", "execute", [{"name": "s", "script": "x"}])

        response = client.get("/api/ensembles/execute")

        assert response.status_code == 200, response.text
        assert response.json()["name"] == "execute"
