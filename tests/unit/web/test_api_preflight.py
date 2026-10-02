"""``POST /api/ensembles/preflight`` (Arc 6, part 2): the gate over the
run's view, answered without running. Over one real OrchestraService on
temp project, state and global dirs; the router is the listing fake and a
fake pull, so nothing is called out."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_orc.providers.llama_server import LlamaServerClient
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    _profile,
    _runs,
    _trees,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_bundles import (  # noqa: F401
    _bundle_file,
    _pack,
    _run,
    client,
)

PREFLIGHT = "/api/ensembles/preflight"


def _marking_request(marker: Path, **more: Any) -> dict[str, Any]:
    """A root with a script that writes ``marker`` when it runs and an
    agent on profile ``seat``, which the host lacks."""
    script = (
        "import json, sys\n"
        "sys.stdin.read()\n"
        f'open(r"{marker}", "a").write("ran")\n'
        'print(json.dumps({"success": True, "data": {}}))\n'
    )
    return {
        "ensemble": {
            "name": "pf",
            "description": "pf",
            "agents": [
                {"name": "first", "script": "tools/mark.py"},
                {"name": "w", "model_profile": "seat", "depends_on": ["first"]},
            ],
        },
        "scripts": {"tools/mark.py": script},
        "input": "hi",
        **more,
    }


def _preflight(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post(PREFLIGHT, json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


def _statuses(document: dict[str, Any]) -> dict[str, str]:
    return {d["name"]: d["status"] for d in document["dependencies"]}


class TestTheAnswer:
    def test_a_profile_the_host_lacks_is_runnable_false_with_its_row(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"

        document = _preflight(client, _marking_request(marker))

        assert document["runnable"] is False
        assert "error" not in document
        assert _statuses(document)["seat"] == "missing_profile"
        assert document["bindings"] == {}
        assert not marker.exists()

    def test_a_bind_reads_runnable_and_names_the_binding_and_nothing_ran(
        self,
        client: TestClient,
        project: Path,
        state_dir: Path,
        tmp_path: Path,
    ) -> None:
        marker = tmp_path / "marker.txt"
        _profile(project / ".llm-orc", "other", model="mock-other")
        before = _trees(project, state_dir)

        document = _preflight(client, _marking_request(marker, bind={"seat": "other"}))

        assert document["runnable"] is True
        assert document["bindings"] == {"seat": "other"}
        assert not marker.exists()
        assert _runs(state_dir) == []
        assert _trees(project, state_dir) == before

    def test_the_rows_are_the_ones_execute_refuses_with(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        body = _marking_request(tmp_path / "marker.txt")

        document = _preflight(client, body)
        refusal = _run(client, body)

        assert refusal["error"]["kind"] == "not_equipped"
        assert document["dependencies"] == refusal["error"]["dependencies"]

    def test_the_input_is_optional(self, client: TestClient, tmp_path: Path) -> None:
        body = _marking_request(tmp_path / "marker.txt")
        del body["input"]

        assert _preflight(client, body)["runnable"] is False

    def test_a_named_root_is_judged_over_the_hosts_tiers(
        self, client: TestClient, project: Path
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "nope"}])

        document = _preflight(client, {"ensemble_name": "top"})

        assert document["runnable"] is False
        assert _statuses(document)["nope"] == "missing_profile"

    def test_a_bundle_root_is_expanded_and_judged_as_a_run_would(
        self, client: TestClient, state_dir: Path, project: Path
    ) -> None:
        assert _run(client, _pack(persist="global"))["status"] == "success"
        before = _trees(project, state_dir)

        document = _preflight(client, {"ensemble_name": "pack"})

        assert document["runnable"] is True
        assert _trees(project, state_dir) == before


class TestAnInvalidRequestIsTheRefusalEnvelope:
    @pytest.mark.parametrize(
        ("change", "named"),
        [
            ({"bind": {"unnamed": "other"}}, "unnamed"),
            ({"ensemble_name": "pf"}, "ensemble"),
            ({"ensemble": {"name": "pf", "agents": "not a list"}}, "does not load"),
        ],
        ids=["bind-key", "both-roots", "root-does-not-load"],
    )
    def test_it_is_http_200_with_the_execute_envelope(
        self,
        client: TestClient,
        project: Path,
        state_dir: Path,
        tmp_path: Path,
        change: dict[str, Any],
        named: str,
    ) -> None:
        _profile(project / ".llm-orc", "other", model="mock-other")
        body = {**_marking_request(tmp_path / "marker.txt"), **change}

        document = _preflight(client, body)

        assert document["status"] == "error"
        assert document["error"]["kind"] == "invalid_request"
        assert named in document["error"]["message"]
        assert _runs(state_dir) == []


class TestNothingIsWritten:
    def test_persist_global_writes_no_bundle(
        self, client: TestClient, project: Path, state_dir: Path
    ) -> None:
        before = _trees(project, state_dir)

        document = _preflight(client, _pack(persist="global"))

        assert document["runnable"] is True
        assert "persisted" not in document
        assert not _bundle_file().exists()
        assert not _bundle_file().parent.exists()
        assert _runs(state_dir) == []
        assert _trees(project, state_dir) == before

    def test_a_persist_over_a_name_a_tier_holds_is_still_invalid(
        self, client: TestClient, project: Path
    ) -> None:
        _ensemble(project / ".llm-orc", "pack", [{"name": "s", "script": "echo hi"}])

        document = _preflight(client, _pack(persist="global"))

        assert document["error"]["kind"] == "invalid_request"


class TestPull:
    @pytest.fixture
    def pulls(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        def pull(self: Any, model: str, **kwargs: Any) -> Any:
            calls.append(model)
            return {"status": "loaded", "failed": False, "exit_code": None}

        monkeypatch.setattr(LlamaServerClient, "pull", pull)
        return calls

    def _pullable(self, project: Path) -> dict[str, Any]:
        _profile(project / ".llm-orc", "seat", hf_repo="x/y:Q4")
        return {
            "ensemble": {
                "name": "pf",
                "description": "pf",
                "agents": [{"name": "w", "model_profile": "seat"}],
            },
            "input": "",
        }

    def test_pull_true_pulls_nothing_and_the_row_stays_pullable(
        self, client: TestClient, project: Path, pulls: list[str]
    ) -> None:
        body = {**self._pullable(project), "pull": True}

        document = _preflight(client, body)

        assert pulls == []
        assert document["runnable"] is False
        assert _statuses(document)["seat"] == "pullable"
        assert document["pull_requested"] is True

    def test_without_pull_the_answer_does_not_say_one_was_asked_for(
        self, client: TestClient, project: Path, pulls: list[str]
    ) -> None:
        document = _preflight(client, self._pullable(project))

        assert "pull_requested" not in document
