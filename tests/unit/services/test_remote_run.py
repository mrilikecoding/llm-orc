"""``run_remote``'s failures say whether anything was sent (Arc 5,
Task 10). Through the real function on a real service's lookups, with the
one transport seam replaced by canned answers. Nothing opens a
connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from llm_orc.services import remote_run
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.services.remote_run import RemoteRunError, run_remote
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    listing,
    project,
    service,
    state_dir,
)

REMOTE_URL = "https://llm-orc.remote.example"


class Canned:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)


@pytest.fixture
def remotes() -> None:
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"remotes": {"remote-host": {"url": REMOTE_URL}}}))


def _run(
    service: OrchestraService, root: str = "top", remote: str = "remote-host"
) -> dict[str, Any]:
    return run_remote(
        root,
        remote,
        find_root=service.find_ensemble_by_name,
        config_manager=service.config_manager,
        project_dir=service.project_path,
        input_text="hi",
    )


def _answer(monkeypatch: pytest.MonkeyPatch, answer: Canned | BaseException) -> None:
    def post(url: str, request: Any) -> Any:
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(remote_run, "post_run", post)


class TestTheKindOfAFailure:
    def test_an_unknown_remote_sent_nothing(
        self, remotes: None, service: OrchestraService, project: Path
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "s", "script": "echo hi"}])

        with pytest.raises(RemoteRunError) as raised:
            _run(service, remote="nowhere")

        assert raised.value.kind == "invalid_request"

    def test_a_closure_that_cannot_ship_sent_nothing(
        self, remotes: None, service: OrchestraService
    ) -> None:
        with pytest.raises(RemoteRunError) as raised:
            _run(service, root="ghost")

        assert raised.value.kind == "invalid_request"

    def test_an_interactive_script_sent_nothing(
        self, remotes: None, service: OrchestraService, project: Path
    ) -> None:
        scripts = project / ".llm-orc" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "get_user_input.py").write_text("print(input())\n")
        _ensemble(
            project / ".llm-orc", "top", [{"name": "a", "script": "get_user_input.py"}]
        )

        with pytest.raises(RemoteRunError) as raised:
            _run(service)

        assert raised.value.kind == "invalid_request"

    @pytest.mark.parametrize(
        "answer",
        [
            requests.ConnectionError("connection refused"),
            Canned(200, "<!doctype html>"),
            Canned(422, '{"detail": "extra inputs are not permitted"}'),
        ],
        ids=["unreachable", "html", "422"],
    )
    def test_a_request_that_went_out_is_a_remote_error(
        self,
        remotes: None,
        service: OrchestraService,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        answer: Canned | BaseException,
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "s", "script": "echo hi"}])
        _answer(monkeypatch, answer)

        with pytest.raises(RemoteRunError) as raised:
            _run(service)

        assert raised.value.kind == "remote_error"
