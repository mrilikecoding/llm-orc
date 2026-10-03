"""``llm-orc invoke --preflight`` (Arc 6, part 2): gate and print, run
nothing. The real command through ``CliRunner`` on a temp project, locally
and against the second real service through the transport seam."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.cli import cli
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.web.server import create_app
from tests.unit.cli.test_invoke_remote import (  # noqa: F401
    REMOTE_URL,
    Canned,
    Remote,
    _canned,
    _write_top,
    _write_top_with_remote_only_child,
    in_project,
    remote,
    remotes,
)
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    _profile,
    listing,
    project,
    state_dir,
)


def _preflight(*args: str) -> Result:
    return CliRunner().invoke(cli, ["invoke", "top", "--preflight", *args])


def _remote_preflight(*args: str) -> Result:
    return _preflight("--remote", "remote-host", *args)


def _nothing_ran(remote: Remote) -> None:
    runs = remote.state / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []
    assert not list(remote.project.rglob("ran.marker"))
    assert not (remote.state / "artifacts").exists()
    assert not (remote.config / "llm-orc" / "bundles").exists()


class TestARemotePreflight:
    def test_a_profile_the_remote_lacks_prints_the_whole_table_and_exits_1(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project, with_profile=True)
        before = remote.tree()

        result = _remote_preflight("--output-format", "text")

        assert result.exit_code == 1, result.output
        lines = result.stdout.splitlines()
        assert lines[0] == "Preflight: not runnable"
        header = next(line for line in lines if line.startswith("kind"))
        assert header.split() == ["kind", "name", "status", "via", "resolve"]
        statuses = {
            line.split()[1]: line.split()[2]
            for line in lines[lines.index(header) + 1 :]
        }
        assert statuses["seat"] == "missing_profile"
        assert "ready" in statuses.values()
        assert [url for url, _ in remote.calls] == [
            REMOTE_URL + "/api/ensembles/preflight"
        ]
        assert remote.tree() == before
        _nothing_ran(remote)

    def test_bind_reads_runnable_exits_0_and_names_the_binding(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project, with_profile=True)
        _profile(remote.project / ".llm-orc", "other", model="mock-other")
        before = remote.tree()

        result = _remote_preflight("--bind", "seat=other", "--output-format", "text")

        assert result.exit_code == 0, result.output
        assert result.stdout.splitlines()[0] == "Preflight: runnable"
        assert "Bindings that would apply: seat -> other" in result.stdout
        assert remote.calls[0][1]["bind"] == {"seat": "other"}
        assert remote.tree() == before
        _nothing_ran(remote)

    def test_with_profile_ships_the_definition_and_it_reads_runnable(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project, with_profile=True)

        result = _remote_preflight("--with-profile", "seat", "--output-format", "text")

        assert result.exit_code == 0, result.output
        assert "seat" in remote.calls[0][1]["profiles"]

    @pytest.mark.parametrize("fmt", [[], ["--output-format", "text"]])
    def test_rich_and_text_both_print_a_row_for_every_dependency(
        self, in_project: Path, remote: Remote, fmt: list[str]
    ) -> None:
        _write_top(in_project, with_profile=True)

        result = _remote_preflight(*fmt)

        assert result.exit_code == 1, result.output
        for name in ("top", "mark.py", "seat"):
            assert name in result.stdout

    def test_json_mode_prints_the_answer_alone(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project, with_profile=True)

        result = _remote_preflight("--output-format", "json")

        assert result.exit_code == 1
        document = json.loads(result.stdout)
        assert document["runnable"] is False
        assert document["bindings"] == {}
        assert {d["name"]: d["status"] for d in document["dependencies"]}[
            "seat"
        ] == "missing_profile"
        assert result.stderr == ""

    def test_a_runnable_json_answer_exits_0(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        result = _remote_preflight("--output-format", "json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["runnable"] is True

    def test_pull_is_sent_and_said_not_done(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        result = _remote_preflight("--pull", "--output-format", "text")

        assert remote.calls[0][1]["pull"] is True
        assert "nothing was pulled" in result.stdout

    def test_the_request_has_no_persist_and_no_input(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        _remote_preflight("--output-format", "text")

        (_, request) = remote.calls[0]
        assert "persist" not in request
        assert request["input"] == ""

    def test_what_the_closure_leaves_out_is_said_on_stderr(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top_with_remote_only_child(in_project, remote)

        result = _remote_preflight("--output-format", "json")

        assert "Left to the remote (not found locally): ensemble 'kids/only'" in (
            result.stderr
        )
        assert result.exit_code == 0, result.output


class TestWhatIsAnAnswer:
    DOCUMENT = {"runnable": True, "dependencies": [], "bindings": {}}

    def test_a_preflight_document_is_accepted(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, Canned(200, json.dumps(self.DOCUMENT)))

        result = _remote_preflight("--output-format", "json")

        assert result.exit_code == 0, result.output

    def test_the_refusal_envelope_is_accepted_and_exits_1(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        envelope = {
            "status": "error",
            "has_errors": True,
            "error": {
                "kind": "invalid_request",
                "message": "bad bind",
                "dependencies": [],
            },
        }
        _canned(monkeypatch, Canned(200, json.dumps(envelope)))

        result = _remote_preflight("--output-format", "text")

        assert result.exit_code == 1
        assert "invalid_request" in result.stdout
        assert "bad bind" in result.stdout

    @pytest.mark.parametrize(
        "answer",
        [
            Canned(200, "<html>the web ui</html>"),
            Canned(200, json.dumps({"status": "success", "has_errors": False})),
            Canned(200, json.dumps({"runnable": "yes"})),
            Canned(200, "[]"),
            Canned(404, '{"detail": "Not Found"}'),
            Canned(422, '{"detail": "extra forbidden"}'),
        ],
        ids=["html", "run-result", "runnable-not-bool", "array", "404", "422"],
    )
    def test_anything_else_exits_1_naming_the_remote(
        self,
        in_project: Path,
        remotes: None,
        monkeypatch: pytest.MonkeyPatch,
        answer: Canned,
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, answer)

        result = _remote_preflight("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "Remote 'remote-host'" in result.output
        assert "Traceback" not in result.output

    def test_a_connection_error_is_a_remote_error(
        self, in_project: Path, remotes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_top(in_project)
        _canned(monkeypatch, httpx.ConnectError("refused"))

        result = _remote_preflight("--output-format", "text")

        assert result.exit_code == 1
        assert "could not reach" in result.output
        assert "refused" in result.output


class TestRefusedBeforeAnythingIsSent:
    def test_persist_is_a_usage_error(self, in_project: Path, remote: Remote) -> None:
        _write_top(in_project)

        result = _remote_preflight("--persist", "global")

        assert result.exit_code == 2
        assert "--preflight" in result.output
        assert "--persist" in result.output
        assert remote.calls == []

    def test_persist_is_a_usage_error_without_a_remote_too(
        self, in_project: Path
    ) -> None:
        result = _preflight("--persist", "global")

        assert result.exit_code == 2

    def test_an_unknown_remote_lists_the_known_names(
        self, in_project: Path, remote: Remote
    ) -> None:
        _write_top(in_project)

        result = _preflight("--remote", "nowhere")

        assert result.exit_code == 1
        assert "known remotes: remote-host" in result.output
        assert remote.calls == []

    def test_an_ensemble_that_is_not_there_is_refused(
        self, in_project: Path, remote: Remote
    ) -> None:
        result = _remote_preflight()

        assert result.exit_code == 1
        assert "not found" in result.output
        assert remote.calls == []


class TestALocalPreflight:
    def _service_rows(self, project: Path, **request: Any) -> list[dict[str, Any]]:
        """REST's answer on a fresh service over the same project."""
        service = OrchestraService()
        assert service.handle_set_project(str(project))["status"] == "ok"
        previous = web_api._orchestra_service
        web_api._orchestra_service = service
        try:
            with TestClient(create_app()) as client:
                response = client.post(
                    "/api/ensembles/preflight", json={"ensemble_name": "top", **request}
                )
        finally:
            web_api._orchestra_service = previous
        rows: list[dict[str, Any]] = response.json()["dependencies"]
        return rows

    def test_the_rows_match_rests_and_nothing_ran(
        self, in_project: Path, state_dir: Path
    ) -> None:
        marker = _write_top(in_project)
        _ensemble(
            in_project / ".llm-orc",
            "top",
            [
                {"name": "first", "script": "mark.py"},
                {"name": "w", "model_profile": "ghost", "depends_on": ["first"]},
            ],
        )

        result = _preflight("--output-format", "json")

        assert result.exit_code == 1, result.output
        document = json.loads(result.stdout)
        assert document["runnable"] is False
        statuses = {d["name"]: d["status"] for d in document["dependencies"]}
        assert statuses["ghost"] == "missing_profile"
        assert document["dependencies"] == self._service_rows(in_project)
        assert not marker.exists()
        runs = state_dir / "runs"
        assert not runs.exists() or list(runs.iterdir()) == []

    def test_bind_reads_runnable_and_exits_0(self, in_project: Path) -> None:
        marker = _write_top(in_project, with_profile=True)
        _profile(in_project / ".llm-orc", "other", model="mock-other")

        result = _preflight("--bind", "seat=other", "--output-format", "text")

        assert result.exit_code == 0, result.output
        assert result.stdout.splitlines()[0] == "Preflight: runnable"
        assert "Bindings that would apply: seat -> other" in result.stdout
        assert not marker.exists()

    def test_it_does_not_wait_for_input(self, in_project: Path) -> None:
        _write_top(in_project)

        result = CliRunner().invoke(cli, ["invoke", "top", "--preflight"], input=None)

        assert result.exit_code == 0, result.output

    def test_an_invalid_bind_key_is_the_refusal_envelope_and_exits_1(
        self, in_project: Path
    ) -> None:
        _write_top(in_project)

        result = _preflight("--bind", "unnamed=other", "--output-format", "text")

        assert result.exit_code == 1
        assert "invalid_request" in result.stdout
        assert "unnamed" in result.stdout

    def test_a_missing_ensemble_is_a_message(self, in_project: Path) -> None:
        result = _preflight()

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_pull_is_not_performed_and_is_said(self, in_project: Path) -> None:
        _write_top(in_project)

        result = _preflight("--pull", "--output-format", "text")

        assert "nothing was pulled" in result.stdout
