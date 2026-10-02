"""``llm-orc remotes`` (Arc 6, part 1): the configured remotes, each with a
live ``GET /health`` probe. The real command through ``CliRunner``; the
transport seam is the second real service's ASGI app or canned answers, so
nothing here opens a connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from click.testing import CliRunner, Result

from llm_orc.cli import cli
from llm_orc.services import remote_run
from llm_orc.web import server as web_server
from tests.unit.cli.test_invoke_remote import (  # noqa: F401
    REMOTE_URL,
    Remote,
    in_project,
    remote,
    remotes,
)
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    listing,
    project,
    state_dir,
)


def _remotes(*args: str) -> Result:
    return CliRunner().invoke(cli, ["remotes", *args])


def _global_config(data: dict[str, Any]) -> None:
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "llm-orc" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _answer(
    monkeypatch: pytest.MonkeyPatch, answer: httpx.Response | BaseException
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))
    return seen


class TestAReachableRemote:
    @pytest.mark.parametrize("fmt", [[], ["--output-format", "text"]])
    def test_the_line_names_the_second_services_version(
        self, remote: Remote, monkeypatch: pytest.MonkeyPatch, fmt: list[str]
    ) -> None:
        monkeypatch.setattr(web_server, "get_version", lambda: "9.9.9")

        result = _remotes(*fmt)

        assert result.exit_code == 0, result.output
        line = f"remote-host  {REMOTE_URL}  reachable, llm-orc 9.9.9\n"
        assert result.stdout == line

    def test_json_mode_prints_the_list_as_an_array(
        self, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web_server, "get_version", lambda: "9.9.9")

        result = _remotes("--output-format", "json")

        assert json.loads(result.stdout) == [
            {
                "name": "remote-host",
                "url": REMOTE_URL,
                "reachable": True,
                "version": "9.9.9",
            }
        ]

    def test_the_probe_is_a_get_of_health_with_five_second_timeouts(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _answer(monkeypatch, httpx.Response(200, json={"version": "1.2.3"}))

        _remotes()

        (request,) = seen
        assert request.method == "GET"
        assert str(request.url) == REMOTE_URL + "/health"
        assert request.extensions["timeout"] == {
            "connect": 5,
            "read": 5,
            "write": 5,
            "pool": 5,
        }


class TestAnUnreachableRemote:
    def test_a_connection_error_reads_unreachable_with_the_error(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, httpx.ConnectError("connection refused"))

        result = _remotes("--output-format", "text")

        assert result.exit_code == 0, result.output
        assert result.stdout == (
            f"remote-host  {REMOTE_URL}  unreachable: connection refused\n"
        )

    def test_a_status_code_is_the_error(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, httpx.Response(502, text="bad gateway"))

        result = _remotes("--output-format", "json")

        (row,) = json.loads(result.stdout)
        assert row["reachable"] is False
        assert "502" in row["error"]
        assert "version" not in row

    @pytest.mark.parametrize(
        "answer",
        [
            httpx.Response(200, text="<html>the web ui</html>"),
            httpx.Response(200, json={"status": "healthy"}),
            httpx.Response(200, json=["not", "an", "object"]),
        ],
        ids=["html", "no-version", "array"],
    )
    def test_a_200_that_is_not_a_health_answer_is_not_reachable(
        self,
        remotes: None,
        in_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        answer: httpx.Response,
    ) -> None:
        _answer(monkeypatch, answer)

        result = _remotes("--output-format", "json")

        (row,) = json.loads(result.stdout)
        assert row["reachable"] is False
        assert "health" in row["error"]

    def test_a_redirect_is_not_followed(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _answer(
            monkeypatch,
            httpx.Response(302, headers={"location": "https://elsewhere.example/"}),
        )

        result = _remotes("--output-format", "json")

        (row,) = json.loads(result.stdout)
        assert row["reachable"] is False
        assert "302" in row["error"]
        assert len(seen) == 1

    def test_the_error_from_a_remote_is_stripped_of_control_characters(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, httpx.ConnectError("bad\x1b[31m\nnews"))

        result = _remotes("--output-format", "json")

        (row,) = json.loads(result.stdout)
        assert "\x1b" not in row["error"]
        assert "\n" not in row["error"]


class TestSeveralRemotes:
    def test_each_row_is_probed_on_its_own_and_a_bad_url_is_one_row(
        self, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _global_config(
            {
                "remotes": {
                    "good": {"url": "https://good.example"},
                    "down": {"url": "https://down.example"},
                    "bad": {"url": "ftp://bad.example"},
                }
            }
        )

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.host == "down.example":
                raise httpx.ConnectError("refused")
            return httpx.Response(200, json={"version": "1.0.0"})

        monkeypatch.setattr(remote_run, "transport", httpx.MockTransport(handle))

        result = _remotes("--output-format", "json")

        assert result.exit_code == 0, result.output
        rows = {row["name"]: row for row in json.loads(result.stdout)}
        assert rows["good"]["reachable"] is True
        assert rows["down"]["error"] == "refused"
        assert rows["bad"]["reachable"] is False
        assert "http or https" in rows["bad"]["error"]
        assert rows["bad"]["url"] == "ftp://bad.example"


class TestTheConfig:
    def test_a_project_config_remotes_key_is_not_read(
        self, remotes: None, in_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (in_project / ".llm-orc" / "config.yaml").write_text(
            yaml.safe_dump({"remotes": {"from-repo": {"url": "https://repo.example"}}})
        )
        _answer(monkeypatch, httpx.Response(200, json={"version": "1.0.0"}))

        result = _remotes("--output-format", "json")

        assert [row["name"] for row in json.loads(result.stdout)] == ["remote-host"]

    def test_none_configured_says_so_and_points_at_the_config(
        self, in_project: Path
    ) -> None:
        result = _remotes()

        assert result.exit_code == 0, result.output
        assert "No remotes are configured" in result.stdout
        assert "remotes:" in result.stdout
        assert "config.yaml" in result.stdout

    def test_none_configured_is_an_empty_array_in_json_mode(
        self, in_project: Path
    ) -> None:
        result = _remotes("--output-format", "json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == []

    def test_a_malformed_remotes_key_exits_1_with_the_message(
        self, in_project: Path
    ) -> None:
        _global_config({"remotes": ["not", "a", "mapping"]})

        result = _remotes()

        assert result.exit_code == 1
        assert "'remotes' in the global config must be a mapping" in result.output
        assert "Traceback" not in result.output
