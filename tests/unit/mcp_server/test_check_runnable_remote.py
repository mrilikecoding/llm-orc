"""``check_ensemble_runnable`` with ``remote`` (Arc 6, part 2): the real
tool ships the closure and asks the second real service's preflight
endpoint. On a mount it does not relay. Nothing here opens a connection."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from tests.unit.mcp_server.test_invoke_tool import (  # noqa: F401
    ToolCall,
    stdio_tool,
    tool,
)
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    _profile,
    listing,
    project,
    service,
    state_dir,
)


def _statuses(result: dict[str, object]) -> dict[str, str]:
    rows = result["dependencies"]
    assert isinstance(rows, list)
    return {row["name"]: row["status"] for row in rows}


class TestWithARemote:
    def test_a_profile_the_remote_lacks_returns_the_rows_and_nothing_ran(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project, with_profile=True)
        before = remote.tree()

        result = stdio_tool(
            "check_ensemble_runnable",
            {"ensemble_name": "top", "remote": "remote-host"},
        )

        assert result["runnable"] is False
        assert _statuses(result)["seat"] == "missing_profile"
        assert _statuses(result)["top"] == "ready"
        assert [url for url, _ in remote.calls] == [
            REMOTE_URL + "/api/ensembles/preflight"
        ]
        assert remote.tree() == before
        assert not list(remote.project.rglob("ran.marker"))

    def test_bind_and_pull_are_shipped_and_the_binding_is_reported(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project, with_profile=True)
        _profile(remote.project / ".llm-orc", "other", model="mock-other")
        before = remote.tree()

        result = stdio_tool(
            "check_ensemble_runnable",
            {
                "ensemble_name": "top",
                "remote": "remote-host",
                "bind": {"seat": "other"},
                "pull": True,
            },
        )

        assert result["runnable"] is True
        assert result["bindings"] == {"seat": "other"}
        assert result["pull_requested"] is True
        sent = remote.calls[0][1]
        assert sent["bind"] == {"seat": "other"}
        assert sent["pull"] is True
        assert "persist" not in sent
        assert remote.tree() == before

    def test_with_profiles_ships_the_definition(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project, with_profile=True)

        result = stdio_tool(
            "check_ensemble_runnable",
            {
                "ensemble_name": "top",
                "remote": "remote-host",
                "with_profiles": ["seat"],
            },
        )

        assert result["runnable"] is True
        assert "seat" in remote.calls[0][1]["profiles"]

    def test_a_child_left_to_the_remote_is_named_in_left_out(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top_with_remote_only_child(project, remote)

        result = stdio_tool(
            "check_ensemble_runnable",
            {"ensemble_name": "top", "remote": "remote-host"},
        )

        assert result["runnable"] is True
        assert result["left_out"] == ["ensemble 'kids/only'"]

    def test_an_invalid_bind_key_is_the_remotes_refusal_envelope(
        self, project: Path, remote: Remote, stdio_tool: ToolCall
    ) -> None:
        _write_top(project)

        result = stdio_tool(
            "check_ensemble_runnable",
            {
                "ensemble_name": "top",
                "remote": "remote-host",
                "bind": {"unnamed": "other"},
            },
        )

        assert result["status"] == "error"
        assert result["error"]["kind"] == "invalid_request"
        assert "unnamed" in result["error"]["message"]

    @pytest.mark.parametrize(
        "answer",
        [
            Canned(200, "<html>the web ui</html>"),
            Canned(200, json.dumps({"status": "success", "has_errors": False})),
            Canned(404, "{}"),
        ],
        ids=["html", "run-result", "404"],
    )
    def test_an_answer_that_is_not_a_preflight_is_a_remote_error(
        self,
        project: Path,
        remotes: None,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        answer: Canned,
    ) -> None:
        _write_top(project)
        _canned(monkeypatch, answer)

        result = stdio_tool(
            "check_ensemble_runnable",
            {"ensemble_name": "top", "remote": "remote-host"},
        )

        assert result["error"]["kind"] == "remote_error"
        assert "remote-host" in result["error"]["message"]

    def test_an_unknown_remote_is_invalid_and_sends_nothing(
        self,
        project: Path,
        remotes: None,
        stdio_tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_top(project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = stdio_tool(
            "check_ensemble_runnable",
            {"ensemble_name": "top", "remote": "nowhere"},
        )

        assert result["error"]["kind"] == "invalid_request"
        assert "known remotes: remote-host" in result["error"]["message"]
        assert calls == []


class TestWithoutARemote:
    def test_it_judges_the_local_ensemble_as_before(
        self, project: Path, stdio_tool: ToolCall
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "x"}])

        result = stdio_tool("check_ensemble_runnable", {"ensemble_name": "top"})

        assert result["runnable"] is False
        assert result["ensemble"] == "top"
        assert "agents" in result

    @pytest.mark.parametrize(
        "extra",
        [{"bind": {"a": "b"}}, {"pull": True}, {"with_profiles": ["seat"]}],
        ids=["bind", "pull", "with_profiles"],
    )
    def test_a_shipping_argument_without_remote_is_invalid_not_dropped(
        self, project: Path, stdio_tool: ToolCall, extra: dict[str, object]
    ) -> None:
        _write_top(project)

        result = stdio_tool(
            "check_ensemble_runnable", {"ensemble_name": "top", **extra}
        )

        assert result["error"]["kind"] == "invalid_request"
        assert "need remote" in result["error"]["message"]


class TestOnAMountedServer:
    @pytest.mark.parametrize(
        "extra",
        [{"remote": "remote-host"}, {"with_profiles": ["seat"]}],
        ids=["remote", "with_profiles"],
    )
    def test_it_does_not_relay_and_sends_nothing(
        self,
        project: Path,
        remotes: None,
        tool: ToolCall,
        monkeypatch: pytest.MonkeyPatch,
        extra: dict[str, object],
    ) -> None:
        _write_top(project)
        calls = _canned(monkeypatch, Canned(200, "{}"))

        result = tool("check_ensemble_runnable", {"ensemble_name": "top", **extra})

        assert result["error"]["kind"] == "invalid_request"
        assert "does not relay" in result["error"]["message"]
        assert calls == []

    def test_the_plain_check_still_works(self, project: Path, tool: ToolCall) -> None:
        _write_top(project)

        result = tool("check_ensemble_runnable", {"ensemble_name": "top"})

        assert result["ensemble"] == "top"
