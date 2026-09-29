"""Where the serve writes: env, then the project, then XDG state (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config import state


class TestResolveStateDir:
    def test_env_override_wins_over_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(state.STATE_DIR_ENV, str(tmp_path / "explicit"))
        assert state.resolve_state_dir(tmp_path / ".llm-orc") == tmp_path / "explicit"

    def test_project_dot_dir_when_there_is_a_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(state.STATE_DIR_ENV, raising=False)
        assert state.resolve_state_dir(tmp_path / ".llm-orc") == tmp_path / ".llm-orc"

    def test_xdg_state_home_without_a_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(state.STATE_DIR_ENV, raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
        assert state.resolve_state_dir(None) == tmp_path / "xdg" / "llm-orc"

    def test_home_local_state_when_nothing_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(state.STATE_DIR_ENV, raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        expected = tmp_path / ".local" / "state" / "llm-orc"
        assert state.resolve_state_dir(None) == expected

    def test_empty_env_is_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(state.STATE_DIR_ENV, "")
        assert state.resolve_state_dir(tmp_path / ".llm-orc") == tmp_path / ".llm-orc"
