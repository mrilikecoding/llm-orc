"""Unit tests for ProfileHandler — covers previously uncovered lines."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)
from llm_orc.services.handlers.profile_handler import ProfileHandler


@pytest.fixture
def mock_config() -> Any:
    config = MagicMock()
    config.get_profiles_dirs.return_value = []
    config.get_model_profiles.return_value = {}
    config.local_config_dir = None
    return config


def _handler(config: Any) -> ProfileHandler:
    return ProfileHandler(config)


# ---------------------------------------------------------------------------
# list_profiles
# ---------------------------------------------------------------------------


class TestListProfiles:
    """Covers lines 36, 55-56 in list_profiles."""

    async def test_skips_nonexistent_directory(self, mock_config: Any) -> None:
        """Nonexistent profiles dir is silently skipped (line 36)."""
        mock_config.get_profiles_dirs.return_value = ["/nonexistent/path/profiles"]
        handler = _handler(mock_config)

        result = await handler.list_profiles({})

        assert result == {"profiles": []}

    async def test_skips_unreadable_yaml_file(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """Corrupt YAML file is silently skipped (lines 55-56)."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        bad_file = profiles_dir / "bad.yaml"
        bad_file.write_text("{corrupt: [yaml: content")
        mock_config.get_profiles_dirs.return_value = [str(profiles_dir)]
        handler = _handler(mock_config)

        result = await handler.list_profiles({})

        assert result == {"profiles": []}

    async def test_provider_filter_excludes_non_matching(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """Profiles not matching provider filter are excluded."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "a.yaml").write_text(
            yaml.safe_dump({"name": "a", "provider": "llama-server", "model": "llama3"})
        )
        (profiles_dir / "b.yaml").write_text(
            yaml.safe_dump({"name": "b", "provider": "anthropic", "model": "claude"})
        )
        mock_config.get_profiles_dirs.return_value = [str(profiles_dir)]
        handler = _handler(mock_config)

        result = await handler.list_profiles({"provider": "llama-server"})

        names = [p["name"] for p in result["profiles"]]
        assert "a" in names
        assert "b" not in names


# ---------------------------------------------------------------------------
# get_local_profiles_dir
# ---------------------------------------------------------------------------


class TestGetLocalProfilesDir:
    """get_local_profiles_dir is the project dir or a raise naming global."""

    def test_returns_the_project_profiles_dir(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        mock_config.local_config_dir = tmp_path / ".llm-orc"
        handler = _handler(mock_config)

        assert handler.get_local_profiles_dir() == tmp_path / ".llm-orc" / "profiles"

    def test_raises_naming_global_when_no_project(self, mock_config: Any) -> None:
        mock_config.local_config_dir = None
        mock_config.get_profiles_dirs.return_value = ["/somewhere/profiles"]
        handler = _handler(mock_config)

        with pytest.raises(ValueError, match="scope 'global'"):
            handler.get_local_profiles_dir()


# ---------------------------------------------------------------------------
# create_profile
# ---------------------------------------------------------------------------


class TestCreateProfile:
    """Covers lines 99, 101, 103, 105 — optional fields in create_profile."""

    async def test_creates_profile_with_all_optional_fields(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """All optional fields are written to YAML (lines 99, 101, 103, 105)."""
        local_dir = tmp_path / ".llm-orc" / "profiles"
        mock_config.get_profiles_dirs.return_value = [str(local_dir)]
        mock_config.local_config_dir = tmp_path / ".llm-orc"
        handler = _handler(mock_config)

        result = await handler.create_profile(
            {
                "name": "full-profile",
                "provider": "llama-server",
                "model": "llama3",
                "system_prompt": "You are helpful.",
                "timeout_seconds": 30,
                "temperature": 0.7,
                "max_tokens": 512,
            }
        )

        assert result["created"] is True
        written = yaml.safe_load(Path(result["path"]).read_text())
        assert written["system_prompt"] == "You are helpful."
        assert written["timeout_seconds"] == 30
        assert written["temperature"] == pytest.approx(0.7)
        assert written["max_tokens"] == 512

    async def test_temperature_zero_is_written(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """temperature=0.0 is written (not skipped) because `is not None` is used."""
        local_dir = tmp_path / ".llm-orc" / "profiles"
        mock_config.get_profiles_dirs.return_value = [str(local_dir)]
        mock_config.local_config_dir = tmp_path / ".llm-orc"
        handler = _handler(mock_config)

        result = await handler.create_profile(
            {
                "name": "zero-temp",
                "provider": "llama-server",
                "model": "llama3",
                "temperature": 0.0,
            }
        )

        written = yaml.safe_load(Path(result["path"]).read_text())
        assert written["temperature"] == pytest.approx(0.0)

    async def test_max_tokens_zero_is_written(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """max_tokens=0 is written (not skipped) because `is not None` is used."""
        local_dir = tmp_path / ".llm-orc" / "profiles"
        mock_config.get_profiles_dirs.return_value = [str(local_dir)]
        mock_config.local_config_dir = tmp_path / ".llm-orc"
        handler = _handler(mock_config)

        result = await handler.create_profile(
            {
                "name": "zero-tokens",
                "provider": "llama-server",
                "model": "llama3",
                "max_tokens": 0,
            }
        )

        written = yaml.safe_load(Path(result["path"]).read_text())
        assert written["max_tokens"] == 0


# ---------------------------------------------------------------------------
# get_all_profiles — nonexistent directory branch
# ---------------------------------------------------------------------------


class TestGetAllProfilesNonexistentDir:
    """Covers line 192 — skipping nonexistent directory in get_all_profiles."""

    def test_skips_nonexistent_profiles_dir(self, mock_config: Any) -> None:
        """get_all_profiles skips dirs that do not exist on disk (line 192)."""
        mock_config.get_profiles_dirs.return_value = ["/does/not/exist"]
        handler = _handler(mock_config)

        result = handler.get_all_profiles()

        assert result == {}


# ---------------------------------------------------------------------------
# _load_profiles_from_file
# ---------------------------------------------------------------------------


class TestLoadProfilesFromFile:
    """Covers lines 210-211 — exception swallowed in _load_profiles_from_file."""

    def test_corrupt_yaml_file_is_ignored(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """A corrupt YAML file does not raise; it is silently skipped."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{not: valid: yaml: [}")
        handler = _handler(mock_config)
        profiles: dict[str, dict[str, Any]] = {}

        handler._load_profiles_from_file(bad_file, profiles)

        assert profiles == {}


# ---------------------------------------------------------------------------
# _parse_profile_data
# ---------------------------------------------------------------------------


class TestParseProfileData:
    """Covers lines 220, 222, 232-235, 243-246 in _parse_profile_data."""

    def test_model_profiles_dict_format(self, mock_config: Any) -> None:
        """model_profiles key triggers dict-format parsing (line 220)."""
        data: dict[str, Any] = {
            "model_profiles": {
                "p1": {"provider": "llama-server", "model": "llama3"},
            }
        }
        handler = _handler(mock_config)
        profiles: dict[str, dict[str, Any]] = {}

        handler._parse_profile_data(data, profiles)

        assert "p1" in profiles
        assert profiles["p1"]["name"] == "p1"
        assert profiles["p1"]["provider"] == "llama-server"

    def test_profiles_list_format(self, mock_config: Any) -> None:
        """profiles key triggers list-format parsing (line 222)."""
        data: dict[str, Any] = {
            "profiles": [
                {"name": "p2", "provider": "anthropic", "model": "claude"},
            ]
        }
        handler = _handler(mock_config)
        profiles: dict[str, dict[str, Any]] = {}

        handler._parse_profile_data(data, profiles)

        assert "p2" in profiles
        assert profiles["p2"]["provider"] == "anthropic"

    def test_single_profile_with_name_key(self, mock_config: Any) -> None:
        """Flat YAML with 'name' key is stored directly."""
        data: dict[str, Any] = {
            "name": "p3",
            "provider": "llama-server",
            "model": "mistral",
        }
        handler = _handler(mock_config)
        profiles: dict[str, dict[str, Any]] = {}

        handler._parse_profile_data(data, profiles)

        assert "p3" in profiles

    def test_dict_format_non_dict_values_are_skipped(self, mock_config: Any) -> None:
        """Non-dict values inside model_profiles are skipped (line 233 branch)."""
        data: dict[str, Any] = {
            "model_profiles": {
                "ok": {"provider": "llama-server", "model": "llama3"},
                "bad": "just-a-string",
            }
        }
        handler = _handler(mock_config)
        profiles: dict[str, dict[str, Any]] = {}

        handler._parse_profile_data(data, profiles)

        assert "ok" in profiles
        assert "bad" not in profiles

    def test_list_format_entries_without_name_are_skipped(
        self, mock_config: Any
    ) -> None:
        """List entries missing 'name' field are silently skipped (line 244 branch)."""
        data: dict[str, Any] = {
            "profiles": [
                {"provider": "llama-server", "model": "llama3"},
                {"name": "named", "provider": "llama-server", "model": "llama3"},
            ]
        }
        handler = _handler(mock_config)
        profiles: dict[str, dict[str, Any]] = {}

        handler._parse_profile_data(data, profiles)

        assert "named" in profiles
        assert len(profiles) == 1


# ---------------------------------------------------------------------------
# set_project_context
# ---------------------------------------------------------------------------


class TestSetProjectContext:
    """set_project_context replaces config_manager."""

    def test_set_project_context_updates_config_manager(self, mock_config: Any) -> None:
        handler = _handler(mock_config)
        new_config = MagicMock()
        ctx = MagicMock()
        ctx.config_manager = new_config

        handler.set_project_context(ctx)

        assert handler._config_manager is new_config


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------


def _real_profile_handler(project: Path) -> ProfileHandler:
    return ProfileHandler(ConfigurationManager(project_dir=project, provision=False))


class TestProfileScope:
    async def test_global_scope_creates_under_global_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)
        handler = _real_profile_handler(tmp_path)
        global_profiles = resolve_global_config_dir() / "profiles"
        assert not global_profiles.exists()

        result = await handler.create_profile(
            {
                "name": "remote-prof",
                "provider": "llama-server",
                "model": "qwen3-8b",
                "scope": "global",
            }
        )

        assert result["scope"] == "global"
        assert Path(result["path"]) == global_profiles / "remote-prof.yaml"
        assert not (tmp_path / ".llm-orc" / "profiles" / "remote-prof.yaml").exists()

    async def test_default_scope_writes_project(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)
        handler = _real_profile_handler(tmp_path)

        result = await handler.create_profile(
            {"name": "mine", "provider": "llama-server", "model": "qwen3-8b"}
        )

        assert result["scope"] == "project"
        assert Path(result["path"]) == tmp_path / ".llm-orc" / "profiles" / "mine.yaml"

    async def test_delete_with_wrong_scope_touches_nothing(
        self, tmp_path: Path
    ) -> None:
        project_file = tmp_path / ".llm-orc" / "profiles" / "keep.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("name: keep\nprovider: llama-server\nmodel: m\n")
        handler = _real_profile_handler(tmp_path)

        with pytest.raises(ValueError, match=r"not in scope 'global'.*local tier"):
            await handler.delete_profile(
                {"name": "keep", "confirm": True, "scope": "global"}
            )

        assert project_file.exists()

    async def test_update_global_scope_edits_global_file_only(
        self, tmp_path: Path
    ) -> None:
        project_file = tmp_path / ".llm-orc" / "profiles" / "twin.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("name: twin\nprovider: llama-server\nmodel: project\n")
        global_file = resolve_global_config_dir() / "profiles" / "twin.yaml"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("name: twin\nprovider: llama-server\nmodel: global\n")
        handler = _real_profile_handler(tmp_path)

        await handler.update_profile(
            {"name": "twin", "changes": {"model": "edited"}, "scope": "global"}
        )

        assert yaml.safe_load(global_file.read_text())["model"] == "edited"
        assert yaml.safe_load(project_file.read_text())["model"] == "project"

    async def test_omitted_scope_never_deletes_a_global_only_profile(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)
        global_file = resolve_global_config_dir() / "profiles" / "only-global.yaml"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("name: only-global\nprovider: llama-server\nmodel: m\n")
        handler = _real_profile_handler(tmp_path)

        with pytest.raises(ValueError, match=r"not in scope 'project'.*global tier"):
            await handler.delete_profile({"name": "only-global", "confirm": True})

        assert global_file.exists()

    async def test_project_scope_without_project_dir_errors_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        # A global profiles dir with unrelated content already exists, so
        # a fallback to "first search dir when nothing matches .llm-orc"
        # would otherwise find it and silently write the new profile there.
        global_profiles = resolve_global_config_dir() / "profiles"
        global_profiles.mkdir(parents=True)
        (global_profiles / "unrelated.yaml").write_text(
            "name: unrelated\nprovider: llama-server\nmodel: m\n"
        )
        handler = _real_profile_handler(tmp_path)  # no .llm-orc created

        with pytest.raises(ValueError, match=r"No project directory.*scope: global"):
            await handler.create_profile(
                {"name": "orphan", "provider": "llama-server", "model": "qwen3-8b"}
            )

        assert not (global_profiles / "orphan.yaml").exists()

    async def test_omitted_scope_without_project_dir_never_deletes_global(
        self, tmp_path: Path
    ) -> None:
        global_file = resolve_global_config_dir() / "profiles" / "only-global.yaml"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("name: only-global\nprovider: llama-server\nmodel: m\n")
        handler = _real_profile_handler(tmp_path)  # no .llm-orc created

        with pytest.raises(ValueError, match=r"not in scope 'project'.*global tier"):
            await handler.delete_profile({"name": "only-global", "confirm": True})

        assert global_file.exists()
