"""Unit tests for ProfileHandler — covers previously uncovered lines."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from llm_orc.services.handlers.profile_handler import ProfileHandler


@pytest.fixture
def mock_config() -> Any:
    config = MagicMock()
    config.get_profiles_dirs.return_value = []
    config.get_model_profiles.return_value = {}
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
            yaml.safe_dump({"name": "a", "provider": "ollama", "model": "llama3"})
        )
        (profiles_dir / "b.yaml").write_text(
            yaml.safe_dump({"name": "b", "provider": "anthropic", "model": "claude"})
        )
        mock_config.get_profiles_dirs.return_value = [str(profiles_dir)]
        handler = _handler(mock_config)

        result = await handler.list_profiles({"provider": "ollama"})

        names = [p["name"] for p in result["profiles"]]
        assert "a" in names
        assert "b" not in names


# ---------------------------------------------------------------------------
# get_local_profiles_dir
# ---------------------------------------------------------------------------


class TestGetLocalProfilesDir:
    """Covers lines 71-73 in get_local_profiles_dir."""

    def test_returns_llm_orc_non_library_path(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """Returns the first .llm-orc non-library path."""
        local_path = tmp_path / ".llm-orc" / "profiles"
        mock_config.get_profiles_dirs.return_value = [str(local_path)]
        handler = _handler(mock_config)

        result = handler.get_local_profiles_dir()

        assert result == local_path

    def test_falls_back_to_first_dir_when_no_llm_orc_path(
        self, mock_config: Any, tmp_path: Path
    ) -> None:
        """Falls back to first entry when no .llm-orc path exists (line 72)."""
        fallback = tmp_path / "other" / "profiles"
        mock_config.get_profiles_dirs.return_value = [str(fallback)]
        handler = _handler(mock_config)

        result = handler.get_local_profiles_dir()

        assert result == fallback

    def test_raises_when_no_dirs_configured(self, mock_config: Any) -> None:
        """Raises ValueError when no profiles directories at all (line 73)."""
        mock_config.get_profiles_dirs.return_value = []
        handler = _handler(mock_config)

        with pytest.raises(ValueError, match="No profiles directory configured"):
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
        handler = _handler(mock_config)

        result = await handler.create_profile(
            {
                "name": "full-profile",
                "provider": "ollama",
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
        handler = _handler(mock_config)

        result = await handler.create_profile(
            {
                "name": "zero-temp",
                "provider": "ollama",
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
        handler = _handler(mock_config)

        result = await handler.create_profile(
            {
                "name": "zero-tokens",
                "provider": "ollama",
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
                "p1": {"provider": "ollama", "model": "llama3"},
            }
        }
        handler = _handler(mock_config)
        profiles: dict[str, dict[str, Any]] = {}

        handler._parse_profile_data(data, profiles)

        assert "p1" in profiles
        assert profiles["p1"]["name"] == "p1"
        assert profiles["p1"]["provider"] == "ollama"

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
            "provider": "ollama",
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
                "ok": {"provider": "ollama", "model": "llama3"},
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
                {"provider": "ollama", "model": "llama3"},
                {"name": "named", "provider": "ollama", "model": "llama3"},
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
