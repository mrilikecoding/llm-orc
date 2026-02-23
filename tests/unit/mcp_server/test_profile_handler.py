"""Unit tests for ProfileHandler.get_all_profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from llm_orc.mcp.handlers.profile_handler import ProfileHandler


@pytest.fixture
def mock_config_manager() -> Any:
    config = MagicMock()
    config.get_profiles_dirs.return_value = []
    config.get_model_profiles.return_value = {}
    return config


class TestGetAllProfiles:
    def test_includes_profiles_from_config_yaml(self, mock_config_manager: Any) -> None:
        """Profiles defined in config.yaml model_profiles are visible."""
        mock_config_manager.get_model_profiles.return_value = {
            "validate-ollama": {"model": "llama3", "provider": "ollama"},
        }
        handler = ProfileHandler(mock_config_manager)

        result = handler.get_all_profiles()

        assert "validate-ollama" in result
        assert result["validate-ollama"]["model"] == "llama3"

    def test_directory_profiles_override_config_yaml(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """Profiles in profiles/ directories take precedence over config.yaml."""
        mock_config_manager.get_model_profiles.return_value = {
            "my-profile": {"model": "old-model", "provider": "ollama"},
        }
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "my-profile.yaml").write_text(
            yaml.safe_dump(
                {"name": "my-profile", "model": "new-model", "provider": "ollama"}
            )
        )
        mock_config_manager.get_profiles_dirs.return_value = [str(profiles_dir)]
        handler = ProfileHandler(mock_config_manager)

        result = handler.get_all_profiles()

        assert result["my-profile"]["model"] == "new-model"

    def test_merges_config_yaml_and_directory_profiles(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """Both config.yaml and directory profiles appear in the result."""
        mock_config_manager.get_model_profiles.return_value = {
            "config-profile": {"model": "llama3", "provider": "ollama"},
        }
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "dir-profile.yaml").write_text(
            yaml.safe_dump(
                {"name": "dir-profile", "model": "qwen3:0.6b", "provider": "ollama"}
            )
        )
        mock_config_manager.get_profiles_dirs.return_value = [str(profiles_dir)]
        handler = ProfileHandler(mock_config_manager)

        result = handler.get_all_profiles()

        assert "config-profile" in result
        assert "dir-profile" in result

    def test_empty_when_no_profiles_anywhere(self, mock_config_manager: Any) -> None:
        """Returns empty dict when neither config.yaml nor directories have profiles."""
        handler = ProfileHandler(mock_config_manager)

        result = handler.get_all_profiles()

        assert result == {}
