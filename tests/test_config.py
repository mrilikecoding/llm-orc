"""Test configuration management functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llm_orc.config import ConfigurationManager


class TestConfigurationManager:
    """Test the ConfigurationManager class."""

    def test_global_config_dir_default(self) -> None:
        """Test default global config directory path."""
        with patch.dict(os.environ, {}, clear=True):
            config_manager = ConfigurationManager()
            expected_path = Path.home() / ".config" / "llm-orc"
            assert config_manager.global_config_dir == expected_path

    def test_global_config_dir_xdg_config_home(self) -> None:
        """Test global config directory with XDG_CONFIG_HOME set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            xdg_config_home = temp_dir + "/config"
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": xdg_config_home}):
                config_manager = ConfigurationManager()
                expected_path = Path(xdg_config_home) / "llm-orc"
                assert config_manager.global_config_dir == expected_path

    def test_load_project_config(self) -> None:
        """Test loading project-specific configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create local config directory with config file
            local_dir = temp_path / ".llm-orc"
            local_dir.mkdir()

            config_data = {
                "project": {"name": "test-project"},
                "model_profiles": {"dev": {"model": "llama3"}},
            }

            config_file = local_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Mock cwd to find the local config
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                loaded_config = config_manager.load_project_config()

                assert loaded_config["project"]["name"] == "test-project"
                assert "dev" in loaded_config["model_profiles"]

    def test_load_project_config_no_local_config(self) -> None:
        """Test loading project config when no local config exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock cwd with no local config
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                loaded_config = config_manager.load_project_config()

                assert loaded_config == {}