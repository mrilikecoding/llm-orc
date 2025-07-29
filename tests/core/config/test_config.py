"""Test configuration management functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager


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

    def test_setup_default_ensembles_from_templates(self) -> None:
        """Test that default ensembles are created from template files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock global config directory
            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                ConfigurationManager()

                # Check that ensembles directory was created
                ensembles_dir = temp_path / "ensembles"
                assert ensembles_dir.exists()

                # Check that template files were copied
                expected_files = [
                    "validate-anthropic-api.yaml",
                    "validate-anthropic-claude-pro-max.yaml",
                    "validate-google-gemini.yaml",
                    "validate-ollama.yaml",
                ]

                for filename in expected_files:
                    ensemble_file = ensembles_dir / filename
                    assert ensemble_file.exists()

                    # Verify the file contains valid YAML
                    with open(ensemble_file) as f:
                        ensemble_config = yaml.safe_load(f)
                        assert "name" in ensemble_config
                        assert "description" in ensemble_config
                        assert "agents" in ensemble_config
                        # New dependency-based architecture doesn't use coordinator
                        assert len(ensemble_config["agents"]) > 0

    def test_setup_default_ensembles_no_templates(self) -> None:
        """Test that setup gracefully handles missing template directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock global config directory and template directory to not exist
            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                with patch.object(
                    ConfigurationManager,
                    "_get_template_ensembles_dir",
                    return_value=temp_path / "missing",
                ):
                    ConfigurationManager()

                    # Should create ensembles directory but not fail
                    ensembles_dir = temp_path / "ensembles"
                    assert ensembles_dir.exists()

                    # Should be empty since no templates exist
                    assert list(ensembles_dir.glob("*.yaml")) == []

    def test_setup_default_config_from_template(self) -> None:
        """Test that global config is created from template file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock global config directory
            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                ConfigurationManager()

                # Check that config.yaml was created
                config_file = temp_path / "config.yaml"
                assert config_file.exists()

                # Verify the file contains valid YAML with model profiles
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                    assert "model_profiles" in config_data
                    assert "micro-local" in config_data["model_profiles"]
                    assert "default" in config_data["model_profiles"]
                    assert "validate-anthropic-api" in config_data["model_profiles"]

    def test_init_local_config_from_templates(self) -> None:
        """Test that local config initialization uses templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock cwd to temp directory
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                config_manager.init_local_config("test-project")

                # Check directory structure was created
                local_dir = temp_path / ".llm-orc"
                assert local_dir.exists()
                assert (local_dir / "ensembles").exists()
                assert (local_dir / "models").exists()
                assert (local_dir / "scripts").exists()

                # Check config file was created with project name
                config_file = local_dir / "config.yaml"
                assert config_file.exists()

                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                    assert config_data["project"]["name"] == "test-project"
                    assert "test" in config_data["project"]["default_models"]
                    assert (
                        config_data["project"]["default_models"]["test"]
                        == "micro-local"
                    )
                    # Only test fallback should exist now
                    assert len(config_data["project"]["default_models"]) == 1

                # Check example ensemble was copied
                example_ensemble = (
                    local_dir / "ensembles" / "example-local-ensemble.yaml"
                )
                assert example_ensemble.exists()

                with open(example_ensemble) as f:
                    ensemble_data = yaml.safe_load(f)
                    assert ensemble_data["name"] == "example-local-ensemble"
                    # New dependency-based architecture doesn't use coordinator
                    assert "agents" in ensemble_data
                    assert len(ensemble_data["agents"]) > 0

                # Check gitignore was created
                gitignore_file = local_dir / ".gitignore"
                assert gitignore_file.exists()

    def test_setup_default_config_template_not_found(self) -> None:
        """Test config setup falls back gracefully when template not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock global config directory and template methods to fail
            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                with patch.object(
                    ConfigurationManager,
                    "_get_template_config_content",
                    side_effect=FileNotFoundError(),
                ):
                    ConfigurationManager()

                    # Check that config.yaml was created with fallback content
                    config_file = temp_path / "config.yaml"
                    assert config_file.exists()

                    # Verify the file contains fallback YAML
                    with open(config_file) as f:
                        config_data = yaml.safe_load(f)
                        assert "model_profiles" in config_data
                        assert config_data["model_profiles"] == {}

    def test_load_project_config_yaml_error(self) -> None:
        """Test loading project config with corrupted YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create local config directory with corrupted config file
            local_dir = temp_path / ".llm-orc"
            local_dir.mkdir()

            config_file = local_dir / "config.yaml"
            with open(config_file, "w") as f:
                f.write("invalid: yaml: content: [")  # Corrupted YAML

            # Mock cwd to find the local config
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                loaded_config = config_manager.load_project_config()

                # Should return empty dict on YAML error
                assert loaded_config == {}

    def test_load_global_config_yaml_error(self) -> None:
        """Test loading global config with corrupted YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config file with corrupted YAML
            config_file = temp_path / "config.yaml"
            with open(config_file, "w") as f:
                f.write("invalid: yaml: content: [")  # Corrupted YAML

            # Mock global config directory
            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                config_manager = ConfigurationManager()
                global_config = config_manager._load_global_config()

                # Should return empty dict on YAML error
                assert global_config == {}

    def test_discover_local_config_with_existing_dir(self) -> None:
        """Test local config discovery finds existing .llm-orc directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create .llm-orc directory
            llm_orc_dir = temp_path / ".llm-orc"
            llm_orc_dir.mkdir()

            # Mock cwd to point to temp directory
            with patch("pathlib.Path.cwd", return_value=temp_path):
                with patch.object(ConfigurationManager, "_setup_default_config"):
                    with patch.object(ConfigurationManager, "_setup_default_ensembles"):
                        config_manager = ConfigurationManager()
                        # Should find the .llm-orc directory
                        assert config_manager.local_config_dir == llm_orc_dir

    def test_config_manager_initialization_with_existing_config(self) -> None:
        """Test config manager doesn't overwrite existing global config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create existing config file
            config_file = temp_path / "config.yaml"
            existing_content = "existing: configuration"
            with open(config_file, "w") as f:
                f.write(existing_content)

            # Mock global config directory
            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                ConfigurationManager()

                # Should not overwrite existing config
                with open(config_file) as f:
                    content = f.read()
                    assert content == existing_content

    def test_get_template_config_content_not_found(self) -> None:
        """Test template config content raises error when file not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                with patch(
                    "llm_orc.cli_library.library.get_template_content",
                    side_effect=FileNotFoundError(),
                ):
                    config_manager = ConfigurationManager()

                    with pytest.raises(
                        FileNotFoundError, match="Template not found: nonexistent.yaml"
                    ):
                        config_manager._get_template_config_content("nonexistent.yaml")

    def test_get_model_profile_existing(self) -> None:
        """Test getting an existing model profile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config with model profiles
            config_data = {
                "model_profiles": {
                    "test-profile": {"model": "test-model", "provider": "test-provider"}
                }
            }

            config_file = temp_path / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                config_manager = ConfigurationManager()
                profile = config_manager.get_model_profile("test-profile")
                assert profile is not None
                assert profile["model"] == "test-model"
                assert profile["provider"] == "test-provider"

    def test_get_model_profile_not_found(self) -> None:
        """Test getting a non-existent model profile returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                config_manager = ConfigurationManager()
                profile = config_manager.get_model_profile("nonexistent-profile")
                assert profile is None

    def test_resolve_model_profile_missing_model(self) -> None:
        """Test resolving a model profile with missing model field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config with incomplete model profile
            config_data = {
                "model_profiles": {
                    "incomplete-profile": {
                        "provider": "test-provider"
                        # Missing "model" field
                    }
                }
            }

            config_file = temp_path / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                config_manager = ConfigurationManager()

                with pytest.raises(ValueError, match="is incomplete.*Both.*required"):
                    config_manager.resolve_model_profile("incomplete-profile")

    def test_resolve_model_profile_missing_provider(self) -> None:
        """Test resolving a model profile with missing provider field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config with incomplete model profile
            config_data = {
                "model_profiles": {
                    "incomplete-profile": {
                        "model": "test-model"
                        # Missing "provider" field
                    }
                }
            }

            config_file = temp_path / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            with patch.object(
                ConfigurationManager, "_get_global_config_dir", return_value=temp_path
            ):
                config_manager = ConfigurationManager()

                with pytest.raises(ValueError, match="is incomplete.*Both.*required"):
                    config_manager.resolve_model_profile("incomplete-profile")
