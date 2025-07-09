"""Tests for configuration management system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llm_orc.config import ConfigurationManager


class TestConfigurationManager:
    """Test the configuration management system."""

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

    def test_global_config_dir_creation(self) -> None:
        """Test that global config directory is created when no migration is needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            xdg_config_home = temp_dir + "/config"

            # Mock home to avoid finding existing .llm-orc directory
            with patch("pathlib.Path.home", return_value=temp_path / "fake_home"):
                with patch.dict(os.environ, {"XDG_CONFIG_HOME": xdg_config_home}):
                    config_manager = ConfigurationManager()
                    assert config_manager.global_config_dir.exists()

    def test_local_config_dir_discovery(self) -> None:
        """Test local config directory discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested directory structure
            project_dir = temp_path / "project"
            project_dir.mkdir()

            sub_dir = project_dir / "src" / "deep"
            sub_dir.mkdir(parents=True)

            # Create .llm-orc directory in project root
            llm_orc_dir = project_dir / ".llm-orc"
            llm_orc_dir.mkdir()

            # Mock cwd to be in the deep subdirectory
            with patch("pathlib.Path.cwd", return_value=sub_dir):
                config_manager = ConfigurationManager()
                assert config_manager.local_config_dir == llm_orc_dir

    def test_local_config_dir_not_found(self) -> None:
        """Test when no local config directory is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested directory structure without .llm-orc
            project_dir = temp_path / "isolated" / "project"
            project_dir.mkdir(parents=True)

            sub_dir = project_dir / "src" / "deep"
            sub_dir.mkdir(parents=True)

            # Mock the _discover_local_config method to use our controlled structure
            def mock_discover_local_config(self: ConfigurationManager) -> Path | None:
                """Mock discover method that only searches in our test structure."""
                current = sub_dir
                while current != temp_path.parent:  # Stop at our test boundary
                    llm_orc_dir = current / ".llm-orc"
                    if llm_orc_dir.exists() and llm_orc_dir.is_dir():
                        return llm_orc_dir
                    current = current.parent
                    if current == temp_path.parent:
                        break
                return None

            # Mock both cwd and the discover method
            with patch("pathlib.Path.cwd", return_value=sub_dir):
                with patch("pathlib.Path.home", return_value=temp_path / "fake_home"):
                    with patch.dict(
                        os.environ, {"XDG_CONFIG_HOME": str(temp_path / "config")}
                    ):
                        with patch.object(
                            ConfigurationManager,
                            "_discover_local_config",
                            mock_discover_local_config,
                        ):
                            config_manager = ConfigurationManager()
                            assert config_manager.local_config_dir is None

    def test_get_ensembles_dirs_local_and_global(self) -> None:
        """Test getting ensembles directories with both local and global present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Setup local config
            local_dir = temp_path / ".llm-orc"
            local_dir.mkdir()
            local_ensembles = local_dir / "ensembles"
            local_ensembles.mkdir()

            # Setup global config
            global_dir = temp_path / "global"
            global_dir.mkdir()
            global_ensembles = global_dir / "ensembles"
            global_ensembles.mkdir()

            # Mock both directories
            with patch("pathlib.Path.cwd", return_value=temp_path):
                with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(temp_path)}):
                    # Override the global config dir for this test
                    config_manager = ConfigurationManager()
                    config_manager._global_config_dir = global_dir

                    dirs = config_manager.get_ensembles_dirs()

                    # Local should come first
                    assert len(dirs) == 2
                    assert dirs[0] == local_ensembles
                    assert dirs[1] == global_ensembles

    def test_get_ensembles_dirs_global_only(self) -> None:
        """Test getting ensembles directories with only global present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Setup global config only
            global_dir = temp_path / "global"
            global_dir.mkdir()
            global_ensembles = global_dir / "ensembles"
            global_ensembles.mkdir()

            # Mock no local config
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                config_manager._global_config_dir = global_dir

                dirs = config_manager.get_ensembles_dirs()

                assert len(dirs) == 1
                assert dirs[0] == global_ensembles

    def test_get_credentials_file(self) -> None:
        """Test getting credentials file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": temp_dir}):
                config_manager = ConfigurationManager()
                expected_path = Path(temp_dir) / "llm-orc" / "credentials.yaml"
                assert config_manager.get_credentials_file() == expected_path

    def test_get_encryption_key_file(self) -> None:
        """Test getting encryption key file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": temp_dir}):
                config_manager = ConfigurationManager()
                expected_path = Path(temp_dir) / "llm-orc" / ".encryption_key"
                assert config_manager.get_encryption_key_file() == expected_path

    def test_needs_migration_true(self) -> None:
        """Test migration is needed when old config exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create old config directory
            old_config = temp_path / ".llm-orc"
            old_config.mkdir()

            # Mock home to point to temp_dir
            with patch("pathlib.Path.home", return_value=temp_path):
                # Use different XDG path so new config doesn't exist
                with patch.dict(
                    os.environ, {"XDG_CONFIG_HOME": str(temp_path / "config")}
                ):
                    config_manager = ConfigurationManager()
                    assert config_manager.needs_migration()

    def test_needs_migration_false_no_old_config(self) -> None:
        """Test migration is not needed when old config doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock home to point to temp_dir (no .llm-orc directory)
            with patch("pathlib.Path.home", return_value=temp_path):
                with patch.dict(
                    os.environ, {"XDG_CONFIG_HOME": str(temp_path / "config")}
                ):
                    config_manager = ConfigurationManager()
                    assert not config_manager.needs_migration()

    def test_needs_migration_false_new_config_exists(self) -> None:
        """Test migration is not needed when new config already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create both old and new config directories
            old_config = temp_path / ".llm-orc"
            old_config.mkdir()

            new_config = temp_path / "config" / "llm-orc"
            new_config.mkdir(parents=True)

            # Mock home to point to temp_dir
            with patch("pathlib.Path.home", return_value=temp_path):
                with patch.dict(
                    os.environ, {"XDG_CONFIG_HOME": str(temp_path / "config")}
                ):
                    config_manager = ConfigurationManager()
                    assert not config_manager.needs_migration()

    def test_migrate_from_old_location(self) -> None:
        """Test migrating from old configuration location."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create old config directory with content
            old_config = temp_path / ".llm-orc"
            old_config.mkdir()

            # Add some content to migrate
            credentials_file = old_config / "credentials.yaml"
            credentials_file.write_text("test: data")

            ensembles_dir = old_config / "ensembles"
            ensembles_dir.mkdir()

            ensemble_file = ensembles_dir / "test.yaml"
            ensemble_file.write_text("name: test")

            # Mock home to point to temp_dir
            with patch("pathlib.Path.home", return_value=temp_path):
                with patch.dict(
                    os.environ, {"XDG_CONFIG_HOME": str(temp_path / "config")}
                ):
                    config_manager = ConfigurationManager()
                    config_manager.migrate_from_old_location()

                    # Check that content was migrated
                    new_config = temp_path / "config" / "llm-orc"
                    assert new_config.exists()
                    assert (new_config / "credentials.yaml").exists()
                    assert (new_config / "ensembles" / "test.yaml").exists()

                    # Check that old location is gone
                    assert not old_config.exists()

                    # Check that breadcrumb file was created
                    breadcrumb = temp_path / ".llm-orc-migrated"
                    assert breadcrumb.exists()

    def test_migrate_from_old_location_new_exists(self) -> None:
        """Test migration fails when new config already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create both old and new config directories
            old_config = temp_path / ".llm-orc"
            old_config.mkdir()

            new_config = temp_path / "config" / "llm-orc"
            new_config.mkdir(parents=True)

            # Mock home to point to temp_dir
            with patch("pathlib.Path.home", return_value=temp_path):
                with patch.dict(
                    os.environ, {"XDG_CONFIG_HOME": str(temp_path / "config")}
                ):
                    config_manager = ConfigurationManager()

                    with pytest.raises(
                        ValueError, match="New configuration directory already exists"
                    ):
                        config_manager.migrate_from_old_location()

    def test_load_project_config(self) -> None:
        """Test loading project-specific configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create local config directory with config file
            local_dir = temp_path / ".llm-orc"
            local_dir.mkdir()

            config_data = {
                "project": {"name": "test-project"},
                "model_profiles": {"dev": [{"model": "llama3"}]},
            }

            config_file = local_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Mock cwd to find the local config
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                loaded_config = config_manager.load_project_config()

                assert loaded_config == config_data

    def test_load_project_config_no_local_config(self) -> None:
        """Test loading project config when no local config exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock cwd with no local config
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                loaded_config = config_manager.load_project_config()

                assert loaded_config == {}

    def test_load_project_config_no_config_file(self) -> None:
        """Test loading project config when local dir exists but no config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create local config directory without config file
            local_dir = temp_path / ".llm-orc"
            local_dir.mkdir()

            # Mock cwd to find the local config
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                loaded_config = config_manager.load_project_config()

                assert loaded_config == {}

    def test_init_local_config(self) -> None:
        """Test initializing local configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock cwd to be in temp directory
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                config_manager.init_local_config("test-project")

                # Check directory structure was created
                local_dir = temp_path / ".llm-orc"
                assert local_dir.exists()
                assert (local_dir / "ensembles").exists()
                assert (local_dir / "models").exists()
                assert (local_dir / "scripts").exists()

                # Check config file was created
                config_file = local_dir / "config.yaml"
                assert config_file.exists()

                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                    assert config_data["project"]["name"] == "test-project"
                    assert "model_profiles" in config_data

                # Check .gitignore was created
                gitignore_file = local_dir / ".gitignore"
                assert gitignore_file.exists()

    def test_init_local_config_default_name(self) -> None:
        """Test initializing local config with default project name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock cwd to be in temp directory
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()
                config_manager.init_local_config()

                # Check config file uses directory name
                config_file = temp_path / ".llm-orc" / "config.yaml"
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                    assert config_data["project"]["name"] == temp_path.name

    def test_init_local_config_already_exists(self) -> None:
        """Test initializing local config when it already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create existing local config
            local_dir = temp_path / ".llm-orc"
            local_dir.mkdir()

            # Mock cwd to be in temp directory
            with patch("pathlib.Path.cwd", return_value=temp_path):
                config_manager = ConfigurationManager()

                with pytest.raises(
                    ValueError, match="Local .llm-orc directory already exists"
                ):
                    config_manager.init_local_config()
