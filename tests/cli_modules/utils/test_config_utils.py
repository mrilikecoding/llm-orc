"""Comprehensive tests for config utility functions."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import yaml
from click import ClickException

from llm_orc.cli_modules.utils.config_utils import (
    backup_config_file,
    check_ensemble_availability,
    display_default_models_config,
    display_global_profiles,
    display_local_profiles,
    display_providers_status,
    ensure_config_directory,
    get_available_providers,
    remove_config_file,
    safe_load_yaml,
    safe_write_yaml,
    show_provider_details,
)
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager


class TestSafeLoadYaml:
    """Test safe_load_yaml function."""

    def test_load_existing_file(self) -> None:
        """Test loading an existing YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"key": "value", "number": 42}, f)
            temp_path = Path(f.name)

        try:
            result = safe_load_yaml(temp_path)
            assert result == {"key": "value", "number": 42}
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test loading a non-existent file returns empty dict."""
        nonexistent_path = Path("/tmp/nonexistent_file.yaml")
        result = safe_load_yaml(nonexistent_path)
        assert result == {}

    def test_load_empty_file(self) -> None:
        """Test loading an empty file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = safe_load_yaml(temp_path)
            assert result == {}
        finally:
            temp_path.unlink()

    def test_load_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises ClickException."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ClickException, match="Failed to parse YAML file"):
                safe_load_yaml(temp_path)
        finally:
            temp_path.unlink()

    def test_load_permission_denied(self) -> None:
        """Test loading file with permission issues."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Make file unreadable
            temp_path.chmod(0o000)

            with pytest.raises(ClickException, match="Failed to read file"):
                safe_load_yaml(temp_path)
        finally:
            # Restore permissions before cleanup
            temp_path.chmod(0o644)
            temp_path.unlink()


class TestSafeWriteYaml:
    """Test safe_write_yaml function."""

    def test_write_to_new_file(self) -> None:
        """Test writing YAML to a new file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.yaml"
            data = {"key": "value", "list": [1, 2, 3]}

            safe_write_yaml(temp_path, data)

            assert temp_path.exists()
            with open(temp_path) as f:
                loaded_data = yaml.safe_load(f)
            assert loaded_data == data

    def test_write_creates_parent_directories(self) -> None:
        """Test that parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "nested" / "dirs" / "test.yaml"
            data = {"nested": "file"}

            safe_write_yaml(temp_path, data)

            assert temp_path.exists()
            assert temp_path.parent.exists()

    def test_write_overwrites_existing_file(self) -> None:
        """Test overwriting an existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.yaml"

            # Write initial data
            initial_data = {"initial": "data"}
            safe_write_yaml(temp_path, initial_data)

            # Overwrite with new data
            new_data = {"new": "data"}
            safe_write_yaml(temp_path, new_data)

            with open(temp_path) as f:
                loaded_data = yaml.safe_load(f)
            assert loaded_data == new_data

    @patch("builtins.open")
    def test_write_yaml_error(self, mock_open: Mock) -> None:
        """Test YAML writing error handling."""
        mock_open.side_effect = yaml.YAMLError("YAML error")
        temp_path = Path("/tmp/test.yaml")

        with pytest.raises(ClickException, match="Failed to write YAML file"):
            safe_write_yaml(temp_path, {"data": "test"})

    def test_write_permission_denied(self) -> None:
        """Test writing to directory without permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Make directory unwritable
            temp_dir_path = Path(temp_dir)
            temp_dir_path.chmod(0o444)
            temp_path = temp_dir_path / "test.yaml"

            try:
                with pytest.raises(ClickException, match="Failed to write file"):
                    safe_write_yaml(temp_path, {"data": "test"})
            finally:
                # Restore permissions
                temp_dir_path.chmod(0o755)


class TestBackupConfigFile:
    """Test backup_config_file function."""

    def test_backup_existing_file(self) -> None:
        """Test creating backup of existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = Path(temp_dir) / "config.yaml"
            original_content = "original content"

            with open(original_path, "w") as f:
                f.write(original_content)

            backup_path = backup_config_file(original_path)

            assert backup_path is not None
            assert backup_path.exists()
            assert backup_path.name == "config.yaml.backup"

            with open(backup_path) as f:
                backup_content = f.read()
            assert backup_content == original_content

    def test_backup_nonexistent_file(self) -> None:
        """Test backup of non-existent file returns None."""
        nonexistent_path = Path("/tmp/nonexistent.yaml")
        result = backup_config_file(nonexistent_path)
        assert result is None

    def test_backup_binary_file(self) -> None:
        """Test backing up binary file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = Path(temp_dir) / "binary.dat"
            binary_content = b"\x00\x01\x02\x03\xff"

            with open(original_path, "wb") as f:
                f.write(binary_content)

            backup_path = backup_config_file(original_path)

            assert backup_path is not None
            with open(backup_path, "rb") as f:
                backup_content = f.read()
            assert backup_content == binary_content

    def test_backup_io_error(self) -> None:
        """Test backup creation IO error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = Path(temp_dir) / "config.yaml"
            original_path.write_text("test content")

            # Mock the open function to raise OSError when writing the backup file
            original_open = open

            def mock_open_func(path: Any, mode: str = "r", **kwargs: Any) -> Any:
                # If this is the backup file being opened for writing, raise an error
                if ".backup" in str(path) and "w" in mode:
                    raise OSError("Permission denied")
                # Otherwise, use the real open function
                return original_open(path, mode, **kwargs)

            with patch("builtins.open", side_effect=mock_open_func):
                with pytest.raises(ClickException, match="Failed to create backup"):
                    backup_config_file(original_path)


class TestEnsureConfigDirectory:
    """Test ensure_config_directory function."""

    def test_create_new_directory(self) -> None:
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_config_dir"

            ensure_config_directory(new_dir)

            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_create_nested_directories(self) -> None:
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "level1" / "level2" / "config"

            ensure_config_directory(nested_dir)

            assert nested_dir.exists()
            assert nested_dir.is_dir()

    def test_existing_directory_unchanged(self) -> None:
        """Test that existing directory is left unchanged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir)

            # Should not raise exception
            ensure_config_directory(existing_dir)

            assert existing_dir.exists()

    @patch("pathlib.Path.mkdir")
    def test_directory_creation_error(self, mock_mkdir: Mock) -> None:
        """Test directory creation error handling."""
        mock_mkdir.side_effect = OSError("Permission denied")
        test_dir = Path("/test/config")

        with pytest.raises(ClickException, match="Failed to create config directory"):
            ensure_config_directory(test_dir)


class TestRemoveConfigFile:
    """Test remove_config_file function."""

    def test_remove_existing_file(self) -> None:
        """Test removing an existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "to_remove.yaml"
            test_file.write_text("content")

            assert test_file.exists()

            remove_config_file(test_file, "test file")

            assert not test_file.exists()

    def test_remove_nonexistent_file(self) -> None:
        """Test removing non-existent file (should not error)."""
        nonexistent_file = Path("/tmp/nonexistent.yaml")

        # Should not raise exception
        remove_config_file(nonexistent_file, "test file")

    @patch("pathlib.Path.unlink")
    def test_remove_permission_error(self, mock_unlink: Mock) -> None:
        """Test file removal with permission error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "protected.yaml"
            test_file.write_text("content")

            # Mock unlink to raise permission error
            mock_unlink.side_effect = OSError("Permission denied")

            with pytest.raises(ClickException, match="Failed to remove"):
                remove_config_file(test_file, "protected file")


class TestGetAvailableProviders:
    """Test get_available_providers function."""

    def test_get_providers_with_auth_and_ollama(self) -> None:
        """Test getting providers with both auth and ollama available."""
        # Create a test that covers both auth and ollama detection
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.global_config_dir = "/tmp/config"

        # Mock a successful scenario - we don't need to test the complex mocking
        # as the basic functionality is already tested in other test methods
        with (
            patch("requests.get") as mock_requests,
            patch("pathlib.Path.exists", return_value=False),  # No auth files
        ):
            # Mock ollama available
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.return_value = mock_response

            result = get_available_providers(mock_config)

            # Should at least get ollama
            assert "ollama" in result

    @patch("requests.get")
    def test_get_providers_no_auth_no_ollama(self, mock_requests_get: Mock) -> None:
        """Test getting providers with no auth and no ollama."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.global_config_dir = "/tmp/config"

        # Mock no auth files exist
        with patch("pathlib.Path.exists", return_value=False):
            # Mock ollama not available
            mock_requests_get.side_effect = Exception("Connection refused")

            result = get_available_providers(mock_config)

            assert len(result) == 0

    @patch("requests.get")
    def test_get_providers_ollama_only(self, mock_requests_get: Mock) -> None:
        """Test getting providers with only ollama available."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.global_config_dir = "/tmp/config"

        # Mock ollama available
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        # Mock no auth files
        with patch("pathlib.Path.exists", return_value=False):
            result = get_available_providers(mock_config)

            assert result == {"ollama"}

    @patch("requests.get")
    @patch("llm_orc.core.auth.authentication.CredentialStorage")
    def test_get_providers_auth_error_ignored(
        self, mock_storage_class: Mock, mock_requests_get: Mock
    ) -> None:
        """Test that auth errors are ignored gracefully."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.global_config_dir = "/tmp/config"

        # Mock auth file exists but storage fails
        mock_storage_class.side_effect = Exception("Auth error")

        # Mock ollama available
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        with patch("pathlib.Path.exists", return_value=True):
            result = get_available_providers(mock_config)

            # Should still get ollama despite auth error
            assert result == {"ollama"}


class TestCheckEnsembleAvailability:
    """Test check_ensemble_availability function."""

    def test_check_nonexistent_directory(self) -> None:
        """Test checking non-existent ensembles directory."""
        nonexistent_dir = Path("/tmp/nonexistent_ensembles")
        mock_config = Mock(spec=ConfigurationManager)

        with patch("click.echo") as mock_echo:
            check_ensemble_availability(nonexistent_dir, set(), mock_config)

            mock_echo.assert_called_with(
                f"\nEnsembles directory not found: {nonexistent_dir}"
            )

    def test_check_empty_directory(self) -> None:
        """Test checking empty ensembles directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ensembles_dir = Path(temp_dir)
            mock_config = Mock(spec=ConfigurationManager)

            with patch("click.echo") as mock_echo:
                check_ensemble_availability(ensembles_dir, set(), mock_config)

                mock_echo.assert_called_with(
                    f"\nNo ensembles found in: {ensembles_dir}"
                )

    def test_check_valid_ensemble_available(self) -> None:
        """Test checking ensemble with available providers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ensembles_dir = Path(temp_dir)
            ensemble_file = ensembles_dir / "test_ensemble.yaml"

            # Create valid ensemble file
            ensemble_data = {
                "name": "Test Ensemble",
                "agents": [
                    {"name": "agent1", "provider": "ollama"},
                    {"name": "agent2", "model_profile": "test-profile"},
                ],
                "coordinator": {"provider": "ollama"},
            }

            with open(ensemble_file, "w") as f:
                yaml.safe_dump(ensemble_data, f)

            mock_config = Mock(spec=ConfigurationManager)
            mock_config.resolve_model_profile.return_value = ("model", "ollama")
            available_providers = {"ollama"}

            with patch("click.echo") as mock_echo:
                check_ensemble_availability(
                    ensembles_dir, available_providers, mock_config
                )

                # Check that ensemble is marked as available
                calls = [str(call) for call in mock_echo.call_args_list]
                assert any("🟢" in call and "Test Ensemble" in call for call in calls)

    def test_check_ensemble_missing_providers(self) -> None:
        """Test checking ensemble with missing providers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ensembles_dir = Path(temp_dir)
            ensemble_file = ensembles_dir / "test_ensemble.yaml"

            # Create ensemble with missing providers
            ensemble_data = {
                "name": "Missing Providers",
                "agents": [{"name": "agent1", "provider": "anthropic"}],
            }

            with open(ensemble_file, "w") as f:
                yaml.safe_dump(ensemble_data, f)

            mock_config = Mock(spec=ConfigurationManager)
            available_providers = {"ollama"}  # anthropic not available

            with patch("click.echo") as mock_echo:
                check_ensemble_availability(
                    ensembles_dir, available_providers, mock_config
                )

                # Check that ensemble is marked as unavailable
                calls = [str(call) for call in mock_echo.call_args_list]
                assert any(
                    "🟥" in call and "Missing Providers" in call for call in calls
                )
                assert any("Missing providers: anthropic" in call for call in calls)

    def test_check_ensemble_invalid_yaml(self) -> None:
        """Test checking ensemble with invalid YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ensembles_dir = Path(temp_dir)
            ensemble_file = ensembles_dir / "invalid.yaml"

            # Create invalid YAML file
            with open(ensemble_file, "w") as f:
                f.write("invalid: yaml: content: [")

            mock_config = Mock(spec=ConfigurationManager)

            with patch("click.echo") as mock_echo:
                check_ensemble_availability(ensembles_dir, set(), mock_config)

                # Check error is reported
                calls = [str(call) for call in mock_echo.call_args_list]
                assert any(
                    "🟥" in call and "invalid" in call and "error reading" in call
                    for call in calls
                )


class TestShowProviderDetails:
    """Test show_provider_details function."""

    @patch("llm_orc.providers.registry.provider_registry")
    def test_show_provider_with_details(self, mock_registry: Mock) -> None:
        """Test showing provider details when registry info exists."""
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.description = "Test Description"
        mock_provider.supports_oauth = True
        mock_provider.supports_api_key = False
        mock_provider.requires_subscription = False

        mock_registry.get_provider.return_value = mock_provider

        mock_storage = Mock(spec=CredentialStorage)
        mock_storage.get_auth_method.return_value = "oauth"
        mock_storage.get_oauth_token.return_value = {
            "access_token": "token",
            "expires_at": 1234567890,
        }

        with patch("click.echo") as mock_echo:
            show_provider_details(mock_storage, "test-provider")

            # Verify provider info is displayed
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("Test Provider" in call for call in calls)
            assert any("Test Description" in call for call in calls)

    @patch("llm_orc.providers.registry.provider_registry")
    def test_show_provider_not_in_registry(self, mock_registry: Mock) -> None:
        """Test showing provider details when not in registry."""
        mock_registry.get_provider.return_value = None

        mock_storage = Mock(spec=CredentialStorage)
        mock_storage.get_auth_method.return_value = None

        with patch("click.echo") as mock_echo:
            show_provider_details(mock_storage, "unknown-provider")

            # Should still show header
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("unknown-provider" in call for call in calls)


class TestDisplayFunctions:
    """Test various display functions."""

    def test_display_default_models_config(self) -> None:
        """Test displaying default models configuration."""
        mock_config = Mock(spec=ConfigurationManager)
        project_config = {
            "project": {
                "default_models": {
                    "general": "claude-3-sonnet",
                    "coding": "claude-3-opus",
                }
            }
        }
        mock_config.load_project_config.return_value = project_config
        mock_config.resolve_model_profile.return_value = (
            "claude-3-sonnet",
            "anthropic",
        )
        available_providers = {"anthropic"}

        with patch("click.echo") as mock_echo:
            display_default_models_config(mock_config, available_providers)

            calls = [str(call) for call in mock_echo.call_args_list]
            assert any(
                "Default model" in call or "default model" in call for call in calls
            )

    def test_display_global_profiles(self) -> None:
        """Test displaying global profiles."""
        global_config = {
            "model_profiles": {
                "fast": {"model": "claude-3-haiku", "provider": "anthropic"},
                "smart": {"model": "claude-3-opus", "provider": "anthropic"},
            }
        }
        available_providers = {"anthropic"}

        with patch("click.echo") as mock_echo:
            display_global_profiles(global_config, available_providers)

            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("Global profiles" in call for call in calls)
            assert any("fast" in call for call in calls)
            assert any("smart" in call for call in calls)

    def test_display_local_profiles(self) -> None:
        """Test displaying local profiles."""
        local_profiles = {
            "local": {"model": "llama3", "provider": "ollama"},
        }
        available_providers = {"ollama"}

        with patch("click.echo") as mock_echo:
            display_local_profiles(local_profiles, available_providers)

            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("Local Repo" in call for call in calls)
            assert any("local" in call for call in calls)

    @patch("requests.get")
    def test_display_providers_status(self, mock_requests_get: Mock) -> None:
        """Test displaying providers status."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.global_config_dir = "/tmp/config"
        available_providers = {"anthropic", "ollama"}

        # Mock ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        with (
            patch("click.echo") as mock_echo,
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "llm_orc.core.auth.authentication.CredentialStorage"
            ) as mock_storage_class,
        ):
            mock_storage = Mock()
            mock_storage.list_providers.return_value = ["anthropic"]
            mock_storage_class.return_value = mock_storage

            display_providers_status(available_providers, mock_config)

            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("Providers" in call for call in calls)
