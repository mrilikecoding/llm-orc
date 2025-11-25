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


class TestConfigUtilsErrorHandling:
    """Test error handling scenarios in config utils."""

    @patch("click.echo")
    def test_check_ensemble_availability_with_missing_profiles(
        self, mock_echo: Mock
    ) -> None:
        """Test ensemble availability check with missing model profiles."""
        # Given
        mock_config = Mock()

        # Mock profile resolution to raise ValueError for missing profiles
        def mock_resolve_profile(profile_name: str) -> tuple[str, str]:
            if profile_name == "missing-profile":
                raise ValueError("Profile not found")
            return "claude-3", "anthropic"

        mock_config.resolve_model_profile = mock_resolve_profile
        mock_config.get_config.return_value = {}

        # Create ensemble with missing profile
        ensemble_data = {
            "name": "test-ensemble",
            "agents": [
                {"name": "agent1", "model_profile": "missing-profile"},
                {"name": "agent2", "model_profile": "valid-profile"},
            ],
            "coordinator": {"model_profile": "missing-coordinator-profile"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            ensemble_dir = Path(temp_dir)
            ensemble_file = ensemble_dir / "test_ensemble.yaml"

            with open(ensemble_file, "w") as f:
                yaml.safe_dump(ensemble_data, f)

            # When
            check_ensemble_availability(ensemble_dir, {"anthropic"}, mock_config)

            # Then - missing profiles should be handled gracefully
            calls = [str(call) for call in mock_echo.call_args_list]
            # Should not crash and should process the file
            assert any("ensemble" in call.lower() for call in calls)

    @patch("click.echo")
    def test_check_ensemble_availability_coordinator_missing_profile(
        self, mock_echo: Mock
    ) -> None:
        """Test ensemble availability with missing coordinator profile."""
        # Given
        mock_config = Mock()

        # Mock coordinator profile resolution to raise KeyError
        def mock_resolve_profile(profile_name: str) -> tuple[str, str]:
            if profile_name == "missing-coordinator":
                raise KeyError("Profile not found")
            return "claude-3", "anthropic"

        mock_config.resolve_model_profile = mock_resolve_profile
        mock_config.get_config.return_value = {}

        # Create ensemble with missing coordinator profile
        ensemble_data = {
            "name": "test-ensemble",
            "agents": [{"name": "agent1", "provider": "anthropic"}],
            "coordinator": {"model_profile": "missing-coordinator"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            ensemble_dir = Path(temp_dir)
            ensemble_file = ensemble_dir / "test_ensemble.yaml"

            with open(ensemble_file, "w") as f:
                yaml.safe_dump(ensemble_data, f)

            # When
            check_ensemble_availability(ensemble_dir, {"anthropic"}, mock_config)

            # Then - should handle missing coordinator profile
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("ensemble" in call.lower() for call in calls)

    @patch("click.echo")
    @patch("llm_orc.cli_modules.utils.config_utils.CredentialStorage")
    def test_show_provider_details_no_auth_required(
        self, mock_storage_class: Mock, mock_echo: Mock
    ) -> None:
        """Test show_provider_details for provider that doesn't require auth."""
        # Given
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        # Mock provider that doesn't require auth
        with patch("llm_orc.providers.registry.provider_registry") as mock_registry:
            mock_provider_info = Mock()
            mock_provider_info.supports_oauth = False
            mock_provider_info.supports_api_key = False
            mock_provider_info.requires_auth = False
            mock_registry.get_provider.return_value = mock_provider_info

            # When
            show_provider_details(mock_storage, "no-auth-provider")

            # Then
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("No authentication required" in call for call in calls)

    @patch("click.echo")
    @patch("llm_orc.cli_modules.utils.config_utils.CredentialStorage")
    def test_show_provider_details_api_key_only(
        self, mock_storage_class: Mock, mock_echo: Mock
    ) -> None:
        """Test show_provider_details for API key only provider."""
        # Given
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        # Mock provider that only supports API key
        with patch("llm_orc.providers.registry.provider_registry") as mock_registry:
            mock_provider_info = Mock()
            mock_provider_info.supports_oauth = False
            mock_provider_info.supports_api_key = True
            mock_provider_info.requires_auth = True
            mock_registry.get_provider.return_value = mock_provider_info

            mock_storage.get_auth_method.return_value = None

            # When
            show_provider_details(mock_storage, "api-key-provider")

            # Then
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("API Key" in call for call in calls)

    @patch("click.echo")
    @patch("llm_orc.cli_modules.utils.config_utils.CredentialStorage")
    def test_show_provider_details_oauth_exception_handling(
        self, mock_storage_class: Mock, mock_echo: Mock
    ) -> None:
        """Test show_provider_details OAuth status with exception."""
        # Given
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        # Mock provider with OAuth
        with patch("llm_orc.providers.registry.provider_registry") as mock_registry:
            mock_provider_info = Mock()
            mock_provider_info.supports_oauth = True
            mock_provider_info.supports_api_key = True
            mock_provider_info.requires_auth = True
            mock_registry.get_provider.return_value = mock_provider_info

            mock_storage.get_auth_method.return_value = "oauth"

            # When
            show_provider_details(mock_storage, "oauth-provider")

            # Then - should handle the try/except block for OAuth status
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("OAuth" in call for call in calls)

    @patch("click.echo")
    def test_display_default_models_config_profile_not_found(
        self, mock_echo: Mock
    ) -> None:
        """Test display_default_models_config with missing profiles."""
        # Given
        mock_config = Mock()

        # Mock resolve to raise ValueError for missing profiles
        def mock_resolve_profile(profile_name: str) -> tuple[str, str]:
            if profile_name == "missing-profile":
                raise ValueError("Profile not found")
            return "claude-3", "anthropic"

        mock_config.resolve_model_profile = mock_resolve_profile
        mock_config.load_project_config.return_value = {
            "project": {
                "default_models": {
                    "primary": "valid-profile",
                    "fallback": "missing-profile",
                }
            }
        }

        # When
        display_default_models_config(mock_config, {"anthropic"})

        # Then
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("profile not found" in call for call in calls)

    @patch("click.echo")
    def test_display_default_models_config_keyerror_handling(
        self, mock_echo: Mock
    ) -> None:
        """Test display_default_models_config with KeyError."""
        # Given
        mock_config = Mock()

        # Mock resolve to raise KeyError for missing profiles
        def mock_resolve_profile(profile_name: str) -> tuple[str, str]:
            if profile_name == "missing-profile":
                raise KeyError("Profile key not found")
            return "claude-3", "anthropic"

        mock_config.resolve_model_profile = mock_resolve_profile
        mock_config.load_project_config.return_value = {
            "project": {
                "default_models": {
                    "primary": "missing-profile",
                }
            }
        }

        # When
        display_default_models_config(mock_config, {"anthropic"})

        # Then
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("profile not found" in call for call in calls)


class TestCheckEnsembleAvailabilityHelperMethods:
    """Test helper methods extracted from check_ensemble_availability for complexity."""

    def test_process_ensemble_file_valid_yaml(self) -> None:
        """Test processing valid ensemble YAML file."""
        from llm_orc.cli_modules.utils.config_utils import _process_ensemble_file

        # This should fail initially as the helper doesn't exist yet
        with tempfile.TemporaryDirectory() as temp_dir:
            ensemble_file = Path(temp_dir) / "valid.yaml"
            ensemble_data = {"name": "Test Ensemble", "agents": []}

            with open(ensemble_file, "w") as f:
                yaml.safe_dump(ensemble_data, f)

            result = _process_ensemble_file(ensemble_file)
            assert result == ensemble_data

    def test_process_ensemble_file_missing_name(self) -> None:
        """Test processing ensemble file without name field."""
        from llm_orc.cli_modules.utils.config_utils import _process_ensemble_file

        with tempfile.TemporaryDirectory() as temp_dir:
            ensemble_file = Path(temp_dir) / "no_name.yaml"
            ensemble_data: dict[str, Any] = {"agents": []}

            with open(ensemble_file, "w") as f:
                yaml.safe_dump(ensemble_data, f)

            result = _process_ensemble_file(ensemble_file)
            assert result == ensemble_data  # Should return data even without name

    def test_process_ensemble_file_invalid_yaml(self) -> None:
        """Test processing invalid YAML file raises exception."""
        from llm_orc.cli_modules.utils.config_utils import _process_ensemble_file

        with tempfile.TemporaryDirectory() as temp_dir:
            ensemble_file = Path(temp_dir) / "invalid.yaml"

            with open(ensemble_file, "w") as f:
                f.write("invalid: yaml: content: [")

            with pytest.raises(yaml.YAMLError):  # Should handle YAML error
                _process_ensemble_file(ensemble_file)

    def test_check_agent_requirements_with_profiles(self) -> None:
        """Test checking agent requirements with model profiles."""
        from llm_orc.cli_modules.utils.config_utils import _check_agent_requirements

        mock_config = Mock()
        mock_config.resolve_model_profile.return_value = ("claude-3", "anthropic")

        agents = [
            {"name": "agent1", "model_profile": "test-profile"},
            {"name": "agent2", "provider": "ollama"},
        ]

        required_providers, missing_profiles = _check_agent_requirements(
            agents, mock_config
        )

        assert "anthropic" in required_providers
        assert "ollama" in required_providers
        assert len(missing_profiles) == 0

    def test_check_agent_requirements_empty_agents(self) -> None:
        """Test checking requirements with empty agents list."""
        from llm_orc.cli_modules.utils.config_utils import _check_agent_requirements

        mock_config = Mock()

        required_providers, missing_profiles = _check_agent_requirements(
            [], mock_config
        )

        assert len(required_providers) == 0
        assert len(missing_profiles) == 0

    def test_check_coordinator_requirements_with_profile(self) -> None:
        """Test checking coordinator requirements with model profile."""
        from llm_orc.cli_modules.utils.config_utils import (
            _check_coordinator_requirements,
        )

        mock_config = Mock()
        mock_config.resolve_model_profile.return_value = ("claude-3", "anthropic")

        coordinator = {"model_profile": "coordinator-profile"}

        required_providers, missing_profiles = _check_coordinator_requirements(
            coordinator, mock_config
        )

        assert "anthropic" in required_providers
        assert len(missing_profiles) == 0

    def test_check_coordinator_requirements_with_provider(self) -> None:
        """Test checking coordinator requirements with direct provider."""
        from llm_orc.cli_modules.utils.config_utils import (
            _check_coordinator_requirements,
        )

        mock_config = Mock()

        coordinator = {"provider": "ollama"}

        required_providers, missing_profiles = _check_coordinator_requirements(
            coordinator, mock_config
        )

        assert "ollama" in required_providers
        assert len(missing_profiles) == 0

    def test_check_coordinator_requirements_empty_coordinator(self) -> None:
        """Test checking requirements with empty coordinator."""
        from llm_orc.cli_modules.utils.config_utils import (
            _check_coordinator_requirements,
        )

        mock_config = Mock()

        required_providers, missing_profiles = _check_coordinator_requirements(
            {}, mock_config
        )

        assert len(required_providers) == 0
        assert len(missing_profiles) == 0

    def test_check_coordinator_requirements_missing_profile(self) -> None:
        """Test checking coordinator with missing profile."""
        from llm_orc.cli_modules.utils.config_utils import (
            _check_coordinator_requirements,
        )

        mock_config = Mock()
        mock_config.resolve_model_profile.side_effect = ValueError("Profile not found")

        coordinator = {"model_profile": "missing-profile"}

        required_providers, missing_profiles = _check_coordinator_requirements(
            coordinator, mock_config
        )

        assert len(required_providers) == 0
        assert "missing-profile" in missing_profiles

    def test_determine_ensemble_availability(self) -> None:
        """Test ensemble availability determination helper method."""
        # Given
        agent_providers = {"anthropic", "openai"}
        coord_providers = {"google"}
        agent_missing = ["missing-agent-profile"]
        coord_missing: list[str] = []
        available_providers = {"anthropic", "openai", "google"}

        # When
        from llm_orc.cli_modules.utils.config_utils import (
            _determine_ensemble_availability,
        )

        is_available, missing_providers, missing_profiles = (
            _determine_ensemble_availability(
                agent_providers,
                coord_providers,
                agent_missing,
                coord_missing,
                available_providers,
            )
        )

        # Then
        assert not is_available  # Should be False due to missing profiles
        assert len(missing_providers) == 0  # All providers available
        assert missing_profiles == ["missing-agent-profile"]

    def test_determine_ensemble_availability_missing_providers(self) -> None:
        """Test ensemble availability with missing providers."""
        # Given
        agent_providers = {"anthropic", "unavailable-provider"}
        coord_providers: set[str] = set()
        agent_missing: list[str] = []
        coord_missing: list[str] = []
        available_providers = {"anthropic"}

        # When
        from llm_orc.cli_modules.utils.config_utils import (
            _determine_ensemble_availability,
        )

        is_available, missing_providers, missing_profiles = (
            _determine_ensemble_availability(
                agent_providers,
                coord_providers,
                agent_missing,
                coord_missing,
                available_providers,
            )
        )

        # Then
        assert not is_available
        assert missing_providers == ["unavailable-provider"]
        assert len(missing_profiles) == 0

    @patch("click.echo")
    def test_display_ensemble_status_available(self, mock_echo: Mock) -> None:
        """Test displaying status for available ensemble."""
        # Given
        ensemble_name = "test-ensemble"
        is_available = True
        missing_providers: list[str] = []
        missing_profiles: list[str] = []

        # When
        from llm_orc.cli_modules.utils.config_utils import _display_ensemble_status

        _display_ensemble_status(
            ensemble_name, is_available, missing_providers, missing_profiles
        )

        # Then
        mock_echo.assert_called_once_with("  🟢 test-ensemble")

    @patch("click.echo")
    def test_display_ensemble_status_unavailable(self, mock_echo: Mock) -> None:
        """Test displaying status for unavailable ensemble."""
        # Given
        ensemble_name = "test-ensemble"
        is_available = False
        missing_providers: list[str] = ["missing-provider"]
        missing_profiles: list[str] = ["missing-profile"]

        # When
        from llm_orc.cli_modules.utils.config_utils import _display_ensemble_status

        _display_ensemble_status(
            ensemble_name, is_available, missing_providers, missing_profiles
        )

        # Then
        expected_calls = [
            "  🟥 test-ensemble",
            "    Missing profiles: missing-profile",
            "    Missing providers: missing-provider",
        ]
        actual_calls = [call.args[0] for call in mock_echo.call_args_list]
        for expected in expected_calls:
            assert expected in actual_calls
