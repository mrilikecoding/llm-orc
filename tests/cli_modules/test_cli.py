"""Tests for CLI interface."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from llm_orc.cli import cli
from llm_orc.cli_modules.utils.config_utils import get_available_providers


@pytest.fixture(autouse=True)
def mock_expensive_file_operations():
    """Mock expensive file I/O operations for CLI tests."""
    # Mock only the expensive I/O operations, not entire subsystems
    with patch(
        "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_config"
    ):
        with patch(
            "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_ensembles"
        ):
            with patch(
                "llm_orc.core.config.config_manager.ConfigurationManager._copy_profile_templates"
            ):
                yield


class TestCLI:
    """Test CLI interface."""

    def test_cli_help(self) -> None:
        """Test that CLI shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "llm orchestra" in result.output.lower()

    def test_cli_invoke_command_exists(self) -> None:
        """Test that invoke command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "--help"])
        assert result.exit_code == 0
        assert "invoke" in result.output.lower()

    def test_cli_invoke_requires_ensemble_name(self) -> None:
        """Test that invoke command requires ensemble name."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke"])
        assert result.exit_code != 0
        assert (
            "ensemble" in result.output.lower() or "required" in result.output.lower()
        )

    def test_cli_invoke_with_ensemble_name(self) -> None:
        """Test basic ensemble invocation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "test_ensemble"])
        # Should fail because ensemble doesn't exist
        assert result.exit_code != 0
        # Either no ensemble directories found or ensemble not found in existing dirs
        assert (
            "No ensemble directories found" in result.output
            or "test_ensemble" in result.output
        )

    def test_cli_invoke_with_config_option(self) -> None:
        """Test invoke command accepts config directory option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "--config-dir", "/tmp", "test_ensemble"])
        assert result.exit_code != 0
        # Should show that it's looking in the specified config directory
        assert "test_ensemble" in result.output

    def test_cli_list_command_exists(self) -> None:
        """Test that list-ensembles command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list-ensembles", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "ensemble" in result.output.lower()

    def test_cli_list_ensembles_with_actual_configs(self) -> None:
        """Test listing ensembles when config files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test ensemble file
            ensemble = {
                "name": "test_ensemble",
                "description": "A test ensemble for CLI testing",
                "agents": [
                    {"name": "agent1", "role": "tester", "model": "claude-3-sonnet"}
                ],  # noqa: E501
                "coordinator": {"synthesis_prompt": "Test", "output_format": "json"},
            }

            with open(f"{temp_dir}/test_ensemble.yaml", "w") as f:
                yaml.dump(ensemble, f)

            runner = CliRunner()
            result = runner.invoke(cli, ["list-ensembles", "--config-dir", temp_dir])
            assert result.exit_code == 0
            assert "test_ensemble" in result.output
            assert "A test ensemble for CLI testing" in result.output

    def test_cli_invoke_existing_ensemble(self) -> None:
        """Test invoking an ensemble that exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test ensemble file
            ensemble = {
                "name": "working_ensemble",
                "description": "A working test ensemble",
                "agents": [
                    {
                        "name": "agent1",
                        "role": "tester",
                        "model": "claude-3-sonnet",
                    },
                    {
                        "name": "agent2",
                        "role": "reviewer",
                        "model": "claude-3-sonnet",
                    },
                ],
                "coordinator": {
                    "synthesis_prompt": "Combine results",
                    "output_format": "json",
                },  # noqa: E501
            }

            with open(f"{temp_dir}/working_ensemble.yaml", "w") as f:
                yaml.dump(ensemble, f)

            runner = CliRunner()
            result = runner.invoke(
                cli, ["invoke", "--config-dir", temp_dir, "working_ensemble"]
            )  # noqa: E501
            # Should now succeed and show execution results (using JSON output)
            assert "working_ensemble" in result.output
            # Should see some execution output or JSON structure
            assert result.exit_code == 0 or "execution" in result.output.lower()

    def test_cli_list_ensembles_grouped_output(self) -> None:
        """Test that list-ensembles groups ensembles by location."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_global_dir:
            with tempfile.TemporaryDirectory() as temp_local_dir:
                # Create mock global config
                global_config_path = Path(temp_global_dir)
                global_ensembles_path = global_config_path / "ensembles"
                global_ensembles_path.mkdir(parents=True)

                # Create mock local config
                local_config_path = Path(temp_local_dir)
                local_ensembles_path = local_config_path / "ensembles"
                local_ensembles_path.mkdir(parents=True)

                # Create global ensemble
                global_ensemble = {
                    "name": "global_ensemble",
                    "description": "Global ensemble for testing",
                    "agents": [
                        {"name": "agent1", "role": "tester", "model": "claude-3-sonnet"}
                    ],
                    "coordinator": {
                        "synthesis_prompt": "Test",
                        "output_format": "json",
                    },
                }
                with open(global_ensembles_path / "global_ensemble.yaml", "w") as f:
                    yaml.dump(global_ensemble, f)

                # Create local ensemble
                local_ensemble = {
                    "name": "local_ensemble",
                    "description": "Local ensemble for testing",
                    "agents": [
                        {"name": "agent1", "role": "tester", "model": "claude-3-sonnet"}
                    ],
                    "coordinator": {
                        "synthesis_prompt": "Test",
                        "output_format": "json",
                    },
                }
                with open(local_ensembles_path / "local_ensemble.yaml", "w") as f:
                    yaml.dump(local_ensemble, f)

                # Mock ConfigurationManager to return our test directories
                with patch(
                    "llm_orc.cli_commands.ConfigurationManager"
                ) as mock_config_manager:
                    mock_instance = mock_config_manager.return_value
                    mock_instance.global_config_dir = global_config_path
                    mock_instance.local_config_dir = local_config_path
                    mock_instance.get_ensembles_dirs.return_value = [
                        local_ensembles_path,
                        global_ensembles_path,
                    ]
                    mock_instance.needs_migration.return_value = False

                    runner = CliRunner()
                    result = runner.invoke(cli, ["list-ensembles"])

                    assert result.exit_code == 0

                    # Check that both ensembles are listed
                    assert "local_ensemble" in result.output
                    assert "global_ensemble" in result.output
                    assert "Local ensemble for testing" in result.output
                    assert "Global ensemble for testing" in result.output

                    # Check that they are grouped by location
                    assert "📁 Local Repo (.llm-orc/ensembles):" in result.output
                    expected_global_label = (
                        f"🌐 Global ({global_config_path}/ensembles):"
                    )
                    assert expected_global_label in result.output

                    # Check that local appears before global in output
                    local_index = result.output.find("📁 Local Repo")
                    global_index = result.output.find("🌐 Global")
                    assert local_index < global_index

    def test_auth_setup_provider_selection(self) -> None:
        """Test that auth setup shows only supported providers."""
        from llm_orc.providers.registry import provider_registry

        # Test that we have the expected providers
        providers = provider_registry.list_providers()
        provider_keys = [p.key for p in providers]

        # Should include the specific provider keys
        assert "anthropic-api" in provider_keys
        assert "anthropic-claude-pro-max" in provider_keys
        assert "google-gemini" in provider_keys
        assert "ollama" in provider_keys

        # Should not include generic "anthropic" or "google"
        assert "anthropic" not in provider_keys
        assert "google" not in provider_keys

    def testget_available_providers_with_auth_and_ollama(self) -> None:
        """Test get_available_providers function with authentication and ollama."""
        with tempfile.TemporaryDirectory() as temp_dir:
            global_config_dir = Path(temp_dir)

            # Create mock config manager
            mock_config_manager = Mock()
            mock_config_manager.global_config_dir = global_config_dir

            # Create auth files
            (global_config_dir / "credentials.yaml").touch()
            (global_config_dir / ".encryption_key").touch()

            # Mock CredentialStorage to return test providers
            with patch(
                "llm_orc.cli_modules.utils.config_utils.CredentialStorage"
            ) as mock_storage_class:
                mock_storage = Mock()
                mock_storage.list_providers.return_value = [
                    "anthropic-api",
                    "google-gemini",
                ]
                mock_storage_class.return_value = mock_storage

                # Mock ollama availability
                with patch("requests.get") as mock_requests_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_requests_get.return_value = mock_response

                    # Test the function
                    providers = get_available_providers(mock_config_manager)

                    # Should include authenticated providers + ollama
                    assert "anthropic-api" in providers
                    assert "google-gemini" in providers
                    assert "ollama" in providers
                    assert len(providers) == 3

    def testget_available_providers_no_auth_no_ollama(self) -> None:
        """Test get_available_providers with no authentication and no ollama."""
        with tempfile.TemporaryDirectory() as temp_dir:
            global_config_dir = Path(temp_dir)

            # Create mock config manager (no auth files)
            mock_config_manager = Mock()
            mock_config_manager.global_config_dir = global_config_dir

            # Mock ollama not available
            with patch("requests.get") as mock_requests_get:
                mock_requests_get.side_effect = Exception("Connection refused")

                # Test the function
                providers = get_available_providers(mock_config_manager)

                # Should be empty
                assert len(providers) == 0

    def testget_available_providers_auth_only(self) -> None:
        """Test get_available_providers with only authentication."""
        with tempfile.TemporaryDirectory() as temp_dir:
            global_config_dir = Path(temp_dir)

            # Create mock config manager
            mock_config_manager = Mock()
            mock_config_manager.global_config_dir = global_config_dir

            # Create auth files
            (global_config_dir / "credentials.yaml").touch()

            # Mock CredentialStorage
            with patch(
                "llm_orc.cli_modules.utils.config_utils.CredentialStorage"
            ) as mock_storage_class:
                mock_storage = Mock()
                mock_storage.list_providers.return_value = ["anthropic-claude-pro-max"]
                mock_storage_class.return_value = mock_storage

                # Mock ollama not available
                with patch("requests.get") as mock_requests_get:
                    mock_requests_get.side_effect = Exception("Connection refused")

                    # Test the function
                    providers = get_available_providers(mock_config_manager)

                    # Should only include authenticated provider
                    assert "anthropic-claude-pro-max" in providers
                    assert "ollama" not in providers
                    assert len(providers) == 1

    def test_config_check_global_command_exists(self) -> None:
        """Test that config check-global command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "check-global", "--help"])
        assert result.exit_code == 0
        assert "global" in result.output.lower()

    def test_config_check_local_command_exists(self) -> None:
        """Test that config check-local command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "check-local", "--help"])
        assert result.exit_code == 0
        assert "local" in result.output.lower()

    def test_config_check_unified_command_exists(self) -> None:
        """Test that unified config check command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "check", "--help"])
        assert result.exit_code == 0
        assert "configuration" in result.output.lower()

    def test_config_check_unified_shows_legend(self) -> None:
        """Test that unified config check shows accessibility legend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            global_config_dir = Path(temp_dir)

            # Create minimal config structure
            config_file = global_config_dir / "config.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w") as f:
                yaml.dump({"model_profiles": {}}, f)

            # Mock configuration manager
            with patch(
                "llm_orc.cli_modules.commands.config_commands.ConfigurationManager"
            ) as mock_config_manager_class:
                mock_config_manager = Mock()
                mock_config_manager.global_config_dir = global_config_dir
                mock_config_manager.local_config_dir = None
                mock_config_manager.load_project_config.return_value = None
                mock_config_manager_class.return_value = mock_config_manager

                # Mock get_available_providers
                with patch(
                    "llm_orc.cli_commands.get_available_providers"
                ) as mock_get_providers:
                    mock_get_providers.return_value = set()

                    runner = CliRunner()
                    result = runner.invoke(cli, ["config", "check"])

                    # Should show legend with accessibility symbols
                    assert "Configuration Status Legend:" in result.output
                    assert "🟢 Ready to use" in result.output
                    assert "🟥 Needs setup" in result.output

    def test_config_check_global_shows_availability_indicators(self) -> None:
        """Test that global config check shows availability indicators."""
        with tempfile.TemporaryDirectory() as temp_dir:
            global_config_dir = Path(temp_dir)

            # Create config with test profiles
            config_file = global_config_dir / "config.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_data = {
                "model_profiles": {
                    "available-profile": {
                        "model": "test-model",
                        "provider": "test-provider",
                        "system_prompt": "Test prompt",
                        "timeout_seconds": 30,
                    },
                    "unavailable-profile": {
                        "model": "test-model-2",
                        "provider": "missing-provider",
                        "system_prompt": "Test prompt 2",
                        "timeout_seconds": 60,
                    },
                }
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Mock configuration manager
            with patch(
                "llm_orc.cli_modules.commands.config_commands.ConfigurationManager"
            ) as mock_config_manager_class:
                mock_config_manager = Mock()
                mock_config_manager.global_config_dir = global_config_dir
                # Mock load_project_config to return empty config for this test
                mock_config_manager.load_project_config.return_value = {}
                mock_config_manager_class.return_value = mock_config_manager

                # Mock available providers (only test-provider available)
                with patch(
                    "llm_orc.cli_modules.commands.config_commands.get_available_providers"
                ) as mock_get_providers:
                    mock_get_providers.return_value = {"test-provider"}

                    runner = CliRunner()
                    result = runner.invoke(cli, ["config", "check-global"])

                    # Should show green for available, red for unavailable
                    assert "🟢 available-profile" in result.output
                    assert "🟥 unavailable-profile" in result.output
                    assert "test-provider" in result.output
                    assert "missing-provider" in result.output

    def test_config_check_local_shows_project_name_first(self) -> None:
        """Test that local config check shows project name at the top."""
        with tempfile.TemporaryDirectory() as temp_dir:
            local_config_dir = Path(temp_dir) / ".llm-orc"
            local_config_dir.mkdir(parents=True)

            # Create local config with project name
            config_file = local_config_dir / "config.yaml"
            config_data = {
                "project": {"name": "Test Project Name"},
                "model_profiles": {},
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Mock configuration manager
            with patch(
                "llm_orc.cli_modules.commands.config_commands.ConfigurationManager"
            ) as mock_config_manager_class:
                mock_config_manager = Mock()
                mock_config_manager.load_project_config.return_value = config_data
                mock_config_manager_class.return_value = mock_config_manager

                # Mock available providers
                with patch(
                    "llm_orc.cli_commands.get_available_providers"
                ) as mock_get_providers:
                    mock_get_providers.return_value = set()

                    # Change to temp directory to simulate being in a project
                    import os

                    original_cwd = os.getcwd()
                    try:
                        os.chdir(temp_dir)

                        runner = CliRunner()
                        result = runner.invoke(cli, ["config", "check-local"])

                        # Should show project name in the header
                        assert (
                            "Local Configuration Status: Test Project Name"
                            in result.output
                        )

                    finally:
                        os.chdir(original_cwd)
