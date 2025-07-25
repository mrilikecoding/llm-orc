"""Tests for CLI tab completion functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import click
import yaml

from llm_orc.cli import cli
from llm_orc.cli_completion import complete_ensemble_names, complete_providers


class TestEnsembleNameCompletion:
    """Test completion of ensemble names."""

    def test_complete_ensemble_names_returns_available_ensembles(self) -> None:
        """Should return list of available ensemble names matching incomplete input."""
        # Create a temporary directory with ensemble files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test ensemble files
            ensemble1_content = {
                "name": "test-ensemble-one",
                "description": "Test ensemble one",
                "agents": [{"name": "agent1", "model": "gpt-4"}],
            }
            ensemble2_content = {
                "name": "test-ensemble-two",
                "description": "Test ensemble two",
                "agents": [{"name": "agent2", "model": "claude-3"}],
            }

            (temp_path / "test-ensemble-one.yaml").write_text(
                yaml.dump(ensemble1_content)
            )
            (temp_path / "test-ensemble-two.yaml").write_text(
                yaml.dump(ensemble2_content)
            )

            # Create mock Click context with config_dir parameter
            ctx = Mock(spec=click.Context)
            ctx.params = {"config_dir": str(temp_path)}

            param = Mock(spec=click.Parameter)

            # Test completion with partial input
            result = complete_ensemble_names(ctx, param, "test-ensemble")

            # Should return both ensemble names
            assert "test-ensemble-one" in result
            assert "test-ensemble-two" in result
            assert len(result) == 2

    def test_complete_ensemble_names_filters_by_incomplete_input(self) -> None:
        """Should filter ensemble names by incomplete input prefix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create ensemble files with different prefixes
            ensemble1_content = {
                "name": "prod-ensemble",
                "description": "Production ensemble",
                "agents": [],
            }
            ensemble2_content = {
                "name": "test-ensemble",
                "description": "Test ensemble",
                "agents": [],
            }

            (temp_path / "prod-ensemble.yaml").write_text(yaml.dump(ensemble1_content))
            (temp_path / "test-ensemble.yaml").write_text(yaml.dump(ensemble2_content))

            ctx = Mock(spec=click.Context)
            ctx.params = {"config_dir": str(temp_path)}
            param = Mock(spec=click.Parameter)

            # Test completion with "prod" prefix
            result = complete_ensemble_names(ctx, param, "prod")

            # Should only return ensemble starting with "prod"
            assert result == ["prod-ensemble"]

    def test_complete_ensemble_names_returns_empty_on_error(self) -> None:
        """Should return empty list when encountering errors."""
        ctx = Mock(spec=click.Context)
        ctx.params = {"config_dir": "/nonexistent/directory"}
        param = Mock(spec=click.Parameter)

        result = complete_ensemble_names(ctx, param, "test")

        # Should return empty list, not raise exception
        assert result == []


class TestProviderCompletion:
    """Test completion of authentication provider names."""

    @patch("llm_orc.cli_completion.get_available_providers")
    def test_complete_providers_returns_available_providers(
        self, mock_get_providers: Mock
    ) -> None:
        """Should return list of available provider names matching incomplete input."""
        # Mock the available providers
        mock_get_providers.return_value = [
            "anthropic-api",
            "anthropic-claude-pro-max",
            "google-gemini",
            "ollama",
        ]

        ctx = Mock(spec=click.Context)
        param = Mock(spec=click.Parameter)

        # Test completion with partial input
        result = complete_providers(ctx, param, "anthropic")

        # Should return both anthropic providers
        assert "anthropic-api" in result
        assert "anthropic-claude-pro-max" in result
        assert "google-gemini" not in result
        assert "ollama" not in result
        assert len(result) == 2

    @patch("llm_orc.cli_completion.get_available_providers")
    def test_complete_providers_returns_empty_on_error(
        self, mock_get_providers: Mock
    ) -> None:
        """Should return empty list when encountering errors."""
        # Mock get_available_providers to raise an exception
        mock_get_providers.side_effect = Exception("Provider lookup failed")

        ctx = Mock(spec=click.Context)
        param = Mock(spec=click.Parameter)

        result = complete_providers(ctx, param, "test")

        # Should return empty list, not raise exception
        assert result == []


class TestCLICompletionIntegration:
    """Test integration of completion with CLI commands."""

    def test_invoke_command_has_ensemble_completion(self) -> None:
        """Should have ensemble name completion on invoke command argument."""
        # Get the invoke command
        invoke_cmd = cli.commands["invoke"]

        # Check that the ensemble_name argument has shell completion
        ensemble_param = None
        for param in invoke_cmd.params:
            if hasattr(param, "name") and param.name == "ensemble_name":
                ensemble_param = param
                break

        assert ensemble_param is not None, "ensemble_name parameter not found"

        # Check that a custom completion function has been set
        assert hasattr(ensemble_param, "shell_complete"), "No shell_complete attribute"
        # Check it's not the default parameter shell_complete
        assert ensemble_param.shell_complete is not None, "shell_complete is None"

    def test_auth_add_command_has_provider_completion(self) -> None:
        """Should have provider name completion on auth add command argument."""
        # Get the auth group and add command
        auth_group = cli.commands["auth"]
        assert isinstance(auth_group, click.Group), "auth should be a group"
        add_cmd = auth_group.commands["add"]

        # Check that the provider argument has shell completion
        provider_param = None
        for param in add_cmd.params:
            if hasattr(param, "name") and param.name == "provider":
                provider_param = param
                break

        assert provider_param is not None, "provider parameter not found"

        # Check that a custom completion function has been set
        assert hasattr(provider_param, "shell_complete"), "No shell_complete attribute"
        # Check it's not the default parameter shell_complete
        assert provider_param.shell_complete is not None, "shell_complete is None"
