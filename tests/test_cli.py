"""Tests for CLI interface."""

import tempfile

import yaml
from click.testing import CliRunner

from llm_orc.cli import cli


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
                    {"name": "agent1", "role": "tester", "model": "claude-3-sonnet"},
                    {"name": "agent2", "role": "reviewer", "model": "claude-3-sonnet"},
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
                with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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
