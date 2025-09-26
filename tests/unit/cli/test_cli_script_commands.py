"""Tests for script and artifact CLI commands."""

import json
from unittest.mock import Mock, patch

from click.testing import CliRunner

from llm_orc.cli import cli


class TestScriptCommands:
    """Test script management CLI commands."""

    def test_scripts_group_exists(self) -> None:
        """Test that scripts command group exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scripts", "--help"])

        assert result.exit_code == 0
        assert "Script management commands" in result.output

    def test_scripts_list_command_basic(self) -> None:
        """Test basic scripts list command."""
        runner = CliRunner()

        mock_resolver = Mock()
        mock_resolver.list_available_scripts.return_value = [
            {
                "name": "test-script.py",
                "display_name": "test-script.py",
                "path": "/path/to/test-script.py",
                "relative_path": None,
            },
            {
                "name": "analyze.sh",
                "display_name": "analyze.sh",
                "path": "/path/to/analyze.sh",
                "relative_path": None,
            },
        ]

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(cli, ["scripts", "list"])

            assert result.exit_code == 0
            assert "Available scripts:" in result.output
            assert "test-script.py" in result.output
            assert "analyze.sh" in result.output

    def test_scripts_list_command_json_output(self) -> None:
        """Test scripts list command with JSON output."""
        runner = CliRunner()

        mock_resolver = Mock()
        mock_resolver.list_available_scripts.return_value = [
            {"name": "test-script.py", "path": "/path/to/test-script.py"},
        ]

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(cli, ["scripts", "list", "--format", "json"])

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            assert len(output_data) == 1
            assert output_data[0]["name"] == "test-script.py"

    def test_scripts_list_command_no_scripts(self) -> None:
        """Test scripts list command when no scripts available."""
        runner = CliRunner()

        mock_resolver = Mock()
        mock_resolver.list_available_scripts.return_value = []

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(cli, ["scripts", "list"])

            assert result.exit_code == 0
            assert "No scripts found in .llm-orc/scripts/" in result.output

    def test_scripts_show_command_basic(self) -> None:
        """Test basic scripts show command."""
        runner = CliRunner()

        script_info = {
            "name": "test-script.py",
            "path": "/path/to/test-script.py",
            "description": "A test script",
            "parameters": ["input", "output"],
        }

        mock_resolver = Mock()
        mock_resolver.get_script_info.return_value = script_info

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(cli, ["scripts", "show", "test-script.py"])

            assert result.exit_code == 0
            assert "Script: test-script.py" in result.output
            assert "Description: A test script" in result.output
            assert "Parameters: input, output" in result.output

    def test_scripts_show_command_script_not_found(self) -> None:
        """Test scripts show command when script not found."""
        runner = CliRunner()

        mock_resolver = Mock()
        mock_resolver.get_script_info.return_value = None

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(cli, ["scripts", "show", "missing-script.py"])

            assert result.exit_code == 1
            assert "Script 'missing-script.py' not found" in result.output

    def test_scripts_test_command_basic(self) -> None:
        """Test basic scripts test command."""
        runner = CliRunner()

        mock_resolver = Mock()
        mock_resolver.test_script.return_value = {
            "success": True,
            "output": "Script executed successfully",
            "duration_ms": 150,
        }

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(
                cli,
                [
                    "scripts",
                    "test",
                    "test-script.py",
                    "--parameters",
                    '{"input": "test"}',
                ],
            )

            assert result.exit_code == 0
            assert "Script executed successfully" in result.output
            assert "Duration: 150ms" in result.output
            mock_resolver.test_script.assert_called_once_with(
                "test-script.py", {"input": "test"}
            )

    def test_scripts_test_command_without_parameters(self) -> None:
        """Test scripts test command without parameters."""
        runner = CliRunner()

        mock_resolver = Mock()
        mock_resolver.test_script.return_value = {
            "success": True,
            "output": "Script executed",
            "duration_ms": 100,
        }

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(cli, ["scripts", "test", "test-script.py"])

            assert result.exit_code == 0
            mock_resolver.test_script.assert_called_once_with("test-script.py", {})

    def test_scripts_test_command_invalid_json(self) -> None:
        """Test scripts test command with invalid JSON parameters."""
        runner = CliRunner()

        result = runner.invoke(
            cli, ["scripts", "test", "test-script.py", "--parameters", "invalid-json"]
        )

        assert result.exit_code == 1
        assert "Invalid JSON in parameters" in result.output

    def test_scripts_test_command_execution_failed(self) -> None:
        """Test scripts test command when execution fails."""
        runner = CliRunner()

        mock_resolver = Mock()
        mock_resolver.test_script.return_value = {
            "success": False,
            "output": "Script failed",
            "error": "Permission denied",
            "duration_ms": 50,
        }

        with patch(
            "llm_orc.core.execution.script_resolver.ScriptResolver",
            return_value=mock_resolver,
        ):
            result = runner.invoke(cli, ["scripts", "test", "test-script.py"])

            assert result.exit_code == 1
            assert "Script failed" in result.output
            assert "Error: Permission denied" in result.output


class TestArtifactCommands:
    """Test artifact management CLI commands."""

    def test_artifacts_group_exists(self) -> None:
        """Test that artifacts command group exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["artifacts", "--help"])

        assert result.exit_code == 0
        assert "Artifact management commands" in result.output

    def test_artifacts_list_command_basic(self) -> None:
        """Test basic artifacts list command."""
        runner = CliRunner()

        mock_manager = Mock()
        mock_manager.list_ensembles.return_value = [
            {
                "name": "test-ensemble",
                "latest_execution": "20240101-120000-123",
                "executions_count": 5,
            },
            {
                "name": "analysis-ensemble",
                "latest_execution": "20240102-140000-456",
                "executions_count": 3,
            },
        ]

        with patch(
            "llm_orc.core.execution.artifact_manager.ArtifactManager",
            return_value=mock_manager,
        ):
            result = runner.invoke(cli, ["artifacts", "list"])

            assert result.exit_code == 0
            assert "Available artifacts:" in result.output
            assert "test-ensemble" in result.output
            assert "analysis-ensemble" in result.output
            assert "5 executions" in result.output

    def test_artifacts_list_command_json_output(self) -> None:
        """Test artifacts list command with JSON output."""
        runner = CliRunner()

        mock_manager = Mock()
        mock_manager.list_ensembles.return_value = [
            {
                "name": "test-ensemble",
                "latest_execution": "20240101-120000-123",
                "executions_count": 2,
            },
        ]

        with patch(
            "llm_orc.core.execution.artifact_manager.ArtifactManager",
            return_value=mock_manager,
        ):
            result = runner.invoke(cli, ["artifacts", "list", "--format", "json"])

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            assert len(output_data) == 1
            assert output_data[0]["name"] == "test-ensemble"

    def test_artifacts_list_command_no_artifacts(self) -> None:
        """Test artifacts list command when no artifacts available."""
        runner = CliRunner()

        mock_manager = Mock()
        mock_manager.list_ensembles.return_value = []

        with patch(
            "llm_orc.core.execution.artifact_manager.ArtifactManager",
            return_value=mock_manager,
        ):
            result = runner.invoke(cli, ["artifacts", "list"])

            assert result.exit_code == 0
            assert "No artifacts found in .llm-orc/artifacts/" in result.output

    def test_artifacts_show_command_basic(self) -> None:
        """Test basic artifacts show command."""
        runner = CliRunner()

        artifact_data = {
            "ensemble_name": "test-ensemble",
            "timestamp": "2024-01-01T12:00:00",
            "total_duration_ms": 2500,
            "agents": [
                {"name": "agent1", "status": "completed", "result": "Success"},
                {"name": "agent2", "status": "failed", "error": "Timeout"},
            ],
        }

        mock_manager = Mock()
        mock_manager.get_latest_results.return_value = artifact_data

        with patch(
            "llm_orc.core.execution.artifact_manager.ArtifactManager",
            return_value=mock_manager,
        ):
            result = runner.invoke(cli, ["artifacts", "show", "test-ensemble"])

            assert result.exit_code == 0
            assert "Ensemble: test-ensemble" in result.output
            assert "Duration: 2.5s" in result.output
            assert "agent1: completed" in result.output
            assert "agent2: failed" in result.output

    def test_artifacts_show_command_ensemble_not_found(self) -> None:
        """Test artifacts show command when ensemble not found."""
        runner = CliRunner()

        mock_manager = Mock()
        mock_manager.get_latest_results.return_value = None

        with patch(
            "llm_orc.core.execution.artifact_manager.ArtifactManager",
            return_value=mock_manager,
        ):
            result = runner.invoke(cli, ["artifacts", "show", "missing-ensemble"])

            assert result.exit_code == 1
            assert "No artifacts found for ensemble 'missing-ensemble'" in result.output

    def test_artifacts_show_command_json_output(self) -> None:
        """Test artifacts show command with JSON output."""
        runner = CliRunner()

        artifact_data = {
            "ensemble_name": "test-ensemble",
            "timestamp": "2024-01-01T12:00:00",
            "agents": [],
        }

        mock_manager = Mock()
        mock_manager.get_latest_results.return_value = artifact_data

        with patch(
            "llm_orc.core.execution.artifact_manager.ArtifactManager",
            return_value=mock_manager,
        ):
            result = runner.invoke(
                cli, ["artifacts", "show", "test-ensemble", "--format", "json"]
            )

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            assert output_data["ensemble_name"] == "test-ensemble"

    def test_artifacts_show_command_with_specific_execution(self) -> None:
        """Test artifacts show command with specific execution timestamp."""
        runner = CliRunner()

        artifact_data = {
            "ensemble_name": "test-ensemble",
            "timestamp": "20240101-120000-123",
            "agents": [],
        }

        mock_manager = Mock()
        mock_manager.get_execution_results.return_value = artifact_data

        with patch(
            "llm_orc.core.execution.artifact_manager.ArtifactManager",
            return_value=mock_manager,
        ):
            result = runner.invoke(
                cli,
                [
                    "artifacts",
                    "show",
                    "test-ensemble",
                    "--execution",
                    "20240101-120000-123",
                ],
            )

            assert result.exit_code == 0
            mock_manager.get_execution_results.assert_called_once_with(
                "test-ensemble", "20240101-120000-123"
            )


class TestCommandIntegration:
    """Test integration of script and artifact commands with main CLI."""

    def test_scripts_command_group_registered(self) -> None:
        """Test that scripts command group is registered with main CLI."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "scripts" in result.output

    def test_artifacts_command_group_registered(self) -> None:
        """Test that artifacts command group is registered with main CLI."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "artifacts" in result.output

    def test_command_aliases_exist(self) -> None:
        """Test that command aliases exist for scripts and artifacts."""
        runner = CliRunner()

        # Test scripts alias
        result = runner.invoke(cli, ["sc", "--help"])
        assert result.exit_code == 0
        assert "Script management commands" in result.output

        # Test artifacts alias
        result = runner.invoke(cli, ["ar", "--help"])
        assert result.exit_code == 0
        assert "Artifact management commands" in result.output
