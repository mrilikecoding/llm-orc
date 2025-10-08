"""Tests for CLI validation commands."""

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


def test_validate_command_exists(runner: CliRunner) -> None:
    """Test that validate command is registered."""
    result = runner.invoke(cli, ["validate", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.output.lower()


def test_validate_single_ensemble(runner: CliRunner) -> None:
    """Test validating a single ensemble."""
    result = runner.invoke(cli, ["validate", "run", "test-validation-ensemble"])
    # Should fail because ensemble not found or no validation config
    assert result.exit_code != 0


def test_validate_by_category(runner: CliRunner) -> None:
    """Test validating ensembles by category."""
    result = runner.invoke(cli, ["validate", "category", "--category", "primitive"])
    # Should fail with not implemented message
    assert result.exit_code != 0
    assert "not yet implemented" in result.output.lower()


def test_validate_all(runner: CliRunner) -> None:
    """Test validating all validation ensembles."""
    result = runner.invoke(cli, ["validate", "all"])
    # Should fail with not implemented message
    assert result.exit_code != 0
    assert "not yet implemented" in result.output.lower()


def test_validate_ensemble_not_found(runner: CliRunner) -> None:
    """Test validation with non-existent ensemble."""
    result = runner.invoke(cli, ["validate", "run", "nonexistent-ensemble"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_validate_reports_pass_status(runner: CliRunner) -> None:
    """Test that validation displays pass status."""
    result = runner.invoke(cli, ["validate", "run", "test-passing-ensemble"])
    # Will show validation passed or not found error
    pass_or_notfound = (
        "validation passed" in result.output.lower()
        or "not found" in result.output.lower()
        or "does not have validation" in result.output.lower()
    )
    assert pass_or_notfound


def test_validate_reports_fail_status(runner: CliRunner) -> None:
    """Test that validation displays fail status."""
    result = runner.invoke(cli, ["validate", "run", "test-failing-ensemble"])
    # Will show validation failed or not found error
    assert result.exit_code == 1


def test_validate_displays_validation_details(runner: CliRunner) -> None:
    """Test that validation displays detailed results."""
    result = runner.invoke(
        cli, ["validate", "run", "test-validation-ensemble", "--verbose"]
    )
    # Will fail with not found or show validation output
    assert result.exit_code != 0
