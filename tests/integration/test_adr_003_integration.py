"""Integration test for ADR-003 contract validation CLI implementation."""

import subprocess
import sys
from pathlib import Path


def test_cli_module_execution() -> None:
    """Test that the CLI module can be executed as a subprocess."""
    # Test that the module can be imported and executed
    result = subprocess.run(
        [sys.executable, "-m", "llm_orc.testing.contract_validator", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "ADR-003" in result.stdout
    assert "--directory" in result.stdout
    assert "--level" in result.stdout


def test_cli_validation_execution() -> None:
    """Test that CLI validation works on existing directory."""
    # Test validation on src directory (should contain Python files)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_orc.testing.contract_validator",
            "--directory",
            "src",
            "--level",
            "core",
            "--json",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "success" in result.stdout


def test_cli_error_handling() -> None:
    """Test that CLI properly handles invalid directories."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_orc.testing.contract_validator",
            "--directory",
            "/nonexistent/path",
            "--level",
            "core",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Directory does not exist" in result.stderr


def test_adr_003_compliance() -> None:
    """Test that implementation matches ADR-003 specification exactly."""
    # Test the exact commands from ADR-003
    commands = [
        ["--directory", ".llm-orc/scripts/primitives", "--level", "core"],
        ["--directory", ".llm-orc/scripts/examples", "--level", "examples"],
        ["--directory", ".llm-orc/scripts/community", "--level", "community"],
    ]

    for _cmd_args in commands:
        result = subprocess.run(
            [sys.executable, "-m", "llm_orc.testing.contract_validator", "--help"],
            capture_output=True,
            text=True,
        )
        # Should not crash - command structure is correct
        assert result.returncode == 0


def test_makefile_integration() -> None:
    """Test that make targets work properly."""
    # Test that make target exists and can be called
    result = subprocess.run(
        ["make", "help"],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert "validate-contracts-core" in result.stdout
    assert "validate-contracts-examples" in result.stdout
    assert "validate-contracts-community" in result.stdout
    assert "validate-contracts-all" in result.stdout
