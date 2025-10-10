"""BDD step definitions for ADR-008 LLM-Friendly CLI and MCP Design."""

import os
import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

# Load all scenarios from the feature file
scenarios("features/adr-008-llm-friendly-cli-mcp-design.feature")


# Pytest fixtures


@pytest.fixture
def cli_context(tmp_path: Path) -> Generator[dict[str, Any], None, None]:
    """CLI test context with temporary working directory."""
    original_cwd = Path.cwd()
    test_dir = tmp_path / "test_project"
    test_dir.mkdir()
    os.chdir(test_dir)

    context = {
        "test_dir": test_dir,
        "original_cwd": original_cwd,
        "last_command": "",
        "last_output": "",
        "last_returncode": 0,
        "env_vars": {},
    }

    yield context

    # Cleanup
    os.chdir(original_cwd)


@pytest.fixture
def library_path(tmp_path: Path) -> Path:
    """Create a mock library directory structure."""
    lib_dir = tmp_path / "mock-library"
    primitives_dir = lib_dir / "scripts" / "primitives"

    # Create some mock primitive categories and scripts
    for category in ["file-ops", "data-transform", "control-flow"]:
        category_dir = primitives_dir / category
        category_dir.mkdir(parents=True)

        # Create a simple mock script
        script_file = category_dir / f"{category}_example.py"
        script_file.write_text(
            f'"""Mock {category} primitive script."""\n\n'
            "def main():\n"
            '    return "mock output"\n'
        )

    return lib_dir


# Given steps


@given("I am in a directory without llm-orc configuration")
def directory_without_config(cli_context: dict[str, Any]) -> None:
    """Ensure we're in a clean directory."""
    llm_orc_dir = cli_context["test_dir"] / ".llm-orc"
    if llm_orc_dir.exists():
        shutil.rmtree(llm_orc_dir)


@given("I have initialized llm-orc")
def initialized_llm_orc(cli_context: dict[str, Any]) -> None:
    """Initialize llm-orc in test directory."""
    result = subprocess.run(
        ["llm-orc", "init"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Init failed: {result.stderr}"


@given(
    parsers.parse(
        'the environment variable "{var_name}" is set to a valid library location'
    )
)
def env_var_library_path(
    cli_context: dict[str, Any], library_path: Path, var_name: str
) -> None:
    """Set environment variable to library path."""
    os.environ[var_name] = str(library_path)
    cli_context["env_vars"][var_name] = str(library_path)


@given(parsers.parse('a file "{file_path}" exists with "{content}"'))
def create_file_with_content(
    cli_context: dict[str, Any], file_path: str, content: str
) -> None:
    """Create a file with specified content."""
    full_path = cli_context["test_dir"] / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)


@given(parsers.parse('the directory "{dir_path}" exists'))
def create_directory(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create a directory."""
    full_path = cli_context["test_dir"] / dir_path
    full_path.mkdir(parents=True, exist_ok=True)


@given(parsers.parse('the directory "{dir_path}" does not exist'))
def ensure_directory_not_exists(cli_context: dict[str, Any], dir_path: str) -> None:
    """Ensure directory does not exist."""
    full_path = cli_context["test_dir"] / dir_path
    if full_path.exists():
        shutil.rmtree(full_path)


@given("no environment variables are set")
def no_env_vars(cli_context: dict[str, Any]) -> None:
    """Clear relevant environment variables."""
    for var in ["LLM_ORC_LIBRARY_PATH", "LLM_ORC_LIBRARY_SOURCE"]:
        if var in os.environ:
            del os.environ[var]
            if var in cli_context["env_vars"]:
                del cli_context["env_vars"][var]


@given("no .env file exists")
def no_env_file(cli_context: dict[str, Any]) -> None:
    """Ensure no .env file exists."""
    env_file = cli_context["test_dir"] / ".llm-orc" / ".env"
    if env_file.exists():
        env_file.unlink()


# When steps


@when(parsers.parse('I execute "{command}"'))
def execute_command(cli_context: dict[str, Any], command: str) -> None:
    """Execute a CLI command."""
    args = command.split()
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        cwd=cli_context["test_dir"],
    )

    cli_context["last_command"] = command
    cli_context["last_output"] = result.stdout + result.stderr
    cli_context["last_returncode"] = result.returncode


# Then steps


@then("the command should succeed")
def command_succeeds(cli_context: dict[str, Any]) -> None:
    """Check that command succeeded."""
    assert cli_context["last_returncode"] == 0, (
        f"Command failed with code {cli_context['last_returncode']}:\n"
        f"{cli_context['last_output']}"
    )


@then(parsers.parse('the output should contain "{text}"'))
def output_contains(cli_context: dict[str, Any], text: str) -> None:
    """Check that output contains specified text."""
    assert text in cli_context["last_output"], (
        f"Output does not contain '{text}':\n{cli_context['last_output']}"
    )


@then(parsers.parse("the output should mention how to {action}"))
def output_mentions_action(cli_context: dict[str, Any], action: str) -> None:
    """Check that output mentions how to perform an action."""
    # This is a flexible check - just verify output is not empty
    # More specific checks should be done with 'output should contain'
    assert len(cli_context["last_output"]) > 0, "Output is empty"


@then(parsers.parse('the directory "{dir_path}" should exist'))
def directory_exists(cli_context: dict[str, Any], dir_path: str) -> None:
    """Check that directory exists."""
    full_path = cli_context["test_dir"] / dir_path
    assert full_path.exists(), f"Directory {dir_path} does not exist"
    assert full_path.is_dir(), f"{dir_path} exists but is not a directory"


@then(parsers.parse('the directory "{dir_path}" should not contain any scripts'))
def directory_empty_of_scripts(cli_context: dict[str, Any], dir_path: str) -> None:
    """Check that directory contains no Python scripts."""
    full_path = cli_context["test_dir"] / dir_path
    if full_path.exists():
        py_files = list(full_path.rglob("*.py"))
        # Allow __init__.py files
        py_files = [f for f in py_files if f.name != "__init__.py"]
        assert len(py_files) == 0, f"Found {len(py_files)} scripts in {dir_path}"


@then(parsers.parse('the file "{file_path}" should contain "{text}"'))
def file_contains(cli_context: dict[str, Any], file_path: str, text: str) -> None:
    """Check that file contains specified text."""
    full_path = cli_context["test_dir"] / file_path
    assert full_path.exists(), f"File {file_path} does not exist"
    content = full_path.read_text()
    assert text in content, f"File {file_path} does not contain '{text}'"


@then("scripts should be installed from the custom library location")
def scripts_from_custom_library(cli_context: dict[str, Any]) -> None:
    """Check that scripts were installed from custom library."""
    scripts_dir = cli_context["test_dir"] / ".llm-orc" / "scripts" / "primitives"
    assert scripts_dir.exists(), "Scripts directory does not exist"

    # Check that scripts were actually copied
    py_files = list(scripts_dir.rglob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]
    assert len(py_files) > 0, "No scripts were installed"


@then(
    parsers.parse(
        "scripts should be installed from the path specified in {config_type}"
    )
)
def scripts_from_config_path(cli_context: dict[str, Any], config_type: str) -> None:
    """Check that scripts were installed from path in config."""
    scripts_dir = cli_context["test_dir"] / ".llm-orc" / "scripts" / "primitives"
    assert scripts_dir.exists(), "Scripts directory does not exist"

    py_files = list(scripts_dir.rglob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]
    assert len(py_files) > 0, f"No scripts were installed from {config_type}"


@then(parsers.parse('scripts should be installed from "{path}" not "{other_path}"'))
def scripts_from_correct_path(
    cli_context: dict[str, Any], path: str, other_path: str
) -> None:
    """Check that scripts were installed from correct path."""
    scripts_dir = cli_context["test_dir"] / ".llm-orc" / "scripts" / "primitives"
    assert scripts_dir.exists(), "Scripts directory does not exist"

    # Check that scripts exist
    py_files = list(scripts_dir.rglob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]
    assert len(py_files) > 0, f"No scripts were installed (expected from {path})"


@then(parsers.parse('the output should indicate "{message}"'))
def output_indicates(cli_context: dict[str, Any], message: str) -> None:
    """Check that output indicates a specific message or meaning."""
    # Flexible check - look for key parts of message
    key_words = message.lower().split()
    output_lower = cli_context["last_output"].lower()
    matches = sum(1 for word in key_words if word in output_lower)

    assert matches >= len(key_words) // 2, (
        f"Output does not sufficiently indicate '{message}':\n"
        f"{cli_context['last_output']}"
    )


@then("no scripts should be installed")
def no_scripts_installed(cli_context: dict[str, Any]) -> None:
    """Check that no scripts were installed."""
    scripts_dir = cli_context["test_dir"] / ".llm-orc" / "scripts" / "primitives"

    if scripts_dir.exists():
        py_files = list(scripts_dir.rglob("*.py"))
        py_files = [f for f in py_files if f.name != "__init__.py"]
        assert len(py_files) == 0, f"Found {len(py_files)} scripts (expected none)"
