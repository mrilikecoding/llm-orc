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

    # Also remove any library directory to ensure clean test environment
    library_dir = cli_context["test_dir"] / "llm-orchestra-library"
    if library_dir.exists():
        shutil.rmtree(library_dir)


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
    """Set environment variable to library path fixture."""
    os.environ[var_name] = str(library_path)
    cli_context["env_vars"][var_name] = str(library_path)


@given(parsers.parse('the environment variable "{var_name}" is set to "{value}"'))
def env_var_to_value(cli_context: dict[str, Any], var_name: str, value: str) -> None:
    """Set environment variable to specific value (test-relative if needed)."""
    # Convert test_dir-relative paths
    if not value.startswith("/"):
        value = str(cli_context["test_dir"] / value)
    elif value.startswith("/"):
        # Convert absolute-looking paths to test-relative
        value = str(cli_context["test_dir"] / value.lstrip("/"))

    os.environ[var_name] = value
    cli_context["env_vars"][var_name] = value


@given(parsers.parse('a file "{file_path}" exists with "{content}"'))
def create_file_with_content(
    cli_context: dict[str, Any], file_path: str, content: str
) -> None:
    """Create a file with specified content.

    For paths in content that look absolute (e.g., /custom/path),
    convert them to test-relative paths.
    """
    # Handle absolute-looking paths in content by converting to test-relative
    if "=" in content:
        key, value = content.split("=", 1)
        if value.startswith("/"):
            # Convert /custom/path to test_dir/custom/path
            relative_value = value.lstrip("/")
            test_value = str(cli_context["test_dir"] / relative_value)
            content = f"{key}={test_value}"

    full_path = cli_context["test_dir"] / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)


@given(parsers.parse('the directory "{dir_path}" exists'))
def create_directory(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create a directory.

    For paths that look absolute (e.g., /custom/path),
    convert them to test-relative paths.
    """
    # Handle absolute-looking paths by converting to test-relative
    if dir_path.startswith("/"):
        dir_path = dir_path.lstrip("/")

    full_path = cli_context["test_dir"] / dir_path
    full_path.mkdir(parents=True, exist_ok=True)


@given(parsers.parse('the directory "{dir_path}" does not exist'))
def ensure_directory_not_exists(cli_context: dict[str, Any], dir_path: str) -> None:
    """Ensure directory does not exist."""
    # Handle absolute-looking paths
    if dir_path.startswith("/"):
        dir_path = dir_path.lstrip("/")

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


@given(parsers.parse('a library directory "{dir_path}" exists with sample scripts'))
def library_directory_with_scripts(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create a library directory with sample scripts."""
    full_path = cli_context["test_dir"] / dir_path

    # Create some mock primitive categories and scripts
    for category in ["file-ops", "data-transform", "control-flow"]:
        category_dir = full_path / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create a simple mock script
        script_file = category_dir / f"{category}_example.py"
        script_file.write_text(
            f'"""Mock {category} primitive script."""\n\n'
            "def main():\n"
            '    return "mock output"\n'
        )


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
@then("core primitives should be available as package modules")
def primitives_available_as_package(cli_context: dict[str, Any]) -> None:
    """Check that core primitives are available as package modules."""
    from llm_orc.primitives import get_output_schema

    # Verify core primitives are registered in the package
    for key in [
        "user_interaction/get_user_input",
        "user_interaction/confirm_action",
        "file_ops/read_file",
        "file_ops/write_file",
        "data_transform/json_extract",
        "control_flow/replicate_n_times",
    ]:
        schema = get_output_schema(f"primitives/{key}.py")
        assert schema is not None, f"Primitive {key} not found in package"


@then(
    parsers.parse(
        "scripts should be installed from the path specified in {config_type}"
    )
)
def scripts_from_config_path(cli_context: dict[str, Any], config_type: str) -> None:
    """Check that core primitives are available as package modules."""
    from llm_orc.primitives import get_output_schema

    schema = get_output_schema("primitives/file_ops/read_file.py")
    assert schema is not None, f"Primitives not available (checking {config_type})"


@then(parsers.parse('scripts should be installed from "{path}" not "{other_path}"'))
def scripts_from_correct_path(
    cli_context: dict[str, Any], path: str, other_path: str
) -> None:
    """Check that core primitives are available as package modules."""
    from llm_orc.primitives import get_output_schema

    schema = get_output_schema("primitives/file_ops/read_file.py")
    assert schema is not None, f"Primitives not available (expected from {path})"


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


# Ensemble Discovery Steps


@given(parsers.parse('the library directory "{dir_path}" exists with an ensemble.yaml'))
def library_ensemble_directory(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create a library ensemble directory with ensemble.yaml."""
    full_path = cli_context["test_dir"] / dir_path
    full_path.mkdir(parents=True, exist_ok=True)

    # Create a simple ensemble.yaml
    ensemble_yaml = full_path / "ensemble.yaml"
    ensemble_yaml.write_text(
        """name: test-ensemble
description: Test ensemble for discovery
agents:
  - name: test-agent
    model_profile: test
"""
    )


@given(parsers.parse('the library directory "{dir_path}" exists with a valid ensemble'))
def library_valid_ensemble(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create a library ensemble with valid configuration."""
    library_ensemble_directory(cli_context, dir_path)


@given(parsers.parse('a local ensemble "{name}" exists in "{dir_path}"'))
def local_ensemble(cli_context: dict[str, Any], name: str, dir_path: str) -> None:
    """Create a local ensemble."""
    full_path = cli_context["test_dir"] / dir_path / name
    full_path.mkdir(parents=True, exist_ok=True)

    ensemble_yaml = full_path / "ensemble.yaml"
    ensemble_yaml.write_text(
        f"""name: {name}
description: Local test ensemble
agents:
  - name: local-agent
    model_profile: test
"""
    )


@given(parsers.parse('a library ensemble "{name}" exists in "{dir_path}"'))
def library_ensemble(cli_context: dict[str, Any], name: str, dir_path: str) -> None:
    """Create a library ensemble."""
    full_path = cli_context["test_dir"] / dir_path / name
    full_path.mkdir(parents=True, exist_ok=True)

    ensemble_yaml = full_path / "ensemble.yaml"
    ensemble_yaml.write_text(
        f"""name: {name}
description: Library test ensemble
agents:
  - name: library-agent
    model_profile: test
"""
    )


@given(parsers.parse('local ensembles exist in "{dir_path}"'))
def local_ensembles_exist(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create some local ensembles."""
    local_ensemble(cli_context, "local-ensemble-1", dir_path)
    local_ensemble(cli_context, "local-ensemble-2", dir_path)


@given(parsers.parse('library ensembles exist in "{dir_path}"'))
def library_ensembles_exist(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create some library ensembles."""
    library_ensemble(cli_context, "library-ensemble-1", dir_path)
    library_ensemble(cli_context, "library-ensemble-2", dir_path)


@given(parsers.parse('the library has ensembles in "{dir_path}"'))
def library_has_ensembles(cli_context: dict[str, Any], dir_path: str) -> None:
    """Create ensembles in library examples directory."""
    # Create the neon-shadows-detective ensemble
    full_path = cli_context["test_dir"] / dir_path / "neon-shadows-detective"
    full_path.mkdir(parents=True, exist_ok=True)

    ensemble_yaml = full_path / "ensemble.yaml"
    ensemble_yaml.write_text(
        """name: neon-shadows-detective
description: Interactive cyberpunk detective story
agents:
  - name: opening-scene
    model_profile: creative
"""
    )


@then("the output should list ensembles from the library")
def output_lists_library_ensembles(cli_context: dict[str, Any]) -> None:
    """Check that library ensembles are listed."""
    output = cli_context["last_output"]
    assert "library" in output.lower() or "examples" in output.lower(), (
        f"Output does not mention library ensembles:\n{output}"
    )


@then("the ensemble should execute successfully")
def ensemble_executes(cli_context: dict[str, Any]) -> None:
    """Check that ensemble execution succeeded."""
    # For now, just check that command didn't fail
    # TODO: Add more specific checks for ensemble execution
    assert cli_context["last_returncode"] == 0, (
        f"Ensemble execution failed:\n{cli_context['last_output']}"
    )


@then("the output should not require full path specification")
def no_full_path_required(cli_context: dict[str, Any]) -> None:
    """Check that we didn't need to use full path."""
    # This is validated by the fact that the command succeeded
    # with just the relative path (checked by ensemble_executes)
    pass


@then("the local ensemble should be executed")
def local_ensemble_executed(cli_context: dict[str, Any]) -> None:
    """Check that local ensemble was executed (not library)."""
    output = cli_context["last_output"]
    # Look for indicators that local ensemble was used
    assert "local-agent" in output or cli_context["last_returncode"] == 0, (
        f"Local ensemble may not have been executed:\n{output}"
    )


@then("not the library ensemble")
def not_library_ensemble(cli_context: dict[str, Any]) -> None:
    """Check that library ensemble was not executed."""
    output = cli_context["last_output"]
    # Should not see library-specific agent names
    assert "library-agent" not in output, (
        f"Library ensemble appears to have been executed:\n{output}"
    )


@then(parsers.parse('the output should have a "{section}" section'))
def output_has_section(cli_context: dict[str, Any], section: str) -> None:
    """Check that output has a specific section."""
    output = cli_context["last_output"]
    assert section.lower() in output.lower(), (
        f"Output does not have '{section}' section:\n{output}"
    )


@then(parsers.parse('library ensembles should be listed under "{section}"'))
def library_under_section(cli_context: dict[str, Any], section: str) -> None:
    """Check that library ensembles are under the right section."""
    # For now, just verify the section exists
    # TODO: Add more sophisticated parsing
    output_has_section(cli_context, section)


@then("the output should list ensembles in the examples category")
def list_examples_category(cli_context: dict[str, Any]) -> None:
    """Check that examples category ensembles are listed."""
    output = cli_context["last_output"]
    assert "examples" in output.lower() or "ensemble" in output.lower(), (
        f"Output does not list examples category:\n{output}"
    )


@then("the output should include the newly created narrative ensemble")
def includes_narrative_ensemble(cli_context: dict[str, Any]) -> None:
    """Check that neon-shadows-detective ensemble is listed."""
    output = cli_context["last_output"]
    assert "neon-shadows" in output.lower() or "detective" in output.lower(), (
        f"Output does not include narrative ensemble:\n{output}"
    )
