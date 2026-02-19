"""Main CLI command implementations."""

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click
import yaml

from llm_orc.cli_modules.utils.config_utils import (
    display_local_profiles,
    get_available_providers,
)
from llm_orc.cli_modules.utils.visualization import (
    run_standard_execution,
    run_streaming_execution,
)
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


def _resolve_input_data(positional_input: str | None, option_input: str | None) -> str:
    """Resolve input data using priority: positional > option > stdin > default.

    Args:
        positional_input: Input data from positional argument
        option_input: Input data from --input option

    Returns:
        str: Resolved input data
    """
    # Handle input data priority: positional > option > stdin > default
    final_input_data = positional_input or option_input

    if final_input_data is None:
        if not sys.stdin.isatty():
            # Read from stdin (piped input)
            final_input_data = sys.stdin.read().strip()
        else:
            # No input provided and not piped, use default
            final_input_data = "Please analyze this."

    return final_input_data


def _find_ensemble_config(
    ensemble_name: str, ensemble_dirs: list[Path]
) -> EnsembleConfig:
    """Find ensemble configuration in the provided directories.

    Args:
        ensemble_name: Name of the ensemble to find
        ensemble_dirs: List of directories to search

    Returns:
        EnsembleConfig: The found ensemble configuration

    Raises:
        click.ClickException: If ensemble is not found in any directory
    """
    # Find ensemble in the directories
    loader = EnsembleLoader()
    ensemble_config = None

    for ensemble_dir in ensemble_dirs:
        ensemble_config = loader.find_ensemble(str(ensemble_dir), ensemble_name)
        if ensemble_config is not None:
            break

    if ensemble_config is None:
        searched_dirs = [str(d) for d in ensemble_dirs]
        raise click.ClickException(
            f"Ensemble '{ensemble_name}' not found in: {', '.join(searched_dirs)}"
        )

    return ensemble_config


def _get_grouped_ensembles(
    config_manager: ConfigurationManager, ensemble_dirs: list[Path]
) -> tuple[list[EnsembleConfig], list[EnsembleConfig], list[EnsembleConfig]]:
    """Group ensembles into local, library, and global categories.

    Args:
        config_manager: Configuration manager instance
        ensemble_dirs: List of ensemble directories to search

    Returns:
        tuple: (local_ensembles, library_ensembles, global_ensembles)
    """
    loader = EnsembleLoader()
    local_ensembles: list[EnsembleConfig] = []
    library_ensembles: list[EnsembleConfig] = []
    global_ensembles: list[EnsembleConfig] = []
    cwd = Path.cwd()

    for dir_path in ensemble_dirs:
        ensembles = loader.list_ensembles(str(dir_path))
        is_local = config_manager.local_config_dir and str(dir_path).startswith(
            str(config_manager.local_config_dir)
        )
        is_library = str(dir_path).startswith(str(cwd / "llm-orchestra-library"))

        if is_local:
            local_ensembles.extend(ensembles)
        elif is_library:
            library_ensembles.extend(ensembles)
        else:
            global_ensembles.extend(ensembles)

    return local_ensembles, library_ensembles, global_ensembles


def _format_ensemble_display_name(ensemble: EnsembleConfig) -> str:
    """Format ensemble name for display."""
    if ensemble.relative_path:
        return f"{ensemble.relative_path}/{ensemble.name}"
    return ensemble.name


def _display_ensemble_group(ensembles: Sequence[EnsembleConfig], header: str) -> None:
    """Display a group of ensembles with header."""
    if not ensembles:
        return
    click.echo(f"\n{header}")
    for ensemble in sorted(ensembles, key=lambda e: (e.relative_path or "", e.name)):
        display_name = _format_ensemble_display_name(ensemble)
        click.echo(f"  {display_name}: {ensemble.description}")


def _display_grouped_ensembles(
    config_manager: ConfigurationManager,
    local_ensembles: Sequence[EnsembleConfig],
    library_ensembles: Sequence[EnsembleConfig],
    global_ensembles: Sequence[EnsembleConfig],
) -> None:
    """Display grouped ensembles with proper formatting.

    Args:
        config_manager: Configuration manager instance
        local_ensembles: List of local ensemble configs
        library_ensembles: List of library ensemble configs
        global_ensembles: List of global ensemble configs
    """
    click.echo("Available ensembles:")

    _display_ensemble_group(local_ensembles, "📁 Local Repo (.llm-orc/ensembles):")
    _display_ensemble_group(
        library_ensembles, "📚 Library (llm-orchestra-library/ensembles):"
    )

    global_header = f"🌐 Global ({config_manager.global_config_dir}/ensembles):"
    _display_ensemble_group(global_ensembles, global_header)


def _determine_ensemble_directories(
    config_manager: ConfigurationManager, config_dir: str | None
) -> list[Path]:
    """Determine ensemble directories from config manager or custom directory.

    Args:
        config_manager: Configuration manager instance
        config_dir: Custom config directory path or None

    Returns:
        List of ensemble directories to search

    Raises:
        click.ClickException: If no ensemble directories found
    """
    if config_dir is None:
        # Use configuration manager to get ensemble directories
        ensemble_dirs = config_manager.get_ensembles_dirs()
        if not ensemble_dirs:
            raise click.ClickException(
                "No ensemble directories found. Run 'llm-orc config init' to set up "
                "local configuration."
            )
        return ensemble_dirs
    else:
        # Use specified config directory
        return [Path(config_dir)]


def _setup_performance_display(
    config_manager: ConfigurationManager,
    executor: "EnsembleExecutor",
    ensemble_name: str,
    ensemble_config: "EnsembleConfig",
    streaming: bool,
    output_format: str | None,
    input_data: str,
) -> None:
    """Setup and display performance configuration for Rich interface.

    Args:
        config_manager: Configuration manager instance
        executor: Ensemble executor
        ensemble_name: Name of the ensemble
        ensemble_config: Ensemble configuration
        streaming: Streaming flag from CLI
        output_format: Output format (None for Rich interface)
        input_data: Input data for display
    """
    if output_format is not None:  # Skip for text/json output
        return

    click.echo(
        f"Executing ensemble '{ensemble_name}' with "
        f"{len(ensemble_config.agents)} agents"
    )
    click.echo("─" * 50)


def _determine_effective_streaming(
    config_manager: ConfigurationManager,
    output_format: str | None,
    streaming: bool,
) -> bool:
    """Determine effective streaming setting based on output format and config.

    Args:
        config_manager: Configuration manager instance
        output_format: Output format (text/json/rich)
        streaming: Streaming flag from CLI

    Returns:
        Whether to use streaming execution
    """
    # For text/JSON output, use standard execution for clean piping output
    # Only use streaming for Rich interface (default) or when explicitly requested
    if output_format in ["json", "text"]:
        return False  # Clean, non-streaming output for piping
    else:
        # Default Rich interface - use streaming
        try:
            performance_config = config_manager.load_performance_config()
            return streaming or performance_config.get("streaming_enabled", True)
        except Exception:
            # Fallback if performance config fails
            return streaming  # Use just the CLI flag


def _execute_ensemble_with_mode(
    executor: "EnsembleExecutor",
    ensemble_config: "EnsembleConfig",
    input_data: str,
    output_format: str | None,
    detailed: bool,
    requires_user_input: bool,
    effective_streaming: bool,
) -> None:
    """Execute ensemble with the appropriate execution mode.

    Args:
        executor: Ensemble executor
        ensemble_config: Ensemble configuration
        input_data: Input data for execution
        output_format: Output format
        detailed: Detailed output flag
        requires_user_input: Whether ensemble requires user input
        effective_streaming: Whether to use streaming execution
    """
    # Convert None output_format to "rich" for execution functions
    execution_format = output_format or "rich"

    if requires_user_input:
        # Interactive execution with streaming visualization for progress control
        asyncio.run(
            run_streaming_execution(
                executor, ensemble_config, input_data, execution_format, detailed
            )
        )
    elif effective_streaming:
        # Streaming execution with Rich status
        asyncio.run(
            run_streaming_execution(
                executor, ensemble_config, input_data, execution_format, detailed
            )
        )
    else:
        # Standard execution
        asyncio.run(
            run_standard_execution(
                executor, ensemble_config, input_data, execution_format, detailed
            )
        )


def invoke_ensemble(
    ensemble_name: str,
    input_data: str | None,
    config_dir: str | None,
    input_data_option: str | None,
    output_format: str | None,
    streaming: bool,
    max_concurrent: int | None,
    detailed: bool,
) -> None:
    """Invoke an ensemble of agents."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Determine ensemble directories
    ensemble_dirs = _determine_ensemble_directories(config_manager, config_dir)

    # Resolve input data using helper method
    input_data = _resolve_input_data(input_data, input_data_option)

    # Find ensemble configuration using helper method
    ensemble_config = _find_ensemble_config(ensemble_name, ensemble_dirs)

    # Create standard executor
    executor = EnsembleExecutor()

    # Check if ensemble contains interactive scripts
    from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler

    input_handler = ScriptUserInputHandler()
    requires_user_input = input_handler.ensemble_requires_user_input(ensemble_config)

    # Override concurrency settings if provided
    if max_concurrent is not None:
        # Apply concurrency limit to executor configuration
        pass  # This would be implemented as needed

    # Show performance configuration only for default Rich interface (not text/json)
    _setup_performance_display(
        config_manager,
        executor,
        ensemble_name,
        ensemble_config,
        streaming,
        output_format,
        input_data,
    )

    # Determine effective streaming setting
    effective_streaming = _determine_effective_streaming(
        config_manager, output_format, streaming
    )

    # Execute the ensemble
    try:
        _execute_ensemble_with_mode(
            executor,
            ensemble_config,
            input_data,
            output_format,
            detailed,
            requires_user_input,
            effective_streaming,
        )
    except Exception as e:
        raise click.ClickException(f"Ensemble execution failed: {str(e)}") from e


def _list_ensembles_from_dir(config_dir: str) -> None:
    """List ensembles from a specific directory."""
    loader = EnsembleLoader()
    ensembles = loader.list_ensembles(config_dir)

    if not ensembles:
        click.echo(f"No ensembles found in {config_dir}")
        click.echo("  (Create .yaml files with ensemble configurations)")
    else:
        click.echo(f"Available ensembles in {config_dir}:")
        for ensemble in ensembles:
            click.echo(f"  {ensemble.name}: {ensemble.description}")


def _list_ensembles_auto(config_manager: ConfigurationManager) -> None:
    """List ensembles from all configured directories."""
    ensemble_dirs = config_manager.get_ensembles_dirs()
    if not ensemble_dirs:
        click.echo("No ensemble directories found.")
        click.echo("Run 'llm-orc config init' to set up local configuration.")
        return

    local_ensembles, library_ensembles, global_ensembles = _get_grouped_ensembles(
        config_manager, ensemble_dirs
    )

    if not local_ensembles and not library_ensembles and not global_ensembles:
        click.echo("No ensembles found in any configured directories:")
        for dir_path in ensemble_dirs:
            click.echo(f"  {dir_path}")
        click.echo("  (Create .yaml files with ensemble configurations)")
        return

    _display_grouped_ensembles(
        config_manager, local_ensembles, library_ensembles, global_ensembles
    )


def list_ensembles_command(config_dir: str | None) -> None:
    """List available ensembles."""
    if config_dir is not None:
        _list_ensembles_from_dir(config_dir)
        return
    _list_ensembles_auto(ConfigurationManager())


def _load_profiles_from_config(config_file: Path) -> dict[str, Any]:
    """Load model profiles from a configuration file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary of model profiles, empty if file doesn't exist or has no profiles
    """
    if not config_file.exists():
        return {}

    with open(config_file) as f:
        config = yaml.safe_load(f) or {}
        profiles: dict[str, Any] = config.get("model_profiles", {})
        return profiles


def _display_global_profile(profile_name: str, profile: Any) -> None:
    """Display a single global profile with validation.

    Args:
        profile_name: Name of the profile
        profile: Profile configuration (should be dict)
    """
    # Handle case where profile is not a dict (malformed YAML)
    if not isinstance(profile, dict):
        click.echo(
            f"  {profile_name}: [Invalid profile format - "
            f"expected dict, got {type(profile).__name__}]"
        )
        return

    model = profile.get("model", "Unknown")
    provider = profile.get("provider", "Unknown")
    cost = profile.get("cost_per_token", "Not specified")

    click.echo(f"  {profile_name}:")
    click.echo(f"    Model: {model}")
    click.echo(f"    Provider: {provider}")
    click.echo(f"    Cost per token: {cost}")


def list_profiles_command() -> None:
    """List available model profiles with their provider/model details."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Get all model profiles (merged global + local)
    all_profiles = config_manager.get_model_profiles()

    if not all_profiles:
        click.echo("No model profiles found.")
        click.echo("Run 'llm-orc config init' to create default profiles.")
        return

    # Load separate global and local profiles for grouping
    global_config_file = config_manager.global_config_dir / "config.yaml"
    global_profiles = _load_profiles_from_config(global_config_file)

    local_profiles = {}
    if config_manager.local_config_dir:
        local_config_file = config_manager.local_config_dir / "config.yaml"
        local_profiles = _load_profiles_from_config(local_config_file)

    click.echo("Available model profiles:")

    # Get available providers for status indicators
    available_providers = get_available_providers(config_manager)

    # Show local profiles first (if any)
    if local_profiles:
        display_local_profiles(local_profiles, available_providers)

    # Show global profiles
    if global_profiles:
        global_config_label = f"Global ({config_manager.global_config_dir}/config.yaml)"
        click.echo(f"\n🌐 {global_config_label}:")
        for profile_name in sorted(global_profiles.keys()):
            # Skip if this profile is overridden by local
            if profile_name in local_profiles:
                click.echo(f"  {profile_name}: (overridden by local)")
                continue

            profile = global_profiles[profile_name]
            _display_global_profile(profile_name, profile)


# Script and Artifact Commands
def _format_json_output(data: Any) -> None:
    """Format and display JSON output."""
    click.echo(json.dumps(data, indent=2))


def scripts_list_command(format_type: str) -> None:
    """List available scripts."""
    from llm_orc.cli_modules.commands.script_commands import list_scripts_impl

    json_output = format_type == "json"
    output = list_scripts_impl(category=None, json_output=json_output)
    click.echo(output)


def scripts_show_command(script_name: str) -> None:
    """Show script documentation."""
    from llm_orc.cli_modules.commands.script_commands import show_script_impl

    try:
        output = show_script_impl(script_name)
        click.echo(output)
    except (FileNotFoundError, KeyError):
        click.echo(f"Script '{script_name}' not found", err=True)
        raise SystemExit(1) from None


def _parse_json_parameters(parameters_json: str | None) -> dict[str, Any]:
    """Parse JSON parameters with error handling."""
    if not parameters_json:
        return {}

    try:
        data = json.loads(parameters_json)
        if isinstance(data, dict):
            return data
        else:
            click.echo("Parameters must be a JSON object", err=True)
            raise SystemExit(1)
    except json.JSONDecodeError as e:
        click.echo("Invalid JSON in parameters", err=True)
        raise SystemExit(1) from e


def scripts_test_command(script_name: str, parameters_json: str | None) -> None:
    """Test script with parameters."""
    from llm_orc.core.execution.script_resolver import ScriptResolver

    resolver = ScriptResolver()
    parameters = _parse_json_parameters(parameters_json)

    # Execute script test
    result = resolver.test_script(script_name, parameters)

    if result["success"]:
        click.echo(result["output"])
        click.echo(f"Duration: {result['duration_ms']}ms")
    else:
        click.echo(result["output"], err=True)
        if "error" in result:
            click.echo(f"Error: {result['error']}", err=True)
        raise SystemExit(1)


def artifacts_list_command(format_type: str) -> None:
    """List execution artifacts."""
    from llm_orc.core.execution.artifact_manager import ArtifactManager

    manager = ArtifactManager()
    ensembles = manager.list_ensembles()

    if format_type == "json":
        _format_json_output(ensembles)
        return

    if not ensembles:
        click.echo("No artifacts found in .llm-orc/artifacts/")
        return

    click.echo("Available artifacts:")
    for ensemble in ensembles:
        count_str = f"{ensemble['executions_count']} execution"
        if ensemble["executions_count"] != 1:
            count_str += "s"
        click.echo(
            f"  {ensemble['name']}: {count_str}, latest: {ensemble['latest_execution']}"
        )


def _display_agent_artifact(agent: dict[str, Any]) -> None:
    """Display a single agent's artifact result."""
    agent_name = agent.get("name", "Unknown")
    status = agent.get("status", "unknown")
    click.echo(f"  {agent_name}: {status}")

    if status == "completed" and "result" in agent:
        result_text = agent["result"]
        preview = result_text[:100] + ("..." if len(result_text) > 100 else "")
        click.echo(f"    → {preview}")
    elif status == "failed" and "error" in agent:
        click.echo(f"    → Error: {agent['error']}")


def _display_artifact_text_format(ensemble_name: str, results: dict[str, Any]) -> None:
    """Display artifact results in text format."""
    click.echo(f"Ensemble: {results.get('ensemble_name', ensemble_name)}")

    if "timestamp" in results:
        click.echo(f"Executed: {results['timestamp']}")

    if "total_duration_ms" in results:
        duration_s = results["total_duration_ms"] / 1000
        click.echo(f"Duration: {duration_s:.1f}s")

    if "agents" in results and results["agents"]:
        click.echo("\nAgent Results:")
        for agent in results["agents"]:
            _display_agent_artifact(agent)


def artifacts_show_command(
    ensemble_name: str, format_type: str, execution_timestamp: str | None
) -> None:
    """Show latest results for an ensemble."""
    from llm_orc.core.execution.artifact_manager import ArtifactManager

    manager = ArtifactManager()

    if execution_timestamp:
        results = manager.get_execution_results(ensemble_name, execution_timestamp)
    else:
        results = manager.get_latest_results(ensemble_name)

    if not results:
        click.echo(f"No artifacts found for ensemble '{ensemble_name}'", err=True)
        raise SystemExit(1)

    if format_type == "json":
        _format_json_output(results)
        return

    _display_artifact_text_format(ensemble_name, results)
