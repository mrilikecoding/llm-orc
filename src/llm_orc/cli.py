"""Command line interface for llm-orc."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.config import ConfigurationManager
from llm_orc.ensemble_config import EnsembleLoader
from llm_orc.ensemble_execution import EnsembleExecutor
from llm_orc.mcp_server_runner import MCPServerRunner


def _get_available_providers(config_manager: ConfigurationManager) -> set[str]:
    """Get set of available providers (authenticated + local services)."""
    available_providers = set()

    # Check for authentication files
    global_config_dir = Path(config_manager.global_config_dir)
    auth_files = [
        global_config_dir / "credentials.yaml",
        global_config_dir / ".encryption_key",
        global_config_dir / ".credentials.yaml",
    ]
    auth_found = any(auth_file.exists() for auth_file in auth_files)

    # Get authenticated providers
    if auth_found:
        try:
            storage = CredentialStorage(config_manager)
            auth_providers = storage.list_providers()
            available_providers.update(auth_providers)
        except Exception:
            pass  # Ignore errors for availability check

    # Check ollama availability
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            available_providers.add("ollama")
    except Exception:
        pass  # Ignore errors for availability check

    return available_providers


def _check_ensemble_availability(
    ensembles_dir: Path,
    available_providers: set[str],
    config_manager: ConfigurationManager,
) -> None:
    """Check and display ensemble availability status."""
    if not ensembles_dir.exists():
        click.echo(f"\nEnsembles directory not found: {ensembles_dir}")
        return

    ensemble_files = list(ensembles_dir.glob("*.yaml"))
    if not ensemble_files:
        click.echo(f"\nNo ensembles found in: {ensembles_dir}")
        return

    click.echo(f"\n📁 Ensembles ({len(ensemble_files)} found):")

    import yaml

    for ensemble_file in sorted(ensemble_files):
        try:
            with open(ensemble_file) as f:
                ensemble_data = yaml.safe_load(f) or {}

            ensemble_name = ensemble_data.get("name", ensemble_file.stem)
            agents = ensemble_data.get("agents", [])
            coordinator = ensemble_data.get("coordinator", {})

            # Check all required providers for this ensemble
            required_providers = set()
            missing_profiles: list[str] = []
            missing_providers: list[str] = []

            # Check agent requirements
            for agent in agents:
                if "model_profile" in agent:
                    profile_name = agent["model_profile"]
                    try:
                        _, provider = config_manager.resolve_model_profile(profile_name)
                        required_providers.add(provider)
                    except (ValueError, KeyError):
                        missing_profiles.append(profile_name)
                elif "provider" in agent:
                    required_providers.add(agent["provider"])

            # Check coordinator requirements
            if "model_profile" in coordinator:
                profile_name = coordinator["model_profile"]
                try:
                    _, provider = config_manager.resolve_model_profile(profile_name)
                    required_providers.add(provider)
                except (ValueError, KeyError):
                    missing_profiles.append(profile_name)
            elif "provider" in coordinator:
                required_providers.add(coordinator["provider"])

            # Determine availability
            missing_providers_set = required_providers - available_providers
            missing_providers = list(missing_providers_set)
            is_available = not missing_providers and not missing_profiles

            status_symbol = "🟢" if is_available else "🟥"
            click.echo(f"  {status_symbol} {ensemble_name}")

            # Show details for unavailable ensembles
            if not is_available:
                if missing_profiles:
                    click.echo(f"    Missing profiles: {', '.join(missing_profiles)}")
                if missing_providers:
                    click.echo(f"    Missing providers: {', '.join(missing_providers)}")

        except Exception as e:
            click.echo(f"  🟥 {ensemble_file.stem} (error reading: {e})")


@click.group()
@click.version_option(package_name="llm-orchestra")
def cli() -> None:
    """LLM Orchestra - Multi-agent LLM communication system."""
    pass


@cli.command()
@click.argument("ensemble_name")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
@click.option(
    "--input-data",
    default=None,
    help="Input data for the ensemble (if not provided, reads from stdin)",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format for results",
)
@click.option(
    "--streaming",
    is_flag=True,
    help="Enable streaming execution for real-time progress updates",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="Maximum number of concurrent agents (overrides config)",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed results and performance metrics",
)
def invoke(
    ensemble_name: str,
    config_dir: str,
    input_data: str,
    output_format: str,
    streaming: bool,
    max_concurrent: int,
    detailed: bool,
) -> None:
    """Invoke an ensemble of agents."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Determine ensemble directories
    if config_dir is None:
        # Use configuration manager to get ensemble directories
        ensemble_dirs = config_manager.get_ensembles_dirs()
        if not ensemble_dirs:
            raise click.ClickException(
                "No ensemble directories found. Run 'llm-orc config init' to set up "
                "local configuration."
            )
    else:
        # Use specified config directory
        ensemble_dirs = [Path(config_dir)]

    # Handle input from stdin if not provided via --input
    if input_data is None:
        if not sys.stdin.isatty():
            # Read from stdin (piped input)
            input_data = sys.stdin.read().strip()
        else:
            # No input provided and not piped, use default
            input_data = "Please analyze this."

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

    # Create standard executor
    executor = EnsembleExecutor()

    # Override concurrency settings if provided
    if max_concurrent is not None:
        # Apply concurrency limit to executor configuration
        pass  # This would be implemented as needed

    # Show performance configuration for text output
    if output_format == "text":
        try:
            performance_config = config_manager.load_performance_config()
            effective_concurrency = executor._get_effective_concurrency_limit(
                len(ensemble_config.agents)
            )
            # Determine effective streaming setting (CLI flag overrides config)
            effective_streaming = streaming or performance_config.get(
                "streaming_enabled", False
            )
            click.echo(
                f"🚀 Executing ensemble '{ensemble_name}' with "
                f"{len(ensemble_config.agents)} agents"
            )
            click.echo(
                f"⚡ Performance: max_concurrent={effective_concurrency}, "
                f"streaming={effective_streaming}"
            )
            click.echo("─" * 50)
        except Exception:
            # Fallback to original output if performance config fails
            click.echo(f"Invoking ensemble: {ensemble_name}")
            click.echo(f"Description: {ensemble_config.description}")
            click.echo(f"Agents: {len(ensemble_config.agents)}")
            click.echo(f"Input: {input_data}")
            click.echo("---")

    # Determine effective streaming setting
    performance_config = config_manager.load_performance_config()
    effective_streaming = streaming or performance_config.get(
        "streaming_enabled", False
    )

    # Execute the ensemble
    try:
        if effective_streaming:
            # Streaming execution with Rich status
            async def run_streaming() -> None:
                console = Console()
                agent_statuses: dict[str, str] = {}

                # Initialize with Rich status
                with console.status("Starting execution...", spinner="dots") as status:
                    async for event in executor.execute_streaming(
                        ensemble_config, input_data
                    ):
                        if output_format == "json":
                            click.echo(json.dumps(event, indent=2))
                        else:
                            event_type = event["type"]
                            if event_type == "agent_progress":
                                # Extract agent status from progress data
                                completed_agents = event["data"].get(
                                    "completed_agents", 0
                                )
                                total_agents = event["data"].get(
                                    "total_agents", len(ensemble_config.agents)
                                )

                                # Mark first N agents as completed, rest as pending
                                for i, agent in enumerate(ensemble_config.agents):
                                    if i < completed_agents:
                                        agent_statuses[agent["name"]] = "completed"
                                    elif (
                                        i == completed_agents
                                        and completed_agents < total_agents
                                    ):
                                        agent_statuses[agent["name"]] = "running"
                                    else:
                                        agent_statuses[agent["name"]] = "pending"

                                # Update status display with current dependency graph
                                current_graph = _create_dependency_graph_with_status(
                                    ensemble_config.agents, agent_statuses
                                )
                                status.update(current_graph)

                            elif event_type == "execution_completed":
                                # Final update with all completed
                                final_statuses = {
                                    agent["name"]: "completed"
                                    for agent in ensemble_config.agents
                                }
                                final_graph = _create_dependency_graph_with_status(
                                    ensemble_config.agents, final_statuses
                                )
                                console.print(f"Final: {final_graph}")
                                console.print(
                                    f"Completed in {event['data']['duration']:.2f}s"
                                )

                                if output_format == "text":
                                    _display_results(
                                        event["data"]["results"],
                                        event["data"]["metadata"],
                                        detailed,
                                    )

            asyncio.run(run_streaming())
        else:
            # Standard execution
            async def run_standard() -> None:
                result = await executor.execute(ensemble_config, input_data)
                if output_format == "json":
                    click.echo(json.dumps(result, indent=2))
                else:
                    _display_results(result["results"], result["metadata"], detailed)

            asyncio.run(run_standard())

    except Exception as e:
        raise click.ClickException(f"Ensemble execution failed: {str(e)}") from e


def _create_dependency_graph(agents: list[dict[str, Any]]) -> str:
    """Create horizontal dependency graph: A,B,C → D → E,F → G"""
    return _create_dependency_graph_with_status(agents, {})


def _create_dependency_graph_with_status(
    agents: list[dict[str, Any]], agent_statuses: dict[str, str]
) -> str:
    """Create horizontal dependency graph with status indicators."""
    # Group agents by dependency level
    agents_by_level: dict[int, list[dict[str, Any]]] = {}

    for agent in agents:
        dependencies = agent.get("depends_on", [])
        level = _calculate_agent_level(agent["name"], dependencies, agents)

        if level not in agents_by_level:
            agents_by_level[level] = []
        agents_by_level[level].append(agent)

    # Build horizontal graph: A,B,C → D → E,F → G
    graph_parts = []
    max_level = max(agents_by_level.keys()) if agents_by_level else 0

    for level in range(max_level + 1):
        if level not in agents_by_level:
            continue

        level_agents = agents_by_level[level]
        agent_displays = []

        for agent in level_agents:
            name = agent["name"]
            status = agent_statuses.get(name, "pending")

            # Status indicators without dots
            if status == "running":
                agent_displays.append(f"[yellow]{name}[/yellow]")
            elif status == "completed":
                agent_displays.append(f"[green]{name}[/green]")
            elif status == "failed":
                agent_displays.append(f"[red]{name}[/red]")
            else:
                agent_displays.append(f"[dim]{name}[/dim]")

        # Join agents at same level with commas
        level_text = ", ".join(agent_displays)
        graph_parts.append(level_text)

    # Join levels with arrows
    return " → ".join(graph_parts)


def _calculate_agent_level(
    agent_name: str, dependencies: list[str], all_agents: list[dict[str, Any]]
) -> int:
    """Calculate the dependency level of an agent (0 = no dependencies)."""
    if not dependencies:
        return 0

    # Find the maximum level of all dependencies
    max_dep_level = 0
    for dep_name in dependencies:
        # Find the dependency agent
        dep_agent = next((a for a in all_agents if a["name"] == dep_name), None)
        if dep_agent:
            dep_dependencies = dep_agent.get("depends_on", [])
            dep_level = _calculate_agent_level(dep_name, dep_dependencies, all_agents)
            max_dep_level = max(max_dep_level, dep_level)

    return max_dep_level + 1


def _display_results(
    results: dict[str, Any], metadata: dict[str, Any], detailed: bool = False
) -> None:
    """Display results in a formatted way."""
    if detailed:
        # Show detailed results with all agents
        click.echo("\n📋 Results:")
        click.echo("=" * 50)

        for agent_name, result in results.items():
            if result.get("status") == "success":
                click.echo(f"\n✅ {agent_name}:")
                click.echo(f"   {result['response']}")
            else:
                click.echo(f"\n❌ {agent_name}:")
                click.echo(f"   Error: {result.get('error', 'Unknown error')}")

        # Show performance metrics
        if "usage" in metadata:
            usage = metadata["usage"]
            totals = usage.get("totals", {})
            click.echo("\n📊 Performance Metrics:")
            click.echo(f"   Duration: {metadata['duration']}")
            click.echo(f"   Total tokens: {totals.get('total_tokens', 0):,}")
            click.echo(f"   Total cost: ${totals.get('total_cost_usd', 0.0):.4f}")
            click.echo(f"   Agents: {totals.get('agents_count', 0)}")

            # Show per-agent usage
            agents_usage = usage.get("agents", {})
            if agents_usage:
                click.echo("\n   Per-Agent Usage:")
                for agent_name, agent_usage in agents_usage.items():
                    tokens = agent_usage.get("total_tokens", 0)
                    cost = agent_usage.get("cost_usd", 0.0)
                    duration = agent_usage.get("duration_ms", 0)
                    model = agent_usage.get("model", "unknown")
                    click.echo(
                        f"     {agent_name} ({model}): {tokens:,} tokens, "
                        f"${cost:.4f}, {duration}ms"
                    )
    else:
        # Simplified output: just show final synthesis/result
        _display_simplified_results(results, metadata)


def _display_simplified_results(
    results: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Display simplified results showing only the final output."""
    # Find the final agent (the one with no dependents)
    final_agent = _find_final_agent(results)

    if final_agent and results[final_agent].get("status") == "success":
        click.echo("\n✅ Final Result:")
        click.echo(f"{results[final_agent]['response']}")
    else:
        # Fallback: show last successful agent
        successful_agents = [
            name
            for name, result in results.items()
            if result.get("status") == "success"
        ]
        if successful_agents:
            last_agent = successful_agents[-1]
            click.echo(f"\n✅ Result from {last_agent}:")
            click.echo(f"{results[last_agent]['response']}")
        else:
            click.echo("\n❌ No successful results found")

    # Show minimal performance summary
    if "usage" in metadata:
        totals = metadata["usage"].get("totals", {})
        agents_count = totals.get("agents_count", 0)
        duration = metadata.get("duration", "unknown")
        click.echo(f"\n⚡ {agents_count} agents completed in {duration}")
        click.echo("   Use --detailed flag for full results and metrics")


def _find_final_agent(results: dict[str, Any]) -> str | None:
    """Find the final agent in the dependency chain (the one with no dependents)."""
    # For now, use a simple heuristic: the agent with the highest token count
    # is likely the final agent (since it got input from all previous agents)
    final_agent = None

    for agent_name in results.keys():
        # This is a simple heuristic - in practice we'd want to track dependencies
        # But for now, we can assume the last successful agent is often the final one
        if results[agent_name].get("status") == "success":
            final_agent = agent_name

    return final_agent


@cli.command("list-ensembles")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
def list_ensembles(config_dir: str) -> None:
    """List available ensembles."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    if config_dir is None:
        # Use configuration manager to get ensemble directories
        ensemble_dirs = config_manager.get_ensembles_dirs()
        if not ensemble_dirs:
            click.echo("No ensemble directories found.")
            click.echo("Run 'llm-orc config init' to set up local configuration.")
            return

        # List ensembles from all directories, grouped by location
        loader = EnsembleLoader()
        local_ensembles = []
        global_ensembles = []

        for dir_path in ensemble_dirs:
            ensembles = loader.list_ensembles(str(dir_path))
            is_local = config_manager.local_config_dir and str(dir_path).startswith(
                str(config_manager.local_config_dir)
            )

            if is_local:
                local_ensembles.extend(ensembles)
            else:
                global_ensembles.extend(ensembles)

        # Check if we have any ensembles at all
        if not local_ensembles and not global_ensembles:
            click.echo("No ensembles found in any configured directories:")
            for dir_path in ensemble_dirs:
                click.echo(f"  {dir_path}")
            click.echo("  (Create .yaml files with ensemble configurations)")
            return

        click.echo("Available ensembles:")

        # Show local ensembles first
        if local_ensembles:
            click.echo("\n📁 Local Repo (.llm-orc/ensembles):")
            for ensemble in sorted(local_ensembles, key=lambda e: e.name):
                click.echo(f"  {ensemble.name}: {ensemble.description}")

        # Show global ensembles
        if global_ensembles:
            global_config_label = (
                f"Global ({config_manager.global_config_dir}/ensembles)"
            )
            click.echo(f"\n🌐 {global_config_label}:")
            for ensemble in sorted(global_ensembles, key=lambda e: e.name):
                click.echo(f"  {ensemble.name}: {ensemble.description}")
    else:
        # Use specified config directory
        loader = EnsembleLoader()
        ensembles = loader.list_ensembles(config_dir)

        if not ensembles:
            click.echo(f"No ensembles found in {config_dir}")
            click.echo("  (Create .yaml files with ensemble configurations)")
        else:
            click.echo(f"Available ensembles in {config_dir}:")
            for ensemble in ensembles:
                click.echo(f"  {ensemble.name}: {ensemble.description}")


@cli.command("list-profiles")
def list_profiles() -> None:
    """List available model profiles with their provider/model details."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Get all model profiles (merged global + local)
    all_profiles = config_manager.get_model_profiles()

    if not all_profiles:
        click.echo("No model profiles found.")
        click.echo("Run 'llm-orc config init' to create default profiles.")
        return

    # Get separate global and local profiles for grouping
    global_profiles = {}
    global_config_file = config_manager.global_config_dir / "config.yaml"
    if global_config_file.exists():
        import yaml

        with open(global_config_file) as f:
            global_config = yaml.safe_load(f) or {}
            global_profiles = global_config.get("model_profiles", {})

    local_profiles = {}
    if config_manager.local_config_dir:
        local_config_file = config_manager.local_config_dir / "config.yaml"
        if local_config_file.exists():
            import yaml

            with open(local_config_file) as f:
                local_config = yaml.safe_load(f) or {}
                local_profiles = local_config.get("model_profiles", {})

    click.echo("Available model profiles:")

    # Show local profiles first (if any)
    if local_profiles:
        click.echo("\n📁 Local Repo (.llm-orc/config.yaml):")
        for profile_name in sorted(local_profiles.keys()):
            profile = local_profiles[profile_name]

            # Handle case where profile is not a dict (malformed YAML)
            if not isinstance(profile, dict):
                click.echo(
                    f"  {profile_name}: [Invalid profile format - "
                    f"expected dict, got {type(profile).__name__}]"
                )
                continue

            model = profile.get("model", "Unknown")
            provider = profile.get("provider", "Unknown")
            cost = profile.get("cost_per_token", "Not specified")

            click.echo(f"  {profile_name}:")
            click.echo(f"    Model: {model}")
            click.echo(f"    Provider: {provider}")
            click.echo(f"    Cost per token: {cost}")

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

            # Handle case where profile is not a dict (malformed YAML)
            if not isinstance(profile, dict):
                click.echo(
                    f"  {profile_name}: [Invalid profile format - "
                    f"expected dict, got {type(profile).__name__}]"
                )
                continue

            model = profile.get("model", "Unknown")
            provider = profile.get("provider", "Unknown")
            cost = profile.get("cost_per_token", "Not specified")

            click.echo(f"  {profile_name}:")
            click.echo(f"    Model: {model}")
            click.echo(f"    Provider: {provider}")
            click.echo(f"    Cost per token: {cost}")


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command()
@click.option(
    "--project-name",
    default=None,
    help="Name for the project (defaults to directory name)",
)
def init(project_name: str) -> None:
    """Initialize local .llm-orc configuration for current project."""
    config_manager = ConfigurationManager()

    try:
        config_manager.init_local_config(project_name)
        click.echo("Local configuration initialized successfully!")
        click.echo("Created .llm-orc directory with:")
        click.echo("  - ensembles/   (project-specific ensembles)")
        click.echo("  - models/      (shared model configurations)")
        click.echo("  - scripts/     (project-specific scripts)")
        click.echo("  - config.yaml  (project configuration)")
        click.echo(
            "\nYou can now create project-specific ensembles in .llm-orc/ensembles/"
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@config.command("reset-global")
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup of existing global config (default: True)",
)
@click.option(
    "--preserve-auth/--reset-auth",
    default=True,
    help="Preserve existing authentication credentials (default: True)",
)
@click.confirmation_option(
    prompt="This will reset your global LLM Orchestra configuration. Continue?"
)
def reset_global(backup: bool, preserve_auth: bool) -> None:
    """Reset global configuration to template defaults."""
    import shutil
    from pathlib import Path

    config_manager = ConfigurationManager()
    global_config_dir = Path(config_manager.global_config_dir)

    # Create backup if requested and config exists
    if backup and global_config_dir.exists():
        backup_path = global_config_dir.with_suffix(".backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(global_config_dir, backup_path)
        click.echo(f"📦 Backed up existing config to {backup_path}")

    # Preserve authentication files if requested
    auth_files = []
    if preserve_auth and global_config_dir.exists():
        potential_auth_files = [
            "credentials.yaml",
            ".encryption_key",
            ".credentials.yaml",  # legacy
        ]
        for auth_file in potential_auth_files:
            auth_path = global_config_dir / auth_file
            if auth_path.exists():
                # Save auth file content
                auth_files.append((auth_file, auth_path.read_bytes()))
                click.echo(f"🔐 Preserving authentication file: {auth_file}")

    # Remove existing config directory
    if global_config_dir.exists():
        shutil.rmtree(global_config_dir)

    # Create fresh config directory
    global_config_dir.mkdir(parents=True, exist_ok=True)

    # Copy template to global config
    template_path = Path(__file__).parent / "templates" / "global-config.yaml"
    global_config_path = global_config_dir / "config.yaml"

    if template_path.exists():
        shutil.copy(template_path, global_config_path)
        click.echo("📋 Installed fresh global config from template")

        # Restore authentication files
        if auth_files:
            for auth_file, auth_content in auth_files:
                auth_path = global_config_dir / auth_file
                auth_path.write_bytes(auth_content)
                click.echo(f"🔐 Restored authentication file: {auth_file}")

        click.echo(f"✅ Global config reset complete at {global_config_dir}")

        if preserve_auth and auth_files:
            click.echo("🔐 Authentication credentials preserved")
        elif not preserve_auth:
            click.echo(
                "💡 Note: You may need to reconfigure authentication "
                "with 'llm-orc auth setup'"
            )
    else:
        raise click.ClickException(f"Template not found at {template_path}")


@config.command("check-global")
def check_global() -> None:
    """Check global configuration status."""
    config_manager = ConfigurationManager()
    global_config_dir = Path(config_manager.global_config_dir)
    global_config_path = global_config_dir / "config.yaml"

    click.echo("Global Configuration Status:")
    click.echo(f"Directory: {global_config_dir}")

    if global_config_path.exists():
        click.echo("Status: configured")

        # Show basic info about the config
        try:
            # Get available providers first
            available_providers = _get_available_providers(config_manager)

            # Show providers FIRST, right after status
            # Check for authentication status and configured providers
            auth_files = [
                global_config_dir / "credentials.yaml",
                global_config_dir / ".encryption_key",
                global_config_dir / ".credentials.yaml",
            ]
            auth_found = any(auth_file.exists() for auth_file in auth_files)

            # Build provider display with detailed status
            provider_display = []
            if auth_found:
                from llm_orc.authentication import CredentialStorage

                try:
                    storage = CredentialStorage(config_manager)
                    auth_providers = storage.list_providers()
                    for provider in auth_providers:
                        provider_display.append(f"{provider} (authenticated)")
                except Exception as e:
                    provider_display.append(f"Error reading auth providers: {e}")

            # Check ollama availability with detailed status
            try:
                import requests

                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    provider_display.append("ollama (available)")
                else:
                    provider_display.append("ollama (service running but API error)")
            except requests.exceptions.ConnectionError:
                provider_display.append("ollama (not running)")
            except requests.exceptions.Timeout:
                provider_display.append("ollama (timeout - may be starting)")
            except Exception as e:
                provider_display.append(f"ollama (error: {e})")

            # Display all providers
            if provider_display:
                click.echo(f"\nProviders: {len(available_providers)} available")
                for provider in sorted(provider_display):
                    click.echo(f"  - {provider}")
            else:
                click.echo("\nProviders: none configured")

            # Read ONLY global config file, not merged profiles
            import yaml

            with open(global_config_path) as f:
                global_config = yaml.safe_load(f) or {}

            # Show default model profiles configuration
            project_config = config_manager.load_project_config()
            if project_config:
                default_models = project_config.get("project", {}).get(
                    "default_models", {}
                )
                if default_models:
                    click.echo(
                        f"\n⚙️ Default model profiles ({len(default_models)} found):"
                    )
                    for purpose, profile in default_models.items():
                        # Resolve profile to show actual model and provider
                        try:
                            (
                                resolved_model,
                                resolved_provider,
                            ) = config_manager.resolve_model_profile(profile)
                            # Check if provider is available for status indicator
                            provider_available = (
                                resolved_provider in available_providers
                            )
                            status_symbol = "🟢" if provider_available else "🟥"
                            click.echo(
                                f"  {status_symbol} {purpose}: {profile} → "
                                f"{resolved_model} ({resolved_provider})"
                            )
                            click.echo("    Purpose: fallback model for reliability")
                        except (ValueError, KeyError):
                            click.echo(f"  🟥 {purpose}: {profile} → profile not found")
                            click.echo("    Purpose: fallback model for reliability")
                else:
                    click.echo("\n⚙️ Default model profiles: none configured")

            global_profiles = global_config.get("model_profiles", {})

            # Check global ensembles SECOND
            global_ensembles_dir = global_config_dir / "ensembles"
            _check_ensemble_availability(
                global_ensembles_dir, available_providers, config_manager
            )

            if global_profiles:
                click.echo(f"\n🌐 Global profiles ({len(global_profiles)} found):")
                for profile_name in sorted(global_profiles.keys()):
                    profile = global_profiles[profile_name]
                    model = profile.get("model", "unknown")
                    provider = profile.get("provider", "unknown")
                    cost = profile.get("cost_per_token", "not specified")
                    timeout = profile.get("timeout_seconds", "not specified")
                    has_system_prompt = "system_prompt" in profile

                    # Check if provider is available
                    provider_available = provider in available_providers
                    status_symbol = "🟢" if provider_available else "🟥"

                    timeout_display = (
                        f"{timeout}s" if timeout != "not specified" else timeout
                    )
                    click.echo(
                        f"  {status_symbol} {profile_name}: {model} ({provider})"
                    )
                    system_prompt_indicator = "✓" if has_system_prompt else "✗"
                    click.echo(
                        f"    Cost: {cost}, Timeout: {timeout_display}, "
                        f"System prompt: {system_prompt_indicator}"
                    )

        except Exception as e:
            click.echo(f"Error reading config: {e}")
    else:
        click.echo("Status: missing")
        click.echo("Run 'llm-orc config init' to create it")


@config.command("reset-local")
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup of existing local config (default: True)",
)
@click.option(
    "--preserve-ensembles/--reset-ensembles",
    default=True,
    help="Preserve existing ensembles directory (default: True)",
)
@click.option(
    "--project-name",
    default=None,
    help="Name for the project (defaults to directory name)",
)
@click.confirmation_option(
    prompt="This will reset your local .llm-orc configuration. Continue?"
)
def reset_local(backup: bool, preserve_ensembles: bool, project_name: str) -> None:
    """Reset local .llm-orc configuration to template defaults."""
    import shutil
    from pathlib import Path

    config_manager = ConfigurationManager()
    local_config_dir = Path(".llm-orc")

    if not local_config_dir.exists():
        click.echo("❌ No local .llm-orc directory found")
        click.echo("💡 Run 'llm-orc config init' to create initial local config")
        return

    # Create backup if requested
    if backup:
        backup_path = Path(".llm-orc.backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(local_config_dir, backup_path)
        click.echo(f"📦 Backed up existing local config to {backup_path}")

    # Preserve ensembles if requested
    ensembles_backup = None
    if preserve_ensembles:
        ensembles_dir = local_config_dir / "ensembles"
        if ensembles_dir.exists():
            # Save ensembles directory content
            ensembles_backup = {}
            for ensemble_file in ensembles_dir.glob("*.yaml"):
                ensembles_backup[ensemble_file.name] = ensemble_file.read_text()
            click.echo(f"🎭 Preserving {len(ensembles_backup)} ensemble(s)")

    # Remove existing local config
    shutil.rmtree(local_config_dir)

    # Initialize fresh local config
    try:
        config_manager.init_local_config(project_name)
        click.echo("📋 Created fresh local config from template")

        # Restore ensembles if preserved
        if ensembles_backup:
            ensembles_dir = local_config_dir / "ensembles"
            for ensemble_name, ensemble_content in ensembles_backup.items():
                ensemble_path = ensembles_dir / ensemble_name
                ensemble_path.write_text(ensemble_content)
                click.echo(f"🎭 Restored ensemble: {ensemble_name}")

        click.echo(f"✅ Local config reset complete at {local_config_dir}")

        if preserve_ensembles and ensembles_backup:
            click.echo("🎭 Existing ensembles preserved")
        elif not preserve_ensembles:
            click.echo("💡 Note: All ensembles were reset to template defaults")

    except ValueError as e:
        raise click.ClickException(str(e)) from e


@config.command("check")
def check() -> None:
    """Check both global and local configuration status."""
    # Show legend at the top
    click.echo("Configuration Status Legend:")
    click.echo("🟢 Ready to use (provider authenticated/available)")
    click.echo("🟥 Needs setup (provider not authenticated/available)")
    click.echo("=" * 50)

    # Show global config first
    check_global.callback()  # type: ignore[misc]

    # Add separator
    click.echo("\n" + "=" * 50)

    # Show local config
    check_local.callback()  # type: ignore[misc]
    click.echo("=" * 50)


@config.command("check-local")
def check_local() -> None:
    """Check local .llm-orc configuration status."""
    from pathlib import Path

    local_config_dir = Path(".llm-orc")
    local_config_path = local_config_dir / "config.yaml"

    if local_config_path.exists():
        # Show basic info about the config
        try:
            config_manager = ConfigurationManager()

            # Check project config first to get project name
            project_config = config_manager.load_project_config()
            if project_config:
                project_name = project_config.get("project", {}).get("name", "Unknown")
                click.echo(f"Local Configuration Status: {project_name}")
                click.echo(f"Directory: {local_config_dir.absolute()}")
                click.echo("Status: configured")

                # Get available providers for ensemble checking
                available_providers = _get_available_providers(config_manager)

                # Check local ensembles with availability indicators
                ensembles_dir = local_config_dir / "ensembles"
                _check_ensemble_availability(
                    ensembles_dir, available_providers, config_manager
                )

                # Show local model profiles
                local_profiles = project_config.get("model_profiles", {})
                if local_profiles:
                    click.echo(
                        f"\n💻 Local model profiles ({len(local_profiles)} found):"
                    )
                    for profile_name in sorted(local_profiles.keys()):
                        profile = local_profiles[profile_name]
                        model = profile.get("model", "unknown")
                        provider = profile.get("provider", "unknown")
                        cost = profile.get("cost_per_token", "not specified")
                        timeout = profile.get("timeout_seconds", "not specified")
                        has_system_prompt = "system_prompt" in profile

                        # Check if provider is available
                        provider_available = provider in available_providers
                        status_symbol = "🟢" if provider_available else "🟥"

                        timeout_display = (
                            f"{timeout}s" if timeout != "not specified" else timeout
                        )
                        click.echo(
                            f"  {status_symbol} {profile_name}: {model} ({provider})"
                        )
                        system_prompt_indicator = "✓" if has_system_prompt else "✗"
                        click.echo(
                            f"    Cost: {cost}, Timeout: {timeout_display}, "
                            f"System prompt: {system_prompt_indicator}"
                        )
            else:
                click.echo("Local Configuration Status:")
                click.echo(f"Directory: {local_config_dir.absolute()}")
                click.echo("Status: configured but no project config found")

        except Exception as e:
            click.echo("Local Configuration Status:")
            click.echo(f"Directory: {local_config_dir.absolute()}")
            click.echo(f"Error reading local config: {e}")
    else:
        click.echo("Local Configuration Status:")
        click.echo(f"Directory: {local_config_dir.absolute()}")
        click.echo("Status: missing")
        click.echo("Run 'llm-orc config init' to create it")


@cli.group()
def auth() -> None:
    """Authentication management commands."""
    pass


@auth.command("add")
@click.argument("provider")
@click.option("--api-key", help="API key for the provider")
@click.option("--client-id", help="OAuth client ID")
@click.option("--client-secret", help="OAuth client secret")
def auth_add(provider: str, api_key: str, client_id: str, client_secret: str) -> None:
    """Add authentication for a provider (API key or OAuth)."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    # Special handling for claude-cli provider
    if provider.lower() == "claude-cli":
        try:
            _handle_claude_cli_auth(storage)
            return
        except Exception as e:
            raise click.ClickException(
                f"Failed to set up Claude CLI authentication: {str(e)}"
            ) from e

    # Special handling for anthropic-claude-pro-max OAuth
    if provider.lower() == "anthropic-claude-pro-max":
        try:
            _handle_claude_pro_max_oauth(auth_manager, storage)
            return
        except Exception as e:
            raise click.ClickException(
                f"Failed to set up Claude Pro/Max OAuth authentication: {str(e)}"
            ) from e

    # Special interactive flow for Anthropic
    is_anthropic_interactive = (
        provider.lower() == "anthropic"
        and not api_key
        and not (client_id and client_secret)
    )
    if is_anthropic_interactive:
        try:
            _handle_anthropic_interactive_auth(auth_manager, storage)
            return
        except Exception as e:
            raise click.ClickException(
                f"Failed to set up Anthropic authentication: {str(e)}"
            ) from e

    # Validate input for non-interactive flow
    if api_key and (client_id or client_secret):
        raise click.ClickException("Cannot use both API key and OAuth credentials")

    if not api_key and not (client_id and client_secret):
        raise click.ClickException(
            "Must provide either --api-key or both --client-id and --client-secret"
        )

    try:
        if api_key:
            # API key authentication
            storage.store_api_key(provider, api_key)
            click.echo(f"API key for {provider} added successfully")
        else:
            # OAuth authentication
            if auth_manager.authenticate_oauth(provider, client_id, client_secret):
                click.echo(
                    f"OAuth authentication for {provider} completed successfully"
                )
            else:
                raise click.ClickException(
                    f"OAuth authentication for {provider} failed"
                )
    except Exception as e:
        raise click.ClickException(f"Failed to add authentication: {str(e)}") from e


def _handle_anthropic_interactive_auth(
    auth_manager: AuthenticationManager, storage: CredentialStorage
) -> None:
    """Handle interactive Anthropic authentication setup."""
    click.echo("How would you like to authenticate with Anthropic?")
    click.echo("1. API Key (for Anthropic API access)")
    click.echo("2. Claude Pro/Max OAuth (for your existing Claude subscription)")
    click.echo("3. Both (set up multiple authentication methods)")
    click.echo()

    choice = click.prompt("Choice", type=click.Choice(["1", "2", "3"]), default="1")

    if choice == "1":
        # API Key only
        api_key = click.prompt("Anthropic API key", hide_input=True)
        storage.store_api_key("anthropic-api", api_key)
        click.echo("✅ API key configured as 'anthropic-api'")

    elif choice == "2":
        # OAuth only
        _setup_anthropic_oauth(auth_manager, "anthropic-claude-pro-max")
        click.echo("✅ OAuth configured as 'anthropic-claude-pro-max'")

    elif choice == "3":
        # Both methods
        click.echo()
        click.echo("🔑 Setting up API key access...")
        api_key = click.prompt("Anthropic API key", hide_input=True)
        storage.store_api_key("anthropic-api", api_key)
        click.echo("✅ API key configured as 'anthropic-api'")

        click.echo()
        click.echo("🔧 Setting up Claude Pro/Max OAuth...")
        _setup_anthropic_oauth(auth_manager, "anthropic-claude-pro-max")
        click.echo("✅ OAuth configured as 'anthropic-claude-pro-max'")


def _setup_anthropic_oauth(
    auth_manager: AuthenticationManager, provider_key: str
) -> None:
    """Set up Anthropic OAuth authentication."""
    from llm_orc.authentication import AnthropicOAuthFlow

    oauth_flow = AnthropicOAuthFlow.create_with_guidance()

    if not auth_manager.authenticate_oauth(
        provider_key, oauth_flow.client_id, oauth_flow.client_secret
    ):
        raise click.ClickException("OAuth authentication failed")


def _handle_claude_pro_max_oauth(
    auth_manager: AuthenticationManager, storage: CredentialStorage
) -> None:
    """Handle Claude Pro/Max OAuth authentication setup using hardcoded client ID."""
    import base64
    import hashlib
    import secrets
    import webbrowser
    from urllib.parse import urlencode

    click.echo("🔧 Setting up Claude Pro/Max OAuth Authentication")
    click.echo("=" * 55)
    click.echo("This will authenticate with your existing Claude Pro/Max subscription.")
    click.echo()

    # Hardcoded OAuth parameters from issue-32
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    redirect_uri = "https://console.anthropic.com/oauth/code/callback"
    scope = "org:create_api_key user:profile user:inference"

    # Generate PKCE parameters
    code_verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    )
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )

    # Build authorization URL
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": code_verifier,
    }

    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    click.echo("📋 OAuth Flow Details:")
    click.echo(f"   • Client ID: {client_id}")
    click.echo(f"   • Scope: {scope}")
    click.echo(f"   • Redirect URI: {redirect_uri}")
    click.echo()

    # Open browser and guide user
    click.echo("🌐 Opening authorization URL in your browser...")
    click.echo(f"   {auth_url}")
    click.echo()

    if click.confirm("Open browser automatically?", default=True):
        webbrowser.open(auth_url)
        click.echo("✅ Browser opened")
    else:
        click.echo("Please manually navigate to the URL above")

    click.echo()
    click.echo("📋 Instructions:")
    click.echo("1. Sign in to your Claude Pro/Max account")
    click.echo("2. Authorize the application")
    click.echo("3. You'll be redirected to a callback page")
    click.echo("4. Copy the full URL from the address bar")
    click.echo("5. Extract the authorization code from the URL")
    click.echo()

    # Get authorization code from user
    auth_code = click.prompt(
        "Authorization code (format: code#state)", type=str
    ).strip()

    # Parse auth code
    splits = auth_code.split("#")
    if len(splits) != 2:
        raise click.ClickException(
            f"Invalid authorization code format. Expected 'code#state', "
            f"got: {auth_code}"
        )

    code_part = splits[0]
    state_part = splits[1]

    # Verify state matches
    if state_part != code_verifier:
        click.echo("⚠️  Warning: State mismatch - proceeding anyway")

    # Exchange code for tokens
    click.echo("🔄 Exchanging authorization code for access tokens...")

    import requests

    token_url = "https://console.anthropic.com/v1/oauth/token"
    data = {
        "code": code_part,
        "state": state_part,
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(token_url, json=data, headers=headers, timeout=30)

        if response.status_code == 200:
            tokens = response.json()

            # Store OAuth tokens
            storage.store_oauth_token(
                "anthropic-claude-pro-max",
                tokens["access_token"],
                tokens.get("refresh_token"),
                int(time.time()) + tokens.get("expires_in", 3600),
                client_id,
            )

            click.echo("✅ OAuth authentication successful!")
            click.echo("✅ Tokens stored as 'anthropic-claude-pro-max'")

        else:
            raise click.ClickException(
                f"Token exchange failed. Status: {response.status_code}, "
                f"Response: {response.text}"
            )

    except requests.exceptions.RequestException as e:
        raise click.ClickException(
            f"Network error during token exchange: {str(e)}"
        ) from e


def _handle_claude_cli_auth(storage: CredentialStorage) -> None:
    """Handle Claude CLI authentication setup."""
    import shutil

    # Check if claude command is available
    claude_path = shutil.which("claude")
    if not claude_path:
        raise click.ClickException(
            "Claude CLI not found. Please install the Claude CLI from: "
            "https://docs.anthropic.com/en/docs/claude-code"
        )

    # Store claude-cli as a special auth method
    # We'll store the path to the claude executable
    storage.store_api_key("claude-cli", claude_path)

    click.echo("✅ Claude CLI authentication configured")
    click.echo(f"Using local claude command at: {claude_path}")


@auth.command("list")
@click.option(
    "--interactive", "-i", is_flag=True, help="Show interactive menu with actions"
)
def auth_list(interactive: bool) -> None:
    """List configured authentication providers."""
    from .menu_system import (
        AuthMenus,
        show_error,
        show_info,
        show_success,
        show_working,
    )

    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        providers = storage.list_providers()

        if not interactive:
            # Simple list view (original behavior)
            if not providers:
                click.echo("No authentication providers configured")
            else:
                click.echo("Configured providers:")
                for provider in providers:
                    auth_method = storage.get_auth_method(provider)
                    if auth_method == "oauth":
                        click.echo(f"  {provider}: OAuth")
                    else:
                        click.echo(f"  {provider}: API key")
            return

        # Interactive mode with action menu
        while True:
            action, selected_provider = AuthMenus.auth_list_actions(providers)

            if action == "quit":
                break
            elif action == "setup" or action == "add":
                # Run the setup wizard
                auth_setup()
                # Refresh provider list
                providers = storage.list_providers()
            elif action == "test" and selected_provider:
                show_working(f"Testing {selected_provider}...")
                try:
                    auth_method = storage.get_auth_method(selected_provider)
                    if not auth_method:
                        show_error(f"No authentication found for {selected_provider}")
                        continue

                    # Test the authentication
                    success = False
                    if auth_method == "api_key":
                        api_key = storage.get_api_key(selected_provider)
                        if api_key:
                            success = auth_manager.authenticate(
                                selected_provider, api_key
                            )
                    elif auth_method == "oauth":
                        oauth_token = storage.get_oauth_token(selected_provider)
                        if oauth_token:
                            # For OAuth, we'll consider it successful if we have a valid token  # noqa: E501
                            import time

                            if "expires_at" in oauth_token:
                                success = time.time() < oauth_token["expires_at"]
                            else:
                                success = True  # No expiration info, assume valid

                    if success:
                        show_success(
                            f"Authentication for {selected_provider} is working!"
                        )
                    else:
                        show_error(f"Authentication for {selected_provider} failed")
                except Exception as e:
                    show_error(f"Test failed: {str(e)}")
            elif action == "remove" and selected_provider:
                from .menu_system import confirm_action

                if confirm_action(f"Remove authentication for {selected_provider}?"):
                    storage.remove_provider(selected_provider)
                    show_success(f"Removed {selected_provider}")
                    providers = storage.list_providers()
            elif action == "details" and selected_provider:
                _show_provider_details(storage, selected_provider)
            elif action == "refresh" and selected_provider:
                show_working(f"Refreshing tokens for {selected_provider}...")
                try:
                    auth_method = storage.get_auth_method(selected_provider)
                    if auth_method == "oauth":
                        # For now, just re-authenticate with OAuth
                        show_info("Re-authentication required for OAuth token refresh")
                        # This would typically trigger a re-auth flow
                        show_success("Token refresh would be performed here")
                    else:
                        show_error("Token refresh only available for OAuth providers")
                except Exception as e:
                    show_error(f"Refresh failed: {str(e)}")

    except Exception as e:
        raise click.ClickException(f"Failed to list providers: {str(e)}") from e


def _show_provider_details(storage: "CredentialStorage", provider: str) -> None:
    """Show detailed information about a provider."""
    from .provider_registry import provider_registry

    click.echo(f"\n📋 Provider Details: {provider}")
    click.echo("=" * 40)

    # Get registry info
    provider_info = provider_registry.get_provider(provider)
    if provider_info:
        click.echo(f"Display Name: {provider_info.display_name}")
        click.echo(f"Description: {provider_info.description}")

        auth_methods = []
        if provider_info.supports_oauth:
            auth_methods.append("OAuth")
        if provider_info.supports_api_key:
            auth_methods.append("API Key")
        if not provider_info.requires_auth:
            auth_methods.append("No authentication required")
        click.echo(f"Supported Auth: {', '.join(auth_methods)}")

    # Get stored auth info
    auth_method = storage.get_auth_method(provider)
    if auth_method:
        click.echo(f"Configured Method: {auth_method.upper()}")

        if auth_method == "oauth":
            # Try to get OAuth details if available
            try:
                # This would need to be implemented in storage
                click.echo("OAuth Status: Configured")
            except Exception:
                pass
    else:
        click.echo("Status: Not configured")

    click.echo()


@auth.command("remove")
@click.argument("provider")
def auth_remove(provider: str) -> None:
    """Remove authentication for a provider."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)

    try:
        # Check if provider exists
        if provider not in storage.list_providers():
            raise click.ClickException(f"No authentication found for {provider}")

        storage.remove_provider(provider)
        click.echo(f"Authentication for {provider} removed")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to remove provider: {str(e)}") from e


@auth.command("setup")
def auth_setup() -> None:
    """Interactive setup wizard for authentication."""
    from .menu_system import (
        AuthMenus,
        confirm_action,
        show_error,
        show_success,
        show_working,
    )
    from .provider_registry import provider_registry

    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    click.echo("🚀 Welcome to LLM Orchestra setup!")
    click.echo("This wizard will help you configure authentication for LLM providers.")

    while True:
        # Use interactive menu for provider selection
        provider_key = AuthMenus.provider_selection()

        # Get provider info
        provider = provider_registry.get_provider(provider_key)

        if not provider:
            show_error(f"Provider '{provider_key}' not found in registry")
            continue

        if not provider.requires_auth:
            show_success(f"{provider.display_name} doesn't require authentication!")
            if not confirm_action("Add another provider?"):
                break
            continue

        # Get authentication method based on provider
        if provider_key == "anthropic-claude-pro-max":
            auth_method = "oauth"  # Claude Pro/Max only supports OAuth
        elif provider_key == "anthropic-api":
            auth_method = "api_key"  # Anthropic API only supports API key
        elif provider_key == "google-gemini":
            auth_method = "api_key"  # Google Gemini only supports API key
        else:
            # For other providers, use the menu system
            auth_method = AuthMenus.get_auth_method_for_provider(provider_key)

        # Handle authentication setup based on method
        try:
            if auth_method == "help":
                _show_auth_method_help()
                continue
            elif auth_method == "oauth" and provider_key == "anthropic-claude-pro-max":
                show_working("Setting up Claude Pro/Max OAuth...")
                _handle_claude_pro_max_oauth(auth_manager, storage)
                show_success("Claude Pro/Max OAuth configured!")
            elif auth_method == "api_key" and provider_key == "anthropic-api":
                api_key = click.prompt("Anthropic API key", hide_input=True)
                storage.store_api_key("anthropic-api", api_key)
                show_success("Anthropic API key configured!")
            elif auth_method == "api_key" and provider_key == "google-gemini":
                api_key = click.prompt("Google Gemini API key", hide_input=True)
                storage.store_api_key("google-gemini", api_key)
                show_success("Google Gemini API key configured!")
            elif auth_method == "api_key" or auth_method == "api-key":
                # Generic API key setup for other providers
                api_key = click.prompt(
                    f"{provider.display_name} API key", hide_input=True
                )
                storage.store_api_key(provider_key, api_key)
                show_success(f"{provider.display_name} API key configured!")
            elif auth_method == "oauth":
                # Generic OAuth setup for other providers
                client_id = click.prompt("OAuth client ID")
                client_secret = click.prompt("OAuth client secret", hide_input=True)

                if auth_manager.authenticate_oauth(
                    provider_key, client_id, client_secret
                ):
                    show_success(f"{provider.display_name} OAuth configured!")
                else:
                    show_error(
                        f"OAuth authentication for {provider.display_name} failed"
                    )
            else:
                show_error(f"Unknown authentication method: {auth_method}")

        except Exception as e:
            show_error(f"Failed to configure {provider.display_name}: {str(e)}")

        if not confirm_action("Add another provider?"):
            break

    click.echo()
    show_success(
        "Setup complete! Use 'llm-orc auth list' to see your configured providers."
    )


def _show_auth_method_help() -> None:
    """Show help for choosing authentication methods."""
    click.echo("\n📚 Authentication Method Guide")
    click.echo("=" * 30)
    click.echo()
    click.echo("🔐 Claude Pro/Max OAuth:")
    click.echo("   • Best if you have a Claude Pro or Claude Max subscription")
    click.echo("   • Uses your existing subscription (no extra API costs)")
    click.echo("   • Automatic token management and refresh")
    click.echo("   • Most convenient for regular Claude users")
    click.echo()
    click.echo("🔑 API Key:")
    click.echo("   • Best for programmatic access or if you don't have Claude Pro/Max")
    click.echo("   • Requires separate API subscription (~$20/month minimum)")
    click.echo("   • Direct API access with manual key management")
    click.echo("   • Good for production applications")
    click.echo()
    click.echo("💡 Recommendation:")
    click.echo("   Choose Claude Pro/Max if you already have a subscription.")
    click.echo(
        "   Choose API Key if you need programmatic access or don't have "
        "Claude Pro/Max."
    )
    click.echo()


@auth.command("logout")
@click.argument("provider", required=False)
@click.option(
    "--all", "logout_all", is_flag=True, help="Logout from all OAuth providers"
)
def auth_logout(provider: str | None, logout_all: bool) -> None:
    """Logout from OAuth providers (revokes tokens and removes credentials)."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        if logout_all:
            # Logout from all OAuth providers
            results = auth_manager.logout_all_oauth_providers()

            if not results:
                click.echo("No OAuth providers found to logout")
                return

            success_count = sum(1 for success in results.values() if success)

            click.echo(f"Logged out from {success_count} OAuth providers:")
            for provider_name, success in results.items():
                status = "✅" if success else "❌"
                click.echo(f"  {provider_name}: {status}")

        elif provider:
            # Logout from specific provider
            if auth_manager.logout_oauth_provider(provider):
                click.echo(f"✅ Logged out from {provider}")
            else:
                raise click.ClickException(
                    f"Failed to logout from {provider}. "
                    f"Provider may not exist or is not an OAuth provider."
                )
        else:
            raise click.ClickException("Must specify a provider name or use --all flag")

    except Exception as e:
        raise click.ClickException(f"Failed to logout: {str(e)}") from e


@cli.command()
@click.argument("ensemble_name")
@click.option("--port", default=3000, help="Port to serve MCP server on")
def serve(ensemble_name: str, port: int) -> None:
    """Serve an ensemble as an MCP server."""
    runner = MCPServerRunner(ensemble_name, port)
    runner.run()


if __name__ == "__main__":
    cli()
