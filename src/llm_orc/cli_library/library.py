"""Library commands for browsing and copying ensembles."""

from pathlib import Path
from typing import Any

import click
import requests
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager


def get_library_categories() -> list[str]:
    """Get list of available library categories."""
    categories = [
        "code-analysis",
        "idea-exploration",
        "research-analysis",
        "decision-support",
        "problem-decomposition",
        "learning-facilitation",
    ]
    return categories


def get_library_categories_with_descriptions() -> list[tuple[str, str]]:
    """Get categories with their descriptions."""
    categories_with_descriptions = [
        ("code-analysis", "Code review and security analysis"),
        ("idea-exploration", "Concept mapping and perspective taking"),
        ("research-analysis", "Literature review and synthesis"),
        ("decision-support", "Strategic decisions and risk assessment"),
        ("problem-decomposition", "System breakdown and root cause analysis"),
        ("learning-facilitation", "Educational exploration and knowledge building"),
    ]
    return categories_with_descriptions


def get_category_ensembles(category: str) -> list[dict[str, Any]]:
    """Get ensembles for a specific category by fetching from GitHub API."""
    base_api_url = (
        "https://api.github.com/repos/mrilikecoding/llm-orchestra-library/contents"
    )

    try:
        # Fetch directory contents from GitHub API
        response = requests.get(f"{base_api_url}/{category}", timeout=10)
        response.raise_for_status()

        files = response.json()
        ensembles = []

        for file_info in files:
            # Only process .yaml files (skip README.md and other files)
            if file_info.get("type") == "file" and file_info.get("name", "").endswith(
                ".yaml"
            ):
                ensemble_name = file_info["name"].replace(".yaml", "")
                ensemble_path = f"{category}/{file_info['name']}"

                # Try to fetch ensemble content to get description
                try:
                    content = fetch_ensemble_content(ensemble_path)
                    ensemble_data = yaml.safe_load(content)
                    description = ensemble_data.get(
                        "description", "No description available"
                    )
                except (requests.RequestException, yaml.YAMLError):
                    description = "No description available"

                ensembles.append(
                    {
                        "name": ensemble_name,
                        "description": description,
                        "path": ensemble_path,
                    }
                )

        return ensembles

    except requests.RequestException:
        # Fallback to empty list if GitHub API is unavailable
        return []


def fetch_ensemble_content(ensemble_path: str) -> str:
    """Fetch ensemble content from GitHub repository."""
    base_url = (
        "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main"
    )

    # Handle .yaml extension
    if not ensemble_path.endswith(".yaml"):
        ensemble_path += ".yaml"

    url = f"{base_url}/{ensemble_path}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise FileNotFoundError(f"Ensemble not found: {ensemble_path}") from e


def ensure_local_ensembles_dir() -> str:
    """Ensure local ensembles directory exists and return path."""
    ensembles_dir = Path(".llm-orc/ensembles")
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    return str(ensembles_dir)


def ensure_global_ensembles_dir() -> str:
    """Ensure global ensembles directory exists and return path."""
    config_manager = ConfigurationManager()
    global_config_dir = config_manager.global_config_dir
    ensembles_dir = Path(global_config_dir) / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    return str(ensembles_dir)


def ensemble_exists(ensemble_name: str, is_global: bool = False) -> bool:
    """Check if ensemble already exists."""
    if is_global:
        ensembles_dir = Path(ensure_global_ensembles_dir())
    else:
        ensembles_dir = Path(ensure_local_ensembles_dir())

    ensemble_file = ensembles_dir / f"{ensemble_name}.yaml"
    return ensemble_file.exists()


def browse_library(category: str | None = None) -> None:
    """Browse library ensembles."""
    if category is None:
        # Show all categories
        categories = get_library_categories()
        click.echo("Available ensemble categories:")
        click.echo()
        for cat in categories:
            click.echo(f"  {cat}")
        click.echo()
        click.echo(
            "Use 'llm-orc library browse <category>' to see ensembles in a category"
        )
    else:
        # Show ensembles in specific category
        ensembles = get_category_ensembles(category)
        if not ensembles:
            click.echo(f"No ensembles found in category: {category}")
            return

        click.echo(f"Ensembles in {category}:")
        click.echo()
        for ensemble in ensembles:
            click.echo(f"  {ensemble['name']}")
            click.echo(f"    {ensemble['description']}")
            click.echo()


def copy_ensemble(ensemble_path: str, is_global: bool = False) -> None:
    """Copy ensemble from library to local or global config."""
    try:
        # Fetch ensemble content
        content = fetch_ensemble_content(ensemble_path)

        # Parse ensemble name from content
        ensemble_data = yaml.safe_load(content)
        ensemble_name = ensemble_data.get("name", Path(ensemble_path).stem)

        # Check if ensemble already exists
        if ensemble_exists(ensemble_name, is_global):
            if not click.confirm(
                f"Ensemble '{ensemble_name}' already exists. Overwrite?"
            ):
                click.echo("Copy cancelled.")
                return

        # Determine target directory
        if is_global:
            target_dir = ensure_global_ensembles_dir()
            location = "global"
        else:
            target_dir = ensure_local_ensembles_dir()
            location = "local"

        # Write ensemble file
        target_file = Path(target_dir) / f"{ensemble_name}.yaml"
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)

        click.echo(f"Copied {ensemble_name} to {location} config ({target_file})")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.ClickException(str(e)) from e


def _analyze_ensemble_metadata(
    agents: list[dict[str, Any]],
) -> tuple[set[str], list[tuple[str, str]], set[str]]:
    """Analyze ensemble metadata from agents list."""
    model_profiles = set()
    dependencies = []
    output_formats = set()

    for agent in agents:
        # Collect model profiles
        profile = agent.get("model_profile", "default")
        model_profiles.add(profile)

        # Collect dependencies
        if "depends_on" in agent:
            deps = agent["depends_on"]
            if isinstance(deps, list):
                dependencies.extend([(agent["name"], dep) for dep in deps])
            else:
                dependencies.append((agent["name"], deps))

        # Collect output formats
        if "output_format" in agent:
            output_formats.add(agent["output_format"])

    return model_profiles, dependencies, output_formats


def _display_agent_details(agents: list[dict[str, Any]]) -> None:
    """Display detailed information about each agent."""
    click.echo("👤 Agent Details:")
    for agent in agents:
        agent_name = agent.get("name", "unnamed")
        profile = agent.get("model_profile", "default")
        click.echo(f"  • {agent_name} ({profile})")

        # Show dependencies
        if "depends_on" in agent:
            deps = agent["depends_on"]
            if isinstance(deps, list):
                deps_str = ", ".join(deps)
            else:
                deps_str = str(deps)
            click.echo(f"    ↳ depends on: {deps_str}")

        # Show special features
        if "output_format" in agent:
            click.echo(f"    ↳ output format: {agent['output_format']}")


def _display_execution_flow(
    agents: list[dict[str, Any]], dependencies: list[tuple[str, str]]
) -> None:
    """Display execution flow information."""
    if not dependencies:
        return

    click.echo()
    click.echo("🔄 Execution Flow:")

    # Simple dependency display
    independent_agents = []
    dependent_agents = []

    for agent in agents:
        if "depends_on" not in agent:
            independent_agents.append(agent["name"])
        else:
            dependent_agents.append(agent["name"])

    if independent_agents:
        click.echo(f"  1. Parallel: {', '.join(independent_agents)}")
    if dependent_agents:
        click.echo(f"  2. Sequential: {', '.join(dependent_agents)}")


def show_ensemble_info(ensemble_path: str) -> None:
    """Show detailed information about an ensemble."""
    try:
        # Fetch ensemble content
        content = fetch_ensemble_content(ensemble_path)
        ensemble_data = yaml.safe_load(content)

        # Extract basic info
        name = ensemble_data.get("name", "Unknown")
        description = ensemble_data.get("description", "No description available")
        agents = ensemble_data.get("agents", [])

        # Display basic info
        click.echo(f"📋 Ensemble: {name}")
        click.echo(f"📝 Description: {description}")
        click.echo(f"👥 Agents: {len(agents)}")
        click.echo()

        # Analyze metadata
        model_profiles, dependencies, output_formats = _analyze_ensemble_metadata(
            agents
        )

        # Display model profiles
        click.echo("🤖 Model Profiles:")
        for profile in sorted(model_profiles):
            click.echo(f"  • {profile}")
        click.echo()

        # Display agent details
        _display_agent_details(agents)

        # Display execution flow
        _display_execution_flow(agents, dependencies)

        click.echo()

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.ClickException(str(e)) from e
    except yaml.YAMLError as e:
        click.echo(f"Error: Invalid YAML in ensemble: {e}", err=True)
        raise click.ClickException(f"Invalid YAML: {e}") from e


def list_categories() -> None:
    """List all available categories with descriptions."""
    categories = get_library_categories_with_descriptions()
    click.echo("Available ensemble categories:")
    click.echo()
    for category, description in categories:
        click.echo(f"  {category:<20} {description}")
    click.echo()
