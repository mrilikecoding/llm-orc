"""Library commands for browsing and copying ensembles."""

import os
from pathlib import Path
from typing import Any

import click
import requests
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager


def _get_library_source_config() -> tuple[str, str]:
    """Get library source configuration from environment or defaults.

    Returns:
        Tuple of (source_type, source_path) where source_type is 'local' or 'remote'
    """
    library_source = os.environ.get("LLM_ORC_LIBRARY_SOURCE", "remote")

    if library_source == "local":
        # Use local submodule - find it relative to this file
        # Go up from src/llm_orc/cli_library/ to project root
        current_dir = Path(__file__).parent.parent.parent.parent
        local_path = current_dir / "llm-orchestra-library"
        if not local_path.exists():
            raise FileNotFoundError(
                f"Local llm-orchestra-library not found at {local_path}. "
                "Please ensure submodule is initialized or set "
                "LLM_ORC_LIBRARY_SOURCE=remote"
            )
        return "local", str(local_path)
    else:
        # Use remote GitHub
        return (
            "remote",
            "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main",
        )


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
    """Get ensembles for a specific category from local or remote source."""
    source_type, source_path = _get_library_source_config()

    if source_type == "local":
        return _get_local_category_ensembles(category, Path(source_path))
    else:
        return _get_remote_category_ensembles(category)


def _get_local_category_ensembles(
    category: str, library_path: Path
) -> list[dict[str, Any]]:
    """Get ensembles from local library directory."""
    ensembles_dir = library_path / "ensembles" / category
    if not ensembles_dir.exists():
        return []

    ensembles = []
    for yaml_file in ensembles_dir.glob("*.yaml"):
        ensemble_name = yaml_file.stem
        ensemble_path = f"{category}/{yaml_file.name}"

        # Try to read ensemble content to get description
        try:
            with open(yaml_file, encoding="utf-8") as f:
                ensemble_data = yaml.safe_load(f)
                description = ensemble_data.get(
                    "description", "No description available"
                )
        except (OSError, yaml.YAMLError):
            description = "No description available"

        ensembles.append(
            {
                "name": ensemble_name,
                "description": description,
                "path": ensemble_path,
            }
        )

    return ensembles


def _get_remote_category_ensembles(category: str) -> list[dict[str, Any]]:
    """Get ensembles from remote GitHub repository."""
    base_api_url = (
        "https://api.github.com/repos/mrilikecoding/llm-orchestra-library/contents"
    )

    try:
        # Fetch directory contents from GitHub API
        response = requests.get(f"{base_api_url}/ensembles/{category}", timeout=10)
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
    """Fetch ensemble content from local or remote source."""
    source_type, source_path = _get_library_source_config()

    if source_type == "local":
        return _fetch_local_ensemble_content(ensemble_path, Path(source_path))
    else:
        return _fetch_remote_ensemble_content(ensemble_path)


def _fetch_local_ensemble_content(ensemble_path: str, library_path: Path) -> str:
    """Fetch ensemble content from local library directory."""
    # Handle .yaml extension
    if not ensemble_path.endswith(".yaml"):
        ensemble_path += ".yaml"

    full_path = library_path / "ensembles" / ensemble_path

    try:
        with open(full_path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise FileNotFoundError(f"Ensemble not found: {ensemble_path}") from e


def _fetch_remote_ensemble_content(ensemble_path: str) -> str:
    """Fetch ensemble content from remote GitHub repository."""
    base_url = (
        "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main"
    )

    # Handle .yaml extension
    if not ensemble_path.endswith(".yaml"):
        ensemble_path += ".yaml"

    url = f"{base_url}/ensembles/{ensemble_path}"

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


def get_template_content(template_name: str) -> str:
    """Fetch template content from local or remote source."""
    source_type, source_path = _get_library_source_config()

    if source_type == "local":
        return _get_local_template_content(template_name, Path(source_path))
    else:
        return _get_remote_template_content(template_name)


def _get_local_template_content(template_name: str, library_path: Path) -> str:
    """Fetch template content from local library directory."""
    # Ensure template has .yaml extension if not already present
    if not template_name.endswith(".yaml"):
        template_name += ".yaml"

    template_path = library_path / "templates" / template_name

    try:
        with open(template_path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise FileNotFoundError(f"Template not found: {template_name}") from e


def _get_remote_template_content(template_name: str) -> str:
    """Fetch template content from remote GitHub repository."""
    base_url = (
        "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main"
    )

    # Ensure template has .yaml extension if not already present
    if not template_name.endswith(".yaml"):
        template_name += ".yaml"

    url = f"{base_url}/templates/{template_name}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise FileNotFoundError(f"Template not found: {template_name}") from e


def copy_primitive_scripts(target_scripts_dir: Path) -> None:
    """Copy primitive scripts from local or remote library source."""
    source_type, source_path = _get_library_source_config()

    try:
        if source_type == "local":
            _copy_local_primitive_scripts(Path(source_path), target_scripts_dir)
        else:
            _copy_remote_primitive_scripts(target_scripts_dir)
    except (OSError, requests.RequestException):
        # If we can't access source, fail silently
        # This allows offline usage without breaking init
        pass


def _copy_local_primitive_scripts(library_path: Path, target_scripts_dir: Path) -> None:
    """Copy primitive scripts from local library directory."""
    source_primitives_dir = library_path / "scripts" / "primitives"
    if not source_primitives_dir.exists():
        return

    # Copy entire primitives directory structure
    target_primitives_dir = target_scripts_dir / "primitives"
    if target_primitives_dir.exists():
        import shutil

        shutil.rmtree(target_primitives_dir)

    import shutil

    shutil.copytree(source_primitives_dir, target_primitives_dir)

    # Make all Python scripts executable
    for script_file in target_primitives_dir.rglob("*.py"):
        script_file.chmod(0o755)


def _copy_remote_primitive_scripts(target_scripts_dir: Path) -> None:
    """Copy primitive scripts from remote GitHub repository."""
    base_api_url = (
        "https://api.github.com/repos/mrilikecoding/llm-orchestra-library/contents"
    )
    base_raw_url = (
        "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main"
    )

    categories = _fetch_primitive_categories(base_api_url)
    _process_script_categories(
        categories, base_api_url, base_raw_url, target_scripts_dir
    )


def _fetch_primitive_categories(base_api_url: str) -> list[dict[str, Any]]:
    """Fetch primitive script categories from GitHub API."""
    primitives_response = requests.get(f"{base_api_url}/scripts/primitives", timeout=10)
    primitives_response.raise_for_status()
    return primitives_response.json()  # type: ignore[no-any-return]


def _process_script_categories(
    categories: list[dict[str, Any]],
    base_api_url: str,
    base_raw_url: str,
    target_scripts_dir: Path,
) -> None:
    """Process each script category and download scripts."""
    for category_info in categories:
        if category_info["type"] == "dir":
            _download_category_scripts(
                category_info["name"], base_api_url, base_raw_url, target_scripts_dir
            )


def _download_category_scripts(
    category_name: str, base_api_url: str, base_raw_url: str, target_scripts_dir: Path
) -> None:
    """Download all Python scripts in a category."""
    try:
        category_response = requests.get(
            f"{base_api_url}/scripts/primitives/{category_name}", timeout=10
        )
        category_response.raise_for_status()

        files = category_response.json()
        category_dir = target_scripts_dir / "primitives" / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        for file_info in files:
            if file_info["name"].endswith(".py"):
                _download_script_file(
                    file_info["name"], category_name, base_raw_url, category_dir
                )

    except requests.RequestException:
        # If a category fails, continue with others
        pass


def _download_script_file(
    filename: str, category_name: str, base_raw_url: str, category_dir: Path
) -> None:
    """Download and save a single script file."""
    script_response = requests.get(
        f"{base_raw_url}/scripts/primitives/{category_name}/{filename}",
        timeout=10,
    )
    script_response.raise_for_status()

    script_file = category_dir / filename
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_response.text)

    # Make script executable
    script_file.chmod(0o755)


def copy_profile_templates(target_profiles_dir: Path) -> None:
    """Copy profile templates from local or remote library source."""
    source_type, source_path = _get_library_source_config()

    try:
        target_profiles_dir.mkdir(parents=True, exist_ok=True)

        if source_type == "local":
            _copy_local_profile_templates(Path(source_path), target_profiles_dir)
        else:
            _copy_remote_profile_templates(target_profiles_dir)
    except (OSError, requests.RequestException):
        # If we can't access source, fail silently
        pass


def _copy_local_profile_templates(
    library_path: Path, target_profiles_dir: Path
) -> None:
    """Copy profile templates from local library directory."""
    source_profiles_dir = library_path / "profiles"
    if not source_profiles_dir.exists():
        return

    # Copy each YAML profile template
    for yaml_file in source_profiles_dir.glob("*.yaml"):
        target_file = target_profiles_dir / yaml_file.name
        # Skip if file already exists (idempotent)
        if not target_file.exists():
            import shutil

            shutil.copy2(yaml_file, target_file)


def _copy_remote_profile_templates(target_profiles_dir: Path) -> None:
    """Copy profile templates from remote GitHub repository."""
    base_api_url = (
        "https://api.github.com/repos/mrilikecoding/llm-orchestra-library/contents"
    )
    base_raw_url = (
        "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main"
    )

    # Get list of profile templates from GitHub API
    profiles_response = requests.get(f"{base_api_url}/profiles", timeout=10)
    profiles_response.raise_for_status()

    files = profiles_response.json()

    # Download each YAML profile template
    for file_info in files:
        if file_info["name"].endswith(".yaml"):
            _download_profile_template(
                file_info["name"], base_raw_url, target_profiles_dir
            )


def _download_profile_template(
    filename: str, base_raw_url: str, target_profiles_dir: Path
) -> None:
    """Download and save a single profile template file."""
    profile_file = target_profiles_dir / filename

    # Skip if file already exists (idempotent)
    if profile_file.exists():
        return

    profile_response = requests.get(f"{base_raw_url}/profiles/{filename}", timeout=10)
    profile_response.raise_for_status()

    with open(profile_file, "w", encoding="utf-8") as f:
        f.write(profile_response.text)


def list_categories() -> None:
    """List all available categories with descriptions."""
    categories = get_library_categories_with_descriptions()
    click.echo("Available ensemble categories:")
    click.echo()
    for category, description in categories:
        click.echo(f"  {category:<20} {description}")
    click.echo()
