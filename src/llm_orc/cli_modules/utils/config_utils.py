"""Configuration utility functions for CLI operations."""

from pathlib import Path
from typing import Any

import click
import yaml

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager


def safe_load_yaml(file_path: Path) -> dict[str, Any]:
    """Safely load YAML file with error handling.

    Args:
        file_path: Path to YAML file

    Returns:
        Loaded YAML data as dictionary, empty dict if file doesn't exist

    Raises:
        click.ClickException: If YAML parsing fails
    """
    if not file_path.exists():
        return {}

    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise click.ClickException(f"Failed to parse YAML file {file_path}: {e}") from e
    except OSError as e:
        raise click.ClickException(f"Failed to read file {file_path}: {e}") from e


def safe_write_yaml(file_path: Path, data: dict[str, Any]) -> None:
    """Safely write YAML file with error handling.

    Args:
        file_path: Path to write YAML file
        data: Data to write as YAML

    Raises:
        click.ClickException: If YAML writing fails
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except yaml.YAMLError as e:
        raise click.ClickException(f"Failed to write YAML file {file_path}: {e}") from e
    except OSError as e:
        raise click.ClickException(f"Failed to write file {file_path}: {e}") from e


def backup_config_file(file_path: Path) -> Path | None:
    """Create a backup of a configuration file.

    Args:
        file_path: Path to file to backup

    Returns:
        Path to backup file if created, None if original doesn't exist

    Raises:
        click.ClickException: If backup creation fails
    """
    if not file_path.exists():
        return None

    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")

    try:
        # Read original and write backup
        with open(file_path, "rb") as original:
            with open(backup_path, "wb") as backup:
                backup.write(original.read())

        return backup_path
    except OSError as e:
        raise click.ClickException(
            f"Failed to create backup of {file_path}: {e}"
        ) from e


def ensure_config_directory(config_dir: Path) -> None:
    """Ensure configuration directory exists.

    Args:
        config_dir: Path to configuration directory

    Raises:
        click.ClickException: If directory creation fails
    """
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise click.ClickException(
            f"Failed to create config directory {config_dir}: {e}"
        ) from e


def remove_config_file(file_path: Path, description: str = "file") -> None:
    """Remove a configuration file with error handling.

    Args:
        file_path: Path to file to remove
        description: Description of file for error messages

    Raises:
        click.ClickException: If file removal fails
    """
    if not file_path.exists():
        return

    try:
        file_path.unlink()
    except OSError as e:
        raise click.ClickException(
            f"Failed to remove {description} {file_path}: {e}"
        ) from e


def get_available_providers(config_manager: ConfigurationManager) -> set[str]:
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


def _process_ensemble_file(ensemble_file: Path) -> dict[str, Any]:
    """Process a single ensemble YAML file and return its data.

    Args:
        ensemble_file: Path to the ensemble YAML file

    Returns:
        Dictionary containing the ensemble configuration data

    Raises:
        Exception: If YAML parsing fails or file cannot be read
    """
    with open(ensemble_file) as f:
        ensemble_data = yaml.safe_load(f) or {}
    return ensemble_data


def _check_agent_requirements(
    agents: list[dict[str, Any]], config_manager: ConfigurationManager
) -> tuple[set[str], list[str]]:
    """Check requirements for all agents in an ensemble.

    Args:
        agents: List of agent configurations
        config_manager: Configuration manager for resolving profiles

    Returns:
        Tuple of (required_providers_set, missing_profiles_list)
    """
    required_providers: set[str] = set()
    missing_profiles: list[str] = []

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

    return required_providers, missing_profiles


def _check_coordinator_requirements(
    coordinator: dict[str, Any], config_manager: ConfigurationManager
) -> tuple[set[str], list[str]]:
    """Check requirements for ensemble coordinator.

    Args:
        coordinator: Coordinator configuration dictionary
        config_manager: Configuration manager for resolving profiles

    Returns:
        Tuple of (required_providers_set, missing_profiles_list)
    """
    required_providers: set[str] = set()
    missing_profiles: list[str] = []

    if "model_profile" in coordinator:
        profile_name = coordinator["model_profile"]
        try:
            _, provider = config_manager.resolve_model_profile(profile_name)
            required_providers.add(provider)
        except (ValueError, KeyError):
            missing_profiles.append(profile_name)
    elif "provider" in coordinator:
        required_providers.add(coordinator["provider"])

    return required_providers, missing_profiles


def _determine_ensemble_availability(
    agent_providers: set[str],
    coord_providers: set[str],
    agent_missing: list[str],
    coord_missing: list[str],
    available_providers: set[str],
) -> tuple[bool, list[str], list[str]]:
    """Determine ensemble availability based on provider and profile status.

    Args:
        agent_providers: Set of providers required by agents
        coord_providers: Set of providers required by coordinator
        agent_missing: List of missing agent profiles
        coord_missing: List of missing coordinator profiles
        available_providers: Set of available providers

    Returns:
        Tuple of (is_available, missing_providers, missing_profiles)
    """
    # Combine results
    required_providers = agent_providers.union(coord_providers)
    missing_profiles = agent_missing + coord_missing

    # Determine availability
    missing_providers_set = required_providers - available_providers
    missing_providers = list(missing_providers_set)
    is_available = not missing_providers and not missing_profiles

    return is_available, missing_providers, missing_profiles


def _display_ensemble_status(
    ensemble_name: str,
    is_available: bool,
    missing_providers: list[str],
    missing_profiles: list[str],
) -> None:
    """Display ensemble status with details for unavailable ensembles.

    Args:
        ensemble_name: Name of the ensemble
        is_available: Whether the ensemble is available
        missing_providers: List of missing providers
        missing_profiles: List of missing profiles
    """
    status_symbol = "🟢" if is_available else "🟥"
    click.echo(f"  {status_symbol} {ensemble_name}")

    # Show details for unavailable ensembles
    if not is_available:
        if missing_profiles:
            click.echo(f"    Missing profiles: {', '.join(missing_profiles)}")
        if missing_providers:
            click.echo(f"    Missing providers: {', '.join(missing_providers)}")


def check_ensemble_availability(
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

    for ensemble_file in sorted(ensemble_files):
        try:
            # Process ensemble file
            ensemble_data = _process_ensemble_file(ensemble_file)
            ensemble_name = ensemble_data.get("name", ensemble_file.stem)
            agents = ensemble_data.get("agents", [])
            coordinator = ensemble_data.get("coordinator", {})

            # Check requirements using helper methods
            agent_providers, agent_missing = _check_agent_requirements(
                agents, config_manager
            )
            coord_providers, coord_missing = _check_coordinator_requirements(
                coordinator, config_manager
            )

            # Determine availability using helper method
            is_available, missing_providers, missing_profiles = (
                _determine_ensemble_availability(
                    agent_providers,
                    coord_providers,
                    agent_missing,
                    coord_missing,
                    available_providers,
                )
            )

            # Display status using helper method
            _display_ensemble_status(
                ensemble_name, is_available, missing_providers, missing_profiles
            )

        except Exception as e:
            click.echo(f"  🟥 {ensemble_file.stem} (error reading: {e})")


def show_provider_details(storage: CredentialStorage, provider: str) -> None:
    """Show detailed information about a provider."""
    from llm_orc.providers.registry import provider_registry

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


def display_default_models_config(
    config_manager: ConfigurationManager, available_providers: set[str]
) -> None:
    """Display default model profiles configuration."""
    project_config = config_manager.load_project_config()
    if project_config:
        default_models = project_config.get("project", {}).get("default_models", {})
        if default_models:
            click.echo(f"\n⚙️ Default model profiles ({len(default_models)} found):")
            for purpose, profile in default_models.items():
                # Resolve profile to show actual model and provider
                try:
                    (
                        resolved_model,
                        resolved_provider,
                    ) = config_manager.resolve_model_profile(profile)
                    # Check if provider is available for status indicator
                    provider_available = resolved_provider in available_providers
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


def display_global_profiles(
    global_config: dict[str, Any], available_providers: set[str]
) -> None:
    """Display global model profiles with availability indicators."""
    global_profiles = global_config.get("model_profiles", {})

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

            timeout_display = f"{timeout}s" if timeout != "not specified" else timeout
            click.echo(f"  {status_symbol} {profile_name}: {model} ({provider})")
            system_prompt_indicator = "✓" if has_system_prompt else "✗"
            click.echo(
                f"    Cost: {cost}, Timeout: {timeout_display}, "
                f"System prompt: {system_prompt_indicator}"
            )


def display_local_profiles(
    local_profiles: dict[str, Any], available_providers: set[str]
) -> None:
    """Display local model profiles with availability indicators."""
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


def display_providers_status(
    available_providers: set[str], config_manager: ConfigurationManager
) -> None:
    """Display provider availability status with detailed information."""
    global_config_dir = Path(config_manager.global_config_dir)

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
