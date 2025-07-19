"""CLI configuration and provider management utilities."""

from pathlib import Path
from typing import Any

import click
import yaml

from llm_orc.authentication import CredentialStorage
from llm_orc.config import ConfigurationManager


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


def show_provider_details(storage: CredentialStorage, provider: str) -> None:
    """Show detailed information about a provider."""
    from llm_orc.provider_registry import provider_registry

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
