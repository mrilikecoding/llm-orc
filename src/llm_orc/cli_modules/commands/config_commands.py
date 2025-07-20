"""Configuration management CLI commands."""

import shutil
from pathlib import Path

import click

from llm_orc.cli_modules.utils.cli_utils import echo_error, echo_info, echo_success
from llm_orc.cli_modules.utils.config_utils import (
    check_ensemble_availability,
    display_default_models_config,
    display_global_profiles,
    display_local_profiles,
    display_providers_status,
    get_available_providers,
    safe_load_yaml,
)
from llm_orc.config import ConfigurationManager


class ConfigCommands:
    """Configuration management commands."""

    @staticmethod
    def init_local_config(project_name: str | None) -> None:
        """Initialize local .llm-orc configuration for current project."""
        config_manager = ConfigurationManager()

        try:
            config_manager.init_local_config(project_name)
            echo_success("Local configuration initialized successfully!")
            click.echo("Created .llm-orc directory with:")
            click.echo("  - ensembles/   (project-specific ensembles)")
            click.echo("  - models/      (shared model configurations)")
            click.echo("  - scripts/     (project-specific scripts)")
            click.echo("  - config.yaml  (project configuration)")
            echo_info(
                "You can now create project-specific ensembles in .llm-orc/ensembles/"
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    @staticmethod
    def reset_global_config(backup: bool, preserve_auth: bool) -> None:
        """Reset global configuration to template defaults."""
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
        template_path = (
            Path(__file__).parent.parent.parent / "templates" / "global-config.yaml"
        )
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

            echo_success(f"Global config reset complete at {global_config_dir}")

            if preserve_auth and auth_files:
                click.echo("🔐 Authentication credentials preserved")
            elif not preserve_auth:
                echo_info(
                    "Note: You may need to reconfigure authentication "
                    "with 'llm-orc auth setup'"
                )
        else:
            raise click.ClickException(f"Template not found at {template_path}")

    @staticmethod
    def check_global_config() -> None:
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
                available_providers = get_available_providers(config_manager)

                # Show providers FIRST, right after status
                display_providers_status(available_providers, config_manager)

                # Read ONLY global config file, not merged profiles
                global_config = safe_load_yaml(global_config_path)

                # Show default model profiles configuration
                display_default_models_config(config_manager, available_providers)

                # Check global ensembles SECOND
                global_ensembles_dir = global_config_dir / "ensembles"
                check_ensemble_availability(
                    global_ensembles_dir, available_providers, config_manager
                )

                # Show global profiles
                display_global_profiles(global_config, available_providers)

            except Exception as e:
                echo_error(f"Error reading config: {e}")
        else:
            click.echo("Status: missing")
            echo_info("Run 'llm-orc config init' to create it")

    @staticmethod
    def check_local_config() -> None:
        """Check local .llm-orc configuration status."""
        local_config_dir = Path(".llm-orc")
        local_config_path = local_config_dir / "config.yaml"

        if local_config_path.exists():
            # Show basic info about the config
            try:
                config_manager = ConfigurationManager()

                # Check project config first to get project name
                project_config = config_manager.load_project_config()
                if project_config:
                    project_name = project_config.get("project", {}).get(
                        "name", "Unknown"
                    )
                    click.echo(f"Local Configuration Status: {project_name}")
                    click.echo(f"Directory: {local_config_dir.absolute()}")
                    click.echo("Status: configured")

                    # Get available providers for ensemble checking
                    available_providers = get_available_providers(config_manager)

                    # Check local ensembles with availability indicators
                    ensembles_dir = local_config_dir / "ensembles"
                    check_ensemble_availability(
                        ensembles_dir, available_providers, config_manager
                    )

                    # Show local model profiles
                    local_profiles = project_config.get("model_profiles", {})
                    if local_profiles:
                        display_local_profiles(local_profiles, available_providers)
                else:
                    click.echo("Local Configuration Status:")
                    click.echo(f"Directory: {local_config_dir.absolute()}")
                    click.echo("Status: configured but no project config found")

            except Exception as e:
                click.echo("Local Configuration Status:")
                click.echo(f"Directory: {local_config_dir.absolute()}")
                echo_error(f"Error reading local config: {e}")
        else:
            click.echo("Local Configuration Status:")
            click.echo(f"Directory: {local_config_dir.absolute()}")
            click.echo("Status: missing")
            echo_info("Run 'llm-orc config init' to create it")

    @staticmethod
    def reset_local_config(
        backup: bool, preserve_ensembles: bool, project_name: str | None
    ) -> None:
        """Reset local .llm-orc configuration to template defaults."""
        config_manager = ConfigurationManager()
        local_config_dir = Path(".llm-orc")

        if not local_config_dir.exists():
            echo_error("No local .llm-orc directory found")
            echo_info("Run 'llm-orc config init' to create initial local config")
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

            echo_success(f"Local config reset complete at {local_config_dir}")

            if preserve_ensembles and ensembles_backup:
                click.echo("🎭 Existing ensembles preserved")
            elif not preserve_ensembles:
                echo_info("Note: All ensembles were reset to template defaults")

        except ValueError as e:
            raise click.ClickException(str(e)) from e
