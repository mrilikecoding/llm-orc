"""Configuration management CLI commands."""

import os
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
from llm_orc.core.config.config_manager import ConfigurationManager


def _create_backup_if_requested(backup: bool, global_config_dir: Path) -> None:
    """Create backup if requested and config exists.

    Args:
        backup: Whether to create a backup
        global_config_dir: Path to the global config directory
    """
    if backup and global_config_dir.exists():
        backup_path = global_config_dir.with_suffix(".backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(global_config_dir, backup_path)
        click.echo(f"📦 Backed up existing config to {backup_path}")


def _preserve_auth_files_if_requested(
    preserve_auth: bool, global_config_dir: Path
) -> list[tuple[str, bytes]]:
    """Preserve authentication files if requested.

    Args:
        preserve_auth: Whether to preserve authentication files
        global_config_dir: Path to the global config directory

    Returns:
        List of tuples containing (filename, file_content) for preserved files
    """
    auth_files: list[tuple[str, bytes]] = []
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
    return auth_files


def _recreate_config_directory(global_config_dir: Path) -> None:
    """Remove existing config directory and create fresh one.

    Args:
        global_config_dir: Path to the global config directory
    """
    # Remove existing config directory
    if global_config_dir.exists():
        shutil.rmtree(global_config_dir)

    # Create fresh config directory
    global_config_dir.mkdir(parents=True, exist_ok=True)


def _install_template_and_restore_auth(
    global_config_dir: Path,
    template_path: Path,
    auth_files: list[tuple[str, bytes]],
) -> None:
    """Install template and restore authentication files.

    Args:
        global_config_dir: Path to the global config directory
        template_path: Path to the template file
        auth_files: List of tuples containing (filename, file_content)
                   for auth files to restore

    Raises:
        click.ClickException: If template file doesn't exist
    """
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

        if auth_files:
            click.echo("🔐 Authentication credentials preserved")
        else:
            echo_info(
                "Note: You may need to reconfigure authentication "
                "with 'llm-orc auth setup'"
            )
    else:
        raise click.ClickException(f"Template not found at {template_path}")


def _create_local_backup_if_requested(backup: bool, local_config_dir: Path) -> None:
    """Create local backup if requested.

    Args:
        backup: Whether to create a backup
        local_config_dir: Path to the local config directory
    """
    if backup:
        backup_path = Path(".llm-orc.backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(local_config_dir, backup_path)
        click.echo(f"📦 Backed up existing local config to {backup_path}")


def _preserve_ensembles_if_requested(
    preserve_ensembles: bool, local_config_dir: Path
) -> dict[str, str]:
    """Preserve ensembles if requested.

    Args:
        preserve_ensembles: Whether to preserve ensembles
        local_config_dir: Path to the local config directory

    Returns:
        Dictionary mapping ensemble filenames to their content
    """
    ensembles_backup: dict[str, str] = {}
    if preserve_ensembles:
        ensembles_dir = local_config_dir / "ensembles"
        if ensembles_dir.exists():
            # Save ensembles directory content
            for ensemble_file in ensembles_dir.glob("*.yaml"):
                ensembles_backup[ensemble_file.name] = ensemble_file.read_text()
            click.echo(f"🎭 Preserving {len(ensembles_backup)} ensemble(s)")
    return ensembles_backup


def _reset_and_initialize_local_config(
    local_config_dir: Path,
    config_manager: ConfigurationManager,
    project_name: str | None,
) -> None:
    """Reset local config and initialize fresh one.

    Args:
        local_config_dir: Path to the local config directory
        config_manager: Configuration manager instance
        project_name: Name for the new project

    Raises:
        click.ClickException: If initialization fails
    """
    # Remove existing local config
    shutil.rmtree(local_config_dir)

    # Initialize fresh local config
    try:
        config_manager.init_local_config(project_name)
        click.echo("📋 Created fresh local config from template")
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _restore_ensembles_and_complete(
    local_config_dir: Path,
    ensembles_backup: dict[str, str],
    preserve_ensembles: bool,
) -> None:
    """Restore ensembles and complete the reset process.

    Args:
        local_config_dir: Path to the local config directory
        ensembles_backup: Dictionary mapping ensemble filenames to their content
        preserve_ensembles: Whether ensembles were requested to be preserved
    """
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


class ConfigCommands:
    """Configuration management commands."""

    @staticmethod
    def init_local_config(project_name: str | None, with_scripts: bool = True) -> None:
        """Initialize local .llm-orc configuration for current project.

        Args:
            project_name: Optional project name (defaults to directory name)
            with_scripts: Install primitive scripts from library (default: True)
        """
        config_manager = ConfigurationManager()

        try:
            config_manager.init_local_config(project_name)
            echo_success("Local configuration initialized successfully!")
            click.echo("Created .llm-orc directory with:")
            click.echo("  - ensembles/   (project-specific ensembles)")
            click.echo("  - models/      (shared model configurations)")
            click.echo("  - scripts/     (project-specific scripts)")
            click.echo("  - config.yaml  (project configuration)")

            if with_scripts:
                # Install primitive scripts from library
                script_count = ConfigCommands._install_library_primitives()
                if script_count > 0:
                    click.echo(f"✓ Installed {script_count} primitive scripts")
                    click.echo("\nReady! Try:")
                    click.echo(
                        "  llm-orc scripts list          # See installed primitives"
                    )
                    click.echo(
                        "  llm-orc list-ensembles        # See example ensembles"
                    )
                else:
                    echo_info("No library primitives found to install")
            else:
                echo_info("Skipped primitive script installation (--no-scripts)")

            echo_info(
                "You can now create project-specific ensembles in .llm-orc/ensembles/"
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    @staticmethod
    def _get_library_scripts_path() -> Path | None:
        """Get path to library scripts directory.

        Supports:
        1. LLM_ORC_LIBRARY_PATH env var (custom location)
        2. .llm-orc/.env file (project-specific config)
        3. LLM_ORC_LIBRARY_SOURCE=local (submodule)
        4. Current working directory (llm-orchestra-library/)

        Returns:
            Path to library scripts/primitives directory, or None if not found
        """
        # Load .llm-orc/.env if it exists (but don't override existing env vars)
        dotenv_path = Path.cwd() / ".llm-orc" / ".env"
        if dotenv_path.exists():
            from dotenv import load_dotenv

            load_dotenv(dotenv_path, override=False)

        # Check for custom library path
        custom_path = os.environ.get("LLM_ORC_LIBRARY_PATH")
        if custom_path:
            library_scripts = Path(custom_path) / "scripts" / "primitives"
            if library_scripts.exists():
                return library_scripts
            return None

        # Check for library source mode
        library_source = os.environ.get("LLM_ORC_LIBRARY_SOURCE", "local")

        if library_source == "local":
            # Try submodule relative to package installation
            package_root = Path(__file__).parent.parent.parent.parent
            submodule_path = package_root / "llm-orchestra-library"
            if submodule_path.exists():
                return submodule_path / "scripts" / "primitives"

        # Try current working directory
        cwd_path = Path.cwd() / "llm-orchestra-library"
        if cwd_path.exists():
            return cwd_path / "scripts" / "primitives"

        return None

    @staticmethod
    def _install_library_primitives() -> int:
        """Copy primitive scripts from library to .llm-orc/scripts/.

        Returns:
            Number of scripts installed
        """
        library_scripts = ConfigCommands._get_library_scripts_path()
        local_scripts = Path(".llm-orc") / "scripts" / "primitives"

        if not library_scripts:
            return 0

        script_count = 0
        # Copy each category directory
        for category_dir in library_scripts.iterdir():
            if category_dir.is_dir() and category_dir.name != "__pycache__":
                dest = local_scripts / category_dir.name
                dest.mkdir(parents=True, exist_ok=True)

                # Copy all Python scripts in the category
                for script in category_dir.glob("*.py"):
                    if script.name != "__init__.py":
                        dest_script = dest / script.name
                        shutil.copy2(script, dest_script)
                        # Make executable
                        dest_script.chmod(dest_script.stat().st_mode | 0o111)
                        script_count += 1

        return script_count

    @staticmethod
    def reset_global_config(backup: bool, preserve_auth: bool) -> None:
        """Reset global configuration to template defaults."""
        config_manager = ConfigurationManager()
        global_config_dir = Path(config_manager.global_config_dir)

        # Create backup if requested
        _create_backup_if_requested(backup, global_config_dir)

        # Preserve authentication files if requested
        auth_files = _preserve_auth_files_if_requested(preserve_auth, global_config_dir)

        # Remove existing config directory and create fresh one
        _recreate_config_directory(global_config_dir)

        # Copy template to global config and restore auth files
        template_path = (
            Path(__file__).parent.parent.parent / "templates" / "global-config.yaml"
        )
        _install_template_and_restore_auth(global_config_dir, template_path, auth_files)

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
        _create_local_backup_if_requested(backup, local_config_dir)

        # Preserve ensembles if requested
        ensembles_backup = _preserve_ensembles_if_requested(
            preserve_ensembles, local_config_dir
        )

        # Reset and initialize fresh local config
        _reset_and_initialize_local_config(
            local_config_dir, config_manager, project_name
        )

        # Restore ensembles and complete
        _restore_ensembles_and_complete(
            local_config_dir, ensembles_backup, preserve_ensembles
        )


# Module-level exports for CLI imports
init_local_config = ConfigCommands.init_local_config
reset_global_config = ConfigCommands.reset_global_config
check_global_config = ConfigCommands.check_global_config
check_local_config = ConfigCommands.check_local_config
reset_local_config = ConfigCommands.reset_local_config
