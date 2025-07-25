"""Tab completion support for llm-orc CLI."""

from pathlib import Path

import click

from llm_orc.cli_modules.utils.config_utils import get_available_providers
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader


def complete_ensemble_names(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> list[str]:
    """Complete ensemble names from available ensemble directories.

    Args:
        ctx: Click context containing command arguments
        param: Click parameter being completed
        incomplete: Partial input to complete

    Returns:
        List of matching ensemble names
    """
    try:
        # Get config directory from context if provided
        config_dir = ctx.params.get("config_dir")

        # Get ensemble directories
        if config_dir:
            ensemble_dirs = [Path(config_dir)]
        else:
            # Initialize configuration manager
            config_manager = ConfigurationManager()
            ensemble_dirs = config_manager.get_ensembles_dirs()

        # Load ensembles from all directories
        loader = EnsembleLoader()
        ensemble_names: set[str] = set()

        for dir_path in ensemble_dirs:
            if dir_path.exists():
                try:
                    ensembles = loader.list_ensembles(str(dir_path))
                    for ensemble in ensembles:
                        ensemble_names.add(ensemble.name)
                except Exception:
                    # Skip directories that can't be read
                    continue

        # Filter by incomplete input
        matches = [name for name in ensemble_names if name.startswith(incomplete)]
        return sorted(matches)

    except Exception:
        # Return empty list on any error to avoid breaking completion
        return []


def complete_providers(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> list[str]:
    """Complete authentication provider names.

    Args:
        ctx: Click context containing command arguments
        param: Click parameter being completed
        incomplete: Partial input to complete

    Returns:
        List of matching provider names
    """
    try:
        config_manager = ConfigurationManager()
        providers = get_available_providers(config_manager)
        matches = [name for name in providers if name.startswith(incomplete)]
        return sorted(matches)
    except Exception:
        return []
