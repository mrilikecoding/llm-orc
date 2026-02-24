"""Tab completion support for llm-orc CLI."""

from pathlib import Path

import click

from llm_orc.cli_modules.utils.config_utils import get_available_providers
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader


def complete_ensemble_names(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> list[str]:
    """Complete ensemble names from available ensemble directories.

    Args:
        ctx: Click context containing command arguments
        _param: Click parameter being completed (unused)
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
            config_manager = ConfigurationManager(provision=False)
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
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> list[str]:
    """Complete authentication provider names.

    Args:
        ctx: Click context containing command arguments
        _param: Click parameter being completed (unused)
        incomplete: Partial input to complete

    Returns:
        List of matching provider names
    """
    try:
        config_manager = ConfigurationManager(provision=False)
        providers = get_available_providers(config_manager)
        matches = [name for name in providers if name.startswith(incomplete)]
        return sorted(matches)
    except Exception:
        return []


def complete_library_ensemble_paths(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> list[str]:
    """Complete library ensemble paths (e.g., 'code-analysis/security-review').

    Args:
        ctx: Click context containing command arguments
        _param: Click parameter being completed (unused)
        incomplete: Partial input to complete

    Returns:
        List of matching library ensemble paths
    """
    try:
        # Import here to avoid circular imports
        from llm_orc.cli_library.library import get_library_categories

        # Get available categories
        categories = get_library_categories()

        # If incomplete doesn't contain '/', suggest categories
        if "/" not in incomplete:
            matches = [f"{cat}/" for cat in categories if cat.startswith(incomplete)]
            return sorted(matches)

        # If incomplete contains '/', try to complete ensemble names within category
        try:
            category, partial_ensemble = incomplete.split("/", 1)
            if category in categories:
                # Fetch actual ensembles from GitHub API
                from llm_orc.cli_library.library import get_category_ensembles

                ensembles = get_category_ensembles(category)
                matches = [
                    f"{category}/{ensemble['name']}"
                    for ensemble in ensembles
                    if ensemble["name"].startswith(partial_ensemble)
                ]
                return sorted(matches)
        except ValueError:
            # Invalid format, return empty
            return []

        return []

    except Exception:
        # Return empty list on any error to avoid breaking completion
        return []
