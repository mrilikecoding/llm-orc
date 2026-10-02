"""What a run looks things up with: the finders and the script resolver.

A run's closure walk and the shipper's are built from one config manager
and one project dir, here, so a verdict and a shipped closure come from
the lookup the executor would use (Arc 3: a parallel lookup that mostly
agreed with the run's gave three wrong accepts).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    EnsembleLoader,
    child_ensemble_search_dirs,
)
from llm_orc.core.execution.scripting.resolver import ScriptResolver
from llm_orc.services.handlers.run_preparation import LOAD_ERRORS, ChildLoadError


@dataclass(frozen=True)
class RunView:
    """The three lookups a closure walk needs, over one manager and one
    project dir: the profile map, the child finder and the resolver."""

    profiles: dict[str, dict[str, str]]
    find_child: Callable[[str], EnsembleConfig | None]
    resolver: ScriptResolver


def run_view(config_manager: ConfigurationManager, project_dir: Path | None) -> RunView:
    """The lookups of a run over ``config_manager`` and ``project_dir``,
    built from that pair and from nothing else."""
    return RunView(
        profiles=config_manager.get_model_profiles(),
        find_child=_child_finder(config_manager, project_dir),
        resolver=ScriptResolver(
            project_dir=project_dir, run_dir=config_manager.run_layer_dir
        ),
    )


def _child_finder(
    config_manager: ConfigurationManager, project_dir: Path | None
) -> Callable[[str], EnsembleConfig | None]:
    """The executor's child lookup over this manager and project dir:
    the same search dirs, the same by-filename finder."""
    loader = EnsembleLoader()

    def find(reference: str) -> EnsembleConfig | None:
        search_dirs = child_ensemble_search_dirs(project_dir, config_manager)
        try:
            return loader._find_ensemble_in_dirs(reference, search_dirs)
        except LOAD_ERRORS as e:
            raise ChildLoadError(reference, e) from e

    return find
