"""Factory for creating EnsembleExecutor instances."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from llm_orc.cli_library.template_provider import LibraryTemplateProvider
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.models.model_factory import ModelFactory

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


class ExecutorFactory:
    """Creates EnsembleExecutor instances with proper infrastructure sharing.

    Centralizes the construction of root and child executors.
    Root executors create fresh infrastructure; child executors
    share immutable infrastructure from their parent.
    """

    @staticmethod
    def create_root_executor(
        project_dir: Path | None = None,
        *,
        save_artifacts: bool = True,
    ) -> EnsembleExecutor:
        """Create a top-level executor with fresh infrastructure.

        Args:
            project_dir: Project directory path.
            save_artifacts: Whether to save execution artifacts.

        Returns:
            Configured EnsembleExecutor.
        """
        from llm_orc.core.execution.ensemble_execution import (
            EnsembleExecutor,
        )

        config_manager = ConfigurationManager(
            template_provider=LibraryTemplateProvider(),
        )
        credential_storage = CredentialStorage(config_manager)
        model_factory = ModelFactory(config_manager, credential_storage)

        return EnsembleExecutor(
            project_dir=project_dir,
            _config_manager=config_manager,
            _credential_storage=credential_storage,
            _model_factory=model_factory,
            _depth=0,
            _save_artifacts=save_artifacts,
        )

    @staticmethod
    def create_child_executor(
        parent: EnsembleExecutor,
        depth: int,
    ) -> EnsembleExecutor:
        """Create a child executor sharing parent's immutable infrastructure.

        Shares config_manager, credential_storage, model_factory
        from parent. Isolates mutable state (usage collector, event
        queue, streaming tracker). Disables artifact saving.

        Args:
            parent: Parent executor to share infrastructure from.
            depth: Nesting depth for the child.

        Returns:
            Child EnsembleExecutor.
        """
        from llm_orc.core.execution.ensemble_execution import (
            EnsembleExecutor,
        )

        return EnsembleExecutor(
            project_dir=parent._project_dir,
            _config_manager=parent._config_manager,
            _credential_storage=parent._credential_storage,
            _model_factory=parent._model_factory,
            _depth=depth,
            _save_artifacts=False,
        )
