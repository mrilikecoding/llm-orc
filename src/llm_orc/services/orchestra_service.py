"""Shared application service for llm-orc.

Composes all handlers and provides a unified API for both
MCP and web ports.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.services.handlers.artifact_handler import ArtifactHandler
from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler
from llm_orc.services.handlers.execution_handler import ExecutionHandler
from llm_orc.services.handlers.help_handler import HelpHandler
from llm_orc.services.handlers.library_handler import LibraryHandler
from llm_orc.services.handlers.profile_handler import ProfileHandler
from llm_orc.services.handlers.promotion_handler import PromotionHandler
from llm_orc.services.handlers.provider_handler import ProviderHandler
from llm_orc.services.handlers.resource_handler import ResourceHandler
from llm_orc.services.handlers.script_handler import ScriptHandler
from llm_orc.services.handlers.validation_handler import ValidationHandler

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


class OrchestraService:
    """Shared application service composing all handlers.

    Both MCP and web ports delegate to this service for
    ensemble management, execution, and configuration.
    """

    def __init__(
        self,
        config_manager: ConfigurationManager | None = None,
        executor: EnsembleExecutor | None = None,
    ) -> None:
        self._project_path: Path | None = None
        self.config_manager = config_manager or ConfigurationManager()
        self._project_context = ProjectContext(
            project_path=None,
            config_manager=self.config_manager,
        )
        self.ensemble_loader = EnsembleLoader()
        self.artifact_manager = ArtifactManager()
        self._executor = executor

        self._project_lock = asyncio.Lock()

        self._help_handler = HelpHandler()
        self._resource_handler = ResourceHandler(
            self.config_manager, self.ensemble_loader
        )
        self._profile_handler = ProfileHandler(self.config_manager)
        self._artifact_handler = ArtifactHandler()
        self._script_handler = ScriptHandler()
        self._library_handler = LibraryHandler(
            self.config_manager, self.ensemble_loader
        )
        self._provider_handler = ProviderHandler(
            self._profile_handler, self.find_ensemble_by_name
        )
        self._validation_handler = ValidationHandler(
            self.config_manager,
            self.find_ensemble_by_name,
            self._profile_handler.get_all_profiles,
        )
        self._execution_handler = ExecutionHandler(
            self.config_manager,
            self.ensemble_loader,
            self.artifact_manager,
            self._get_executor,
            self.find_ensemble_by_name,
        )
        self._ensemble_crud_handler = EnsembleCrudHandler(
            self.config_manager,
            self.ensemble_loader,
            self.find_ensemble_by_name,
            self._resource_handler.read_artifact,
        )
        self._promotion_handler = PromotionHandler(
            self.config_manager,
            self._profile_handler,
            self._library_handler,
            self._provider_handler,
            self.find_ensemble_by_name,
        )

    @property
    def project_path(self) -> Path | None:
        return self._project_path

    def _get_executor(self) -> EnsembleExecutor:
        if self._executor is None:
            from llm_orc.core.execution.executor_factory import (
                ExecutorFactory,
            )

            self._executor = ExecutorFactory.create_root_executor(
                project_dir=self._project_path
            )
        return self._executor

    def find_ensemble_by_name(self, ensemble_name: str) -> Any:
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        for ensemble_dir in ensemble_dirs:
            config = self.ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                return config
        return None

    def list_ensembles_grouped(self) -> dict[str, list[Any]]:
        """List all ensembles grouped by tier."""
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        local: list[Any] = []
        library: list[Any] = []
        global_: list[Any] = []
        for dir_path in ensemble_dirs:
            ensembles = self.ensemble_loader.list_ensembles(str(dir_path))
            tier = self.config_manager.classify_tier(dir_path)
            if tier == "local":
                local.extend(ensembles)
            elif tier == "library":
                library.extend(ensembles)
            else:
                global_.extend(ensembles)
        return {"local": local, "library": library, "global": global_}

    def find_ensemble_in_dir(self, ensemble_name: str, dir_path: str) -> Any:
        """Find an ensemble by name in a specific directory.

        Args:
            ensemble_name: Name of the ensemble to find
            dir_path: Directory path to search in

        Returns:
            EnsembleConfig if found, None otherwise
        """
        return self.ensemble_loader.find_ensemble(dir_path, ensemble_name)

    def list_ensembles_in_dir(self, dir_path: str) -> list[Any]:
        """List all ensembles in a specific directory.

        Args:
            dir_path: Directory path to list ensembles from

        Returns:
            List of EnsembleConfig objects
        """
        return self.ensemble_loader.list_ensembles(dir_path)

    # === Context management ===

    def handle_set_project(self, path: str) -> dict[str, Any]:
        """Handle set_project logic."""
        project_dir = Path(path).resolve()
        if not project_dir.exists():
            return {
                "status": "error",
                "error": f"Path does not exist: {path}",
            }

        ctx = ProjectContext.create(project_dir)
        self._project_context = ctx
        self._project_path = ctx.project_path
        self.config_manager = ctx.config_manager
        self._executor = None

        self._ensemble_crud_handler.set_project_context(ctx)
        self._execution_handler.set_project_context(ctx)
        self._validation_handler.set_project_context(ctx)
        self._profile_handler.set_project_context(ctx)
        self._promotion_handler.set_project_context(ctx)
        self._resource_handler.set_project_context(ctx)
        self._library_handler.set_project_context(ctx)
        self._script_handler.set_project_context(ctx)
        self._artifact_handler.set_project_context(ctx)

        result: dict[str, Any] = {
            "status": "ok",
            "project_path": str(project_dir),
        }
        llm_orc_dir = project_dir / ".llm-orc"
        if not llm_orc_dir.exists():
            result["note"] = "No .llm-orc directory found; using global config only"
        return result

    async def handle_set_project_async(self, path: str) -> dict[str, Any]:
        """Thread-safe async wrapper for handle_set_project.

        Serializes concurrent project switches via a lock to prevent
        partial state corruption when multiple callers race.
        """
        async with self._project_lock:
            return self.handle_set_project(path)

    # === Resource reading ===

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return await self._resource_handler.read_ensembles()

    async def read_ensemble(self, name: str) -> dict[str, Any]:
        return await self._resource_handler.read_ensemble(name)

    async def read_artifacts(self, ensemble_name: str) -> list[dict[str, Any]]:
        return await self._resource_handler.read_artifacts(ensemble_name)

    async def read_artifact(
        self, ensemble_name: str, artifact_id: str
    ) -> dict[str, Any]:
        return await self._resource_handler.read_artifact(ensemble_name, artifact_id)

    async def read_metrics(self, ensemble_name: str) -> dict[str, Any]:
        return await self._resource_handler.read_metrics(ensemble_name)

    async def read_profiles(self) -> list[dict[str, Any]]:
        return await self._resource_handler.read_profiles()

    async def read_resource(self, uri: str) -> Any:
        return await self._resource_handler.read_resource(uri)

    # === Execution ===

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execution_handler.invoke(arguments)

    async def execute_streaming(
        self,
        ensemble_name: str,
        input_data: str,
        reporter: Any,
    ) -> dict[str, Any]:
        return await self._execution_handler.execute_streaming(
            ensemble_name, input_data, reporter
        )

    async def handle_streaming_event(
        self,
        event: dict[str, Any],
        reporter: Any,
        total_agents: int,
        state: dict[str, Any],
    ) -> None:
        await self._execution_handler.handle_streaming_event(
            event, reporter, total_agents, state
        )

    async def invoke_streaming(
        self, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield streaming events from execution."""
        async for event in self._execution_handler.invoke_streaming(params):
            yield event

    # === Validation ===

    async def validate_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._validation_handler.validate_ensemble(arguments)

    # === Ensemble CRUD ===

    async def create_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._ensemble_crud_handler.create_ensemble(arguments)

    async def update_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._ensemble_crud_handler.update_ensemble(arguments)

    async def delete_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._ensemble_crud_handler.delete_ensemble(arguments)

    async def analyze_execution(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._ensemble_crud_handler.analyze_execution(arguments)

    def get_local_ensembles_dir(self) -> Path:
        return self._ensemble_crud_handler.get_local_ensembles_dir()

    # === Profile CRUD ===

    async def list_profiles_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._profile_handler.list_profiles(arguments)

    async def create_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._profile_handler.create_profile(arguments)

    async def update_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._profile_handler.update_profile(arguments)

    async def delete_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._profile_handler.delete_profile(arguments)

    # === Artifact management ===

    async def delete_artifact(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._artifact_handler.delete_artifact(arguments)

    async def cleanup_artifacts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._artifact_handler.cleanup_artifacts(arguments)

    def list_artifact_ensembles(self) -> list[Any]:
        return self.artifact_manager.list_ensembles()

    # === Scripts ===

    async def list_scripts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._script_handler.list_scripts(arguments)

    async def get_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._script_handler.get_script(arguments)

    async def test_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._script_handler.test_script(arguments)

    async def create_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._script_handler.create_script(arguments)

    async def delete_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._script_handler.delete_script(arguments)

    # === Library ===

    async def library_browse(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._library_handler.browse(arguments)

    async def library_copy(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._library_handler.copy(arguments)

    async def library_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._library_handler.search(arguments)

    async def library_info(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._library_handler.info(arguments)

    # === Provider discovery ===

    async def get_provider_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._provider_handler.get_provider_status(arguments)

    async def check_ensemble_runnable(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._provider_handler.check_ensemble_runnable(arguments)

    # === Promotion ===

    async def promote_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._promotion_handler.promote_ensemble(arguments)

    async def demote_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._promotion_handler.demote_ensemble(arguments)

    async def list_dependencies(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._promotion_handler.list_dependencies(arguments)

    async def check_promotion_readiness(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._promotion_handler.check_promotion_readiness(arguments)

    # === Help ===

    def get_help_documentation(self) -> dict[str, Any]:
        return self._help_handler.get_help_documentation()
