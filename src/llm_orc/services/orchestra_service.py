"""Shared application service for llm-orc.

Composes all handlers and provides a unified API for both
MCP and web ports.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_orc.cli_library.template_provider import LibraryTemplateProvider
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    EnsembleLoader,
    child_ensemble_search_dirs,
)
from llm_orc.core.config.state import ARTIFACTS_DIRNAME, resolve_state_dir
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.scripting.resolver import ScriptResolver
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.models.base import HTTPConnectionPool
from llm_orc.services.handlers.artifact_handler import ArtifactHandler
from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler
from llm_orc.services.handlers.execution_handler import (
    ExecutionHandler,
    PreparedRun,
)
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
        self.config_manager = config_manager or ConfigurationManager(
            template_provider=LibraryTemplateProvider(),
        )
        self._project_context = ProjectContext(
            project_path=None,
            config_manager=self.config_manager,
        )
        self.ensemble_loader = EnsembleLoader()
        self.artifact_manager = ArtifactManager(
            artifacts_dir=resolve_state_dir(self.config_manager.local_config_dir)
            / ARTIFACTS_DIRNAME
        )
        self._executor = executor
        # A caller-injected executor (tests, embedders) is returned as-is
        # forever - it bypasses ExecutorFactory entirely, so there is
        # nothing of ours to mint a fresh execution_id for. Only
        # handle_set_project's reset re-enables the auto-build path.
        self._executor_injected = executor is not None

        self._project_lock = asyncio.Lock()

        # Configure HTTP connection pool with project performance settings
        self._configure_http_pool()

        self._help_handler = HelpHandler()
        self._resource_handler = ResourceHandler(
            self.config_manager, self.ensemble_loader
        )
        self._profile_handler = ProfileHandler(self.config_manager)
        self._artifact_handler = ArtifactHandler(config_manager=self.config_manager)
        self._script_handler = ScriptHandler(config_manager=self.config_manager)
        self._library_handler = LibraryHandler(
            self.config_manager, self.ensemble_loader
        )
        self._provider_handler = ProviderHandler(
            self._profile_handler,
            self.find_ensemble_by_name,
            lambda: ScriptResolver(project_dir=self._project_path),
            find_child=self.find_child_ensemble,
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
            preflight_fn=self._provider_handler.preflight,
            layer_executor_fn=self._get_layer_executor,
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

    def _configure_http_pool(self) -> None:
        """Configure the HTTP connection pool from current project config."""
        try:
            performance_config = self.config_manager.load_performance_config()
            HTTPConnectionPool.configure(performance_config)
        except Exception:
            pass  # Pool will use module-level defaults

    def _get_executor(self) -> EnsembleExecutor:
        """Build the executor for one MCP/REST invocation.

        A caller-injected executor (the ``executor=`` constructor param —
        tests, embedders) is returned as-is, every call: it bypasses
        ExecutorFactory entirely, so there is no ModelFactory of ours to
        mint a fresh id for.

        Otherwise, a fresh execution_id — and therefore a fresh
        ModelFactory — is minted on every call, so two invocations
        through this service never share one x-opencode-session on the
        wire (SF5); any child ensemble spawned within a single
        invocation still shares that invocation's id (Invariant 10:
        child executors share their parent's ModelFactory instance).
        The first call's ConfigurationManager and CredentialStorage —
        real disk I/O — are cached and reused by every later call; only
        the cheap ModelFactory/EnsembleExecutor wiring is rebuilt each
        time.
        """
        if self._executor_injected:
            assert self._executor is not None
            return self._executor

        from llm_orc.core.execution.executor_factory import (
            ExecutorFactory,
        )

        if self._executor is None:
            # The executor's artifact manager and script cache resolve their
            # state dir from this manager's local_config_dir, so it must be
            # the project's, not a cwd discovery.
            self._executor = ExecutorFactory.create_root_executor(
                project_dir=self._project_path,
                config_manager=self.config_manager,
            )
            return self._executor

        self._executor = ExecutorFactory.create_root_executor(
            project_dir=self._project_path,
            config_manager=self._executor._config_manager,
            credential_storage=self._executor._credential_storage,
        )
        return self._executor

    def _get_layer_executor(
        self, view: ConfigurationManager, save_artifacts: bool
    ) -> EnsembleExecutor:
        """The executor of one run with a layer: built on the view, never
        cached, so the view cannot reach a later run (Arc 4). Credentials
        are the service's cached storage; a caller-injected executor is
        only ever used for runs with no layer."""
        from llm_orc.core.execution.executor_factory import ExecutorFactory

        storage = None
        if not self._executor_injected:
            if self._executor is None:
                self._get_executor()  # builds and caches the credential storage
            assert self._executor is not None
            storage = self._executor._credential_storage
        return ExecutorFactory.create_root_executor(
            project_dir=self._project_path,
            config_manager=view,
            credential_storage=storage,
            save_artifacts=save_artifacts,
        )

    def find_ensemble_by_name(self, ensemble_name: str) -> Any:
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        for ensemble_dir in ensemble_dirs:
            config = self.ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                return config
        return None

    def find_child_ensemble(self, reference: str) -> EnsembleConfig | None:
        """Resolve a child reference exactly as the executor does
        (``EnsembleExecutor._resolve_ensemble_reference``): the same
        search dirs, the same by-filename finder. Reads the project path
        and config manager at call time, so ``set_project`` is honored."""
        search_dirs = child_ensemble_search_dirs(
            self._project_path, self.config_manager
        )
        return self.ensemble_loader._find_ensemble_in_dirs(reference, search_dirs)

    def list_ensembles_grouped(self) -> dict[str, list[Any]]:
        """List all ensembles grouped by tier (local, library, global, packaged)."""
        groups: dict[str, list[Any]] = {
            "local": [],
            "library": [],
            "global": [],
            "packaged": [],
        }
        for dir_path in self.config_manager.get_ensembles_dirs():
            ensembles = self.ensemble_loader.list_ensembles(str(dir_path))
            tier = self.config_manager.classify_tier(dir_path)
            groups.get(tier, groups["global"]).extend(ensembles)
        return groups

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
        self.artifact_manager = ArtifactManager(
            artifacts_dir=resolve_state_dir(self.config_manager.local_config_dir)
            / ARTIFACTS_DIRNAME
        )
        self._executor = None
        self._executor_injected = False

        self._configure_http_pool()
        self._ensemble_crud_handler.set_project_context(ctx)
        self._execution_handler.set_project_context(ctx)
        self._execution_handler.set_artifact_manager(self.artifact_manager)
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

    @asynccontextmanager
    async def prepared_run(
        self,
        request: Mapping[str, Any],
        lookup: Callable[[str], Any] | None = None,
    ) -> AsyncIterator[PreparedRun]:
        """The root and executor of one gated run, the preparation step
        REST and MCP share. ``lookup`` finds a named root (default: the
        service's tiers). Raises ``RunRefusedError`` before any agent
        starts."""
        async with self._execution_handler.prepared(
            request, lookup or self.find_ensemble_by_name, "Ensemble does not exist"
        ) as run:
            yield run

    async def execute_streaming(
        self,
        ensemble_name: str | None,
        input_data: str,
        reporter: Any,
        injection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._execution_handler.execute_streaming(
            ensemble_name, input_data, reporter, injection
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
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield streaming events from execution."""
        events = self._execution_handler.invoke_streaming(params)
        async with aclosing(events):
            async for event in events:
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
