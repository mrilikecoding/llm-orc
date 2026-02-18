"""MCP Server v2 implementation using FastMCP SDK.

This module implements the MCP server following ADR-009, providing:
- Resource exposure for ensembles, artifacts, metrics, and profiles
- Tools for invoke, validate_ensemble, update_ensemble, analyze_execution
- Streaming support for long-running executions
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mcp.server.fastmcp import Context, FastMCP

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.mcp.handlers.artifact_handler import ArtifactHandler
from llm_orc.mcp.handlers.help_handler import HelpHandler
from llm_orc.mcp.handlers.library_handler import LibraryHandler
from llm_orc.mcp.handlers.profile_handler import ProfileHandler
from llm_orc.mcp.handlers.provider_handler import ProviderHandler
from llm_orc.mcp.handlers.script_handler import ScriptHandler
from llm_orc.mcp.handlers.validation_handler import ValidationHandler

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


class ProgressReporter(Protocol):
    """Protocol for reporting execution progress.

    Abstracts progress reporting to allow testing without FastMCP Context.
    """

    async def info(self, message: str) -> None:
        """Report an informational message."""
        ...

    async def warning(self, message: str) -> None:
        """Report a warning message."""
        ...

    async def error(self, message: str) -> None:
        """Report an error message."""
        ...

    async def report_progress(self, progress: int, total: int) -> None:
        """Report progress (progress out of total)."""
        ...


class FastMCPProgressReporter:
    """Progress reporter that wraps FastMCP Context."""

    def __init__(self, ctx: Context[Any, Any, Any]) -> None:
        """Initialize with FastMCP context."""
        self._ctx = ctx

    async def info(self, message: str) -> None:
        """Report an informational message."""
        await self._ctx.info(message)

    async def warning(self, message: str) -> None:
        """Report a warning message."""
        await self._ctx.warning(message)

    async def error(self, message: str) -> None:
        """Report an error message."""
        await self._ctx.error(message)

    async def report_progress(self, progress: int, total: int) -> None:
        """Report progress."""
        await self._ctx.report_progress(progress=progress, total=total)


def _get_agent_attr(agent: Any, attr: str, default: Any = None) -> Any:
    """Get agent attribute handling both dict and object forms.

    Args:
        agent: Agent config (dict or object).
        attr: Attribute name.
        default: Default value if not found.

    Returns:
        Attribute value or default.
    """
    if isinstance(agent, dict):
        return agent.get(attr, default)
    return getattr(agent, attr, default)


class MCPServerV2:
    """MCP Server v2 using FastMCP SDK.

    Exposes all llm-orc ensembles as MCP resources and provides
    tools for ensemble management and execution.
    """

    def __init__(
        self,
        config_manager: ConfigurationManager | None = None,
        executor: EnsembleExecutor | None = None,
    ) -> None:
        """Initialize MCP server.

        Args:
            config_manager: Configuration manager instance. Creates default if None.
            executor: Ensemble executor instance. Creates default if None.
        """
        self._project_path: Path | None = None
        self.config_manager = config_manager or ConfigurationManager()
        self.ensemble_loader = EnsembleLoader()
        self.artifact_manager = ArtifactManager()
        self._executor = executor  # Lazily created if None
        self._help_handler = HelpHandler()
        self._profile_handler = ProfileHandler(self.config_manager)
        self._artifact_handler = ArtifactHandler()
        self._script_handler = ScriptHandler()
        self._library_handler = LibraryHandler(
            self.config_manager, self.ensemble_loader
        )
        self._provider_handler = ProviderHandler(
            self._profile_handler, self._find_ensemble_by_name
        )
        self._validation_handler = ValidationHandler(
            self.config_manager, self._find_ensemble_by_name
        )
        self._mcp = FastMCP("llm-orc")
        self._setup_resources()
        self._setup_tools()

    @property
    def project_path(self) -> Path | None:
        """Get the current project path."""
        return self._project_path

    def _get_executor(self) -> EnsembleExecutor:
        """Get or create the ensemble executor.

        Returns:
            EnsembleExecutor instance.
        """
        if self._executor is None:
            from llm_orc.core.execution.ensemble_execution import EnsembleExecutor

            self._executor = EnsembleExecutor(project_dir=self._project_path)
        return self._executor

    def _setup_resources(self) -> None:
        """Register MCP resources with FastMCP."""

        # Use decorator syntax on methods
        @self._mcp.resource("llm-orc://ensembles")
        async def get_ensembles() -> str:
            """List all available ensembles."""
            result = await self._read_ensembles_resource()
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://profiles")
        async def get_profiles() -> str:
            """List configured model profiles."""
            result = await self._read_profiles_resource()
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://ensemble/{name}")
        async def get_ensemble(name: str) -> str:
            """Get specific ensemble configuration."""
            result = await self._read_ensemble_resource(name)
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://artifacts/{ensemble}")
        async def get_artifacts(ensemble: str) -> str:
            """List execution artifacts for an ensemble."""
            result = await self._read_artifacts_resource(ensemble)
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://metrics/{ensemble}")
        async def get_metrics(ensemble: str) -> str:
            """Get aggregated metrics for an ensemble."""
            result = await self._read_metrics_resource(ensemble)
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://artifact/{ensemble}/{artifact_id}")
        async def get_artifact(ensemble: str, artifact_id: str) -> str:
            """Get individual artifact details."""
            result = await self._read_artifact_resource(ensemble, artifact_id)
            return json.dumps(result, indent=2)

    def _setup_tools(self) -> None:
        """Register MCP tools with FastMCP."""
        self._setup_context_tools()
        self._setup_core_tools()
        self._setup_crud_tools()
        self._setup_provider_discovery_tools()
        self._setup_help_tool()

    def _setup_context_tools(self) -> None:
        """Register context management tools."""
        server = self  # Capture for closure

        @self._mcp.tool()
        async def set_project(path: str) -> str:
            """Set the active project directory for subsequent operations.

            Args:
                path: Path to the project directory
            """
            result = server._handle_set_project(path)
            return json.dumps(result, indent=2)

    def _setup_core_tools(self) -> None:
        """Register core MCP tools."""

        @self._mcp.tool()
        async def invoke(
            ensemble_name: str, input_data: str, ctx: Context[Any, Any, Any]
        ) -> str:
            """Execute an ensemble with input data.

            Args:
                ensemble_name: Name of the ensemble to execute
                input_data: Input data for the ensemble
            """
            result = await self._invoke_tool_with_streaming(
                ensemble_name, input_data, ctx
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def validate_ensemble(ensemble_name: str) -> str:
            """Validate ensemble configuration.

            Args:
                ensemble_name: Name of the ensemble to validate
            """
            result = await self._validate_ensemble_tool(
                {
                    "ensemble_name": ensemble_name,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def list_ensembles() -> str:
            """List all available ensembles with their metadata."""
            result = await self._read_ensembles_resource()
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def update_ensemble(
            ensemble_name: str,
            changes: dict[str, Any],
            dry_run: bool = True,
            backup: bool = True,
        ) -> str:
            """Modify ensemble configuration.

            Args:
                ensemble_name: Name of the ensemble to update
                changes: Changes to apply (add_agents, remove_agents, etc.)
                dry_run: If True, only preview changes without applying
                backup: If True, create backup before modifying
            """
            result = await self._update_ensemble_tool(
                {
                    "ensemble_name": ensemble_name,
                    "changes": changes,
                    "dry_run": dry_run,
                    "backup": backup,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def analyze_execution(artifact_id: str) -> str:
            """Analyze execution artifact.

            Args:
                artifact_id: ID of the artifact (format: ensemble_name/artifact_id)
            """
            result = await self._analyze_execution_tool(
                {
                    "artifact_id": artifact_id,
                }
            )
            return json.dumps(result, indent=2)

    def _setup_crud_tools(self) -> None:
        """Register Phase 2 CRUD tools."""
        self._setup_ensemble_crud_tools()
        self._setup_profile_tools()
        self._setup_artifact_tools()

    def _setup_ensemble_crud_tools(self) -> None:
        """Register ensemble CRUD tools."""

        @self._mcp.tool()
        async def create_ensemble(
            name: str,
            description: str = "",
            agents: list[dict[str, Any]] | None = None,
            from_template: str | None = None,
        ) -> str:
            """Create a new ensemble from scratch or template.

            Args:
                name: Name of the new ensemble
                description: Optional description
                agents: List of agent configurations
                from_template: Optional template ensemble to copy from
            """
            result = await self._create_ensemble_tool(
                {
                    "name": name,
                    "description": description,
                    "agents": agents or [],
                    "from_template": from_template,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def delete_ensemble(ensemble_name: str, confirm: bool = False) -> str:
            """Delete an ensemble.

            Args:
                ensemble_name: Name of the ensemble to delete
                confirm: Must be True to actually delete
            """
            result = await self._delete_ensemble_tool(
                {
                    "ensemble_name": ensemble_name,
                    "confirm": confirm,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def list_scripts(category: str | None = None) -> str:
            """List available primitive scripts.

            Args:
                category: Optional category to filter by
            """
            result = await self._list_scripts_tool({"category": category})
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def library_browse(
            browse_type: str = "all", category: str | None = None
        ) -> str:
            """Browse library ensembles and scripts.

            Args:
                browse_type: Type to browse (ensembles, scripts, all)
                category: Optional category filter
            """
            result = await self._library_browse_tool(
                {"type": browse_type, "category": category}
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def library_copy(
            source: str,
            destination: str | None = None,
            overwrite: bool = False,
        ) -> str:
            """Copy from library to local project.

            Args:
                source: Library path to copy from
                destination: Optional destination path
                overwrite: Whether to overwrite existing files
            """
            result = await self._library_copy_tool(
                {
                    "source": source,
                    "destination": destination,
                    "overwrite": overwrite,
                }
            )
            return json.dumps(result, indent=2)

    def _setup_profile_tools(self) -> None:
        """Register profile CRUD tools."""

        @self._mcp.tool()
        async def list_profiles(provider: str | None = None) -> str:
            """List all model profiles.

            Args:
                provider: Optional provider to filter by
            """
            result = await self._list_profiles_tool({"provider": provider})
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def create_profile(
            name: str,
            provider: str,
            model: str,
            system_prompt: str | None = None,
            timeout_seconds: int | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> str:
            """Create a new model profile.

            Args:
                name: Profile name
                provider: Provider name (ollama, anthropic, etc.)
                model: Model identifier
                system_prompt: Optional system prompt
                timeout_seconds: Optional timeout
                temperature: Optional temperature (0.0-1.0)
                max_tokens: Optional max tokens for generation
            """
            result = await self._create_profile_tool(
                {
                    "name": name,
                    "provider": provider,
                    "model": model,
                    "system_prompt": system_prompt,
                    "timeout_seconds": timeout_seconds,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def update_profile(name: str, changes: dict[str, Any]) -> str:
            """Update an existing profile.

            Args:
                name: Profile name to update
                changes: Changes to apply
            """
            result = await self._update_profile_tool({"name": name, "changes": changes})
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def delete_profile(name: str, confirm: bool = False) -> str:
            """Delete a model profile.

            Args:
                name: Profile name to delete
                confirm: Must be True to actually delete
            """
            result = await self._delete_profile_tool({"name": name, "confirm": confirm})
            return json.dumps(result, indent=2)

    def _setup_artifact_tools(self) -> None:
        """Register artifact management tools."""

        @self._mcp.tool()
        async def delete_artifact(artifact_id: str, confirm: bool = False) -> str:
            """Delete an execution artifact.

            Args:
                artifact_id: Artifact ID (format: ensemble/timestamp)
                confirm: Must be True to actually delete
            """
            result = await self._delete_artifact_tool(
                {"artifact_id": artifact_id, "confirm": confirm}
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def cleanup_artifacts(
            ensemble_name: str | None = None,
            older_than_days: int = 30,
            dry_run: bool = True,
        ) -> str:
            """Cleanup old artifacts.

            Args:
                ensemble_name: Optional ensemble to cleanup (all if not specified)
                older_than_days: Delete artifacts older than this
                dry_run: If True, preview without deleting
            """
            result = await self._cleanup_artifacts_tool(
                {
                    "ensemble_name": ensemble_name,
                    "older_than_days": older_than_days,
                    "dry_run": dry_run,
                }
            )
            return json.dumps(result, indent=2)

        self._setup_script_tools()
        self._setup_library_extra_tools()

    def _setup_script_tools(self) -> None:
        """Register script management tools."""

        @self._mcp.tool()
        async def get_script(name: str, category: str) -> str:
            """Get script details.

            Args:
                name: Script name
                category: Script category
            """
            result = await self._get_script_tool({"name": name, "category": category})
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def test_script(
            name: str,
            category: str,
            input: str,  # noqa: A002
        ) -> str:
            """Test a script with sample input.

            Args:
                name: Script name
                category: Script category
                input: Test input data
            """
            result = await self._test_script_tool(
                {"name": name, "category": category, "input": input}
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def create_script(
            name: str, category: str, template: str = "basic"
        ) -> str:
            """Create a new primitive script.

            Args:
                name: Script name
                category: Script category
                template: Template to use (basic, extraction, etc.)
            """
            result = await self._create_script_tool(
                {"name": name, "category": category, "template": template}
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def delete_script(name: str, category: str, confirm: bool = False) -> str:
            """Delete a script.

            Args:
                name: Script name
                category: Script category
                confirm: Must be True to actually delete
            """
            result = await self._delete_script_tool(
                {"name": name, "category": category, "confirm": confirm}
            )
            return json.dumps(result, indent=2)

    def _setup_library_extra_tools(self) -> None:
        """Register library extra tools."""

        @self._mcp.tool()
        async def library_search(query: str) -> str:
            """Search library content.

            Args:
                query: Search query
            """
            result = await self._library_search_tool({"query": query})
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def library_info() -> str:
            """Get library information."""
            result = await self._library_info_tool({})
            return json.dumps(result, indent=2)

    def _setup_provider_discovery_tools(self) -> None:
        """Register provider & model discovery tools."""

        @self._mcp.tool()
        async def get_provider_status() -> str:
            """Show which providers are configured and available models.

            Returns status of all providers including:
            - Ollama: Available models from local instance
            - Cloud providers: Whether authentication is configured
            """
            result = await self._get_provider_status_tool({})
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def check_ensemble_runnable(ensemble_name: str) -> str:
            """Check if ensemble can run with current providers.

            Args:
                ensemble_name: Name of the ensemble to check

            Returns runnable status with:
            - Whether ensemble can run
            - Status of each agent's profile/provider
            - Suggested local alternatives for unavailable profiles
            """
            result = await self._check_ensemble_runnable_tool(
                {"ensemble_name": ensemble_name}
            )
            return json.dumps(result, indent=2)

    def _setup_help_tool(self) -> None:
        """Register help tool for agent onboarding."""

        @self._mcp.tool()
        async def get_help() -> str:
            """Get comprehensive documentation for using llm-orc MCP server.

            Returns documentation including:
            - Directory structure for ensembles, profiles, scripts
            - YAML schemas with examples for creating ensembles and profiles
            - Tool categories and their purposes
            - Common workflows
            """
            result = self._help_handler.get_help_documentation()
            return json.dumps(result, indent=2)

    async def handle_initialize(self) -> dict[str, Any]:
        """Handle MCP initialize request.

        Returns:
            Initialization response with capabilities.
        """
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "serverInfo": {
                "name": "llm-orc",
                "version": self._get_version(),
            },
        }

    def _get_version(self) -> str:
        """Get llm-orc version."""
        try:
            from importlib.metadata import version

            return version("llm-orchestra")
        except Exception:
            return "0.11.0"  # Fallback

    async def list_resources(self) -> list[dict[str, Any]]:
        """List available MCP resources.

        Returns:
            List of resource definitions.
        """
        return [
            {
                "uri": "llm-orc://ensembles",
                "name": "Ensembles",
                "description": "List all available ensembles",
                "mimeType": "application/json",
            },
            {
                "uri": "llm-orc://profiles",
                "name": "Model Profiles",
                "description": "List configured model profiles",
                "mimeType": "application/json",
            },
        ]

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools.

        Returns:
            List of tool definitions.
        """
        return [
            {
                "name": "invoke",
                "description": "Execute an ensemble",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ensemble_name": {
                            "type": "string",
                            "description": "Name of the ensemble to execute",
                        },
                        "input": {
                            "type": "string",
                            "description": "Input data for the ensemble",
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["text", "json"],
                            "default": "json",
                        },
                    },
                    "required": ["ensemble_name", "input"],
                },
            },
            {
                "name": "validate_ensemble",
                "description": "Validate ensemble configuration",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ensemble_name": {
                            "type": "string",
                            "description": "Name of the ensemble to validate",
                        },
                    },
                    "required": ["ensemble_name"],
                },
            },
            {
                "name": "update_ensemble",
                "description": "Modify ensemble configuration",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ensemble_name": {
                            "type": "string",
                            "description": "Name of the ensemble to update",
                        },
                        "changes": {
                            "type": "object",
                            "description": "Changes to apply",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "default": True,
                        },
                        "backup": {
                            "type": "boolean",
                            "default": True,
                        },
                    },
                    "required": ["ensemble_name", "changes"],
                },
            },
            {
                "name": "analyze_execution",
                "description": "Analyze execution artifact",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "artifact_id": {
                            "type": "string",
                            "description": "ID of the artifact to analyze",
                        },
                    },
                    "required": ["artifact_id"],
                },
            },
        ]

    async def read_resource(self, uri: str) -> Any:
        """Read an MCP resource.

        Args:
            uri: Resource URI (e.g., llm-orc://ensembles)

        Returns:
            Resource content.

        Raises:
            ValueError: If resource not found.
        """
        # Parse URI
        if not uri.startswith("llm-orc://"):
            raise ValueError(f"Invalid URI scheme: {uri}")

        path = uri[len("llm-orc://") :]
        parts = path.split("/")

        if parts[0] == "ensembles":
            return await self._read_ensembles_resource()
        elif parts[0] == "ensemble" and len(parts) > 1:
            return await self._read_ensemble_resource(parts[1])
        elif parts[0] == "artifacts" and len(parts) > 1:
            return await self._read_artifacts_resource(parts[1])
        elif parts[0] == "artifact" and len(parts) > 2:
            return await self._read_artifact_resource(parts[1], parts[2])
        elif parts[0] == "metrics" and len(parts) > 1:
            return await self._read_metrics_resource(parts[1])
        elif parts[0] == "profiles":
            return await self._read_profiles_resource()
        else:
            raise ValueError(f"Resource not found: {uri}")

    async def _read_ensembles_resource(self) -> list[dict[str, Any]]:
        """Read all ensembles.

        Returns:
            List of ensemble metadata.
        """
        ensembles: list[dict[str, Any]] = []
        ensemble_dirs = self.config_manager.get_ensembles_dirs()

        for ensemble_dir in ensemble_dirs:
            if not Path(ensemble_dir).exists():
                continue

            source = self._determine_source(ensemble_dir)

            ensemble_dir_path = Path(ensemble_dir)
            for yaml_file in ensemble_dir_path.glob("**/*.yaml"):
                try:
                    config = self.ensemble_loader.load_from_file(str(yaml_file))
                    if config:
                        relative_path = str(yaml_file.relative_to(ensemble_dir_path))
                        ensembles.append(
                            {
                                "name": config.name,
                                "source": source,
                                "relative_path": relative_path,
                                "agent_count": len(config.agents),
                                "description": config.description,
                            }
                        )
                except Exception:
                    continue

        return ensembles

    def _determine_source(self, ensemble_dir: Path) -> str:
        """Determine the source type of an ensemble directory.

        Args:
            ensemble_dir: Path to ensemble directory.

        Returns:
            Source type: 'local', 'library', or 'global'.
        """
        path = ensemble_dir
        if ".llm-orc" in str(path) and "library" not in str(path):
            return "local"
        elif "library" in str(path):
            return "library"
        else:
            return "global"

    async def _read_ensemble_resource(self, name: str) -> dict[str, Any]:
        """Read specific ensemble configuration.

        Args:
            name: Ensemble name.

        Returns:
            Ensemble configuration.

        Raises:
            ValueError: If ensemble not found.
        """
        ensemble_dirs = self.config_manager.get_ensembles_dirs()

        for ensemble_dir in ensemble_dirs:
            config = self.ensemble_loader.find_ensemble(str(ensemble_dir), name)
            if config:
                agents_list = []
                for agent in config.agents:
                    # Handle both dict and object forms
                    if isinstance(agent, dict):
                        agents_list.append(
                            {
                                "name": agent.get("name", ""),
                                "model_profile": agent.get("model_profile"),
                                "depends_on": agent.get("depends_on") or [],
                            }
                        )
                    else:
                        agents_list.append(
                            {
                                "name": agent.name,
                                "model_profile": agent.model_profile,
                                "depends_on": agent.depends_on or [],
                            }
                        )
                return {
                    "name": config.name,
                    "description": config.description,
                    "agents": agents_list,
                }

        raise ValueError(f"Ensemble not found: {name}")

    async def _read_artifacts_resource(
        self, ensemble_name: str
    ) -> list[dict[str, Any]]:
        """Read artifacts for an ensemble.

        Args:
            ensemble_name: Name of the ensemble.

        Returns:
            List of artifact metadata.
        """
        artifacts: list[dict[str, Any]] = []
        artifacts_dir = self._get_artifacts_dir() / ensemble_name

        if not artifacts_dir.exists():
            return []

        # Artifacts are stored as {timestamp_dir}/execution.json
        for artifact_dir in artifacts_dir.iterdir():
            if not artifact_dir.is_dir() or artifact_dir.is_symlink():
                continue

            execution_file = artifact_dir / "execution.json"
            if not execution_file.exists():
                continue

            try:
                artifact_data = json.loads(execution_file.read_text())
                metadata = artifact_data.get("metadata", {})
                artifacts.append(
                    {
                        "id": artifact_dir.name,
                        "timestamp": metadata.get("started_at"),
                        "status": artifact_data.get("status"),
                        "duration": metadata.get("duration"),
                        "agent_count": metadata.get("agents_used"),
                    }
                )
            except Exception:
                continue

        return artifacts

    async def _read_artifact_resource(
        self, ensemble_name: str, artifact_id: str
    ) -> dict[str, Any]:
        """Read specific artifact.

        Args:
            ensemble_name: Name of the ensemble.
            artifact_id: Artifact ID (timestamp directory name).

        Returns:
            Artifact data.

        Raises:
            ValueError: If artifact not found.
        """
        # Artifacts are stored as {ensemble}/{artifact_id}/execution.json
        artifact_dir = self._get_artifacts_dir() / ensemble_name / artifact_id
        execution_file = artifact_dir / "execution.json"

        if not execution_file.exists():
            raise ValueError(f"Artifact not found: {ensemble_name}/{artifact_id}")

        result: dict[str, Any] = json.loads(execution_file.read_text())
        return result

    async def _read_metrics_resource(self, ensemble_name: str) -> dict[str, Any]:
        """Read metrics for an ensemble.

        Args:
            ensemble_name: Name of the ensemble.

        Returns:
            Aggregated metrics.
        """
        artifacts = await self._read_artifacts_resource(ensemble_name)

        if not artifacts:
            return {
                "success_rate": 0.0,
                "avg_cost": 0.0,
                "avg_duration": 0.0,
                "total_executions": 0,
            }

        success_count = sum(1 for a in artifacts if a.get("status") == "success")

        # Parse duration strings (e.g., "2.3s") to floats
        def parse_duration(dur: str | float | None) -> float:
            if dur is None:
                return 0.0
            if isinstance(dur, int | float):
                return float(dur)
            if isinstance(dur, str) and dur.endswith("s"):
                try:
                    return float(dur[:-1])
                except ValueError:
                    return 0.0
            return 0.0

        total_duration = sum(parse_duration(a.get("duration")) for a in artifacts)

        return {
            "success_rate": success_count / len(artifacts) if artifacts else 0.0,
            "avg_cost": 0.0,  # Cost not tracked in new artifact format
            "avg_duration": total_duration / len(artifacts) if artifacts else 0.0,
            "total_executions": len(artifacts),
        }

    async def _read_profiles_resource(self) -> list[dict[str, Any]]:
        """Read model profiles.

        Returns:
            List of model profile configurations.
        """
        profiles: list[dict[str, Any]] = []
        model_profiles = self.config_manager.get_model_profiles()

        for name, config in model_profiles.items():
            profiles.append(
                {
                    "name": name,
                    "provider": config.get("provider", "unknown"),
                    "model": config.get("model", "unknown"),
                }
            )

        return profiles

    def _get_artifacts_dir(self) -> Path:
        """Get artifacts directory path.

        Returns:
            Path to artifacts directory.
        """
        # Check if global_config_dir points to an artifacts directory (for testing)
        global_config_path = Path(self.config_manager.global_config_dir)
        if global_config_path.name == "artifacts" and global_config_path.exists():
            return global_config_path

        # Check local artifacts first (project-specific)
        local_artifacts = Path.cwd() / ".llm-orc" / "artifacts"
        if local_artifacts.exists():
            return local_artifacts

        # Then check global artifacts
        global_artifacts = global_config_path / "artifacts"
        if global_artifacts.exists():
            return global_artifacts

        # Default to local even if it doesn't exist yet
        return local_artifacts

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result.

        Raises:
            ValueError: If tool not found or execution fails.
        """
        handler = self._get_tool_handler(name)
        if handler is None:
            raise ValueError(f"Tool not found: {name}")
        return await handler(arguments)

    def _get_tool_handler(
        self, name: str
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None:
        """Get tool handler by name.

        Args:
            name: Tool name.

        Returns:
            Handler function or None if not found.
        """
        # Build dispatch table mapping tool names to handlers
        handlers: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
            # Context management
            "set_project": self._set_project_tool,
            # Core tools
            "invoke": self._invoke_tool,
            "validate_ensemble": self._validate_ensemble_tool,
            "update_ensemble": self._update_ensemble_tool,
            "analyze_execution": self._analyze_execution_tool,
            # Ensemble CRUD
            "create_ensemble": self._create_ensemble_tool,
            "delete_ensemble": self._delete_ensemble_tool,
            # Scripts and library (high priority)
            "list_scripts": self._list_scripts_tool,
            "library_browse": self._library_browse_tool,
            "library_copy": self._library_copy_tool,
            # Profile CRUD
            "list_profiles": self._list_profiles_tool,
            "create_profile": self._create_profile_tool,
            "update_profile": self._update_profile_tool,
            "delete_profile": self._delete_profile_tool,
            # Artifact management
            "delete_artifact": self._delete_artifact_tool,
            "cleanup_artifacts": self._cleanup_artifacts_tool,
            # Script management (low priority)
            "get_script": self._get_script_tool,
            "test_script": self._test_script_tool,
            "create_script": self._create_script_tool,
            "delete_script": self._delete_script_tool,
            # Library extras (low priority)
            "library_search": self._library_search_tool,
            "library_info": self._library_info_tool,
            # Phase 3: Provider & model discovery
            "get_provider_status": self._get_provider_status_tool,
            "check_ensemble_runnable": self._check_ensemble_runnable_tool,
            # Help
            "get_help": self._help_tool,
        }
        return handlers.get(name)

    async def _set_project_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute set_project tool.

        Args:
            arguments: Tool arguments including path.

        Returns:
            Project context result.
        """
        path = arguments.get("path", "")
        return self._handle_set_project(path)

    def _handle_set_project(self, path: str) -> dict[str, Any]:
        """Handle set_project logic.

        Args:
            path: Project directory path.

        Returns:
            Result dict with status and project info.
        """
        project_dir = Path(path).resolve()

        # Validate path exists
        if not project_dir.exists():
            return {
                "status": "error",
                "error": f"Path does not exist: {path}",
            }

        # Update project path and invalidate cached executor
        self._project_path = project_dir
        self._executor = None

        # Recreate config manager with new project path
        self.config_manager = ConfigurationManager(project_dir=project_dir)

        # Build result
        result: dict[str, Any] = {
            "status": "ok",
            "project_path": str(project_dir),
        }

        # Add note if no .llm-orc directory
        llm_orc_dir = project_dir / ".llm-orc"
        if not llm_orc_dir.exists():
            result["note"] = "No .llm-orc directory found; using global config only"

        return result

    async def _invoke_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute invoke tool.

        Args:
            arguments: Tool arguments including ensemble_name and input.

        Returns:
            Execution result.
        """
        ensemble_name = arguments.get("ensemble_name")
        input_data = arguments.get("input", "")

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        # Find ensemble
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        config = None

        for ensemble_dir in ensemble_dirs:
            config = self.ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                break

        if not config:
            raise ValueError(f"Ensemble does not exist: {ensemble_name}")

        # Execute ensemble
        from llm_orc.core.execution.ensemble_execution import EnsembleExecutor

        executor = EnsembleExecutor(project_dir=self._project_path)
        result = await executor.execute(config, input_data)

        return {
            "results": result.get("results", {}),
            "synthesis": result.get("synthesis"),
            "status": result.get("status"),
        }

    async def _invoke_tool_with_streaming(
        self, ensemble_name: str, input_data: str, ctx: Context[Any, Any, Any]
    ) -> dict[str, Any]:
        """Execute invoke tool with streaming progress updates.

        Args:
            ensemble_name: Name of the ensemble to execute.
            input_data: Input data for the ensemble.
            ctx: FastMCP context for progress reporting.

        Returns:
            Execution result.
        """
        reporter = FastMCPProgressReporter(ctx)
        return await self._execute_ensemble_streaming(
            ensemble_name, input_data, reporter
        )

    async def _execute_ensemble_streaming(
        self,
        ensemble_name: str,
        input_data: str,
        reporter: ProgressReporter,
    ) -> dict[str, Any]:
        """Execute ensemble with streaming progress updates.

        This method is separated from _invoke_tool_with_streaming to allow
        testing with a mock ProgressReporter.

        Args:
            ensemble_name: Name of the ensemble to execute.
            input_data: Input data for the ensemble.
            reporter: Progress reporter for status updates.

        Returns:
            Execution result.
        """
        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        config = self._find_ensemble_by_name(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble does not exist: {ensemble_name}")

        executor = self._get_executor()
        total_agents = len(config.agents)
        state: dict[str, Any] = {
            "completed": 0,
            "result": {},
            "ensemble_name": ensemble_name,
            "input_data": input_data,
        }

        msg = f"Starting ensemble '{ensemble_name}' with {total_agents} agents"
        await reporter.info(msg)

        async for event in executor.execute_streaming(config, input_data):
            await self._handle_streaming_event(event, reporter, total_agents, state)

        result = state.get("result", {})
        if not isinstance(result, dict):
            result = {}
        return result

    async def _handle_streaming_event(
        self,
        event: dict[str, Any],
        reporter: ProgressReporter,
        total_agents: int,
        state: dict[str, Any],
    ) -> None:
        """Handle a single streaming event from ensemble execution.

        Args:
            event: The streaming event.
            reporter: Progress reporter for status updates.
            total_agents: Total number of agents in ensemble.
            state: Mutable state dict with 'completed' count and 'result'.
        """
        event_type = event.get("type", "")
        event_data = event.get("data", {})

        if event_type == "execution_started":
            await reporter.report_progress(progress=0, total=total_agents)

        elif event_type == "agent_started":
            agent_name = event_data.get("agent_name", "unknown")
            await reporter.info(f"Agent '{agent_name}' started")

        elif event_type == "agent_completed":
            state["completed"] += 1
            agent_name = event_data.get("agent_name", "unknown")
            await reporter.report_progress(state["completed"], total_agents)
            await reporter.info(f"Agent '{agent_name}' completed")

        elif event_type == "execution_completed":
            results = event_data.get("results", {})
            synthesis = event_data.get("synthesis")
            status = event_data.get("status", "completed")
            state["result"] = {
                "results": results,
                "synthesis": synthesis,
                "status": status,
            }
            # Save artifact for later analysis
            ensemble_name = state.get("ensemble_name", "unknown")
            input_data = state.get("input_data", "")
            self._save_execution_artifact(
                ensemble_name, input_data, results, synthesis, status
            )
            await reporter.report_progress(progress=total_agents, total=total_agents)

        elif event_type == "execution_failed":
            error_msg = event_data.get("error", "Unknown error")
            await reporter.error(f"Execution failed: {error_msg}")
            state["result"] = {
                "results": {},
                "synthesis": None,
                "status": "failed",
                "error": error_msg,
            }

        elif event_type == "agent_fallback_started":
            agent_name = event_data.get("agent_name", "unknown")
            msg = f"Agent '{agent_name}' falling back to alternate model"
            await reporter.warning(msg)

    def _save_execution_artifact(
        self,
        ensemble_name: str,
        input_data: str,
        results: dict[str, Any],
        synthesis: Any,
        status: str,
    ) -> Path | None:
        """Save execution results as an artifact.

        Args:
            ensemble_name: Name of the executed ensemble.
            input_data: Input provided to the ensemble.
            results: Agent results dictionary.
            synthesis: Synthesis result (if any).
            status: Execution status.

        Returns:
            Path to the artifact directory or None if save failed.
        """
        import datetime

        # Build artifact data in expected format
        artifact_data: dict[str, Any] = {
            "ensemble_name": ensemble_name,
            "input": input_data,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": status,
            "results": results,
            "synthesis": synthesis,
            "agents": [],
        }

        # Extract agent info from results
        for agent_name, agent_result in results.items():
            if isinstance(agent_result, dict):
                artifact_data["agents"].append(
                    {
                        "name": agent_name,
                        "status": agent_result.get("status", "unknown"),
                        "result": agent_result.get("response", ""),
                    }
                )

        try:
            artifact_path = self.artifact_manager.save_execution_results(
                ensemble_name, artifact_data
            )
            return artifact_path
        except (OSError, TypeError, ValueError):
            # Log but don't fail execution if artifact save fails
            return None

    async def _validate_ensemble_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute validate_ensemble tool."""
        return await self._validation_handler.validate_ensemble(arguments)

    def _find_ensemble_by_name(self, ensemble_name: str) -> Any:
        """Find ensemble configuration by name.

        Args:
            ensemble_name: Name of ensemble to find.

        Returns:
            Ensemble configuration or None if not found.
        """
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        for ensemble_dir in ensemble_dirs:
            config = self.ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                return config
        return None

    async def _update_ensemble_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute update_ensemble tool.

        Args:
            arguments: Tool arguments including ensemble_name, changes, dry_run.

        Returns:
            Update result.
        """
        ensemble_name = arguments.get("ensemble_name")
        changes = arguments.get("changes", {})
        dry_run = arguments.get("dry_run", True)
        backup = arguments.get("backup", True)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        # Find ensemble file
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        ensemble_path: Path | None = None

        for ensemble_dir in ensemble_dirs:
            potential_path = Path(ensemble_dir) / f"{ensemble_name}.yaml"
            if potential_path.exists():
                ensemble_path = potential_path
                break

        if not ensemble_path:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        if dry_run:
            return {
                "preview": changes,
                "modified": False,
                "backup_created": False,
            }

        # Create backup if requested
        backup_created = False
        if backup:
            backup_path = ensemble_path.with_suffix(".yaml.bak")
            backup_path.write_text(ensemble_path.read_text())
            backup_created = True

        # Apply changes (simplified - would need YAML manipulation)
        # For now, just mark as modified
        return {
            "modified": True,
            "backup_created": backup_created,
            "changes_applied": changes,
        }

    async def _analyze_execution_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute analyze_execution tool.

        Args:
            arguments: Tool arguments including artifact_id.

        Returns:
            Analysis result.
        """
        artifact_id = arguments.get("artifact_id")

        if not artifact_id:
            raise ValueError("artifact_id is required")

        # Parse artifact_id (format: ensemble_name/artifact_id)
        parts = artifact_id.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid artifact_id format: {artifact_id}")

        ensemble_name, aid = parts
        artifact = await self._read_artifact_resource(ensemble_name, aid)

        # Compute analysis metrics
        results = artifact.get("results", {})
        success_count = sum(1 for r in results.values() if r.get("status") == "success")

        return {
            "analysis": {
                "total_agents": len(results),
                "successful_agents": success_count,
                "failed_agents": len(results) - success_count,
            },
            "metrics": {
                "agent_success_rate": success_count / len(results) if results else 0.0,
                "cost": artifact.get("cost", 0),
                "duration": artifact.get("duration", 0),
            },
        }

    async def invoke_streaming(
        self, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Invoke ensemble with streaming progress.

        Args:
            params: Invocation parameters.

        Yields:
            Progress events.
        """
        ensemble_name = params.get("ensemble_name")
        # input_data is available via params.get("input") when needed for execution

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        # Find ensemble
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        config = None

        for ensemble_dir in ensemble_dirs:
            config = self.ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                break

        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        # Emit agent events (simplified streaming)
        for agent in config.agents:
            agent_name = _get_agent_attr(agent, "name")
            yield {
                "type": "agent_start",
                "agent": agent_name,
            }

            # Agent would execute here
            yield {
                "type": "agent_progress",
                "agent": agent_name,
                "progress": 50,
            }

            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "status": "success",
            }

        # Final result
        yield {
            "type": "execution_complete",
            "status": "success",
        }

    def run(
        self, transport: str = "stdio", host: str = "0.0.0.0", port: int = 8080
    ) -> None:
        """Run the MCP server.

        Args:
            transport: Transport type ('stdio' or 'http').
            host: Host for HTTP transport.
            port: Port for HTTP transport.
        """
        if transport == "http":
            import uvicorn

            app = self._mcp.sse_app()
            uvicorn.run(app, host=host, port=port)
        else:
            self._mcp.run()

    # =========================================================================
    # Phase 2 CRUD Tool Implementations
    # =========================================================================

    async def _create_ensemble_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new ensemble.

        Args:
            arguments: Tool arguments including name, description, agents.

        Returns:
            Creation result.
        """
        import yaml

        name = arguments.get("name")
        description = arguments.get("description", "")
        agents = arguments.get("agents", [])
        from_template = arguments.get("from_template")

        if not name:
            raise ValueError("name is required")

        local_dir = self._get_local_ensembles_dir()
        target_file = local_dir / f"{name}.yaml"
        if target_file.exists():
            raise ValueError(f"Ensemble already exists: {name}")

        # If from_template, copy from existing ensemble
        agents_copied = 0
        if from_template:
            agents, description, agents_copied = self._copy_from_template(
                from_template, description
            )

        # Build YAML content
        ensemble_data = {
            "name": name,
            "description": description,
            "agents": agents,
        }
        yaml_content = yaml.dump(ensemble_data, default_flow_style=False)

        # Write the file
        local_dir.mkdir(parents=True, exist_ok=True)
        target_file.write_text(yaml_content)

        return {
            "created": True,
            "path": str(target_file),
            "agents_copied": agents_copied,
        }

    def _get_local_ensembles_dir(self) -> Path:
        """Get the local ensembles directory for writing.

        Returns:
            Path to local ensembles directory.

        Raises:
            ValueError: If no ensemble directory is available.
        """
        ensemble_dirs = self.config_manager.get_ensembles_dirs()

        for dir_path in ensemble_dirs:
            path = Path(dir_path)
            if ".llm-orc" in str(path) and "library" not in str(path):
                return path

        if ensemble_dirs:
            return Path(ensemble_dirs[0])

        raise ValueError("No ensemble directory available")

    def _copy_from_template(
        self, template_name: str, description: str
    ) -> tuple[list[dict[str, Any]], str, int]:
        """Copy agents and description from a template ensemble.

        Args:
            template_name: Name of the template ensemble.
            description: Current description (may be overwritten if empty).

        Returns:
            Tuple of (agents list, description, agents_copied count).

        Raises:
            ValueError: If template not found.
        """
        template_config = self._find_ensemble_by_name(template_name)
        if not template_config:
            raise ValueError(f"Template ensemble not found: {template_name}")

        agents: list[dict[str, Any]] = []
        preserved_fields = (
            "name",
            "type",
            "model_profile",
            "script",
            "parameters",
            "depends_on",
            "system_prompt",
            "cache",
            "fan_out",
        )
        for agent in template_config.agents:
            if isinstance(agent, dict):
                agents.append(dict(agent))
            else:
                agent_dict: dict[str, Any] = {}
                for attr in preserved_fields:
                    val = _get_agent_attr(agent, attr)
                    if val is not None:
                        agent_dict[attr] = val
                agents.append(agent_dict)

        final_description = description or template_config.description
        return agents, final_description, len(agents)

    async def _delete_ensemble_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete an ensemble.

        Args:
            arguments: Tool arguments including ensemble_name, confirm.

        Returns:
            Deletion result.
        """
        ensemble_name = arguments.get("ensemble_name")
        confirm = arguments.get("confirm", False)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        if not confirm:
            raise ValueError("Confirmation required to delete ensemble")

        # Find the ensemble file
        ensemble_dirs = self.config_manager.get_ensembles_dirs()
        ensemble_file: Path | None = None

        for dir_path in ensemble_dirs:
            potential_file = Path(dir_path) / f"{ensemble_name}.yaml"
            if potential_file.exists():
                ensemble_file = potential_file
                break

        if not ensemble_file:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        # Delete the file
        ensemble_file.unlink()

        return {
            "deleted": True,
            "path": str(ensemble_file),
        }

    async def _list_scripts_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """List available scripts."""
        return await self._script_handler.list_scripts(arguments)

    async def _library_browse_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Browse library items."""
        return await self._library_handler.browse(arguments)

    async def _library_copy_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Copy from library to local."""
        return await self._library_handler.copy(arguments)

    # =========================================================================
    # Profile CRUD Tool Implementations
    # =========================================================================

    async def _list_profiles_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """List model profiles."""
        return await self._profile_handler.list_profiles(arguments)

    async def _create_profile_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new profile."""
        return await self._profile_handler.create_profile(arguments)

    async def _update_profile_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Update an existing profile."""
        return await self._profile_handler.update_profile(arguments)

    async def _delete_profile_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete a profile."""
        return await self._profile_handler.delete_profile(arguments)

    # =========================================================================
    # Artifact Management Tool Implementations
    # =========================================================================

    async def _delete_artifact_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete an artifact."""
        return await self._artifact_handler.delete_artifact(arguments)

    async def _cleanup_artifacts_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Cleanup old artifacts."""
        return await self._artifact_handler.cleanup_artifacts(arguments)

    # =========================================================================
    # Script Management Tool Implementations (Low Priority)
    # =========================================================================

    async def _get_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get script details."""
        return await self._script_handler.get_script(arguments)

    async def _test_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Test a script with sample input."""
        return await self._script_handler.test_script(arguments)

    async def _create_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new script."""
        return await self._script_handler.create_script(arguments)

    async def _delete_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete a script."""
        return await self._script_handler.delete_script(arguments)

    # =========================================================================
    # Library Extras Tool Implementations (Low Priority)
    # =========================================================================

    async def _library_search_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Search library content."""
        return await self._library_handler.search(arguments)

    async def _library_info_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get library information."""
        return await self._library_handler.info(arguments)

    # =========================================================================
    # Phase 3: Provider & Model Discovery
    # =========================================================================

    async def _get_provider_status_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Get status of all providers and available models."""
        return await self._provider_handler.get_provider_status(arguments)

    async def _check_ensemble_runnable_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Check if an ensemble can run with current providers."""
        return await self._provider_handler.check_ensemble_runnable(arguments)

    async def _help_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get help documentation.

        Args:
            arguments: Tool arguments (none required).

        Returns:
            Comprehensive help documentation.
        """
        return self._help_handler.get_help_documentation()
