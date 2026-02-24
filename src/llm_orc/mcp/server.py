"""MCP Server implementation using FastMCP SDK.

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
from llm_orc.services.orchestra_service import OrchestraService

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


class MCPServer:
    """MCP Server using FastMCP SDK.

    Exposes all llm-orc ensembles as MCP resources and provides
    tools for ensemble management and execution.
    """

    def __init__(
        self,
        config_manager: ConfigurationManager | None = None,
        executor: EnsembleExecutor | None = None,
        service: OrchestraService | None = None,
    ) -> None:
        """Initialize MCP server.

        Args:
            config_manager: Configuration manager instance. Creates default if None.
            executor: Ensemble executor instance. Creates default if None.
            service: OrchestraService instance. Creates default if None.
        """
        if service is not None:
            self._service = service
        else:
            self._service = OrchestraService(
                config_manager=config_manager,
                executor=executor,
            )

        self._mcp = FastMCP("llm-orc")
        self._setup_resources()
        self._setup_tools()

    @property
    def project_path(self) -> Path | None:
        """Get the current project path."""
        return self._service.project_path

    @property
    def config_manager(self) -> ConfigurationManager:
        """Get the configuration manager."""
        return self._service.config_manager

    @property
    def ensemble_loader(self) -> EnsembleLoader:
        """Get the ensemble loader."""
        return self._service.ensemble_loader

    @property
    def artifact_manager(self) -> ArtifactManager:
        """Get the artifact manager."""
        return self._service.artifact_manager

    # Expose handler attributes for tests that access them directly
    @property
    def _library_handler(self) -> Any:
        return self._service._library_handler

    @property
    def _script_handler(self) -> Any:
        return self._service._script_handler

    @property
    def _artifact_handler(self) -> Any:
        return self._service._artifact_handler

    @property
    def _provider_handler(self) -> Any:
        return self._service._provider_handler

    @property
    def _profile_handler(self) -> Any:
        return self._service._profile_handler

    @property
    def _ensemble_crud_handler(self) -> Any:
        return self._service._ensemble_crud_handler

    @property
    def _validation_handler(self) -> Any:
        return self._service._validation_handler

    @property
    def _help_handler(self) -> Any:
        return self._service._help_handler

    @property
    def _execution_handler(self) -> Any:
        return self._service._execution_handler

    @property
    def _promotion_handler(self) -> Any:
        return self._service._promotion_handler

    @property
    def _executor(self) -> Any:
        return self._service._executor

    def _setup_resources(self) -> None:
        """Register MCP resources with FastMCP."""

        @self._mcp.resource("llm-orc://ensembles")
        async def get_ensembles() -> str:
            """List all available ensembles."""
            result = await self._service.read_ensembles()
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://profiles")
        async def get_profiles() -> str:
            """List configured model profiles."""
            result = await self._service.read_profiles()
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://ensemble/{name}")
        async def get_ensemble(name: str) -> str:
            """Get specific ensemble configuration."""
            result = await self._service.read_ensemble(name)
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://artifacts/{ensemble}")
        async def get_artifacts(ensemble: str) -> str:
            """List execution artifacts for an ensemble."""
            result = await self._service.read_artifacts(ensemble)
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://metrics/{ensemble}")
        async def get_metrics(ensemble: str) -> str:
            """Get aggregated metrics for an ensemble."""
            result = await self._service.read_metrics(ensemble)
            return json.dumps(result, indent=2)

        @self._mcp.resource("llm-orc://artifact/{ensemble}/{artifact_id}")
        async def get_artifact(ensemble: str, artifact_id: str) -> str:
            """Get individual artifact details."""
            result = await self._service.read_artifact(ensemble, artifact_id)
            return json.dumps(result, indent=2)

    def _setup_tools(self) -> None:
        """Register MCP tools with FastMCP."""
        self._setup_context_tools()
        self._setup_core_tools()
        self._setup_crud_tools()
        self._setup_provider_discovery_tools()
        self._setup_promotion_tools()
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
            result = server._set_project_tool_sync(path)
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
            result = await self._service.validate_ensemble(
                {
                    "ensemble_name": ensemble_name,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def list_ensembles() -> str:
            """List all available ensembles with their metadata."""
            result = await self._service.read_ensembles()
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
            result = await self._service.update_ensemble(
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
            result = await self._service.analyze_execution(
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
            result = await self._service.create_ensemble(
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
            result = await self._service.delete_ensemble(
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
            result = await self._service.list_scripts({"category": category})
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
            result = await self._service.library_browse(
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
            result = await self._service.library_copy(
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
            result = await self._service.list_profiles_tool({"provider": provider})
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
            result = await self._service.create_profile(
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
            result = await self._service.update_profile(
                {"name": name, "changes": changes}
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def delete_profile(name: str, confirm: bool = False) -> str:
            """Delete a model profile.

            Args:
                name: Profile name to delete
                confirm: Must be True to actually delete
            """
            result = await self._service.delete_profile(
                {"name": name, "confirm": confirm}
            )
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
            result = await self._service.delete_artifact(
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
            result = await self._service.cleanup_artifacts(
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
            result = await self._service.get_script(
                {"name": name, "category": category}
            )
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
            result = await self._service.test_script(
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
            result = await self._service.create_script(
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
            result = await self._service.delete_script(
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
            result = await self._service.library_search({"query": query})
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def library_info() -> str:
            """Get library information."""
            result = await self._service.library_info({})
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
            result = await self._service.get_provider_status({})
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
            result = await self._service.check_ensemble_runnable(
                {"ensemble_name": ensemble_name}
            )
            return json.dumps(result, indent=2)

    def _setup_promotion_tools(self) -> None:
        """Register promotion and demotion tools."""

        @self._mcp.tool()
        async def promote_ensemble(
            ensemble_name: str,
            destination: str,
            include_profiles: bool = True,
            dry_run: bool = True,
            overwrite: bool = False,
        ) -> str:
            """Promote an ensemble from local to global or
            library tier, including profile dependencies.

            Args:
                ensemble_name: Name of the ensemble to promote
                destination: Target tier: 'global' or 'library'
                include_profiles: Copy referenced profiles missing at destination
                dry_run: Preview what would be copied without actually copying
                overwrite: Overwrite if ensemble already exists at destination
            """
            result = await self._service.promote_ensemble(
                {
                    "ensemble_name": ensemble_name,
                    "destination": destination,
                    "include_profiles": include_profiles,
                    "dry_run": dry_run,
                    "overwrite": overwrite,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def list_dependencies(ensemble_name: str) -> str:
            """Show all external dependencies an ensemble
            requires: profiles, models, and providers.

            Args:
                ensemble_name: Name of the ensemble to inspect
            """
            result = await self._service.list_dependencies(
                {"ensemble_name": ensemble_name}
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def check_promotion_readiness(
            ensemble_name: str, destination: str
        ) -> str:
            """Assess whether an ensemble can be promoted
            to a target tier, and what's missing.

            Args:
                ensemble_name: Name of the ensemble to check
                destination: Target tier: 'global' or 'library'
            """
            result = await self._service.check_promotion_readiness(
                {
                    "ensemble_name": ensemble_name,
                    "destination": destination,
                }
            )
            return json.dumps(result, indent=2)

        @self._mcp.tool()
        async def demote_ensemble(
            ensemble_name: str,
            tier: str,
            remove_orphaned_profiles: bool = False,
            confirm: bool = False,
        ) -> str:
            """Remove an ensemble from a higher tier
            (does not delete lower-tier copies).

            Args:
                ensemble_name: Name of the ensemble to demote
                tier: Tier to remove from: 'global' or 'library'
                remove_orphaned_profiles: Also remove profiles
                    no longer referenced by any ensemble
                confirm: Must be True to actually delete
            """
            result = await self._service.demote_ensemble(
                {
                    "ensemble_name": ensemble_name,
                    "tier": tier,
                    "remove_orphaned_profiles": remove_orphaned_profiles,
                    "confirm": confirm,
                }
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
            result = self._service.get_help_documentation()
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
        return await self._service.read_resource(uri)

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
        svc = self._service
        handlers: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
            # Context management
            "set_project": self._set_project_tool,
            # Core tools
            "invoke": svc.invoke,
            "validate_ensemble": svc.validate_ensemble,
            "update_ensemble": svc.update_ensemble,
            "analyze_execution": svc.analyze_execution,
            # Ensemble CRUD
            "create_ensemble": svc.create_ensemble,
            "delete_ensemble": svc.delete_ensemble,
            # Scripts and library
            "list_scripts": svc.list_scripts,
            "library_browse": svc.library_browse,
            "library_copy": svc.library_copy,
            # Profile CRUD
            "list_profiles": svc.list_profiles_tool,
            "create_profile": svc.create_profile,
            "update_profile": svc.update_profile,
            "delete_profile": svc.delete_profile,
            # Artifact management
            "delete_artifact": svc.delete_artifact,
            "cleanup_artifacts": svc.cleanup_artifacts,
            # Script management
            "get_script": svc.get_script,
            "test_script": svc.test_script,
            "create_script": svc.create_script,
            "delete_script": svc.delete_script,
            # Library extras
            "library_search": svc.library_search,
            "library_info": svc.library_info,
            # Provider & model discovery
            "get_provider_status": svc.get_provider_status,
            "check_ensemble_runnable": svc.check_ensemble_runnable,
            # Promotion
            "promote_ensemble": svc.promote_ensemble,
            "list_dependencies": svc.list_dependencies,
            "check_promotion_readiness": svc.check_promotion_readiness,
            "demote_ensemble": svc.demote_ensemble,
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
        return await self._service.handle_set_project_async(path)

    def _set_project_tool_sync(self, path: str) -> dict[str, Any]:
        """Thin synchronous wrapper for set_project (used in FastMCP closure)."""
        return self._service.handle_set_project(path)

    def _handle_set_project(self, path: str) -> dict[str, Any]:
        """Handle set_project logic (kept for backward compatibility in tests).

        Args:
            path: Project directory path.

        Returns:
            Result dict with status and project info.
        """
        return self._service.handle_set_project(path)

    async def _invoke_tool_with_streaming(
        self, ensemble_name: str, input_data: str, ctx: Context[Any, Any, Any]
    ) -> dict[str, Any]:
        """Execute invoke tool with streaming progress updates.

        Bridges FastMCP Context to ProgressReporter.
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
        """Execute streaming (thin wrapper for tests)."""
        return await self._service.execute_streaming(
            ensemble_name, input_data, reporter
        )

    async def _handle_streaming_event(
        self,
        event: dict[str, Any],
        reporter: ProgressReporter,
        total_agents: int,
        state: dict[str, Any],
    ) -> None:
        """Handle streaming event (thin wrapper for tests)."""
        await self._service.handle_streaming_event(event, reporter, total_agents, state)

    async def invoke_streaming(
        self, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Invoke ensemble with streaming progress."""
        async for event in self._service.invoke_streaming(params):
            yield event

    async def _help_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get help documentation.

        Args:
            arguments: Tool arguments (none required).

        Returns:
            Comprehensive help documentation.
        """
        return self._service.get_help_documentation()

    # === Backward-compat delegation methods used by tests ===

    async def _get_provider_status_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Get provider status (backward-compat wrapper for tests)."""
        return await self._service.get_provider_status(arguments)

    async def _check_ensemble_runnable_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Check ensemble runnable (backward-compat wrapper for tests)."""
        return await self._service.check_ensemble_runnable(arguments)

    async def _read_ensembles_resource(self) -> list[dict[str, Any]]:
        """Read all ensembles (backward-compat wrapper for tests)."""
        return await self._service.read_ensembles()

    async def _validate_ensemble_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute validate_ensemble tool (backward-compat wrapper)."""
        return await self._service.validate_ensemble(arguments)

    def _get_executor(self) -> EnsembleExecutor:
        """Get executor (backward-compat wrapper for tests)."""
        return self._service._get_executor()

    def _get_local_ensembles_dir(self) -> Path:
        """Get local ensembles dir (backward-compat wrapper for tests)."""
        return self._service.get_local_ensembles_dir()

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
