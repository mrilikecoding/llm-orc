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

    # Optional test overrides for directories
    _test_library_dir: Path | None = None
    _test_scripts_dir: Path | None = None
    _test_artifacts_base: Path | None = None
    _test_ollama_status: dict[str, Any] | None = None

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

            self._executor = EnsembleExecutor()
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
        ) -> str:
            """Create a new model profile.

            Args:
                name: Profile name
                provider: Provider name (ollama, anthropic, etc.)
                model: Model identifier
                system_prompt: Optional system prompt
                timeout_seconds: Optional timeout
            """
            result = await self._create_profile_tool(
                {
                    "name": name,
                    "provider": provider,
                    "model": model,
                    "system_prompt": system_prompt,
                    "timeout_seconds": timeout_seconds,
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
            result = self._get_help_documentation()
            return json.dumps(result, indent=2)

    def _get_help_documentation(self) -> dict[str, Any]:
        """Build comprehensive help documentation."""
        return {
            "overview": (
                "llm-orc orchestrates multi-agent LLM ensembles. "
                "Use these tools to discover, run, and manage ensembles."
            ),
            "directory_structure": self._get_directory_structure_help(),
            "schemas": self._get_schema_help(),
            "tools": self._get_tools_help(),
            "workflows": self._get_workflow_help(),
        }

    def _get_directory_structure_help(self) -> dict[str, Any]:
        """Get directory structure documentation."""
        return {
            "local": {
                "path": ".llm-orc/",
                "description": "Project-specific configuration (highest priority)",
                "subdirs": {
                    "ensembles/": "Project ensembles (YAML files)",
                    "profiles/": "Model profiles",
                    "scripts/": "Primitive scripts by category",
                    "artifacts/": "Execution results (auto-generated)",
                },
            },
            "global": {
                "path": "~/.config/llm-orc/",
                "description": "User-wide configuration",
                "subdirs": {
                    "ensembles/": "Global ensembles",
                    "profiles/": "Global model profiles",
                    "credentials.yaml": "API keys (encrypted)",
                },
            },
            "priority": "Local config overrides global config",
        }

    def _get_schema_help(self) -> dict[str, Any]:
        """Get YAML schema documentation."""
        return {
            "ensemble": {
                "description": "Multi-agent workflow definition",
                "required_fields": ["name", "agents"],
                "example": {
                    "name": "code-review",
                    "description": "Multi-perspective code review",
                    "agents": [
                        {
                            "name": "security-reviewer",
                            "model_profile": "ollama-llama3",
                            "system_prompt": "Focus on security issues...",
                        },
                        {
                            "name": "synthesizer",
                            "model_profile": "ollama-llama3",
                            "depends_on": ["security-reviewer"],
                            "system_prompt": "Synthesize the analysis...",
                        },
                    ],
                },
            },
            "profile": {
                "description": "Model configuration shortcut",
                "required_fields": ["provider", "model"],
                "example": {
                    "name": "ollama-llama3",
                    "provider": "ollama",
                    "model": "llama3:latest",
                    "system_prompt": "You are a helpful assistant.",
                    "timeout_seconds": 60,
                },
                "providers": ["ollama", "anthropic", "anthropic-claude-pro-max"],
            },
            "agent": {
                "description": "Agent within an ensemble",
                "required_fields": ["name", "model_profile"],
                "optional_fields": [
                    "system_prompt",
                    "depends_on",
                    "output_format",
                    "timeout_seconds",
                ],
            },
        }

    def _get_tools_help(self) -> dict[str, str]:
        """Get tool category documentation."""
        return {
            "context_management": (
                "set_project - Set active project directory for all operations"
            ),
            "core_execution": (
                "invoke, list_ensembles, validate_ensemble, "
                "update_ensemble, analyze_execution"
            ),
            "provider_discovery": (
                "get_provider_status (check available models), "
                "check_ensemble_runnable (verify ensemble can run)"
            ),
            "ensemble_crud": "create_ensemble, delete_ensemble",
            "profile_crud": (
                "list_profiles, create_profile, update_profile, delete_profile"
            ),
            "script_management": (
                "list_scripts, get_script, test_script, create_script, delete_script"
            ),
            "library": ("library_browse, library_copy, library_search, library_info"),
            "artifacts": "delete_artifact, cleanup_artifacts",
        }

    def _get_workflow_help(self) -> dict[str, list[str]]:
        """Get common workflow documentation."""
        return {
            "start_session": [
                "1. set_project - Point to project directory (optional)",
                "2. get_provider_status - See available models",
                "3. list_ensembles - Find available ensembles",
            ],
            "discover_and_run": [
                "1. set_project - Set project context (if not done)",
                "2. list_ensembles - Find available ensembles",
                "3. check_ensemble_runnable - Verify it can run",
                "4. invoke - Execute the ensemble",
            ],
            "adapt_from_library": [
                "1. set_project - Set target project",
                "2. library_search - Find relevant ensembles",
                "3. library_copy - Copy to local project",
                "4. update_ensemble - Adapt for local models",
                "5. invoke - Run the adapted ensemble",
            ],
            "create_new_ensemble": [
                "1. set_project - Set project context",
                "2. list_profiles - See available model profiles",
                "3. create_ensemble - Create with agents",
                "4. validate_ensemble - Check configuration",
                "5. invoke - Test execution",
            ],
        }

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

            for yaml_file in Path(ensemble_dir).glob("**/*.yaml"):
                try:
                    config = self.ensemble_loader.load_from_file(str(yaml_file))
                    if config:
                        ensembles.append(
                            {
                                "name": config.name,
                                "source": source,
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

        # Update project path
        self._project_path = project_dir

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

        executor = EnsembleExecutor()
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
        """Execute validate_ensemble tool.

        Args:
            arguments: Tool arguments including ensemble_name.

        Returns:
            Validation result.
        """
        ensemble_name = arguments.get("ensemble_name")

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        # Find ensemble
        config = self._find_ensemble_by_name(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        # Validate configuration
        validation_errors = self._collect_validation_errors(config)

        return {
            "valid": len(validation_errors) == 0,
            "details": {
                "errors": validation_errors,
                "agent_count": len(config.agents),
            },
        }

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

    def _collect_validation_errors(self, config: Any) -> list[str]:
        """Collect validation errors for an ensemble configuration.

        Args:
            config: Ensemble configuration.

        Returns:
            List of validation error messages.
        """
        validation_errors: list[str] = []

        # Check for circular dependencies
        try:
            self._check_circular_dependencies(config)
        except ValueError as e:
            validation_errors.append(str(e))

        # Check agent references
        validation_errors.extend(self._validate_agent_references(config))

        # Check model profiles
        validation_errors.extend(self._validate_model_profiles(config))

        return validation_errors

    def _validate_agent_references(self, config: Any) -> list[str]:
        """Validate agent dependency references.

        Args:
            config: Ensemble configuration.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []
        agent_names = {_get_agent_attr(agent, "name") for agent in config.agents}

        for agent in config.agents:
            depends_on = _get_agent_attr(agent, "depends_on") or []
            for dep in depends_on:
                if dep not in agent_names:
                    agent_name = _get_agent_attr(agent, "name")
                    errors.append(
                        f"Agent '{agent_name}' depends on unknown agent '{dep}'"
                    )

        return errors

    def _validate_model_profiles(self, config: Any) -> list[str]:
        """Validate that model profiles exist and are properly configured.

        Args:
            config: Ensemble configuration.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []
        available_profiles = self.config_manager.get_model_profiles()

        for agent in config.agents:
            agent_name = _get_agent_attr(agent, "name")
            agent_type = _get_agent_attr(agent, "type")

            # Script agents don't need model profiles
            if agent_type == "script":
                continue

            model_profile = _get_agent_attr(agent, "model_profile")
            if not model_profile:
                errors.append(f"Agent '{agent_name}' has no model_profile configured")
                continue

            if model_profile not in available_profiles:
                errors.append(
                    f"Agent '{agent_name}' uses unknown profile '{model_profile}'"
                )
                continue

            # Check profile has required fields
            profile_config = available_profiles[model_profile]
            provider = profile_config.get("provider")
            if not provider:
                errors.append(
                    f"Profile '{model_profile}' missing 'provider' configuration"
                )
            else:
                # Check if provider is valid
                from llm_orc.providers.registry import provider_registry

                if not provider_registry.provider_exists(provider):
                    errors.append(
                        f"Profile '{model_profile}' uses unknown provider '{provider}'"
                    )

            if not profile_config.get("model"):
                errors.append(
                    f"Profile '{model_profile}' missing 'model' configuration"
                )

        return errors

    def _check_circular_dependencies(self, config: Any) -> None:
        """Check for circular dependencies in ensemble config.

        Args:
            config: Ensemble configuration.

        Raises:
            ValueError: If circular dependency detected.
        """
        # Build dependency graph
        graph: dict[str, list[str]] = {}
        for agent in config.agents:
            name = _get_agent_attr(agent, "name")
            depends_on = _get_agent_attr(agent, "depends_on") or []
            graph[name] = depends_on

        # DFS to detect cycles
        visited: set[str] = set()
        path: set[str] = set()

        def visit(node: str) -> bool:
            if node in path:
                return True  # Cycle detected
            if node in visited:
                return False

            visited.add(node)
            path.add(node)

            for neighbor in graph.get(node, []):
                if visit(neighbor):
                    return True

            path.remove(node)
            return False

        for agent_name in graph:
            if visit(agent_name):
                raise ValueError(f"Circular dependency detected involving {agent_name}")

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
        for agent in template_config.agents:
            agents.append(
                {
                    "name": _get_agent_attr(agent, "name"),
                    "model_profile": _get_agent_attr(agent, "model_profile"),
                    "depends_on": _get_agent_attr(agent, "depends_on") or [],
                }
            )

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
        """List available scripts.

        Args:
            arguments: Tool arguments including optional category.

        Returns:
            Scripts list.
        """
        category = arguments.get("category")
        scripts: list[dict[str, Any]] = []

        # Look in local scripts directory
        scripts_dir = Path.cwd() / ".llm-orc" / "scripts"

        if scripts_dir.exists():
            for category_dir in scripts_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                cat_name = category_dir.name
                if category and cat_name != category:
                    continue

                for script_file in category_dir.glob("*.py"):
                    scripts.append(
                        {
                            "name": script_file.stem,
                            "category": cat_name,
                            "path": str(script_file),
                        }
                    )

        return {"scripts": scripts}

    async def _library_browse_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Browse library items.

        Args:
            arguments: Tool arguments including type and category.

        Returns:
            Library items.
        """
        browse_type = arguments.get("type", "all")
        library_dir = self._get_library_dir()
        result: dict[str, list[dict[str, Any]]] = {}

        if browse_type in ("all", "ensembles"):
            result["ensembles"] = self._browse_library_ensembles(library_dir)

        if browse_type in ("all", "scripts"):
            result["scripts"] = self._browse_library_scripts(library_dir)

        return result

    def _browse_library_ensembles(self, library_dir: Path) -> list[dict[str, Any]]:
        """Browse ensembles in library directory."""
        ensembles: list[dict[str, Any]] = []
        ensembles_dir = library_dir / "ensembles"
        if not ensembles_dir.exists():
            return ensembles

        for yaml_file in ensembles_dir.glob("**/*.yaml"):
            try:
                config = self.ensemble_loader.load_from_file(str(yaml_file))
                if config:
                    ensembles.append(
                        {
                            "name": config.name,
                            "description": config.description,
                            "path": str(yaml_file),
                        }
                    )
            except Exception:
                continue
        return ensembles

    def _browse_library_scripts(self, library_dir: Path) -> list[dict[str, Any]]:
        """Browse scripts in library directory."""
        scripts: list[dict[str, Any]] = []
        scripts_dir = library_dir / "scripts"
        if not scripts_dir.exists():
            return scripts

        for category_dir in scripts_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for script_file in category_dir.glob("*.py"):
                scripts.append(
                    {
                        "name": script_file.stem,
                        "category": category_dir.name,
                        "path": str(script_file),
                    }
                )
        return scripts

    async def _library_copy_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Copy from library to local.

        Args:
            arguments: Tool arguments including source, destination, overwrite.

        Returns:
            Copy result.
        """
        source = arguments.get("source")
        destination = arguments.get("destination")
        overwrite = arguments.get("overwrite", False)

        if not source:
            raise ValueError("source is required")

        library_dir = self._get_library_dir()
        source_path = library_dir / source

        # Add .yaml extension if not present
        if not source_path.exists() and not source.endswith(".yaml"):
            source_path = library_dir / f"{source}.yaml"

        if not source_path.exists():
            raise ValueError(f"Source not found in library: {source}")

        # Determine destination
        if destination:
            dest_path = Path(destination)
        else:
            # Default to local .llm-orc directory
            ensemble_dirs = self.config_manager.get_ensembles_dirs()
            local_dir = Path.cwd() / ".llm-orc"
            library_dir = self._get_library_dir()

            # Try to find a local (non-library) ensembles dir
            for dir_path in ensemble_dirs:
                path = Path(dir_path)
                # Check it's a .llm-orc dir but not under the library
                is_local = ".llm-orc" in str(path)
                is_library = str(path).startswith(str(library_dir))
                if is_local and not is_library:
                    local_dir = path.parent  # Go up from ensembles to .llm-orc
                    break

            if "ensembles" in str(source_path):
                dest_path = local_dir / "ensembles" / source_path.name
            else:
                dest_path = local_dir / "scripts" / source_path.name

        # Check if exists
        if dest_path.exists() and not overwrite:
            raise ValueError(f"File already exists: {dest_path}")

        # Copy the file
        import shutil

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        return {
            "copied": True,
            "source": str(source_path),
            "destination": str(dest_path),
        }

    def _get_library_dir(self) -> Path:
        """Get library directory path.

        Returns:
            Path to library directory.
        """
        # Check for test override
        if self._test_library_dir is not None:
            return self._test_library_dir

        # Check for library in ensemble dirs
        for dir_path in self.config_manager.get_ensembles_dirs():
            if "library" in str(dir_path):
                # Return parent of ensembles dir
                return Path(dir_path).parent

        # Default library location
        return Path.cwd() / "llm-orchestra-library"

    # =========================================================================
    # Profile CRUD Tool Implementations
    # =========================================================================

    async def _list_profiles_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """List model profiles.

        Args:
            arguments: Tool arguments including optional provider filter.

        Returns:
            List of profiles.
        """
        import yaml

        provider_filter = arguments.get("provider")
        profiles: list[dict[str, Any]] = []

        # Get profiles from configured directories
        profiles_dirs = self.config_manager.get_profiles_dirs()

        for dir_path in profiles_dirs:
            profiles_dir = Path(dir_path)
            if not profiles_dir.exists():
                continue

            for yaml_file in profiles_dir.glob("*.yaml"):
                try:
                    content = yaml_file.read_text()
                    data = yaml.safe_load(content) or {}
                    profile_provider = data.get("provider", "")

                    # Apply filter if specified
                    if provider_filter and profile_provider != provider_filter:
                        continue

                    profiles.append(
                        {
                            "name": data.get("name", yaml_file.stem),
                            "provider": profile_provider,
                            "model": data.get("model", ""),
                            "path": str(yaml_file),
                        }
                    )
                except Exception:
                    continue

        return {"profiles": profiles}

    def _get_local_profiles_dir(self) -> Path:
        """Get the local profiles directory for writing.

        Returns:
            Path to local profiles directory.

        Raises:
            ValueError: If no profiles directory is configured.
        """
        profiles_dirs = self.config_manager.get_profiles_dirs()
        for dir_path in profiles_dirs:
            path = Path(dir_path)
            if ".llm-orc" in str(path) and "library" not in str(path):
                return path
        if profiles_dirs:
            return Path(profiles_dirs[0])
        raise ValueError("No profiles directory configured")

    async def _create_profile_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new profile.

        Args:
            arguments: Tool arguments.

        Returns:
            Creation result.
        """
        import yaml

        name = arguments.get("name")
        provider = arguments.get("provider")
        model = arguments.get("model")

        if not name:
            raise ValueError("name is required")
        if not provider:
            raise ValueError("provider is required")
        if not model:
            raise ValueError("model is required")

        local_dir = self._get_local_profiles_dir()
        target_file = local_dir / f"{name}.yaml"
        if target_file.exists():
            raise ValueError(f"Profile '{name}' already exists")

        # Build profile data
        profile_data: dict[str, Any] = {
            "name": name,
            "provider": provider,
            "model": model,
        }
        if arguments.get("system_prompt"):
            profile_data["system_prompt"] = arguments["system_prompt"]
        if arguments.get("timeout_seconds"):
            profile_data["timeout_seconds"] = arguments["timeout_seconds"]

        # Write file
        local_dir.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.safe_dump(profile_data, default_flow_style=False)
        target_file.write_text(yaml_content)

        return {"created": True, "path": str(target_file)}

    async def _update_profile_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Update an existing profile.

        Args:
            arguments: Tool arguments.

        Returns:
            Update result.
        """
        import yaml

        name = arguments.get("name")
        changes = arguments.get("changes", {})

        if not name:
            raise ValueError("name is required")

        # Find profile file
        profiles_dirs = self.config_manager.get_profiles_dirs()
        profile_file = None

        for dir_path in profiles_dirs:
            path = Path(dir_path) / f"{name}.yaml"
            if path.exists():
                profile_file = path
                break

        if not profile_file:
            raise ValueError(f"Profile '{name}' not found")

        # Load and update
        content = profile_file.read_text()
        data = yaml.safe_load(content) or {}
        data.update(changes)

        # Write back
        yaml_content = yaml.safe_dump(data, default_flow_style=False)
        profile_file.write_text(yaml_content)

        return {"updated": True, "path": str(profile_file)}

    async def _delete_profile_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete a profile.

        Args:
            arguments: Tool arguments.

        Returns:
            Deletion result.
        """
        name = arguments.get("name")
        confirm = arguments.get("confirm", False)

        if not name:
            raise ValueError("name is required")
        if not confirm:
            raise ValueError("Confirmation required to delete profile")

        # Find profile file
        profiles_dirs = self.config_manager.get_profiles_dirs()
        profile_file = None

        for dir_path in profiles_dirs:
            path = Path(dir_path) / f"{name}.yaml"
            if path.exists():
                profile_file = path
                break

        if not profile_file:
            raise ValueError(f"Profile '{name}' not found")

        # Delete
        profile_file.unlink()

        return {"deleted": True, "name": name}

    # =========================================================================
    # Artifact Management Tool Implementations
    # =========================================================================

    def _get_artifacts_base(self) -> Path:
        """Get artifacts base directory."""
        if self._test_artifacts_base is not None:
            return self._test_artifacts_base
        return Path.cwd() / ".llm-orc" / "artifacts"

    async def _delete_artifact_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete an artifact.

        Args:
            arguments: Tool arguments.

        Returns:
            Deletion result.
        """
        import shutil

        artifact_id = arguments.get("artifact_id")
        confirm = arguments.get("confirm", False)

        if not artifact_id:
            raise ValueError("artifact_id is required")
        if not confirm:
            raise ValueError("Confirmation required to delete artifact")

        # Parse artifact_id (format: ensemble/timestamp)
        parts = artifact_id.split("/")
        if len(parts) != 2:
            raise ValueError("Invalid artifact_id format (expected ensemble/timestamp)")

        ensemble_name, timestamp = parts

        # Find artifact directory
        artifacts_base = self._get_artifacts_base()
        artifact_dir = artifacts_base / ensemble_name / timestamp

        if not artifact_dir.exists():
            raise ValueError(f"Artifact '{artifact_id}' not found")

        # Delete
        shutil.rmtree(artifact_dir)

        return {"deleted": True, "artifact_id": artifact_id}

    async def _cleanup_artifacts_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Cleanup old artifacts.

        Args:
            arguments: Tool arguments.

        Returns:
            Cleanup result.
        """
        import time

        ensemble_name = arguments.get("ensemble_name")
        older_than_days = arguments.get("older_than_days", 30)
        dry_run = arguments.get("dry_run", True)

        artifacts_base = self._get_artifacts_base()
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

        ensemble_dirs = self._get_ensemble_artifact_dirs(artifacts_base, ensemble_name)
        would_delete, deleted = self._process_old_artifacts(
            ensemble_dirs, cutoff_time, dry_run
        )

        if dry_run:
            return {
                "dry_run": True,
                "would_delete": would_delete,
                "count": len(would_delete),
            }
        return {"dry_run": False, "deleted": deleted, "count": len(deleted)}

    def _get_ensemble_artifact_dirs(
        self, artifacts_base: Path, ensemble_name: str | None
    ) -> list[Path]:
        """Get list of ensemble artifact directories to check.

        Args:
            artifacts_base: Base artifacts directory.
            ensemble_name: Optional specific ensemble to check.

        Returns:
            List of ensemble directories.
        """
        if ensemble_name:
            return [artifacts_base / ensemble_name]
        if artifacts_base.exists():
            return [d for d in artifacts_base.iterdir() if d.is_dir()]
        return []

    def _process_old_artifacts(
        self, ensemble_dirs: list[Path], cutoff_time: float, dry_run: bool
    ) -> tuple[list[str], list[str]]:
        """Process and optionally delete old artifacts.

        Args:
            ensemble_dirs: List of ensemble directories to check.
            cutoff_time: Unix timestamp cutoff for deletion.
            dry_run: If True, don't actually delete.

        Returns:
            Tuple of (would_delete list, deleted list).
        """
        import shutil

        would_delete: list[str] = []
        deleted: list[str] = []

        for ensemble_dir in ensemble_dirs:
            if not ensemble_dir.exists():
                continue

            for artifact_dir in ensemble_dir.iterdir():
                if not artifact_dir.is_dir():
                    continue

                mtime = artifact_dir.stat().st_mtime
                if mtime < cutoff_time:
                    artifact_id = f"{ensemble_dir.name}/{artifact_dir.name}"
                    would_delete.append(artifact_id)

                    if not dry_run:
                        shutil.rmtree(artifact_dir)
                        deleted.append(artifact_id)

        return would_delete, deleted

    # =========================================================================
    # Script Management Tool Implementations (Low Priority)
    # =========================================================================

    def _get_scripts_dir(self) -> Path:
        """Get scripts directory path.

        Returns:
            Path to scripts directory.
        """
        if self._test_scripts_dir is not None:
            return self._test_scripts_dir
        return Path.cwd() / ".llm-orc" / "scripts"

    async def _get_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get script details.

        Args:
            arguments: Tool arguments including name and category.

        Returns:
            Script details.
        """
        name = arguments.get("name")
        category = arguments.get("category")

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")

        scripts_dir = self._get_scripts_dir()
        script_file = scripts_dir / category / f"{name}.py"

        if not script_file.exists():
            raise ValueError(f"Script '{category}/{name}' not found")

        content = script_file.read_text()
        description = self._extract_docstring(content)

        return {
            "name": name,
            "category": category,
            "path": str(script_file),
            "description": description,
            "source": content,
        }

    def _extract_docstring(self, content: str) -> str:
        """Extract module docstring from Python source code.

        Args:
            content: Python source code.

        Returns:
            Extracted docstring or empty string if not found.
        """
        lines = content.split("\n")
        in_docstring = False
        docstring_lines: list[str] = []

        for line in lines:
            if '"""' in line or "'''" in line:
                if in_docstring:
                    break
                in_docstring = True
                # Handle single-line docstring
                if line.count('"""') == 2 or line.count("'''") == 2:
                    return self._strip_docstring_quotes(line.strip())
            elif in_docstring:
                docstring_lines.append(line)

        if docstring_lines:
            return "\n".join(docstring_lines).strip()
        return ""

    def _strip_docstring_quotes(self, text: str) -> str:
        """Remove docstring quote marks from text.

        Args:
            text: Text with potential docstring quotes.

        Returns:
            Text with quotes removed.
        """
        for quote in ('"""', "'''"):
            if text.startswith(quote):
                text = text[3:]
            if text.endswith(quote):
                text = text[:-3]
        return text.strip()

    async def _test_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Test a script with sample input.

        Args:
            arguments: Tool arguments including name, category, and input.

        Returns:
            Test result.
        """
        import subprocess
        import sys

        name = arguments.get("name")
        category = arguments.get("category")
        input_data = arguments.get("input", "")

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")

        scripts_dir = self._get_scripts_dir()
        script_file = scripts_dir / category / f"{name}.py"

        if not script_file.exists():
            raise ValueError(f"Script '{category}/{name}' not found")

        # Run the script with input
        try:
            result = subprocess.run(
                [sys.executable, str(script_file)],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _create_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new script.

        Args:
            arguments: Tool arguments including name, category, and template.

        Returns:
            Creation result.
        """
        name = arguments.get("name")
        category = arguments.get("category")
        template = arguments.get("template", "basic")

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")

        scripts_dir = self._get_scripts_dir()
        category_dir = scripts_dir / category
        script_file = category_dir / f"{name}.py"

        if script_file.exists():
            raise ValueError(f"Script '{category}/{name}' already exists")

        # Generate script from template
        if template == "extraction":
            content = f'''"""Extraction script: {name}

Extracts structured data from input text.
"""

import json
import sys


def extract(text: str) -> dict:
    """Extract data from text."""
    # TODO: Implement extraction logic
    return {{"input_length": len(text)}}


if __name__ == "__main__":
    input_text = sys.stdin.read()
    result = extract(input_text)
    print(json.dumps(result))
'''
        else:  # basic template
            content = f'''"""Primitive script: {name}

Process input and produce output.
"""

import sys


def process(text: str) -> str:
    """Process input text."""
    # TODO: Implement processing logic
    return text


if __name__ == "__main__":
    input_text = sys.stdin.read()
    result = process(input_text)
    print(result)
'''

        # Create directory and file
        category_dir.mkdir(parents=True, exist_ok=True)
        script_file.write_text(content)

        return {
            "created": True,
            "path": str(script_file),
            "template": template,
        }

    async def _delete_script_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete a script.

        Args:
            arguments: Tool arguments including name, category, and confirm.

        Returns:
            Deletion result.
        """
        name = arguments.get("name")
        category = arguments.get("category")
        confirm = arguments.get("confirm", False)

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")
        if not confirm:
            raise ValueError("Confirmation required to delete script")

        scripts_dir = self._get_scripts_dir()
        script_file = scripts_dir / category / f"{name}.py"

        if not script_file.exists():
            raise ValueError(f"Script '{category}/{name}' not found")

        script_file.unlink()

        return {
            "deleted": True,
            "name": name,
            "category": category,
        }

    # =========================================================================
    # Library Extras Tool Implementations (Low Priority)
    # =========================================================================

    async def _library_search_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Search library content.

        Args:
            arguments: Tool arguments including query.

        Returns:
            Search results.
        """
        query = arguments.get("query", "").lower()

        if not query:
            raise ValueError("query is required")

        library_dir = self._get_library_dir()
        ensemble_results = self._search_library_ensembles(library_dir, query)
        script_results = self._search_library_scripts(library_dir, query)

        return {
            "query": query,
            "results": {
                "ensembles": ensemble_results,
                "scripts": script_results,
            },
            "total": len(ensemble_results) + len(script_results),
        }

    def _search_library_ensembles(
        self, library_dir: Path, query: str
    ) -> list[dict[str, Any]]:
        """Search ensembles in library directory by query."""
        results: list[dict[str, Any]] = []
        ensembles_dir = library_dir / "ensembles"

        if not ensembles_dir.exists():
            return results

        for yaml_file in ensembles_dir.glob("**/*.yaml"):
            try:
                config = self.ensemble_loader.load_from_file(str(yaml_file))
                if config:
                    name_match = query in config.name.lower()
                    desc_match = query in (config.description or "").lower()
                    if name_match or desc_match:
                        results.append(
                            {
                                "name": config.name,
                                "description": config.description,
                                "path": str(yaml_file),
                                "match": "name" if name_match else "description",
                            }
                        )
            except Exception:
                continue

        return results

    def _search_library_scripts(
        self, library_dir: Path, query: str
    ) -> list[dict[str, Any]]:
        """Search scripts in library directory by query."""
        results: list[dict[str, Any]] = []
        scripts_dir = library_dir / "scripts"

        if not scripts_dir.exists():
            return results

        for category_dir in scripts_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for script_file in category_dir.glob("*.py"):
                name_match = query in script_file.stem.lower()
                cat_match = query in category_dir.name.lower()
                if name_match or cat_match:
                    results.append(
                        {
                            "name": script_file.stem,
                            "category": category_dir.name,
                            "path": str(script_file),
                            "match": "name" if name_match else "category",
                        }
                    )

        return results

    async def _library_info_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get library information.

        Args:
            arguments: Tool arguments (unused).

        Returns:
            Library info.
        """
        library_dir = self._get_library_dir()

        info: dict[str, Any] = {
            "path": str(library_dir),
            "exists": library_dir.exists(),
            "ensembles_count": 0,
            "scripts_count": 0,
            "categories": [],
        }

        if not library_dir.exists():
            return info

        # Count ensembles
        ensembles_dir = library_dir / "ensembles"
        if ensembles_dir.exists():
            info["ensembles_count"] = len(list(ensembles_dir.glob("**/*.yaml")))

        # Count scripts and categories
        scripts_dir = library_dir / "scripts"
        if scripts_dir.exists():
            categories: list[str] = []
            for category_dir in scripts_dir.iterdir():
                if category_dir.is_dir():
                    categories.append(category_dir.name)
                    info["scripts_count"] += len(list(category_dir.glob("*.py")))
            info["categories"] = sorted(categories)

        return info

    # =========================================================================
    # Phase 3: Provider & Model Discovery
    # =========================================================================

    async def _get_provider_status_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Get status of all providers and available models.

        Args:
            arguments: Tool arguments (unused).

        Returns:
            Provider status with available models.
        """
        providers: dict[str, Any] = {}

        # Check Ollama
        providers["ollama"] = await self._get_ollama_status()

        # Check cloud providers via credential storage
        providers["anthropic-api"] = self._get_cloud_provider_status("anthropic-api")
        providers["anthropic-claude-pro-max"] = self._get_cloud_provider_status(
            "anthropic-claude-pro-max"
        )
        providers["google-gemini"] = self._get_cloud_provider_status("google-gemini")

        return {"providers": providers}

    async def _get_ollama_status(self) -> dict[str, Any]:
        """Check Ollama availability and list models.

        Returns:
            Ollama status with available models.
        """
        # Check for test override
        if self._test_ollama_status is not None:
            return self._test_ollama_status

        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return {
                        "available": True,
                        "models": sorted(models),
                        "model_count": len(models),
                    }
        except Exception:
            pass

        return {"available": False, "models": [], "reason": "Ollama not running"}

    def _get_cloud_provider_status(self, provider: str) -> dict[str, Any]:
        """Check if a cloud provider is configured.

        Args:
            provider: Provider name.

        Returns:
            Provider status.
        """
        from llm_orc.core.auth.authentication import CredentialStorage

        storage = CredentialStorage()
        configured_providers = storage.list_providers()

        if provider in configured_providers:
            return {"available": True, "reason": "configured"}

        return {"available": False, "reason": "not configured"}

    async def _check_ensemble_runnable_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Check if an ensemble can run with current providers.

        Args:
            arguments: Tool arguments with ensemble_name.

        Returns:
            Runnable status with agent details.
        """
        ensemble_name = arguments.get("ensemble_name")
        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        # Find ensemble
        config = self._find_ensemble_by_name(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        # Get provider status
        provider_status = await self._get_provider_status_tool({})
        providers = provider_status.get("providers", {})

        # Get all profiles
        all_profiles = self._get_all_profiles()

        # Check each agent
        agents: list[dict[str, Any]] = []
        all_runnable = True

        for agent in config.agents:
            agent_name = _get_agent_attr(agent, "name", "unknown")
            profile_name = _get_agent_attr(agent, "model_profile", "")

            agent_status = self._check_agent_runnable(
                agent_name, profile_name, all_profiles, providers
            )
            agents.append(agent_status)

            if agent_status["status"] != "available":
                all_runnable = False

        return {
            "ensemble": ensemble_name,
            "runnable": all_runnable,
            "agents": agents,
        }

    def _check_agent_runnable(
        self,
        agent_name: str,
        profile_name: str,
        all_profiles: dict[str, dict[str, Any]],
        providers: dict[str, Any],
    ) -> dict[str, Any]:
        """Check if an agent can run with current providers.

        Args:
            agent_name: Name of the agent.
            profile_name: Name of the profile the agent uses.
            all_profiles: All available profiles.
            providers: Provider status from _get_provider_status_tool.

        Returns:
            Agent runnable status with alternatives.
        """
        result: dict[str, Any] = {
            "name": agent_name,
            "profile": profile_name,
            "provider": "",
            "status": "available",
            "alternatives": [],
        }

        # Check if profile exists
        if profile_name not in all_profiles:
            result["status"] = "missing_profile"
            result["alternatives"] = self._suggest_local_alternatives(providers)
            return result

        # Get profile's provider
        profile = all_profiles[profile_name]
        provider = profile.get("provider", "")
        result["provider"] = provider

        # Check if provider is available
        provider_info = providers.get(provider, {})
        if not provider_info.get("available", False):
            result["status"] = "provider_unavailable"
            result["alternatives"] = self._suggest_local_alternatives(providers)
            return result

        # For Ollama, check if model is available
        if provider == "ollama":
            model = profile.get("model", "")
            available_models = provider_info.get("models", [])
            # Normalize model name (handle tags like :latest)
            model_base = model.split(":")[0] if ":" in model else model
            model_found = any(
                m == model or m.startswith(f"{model_base}:") for m in available_models
            )
            if not model_found:
                result["status"] = "model_unavailable"
                result["alternatives"] = self._suggest_available_models(
                    available_models
                )

        return result

    def _suggest_local_alternatives(self, providers: dict[str, Any]) -> list[str]:
        """Suggest local profile alternatives.

        Args:
            providers: Provider status.

        Returns:
            List of suggested local profile names.
        """
        ollama = providers.get("ollama", {})
        if not ollama.get("available", False):
            return []

        # Get profiles using ollama
        all_profiles = self._get_all_profiles()
        local_profiles: list[str] = []

        for name, profile in all_profiles.items():
            if profile.get("provider") == "ollama":
                local_profiles.append(name)

        return sorted(local_profiles)[:5]  # Return top 5

    def _suggest_available_models(self, available_models: list[str]) -> list[str]:
        """Suggest available Ollama models.

        Args:
            available_models: List of available Ollama model names.

        Returns:
            List of suggested models.
        """
        return sorted(available_models)[:5]  # Return top 5

    def _get_all_profiles(self) -> dict[str, dict[str, Any]]:
        """Get all profiles as a dict keyed by name.

        Returns:
            Dict mapping profile name to profile config.
        """
        profiles: dict[str, dict[str, Any]] = {}

        for dir_path in self.config_manager.get_profiles_dirs():
            profile_dir = Path(dir_path)
            if not profile_dir.exists():
                continue

            for yaml_file in profile_dir.glob("*.yaml"):
                self._load_profiles_from_file(yaml_file, profiles)

        return profiles

    def _load_profiles_from_file(
        self, yaml_file: Path, profiles: dict[str, dict[str, Any]]
    ) -> None:
        """Load profiles from a YAML file into the profiles dict.

        Args:
            yaml_file: Path to YAML file.
            profiles: Dict to populate with profiles.
        """
        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f) or {}

            self._parse_profile_data(data, profiles)
        except Exception:
            pass

    def _parse_profile_data(
        self, data: dict[str, Any], profiles: dict[str, dict[str, Any]]
    ) -> None:
        """Parse profile data from various YAML formats.

        Args:
            data: Parsed YAML data.
            profiles: Dict to populate with profiles.
        """
        if "model_profiles" in data:
            self._parse_dict_format_profiles(data["model_profiles"], profiles)
        elif "profiles" in data:
            self._parse_list_format_profiles(data["profiles"], profiles)
        elif "name" in data:
            profiles[data["name"]] = data

    def _parse_dict_format_profiles(
        self, model_profiles: dict[str, Any], profiles: dict[str, dict[str, Any]]
    ) -> None:
        """Parse dict format: model_profiles: {name: {config...}}."""
        for name, config in model_profiles.items():
            if isinstance(config, dict):
                config["name"] = name
                profiles[name] = config

    def _parse_list_format_profiles(
        self, profile_list: list[dict[str, Any]], profiles: dict[str, dict[str, Any]]
    ) -> None:
        """Parse list format: profiles: [{name: ..., ...}]."""
        for p in profile_list:
            name = p.get("name", "")
            if name:
                profiles[name] = p

    async def _help_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get help documentation.

        Args:
            arguments: Tool arguments (none required).

        Returns:
            Comprehensive help documentation.
        """
        return self._get_help_documentation()
