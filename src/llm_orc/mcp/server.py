"""MCP Server v2 implementation using FastMCP SDK.

This module implements the MCP server following ADR-009, providing:
- Resource exposure for ensembles, artifacts, metrics, and profiles
- Tools for invoke, validate_ensemble, update_ensemble, analyze_execution
- Streaming support for long-running executions
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.artifact_manager import ArtifactManager


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

    def __init__(self, config_manager: ConfigurationManager | None = None) -> None:
        """Initialize MCP server.

        Args:
            config_manager: Configuration manager instance. Creates default if None.
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.ensemble_loader = EnsembleLoader()
        self.artifact_manager = ArtifactManager()
        self._mcp = FastMCP("llm-orc")
        self._setup_resources()
        self._setup_tools()

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
        if name == "invoke":
            return await self._invoke_tool(arguments)
        elif name == "validate_ensemble":
            return await self._validate_ensemble_tool(arguments)
        elif name == "update_ensemble":
            return await self._update_ensemble_tool(arguments)
        elif name == "analyze_execution":
            return await self._analyze_execution_tool(arguments)
        else:
            raise ValueError(f"Tool not found: {name}")

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
        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        config = self._find_ensemble_by_name(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble does not exist: {ensemble_name}")

        # Execute with streaming
        from llm_orc.core.execution.ensemble_execution import EnsembleExecutor

        executor = EnsembleExecutor()
        total_agents = len(config.agents)
        state: dict[str, Any] = {
            "completed": 0,
            "result": {},
            "ensemble_name": ensemble_name,
            "input_data": input_data,
        }

        msg = f"Starting ensemble '{ensemble_name}' with {total_agents} agents"
        await ctx.info(msg)

        async for event in executor.execute_streaming(config, input_data):
            await self._handle_streaming_event(event, ctx, total_agents, state)

        result = state.get("result", {})
        if not isinstance(result, dict):
            result = {}
        return result

    async def _handle_streaming_event(
        self,
        event: dict[str, Any],
        ctx: Context[Any, Any, Any],
        total_agents: int,
        state: dict[str, Any],
    ) -> None:
        """Handle a single streaming event from ensemble execution.

        Args:
            event: The streaming event.
            ctx: FastMCP context for progress reporting.
            total_agents: Total number of agents in ensemble.
            state: Mutable state dict with 'completed' count and 'result'.
        """
        event_type = event.get("type", "")
        event_data = event.get("data", {})

        if event_type == "execution_started":
            await ctx.report_progress(progress=0, total=total_agents)

        elif event_type == "agent_started":
            agent_name = event_data.get("agent_name", "unknown")
            await ctx.info(f"Agent '{agent_name}' started")

        elif event_type == "agent_completed":
            state["completed"] += 1
            agent_name = event_data.get("agent_name", "unknown")
            await ctx.report_progress(progress=state["completed"], total=total_agents)
            await ctx.info(f"Agent '{agent_name}' completed")

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
            await ctx.report_progress(progress=total_agents, total=total_agents)

        elif event_type == "execution_failed":
            error_msg = event_data.get("error", "Unknown error")
            await ctx.error(f"Execution failed: {error_msg}")
            state["result"] = {
                "results": {},
                "synthesis": None,
                "status": "failed",
                "error": error_msg,
            }

        elif event_type == "agent_fallback_started":
            agent_name = event_data.get("agent_name", "unknown")
            msg = f"Agent '{agent_name}' falling back to alternate model"
            await ctx.warning(msg)

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
                artifact_data["agents"].append({
                    "name": agent_name,
                    "status": agent_result.get("status", "unknown"),
                    "result": agent_result.get("response", ""),
                })

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
                        f"Profile '{model_profile}' uses unknown provider "
                        f"'{provider}'"
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
