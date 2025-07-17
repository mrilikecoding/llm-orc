"""Ensemble execution with agent coordination."""

import asyncio
import time
from typing import Any

import click

from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.config import ConfigurationManager
from llm_orc.ensemble_config import EnsembleConfig
from llm_orc.models import (
    ClaudeCLIModel,
    ClaudeModel,
    ModelInterface,
    OAuthClaudeModel,
    OllamaModel,
)
from llm_orc.orchestration import Agent
from llm_orc.roles import RoleDefinition
from llm_orc.script_agent import ScriptAgent


class EnsembleExecutor:
    """Executes ensembles of agents and coordinates their responses."""

    def __init__(self) -> None:
        """Initialize the ensemble executor with shared infrastructure."""
        # Share configuration and credential infrastructure across model loads
        # but keep model instances separate for independent contexts
        self._config_manager = ConfigurationManager()
        self._credential_storage = CredentialStorage(self._config_manager)

    async def execute(self, config: EnsembleConfig, input_data: str) -> dict[str, Any]:
        """Execute an ensemble and return structured results."""
        start_time = time.time()

        # Initialize result structure
        result: dict[str, Any] = {
            "ensemble": config.name,
            "status": "running",
            "input": {"data": input_data},
            "results": {},
            "synthesis": None,
            "metadata": {"agents_used": len(config.agents), "started_at": start_time},
        }

        # Ensure results is properly typed
        results_dict: dict[str, Any] = result["results"]

        # Execute agents in phases: script agents first, then LLM agents
        has_errors = False
        agent_usage: dict[str, Any] = {}
        context_data = {}

        # Phase 1: Execute script agents to gather context
        script_agents = [a for a in config.agents if a.get("type") == "script"]
        for agent_config in script_agents:
            try:
                # Resolve model profile to get enhanced configuration
                enhanced_config = await self._resolve_model_profile_to_config(
                    agent_config
                )
                timeout = enhanced_config.get("timeout_seconds") or (
                    config.coordinator.get("timeout_seconds")
                )
                agent_result, model_instance = await self._execute_agent_with_timeout(
                    agent_config, input_data, timeout
                )
                results_dict[agent_config["name"]] = {
                    "response": agent_result,
                    "status": "success",
                }
                # Store script results as context for LLM agents
                context_data[agent_config["name"]] = agent_result
            except Exception as e:
                results_dict[agent_config["name"]] = {
                    "error": str(e),
                    "status": "failed",
                }
                has_errors = True

        # Phase 2: Execute LLM agents with dependency-aware phasing
        llm_agents = [a for a in config.agents if a.get("type") != "script"]

        # Prepare enhanced input for LLM agents
        # CLI input overrides config default_task when provided
        # Fall back to config.default_task or config.task (backward compatibility)
        if input_data and input_data.strip() and input_data != "Please analyze this.":
            # Use CLI input when explicitly provided
            task_input = input_data
        else:
            # Fall back to config default task (support both new and old field names)
            task_input = (
                getattr(config, "default_task", None)
                or getattr(config, "task", None)
                or input_data
            )
        enhanced_input = task_input
        if context_data:
            context_text = "\n\n".join(
                [f"=== {name} ===\n{data}" for name, data in context_data.items()]
            )
            enhanced_input = f"{task_input}\n\n{context_text}"

        # Analyze dependencies to determine execution phases
        independent_agents, dependent_agents = self._analyze_dependencies(llm_agents)

        # Phase 2.1: Execute independent LLM agents in parallel
        if independent_agents:
            await self._execute_agents_parallel(
                independent_agents, enhanced_input, config, results_dict, agent_usage
            )
            if any(
                result.get("status") == "failed" for result in results_dict.values()
            ):
                has_errors = True

        # Phase 2.2: Execute dependent agents with results from independent agents
        if dependent_agents:
            # Enhance input with results from independent agents
            enhanced_input_with_deps = self._enhance_input_with_dependencies(
                enhanced_input, dependent_agents, results_dict
            )

            await self._execute_agents_parallel(
                dependent_agents,
                enhanced_input_with_deps,
                config,
                results_dict,
                agent_usage,
            )
            if any(
                result.get("status") == "failed" for result in results_dict.values()
            ):
                has_errors = True

        # Synthesize results if coordinator is configured
        synthesis_usage = None
        if config.coordinator.get("synthesis_prompt"):
            try:
                synthesis_timeout = config.coordinator.get("synthesis_timeout_seconds")
                synthesis_result = await self._synthesize_results_with_timeout(
                    config, results_dict, synthesis_timeout
                )
                synthesis, synthesis_model = synthesis_result
                result["synthesis"] = synthesis
                synthesis_usage = synthesis_model.get_last_usage()
            except Exception as e:
                result["synthesis"] = f"Synthesis failed: {str(e)}"
                has_errors = True

        # Calculate usage totals
        usage_summary = self._calculate_usage_summary(agent_usage, synthesis_usage)

        # Finalize result
        end_time = time.time()
        result["status"] = "completed_with_errors" if has_errors else "completed"
        metadata_dict: dict[str, Any] = result["metadata"]
        metadata_dict["duration"] = f"{(end_time - start_time):.2f}s"
        metadata_dict["completed_at"] = end_time
        metadata_dict["usage"] = usage_summary

        return result

    async def _execute_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute a single agent and return its response and model instance."""
        agent_type = agent_config.get("type", "llm")

        if agent_type == "script":
            # Execute script agent
            script_agent = ScriptAgent(agent_config["name"], agent_config)
            response = await script_agent.execute(input_data)
            return response, None  # Script agents don't have model instances
        else:
            # Execute LLM agent
            # Load role and model for this agent
            role = await self._load_role_from_config(agent_config)
            model = await self._load_model_from_agent_config(agent_config)

            # Create agent
            agent = Agent(agent_config["name"], role, model)

            # Generate response
            response = await agent.respond_to_message(input_data)
            return response, model

    async def _load_role_from_config(
        self, agent_config: dict[str, Any]
    ) -> RoleDefinition:
        """Load a role definition from agent configuration."""
        agent_name = agent_config["name"]

        # Resolve model profile to get enhanced configuration
        enhanced_config = await self._resolve_model_profile_to_config(agent_config)

        # Use system_prompt from enhanced config if available, otherwise use fallback
        if "system_prompt" in enhanced_config:
            prompt = enhanced_config["system_prompt"]
        else:
            prompt = f"You are a {agent_name}. Provide helpful analysis."

        return RoleDefinition(name=agent_name, prompt=prompt)

    async def _resolve_model_profile_to_config(
        self, agent_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve model profile and merge with agent config.

        Agent config takes precedence over model profile defaults.
        """
        enhanced_config = agent_config.copy()

        # If model_profile is specified, get its configuration
        if "model_profile" in agent_config:
            profiles = self._config_manager.get_model_profiles()

            profile_name = agent_config["model_profile"]
            if profile_name in profiles:
                profile_config = profiles[profile_name]
                # Merge profile defaults with agent config
                # (agent config takes precedence)
                enhanced_config = {**profile_config, **agent_config}

        return enhanced_config

    async def _load_role(self, role_name: str) -> RoleDefinition:
        """Load a role definition."""
        # For now, create a simple role
        # TODO: Load from role configuration files
        return RoleDefinition(
            name=role_name, prompt=f"You are a {role_name}. Provide helpful analysis."
        )

    async def _load_model_from_agent_config(
        self, agent_config: dict[str, Any]
    ) -> ModelInterface:
        """Load a model based on agent configuration.

        Configuration can specify model_profile or model+provider.
        """
        # Check if model_profile is specified (takes precedence)
        if "model_profile" in agent_config:
            profile_name = agent_config["model_profile"]
            resolved_model, resolved_provider = (
                self._config_manager.resolve_model_profile(profile_name)
            )
            return await self._load_model(resolved_model, resolved_provider)

        # Fall back to explicit model+provider
        model: str | None = agent_config.get("model")
        provider: str | None = agent_config.get("provider")

        if not model:
            raise ValueError(
                "Agent configuration must specify either 'model_profile' or 'model'"
            )

        return await self._load_model(model, provider)

    async def _load_model(
        self, model_name: str, provider: str | None = None
    ) -> ModelInterface:
        """Load a model interface based on authentication configuration."""
        # Handle mock models for testing
        if model_name.startswith("mock"):
            from unittest.mock import AsyncMock

            mock = AsyncMock(spec=ModelInterface)
            mock.generate_response.return_value = f"Response from {model_name}"
            return mock

        # Use shared configuration and credential storage for efficiency
        # Each model instance remains independent for separate contexts
        storage = self._credential_storage

        try:
            # Get authentication method for the provider configuration
            # Use provider if specified, otherwise use model_name as lookup key
            lookup_key = provider if provider else model_name
            auth_method = storage.get_auth_method(lookup_key)

            if not auth_method:
                # Prompt user to set up authentication if not configured
                if _should_prompt_for_auth(model_name):
                    auth_configured = _prompt_auth_setup(model_name, storage)
                    if auth_configured:
                        # Retry model loading after auth setup
                        return await self._load_model(model_name, provider)

                # Handle based on provider
                if provider == "ollama":
                    # Expected behavior for Ollama - no auth needed
                    return OllamaModel(model_name=model_name)
                elif provider:
                    # Other providers require authentication
                    raise ValueError(
                        f"No authentication configured for provider '{provider}' "
                        f"with model '{model_name}'. "
                        f"Run 'llm-orc auth setup' to configure authentication."
                    )
                else:
                    # No provider specified, fallback to Ollama
                    click.echo(
                        f"ℹ️  No provider specified for '{model_name}', "
                        f"treating as local Ollama model"
                    )
                    return OllamaModel(model_name=model_name)

            if auth_method == "api_key":
                lookup_key = provider if provider else model_name
                api_key = storage.get_api_key(lookup_key)
                if not api_key:
                    raise ValueError(f"No API key found for {lookup_key}")

                # Check if this is a claude-cli configuration
                # (stored as api_key but path-like)
                if model_name == "claude-cli" or api_key.startswith("/"):
                    return ClaudeCLIModel(claude_path=api_key)
                elif provider == "google-gemini":
                    from .models import GeminiModel

                    return GeminiModel(api_key=api_key, model=model_name)
                else:
                    # Assume it's an Anthropic API key for Claude
                    return ClaudeModel(api_key=api_key)

            elif auth_method == "oauth":
                lookup_key = provider if provider else model_name
                oauth_token = storage.get_oauth_token(lookup_key)
                if not oauth_token:
                    raise ValueError(f"No OAuth token found for {lookup_key}")

                # Use stored client_id or fallback for anthropic-claude-pro-max
                client_id = oauth_token.get("client_id")
                if not client_id and lookup_key == "anthropic-claude-pro-max":
                    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

                return OAuthClaudeModel(
                    access_token=oauth_token["access_token"],
                    refresh_token=oauth_token.get("refresh_token"),
                    client_id=client_id,
                    credential_storage=storage,
                    provider_key=model_name,
                )

            else:
                raise ValueError(f"Unknown authentication method: {auth_method}")

        except Exception as e:
            # Fallback: use configured default model or treat as Ollama
            click.echo(f"⚠️  Failed to load model '{model_name}': {str(e)}")
            if model_name in ["llama3", "llama2"]:  # Known local models
                click.echo(f"🔄 Treating '{model_name}' as local Ollama model")
                return OllamaModel(model_name=model_name)
            else:
                # For unknown models, use configured fallback
                click.echo(f"🔄 Using configured fallback instead of '{model_name}'")
                return await self._get_fallback_model("general")

    async def _synthesize_results(
        self, config: EnsembleConfig, agent_results: dict[str, Any]
    ) -> tuple[str, ModelInterface]:
        """Synthesize results from all agents."""
        synthesis_model = await self._get_synthesis_model(config)

        # Prepare synthesis prompt with agent results
        results_text = ""
        for agent_name, result in agent_results.items():
            if result["status"] == "success":
                results_text += f"\n{agent_name}: {result['response']}\n"
            else:
                results_text += f"\n{agent_name}: [Error: {result['error']}]\n"

        # Prepare role and message for coordinator
        coordinator_role = config.coordinator.get("system_prompt")
        synthesis_instructions = config.coordinator["synthesis_prompt"]

        # If no coordinator system_prompt, use synthesis_prompt as role
        if coordinator_role:
            role_prompt = coordinator_role
            message = f"{synthesis_instructions}\n\nAgent Results:{results_text}"
        else:
            role_prompt = synthesis_instructions
            message = (
                f"Please synthesize these results:\n\nAgent Results:{results_text}"
            )

        # Generate synthesis
        response = await synthesis_model.generate_response(
            message=message, role_prompt=role_prompt
        )

        return response, synthesis_model

    async def _get_synthesis_model(self, config: EnsembleConfig) -> ModelInterface:
        """Get model for synthesis based on coordinator configuration."""
        # Check if coordinator specifies a model_profile or model
        if config.coordinator.get("model_profile") or config.coordinator.get("model"):
            try:
                # Use the configured coordinator model
                # (supports both model_profile and explicit model+provider)
                return await self._load_model_from_agent_config(config.coordinator)
            except Exception as e:
                # Fallback to configured default model
                click.echo(f"⚠️  Failed to load coordinator model: {str(e)}")
                return await self._get_fallback_model("coordinator")
        else:
            # Use configured default for backward compatibility
            click.echo("ℹ️  No coordinator model specified, using configured default")
            return await self._get_fallback_model("coordinator")

    async def _get_fallback_model(self, context: str = "general") -> ModelInterface:
        """Get a fallback model - always use free local model for reliability."""
        import click

        # Load project configuration to get default models
        config_manager = ConfigurationManager()
        project_config = config_manager.load_project_config()

        default_models = project_config.get("project", {}).get("default_models", {})

        # Log that we're falling back
        click.echo(f"⚠️  Falling back to free local model for {context}")

        # Always prefer free local models for fallback reliability
        # Look for a "test" profile first (typically free/local)
        fallback_profile = default_models.get("test")

        if fallback_profile and isinstance(fallback_profile, str):
            try:
                # Try to resolve the test profile to get a free local model
                resolved_model, resolved_provider = (
                    config_manager.resolve_model_profile(fallback_profile)
                )
                # Only use if it's a local/free provider (ollama)
                if resolved_provider == "ollama":
                    click.echo(
                        f"🔄 Using configured free local model '{fallback_profile}' "
                        f"→ {resolved_model}"
                    )
                    try:
                        return await self._load_model(resolved_model, resolved_provider)
                    except Exception as e:
                        click.echo(
                            f"❌ Failed to load configured fallback "
                            f"'{fallback_profile}': {e}"
                        )
                        # Don't raise immediately, try hardcoded fallback first
                else:
                    click.echo(
                        f"⚠️  Configured test profile '{fallback_profile}' uses "
                        f"{resolved_provider}, not ollama"
                    )
            except (ValueError, KeyError) as e:
                click.echo(
                    f"⚠️  Configured test profile '{fallback_profile}' not found: {e}"
                )

        # Last resort: hardcoded free local fallback
        click.echo("🔄 Using hardcoded fallback: llama3 via ollama")
        try:
            return await self._load_model("llama3", "ollama")
        except Exception as e:
            click.echo(f"❌ Hardcoded fallback model 'llama3' failed: {e}")
            # For tests and when Ollama is not available, return basic model
            # In production, this would indicate a serious configuration issue
            click.echo("🆘 Creating basic Ollama model as last resort")
            return OllamaModel(model_name="llama3")

    def _calculate_usage_summary(
        self, agent_usage: dict[str, Any], synthesis_usage: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Calculate aggregated usage summary."""
        summary = {
            "agents": agent_usage,
            "totals": {
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "total_duration_ms": 0,
                "agents_count": len(agent_usage),
            },
        }

        # Aggregate agent usage
        for usage in agent_usage.values():
            summary["totals"]["total_tokens"] += usage.get("total_tokens", 0)
            summary["totals"]["total_input_tokens"] += usage.get("input_tokens", 0)
            summary["totals"]["total_output_tokens"] += usage.get("output_tokens", 0)
            summary["totals"]["total_cost_usd"] += usage.get("cost_usd", 0.0)
            summary["totals"]["total_duration_ms"] += usage.get("duration_ms", 0)

        # Add synthesis usage
        if synthesis_usage:
            summary["synthesis"] = synthesis_usage
            summary["totals"]["total_tokens"] += synthesis_usage.get("total_tokens", 0)
            summary["totals"]["total_input_tokens"] += synthesis_usage.get(
                "input_tokens", 0
            )
            summary["totals"]["total_output_tokens"] += synthesis_usage.get(
                "output_tokens", 0
            )
            summary["totals"]["total_cost_usd"] += synthesis_usage.get("cost_usd", 0.0)
            summary["totals"]["total_duration_ms"] += synthesis_usage.get(
                "duration_ms", 0
            )

        return summary

    async def _execute_agent_with_timeout(
        self, agent_config: dict[str, Any], input_data: str, timeout_seconds: int | None
    ) -> tuple[str, ModelInterface | None]:
        """Execute an agent with optional timeout."""
        if timeout_seconds is None:
            # No timeout specified, execute normally
            return await self._execute_agent(agent_config, input_data)

        try:
            return await asyncio.wait_for(
                self._execute_agent(agent_config, input_data), timeout=timeout_seconds
            )
        except TimeoutError as e:
            raise Exception(
                f"Agent execution timed out after {timeout_seconds} seconds"
            ) from e

    async def _synthesize_results_with_timeout(
        self,
        config: EnsembleConfig,
        agent_results: dict[str, Any],
        timeout_seconds: int | None,
    ) -> tuple[str, ModelInterface]:
        """Synthesize results with optional timeout."""
        if timeout_seconds is None:
            # No timeout specified, execute normally
            return await self._synthesize_results(config, agent_results)

        try:
            return await asyncio.wait_for(
                self._synthesize_results(config, agent_results), timeout=timeout_seconds
            )
        except TimeoutError as e:
            raise Exception(
                f"Synthesis timed out after {timeout_seconds} seconds"
            ) from e

    def _analyze_dependencies(
        self, llm_agents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Analyze agent dependencies and return independent and dependent agents."""
        independent_agents = []
        dependent_agents = []

        for agent_config in llm_agents:
            dependencies = agent_config.get("dependencies", [])
            if dependencies and len(dependencies) > 0:
                dependent_agents.append(agent_config)
            else:
                independent_agents.append(agent_config)

        return independent_agents, dependent_agents

    async def _execute_agents_parallel(
        self,
        agents: list[dict[str, Any]],
        input_data: str,
        config: EnsembleConfig,
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> None:
        """Execute a list of agents in parallel."""
        if not agents:
            return

        # Execute all agents in parallel using asyncio.create_task for concurrent
        # execution
        try:

            async def execute_agent_task(
                agent_config: dict[str, Any],
            ) -> tuple[str, Any]:
                """Execute a single agent task."""
                agent_name = agent_config["name"]
                try:
                    # Resolve config and execute - all happening in parallel per agent
                    enhanced_config = await self._resolve_model_profile_to_config(
                        agent_config
                    )
                    timeout = enhanced_config.get("timeout_seconds") or (
                        config.coordinator.get("timeout_seconds")
                    )
                    result = await self._execute_agent_with_timeout(
                        agent_config, input_data, timeout
                    )
                    return agent_name, result
                except Exception as e:
                    # Record error in results dict and return error indicator
                    results_dict[agent_name] = {
                        "error": str(e),
                        "status": "failed",
                    }
                    return agent_name, None

            # Create tasks using create_task to ensure they start immediately
            tasks = [
                asyncio.create_task(execute_agent_task(agent_config))
                for agent_config in agents
            ]

            # Wait for all tasks to complete
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            for execution_result in agent_results:
                if isinstance(execution_result, Exception):
                    # If we can't determine agent name from exception, skip this result
                    # The agent task should handle its own error recording
                    continue
                else:
                    # execution_result is tuple[str, Any]
                    agent_name, result = execution_result  # type: ignore[misc]
                    if result is None:
                        # Error was already recorded in create_agent_task
                        continue

                    # result is tuple[str, ModelInterface | None]
                    response, model_instance = result
                    results_dict[agent_name] = {
                        "response": response,
                        "status": "success",
                    }
                    # Collect usage metrics (only for LLM agents)
                    if model_instance is not None:
                        usage = model_instance.get_last_usage()
                        if usage:
                            agent_usage[agent_name] = usage
        except Exception as e:
            # Fallback: if gather fails, mark all agents as failed
            for agent_config in agents:
                agent_name = agent_config["name"]
                results_dict[agent_name] = {"error": str(e), "status": "failed"}

    def _enhance_input_with_dependencies(
        self,
        base_input: str,
        dependent_agents: list[dict[str, Any]],
        results_dict: dict[str, Any],
    ) -> str:
        """Enhance input with dependency results for dependent agents."""
        # For now, use the same enhanced input for all dependent agents
        # In future iterations, we could customize input per agent based on dependencies
        dependency_results = []

        for agent_config in dependent_agents:
            dependencies = agent_config.get("dependencies", [])
            for dep_name in dependencies:
                if (
                    dep_name in results_dict
                    and results_dict[dep_name].get("status") == "success"
                ):
                    response = results_dict[dep_name]["response"]
                    dependency_results.append(f"=== {dep_name} ===\n{response}")

        if dependency_results:
            deps_text = "\n\n".join(dependency_results)
            return f"{base_input}\n\nDependency Results:\n{deps_text}"

        return base_input

    def _analyze_enhanced_dependency_graph(
        self, agent_configs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze agent dependencies and organize into execution phases.

        Uses topological sort to determine optimal execution order for agents
        with dependencies. Agents with no dependencies can run in parallel.

        Args:
            agent_configs: List of agent configurations with optional dependencies

        Returns:
            Dictionary containing:
            - phases: List of agent lists, each phase can run in parallel
            - dependency_map: Mapping of agent names to their dependencies
            - total_phases: Number of execution phases

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build dependency map for efficient lookup
        dependency_map = {
            agent_config["name"]: agent_config.get("dependencies", [])
            for agent_config in agent_configs
        }

        # Topological sort to determine execution phases
        phases = []
        remaining_agents = agent_configs.copy()
        processed_agents: set[str] = set()

        while remaining_agents:
            current_phase = []
            agents_to_remove = []

            # Find agents whose dependencies have been processed
            for agent_config in remaining_agents:
                agent_name = agent_config["name"]
                dependencies = dependency_map[agent_name]

                # Agent is ready if it has no dependencies or all are processed
                if self._agent_dependencies_satisfied(dependencies, processed_agents):
                    current_phase.append(agent_config)
                    agents_to_remove.append(agent_config)

            # Update processed agents and remove from remaining
            for agent_config in agents_to_remove:
                processed_agents.add(agent_config["name"])
                remaining_agents.remove(agent_config)

            # Detect circular dependencies
            if not current_phase:
                raise ValueError("Circular dependency detected in agent configuration")

            phases.append(current_phase)

        return {
            "phases": phases,
            "dependency_map": dependency_map,
            "total_phases": len(phases),
        }

    def _agent_dependencies_satisfied(
        self, dependencies: list[str], processed_agents: set[str]
    ) -> bool:
        """Check if an agent's dependencies have been processed."""
        return len(dependencies) == 0 or all(
            dep in processed_agents for dep in dependencies
        )


def _should_prompt_for_auth(model_name: str) -> bool:
    """Determine if we should prompt for authentication setup."""
    # Don't prompt for mock models or generic model names
    if model_name.startswith("mock") or model_name in ["llama3", "llama2"]:
        return False

    # Prompt for known authentication configurations
    known_auth_configs = [
        "anthropic-api",
        "anthropic-claude-pro-max",
        "claude-cli",
        "openai-api",
        "google-api",
    ]

    return model_name in known_auth_configs


def _prompt_auth_setup(model_name: str, storage: CredentialStorage) -> bool:
    """Prompt user to set up authentication for the specified model."""
    # Ask if user wants to set up authentication
    if not click.confirm(
        f"Authentication not configured for '{model_name}'. "
        f"Would you like to set it up now?"
    ):
        return False

    try:
        auth_manager = AuthenticationManager(storage)

        # Handle different authentication types
        if model_name == "anthropic-api":
            return _setup_anthropic_api_auth(storage)
        elif model_name == "anthropic-claude-pro-max":
            return _setup_anthropic_oauth_auth(auth_manager, model_name)
        elif model_name == "claude-cli":
            return _setup_claude_cli_auth(storage)
        else:
            click.echo(f"Don't know how to set up authentication for '{model_name}'")
            return False

    except Exception as e:
        click.echo(f"Failed to set up authentication: {str(e)}")
        return False


def _setup_anthropic_api_auth(storage: CredentialStorage) -> bool:
    """Set up Anthropic API key authentication."""
    api_key = click.prompt("Enter your Anthropic API key", hide_input=True)
    storage.store_api_key("anthropic-api", api_key)
    click.echo("✅ Anthropic API key configured")
    return True


def _setup_anthropic_oauth_auth(
    auth_manager: AuthenticationManager, provider_key: str
) -> bool:
    """Set up Anthropic OAuth authentication."""
    try:
        from llm_orc.authentication import AnthropicOAuthFlow

        oauth_flow = AnthropicOAuthFlow.create_with_guidance()

        if auth_manager.authenticate_oauth(
            provider_key, oauth_flow.client_id, oauth_flow.client_secret
        ):
            click.echo("✅ Anthropic OAuth configured")
            return True
        else:
            click.echo("❌ OAuth authentication failed")
            return False

    except Exception as e:
        click.echo(f"❌ OAuth setup failed: {str(e)}")
        return False


def _setup_claude_cli_auth(storage: CredentialStorage) -> bool:
    """Set up Claude CLI authentication."""
    import shutil

    # Check if claude command is available
    claude_path = shutil.which("claude")
    if not claude_path:
        click.echo("❌ Claude CLI not found. Please install the Claude CLI from:")
        click.echo("   https://docs.anthropic.com/en/docs/claude-code")
        return False

    # Store claude-cli configuration
    storage.store_api_key("claude-cli", claude_path)
    click.echo("✅ Claude CLI authentication configured")
    click.echo(f"   Using local claude command at: {claude_path}")
    return True
