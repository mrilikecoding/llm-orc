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
                timeout = agent_config.get("timeout_seconds") or config.coordinator.get(
                    "timeout_seconds"
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

        # Phase 2: Execute LLM agents with context from script agents
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

        # Execute LLM agents concurrently with enhanced input
        agent_tasks = []
        for agent_config in llm_agents:
            timeout = agent_config.get("timeout_seconds") or config.coordinator.get(
                "timeout_seconds"
            )
            task = self._execute_agent_with_timeout(
                agent_config, enhanced_input, timeout
            )
            agent_tasks.append((agent_config["name"], task))

        # Wait for all LLM agents to complete
        for agent_name, task in agent_tasks:
            try:
                agent_result, model_instance = await task
                results_dict[agent_name] = {
                    "response": agent_result,
                    "status": "success",
                }
                # Collect usage metrics (only for LLM agents)
                if model_instance is not None:
                    usage = model_instance.get_last_usage()
                    if usage:
                        agent_usage[agent_name] = usage
            except Exception as e:
                results_dict[agent_name] = {"error": str(e), "status": "failed"}
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

        # Use system_prompt from config if available, otherwise use fallback
        if "system_prompt" in agent_config:
            prompt = agent_config["system_prompt"]
        else:
            prompt = f"You are a {agent_name}. Provide helpful analysis."

        return RoleDefinition(name=agent_name, prompt=prompt)

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
        config_manager = ConfigurationManager()

        # Check if model_profile is specified (takes precedence)
        if "model_profile" in agent_config:
            profile_name = agent_config["model_profile"]
            resolved_model, resolved_provider = config_manager.resolve_model_profile(
                profile_name
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

        # Initialize configuration and credential storage
        config_manager = ConfigurationManager()
        storage = CredentialStorage(config_manager)

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
        """Get a fallback model based on configured defaults."""
        # Load project configuration to get default models
        config_manager = ConfigurationManager()
        project_config = config_manager.load_project_config()

        default_models = project_config.get("project", {}).get("default_models", {})

        # Choose fallback model based on context
        if context == "coordinator":
            # For coordinators, prefer quality > test > hardcoded fallback
            fallback_model = (
                default_models.get("quality")
                or default_models.get("test")
                or "llama3"
            )
        else:
            # For general use, prefer test > quality > hardcoded fallback
            fallback_model = (
                default_models.get("test")
                or default_models.get("quality")
                or "llama3"
            )

        try:
            click.echo(
                f"🔄 Using fallback model '{fallback_model}' (from configured defaults)"
            )
            return await self._load_model(fallback_model, "ollama")
        except Exception as e:
            # Last resort: hardcoded Ollama fallback
            click.echo(f"❌ Fallback model '{fallback_model}' failed to load: {str(e)}")
            click.echo(
                "🆘 Using hardcoded fallback: llama3 "
                "(consider configuring default_models: test/quality)"
            )
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
