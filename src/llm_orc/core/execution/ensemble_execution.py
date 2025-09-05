"""Ensemble execution with agent coordination."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from llm_orc.agents.enhanced_script_agent import EnhancedScriptAgent
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.core.execution.agent_execution_coordinator import AgentExecutionCoordinator
from llm_orc.core.execution.agent_executor import AgentExecutor
from llm_orc.core.execution.dependency_analyzer import DependencyAnalyzer
from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.core.execution.input_enhancer import InputEnhancer
from llm_orc.core.execution.orchestration import Agent
from llm_orc.core.execution.results_processor import ResultsProcessor
from llm_orc.core.execution.streaming_progress_tracker import StreamingProgressTracker
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.models.base import ModelInterface


class EnsembleExecutor:
    """Executes ensembles of agents and coordinates their responses."""

    def __init__(self) -> None:
        """Initialize the ensemble executor with shared infrastructure."""
        # Share configuration and credential infrastructure across model loads
        # but keep model instances separate for independent contexts
        self._config_manager = ConfigurationManager()
        self._credential_storage = CredentialStorage(self._config_manager)

        # Load performance configuration
        self._performance_config = self._config_manager.load_performance_config()

        # Phase 5: Unified event system - shared event queue for streaming
        self._streaming_event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Initialize extracted components
        self._model_factory = ModelFactory(
            self._config_manager, self._credential_storage
        )
        self._dependency_analyzer = DependencyAnalyzer()
        self._dependency_resolver = DependencyResolver(self._get_agent_role_description)
        self._input_enhancer = InputEnhancer()
        self._usage_collector = UsageCollector()
        self._results_processor = ResultsProcessor()
        self._streaming_progress_tracker = StreamingProgressTracker()

        # Initialize execution coordinator with agent executor function
        # Use a wrapper to avoid circular dependency with _execute_agent_with_timeout
        async def agent_executor_wrapper(
            agent_config: dict[str, Any], input_data: str
        ) -> tuple[str, ModelInterface | None]:
            return await self._execute_agent(agent_config, input_data)

        self._execution_coordinator = AgentExecutionCoordinator(
            self._performance_config, agent_executor_wrapper
        )

        # Note: AgentOrchestrator not used in current simplified implementation

        # Keep existing agent executor for backward compatibility
        self._agent_executor = AgentExecutor(
            self._performance_config,
            self._emit_performance_event,
            self._resolve_model_profile_to_config,
            self._execute_agent_with_timeout,
            self._input_enhancer.get_agent_input,
        )

    async def _load_model_from_agent_config(
        self, agent_config: dict[str, Any]
    ) -> ModelInterface:
        """Delegate to model factory."""
        return await self._model_factory.load_model_from_agent_config(agent_config)

    # Phase 5: Performance hooks system removed - events go directly to streaming queue

    def _emit_performance_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit performance monitoring events to unified streaming queue.

        Phase 5: Events go directly to streaming queue instead of hooks.
        This eliminates the dual event system architecture.
        """
        event = {
            "type": event_type,
            "data": data,
        }

        # Put event in queue (non-blocking)
        try:
            self._streaming_event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # Silently ignore if queue is full to avoid breaking execution
            pass

    def _classify_failure_type(self, error_message: str) -> str:
        """Classify failure type based on error message for enhanced events.

        Args:
            error_message: The error message to classify

        Returns:
            Failure type: 'oauth_error', 'authentication_error', 'model_loading',
            or 'runtime_error'
        """
        error_lower = error_message.lower()

        # OAuth-specific errors
        if any(
            oauth_term in error_lower
            for oauth_term in [
                "oauth",
                "token refresh",
                "invalid_grant",
                "refresh token",
            ]
        ):
            return "oauth_error"

        # Authentication errors (API keys, etc.)
        if any(
            auth_term in error_lower
            for auth_term in [
                "authentication",
                "invalid x-api-key",
                "unauthorized",
                "401",
            ]
        ):
            return "authentication_error"

        # Model loading errors
        if any(
            loading_term in error_lower
            for loading_term in [
                "model loading",
                "failed to load model",
                "network error",
                "connection failed",
                "timeout",
                "not found",
                "not available",
                "model provider",
            ]
        ):
            return "model_loading"

        # Default to runtime error
        return "runtime_error"

    async def execute_streaming(
        self, config: EnsembleConfig, input_data: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute ensemble with streaming progress updates.

        Yields progress events during execution for real-time monitoring.
        Events include: execution_started, agent_progress, execution_completed,
        agent_fallback_started, agent_fallback_completed, agent_fallback_failed.

        Phase 5: Unified event system - merges progress and performance events.
        """
        # Clear the event queue before starting
        while not self._streaming_event_queue.empty():
            try:
                self._streaming_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Use StreamingProgressTracker for execution tracking
        start_time = time.time()
        execution_task = asyncio.create_task(self.execute(config, input_data))

        # Merge events from progress tracker and performance queue
        async for event in self._merge_streaming_events(
            self._streaming_progress_tracker.track_execution_progress(
                config, execution_task, start_time
            )
        ):
            yield event

    async def _merge_streaming_events(
        self, progress_events: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Merge progress events with performance events from the unified queue.

        Phase 5: This eliminates the dual event system by combining both streams.
        """
        try:
            async for progress_event in progress_events:
                yield progress_event

                # Yield any accumulated performance events
                async for perf_event in self._yield_queued_performance_events():
                    yield perf_event

                # Small delay to allow any concurrent performance events to be queued
                await asyncio.sleep(0.001)

                # Yield performance events again after delay
                async for perf_event in self._yield_queued_performance_events():
                    yield perf_event

                # If execution is completed, mark progress as done
                if progress_event.get("type") == "execution_completed":
                    break
        except Exception:
            pass

        # After progress is done, yield any remaining performance events
        async for perf_event in self._yield_queued_performance_events():
            yield perf_event

    async def _yield_queued_performance_events(
        self,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield all currently queued performance events."""
        while not self._streaming_event_queue.empty():
            try:
                performance_event = self._streaming_event_queue.get_nowait()
                yield performance_event
            except asyncio.QueueEmpty:
                break

    async def execute(self, config: EnsembleConfig, input_data: str) -> dict[str, Any]:
        """Execute an ensemble and return structured results."""
        start_time = time.time()

        # Store agent configs for role descriptions
        self._current_agent_configs = config.agents

        # Initialize result structure using ResultsProcessor
        result = self._results_processor.create_initial_result(
            config.name, input_data, len(config.agents)
        )
        results_dict: dict[str, Any] = result["results"]

        # Reset usage collector for this execution
        self._usage_collector.reset()

        # Execute agents in phases: script agents first, then LLM agents
        has_errors = False
        context_data: dict[str, Any] = {}

        # Phase 1: Execute script agents to gather context
        context_data, script_errors = await self._execute_script_agents(
            config, input_data, results_dict
        )
        has_errors = has_errors or script_errors

        # Phase 2: Execute LLM agents with dependency-aware phasing
        llm_agent_errors = await self._execute_llm_agents(
            config, input_data, context_data, results_dict
        )
        has_errors = has_errors or llm_agent_errors

        # Get collected usage and adaptive stats, then finalize result using processor
        agent_usage = self._usage_collector.get_agent_usage()
        adaptive_stats = self._agent_executor.get_adaptive_stats()
        return self._results_processor.finalize_result(
            result, agent_usage, has_errors, start_time, adaptive_stats
        )

    async def _execute_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute a single agent and return its response and model instance.

        Agent type is determined implicitly based on configuration fields:
        - Has 'script' field -> Script agent
        - Has 'model_profile' field -> LLM agent
        - Has explicit 'type' field -> Use that (backward compatibility)
        """
        # Check for explicit type first (backward compatibility)
        explicit_type = agent_config.get("type")

        if explicit_type == "script":
            return await self._execute_script_agent(agent_config, input_data)
        elif explicit_type == "llm":
            return await self._execute_llm_agent(agent_config, input_data)

        # Implicit type detection based on fields present
        if "script" in agent_config:
            return await self._execute_script_agent(agent_config, input_data)
        elif "model_profile" in agent_config:
            return await self._execute_llm_agent(agent_config, input_data)
        else:
            agent_name = agent_config.get("name", "unknown")
            raise ValueError(
                f"Agent '{agent_name}' must have either 'script' or 'model_profile'"
            )

    async def _execute_script_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute script agent with resource monitoring using EnhancedScriptAgent."""
        agent_name = agent_config["name"]

        # Start resource monitoring for this agent
        self._usage_collector.start_agent_resource_monitoring(agent_name)

        try:
            # Use EnhancedScriptAgent for JSON I/O support
            script_agent = EnhancedScriptAgent(agent_name, agent_config)

            # Sample resources during execution
            self._usage_collector.sample_agent_resources(agent_name)

            response = await script_agent.execute(input_data)

            # Final sample before completion
            self._usage_collector.sample_agent_resources(agent_name)

            # Convert response to string if it's a dict (JSON output)
            if isinstance(response, dict):
                import json

                response = json.dumps(response)

            return response, None  # Script agents don't have model instances
        finally:
            # Always finalize resource monitoring
            self._usage_collector.finalize_agent_resource_monitoring(agent_name)

    async def _execute_llm_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute LLM agent with fallback handling and resource monitoring."""
        agent_name = agent_config["name"]

        # Start resource monitoring for this agent
        self._usage_collector.start_agent_resource_monitoring(agent_name)

        try:
            role = await self._load_role_from_config(agent_config)
            model = await self._load_model_with_fallback(agent_config)
            agent = Agent(agent_name, role, model)

            # Take periodic resource samples during execution
            self._usage_collector.sample_agent_resources(agent_name)

            # Generate response with fallback handling for runtime failures
            try:
                response = await agent.respond_to_message(input_data)

                # Final resource sample before completing
                self._usage_collector.sample_agent_resources(agent_name)

                return response, model
            except Exception as e:
                return await self._handle_runtime_fallback(
                    agent_config, role, input_data, e
                )
        finally:
            # Always finalize resource monitoring, even if execution failed
            self._usage_collector.finalize_agent_resource_monitoring(agent_name)

    async def _load_model_with_fallback(
        self, agent_config: dict[str, Any]
    ) -> ModelInterface:
        """Load model with fallback handling for loading failures."""
        try:
            return await self._model_factory.load_model_from_agent_config(agent_config)
        except Exception as model_loading_error:
            return await self._handle_model_loading_fallback(
                agent_config, model_loading_error
            )

    async def _handle_model_loading_fallback(
        self, agent_config: dict[str, Any], model_loading_error: Exception
    ) -> ModelInterface:
        """Handle model loading failure with fallback."""
        fallback_model = await self._model_factory.get_fallback_model(
            context=f"agent_{agent_config['name']}",
            original_profile=agent_config.get("model_profile"),
        )
        fallback_model_name = getattr(fallback_model, "model_name", "unknown")

        # Emit enhanced fallback event for model loading failure
        failure_type = self._classify_failure_type(str(model_loading_error))
        self._emit_performance_event(
            "agent_fallback_started",
            {
                "agent_name": agent_config["name"],
                "failure_type": failure_type,
                "original_error": str(model_loading_error),
                "original_model_profile": agent_config.get("model_profile", "unknown"),
                "fallback_model_profile": None,  # No configurable fallback
                "fallback_model_name": fallback_model_name,
            },
        )
        return fallback_model

    async def _handle_runtime_fallback(
        self,
        agent_config: dict[str, Any],
        role: RoleDefinition,
        input_data: str,
        error: Exception,
    ) -> tuple[str, ModelInterface]:
        """Handle runtime failure with fallback model."""
        fallback_model = await self._model_factory.get_fallback_model(
            context=f"agent_{agent_config['name']}"
        )
        fallback_model_name = getattr(fallback_model, "model_name", "unknown")

        # Emit enhanced fallback event for runtime failure
        failure_type = self._classify_failure_type(str(error))
        self._emit_performance_event(
            "agent_fallback_started",
            {
                "agent_name": agent_config["name"],
                "failure_type": failure_type,
                "original_error": str(error),
                "original_model_profile": agent_config.get("model_profile", "unknown"),
                "fallback_model_profile": None,  # No configurable fallback
                "fallback_model_name": fallback_model_name,
            },
        )

        # Create new agent with fallback model
        fallback_agent = Agent(agent_config["name"], role, fallback_model)

        # Try with fallback model
        try:
            response = await fallback_agent.respond_to_message(input_data)
            self._emit_fallback_success_event(
                agent_config["name"], fallback_model, response
            )
            return response, fallback_model
        except Exception as fallback_error:
            self._emit_fallback_failure_event(
                agent_config["name"], fallback_model_name, fallback_error
            )
            raise fallback_error

    def _emit_fallback_success_event(
        self, agent_name: str, fallback_model: ModelInterface, response: str
    ) -> None:
        """Emit fallback success event."""
        fallback_model_name = getattr(fallback_model, "model_name", "unknown")
        response_preview = response[:100] + "..." if len(response) > 100 else response
        self._emit_performance_event(
            "agent_fallback_completed",
            {
                "agent_name": agent_name,
                "fallback_model_name": fallback_model_name,
                "response_preview": response_preview,
            },
        )

    def _emit_fallback_failure_event(
        self, agent_name: str, fallback_model_name: str, fallback_error: Exception
    ) -> None:
        """Emit fallback failure event."""
        fallback_failure_type = self._classify_failure_type(str(fallback_error))
        self._emit_performance_event(
            "agent_fallback_failed",
            {
                "agent_name": agent_name,
                "failure_type": fallback_failure_type,
                "fallback_error": str(fallback_error),
                "fallback_model_name": fallback_model_name,
            },
        )

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

    async def _execute_script_agents(
        self,
        config: EnsembleConfig,
        input_data: str,
        results_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Execute script agents and return context data and error status."""
        context_data = {}
        has_errors = False
        # Identify script agents using implicit detection
        script_agents = [
            a
            for a in config.agents
            if a.get("type") == "script" or ("script" in a and "model_profile" not in a)
        ]

        for agent_config in script_agents:
            try:
                # Resolve model profile to get enhanced configuration
                enhanced_config = await self._resolve_model_profile_to_config(
                    agent_config
                )
                timeout = enhanced_config.get("timeout_seconds") or (
                    self._performance_config.get("execution", {}).get(
                        "default_timeout", 60
                    )
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

                # Collect usage for script agents
                model_profile = agent_config.get("model_profile", "unknown")
                self._usage_collector.collect_agent_usage(
                    agent_config["name"], model_instance, model_profile
                )
            except Exception as e:
                results_dict[agent_config["name"]] = {
                    "error": str(e),
                    "status": "failed",
                }
                has_errors = True

        return context_data, has_errors

    async def _execute_agents_in_phase_parallel(
        self, phase_agents: list[dict[str, Any]], phase_input: str | dict[str, str]
    ) -> dict[str, Any]:
        """Execute agents in parallel within a phase.

        Based on Issue #43 analysis, this provides 3-15x performance improvement
        for I/O bound LLM API calls using asyncio.gather().
        """

        async def execute_single_agent(
            agent_config: dict[str, Any],
        ) -> tuple[str, dict[str, Any]]:
            """Execute a single agent and return (name, result)."""
            agent_name = agent_config["name"]

            try:
                agent_start_time = time.time()

                # Emit agent started event
                self._emit_performance_event(
                    "agent_started",
                    {"agent_name": agent_name, "timestamp": agent_start_time},
                )

                # Get agent input
                agent_input = self._input_enhancer.get_agent_input(
                    phase_input, agent_name
                )

                # Get timeout from enhanced config
                enhanced_config = await self._resolve_model_profile_to_config(
                    agent_config
                )
                timeout = enhanced_config.get("timeout_seconds") or (
                    self._performance_config.get("execution", {}).get(
                        "default_timeout", 60
                    )
                )

                # Execute agent with timeout coordination
                coordinator = self._execution_coordinator
                response, model_instance = await coordinator.execute_agent_with_timeout(
                    agent_config, agent_input, timeout
                )

                # Emit agent completed event with duration
                agent_end_time = time.time()
                duration_ms = int((agent_end_time - agent_start_time) * 1000)
                self._emit_performance_event(
                    "agent_completed",
                    {
                        "agent_name": agent_name,
                        "timestamp": agent_end_time,
                        "duration_ms": duration_ms,
                    },
                )

                return agent_name, {
                    "response": response,
                    "status": "success",
                    "model_instance": model_instance,
                }

            except Exception as e:
                # Handle agent failure
                agent_end_time = time.time()
                duration_ms = int((agent_end_time - agent_start_time) * 1000)
                self._emit_performance_event(
                    "agent_completed",
                    {
                        "agent_name": agent_name,
                        "timestamp": agent_end_time,
                        "duration_ms": duration_ms,
                        "error": str(e),
                    },
                )

                return agent_name, {
                    "error": str(e),
                    "status": "failed",
                    "model_instance": None,
                }

        # Execute all agents in parallel using asyncio.gather
        tasks = [execute_single_agent(agent_config) for agent_config in phase_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        phase_results: dict[str, Any] = {}
        for result in results:
            if isinstance(result, BaseException):
                # Handle unexpected errors during gather
                # This should not happen since we catch exceptions in execute_single_agent  # noqa: E501
                continue
            # At this point, result should be a tuple[str, dict[str, Any]]
            agent_name, agent_result = result
            phase_results[agent_name] = agent_result

        return phase_results

    def _prepare_enhanced_input(
        self, input_data: str, config: EnsembleConfig, context_data: dict[str, Any]
    ) -> str:
        """Prepare enhanced input for LLM agents with context data."""
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

        return enhanced_input

    def _process_phase_results(
        self,
        phase_results: dict[str, Any],
        results_dict: dict[str, Any],
        phase_agents: list[dict[str, Any]],
    ) -> bool:
        """Process parallel execution results and return if any errors occurred."""
        has_errors = False

        # Create agent lookup for model profile information
        agent_configs = {agent["name"]: agent for agent in phase_agents}

        for agent_name, agent_result in phase_results.items():
            # Store result in results_dict
            results_dict[agent_name] = {
                "response": agent_result.get("response"),
                "status": agent_result["status"],
            }

            # Handle errors
            if agent_result["status"] == "failed":
                has_errors = True
                results_dict[agent_name]["error"] = agent_result["error"]

            # Collect usage for successful agents
            if (
                agent_result["status"] == "success"
                and agent_result["model_instance"] is not None
            ):
                # Get model profile from agent config
                agent_config = agent_configs.get(agent_name, {})
                model_profile = agent_config.get("model_profile", "unknown")

                self._usage_collector.collect_agent_usage(
                    agent_name, agent_result["model_instance"], model_profile
                )

        return has_errors

    def _emit_phase_completed_event(
        self,
        phase_index: int,
        phase_agents: list[dict[str, Any]],
        results_dict: dict[str, Any],
    ) -> None:
        """Emit phase completion event with success/failure counts."""
        successful_agents = [
            a
            for a in phase_agents
            if results_dict.get(a["name"], {}).get("status") == "success"
        ]
        failed_agents = [
            a
            for a in phase_agents
            if results_dict.get(a["name"], {}).get("status") == "failed"
        ]

        self._emit_performance_event(
            "phase_completed",
            {
                "phase_index": phase_index,
                "successful_agents": len(successful_agents),
                "failed_agents": len(failed_agents),
            },
        )

    async def _execute_llm_agents(
        self,
        config: EnsembleConfig,
        input_data: str,
        context_data: dict[str, Any],
        results_dict: dict[str, Any],
    ) -> bool:
        """Execute LLM agents with dependency-aware phasing."""
        has_errors = False
        # Identify LLM agents using implicit detection
        llm_agents = [
            a
            for a in config.agents
            if (
                a.get("type") not in ["script", None]  # Explicit LLM type
                or ("model_profile" in a and "script" not in a)
            )  # Implicit LLM via model_profile
        ]

        # Prepare enhanced input for LLM agents
        enhanced_input = self._prepare_enhanced_input(input_data, config, context_data)

        # Use enhanced dependency analysis for multi-level execution
        if llm_agents:
            # Update input enhancer with current agent configs for role descriptions
            self._input_enhancer.update_agent_configs(llm_agents)
            dependency_analysis = (
                self._dependency_analyzer.analyze_enhanced_dependency_graph(llm_agents)
            )
            phases = dependency_analysis["phases"]

            # Execute phases with monitoring hooks
            has_errors = await self._execute_phases_standard(
                phases, enhanced_input, results_dict, has_errors
            )

        return has_errors

    async def _execute_phases_standard(
        self,
        phases: list[list[dict[str, Any]]],
        enhanced_input: str,
        results_dict: dict[str, Any],
        has_errors: bool,
    ) -> bool:
        """Execute phases using standard approach without per-phase monitoring."""
        for phase_index, phase_agents in enumerate(phases):
            self._emit_performance_event(
                "phase_started",
                {
                    "phase_index": phase_index,
                    "phase_agents": [agent["name"] for agent in phase_agents],
                    "total_phases": len(phases),
                },
            )

            # Determine input for this phase using DependencyResolver
            if phase_index == 0:
                # First phase uses the base enhanced input
                phase_input: str | dict[str, str] = enhanced_input
            else:
                # Subsequent phases get enhanced input with dependencies
                phase_input = self._dependency_resolver.enhance_input_with_dependencies(
                    enhanced_input, phase_agents, results_dict
                )

            # Start per-phase monitoring for performance feedback
            phase_start_time = time.time()
            await self._start_phase_monitoring(phase_index, phase_agents)

            try:
                # Execute agents in this phase in parallel (Issue #43 implementation)
                phase_results = await self._execute_agents_in_phase_parallel(
                    phase_agents, phase_input
                )

                # Process parallel execution results
                phase_has_errors = self._process_phase_results(
                    phase_results, results_dict, phase_agents
                )
                has_errors = has_errors or phase_has_errors

            finally:
                # Stop per-phase monitoring and collect metrics
                phase_duration = time.time() - phase_start_time
                await self._stop_phase_monitoring(
                    phase_index, phase_agents, phase_duration
                )

            # Emit phase completion event
            self._emit_phase_completed_event(phase_index, phase_agents, results_dict)

        return has_errors

    async def _execute_agent_with_timeout(
        self, agent_config: dict[str, Any], input_data: str, timeout_seconds: int | None
    ) -> tuple[str, ModelInterface | None]:
        """Execute agent with timeout using the extracted coordinator."""
        return await self._execution_coordinator.execute_agent_with_timeout(
            agent_config, input_data, timeout_seconds
        )

    def _analyze_dependencies(
        self, llm_agents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Analyze agent dependencies and return independent and dependent agents."""
        independent_agents = []
        dependent_agents = []

        for agent_config in llm_agents:
            dependencies = agent_config.get("depends_on", [])
            if dependencies and len(dependencies) > 0:
                dependent_agents.append(agent_config)
            else:
                independent_agents.append(agent_config)

        return independent_agents, dependent_agents

    def _get_agent_role_description(self, agent_name: str) -> str | None:
        """Get a human-readable role description for an agent."""
        # Try to find the agent in the current ensemble config
        if hasattr(self, "_current_agent_configs"):
            for agent_config in self._current_agent_configs:
                if agent_config["name"] == agent_name:
                    # Try model_profile first, then infer from name
                    if "model_profile" in agent_config:
                        profile = str(agent_config["model_profile"])
                        # Convert kebab-case to title case
                        return profile.replace("-", " ").title()
                    else:
                        # Convert agent name to readable format
                        return agent_name.replace("-", " ").title()

        # Fallback: convert name to readable format
        return agent_name.replace("-", " ").title()

    async def _start_phase_monitoring(
        self, phase_index: int, phase_agents: list[dict[str, Any]]
    ) -> None:
        """Start monitoring for a specific phase."""
        # Phase metrics are always initialized in AgentExecutor

        # Collect initial phase metrics
        phase_name = f"phase_{phase_index}"
        agent_names = [agent["name"] for agent in phase_agents]

        phase_metrics = await self._agent_executor.monitor.collect_phase_metrics(
            phase_index=phase_index,
            phase_name=phase_name,
            agent_count=len(phase_agents),
        )

        # Add agent details
        phase_metrics.update(
            {
                "agent_names": agent_names,
                "start_time": time.time(),
            }
        )

        self._agent_executor._phase_metrics.append(phase_metrics)

        # Start continuous monitoring for this phase
        await self._agent_executor.monitor.start_execution_monitoring()

        self._emit_performance_event(
            "phase_monitoring_started",
            {
                "phase_index": phase_index,
                "agent_count": len(phase_agents),
                "agent_names": agent_names,
            },
        )

    async def _stop_phase_monitoring(
        self, phase_index: int, phase_agents: list[dict[str, Any]], duration: float
    ) -> None:
        """Stop monitoring for a specific phase and collect final metrics."""

        # Find the phase metrics entry
        phase_metrics = None
        for metrics in self._agent_executor._phase_metrics:
            if metrics.get("phase_index") == phase_index:
                phase_metrics = metrics
                break

        if phase_metrics:
            # Stop monitoring and get aggregated metrics for this phase
            try:
                phase_execution_metrics = await (
                    self._agent_executor.monitor.stop_execution_monitoring()
                )

                # Update with completion data and monitoring results
                phase_metrics.update(
                    {
                        "duration_seconds": duration,
                        "end_time": time.time(),
                        "agents_completed": len(phase_agents),
                        # Add aggregated monitoring data
                        "peak_cpu": phase_execution_metrics.get("peak_cpu", 0.0),
                        "avg_cpu": phase_execution_metrics.get("avg_cpu", 0.0),
                        "peak_memory": phase_execution_metrics.get("peak_memory", 0.0),
                        "avg_memory": phase_execution_metrics.get("avg_memory", 0.0),
                        "sample_count": phase_execution_metrics.get("sample_count", 0),
                    }
                )
            except Exception:
                # Fallback to current snapshot if continuous monitoring fails
                try:
                    current_metrics = await (
                        self._agent_executor.monitor.get_current_metrics()
                    )
                    phase_metrics.update(
                        {
                            "duration_seconds": duration,
                            "end_time": time.time(),
                            "agents_completed": len(phase_agents),
                            "final_cpu_percent": current_metrics.get(
                                "cpu_percent", 0.0
                            ),
                            "final_memory_percent": current_metrics.get(
                                "memory_percent", 0.0
                            ),
                        }
                    )
                except Exception:
                    # Final fallback - just timing data
                    phase_metrics.update(
                        {
                            "duration_seconds": duration,
                            "end_time": time.time(),
                            "agents_completed": len(phase_agents),
                        }
                    )

        self._emit_performance_event(
            "phase_monitoring_stopped",
            {
                "phase_index": phase_index,
                "duration_seconds": duration,
                "agent_count": len(phase_agents),
            },
        )
