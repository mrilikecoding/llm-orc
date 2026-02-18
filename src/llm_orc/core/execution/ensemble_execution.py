"""Ensemble execution with agent coordination."""

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from llm_orc.agents.enhanced_script_agent import EnhancedScriptAgent
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.core.execution.agent_execution_coordinator import AgentExecutionCoordinator
from llm_orc.core.execution.agent_executor import AgentExecutor
from llm_orc.core.execution.agent_request_processor import AgentRequestProcessor
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.dependency_analyzer import DependencyAnalyzer
from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.core.execution.fan_out_expander import FanOutExpander
from llm_orc.core.execution.fan_out_gatherer import FanOutGatherer
from llm_orc.core.execution.orchestration import Agent
from llm_orc.core.execution.progress_controller import NoOpProgressController
from llm_orc.core.execution.results_processor import ResultsProcessor
from llm_orc.core.execution.script_cache import ScriptCache, ScriptCacheConfig
from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler
from llm_orc.core.execution.streaming_progress_tracker import StreamingProgressTracker
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.core.validation import (
    EnsembleExecutionResult,
    ValidationConfig,
    ValidationEvaluator,
)
from llm_orc.models.base import ModelInterface


class EnsembleExecutor:
    """Executes ensembles of agents and coordinates their responses."""

    def __init__(self, project_dir: Path | None = None) -> None:
        """Initialize the ensemble executor with shared infrastructure."""
        self._project_dir = project_dir

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
        self._usage_collector = UsageCollector()
        self._results_processor = ResultsProcessor()
        self._streaming_progress_tracker = StreamingProgressTracker()
        self._artifact_manager = ArtifactManager()
        self._progress_controller = NoOpProgressController()
        self._agent_request_processor = AgentRequestProcessor(self._dependency_resolver)

        # Fan-out support (issue #73)
        self._fan_out_expander = FanOutExpander()
        self._fan_out_gatherer = FanOutGatherer(self._fan_out_expander)

        # Initialize execution coordinator with agent executor function
        # Use a wrapper to avoid circular dependency with _execute_agent_with_timeout
        async def agent_executor_wrapper(
            agent_config: dict[str, Any], input_data: str
        ) -> tuple[str, ModelInterface | None]:
            return await self._execute_agent(agent_config, input_data)

        self._execution_coordinator = AgentExecutionCoordinator(
            self._performance_config, agent_executor_wrapper
        )

        self._agent_executor = AgentExecutor(self._performance_config)

        # Initialize script cache for reproducible research
        self._script_cache_config = self._load_script_cache_config()
        self._script_cache = ScriptCache(self._script_cache_config)

    def _load_script_cache_config(self) -> ScriptCacheConfig:
        """Load script cache configuration from performance config."""
        cache_config = self._performance_config.get("script_cache", {})

        return ScriptCacheConfig(
            enabled=cache_config.get("enabled", True),
            ttl_seconds=cache_config.get("ttl_seconds", 3600),
            max_size=cache_config.get("max_size", 1000),
            persist_to_artifacts=cache_config.get("persist_to_artifacts", False),
            artifact_base_dir=self._artifact_manager.base_dir,
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
        """Execute an ensemble and return structured results.

        Automatically detects if the ensemble requires user input and switches to
        interactive mode if needed.
        """
        # Check if this ensemble requires interactive mode
        if self._detect_interactive_ensemble(config):
            # Automatically switch to interactive execution
            user_input_handler = self._create_user_input_handler()
            return await self.execute_with_user_input(
                config, input_data, user_input_handler
            )

        # Continue with standard execution for non-interactive ensembles
        start_time = time.time()

        # Start ensemble execution with progress controller
        if self._progress_controller:
            await self._progress_controller.start_ensemble(config.name)

        # Initialize execution setup
        result, results_dict = await self._initialize_execution_setup(
            config, input_data
        )

        # Analyze dependencies and prepare phases
        phases = await self._analyze_and_prepare_phases(config)

        # Execute agents in dependency-based phases
        has_errors = False
        for phase_index, phase_agents in enumerate(phases):
            # Determine input for this phase using DependencyResolver
            if phase_index == 0:
                # First phase uses the base input
                phase_input: str | dict[str, str] = input_data
            else:
                # Subsequent phases get enhanced input with dependencies
                phase_input = self._dependency_resolver.enhance_input_with_dependencies(
                    input_data, phase_agents, results_dict
                )

            # Execute phase with full monitoring
            phase_has_errors = await self._execute_phase_with_monitoring(
                phase_index, phase_agents, phase_input, results_dict, len(phases)
            )
            has_errors = has_errors or phase_has_errors

        # Finalize results with usage, stats, and artifact saving
        final_result = await self._finalize_execution_results(
            config, result, has_errors, start_time
        )

        # Add processed agent requests to metadata if any exist
        if hasattr(self, "_ensemble_metadata") and self._ensemble_metadata.get(
            "processed_agent_requests"
        ):
            final_result["metadata"]["processed_agent_requests"] = (
                self._ensemble_metadata["processed_agent_requests"]
            )

        # Add execution order for validation
        final_result["execution_order"] = [
            agent["name"]
            for agent in config.agents
            if agent["name"] in final_result["results"]
        ]

        # Complete ensemble execution with progress controller
        if self._progress_controller:
            await self._progress_controller.complete_ensemble()

        return final_result

    def _detect_interactive_ensemble(self, config: EnsembleConfig) -> bool:
        """Detect if ensemble contains scripts that require user input.

        Uses ScriptUserInputHandler to analyze the ensemble configuration
        and determine if any agents require interactive execution.

        Args:
            config: Ensemble configuration to analyze

        Returns:
            True if ensemble requires user input, False otherwise
        """
        handler = ScriptUserInputHandler()
        return handler.ensemble_requires_user_input(config)

    def _create_user_input_handler(self) -> ScriptUserInputHandler:
        """Create a user input handler for interactive execution.

        Returns:
            Configured ScriptUserInputHandler instance
        """
        return ScriptUserInputHandler()

    async def _initialize_execution_setup(
        self, config: EnsembleConfig, input_data: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Initialize execution setup.

        Args:
            config: Ensemble configuration
            input_data: Initial input data

        Returns:
            Tuple of (result, results_dict)
        """
        # Store agent configs for role descriptions
        self._current_agent_configs = config.agents

        # Initialize result structure using ResultsProcessor
        result = self._results_processor.create_initial_result(
            config.name, input_data, len(config.agents)
        )
        results_dict: dict[str, Any] = result["results"]

        # Reset usage collector for this execution
        self._usage_collector.reset()

        return result, results_dict

    async def _analyze_and_prepare_phases(
        self, config: EnsembleConfig
    ) -> list[list[dict[str, Any]]]:
        """Analyze dependencies and prepare execution phases.

        Args:
            config: Ensemble configuration

        Returns:
            List of phases, each containing list of agent configs
        """
        # Use dependency analyzer for ALL agents (script and LLM)
        dependency_analysis = (
            self._dependency_analyzer.analyze_enhanced_dependency_graph(config.agents)
        )
        phases: list[list[dict[str, Any]]] = dependency_analysis["phases"]
        return phases

    async def _execute_phase_with_monitoring(
        self,
        phase_index: int,
        phase_agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        results_dict: dict[str, Any],
        total_phases: int = 1,
    ) -> bool:
        """Execute a phase with full monitoring and event handling.

        Args:
            phase_index: Index of the current phase
            phase_agents: List of agent configs for this phase
            input_data: Input data for this phase (base or enhanced with dependencies)
            results_dict: Results dictionary to populate
            total_phases: Total number of phases for event reporting

        Returns:
            True if any errors occurred in this phase
        """
        # Detect and expand fan-out agents (issue #73)
        fan_out_agents = self._detect_fan_out_in_phase(phase_agents, results_dict)
        expanded_agents = list(phase_agents)  # Copy to avoid modifying original
        fan_out_original_names: list[str] = []

        for agent_config, upstream_array in fan_out_agents:
            # Remove original fan-out agent from list
            expanded_agents = [
                a for a in expanded_agents if a["name"] != agent_config["name"]
            ]
            # Add expanded instances
            instances = self._expand_fan_out_agent(agent_config, upstream_array)
            expanded_agents.extend(instances)
            fan_out_original_names.append(agent_config["name"])

        # Emit phase started event
        self._emit_performance_event(
            "phase_started",
            {
                "phase_index": phase_index,
                "phase_agents": [agent["name"] for agent in expanded_agents],
                "total_phases": total_phases,
            },
        )

        # Start per-phase monitoring for performance feedback
        phase_start_time = time.time()
        await self._start_phase_monitoring(phase_index, expanded_agents)

        try:
            # Execute agents in this phase in parallel (including fan-out instances)
            phase_results = await self._execute_agents_in_phase_parallel(
                expanded_agents, input_data
            )

            # Process parallel execution results
            phase_has_errors = await self._process_phase_results(
                phase_results, results_dict, expanded_agents
            )

            # Gather fan-out instance results under original agent names
            for original_name in fan_out_original_names:
                gathered = self._gather_fan_out_results(original_name, results_dict)
                results_dict[original_name] = gathered

        finally:
            # Stop per-phase monitoring and collect metrics
            phase_duration = time.time() - phase_start_time
            await self._stop_phase_monitoring(
                phase_index, expanded_agents, phase_duration
            )

        # Emit phase completion event
        self._emit_phase_completed_event(phase_index, phase_agents, results_dict)

        return phase_has_errors

    async def execute_with_user_input(
        self,
        config: EnsembleConfig,
        input_data: str,
        user_input_handler: ScriptUserInputHandler,
    ) -> dict[str, Any]:
        """Execute an ensemble with user input handling support.

        This enhanced version integrates user input collection during execution
        and includes metadata about the interactive session.

        Args:
            config: Ensemble configuration
            input_data: Initial input data
            user_input_handler: Handler for user input collection

        Returns:
            Execution results with interactive metadata
        """
        start_time = time.time()

        # Initialize execution setup using existing helper
        result, results_dict = await self._initialize_execution_setup(
            config, input_data
        )

        # Analyze dependencies and prepare phases using existing helper
        phases = await self._analyze_and_prepare_phases(config)

        # Execute agents in dependency-based phases with user input tracking
        has_errors = False
        user_inputs_collected = 0

        for phase_index, phase_agents in enumerate(phases):
            # Determine input for this phase using DependencyResolver
            if phase_index == 0:
                # First phase uses the base input
                phase_input: str | dict[str, str] = input_data
            else:
                # Subsequent phases get enhanced input with dependencies
                phase_input = self._dependency_resolver.enhance_input_with_dependencies(
                    input_data, phase_agents, results_dict
                )

            # Execute phase with monitoring and user input counting
            (
                phase_has_errors,
                user_inputs_from_phase,
            ) = await self._execute_phase_with_monitoring_interactive(
                phase_index, phase_agents, phase_input, results_dict, len(phases)
            )

            has_errors = has_errors or phase_has_errors
            user_inputs_collected += user_inputs_from_phase

        # Finalize results using existing helper
        final_result = await self._finalize_execution_results(
            config, result, has_errors, start_time
        )

        # Add processed agent requests to metadata if any exist
        if hasattr(self, "_ensemble_metadata") and self._ensemble_metadata.get(
            "processed_agent_requests"
        ):
            final_result["metadata"]["processed_agent_requests"] = (
                self._ensemble_metadata["processed_agent_requests"]
            )

        # Add interactive-specific metadata using helper
        self._add_interactive_metadata(final_result, user_inputs_collected)

        return final_result

    def _count_user_inputs_from_phase_results(
        self, phase_results: dict[str, Any]
    ) -> int:
        """Count user inputs collected from phase results.

        Args:
            phase_results: Results from executing agents in a phase

        Returns:
            Number of user inputs collected
        """
        user_inputs_collected = 0
        for _agent_name, agent_result in phase_results.items():
            if agent_result.get("response") and isinstance(
                agent_result["response"], dict
            ):
                # Check if this was an interactive script result
                if agent_result["response"].get("collected_data"):
                    user_inputs_collected += 1
        return user_inputs_collected

    def _add_interactive_metadata(
        self, final_result: dict[str, Any], user_inputs_collected: int
    ) -> None:
        """Add interactive mode metadata to final result.

        Args:
            final_result: The final result dictionary to modify
            user_inputs_collected: Number of user inputs collected
        """
        final_result["metadata"]["interactive_mode"] = True
        final_result["metadata"]["user_inputs_collected"] = user_inputs_collected

    async def _execute_phase_with_monitoring_interactive(
        self,
        phase_index: int,
        phase_agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        results_dict: dict[str, Any],
        total_phases: int = 1,
    ) -> tuple[bool, int]:
        """Execute a phase with monitoring and user input counting.

        Args:
            phase_index: Index of the current phase
            phase_agents: List of agent configs for this phase
            input_data: Input data for this phase (base or enhanced with dependencies)
            results_dict: Results dictionary to populate
            total_phases: Total number of phases for event reporting

        Returns:
            Tuple of (has_errors, user_inputs_collected)
        """
        # Emit phase started event
        self._emit_performance_event(
            "phase_started",
            {
                "phase_index": phase_index,
                "phase_agents": [agent["name"] for agent in phase_agents],
                "total_phases": total_phases,
            },
        )

        # Start per-phase monitoring for performance feedback
        phase_start_time = time.time()
        await self._start_phase_monitoring(phase_index, phase_agents)

        try:
            # Execute agents in this phase in parallel
            # For interactive mode, we handle user input collection here
            phase_results = await self._execute_agents_in_phase_parallel(
                phase_agents, input_data
            )

            # Count user inputs from phase results
            user_inputs_from_phase = self._count_user_inputs_from_phase_results(
                phase_results
            )

            # Process parallel execution results
            phase_has_errors = await self._process_phase_results(
                phase_results, results_dict, phase_agents
            )

        finally:
            # Stop per-phase monitoring and collect metrics
            phase_duration = time.time() - phase_start_time
            await self._stop_phase_monitoring(phase_index, phase_agents, phase_duration)

        # Emit phase completion event
        self._emit_phase_completed_event(phase_index, phase_agents, results_dict)

        return phase_has_errors, user_inputs_from_phase

    async def _finalize_execution_results(
        self,
        config: EnsembleConfig,
        result: dict[str, Any],
        has_errors: bool,
        start_time: float,
    ) -> dict[str, Any]:
        """Finalize execution results with usage, stats, and artifact saving.

        Args:
            config: Ensemble configuration
            result: Initial result structure
            has_errors: Whether any errors occurred during execution
            start_time: Execution start time

        Returns:
            Finalized result dictionary
        """
        # Get collected usage and adaptive stats, then finalize result using processor
        agent_usage = self._usage_collector.get_agent_usage()
        adaptive_stats = self._agent_executor.get_adaptive_stats()
        final_result = self._results_processor.finalize_result(
            result, agent_usage, has_errors, start_time, adaptive_stats
        )

        # Run validation if config is present
        if config.validation is not None:
            validation_result = await self._run_validation(
                config, final_result, start_time
            )
            final_result["validation_result"] = validation_result

        # Save artifacts (don't fail execution if saving fails)
        try:
            self._artifact_manager.save_execution_results(
                config.name, final_result, relative_path=config.relative_path
            )
        except Exception:
            # Silently ignore artifact saving errors to not break execution
            pass

        return final_result

    async def _run_validation(
        self, config: EnsembleConfig, result: dict[str, Any], start_time: float
    ) -> Any:
        """Run validation on execution results.

        Args:
            config: Ensemble configuration
            result: Execution results
            start_time: Execution start time

        Returns:
            ValidationResult object
        """
        from datetime import datetime

        # Parse validation config (mypy needs explicit type annotation)
        validation_dict: dict[str, Any] = config.validation or {}
        validation_config = ValidationConfig(**validation_dict)

        # Convert execution results to EnsembleExecutionResult format
        execution_order = [
            agent["name"]
            for agent in config.agents
            if agent["name"] in result["results"]
        ]

        # Convert agent outputs, handling both dict and string responses
        agent_outputs = {}
        for agent_name, agent_result in result["results"].items():
            response = agent_result.get("response", {})
            # If response is a string, try to parse as JSON first
            if isinstance(response, str):
                try:
                    import json as json_module

                    agent_outputs[agent_name] = json_module.loads(response)
                except (json_module.JSONDecodeError, ValueError):
                    # Not JSON, wrap in dict
                    agent_outputs[agent_name] = {"output": response}
            else:
                agent_outputs[agent_name] = response

        execution_time = time.time() - start_time

        ensemble_result = EnsembleExecutionResult(
            ensemble_name=config.name,
            execution_order=execution_order,
            agent_outputs=agent_outputs,
            execution_time=execution_time,
            timestamp=datetime.now(),
        )

        # Run validation
        evaluator = ValidationEvaluator()
        return await evaluator.evaluate(config.name, ensemble_result, validation_config)

    async def _execute_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute a single agent and return its response and model instance.

        Agent type is determined implicitly based on configuration fields:
        - Has 'script' field -> Script agent
        - Has 'model_profile' field -> LLM agent
        - Has explicit 'type' field -> Use that (backward compatibility)
        """
        agent_type = self._determine_agent_type(agent_config)

        if agent_type == "script":
            return await self._execute_script_agent(agent_config, input_data)
        elif agent_type == "llm":
            return await self._execute_llm_agent(agent_config, input_data)
        else:
            agent_name = agent_config.get("name", "unknown")
            raise ValueError(
                f"Agent '{agent_name}' must have either 'script' or 'model_profile'"
            )

    def _determine_agent_type(self, agent_config: dict[str, Any]) -> str | None:
        """Determine agent type from configuration.

        Args:
            agent_config: Agent configuration

        Returns:
            Agent type: 'script', 'llm', or None if cannot be determined
        """
        # Check for explicit type first (backward compatibility)
        explicit_type = agent_config.get("type")
        if explicit_type == "script":
            return "script"
        elif explicit_type == "llm":
            return "llm"

        # Implicit type detection based on fields present
        if "script" in agent_config:
            return "script"
        elif "model_profile" in agent_config or "model" in agent_config:
            return "llm"

        return None

    async def _execute_script_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute script agent with caching and resource monitoring."""
        script_content = agent_config.get("script", "")
        parameters = agent_config.get("parameters", {})

        # Check cache first
        cache_key_params = {
            "input_data": input_data,
            "parameters": parameters,
        }

        cached_result = self._script_cache.get(script_content, cache_key_params)
        if cached_result is not None:
            # Cache hit - return cached result
            return cached_result.get("output", ""), None

        # Cache miss - execute script and cache result
        start_time = time.time()
        response, model_instance = await self._execute_script_agent_without_cache(
            agent_config, input_data
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # Cache the result
        cache_result = {
            "output": response,
            "execution_metadata": {"duration_ms": duration_ms},
            "success": True,
        }
        self._script_cache.set(script_content, cache_key_params, cache_result)

        return response, model_instance

    async def _execute_script_agent_without_cache(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute script agent with resource monitoring using EnhancedScriptAgent."""
        agent_name = agent_config["name"]

        # Start resource monitoring for this agent
        self._usage_collector.start_agent_resource_monitoring(agent_name)

        try:
            # Use EnhancedScriptAgent for JSON I/O support
            script_agent = EnhancedScriptAgent(
                agent_name, agent_config, project_dir=self._project_dir
            )

            # Sample resources during execution
            self._usage_collector.sample_agent_resources(agent_name)

            # Execute script with appropriate input handling
            response = await self._execute_script_with_input_handling(
                script_agent, agent_config, input_data
            )

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

    async def _execute_script_with_input_handling(
        self,
        script_agent: EnhancedScriptAgent,
        agent_config: dict[str, Any],
        input_data: str,
    ) -> str | dict[str, Any]:
        """Execute script with appropriate input format and interaction handling.

        Args:
            script_agent: Script agent to execute
            agent_config: Agent configuration
            input_data: Input data as JSON string

        Returns:
            Script execution response
        """
        import json

        try:
            parsed_input = json.loads(input_data)
            return await self._execute_with_parsed_input(
                script_agent, agent_config, input_data, parsed_input
            )
        except (json.JSONDecodeError, TypeError):
            return await self._execute_with_raw_input(
                script_agent, agent_config, input_data
            )

    async def _execute_with_parsed_input(
        self,
        script_agent: EnhancedScriptAgent,
        agent_config: dict[str, Any],
        input_data: str,
        parsed_input: dict[str, Any],
    ) -> str | dict[str, Any]:
        """Execute script with parsed JSON input.

        Args:
            script_agent: Script agent to execute
            agent_config: Agent configuration
            input_data: Original input data string
            parsed_input: Parsed input dictionary

        Returns:
            Script execution response
        """
        import json

        # Check if input is ScriptAgentInput (ADR-001)
        if self._is_script_agent_input(parsed_input):
            # ScriptAgentInput JSON - pass directly to script without wrapping
            return await script_agent.execute_with_schema_json(input_data)

        # Legacy input format - check if user input is needed
        if self._requires_user_input(agent_config):
            return await self._execute_interactive_script_agent(
                script_agent, parsed_input
            )

        # Use regular execute for non-interactive scripts (convert dict to string)
        return await script_agent.execute(json.dumps(parsed_input))

    async def _execute_with_raw_input(
        self,
        script_agent: EnhancedScriptAgent,
        agent_config: dict[str, Any],
        input_data: str,
    ) -> str | dict[str, Any]:
        """Execute script with raw string input.

        Args:
            script_agent: Script agent to execute
            agent_config: Agent configuration
            input_data: Raw input data string

        Returns:
            Script execution response
        """
        if self._requires_user_input(agent_config):
            return await self._execute_interactive_script_agent(
                script_agent, input_data
            )

        return await script_agent.execute(input_data)

    def _is_script_agent_input(self, parsed_input: dict[str, Any]) -> bool:
        """Check if parsed input is a ScriptAgentInput object.

        Args:
            parsed_input: Parsed input dictionary

        Returns:
            True if input is ScriptAgentInput format
        """
        return (
            isinstance(parsed_input, dict)
            and "agent_name" in parsed_input
            and "input_data" in parsed_input
        )

    def _requires_user_input(self, agent_config: dict[str, Any]) -> bool:
        """Check if script requires user input.

        Args:
            agent_config: Agent configuration

        Returns:
            True if script requires user input
        """
        user_input_detection = ScriptUserInputHandler()
        script_ref = agent_config.get("script", "")
        return user_input_detection.requires_user_input(script_ref)

    async def _execute_interactive_script_agent(
        self, script_agent: EnhancedScriptAgent, input_data: str | dict[str, Any]
    ) -> str:
        """Execute script agent interactively with terminal access for input().

        Args:
            script_agent: The script agent to execute
            input_data: Input data for the agent

        Returns:
            Script output as string
        """
        # First try to directly pause the progress display if available
        if hasattr(self, "_progress_controller"):
            if self._progress_controller:
                try:
                    # Extract prompt from script parameters if available
                    prompt = script_agent.parameters.get("prompt", "")
                    self._progress_controller.pause_for_user_input(
                        script_agent.name, prompt
                    )
                except Exception:
                    pass  # Fall back to event system if direct control fails

        # Also emit event for any other listeners
        self._emit_performance_event(
            "user_input_required",
            {
                "agent_name": script_agent.name,
                "script": script_agent.script,
                "message": "Waiting for user input...",
            },
        )
        import json
        import os
        import subprocess

        # Get the resolved script path
        resolved_script = script_agent._script_resolver.resolve_script_path(
            script_agent.script
        )

        if not os.path.exists(resolved_script):
            raise RuntimeError(f"Script file not found: {resolved_script}")

        # Prepare environment with input data and parameters
        env = os.environ.copy()
        env.update(script_agent.environment)

        # Pass data via environment instead of stdin so script can access terminal
        # Convert input_data to JSON string if it's a dict
        if isinstance(input_data, dict):
            env["INPUT_DATA"] = json.dumps(input_data)
        else:
            env["INPUT_DATA"] = input_data
        env["AGENT_PARAMETERS"] = json.dumps(script_agent.parameters)

        # Determine interpreter
        interpreter = script_agent._get_interpreter(resolved_script)

        # Execute with stdin inherited but stdout captured
        # We need stdout for the result, but stdin must be connected to terminal
        result = subprocess.run(
            interpreter + [resolved_script],
            env=env,
            timeout=script_agent.timeout,
            stdin=None,  # Inherit stdin from parent (terminal access)
            stdout=subprocess.PIPE,  # Capture stdout for result
            stderr=None,  # Let stderr show in terminal
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Script exited with code {result.returncode}",
                }
            )

        # First try to directly resume the progress display if available
        if hasattr(self, "_progress_controller") and self._progress_controller:
            try:
                self._progress_controller.resume_from_user_input(script_agent.name)
            except Exception:
                # Fall back to event system if direct control fails  # nosec B110
                pass

        # Also emit event for any other listeners
        self._emit_performance_event(
            "user_input_completed",
            {
                "agent_name": script_agent.name,
                "message": "User input completed, continuing...",
            },
        )

        # Return the actual script output
        if result.stdout:
            return result.stdout.strip()
        else:
            return json.dumps(
                {
                    "success": True,
                    "message": "Interactive script completed (no output)",
                }
            )

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

    async def _execute_agents_in_phase_parallel(
        self, phase_agents: list[dict[str, Any]], phase_input: str | dict[str, str]
    ) -> dict[str, Any]:
        """Execute agents in parallel within a phase.

        Based on Issue #43 analysis, this provides 3-15x performance improvement
        for I/O bound LLM API calls using asyncio.gather().
        """
        # Execute all agents in parallel using asyncio.gather
        tasks = [
            self._execute_single_agent_in_phase(agent_config, phase_input)
            for agent_config in phase_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        phase_results: dict[str, Any] = {}
        for result in results:
            if isinstance(result, BaseException):
                # Handle unexpected errors during gather
                continue
            # At this point, result should be a tuple[str, dict[str, Any]]
            agent_name, agent_result = result
            phase_results[agent_name] = agent_result

        return phase_results

    async def _execute_single_agent_in_phase(
        self, agent_config: dict[str, Any], phase_input: str | dict[str, str]
    ) -> tuple[str, dict[str, Any]]:
        """Execute a single agent in a phase and return (name, result)."""
        agent_name = agent_config["name"]

        # Handle fan-out instance input preparation (issue #73)
        if self._dependency_resolver.is_fan_out_instance_config(agent_config):
            base_input = (
                phase_input
                if isinstance(phase_input, str)
                else phase_input.get(agent_name, "")
            )
            phase_input = self._dependency_resolver.prepare_fan_out_instance_input(
                agent_config, base_input
            )

        try:
            return await self._execute_agent_with_monitoring(
                agent_config, agent_name, phase_input
            )
        except Exception as e:
            return await self._handle_agent_execution_failure(agent_config, e)

    async def _execute_agent_with_monitoring(
        self,
        agent_config: dict[str, Any],
        agent_name: str,
        phase_input: str | dict[str, str],
    ) -> tuple[str, dict[str, Any]]:
        """Execute agent with full monitoring and progress tracking."""
        agent_start_time = time.time()

        # Emit agent started event
        self._emit_performance_event(
            "agent_started",
            {"agent_name": agent_name, "timestamp": agent_start_time},
        )

        # Update progress controller with agent progress
        if self._progress_controller:
            await self._progress_controller.update_agent_progress(agent_name, "started")

        # Get agent input and timeout
        agent_input = self._dependency_resolver.get_agent_input(phase_input, agent_name)
        timeout = await self._get_agent_timeout(agent_config)

        # Execute agent with timeout coordination
        (
            response,
            model_instance,
        ) = await self._execution_coordinator.execute_agent_with_timeout(
            agent_config, agent_input, timeout
        )

        # Emit completion events
        await self._emit_agent_completion_events(agent_name, agent_start_time)

        return agent_name, {
            "response": response,
            "status": "success",
            "model_instance": model_instance,
        }

    async def _get_agent_timeout(self, agent_config: dict[str, Any]) -> int:
        """Get timeout for agent execution."""
        enhanced_config = await self._resolve_model_profile_to_config(agent_config)
        timeout = enhanced_config.get("timeout_seconds")
        if timeout is not None:
            return int(timeout)
        return int(
            self._performance_config.get("execution", {}).get("default_timeout", 60)
        )

    async def _emit_agent_completion_events(
        self, agent_name: str, start_time: float
    ) -> None:
        """Emit agent completion events and update progress."""
        agent_end_time = time.time()
        duration_ms = int((agent_end_time - start_time) * 1000)

        self._emit_performance_event(
            "agent_completed",
            {
                "agent_name": agent_name,
                "timestamp": agent_end_time,
                "duration_ms": duration_ms,
            },
        )

        # Update progress controller with agent completion
        if self._progress_controller:
            await self._progress_controller.update_agent_progress(
                agent_name, "completed"
            )

    async def _handle_agent_execution_failure(
        self, agent_config: dict[str, Any], error: Exception
    ) -> tuple[str, dict[str, Any]]:
        """Handle agent execution failure and return error result."""
        agent_name = agent_config["name"]
        agent_end_time = time.time()

        self._emit_performance_event(
            "agent_completed",
            {
                "agent_name": agent_name,
                "timestamp": agent_end_time,
                "duration_ms": 0,
                "error": str(error),
            },
        )

        return agent_name, {
            "error": str(error),
            "status": "failed",
            "model_instance": None,
        }

    async def _process_phase_results(
        self,
        phase_results: dict[str, Any],
        results_dict: dict[str, Any],
        phase_agents: list[dict[str, Any]],
    ) -> bool:
        """Process parallel execution results and return if any errors occurred."""
        has_errors = False
        processed_agent_requests: list[dict[str, Any]] = []

        # Create agent lookup for model profile information
        agent_configs = {agent["name"]: agent for agent in phase_agents}

        for agent_name, agent_result in phase_results.items():
            # Store result in results_dict
            self._store_agent_result(results_dict, agent_name, agent_result)

            # Handle errors
            if agent_result["status"] == "failed":
                has_errors = True

            # Process successful agents
            if agent_result["status"] == "success":
                await self._process_successful_agent_result(
                    agent_result,
                    agent_name,
                    results_dict,
                    processed_agent_requests,
                    phase_agents,
                    agent_configs,
                )

        # Store processed agent requests in results metadata for coordination
        self._store_agent_requests_metadata(processed_agent_requests)

        return has_errors

    def _store_agent_result(
        self,
        results_dict: dict[str, Any],
        agent_name: str,
        agent_result: dict[str, Any],
    ) -> None:
        """Store agent result in results dictionary."""
        results_dict[agent_name] = {
            "response": agent_result.get("response"),
            "status": agent_result["status"],
        }
        if agent_result["status"] == "failed":
            results_dict[agent_name]["error"] = agent_result["error"]

    async def _process_successful_agent_result(
        self,
        agent_result: dict[str, Any],
        agent_name: str,
        results_dict: dict[str, Any],
        processed_agent_requests: list[dict[str, Any]],
        phase_agents: list[dict[str, Any]],
        agent_configs: dict[str, dict[str, Any]],
    ) -> None:
        """Process successful agent result for requests and usage."""
        # Process AgentRequest objects from successful script agents
        response = agent_result.get("response")
        if response and isinstance(response, str):
            await self._process_agent_requests(
                response,
                agent_name,
                results_dict,
                processed_agent_requests,
                phase_agents,
            )

        # Collect usage for successful agents with model instances
        if agent_result["model_instance"] is not None:
            agent_config = agent_configs.get(agent_name, {})
            model_profile = agent_config.get("model_profile", "unknown")
            self._usage_collector.collect_agent_usage(
                agent_name, agent_result["model_instance"], model_profile
            )

    async def _process_agent_requests(
        self,
        response: str,
        agent_name: str,
        results_dict: dict[str, Any],
        processed_agent_requests: list[dict[str, Any]],
        phase_agents: list[dict[str, Any]],
    ) -> None:
        """Process agent requests from script output."""
        try:
            processed_result = (
                await self._agent_request_processor.process_script_output_with_requests(
                    response, agent_name, phase_agents
                )
            )

            # Store processed agent requests for coordination
            if processed_result.get("agent_requests"):
                processed_agent_requests.extend(processed_result["agent_requests"])

            # Add metadata about processed requests
            results_dict[agent_name]["agent_requests"] = processed_result.get(
                "agent_requests", []
            )

        except Exception:
            # Silently ignore AgentRequest processing errors
            pass

    def _store_agent_requests_metadata(
        self, processed_agent_requests: list[dict[str, Any]]
    ) -> None:
        """Store processed agent requests in ensemble metadata."""
        if processed_agent_requests:
            if not hasattr(self, "_ensemble_metadata"):
                self._ensemble_metadata = {}
            self._ensemble_metadata["processed_agent_requests"] = (
                processed_agent_requests
            )

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

    # ========== Fan-Out Support (Issue #73) ==========

    def _detect_fan_out_in_phase(
        self,
        phase_agents: list[dict[str, Any]],
        results_dict: dict[str, Any],
    ) -> list[tuple[dict[str, Any], list[Any]]]:
        """Detect fan-out agents in phase with array upstream results.

        Args:
            phase_agents: List of agent configurations in the current phase
            results_dict: Dictionary of previous agent results

        Returns:
            List of (agent_config, upstream_array) tuples for fan-out agents
        """
        fan_out_agents: list[tuple[dict[str, Any], list[Any]]] = []

        for agent_config in phase_agents:
            if not agent_config.get("fan_out"):
                continue

            # Get upstream dependency
            depends_on = agent_config.get("depends_on", [])
            if not depends_on:
                continue

            # Check first dependency for array result
            upstream_name = depends_on[0]
            upstream_result = results_dict.get(upstream_name, {})

            if upstream_result.get("status") != "success":
                continue

            response = upstream_result.get("response", "")
            array_result = self._fan_out_expander.parse_array_from_result(response)

            if array_result is not None and len(array_result) > 0:
                fan_out_agents.append((agent_config, array_result))

        return fan_out_agents

    def _expand_fan_out_agent(
        self,
        agent_config: dict[str, Any],
        upstream_array: list[Any],
    ) -> list[dict[str, Any]]:
        """Expand a fan-out agent into N instances.

        Args:
            agent_config: Original agent configuration with fan_out: true
            upstream_array: Array from upstream agent

        Returns:
            List of instance configurations with indexed names
        """
        return self._fan_out_expander.expand_fan_out_agent(agent_config, upstream_array)

    def _gather_fan_out_results(
        self,
        original_agent_name: str,
        instance_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Gather results from fan-out instances into ordered array.

        Args:
            original_agent_name: Original agent name (e.g., 'extractor')
            instance_results: Dict of instance results keyed by instance name

        Returns:
            Gathered result with response array and status
        """
        # Clear any previous results for this agent
        self._fan_out_gatherer.clear(original_agent_name)

        # Record each instance result
        for instance_name, result in instance_results.items():
            if not self._fan_out_expander.is_fan_out_instance_name(instance_name):
                continue

            original = self._fan_out_expander.get_original_agent_name(instance_name)
            if original != original_agent_name:
                continue

            success = result.get("status") == "success"
            self._fan_out_gatherer.record_instance_result(
                instance_name=instance_name,
                result=result.get("response"),
                success=success,
                error=result.get("error"),
            )

        return self._fan_out_gatherer.gather_results(original_agent_name)
