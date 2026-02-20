"""Ensemble execution with agent coordination."""

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.agent_dispatcher import AgentDispatcher
from llm_orc.core.execution.agent_execution_coordinator import (
    AgentExecutionCoordinator,
)
from llm_orc.core.execution.agent_executor import AgentExecutor
from llm_orc.core.execution.agent_request_processor import AgentRequestProcessor
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.dependency_analyzer import DependencyAnalyzer
from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.core.execution.fan_out_coordinator import FanOutCoordinator
from llm_orc.core.execution.fan_out_expander import FanOutExpander
from llm_orc.core.execution.fan_out_gatherer import FanOutGatherer
from llm_orc.core.execution.llm_agent_runner import LlmAgentRunner
from llm_orc.core.execution.phase_monitor import PhaseMonitor
from llm_orc.core.execution.phase_result_processor import PhaseResultProcessor
from llm_orc.core.execution.progress_controller import NoOpProgressController
from llm_orc.core.execution.results_processor import (
    create_initial_result,
    finalize_result,
)
from llm_orc.core.execution.script_agent_runner import ScriptAgentRunner
from llm_orc.core.execution.script_cache import ScriptCache, ScriptCacheConfig
from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler
from llm_orc.core.execution.streaming_progress_tracker import (
    StreamingProgressTracker,
)
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.core.validation import (
    EnsembleExecutionResult,
    ValidationConfig,
    ValidationEvaluator,
)
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import (
    AgentConfig,
    LlmAgentConfig,
    ScriptAgentConfig,
)


def classify_failure_type(error_message: str) -> str:
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


async def _run_validation(
    config: EnsembleConfig, result: dict[str, Any], start_time: float
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
        agent.name for agent in config.agents if agent.name in result["results"]
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

        self._streaming_progress_tracker = StreamingProgressTracker()
        self._artifact_manager = ArtifactManager()
        self._progress_controller = NoOpProgressController()
        self._agent_request_processor = AgentRequestProcessor(self._dependency_resolver)

        # Fan-out support (issue #73)
        self._fan_out_expander = FanOutExpander()
        self._fan_out_gatherer = FanOutGatherer(self._fan_out_expander)
        self._fan_out_coordinator = FanOutCoordinator(
            self._fan_out_expander, self._fan_out_gatherer
        )

        self._agent_executor = AgentExecutor(self._performance_config)

        # Initialize script cache for reproducible research
        self._script_cache_config = self._load_script_cache_config()
        self._script_cache = ScriptCache(self._script_cache_config)

        self._script_agent_runner = ScriptAgentRunner(
            self._script_cache,
            self._usage_collector,
            self._progress_controller,
            self._emit_performance_event,
            self._project_dir,
        )

        self._llm_agent_runner = LlmAgentRunner(
            self._model_factory,
            self._config_manager,
            self._usage_collector,
            self._emit_performance_event,
            classify_failure_type,
        )

        # Initialize execution coordinator with agent executor function
        # Use a wrapper that goes through _execute_agent for test patchability
        async def agent_executor_wrapper(
            agent_config: AgentConfig, input_data: str
        ) -> tuple[str, ModelInterface | None]:
            return await self._execute_agent(agent_config, input_data)

        self._execution_coordinator = AgentExecutionCoordinator(
            self._performance_config, agent_executor_wrapper
        )

        self._agent_dispatcher = AgentDispatcher(
            self._execution_coordinator,
            self._dependency_resolver,
            self._progress_controller,
            self._emit_performance_event,
            lambda cfg: self._llm_agent_runner._resolve_model_profile_to_config(cfg),
            self._performance_config,
        )

        self._phase_monitor = PhaseMonitor(
            self._agent_executor, self._emit_performance_event
        )
        self._phase_result_processor = PhaseResultProcessor(
            self._agent_request_processor,
            self._usage_collector,
            self._emit_performance_event,
        )
        self._ensemble_metadata: dict[str, Any] = {}

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
            user_input_handler = self._create_user_input_handler()
            return await self.execute_with_user_input(
                config, input_data, user_input_handler
            )

        # Start ensemble execution with progress controller
        if self._progress_controller:
            await self._progress_controller.start_ensemble(config.name)

        final_result, _ = await self._execute_core(config, input_data)

        # Add execution order for validation
        final_result["execution_order"] = [
            agent.name
            for agent in config.agents
            if agent.name in final_result["results"]
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

        # Initialize result structure
        result = create_initial_result(config.name, input_data, len(config.agents))
        results_dict: dict[str, Any] = result["results"]

        # Reset usage collector for this execution
        self._usage_collector.reset()

        return result, results_dict

    async def _analyze_and_prepare_phases(
        self, config: EnsembleConfig
    ) -> list[list[AgentConfig]]:
        """Analyze dependencies and prepare execution phases.

        Args:
            config: Ensemble configuration

        Returns:
            List of phases, each containing list of agent configs
        """
        dependency_analysis = (
            self._dependency_analyzer.analyze_enhanced_dependency_graph(config.agents)
        )
        phases: list[list[AgentConfig]] = dependency_analysis["phases"]
        return phases

    async def _execute_phase_with_monitoring(
        self,
        phase_index: int,
        phase_agents: list[AgentConfig],
        input_data: str | dict[str, str],
        results_dict: dict[str, Any],
        total_phases: int = 1,
    ) -> tuple[bool, int]:
        """Execute a phase with full monitoring, fan-out, and user input counting.

        Returns:
            Tuple of (has_errors, user_inputs_collected)
        """
        fan_out_agents = self._fan_out_coordinator.detect_in_phase(
            phase_agents, results_dict
        )
        expanded_agents = list(phase_agents)
        fan_out_original_names: list[str] = []

        for agent_config, upstream_array in fan_out_agents:
            expanded_agents = [
                a for a in expanded_agents if a.name != agent_config.name
            ]
            instances = self._fan_out_coordinator.expand_agent(
                agent_config, upstream_array
            )
            expanded_agents.extend(instances)
            fan_out_original_names.append(agent_config.name)

        self._emit_performance_event(
            "phase_started",
            {
                "phase_index": phase_index,
                "phase_agents": [agent.name for agent in expanded_agents],
                "total_phases": total_phases,
            },
        )

        # Start per-phase monitoring for performance feedback
        phase_start_time = time.time()
        await self._phase_monitor.start(phase_index, expanded_agents)

        try:
            # Execute agents in this phase in parallel
            phase_results = await self._agent_dispatcher.execute_agents_in_phase(
                expanded_agents, input_data
            )

            # Count user inputs from phase results
            user_inputs_from_phase = self._count_user_inputs_from_phase_results(
                phase_results
            )

            # Process parallel execution results
            phase_has_errors = await self._phase_result_processor.process_phase_results(
                phase_results, results_dict, expanded_agents
            )

            # Gather fan-out instance results under original agent names
            for original_name in fan_out_original_names:
                gathered = self._fan_out_coordinator.gather_results(
                    original_name, results_dict
                )
                results_dict[original_name] = gathered

        finally:
            # Stop per-phase monitoring and collect metrics
            phase_duration = time.time() - phase_start_time
            await self._phase_monitor.stop(phase_index, expanded_agents, phase_duration)

        # Emit phase completion event
        self._phase_result_processor.emit_phase_completed_event(
            phase_index, phase_agents, results_dict
        )

        return phase_has_errors, user_inputs_from_phase

    async def execute_with_user_input(
        self,
        config: EnsembleConfig,
        input_data: str,
        user_input_handler: ScriptUserInputHandler,
    ) -> dict[str, Any]:
        """Execute an ensemble with user input handling support."""
        final_result, user_inputs_collected = await self._execute_core(
            config, input_data, track_user_inputs=True
        )

        # Add interactive-specific metadata
        self._add_interactive_metadata(final_result, user_inputs_collected)

        return final_result

    async def _execute_core(
        self,
        config: EnsembleConfig,
        input_data: str,
        *,
        track_user_inputs: bool = False,
    ) -> tuple[dict[str, Any], int]:
        """Core execution logic shared by execute() and execute_with_user_input().

        Returns:
            Tuple of (final_result, user_inputs_collected)
        """
        start_time = time.time()

        result, results_dict = await self._initialize_execution_setup(
            config, input_data
        )

        phases = await self._analyze_and_prepare_phases(config)

        has_errors = False
        user_inputs_collected = 0

        for phase_index, phase_agents in enumerate(phases):
            if phase_index == 0:
                phase_input: str | dict[str, str] = input_data
            else:
                phase_input = self._dependency_resolver.enhance_input_with_dependencies(
                    input_data, phase_agents, results_dict
                )

            (
                phase_has_errors,
                user_inputs_from_phase,
            ) = await self._execute_phase_with_monitoring(
                phase_index,
                phase_agents,
                phase_input,
                results_dict,
                len(phases),
            )
            has_errors = has_errors or phase_has_errors
            if track_user_inputs:
                user_inputs_collected += user_inputs_from_phase

        final_result = await self._finalize_execution_results(
            config, result, has_errors, start_time
        )

        # Add processed agent requests to metadata if any exist
        ensemble_metadata = self._phase_result_processor.get_ensemble_metadata()
        if ensemble_metadata.get("processed_agent_requests"):
            final_result["metadata"]["processed_agent_requests"] = ensemble_metadata[
                "processed_agent_requests"
            ]

        return final_result, user_inputs_collected

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
        final_result = finalize_result(
            result, agent_usage, has_errors, start_time, adaptive_stats
        )

        # Run validation if config is present
        if config.validation is not None:
            validation_result = await _run_validation(config, final_result, start_time)
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

    async def _execute_agent(
        self, agent_config: AgentConfig, input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute a single agent, routing by type."""
        if isinstance(agent_config, ScriptAgentConfig):
            return await self._script_agent_runner.execute(agent_config, input_data)
        if isinstance(agent_config, LlmAgentConfig):
            return await self._llm_agent_runner.execute(agent_config, input_data)
        raise ValueError(
            f"Agent '{agent_config.name}' has unknown type: "
            f"{type(agent_config).__name__}"
        )

    @property
    def execution_coordinator(self) -> AgentExecutionCoordinator:
        """Expose execution coordinator for composition-based callers."""
        return self._execution_coordinator

    def _get_agent_role_description(self, agent_name: str) -> str | None:
        """Get a human-readable role description for an agent."""
        if hasattr(self, "_current_agent_configs"):
            for agent_config in self._current_agent_configs:
                if agent_config.name == agent_name:
                    if (
                        isinstance(agent_config, LlmAgentConfig)
                        and agent_config.model_profile
                    ):
                        return agent_config.model_profile.replace("-", " ").title()
                    return agent_name.replace("-", " ").title()

        return agent_name.replace("-", " ").title()
