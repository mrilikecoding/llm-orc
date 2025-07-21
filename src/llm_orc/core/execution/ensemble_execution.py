"""Ensemble execution with agent coordination."""

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

from llm_orc.agents.script_agent import ScriptAgent
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.core.execution.agent_executor import AgentExecutor
from llm_orc.core.execution.dependency_analyzer import DependencyAnalyzer
from llm_orc.core.execution.input_enhancer import InputEnhancer
from llm_orc.core.execution.orchestration import Agent
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

        # Performance monitoring hooks for Issue #27 visualization integration
        self._performance_hooks: list[Callable[[str, dict[str, Any]], None]] = []

        # Initialize extracted components
        self._model_factory = ModelFactory(
            self._config_manager, self._credential_storage
        )
        self._dependency_analyzer = DependencyAnalyzer()
        self._input_enhancer = InputEnhancer()
        self._agent_executor = AgentExecutor(
            self._performance_config,
            self._emit_performance_event,
            self._resolve_model_profile_to_config,
            self._execute_agent_with_timeout,
            self._input_enhancer.get_agent_input,
        )

    # Delegator methods for backward compatibility (especially for tests)
    async def _load_model(
        self, model_name: str, provider: str | None = None
    ) -> ModelInterface:
        """Delegate to model factory."""
        return await self._model_factory.load_model(model_name, provider)

    async def _load_model_from_agent_config(
        self, agent_config: dict[str, Any]
    ) -> ModelInterface:
        """Delegate to model factory."""
        return await self._model_factory.load_model_from_agent_config(agent_config)

    def register_performance_hook(
        self, hook: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Register a performance monitoring hook for Issue #27 visualization."""
        self._performance_hooks.append(hook)

    def _emit_performance_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit performance monitoring events to registered hooks."""
        for hook in self._performance_hooks:
            try:
                hook(event_type, data)
            except Exception:
                # Silently ignore hook failures to avoid breaking execution
                pass

    async def execute_streaming(
        self, config: EnsembleConfig, input_data: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute ensemble with streaming progress updates.

        Yields progress events during execution for real-time monitoring.
        Events include: execution_started, agent_progress, execution_completed.
        """
        start_time = time.time()

        # Emit execution started event
        yield {
            "type": "execution_started",
            "data": {
                "ensemble": config.name,
                "timestamp": start_time,
                "total_agents": len(config.agents),
            },
        }

        # Set up progress tracking
        progress_events: list[dict[str, Any]] = []

        # Register a progress hook to capture streaming events
        def progress_hook(event_type: str, data: dict[str, Any]) -> None:
            progress_events.append({"type": event_type, "data": data})

        # Register our progress hook
        self.register_performance_hook(progress_hook)

        # Execute the ensemble using the existing execute method
        try:
            # Create task for execution
            execution_task = asyncio.create_task(self.execute(config, input_data))

            # Monitor progress while execution runs
            last_progress_count = 0
            while not execution_task.done():
                # Check for new progress events
                completed_count = len(
                    [e for e in progress_events if e["type"] == "agent_completed"]
                )

                # Emit progress update if we have new completions
                if completed_count > last_progress_count:
                    yield {
                        "type": "agent_progress",
                        "data": {
                            "completed_agents": completed_count,
                            "total_agents": len(config.agents),
                            "progress_percentage": (
                                completed_count / len(config.agents)
                            )
                            * 100,
                            "timestamp": time.time(),
                        },
                    }
                    last_progress_count = completed_count

                # Small delay to avoid busy waiting
                await asyncio.sleep(0.05)

            # Get final results
            final_result = await execution_task

            # Emit execution completed event with full results
            yield {
                "type": "execution_completed",
                "data": {
                    "ensemble": config.name,
                    "timestamp": time.time(),
                    "duration": time.time() - start_time,
                    "results": final_result["results"],
                    "metadata": final_result["metadata"],
                    "status": final_result["status"],
                },
            }

        finally:
            # Clean up the progress hook
            if progress_hook in self._performance_hooks:
                self._performance_hooks.remove(progress_hook)

    async def execute(self, config: EnsembleConfig, input_data: str) -> dict[str, Any]:
        """Execute an ensemble and return structured results."""
        start_time = time.time()

        # Store agent configs for role descriptions
        self._current_agent_configs = config.agents

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
        context_data: dict[str, Any] = {}

        # Phase 1: Execute script agents to gather context
        context_data, script_errors = await self._execute_script_agents(
            config, input_data, results_dict
        )
        has_errors = has_errors or script_errors

        # Phase 2: Execute LLM agents with dependency-aware phasing
        llm_agent_errors = await self._execute_llm_agents(
            config, input_data, context_data, results_dict, agent_usage
        )
        has_errors = has_errors or llm_agent_errors

        # Finalize and return result
        return self._finalize_result(result, agent_usage, has_errors, start_time)

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
            model = await self._model_factory.load_model_from_agent_config(agent_config)

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

    async def _execute_script_agents(
        self,
        config: EnsembleConfig,
        input_data: str,
        results_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Execute script agents and return context data and error status."""
        context_data = {}
        has_errors = False
        script_agents = [a for a in config.agents if a.get("type") == "script"]

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
            except Exception as e:
                results_dict[agent_config["name"]] = {
                    "error": str(e),
                    "status": "failed",
                }
                has_errors = True

        return context_data, has_errors

    async def _execute_llm_agents(
        self,
        config: EnsembleConfig,
        input_data: str,
        context_data: dict[str, Any],
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> bool:
        """Execute LLM agents with dependency-aware phasing."""
        has_errors = False
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

        # Use enhanced dependency analysis for multi-level execution
        if llm_agents:
            # Update input enhancer with current agent configs for role descriptions
            self._input_enhancer.update_agent_configs(llm_agents)
            dependency_analysis = (
                self._dependency_analyzer.analyze_enhanced_dependency_graph(llm_agents)
            )
            phases = dependency_analysis["phases"]

            # Execute each phase sequentially, with parallelization within each phase
            for phase_index, phase_agents in enumerate(phases):
                self._emit_performance_event(
                    "phase_started",
                    {
                        "phase_index": phase_index,
                        "phase_agents": [agent["name"] for agent in phase_agents],
                        "total_phases": len(phases),
                    },
                )

                # Determine input for this phase
                if phase_index == 0:
                    # First phase uses the base enhanced input
                    phase_input: str | dict[str, str] = enhanced_input
                else:
                    # Subsequent phases get enhanced input with dependencies
                    phase_input = self._input_enhancer.enhance_input_with_dependencies(
                        enhanced_input, phase_agents, results_dict
                    )

                # Get performance config
                performance_config = ConfigurationManager().load_performance_config()
                effective_concurrency = self._get_effective_concurrency_limit(
                    len(phase_agents)
                )

                # Execute agents in this phase concurrently
                semaphore = asyncio.Semaphore(effective_concurrency)

                async def execute_agent_with_semaphore(
                    agent_cfg: dict[str, Any],
                    sem: asyncio.Semaphore,
                    p_input: str | dict[str, str],
                    enhanced_inp: str,
                    perf_config: dict[str, Any],
                ) -> None:
                    async with sem:
                        try:
                            agent_name = agent_cfg["name"]
                            self._emit_performance_event(
                                "agent_started", {"agent_name": agent_name}
                            )

                            # Determine input for this specific agent
                            if isinstance(p_input, dict):
                                agent_input = p_input.get(agent_name, enhanced_inp)
                            else:
                                agent_input = p_input

                            # Resolve model profile to get enhanced configuration
                            enhanced_config = (
                                await self._resolve_model_profile_to_config(agent_cfg)
                            )
                            timeout = enhanced_config.get("timeout_seconds") or (
                                perf_config.get("execution", {}).get(
                                    "default_timeout", 60
                                )
                            )

                            (
                                agent_result,
                                model_instance,
                            ) = await self._execute_agent_with_timeout(
                                agent_cfg, agent_input, timeout
                            )

                            # Store successful result
                            results_dict[agent_name] = {
                                "response": agent_result,
                                "status": "success",
                            }

                            # Capture model usage if available
                            if model_instance and hasattr(
                                model_instance, "get_last_usage"
                            ):
                                usage = model_instance.get_last_usage()
                                if usage:
                                    agent_usage[agent_name] = usage

                            self._emit_performance_event(
                                "agent_completed", {"agent_name": agent_name}
                            )

                        except Exception as e:
                            # Handle agent failure
                            agent_name = agent_cfg["name"]
                            results_dict[agent_name] = {
                                "error": str(e),
                                "status": "failed",
                            }
                            nonlocal has_errors
                            has_errors = True
                            self._emit_performance_event(
                                "agent_failed",
                                {"agent_name": agent_name, "error": str(e)},
                            )

                # Execute all agents in this phase
                tasks = [
                    execute_agent_with_semaphore(
                        agent,
                        semaphore,
                        phase_input,
                        enhanced_input,
                        performance_config,
                    )
                    for agent in phase_agents
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

                self._emit_performance_event(
                    "phase_completed",
                    {
                        "phase_index": phase_index,
                        "successful_agents": len(
                            [
                                a
                                for a in phase_agents
                                if results_dict.get(a["name"], {}).get("status")
                                == "success"
                            ]
                        ),
                        "failed_agents": len(
                            [
                                a
                                for a in phase_agents
                                if results_dict.get(a["name"], {}).get("status")
                                == "failed"
                            ]
                        ),
                    },
                )

        return has_errors

    def _finalize_result(
        self,
        result: dict[str, Any],
        agent_usage: dict[str, Any],
        has_errors: bool,
        start_time: float,
    ) -> dict[str, Any]:
        """Finalize execution result with metadata and usage summary."""
        # Calculate usage totals (no coordinator synthesis in dependency-based model)
        usage_summary = self._calculate_usage_summary(agent_usage, None)

        # Finalize result
        end_time = time.time()
        result["status"] = "completed_with_errors" if has_errors else "completed"
        metadata_dict: dict[str, Any] = result["metadata"]
        metadata_dict["duration"] = f"{(end_time - start_time):.2f}s"
        metadata_dict["completed_at"] = end_time
        metadata_dict["usage"] = usage_summary
        return result

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

    async def _execute_agents_parallel(
        self,
        agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        config: EnsembleConfig,
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> None:
        """Execute a list of agents in parallel with resource management.

        Args:
            input_data: Either a string for uniform input, or a dict mapping
                       agent names to their specific enhanced input.
        """
        if not agents:
            return

        # Get concurrency limit from performance config or use sensible default
        max_concurrent = self._get_effective_concurrency_limit(len(agents))

        # For small ensembles, run all in parallel
        # For large ensembles, use semaphore to limit concurrent execution
        if len(agents) <= max_concurrent:
            await self._execute_agents_unlimited(
                agents, input_data, config, results_dict, agent_usage
            )
        else:
            await self._execute_agents_with_semaphore(
                agents, input_data, config, results_dict, agent_usage, max_concurrent
            )

    def _get_effective_concurrency_limit(self, agent_count: int) -> int:
        """Get effective concurrency limit based on configuration and agent count."""
        # Check performance configuration first
        configured_limit = self._performance_config.get("concurrency", {}).get(
            "max_concurrent_agents", 0
        )

        # If explicitly configured and > 0, use it
        if isinstance(configured_limit, int) and configured_limit > 0:
            return configured_limit

        # Otherwise use smart defaults based on agent count and system resources
        if agent_count <= 3:
            return agent_count  # Small ensembles: run all in parallel
        elif agent_count <= 10:
            return 5  # Medium ensembles: limit to 5 concurrent
        elif agent_count <= 20:
            return 8  # Large ensembles: limit to 8 concurrent
        else:
            return 10  # Very large ensembles: cap at 10 concurrent

    async def _execute_agents_unlimited(
        self,
        agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        config: EnsembleConfig,
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> None:
        """Execute agents without concurrency limits (for small ensembles)."""
        try:

            async def execute_agent_task(
                agent_config: dict[str, Any],
            ) -> tuple[str, Any]:
                """Execute a single agent task."""
                agent_name = agent_config["name"]
                agent_start_time = time.time()

                # Emit agent started event
                self._emit_performance_event(
                    "agent_started",
                    {"agent_name": agent_name, "timestamp": agent_start_time},
                )

                try:
                    # Resolve config and execute - all happening in parallel per agent
                    enhanced_config = await self._resolve_model_profile_to_config(
                        agent_config
                    )
                    timeout = enhanced_config.get("timeout_seconds") or (
                        self._performance_config.get("execution", {}).get(
                            "default_timeout", 60
                        )
                    )
                    # Get the appropriate input for this agent
                    agent_input = self._input_enhancer.get_agent_input(
                        input_data, agent_config["name"]
                    )
                    result = await self._execute_agent_with_timeout(
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

                    return agent_name, result
                except Exception as e:
                    # Emit agent completed event with error
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

            self._process_agent_results(agent_results, results_dict, agent_usage)

        except Exception as e:
            # Fallback: if gather fails, mark all agents as failed
            for agent_config in agents:
                agent_name = agent_config["name"]
                results_dict[agent_name] = {"error": str(e), "status": "failed"}

    async def _execute_agents_with_semaphore(
        self,
        agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        config: EnsembleConfig,
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
        max_concurrent: int,
    ) -> None:
        """Execute agents with semaphore-based concurrency control."""
        # Create semaphore to limit concurrent execution
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_agent_with_semaphore(
            agent_config: dict[str, Any],
        ) -> tuple[str, Any]:
            """Execute agent with semaphore control."""
            async with semaphore:
                agent_name = agent_config["name"]
                agent_start_time = time.time()

                # Emit agent started event
                self._emit_performance_event(
                    "agent_started",
                    {"agent_name": agent_name, "timestamp": agent_start_time},
                )

                try:
                    # Resolve config and execute
                    enhanced_config = await self._resolve_model_profile_to_config(
                        agent_config
                    )
                    timeout = enhanced_config.get("timeout_seconds") or (
                        self._performance_config.get("execution", {}).get(
                            "default_timeout", 60
                        )
                    )
                    # Get the appropriate input for this agent
                    agent_input = self._input_enhancer.get_agent_input(
                        input_data, agent_config["name"]
                    )
                    result = await self._execute_agent_with_timeout(
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

                    return agent_name, result
                except Exception as e:
                    # Emit agent completed event with error
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

                    # Record error in results dict and return error indicator
                    results_dict[agent_name] = {
                        "error": str(e),
                        "status": "failed",
                    }
                    return agent_name, None

        try:
            # Create tasks with semaphore control
            tasks = [
                asyncio.create_task(execute_agent_with_semaphore(agent_config))
                for agent_config in agents
            ]

            # Wait for all tasks to complete
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            self._process_agent_results(agent_results, results_dict, agent_usage)

        except Exception as e:
            # Fallback: if gather fails, mark all agents as failed
            for agent_config in agents:
                agent_name = agent_config["name"]
                results_dict[agent_name] = {"error": str(e), "status": "failed"}

    def _process_agent_results(
        self,
        agent_results: list[Any],
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> None:
        """Process results from agent execution."""
        for execution_result in agent_results:
            if isinstance(execution_result, Exception):
                # If we can't determine agent name from exception, skip this result
                # The agent task should handle its own error recording
                continue
            else:
                # execution_result is tuple[str, Any]
                agent_name, result = execution_result
                if result is None:
                    # Error was already recorded in execute_agent_task
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

    def _enhance_input_with_dependencies(
        self,
        base_input: str,
        dependent_agents: list[dict[str, Any]],
        results_dict: dict[str, Any],
    ) -> dict[str, str]:
        """Enhance input with dependency results for each dependent agent.

        Returns a dictionary mapping agent names to their enhanced input.
        Each agent gets only the results from their specific dependencies.
        """
        enhanced_inputs = {}

        for agent_config in dependent_agents:
            agent_name = agent_config["name"]
            dependencies = agent_config.get("depends_on", [])

            if not dependencies:
                enhanced_inputs[agent_name] = base_input
                continue

            # Build structured dependency results for this specific agent
            dependency_results = []
            for dep_name in dependencies:
                if (
                    dep_name in results_dict
                    and results_dict[dep_name].get("status") == "success"
                ):
                    response = results_dict[dep_name]["response"]
                    # Get agent role/profile for better attribution
                    dep_role = self._get_agent_role_description(dep_name)
                    role_text = f" ({dep_role})" if dep_role else ""

                    dependency_results.append(
                        f"Agent {dep_name}{role_text}:\n{response}"
                    )

            if dependency_results:
                deps_text = "\n\n".join(dependency_results)
                enhanced_inputs[agent_name] = (
                    f"You are {agent_name}. Please respond to the following input, "
                    f"taking into account the results from the previous agents "
                    f"in the dependency chain.\n\n"
                    f"Original Input:\n{base_input}\n\n"
                    f"Previous Agent Results (for your reference):\n"
                    f"{deps_text}\n\n"
                    f"Please provide your own analysis as {agent_name}, building upon "
                    f"(but not simply repeating) the previous results."
                )
            else:
                enhanced_inputs[agent_name] = (
                    f"You are {agent_name}. Please respond to: {base_input}"
                )

        return enhanced_inputs

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
            agent_config["name"]: agent_config.get("depends_on", [])
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
                if self._dependency_analyzer.agent_dependencies_satisfied(
                    dependencies, processed_agents
                ):
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
