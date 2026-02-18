"""Script agent runner extracted from EnsembleExecutor."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_orc.agents.enhanced_script_agent import (
    EnhancedScriptAgent,
)
from llm_orc.core.execution.script_cache import ScriptCache
from llm_orc.core.execution.script_user_input_handler import (
    ScriptUserInputHandler,
)
from llm_orc.core.execution.usage_collector import (
    UsageCollector,
)
from llm_orc.models.base import ModelInterface

logger = logging.getLogger(__name__)


class ScriptAgentRunner:
    """Runs script agents with caching and resource monitoring."""

    def __init__(
        self,
        script_cache: ScriptCache,
        usage_collector: UsageCollector,
        progress_controller: Any,
        emit_event: Callable[[str, dict[str, Any]], None],
        project_dir: Path | None,
    ) -> None:
        self._script_cache = script_cache
        self._usage_collector = usage_collector
        self._progress_controller = progress_controller
        self._emit_event = emit_event
        self._project_dir = project_dir
        self._input_lock = asyncio.Lock()

    async def execute(
        self,
        agent_config: dict[str, Any],
        input_data: str,
    ) -> tuple[str, ModelInterface | None]:
        """Execute script agent with caching."""
        script_content = agent_config.get("script", "")
        parameters = agent_config.get("parameters", {})

        cache_key_params = {
            "input_data": input_data,
            "parameters": parameters,
        }

        cached_result = self._script_cache.get(script_content, cache_key_params)
        if cached_result is not None:
            return cached_result.get("output", ""), None

        start_time = time.time()
        response, model_instance = await self._execute_without_cache(
            agent_config, input_data
        )
        duration_ms = int((time.time() - start_time) * 1000)

        cache_result = {
            "output": response,
            "execution_metadata": {"duration_ms": duration_ms},
            "success": True,
        }
        self._script_cache.set(script_content, cache_key_params, cache_result)

        return response, model_instance

    async def _execute_without_cache(
        self,
        agent_config: dict[str, Any],
        input_data: str,
    ) -> tuple[str, ModelInterface | None]:
        """Execute script agent with resource monitoring."""
        agent_name = agent_config["name"]

        self._usage_collector.start_agent_resource_monitoring(agent_name)

        try:
            script_agent = EnhancedScriptAgent(
                agent_name,
                agent_config,
                project_dir=self._project_dir,
            )

            self._usage_collector.sample_agent_resources(agent_name)

            response = await self._execute_with_input_handling(
                script_agent, agent_config, input_data
            )

            self._usage_collector.sample_agent_resources(agent_name)

            if isinstance(response, dict):
                response = json.dumps(response)

            self._validate_primitive_output(agent_config.get("script", ""), response)

            return response, None
        finally:
            self._usage_collector.finalize_agent_resource_monitoring(agent_name)

    async def _execute_with_input_handling(
        self,
        script_agent: EnhancedScriptAgent,
        agent_config: dict[str, Any],
        input_data: str,
    ) -> str | dict[str, Any]:
        """Execute script with appropriate input format."""
        try:
            parsed_input = json.loads(input_data)
            return await self._execute_with_parsed_input(
                script_agent,
                agent_config,
                input_data,
                parsed_input,
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
        """Execute script with parsed JSON input."""
        if self._is_script_agent_input(parsed_input):
            return await script_agent.execute_with_schema_json(input_data)

        if self._requires_user_input(agent_config):
            return await self._execute_interactive(script_agent, parsed_input)

        return await script_agent.execute(json.dumps(parsed_input))

    async def _execute_with_raw_input(
        self,
        script_agent: EnhancedScriptAgent,
        agent_config: dict[str, Any],
        input_data: str,
    ) -> str | dict[str, Any]:
        """Execute script with raw string input."""
        if self._requires_user_input(agent_config):
            return await self._execute_interactive(script_agent, input_data)

        return await script_agent.execute(input_data)

    def _is_script_agent_input(self, parsed_input: dict[str, Any]) -> bool:
        """Check if parsed input is ScriptAgentInput."""
        return (
            isinstance(parsed_input, dict)
            and "agent_name" in parsed_input
            and "input_data" in parsed_input
        )

    def _requires_user_input(self, agent_config: dict[str, Any]) -> bool:
        """Check if script requires user input."""
        handler = ScriptUserInputHandler()
        script_ref = agent_config.get("script", "")
        return handler.requires_user_input(script_ref)

    def _validate_primitive_output(self, script_ref: str, response: str) -> None:
        """Validate output against Pydantic schema for known primitives.

        Opt-in: only fires for registered primitives. On failure, logs a
        warning but does not block output (preserves existing workflows).
        """
        if not isinstance(response, str):
            return

        try:
            from llm_orc.primitives import get_output_schema
        except ImportError:
            return

        output_schema = get_output_schema(script_ref)
        if output_schema is None:
            return

        try:
            output_schema.model_validate_json(response)
        except Exception:
            logger.warning(
                "Primitive output validation failed for %s",
                script_ref,
            )

    async def _execute_interactive(
        self,
        script_agent: EnhancedScriptAgent,
        input_data: str | dict[str, Any],
    ) -> str:
        """Execute script interactively, collecting input at Python layer.

        Uses an asyncio.Lock to serialize terminal access so multiple
        interactive agents in the same phase queue their prompts.
        """
        prompt = script_agent.parameters.get("prompt", "Enter input:")
        parameters = script_agent.parameters

        # Serialize terminal access across concurrent interactive agents
        async with self._input_lock:
            if self._progress_controller:
                try:
                    self._progress_controller.pause_for_user_input(
                        script_agent.name, prompt
                    )
                except Exception:
                    pass

            self._emit_event(
                "user_input_required",
                {
                    "agent_name": script_agent.name,
                    "script": script_agent.script,
                    "message": "Waiting for user input...",
                },
            )

            loop = asyncio.get_running_loop()
            try:
                user_response = await loop.run_in_executor(
                    None, lambda: input(f"{prompt} ")
                )
            except (EOFError, KeyboardInterrupt):
                user_response = ""

            if self._progress_controller:
                try:
                    self._progress_controller.resume_from_user_input(script_agent.name)
                except Exception:
                    pass  # nosec B110

        # Run subprocess outside the lock
        resolved_script = script_agent._script_resolver.resolve_script_path(
            script_agent.script
        )

        if not os.path.exists(resolved_script):
            raise RuntimeError(f"Script file not found: {resolved_script}")

        env = os.environ.copy()
        env.update(script_agent.environment)

        if isinstance(input_data, dict):
            env["INPUT_DATA"] = json.dumps(input_data)
        else:
            env["INPUT_DATA"] = str(input_data)
        env["AGENT_PARAMETERS"] = json.dumps(parameters)

        interpreter = script_agent._get_interpreter(resolved_script)

        stdin_payload = json.dumps(
            {
                "input": user_response,
                "parameters": parameters,
            }
        )

        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                interpreter + [resolved_script],
                input=stdin_payload,
                stdout=subprocess.PIPE,
                stderr=None,
                env=env,
                timeout=script_agent.timeout,
                text=True,
                check=False,
            ),
        )

        self._emit_event(
            "user_input_completed",
            {
                "agent_name": script_agent.name,
                "message": "User input completed, continuing...",
            },
        )

        if result.returncode != 0:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Script exited with code {result.returncode}",
                }
            )

        if result.stdout:
            return result.stdout.strip()
        return json.dumps(
            {
                "success": True,
                "message": "Interactive script completed (no output)",
            }
        )
