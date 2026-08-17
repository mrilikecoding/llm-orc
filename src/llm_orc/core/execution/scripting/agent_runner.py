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

from llm_orc.agents.script_agent import (
    ScriptAgent,
)
from llm_orc.core.execution.scripting.cache import ScriptCache
from llm_orc.core.execution.scripting.user_input_handler import (
    ScriptUserInputHandler,
)
from llm_orc.core.execution.usage_collector import (
    UsageCollector,
)
from llm_orc.core.execution.utils import resolve_agent_timeout
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import AgentConfig, ScriptAgentConfig

logger = logging.getLogger(__name__)


def _reports_failure(response: Any) -> bool:
    """Whether a script's own response says it did not succeed (#159).

    Two clauses, because one does not cover the corpus:

    - ``success`` read for TRUTHINESS with a ``True`` default, so
      ``{"success": 0}`` and ``{"success": null}`` count while a response
      that simply omits the key does NOT (a bare
      ``not parsed.get("success")`` would stop caching everything).
    - a truthy ``error`` key, which is what catches the case this issue
      was filed about: ``web_searcher`` reports every failure as
      ``{"error": ...}`` and exits 0, so neither a ``success`` key nor an
      exception envelope exists. NONE of the 33 scripts in
      ``.llm-orc/scripts/agentic_serving/`` emit a boolean ``success``.

    Two shape guards, and only ONE of them is doing real work. The
    ``isinstance(response, str)`` check is redundant with ``TypeError``
    in the except below (``json.loads`` raises it for list/None/int/bool),
    kept for explicitness and for mypy narrowing — removing either alone
    changes nothing. The ``isinstance(parsed, dict)`` check IS
    load-bearing: ``execute_with_schema_json`` returns RAW stdout rather
    than routing through ``_parse_output``, so on the ScriptAgentInput
    dispatch shape a script printing ``[1,2,3]`` yields a str that parses
    to a list, and ``.get`` on it raises ``AttributeError`` mid-run.
    Do not tidy them as a pair; they are not one.

    Known bounds. ADR-024's ``{"status": "error"|"timeout"|"partial"}``
    envelopes are invisible here — the predicate never looks at
    ``status`` — which is correct today only because every envelope
    builder in the repo hardcodes ``"status": "success"``. And the
    ``error`` clause assumes ``error`` means "a failure message", which
    is convention rather than contract: a success carrying a truthy
    non-string ``error`` (a count, a findings list, a standard-error
    float) stops being cached. Nothing shipped does that; the cost if
    something did would be performance, never correctness.
    """
    if not isinstance(response, str):
        return False
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    return not parsed.get("success", True) or bool(parsed.get("error"))


class ScriptAgentRunner:
    """Runs script agents with caching and resource monitoring."""

    def __init__(
        self,
        script_cache: ScriptCache,
        usage_collector: UsageCollector,
        progress_controller: Any,
        emit_event: Callable[[str, dict[str, Any]], None],
        project_dir: Path | None,
        strict_schema: bool = False,
        performance_config: dict[str, Any] | None = None,
    ) -> None:
        self._script_cache = script_cache
        self._usage_collector = usage_collector
        self._progress_controller = progress_controller
        self._emit_event = emit_event
        self._project_dir = project_dir
        self._strict_schema = strict_schema
        # #157: the subprocess bound has to be the SAME number the
        # dispatcher resolved, or a script agent runs unbounded — which is
        # exactly what happened, since model_dump always supplies
        # timeout_seconds and supplies None when unset.
        self._performance_config = performance_config or {}
        self._input_lock = asyncio.Lock()

    async def execute(
        self,
        agent_config: AgentConfig,
        input_data: str,
    ) -> tuple[str, ModelInterface | None, bool]:
        """Execute script agent with caching.

        Returns:
            Tuple of (response, model_instance, model_substituted).
            model_substituted is always False for script agents.
        """
        script_content = (
            agent_config.script if isinstance(agent_config, ScriptAgentConfig) else ""
        )
        parameters = (
            agent_config.parameters
            if isinstance(agent_config, ScriptAgentConfig)
            else {}
        )

        cache_key_params = {
            "input_data": input_data,
            "parameters": parameters,
        }

        cached_result = self._script_cache.get(script_content, cache_key_params)
        if cached_result is not None:
            cached_output = cached_result.get("output", "")
            self._validate_primitive_output(script_content, cached_output)
            return cached_output, None, False

        start_time = time.time()
        response, model_instance, substituted = await self._execute_without_cache(
            agent_config, input_data
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # A failure is never cached (#159). ScriptCache replays entries for
        # a 3600s TTL on the same (script, input, parameters) key, and with
        # persist_to_artifacts it survives a restart, so one rate-limited
        # search or one timeout under momentary load used to poison that
        # key across processes. The old entry also carried a hardcoded
        # "success": True that nothing read, on entries that might hold a
        # failure.
        if not _reports_failure(response):
            cache_result = {
                "output": response,
                "execution_metadata": {"duration_ms": duration_ms},
            }
            self._script_cache.set(script_content, cache_key_params, cache_result)

        return response, model_instance, substituted

    async def _execute_without_cache(
        self,
        agent_config: AgentConfig,
        input_data: str,
    ) -> tuple[str, ModelInterface | None, bool]:
        """Execute script agent with resource monitoring."""
        agent_name = agent_config.name

        self._usage_collector.start_agent_resource_monitoring(agent_name)

        try:
            # ScriptAgent.__init__ expects dict — convert at boundary
            config_dict = agent_config.model_dump()
            # Fill the resolved bound in rather than leaving the dumped
            # None, so the subprocess is bounded by the same number the
            # dispatcher applies as its outer timeout (#157).
            config_dict["timeout_seconds"] = resolve_agent_timeout(
                config_dict, self._performance_config
            )
            script_agent = ScriptAgent(
                agent_name,
                config_dict,
                project_dir=self._project_dir,
            )

            self._usage_collector.sample_agent_resources(agent_name)

            response = await self._execute_with_input_handling(
                script_agent, agent_config, input_data
            )

            self._usage_collector.sample_agent_resources(agent_name)

            if isinstance(response, dict):
                response = json.dumps(response)

            script_ref = (
                agent_config.script
                if isinstance(agent_config, ScriptAgentConfig)
                else ""
            )
            self._validate_primitive_output(script_ref, response)

            return response, None, False
        finally:
            self._usage_collector.finalize_agent_resource_monitoring(agent_name)

    async def _execute_with_input_handling(
        self,
        script_agent: ScriptAgent,
        agent_config: AgentConfig,
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
        script_agent: ScriptAgent,
        agent_config: AgentConfig,
        input_data: str,
        parsed_input: dict[str, Any],
    ) -> str | dict[str, Any]:
        """Execute script with parsed JSON input."""
        if self._requires_user_input(agent_config):
            return await self._execute_interactive(script_agent, parsed_input)

        if self._is_script_agent_input(parsed_input):
            return await script_agent.execute_with_schema_json(input_data)

        return await script_agent.execute(json.dumps(parsed_input))

    async def _execute_with_raw_input(
        self,
        script_agent: ScriptAgent,
        agent_config: AgentConfig,
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

    def _requires_user_input(self, agent_config: AgentConfig) -> bool:
        """Check if script requires user input."""
        handler = ScriptUserInputHandler()
        script_ref = (
            agent_config.script if isinstance(agent_config, ScriptAgentConfig) else ""
        )
        return handler.requires_user_input(script_ref)

    def _validate_primitive_output(self, script_ref: str, response: str) -> None:
        """Validate output against Pydantic schema for known primitives.

        Opt-in: only fires for registered primitives. When strict_schema
        is enabled, validation failures raise ValueError. Otherwise they
        log a warning and allow output to pass through.
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
        except Exception as exc:
            if self._strict_schema:
                raise ValueError(
                    f"Primitive output schema validation failed for {script_ref}"
                ) from exc
            logger.warning(
                "Primitive output validation failed for %s",
                script_ref,
            )

    async def _execute_interactive(
        self,
        script_agent: ScriptAgent,
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
                    logger.debug(
                        "progress_controller.pause_for_user_input failed for %r",
                        script_agent.name,
                        exc_info=True,
                    )

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
                except Exception:  # nosec B110
                    logger.debug(
                        "progress_controller.resume_from_user_input failed for %r",
                        script_agent.name,
                        exc_info=True,
                    )

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
