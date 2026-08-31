"""Script agent runner extracted from EnsembleExecutor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import stat
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_orc.agents.script_agent import (
    ScriptAgent,
)
from llm_orc.core.execution.scripting.cache import ScriptCache
from llm_orc.core.execution.scripting.resolver import ScriptResolver
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
    kept for explicitness alone — mypy needs no narrowing here, since
    ``response`` is ``Any``, and removing either guard alone changes
    nothing. The ``isinstance(parsed, dict)`` check IS
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
        # An agent whose whole purpose is to ask a human is never cached
        # (#160): the ANSWER is not part of the key, so a second identical
        # one replayed the first person's response without prompting.
        # Checked at BOTH get and set — the byte-identity below aliases
        # several references onto one key while this predicate still judges
        # each reference separately, so a get-only skip would let an
        # interactive agent's entry be hit by a non-interactive alias.
        # A DISABLED cache is checked here rather than only inside
        # ScriptCache.get/set, because computing the identity is the
        # expensive part and those check `enabled` only after being
        # called. Without this the shipped default — off — still paid a
        # resolution and a whole-file read, twice per execution, to build
        # a key nothing would ever look up.
        cacheable = self._cache_is_enabled() and not self._requires_user_input(
            agent_config
        )
        # NOTE: script_content stays the RAW reference below, because
        # _validate_primitive_output needs it (_normalize_script_ref returns
        # None for an identity string, which would silently disable the
        # primitive schema check on every cache hit).
        # None means "no identity that names this script's bytes" (#163):
        # neither half of the cache runs, rather than degrading to the
        # path-only key that IS the #160 bug.
        cache_identity = self._cache_identity(script_content) if cacheable else None

        cached_result = (
            self._script_cache.get(cache_identity, cache_key_params)
            if cache_identity is not None
            else None
        )
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
        # The digest above was taken BEFORE the subprocess opened the file,
        # so a script edited during its own run produced output from the NEW
        # bytes and stored it under the OLD bytes' key — two executions
        # sharing an entry although they ran different bytes, which is the
        # exact invariant this issue exists to establish, and it persisted
        # for the full TTL rather than self-correcting. Re-reading here
        # closes the window: if the bytes moved, this run's output belongs
        # to no key we can name, so it belongs in no entry.
        # A post-run identity of None (the bytes stopped being digestable
        # during the run) compares unequal here, so it falls closed with
        # the edited case rather than sharing an entry (#163).
        edited_mid_run = (
            cache_identity is not None
            and self._cache_identity(script_content) != cache_identity
        )
        if (
            cache_identity is not None
            and not edited_mid_run
            and not _reports_failure(response)
        ):
            cache_result = {
                "output": response,
                "execution_metadata": {"duration_ms": duration_ms},
            }
            self._script_cache.set(cache_identity, cache_key_params, cache_result)

        return response, model_instance, substituted

    def _cache_is_enabled(self) -> bool:
        """Whether the cache would store anything at all.

        ``bool()`` rather than the attribute itself because several
        suites pass a ``Mock`` here, whose auto-created ``config.enabled``
        is a truthy Mock — which keeps those tests where they were.

        An earlier draft wrapped both lookups in ``getattr`` defaults and
        justified it as tolerance for Mocks. Review measured that the
        defaults never fire: a plain ``Mock`` answers with its own
        auto-created attribute, and only ``Mock(spec=...)`` would reach a
        default, which nothing in the suite uses. Deleting the layer
        changed no test. Dead defensive code with a rationale naming the
        wrong objects is worse than none.
        """
        return bool(self._script_cache.config.enabled)

    def _cache_identity(self, script_ref: str) -> str | None:
        """What identifies this script for caching (#160), or ``None`` when
        nothing does and the run must not be cached at all (#163).

        The cache used to key on ``agent_config.script``, a REFERENCE, which
        in every shipped ensemble is a path — so the key named the file and
        never its contents, and editing a script served the pre-edit result
        for the TTL, crossing processes under ``persist_to_artifacts``.

        **The invariant.** The cache is read and written only under an
        identity that names the script's BYTES. When the bytes cannot be
        named, there is no identity and the run is not cached.

        Two kinds of reference. The CLASSIFICATION is syntactic and happens
        before any filesystem call; the ANSWER for inline content still
        costs one ``os.path.exists``, because of the disagreement below.

        - **Inline content.** ``resolve_script_path`` returns it verbatim, so
          the reference IS its own bytes and identifies itself. The resolver
          answers this (``is_inline_content``) because it is what the
          RESOLVER will do with the reference — but it is not the last word,
          see the disagreement below.
        - **A path.** The resolver will go and find a file, so anything that
          stops us naming that file's bytes — the resolve, the stat, a
          non-regular file, an unreadable one — is a refusal. There is no
          "which call threw" to get wrong; they all answer ``None``.

        That shape is the third. It replaced an errno rule, which could not
        work: ``ENOENT`` is both "this is inline content" and "the file
        vanished between the resolve and the stat", and one errno cannot
        answer two questions. Before that it was ``os.path.isfile`` plus
        ``os.path.exists``, which are ``genericpath`` — they SWALLOW
        ``OSError`` and answer ``False``, so a stale-mount stat fell through
        to the bare path. Each shape was written against the call that had
        just been seen to fail; this one is written against the question.

        **The disagreement.** ``ScriptAgent`` classifies file-vs-inline
        separately, with ``os.path.exists``, at three sites — so for a bare
        name that happens to name a file in the process CWD the resolver says
        content and the agent EXECUTES a file. Failed closed below (such a
        reference is not cached at all) and tracked as #177. The cost is one
        ``exists`` on the inline path and a silent over-refusal: a
        legitimately inline reference that collides with a CWD entry stops
        being cacheable.

        Reachable triggers for the refusal path, all measured: fd exhaustion
        (``EMFILE``/``ENFILE``), ``EIO``/``ESTALE`` on a network filesystem,
        and an ENOENT race where the file is replaced mid-turn — #158 made
        the first likelier by overlapping script agents. The issue's own
        hypothesis, a script executable but not readable, is refuted:
        ``script_agent.py`` runs every extension through an interpreter
        (``bash`` by default), all of which must read the file, so such a
        script never executes and #159 never caches its failure.
        """
        # execute() passes "" for a non-ScriptAgentConfig, which has no
        # script and therefore no bytes to name. It used to return "" here
        # — a constant identity naming nothing, which the cache was then
        # consulted under (#163 review round 1). Nothing can ever write
        # there, but the invariant is about the READ too.
        if not script_ref:
            return None
        resolver = ScriptResolver(project_dir=self._project_dir)
        # The RESOLVER answers what IT will do with the reference (#163
        # review round 3). That has to be settled before any errno is seen:
        # an errno cannot make the split, because ENOENT is both "this is
        # inline content" and "the file vanished between the resolve and the
        # stat", and the shape before this answered the bare path for the
        # second — the #160 key, reachable by one of the three triggers this
        # fix is named for. It is not the whole answer, though; see below.
        if resolver.is_inline_content(script_ref):
            # ScriptAgent does NOT use this predicate. It decides
            # file-vs-inline with os.path.exists, at three separate sites —
            # its own _execute_interactive among them (this class's
            # _execute_interactive is the one that raises instead — never
            # cached), so
            # a bare name that happens to name a file in the process CWD gets
            # EXECUTED as a file while this call would name it content — the
            # #160 key, and a regression review round 4 caught round 3
            # introducing. Fail closed where the two classifiers disagree;
            # unifying them changes what gets EXECUTED and is #177.
            # This costs inline content one stat, which is why the
            # docstring above does NOT claim it stays off the filesystem.
            if os.path.exists(script_ref):
                logger.debug(
                    "no cache identity: %r is inline content but names a file",
                    script_ref,
                )
                return None
            return script_ref
        # From here the reference denotes a file the resolver will go and
        # find, so ANYTHING that stops us naming its bytes is a refusal.
        # There is no "which call threw" left to get wrong: the resolve, the
        # stat and the read all answer None, and only a regular file that
        # digests produces an identity.
        try:
            resolved = resolver.resolve_script_path(script_ref)
            info = os.stat(resolved)
            if not stat.S_ISREG(info.st_mode):
                return None
            digest = hashlib.sha256(Path(resolved).read_bytes()).hexdigest()
        except Exception:
            # Not caching is the safe direction, so no pin can catch a
            # programming error here (a resolver signature change, an
            # AttributeError) silently disabling the cache forever. A log
            # line makes it visible without changing behaviour.
            logger.debug("no cache identity for %r", script_ref, exc_info=True)
            return None
        return f"{resolved}:{digest}"

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
