"""Serving Ensemble caller — invokes the declarative per-turn handler (WP-A8).

Satisfies the endpoint's ``_ChatCompletionsCaller`` Protocol: ``run(context)``
yields the shared ``OrchestratorChunk`` vocabulary the SSE formatter and the
non-streaming collector already consume, so the surviving transport (session
resolution, SSE, body shaping) is reused unchanged. The caller runs ONE
declarative ensemble (classify -> seat -> marshal) on the L0 engine (ADR-046
§1; AS-11) and maps its serve outcome onto the client permission seam.

Design: keep the caller thin. The only client-shaped concern it carries is the
toolless-meta-call discrimination OpenCode requires (session-title/summary calls
arrive with no tools and must not drive the pipeline); everything else lives in
the ensemble.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.web.serving.chunks import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
    ToolCallInvocation,
)
from llm_orc.web.serving.turn_trace import emit_turn_trace

if TYPE_CHECKING:
    # A surviving ADR-013 container that still lives under agentic/ during the
    # parity window; WP-B8 relocates it. Imported for typing only — the caller
    # accesses messages/tools structurally, with no runtime agentic/ coupling.
    from llm_orc.web.serving.session_start import SessionContext

_WRITE_TOOL = "write"

# Rung-1 conversation-context caps (memory design §Rung 1): bounded render,
# flat per-turn cost regardless of session length.
_CTX_MAX_MESSAGES = 8
_CTX_TEXT_CAP = 500
_CTX_FILE_CAP = 2000
_CTX_TOTAL_CAP = 4000


def _task_from(messages: Sequence[Any]) -> str:
    """The latest user message — clients send the full history every turn.

    Strips one symmetric surrounding double-quote pair: ``opencode run -c``
    (continued sessions) delivers the content as a quoted literal, which
    breaks anchored routing signals in classify.
    """
    for message in reversed(list(messages)):
        content = getattr(message, "content", None)
        if getattr(message, "role", None) == "user" and (content or "").strip():
            task = (content or "").strip()
            if len(task) >= 2 and task[0] == '"' and task[-1] == '"':
                task = task[1:-1]
            return task
    return ""


def _aux_reply(messages: Sequence[Any]) -> str:
    """A short plain-text reply for OpenCode's toolless meta calls (title /
    summary) — the last user message's subject, never the build pipeline."""
    for message in reversed(list(messages)):
        content = getattr(message, "content", None)
        if getattr(message, "role", None) == "user" and isinstance(content, str):
            words = content.strip().strip('"').split()
            return " ".join(words[:6]) if words else "Task"
    return "Task"


def _render_context(messages: Sequence[Any]) -> str:
    """Prior turns as a deterministic, capped transcript (rung-1 memory).

    Everything before the latest user message renders as ``role: text``
    lines; an assistant write tool_call renders as ``[wrote <path>]`` plus
    the written body (that is what lets a later "add tests for it" see the
    code it refers to); tool-result rows are skipped. Bounded by the module
    caps so per-turn cost stays flat regardless of session length.
    """
    items = list(messages)
    prior: list[Any] = []
    for index in range(len(items) - 1, -1, -1):
        content = getattr(items[index], "content", None)
        if getattr(items[index], "role", None) == "user" and (content or "").strip():
            prior = items[:index]
            break
    lines: list[str] = []
    for message in prior[-_CTX_MAX_MESSAGES:]:
        role = getattr(message, "role", "")
        if role == "tool":
            continue
        line = _render_write(message) or _render_text(message, role)
        if line:
            lines.append(line)
    rendered = "\n".join(lines)
    return rendered[-_CTX_TOTAL_CAP:]


def _render_write(message: Any) -> str | None:
    """An assistant write tool_call as ``[wrote <path>]`` + capped body."""
    for call in getattr(message, "tool_calls", ()) or ():
        function = call.get("function", {}) if isinstance(call, dict) else {}
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(arguments, dict) and arguments.get("filePath"):
            body = str(arguments.get("content", ""))[:_CTX_FILE_CAP]
            return f"assistant: [wrote {arguments['filePath']}]\n{body}"
    return None


def _render_text(message: Any, role: str) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return f"{role}: {content.strip()[:_CTX_TEXT_CAP]}"
    return None


def _tool_result_ack(messages: Sequence[Any]) -> str | None:
    """A short acknowledgment when the call is a tool-result continuation.

    After the serve emits a tool_call and the client performs it, the client
    calls back with the tool result appended. That call continues the SAME
    turn — re-running the pipeline would redo (and possibly re-judge) work
    the client already applied. Returns None when the call is a fresh turn.
    """
    last = messages[-1] if messages else None
    if getattr(last, "role", None) != "tool":
        return None
    for message in reversed(list(messages)):
        written = _written_file_path(getattr(message, "tool_calls", ()) or ())
        if written:
            return f"Wrote {written}."
        if getattr(message, "role", None) == "user":
            break
    content = getattr(last, "content", None)
    return content if isinstance(content, str) and content.strip() else "Done."


def _written_file_path(tool_calls: Sequence[Any]) -> str | None:
    """The filePath of the first write-shaped tool call, if any."""
    for call in tool_calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(arguments, dict) and arguments.get("filePath"):
            return str(arguments["filePath"])
    return None


def _find_ensemble(project_dir: Path, name: str) -> Path:
    direct = project_dir / "ensembles" / f"{name}.yaml"
    if direct.exists():
        return direct
    for path in (project_dir / "ensembles").rglob(f"{name}.yaml"):
        return path
    raise FileNotFoundError(
        f"serving ensemble '{name}' not found under {project_dir}/ensembles"
    )


def _serve_outcome(result: dict[str, Any]) -> dict[str, Any]:
    """The terminal ``emit`` node's serve outcome (shape -> form-gate -> emit)."""
    results = result.get("results", {}) if isinstance(result, dict) else {}
    emit = results.get("emit", {}) if isinstance(results, dict) else {}
    response = emit.get("response", "") if isinstance(emit, dict) else ""
    try:
        outcome = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return {"finish": True, "content": response or ""}
    if isinstance(outcome, dict):
        return outcome
    return {"finish": True, "content": str(outcome)}


def _outcome_chunks(outcome: dict[str, Any]) -> list[OrchestratorChunk]:
    if outcome.get("finish"):
        return [
            ContentDelta(content=str(outcome.get("content", "Done."))),
            Completion(finish_reason="stop"),
        ]
    arguments = json.dumps(
        {
            "filePath": outcome.get("file", "solution.py"),
            "content": outcome.get("content", ""),
        }
    )
    invocation = ToolCallInvocation(
        id=f"call_{uuid.uuid4().hex[:8]}", name=_WRITE_TOOL, arguments=arguments
    )
    return [ClientToolCall(tool_calls=(invocation,))]


class ServingEnsembleCaller:
    """Runs the declarative Serving Ensemble for one turn and yields chunks."""

    def __init__(
        self,
        *,
        project_dir: Path,
        ensemble: str = "serving",
        trace_root: Path | None = None,
    ) -> None:
        self._project_dir = Path(project_dir)
        self._ensemble = ensemble
        self._trace_root = trace_root or (self._project_dir / ".serve-trace")

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        if not context.tools:
            yield ContentDelta(content=_aux_reply(context.messages))
            yield Completion(finish_reason="stop")
            return
        ack = _tool_result_ack(context.messages)
        if ack is not None:
            yield ContentDelta(content=ack)
            yield Completion(finish_reason="stop")
            return
        outcome = await self._serve(
            _task_from(context.messages), _render_context(context.messages)
        )
        for chunk in _outcome_chunks(outcome):
            yield chunk

    async def _serve(self, task: str, conversation: str = "") -> dict[str, Any]:
        config = EnsembleLoader().load_from_file(
            str(_find_ensemble(self._project_dir, self._ensemble))
        )
        executor = ExecutorFactory.create_root_executor(project_dir=self._project_dir)
        result = await executor.execute(
            config, json.dumps({"task": task, "context": conversation})
        )
        emit_turn_trace(config.name, result, self._trace_root)
        return _serve_outcome(result)
